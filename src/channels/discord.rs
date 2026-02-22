use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::tungstenite;

use crate::channels::{Channel, ChannelError, InboundMessage, OutboundError, OutboundMessage};
use crate::message::Message;
use crate::namespace::Namespace;
use crate::tools::discord::DiscordConfig;
use tokio::sync::watch;

// ---------------------------------------------------------------------------
// Discord Gateway types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct GatewayEvent {
    op: u8,
    #[serde(default)]
    t: Option<String>,
    #[serde(default)]
    s: Option<u64>,
    d: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct GatewayPayload {
    op: u8,
    d: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ReadyPayload {
    user: BotUser,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct BotUser {
    id: String,
    username: String,
}

#[derive(Debug, Deserialize)]
struct MessageCreate {
    id: String,
    channel_id: String,
    guild_id: Option<String>,
    content: String,
    author: MessageAuthor,
    #[serde(default)]
    mentions: Vec<MessageAuthor>,
}

#[derive(Debug, Deserialize)]
struct MessageAuthor {
    id: String,
    username: String,
    #[serde(default)]
    bot: bool,
}

// Gateway opcodes
const OP_DISPATCH: u8 = 0;
const OP_HEARTBEAT: u8 = 1;
const OP_IDENTIFY: u8 = 2;
const OP_HELLO: u8 = 10;
const OP_HEARTBEAT_ACK: u8 = 11;

// Intents
const INTENT_GUILDS: u64 = 1 << 0;
const INTENT_GUILD_MESSAGES: u64 = 1 << 9;
const INTENT_DIRECT_MESSAGES: u64 = 1 << 12;
const INTENT_MESSAGE_CONTENT: u64 = 1 << 15;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls which messages the Discord channel forwards to the runtime.
#[derive(Debug, Clone)]
pub enum MessageFilter {
    /// Only messages that @mention the bot.
    MentionsOnly,
    /// All messages (excluding the bot's own).
    All,
    /// Only direct messages from the specified usernames.
    DirectMessagesFrom(Vec<String>),
}

/// Configuration for a Discord Gateway channel.
#[derive(Clone)]
pub struct DiscordChannelConfig {
    /// The Discord bot config (token + HTTP client for sending replies).
    pub discord: DiscordConfig,
    /// How to build a namespace from a Discord message.
    /// Default: `discord:<guild_id>:<channel_id>:<author_id>`.
    pub namespace_prefix: String,
    /// Which messages to forward.
    pub filter: MessageFilter,
    /// Known agent names for @mention routing. When a message contains
    /// `@AgentName` (case-insensitive), the matched agent name is stored
    /// in metadata under the key `"agent"`.
    pub agent_names: Vec<String>,
}

impl DiscordChannelConfig {
    pub fn new(discord: DiscordConfig) -> Self {
        Self {
            discord,
            namespace_prefix: "discord".into(),
            filter: MessageFilter::MentionsOnly,
            agent_names: Vec::new(),
        }
    }

    pub fn with_filter(mut self, filter: MessageFilter) -> Self {
        self.filter = filter;
        self
    }

    pub fn with_namespace_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.namespace_prefix = prefix.into();
        self
    }

    pub fn with_agent_names(mut self, names: Vec<String>) -> Self {
        self.agent_names = names;
        self
    }
}

// ---------------------------------------------------------------------------
// DiscordChannel
// ---------------------------------------------------------------------------

/// A channel that connects to the Discord Gateway via WebSocket.
///
/// Call `connect()` to establish the WebSocket connection and start the
/// background reader task. Then use as a `Channel` with `ChannelAdapter::run()`.
///
/// Outbound messages are sent back to the same Discord channel via the REST API.
pub struct DiscordChannel {
    config: DiscordChannelConfig,
    inbound_rx: Mutex<mpsc::Receiver<InboundMessage>>,
    inbound_tx: mpsc::Sender<InboundMessage>,
    bot_id: Mutex<Option<String>>,
    /// Shutdown signal — send `true` to stop the background tasks.
    shutdown_tx: watch::Sender<bool>,
}

impl DiscordChannel {
    pub fn new(config: DiscordChannelConfig) -> Self {
        let (inbound_tx, inbound_rx) = mpsc::channel(256);
        let (shutdown_tx, _) = watch::channel(false);
        Self {
            config,
            inbound_rx: Mutex::new(inbound_rx),
            inbound_tx,
            bot_id: Mutex::new(None),
            shutdown_tx,
        }
    }

    /// Shut down the Discord connection gracefully.
    ///
    /// Signals all background tasks (heartbeat, event reader) to stop.
    /// The `receive()` method checks this signal and will return `None`.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }

    /// Connect to the Discord Gateway and start receiving events.
    ///
    /// This spawns a background task that:
    /// 1. Opens a WebSocket to `wss://gateway.discord.gg/?v=10&encoding=json`
    /// 2. Sends the Identify payload
    /// 3. Maintains heartbeats
    /// 4. Parses MESSAGE_CREATE events and pushes them into the channel
    ///
    /// Returns `Ok(())` once the connection is established and the READY event
    /// has been received. The background task continues running until the
    /// WebSocket closes or an error occurs.
    pub async fn connect(&self) -> Result<(), DiscordChannelError> {
        let url = "wss://gateway.discord.gg/?v=10&encoding=json";
        let (ws_stream, _) = tokio_tungstenite::connect_async(url)
            .await
            .map_err(|e| DiscordChannelError::Connection(e.to_string()))?;

        let (write, read) = ws_stream.split();
        let write = Arc::new(Mutex::new(write));

        // Wait for Hello to get heartbeat interval
        let mut read = read;
        let hello_msg = read
            .next()
            .await
            .ok_or_else(|| {
                DiscordChannelError::Connection("connection closed before Hello".into())
            })?
            .map_err(|e| DiscordChannelError::Connection(e.to_string()))?;

        let hello: GatewayEvent = parse_ws_message(&hello_msg)?;
        if hello.op != OP_HELLO {
            return Err(DiscordChannelError::Protocol(format!(
                "expected Hello (op 10), got op {}",
                hello.op
            )));
        }

        let heartbeat_interval = hello.d["heartbeat_interval"]
            .as_u64()
            .ok_or_else(|| DiscordChannelError::Protocol("missing heartbeat_interval".into()))?;

        // Send Identify
        let mut intents = INTENT_GUILDS | INTENT_GUILD_MESSAGES | INTENT_MESSAGE_CONTENT;
        if matches!(self.config.filter, MessageFilter::DirectMessagesFrom(_)) {
            intents |= INTENT_DIRECT_MESSAGES;
        }
        let identify = GatewayPayload {
            op: OP_IDENTIFY,
            d: serde_json::json!({
                "token": self.config.discord.token(),
                "intents": intents,
                "properties": {
                    "os": std::env::consts::OS,
                    "browser": "orra",
                    "device": "orra"
                }
            }),
        };

        {
            let mut w = write.lock().await;
            w.send(tungstenite::Message::Text(
                serde_json::to_string(&identify).unwrap().into(),
            ))
            .await
            .map_err(|e| DiscordChannelError::Connection(e.to_string()))?;
        }

        // Wait for READY to get bot user ID
        let ready_msg = read
            .next()
            .await
            .ok_or_else(|| {
                DiscordChannelError::Connection("connection closed before Ready".into())
            })?
            .map_err(|e| DiscordChannelError::Connection(e.to_string()))?;

        let ready_event: GatewayEvent = parse_ws_message(&ready_msg)?;
        if ready_event.t.as_deref() != Some("READY") {
            return Err(DiscordChannelError::Protocol(format!(
                "expected READY, got {:?}",
                ready_event.t
            )));
        }

        let ready: ReadyPayload = serde_json::from_value(ready_event.d)
            .map_err(|e| DiscordChannelError::Protocol(format!("parse READY: {e}")))?;

        let bot_user_id = ready.user.id.clone();
        *self.bot_id.lock().await = Some(bot_user_id.clone());

        // Spawn heartbeat task
        let heartbeat_write = write.clone();
        let sequence = Arc::new(Mutex::new(ready_event.s));
        let seq_for_heartbeat = sequence.clone();
        let mut shutdown_rx_heartbeat = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(heartbeat_interval));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let seq = *seq_for_heartbeat.lock().await;
                        let payload = GatewayPayload {
                            op: OP_HEARTBEAT,
                            d: match seq {
                                Some(s) => serde_json::json!(s),
                                None => serde_json::Value::Null,
                            },
                        };
                        let msg = serde_json::to_string(&payload).unwrap();
                        let mut w = heartbeat_write.lock().await;
                        if w.send(tungstenite::Message::Text(msg.into())).await.is_err() {
                            break;
                        }
                    }
                    _ = shutdown_rx_heartbeat.changed() => {
                        break;
                    }
                }
            }
        });

        // Spawn event reader task
        let inbound_tx = self.inbound_tx.clone();
        let filter = self.config.filter.clone();
        let ns_prefix = self.config.namespace_prefix.clone();
        let agent_names = self.config.agent_names.clone();
        let mut shutdown_rx_reader = self.shutdown_tx.subscribe();

        tokio::spawn(async move {
            loop {
                let msg_result = tokio::select! {
                    result = read.next() => {
                        match result {
                            Some(r) => r,
                            None => break,
                        }
                    }
                    _ = shutdown_rx_reader.changed() => {
                        eprintln!("[discord] Shutdown signal received, disconnecting...");
                        break;
                    }
                };
                let msg = match msg_result {
                    Ok(m) => m,
                    Err(_) => break,
                };

                let event: GatewayEvent = match parse_ws_message(&msg) {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                // Update sequence
                if let Some(s) = event.s {
                    *sequence.lock().await = Some(s);
                }

                match event.op {
                    OP_DISPATCH => {
                        if event.t.as_deref() == Some("MESSAGE_CREATE") {
                            let mc: MessageCreate = match serde_json::from_value(event.d) {
                                Ok(m) => m,
                                Err(_) => continue,
                            };

                            // Skip bot's own messages
                            if mc.author.id == bot_user_id {
                                continue;
                            }

                            // Skip other bot messages
                            if mc.author.bot {
                                continue;
                            }

                            // Apply filter
                            let should_process = match &filter {
                                MessageFilter::All => true,
                                MessageFilter::MentionsOnly => {
                                    mc.mentions.iter().any(|m| m.id == bot_user_id)
                                }
                                MessageFilter::DirectMessagesFrom(users) => {
                                    // Must be a DM (no guild_id) from an allowed user
                                    mc.guild_id.is_none()
                                        && users
                                            .iter()
                                            .any(|u| u.eq_ignore_ascii_case(&mc.author.username))
                                }
                            };

                            if !should_process {
                                continue;
                            }

                            // Strip the bot mention from content for cleaner input
                            let content = strip_mention(&mc.content, &bot_user_id);

                            // Detect @AgentName mentions for multi-agent routing
                            let (agent, content) = detect_agent_mention(&content, &agent_names);

                            if content.trim().is_empty() {
                                continue;
                            }

                            // Build namespace: prefix:guild:channel:user
                            let guild = mc.guild_id.as_deref().unwrap_or("dm");
                            let ns = Namespace::new(&ns_prefix)
                                .child(guild)
                                .child(&mc.channel_id)
                                .child(&mc.author.id);

                            let mut metadata = HashMap::new();
                            metadata.insert("channel_id".into(), serde_json::json!(mc.channel_id));
                            metadata.insert("message_id".into(), serde_json::json!(mc.id));
                            metadata.insert("author_id".into(), serde_json::json!(mc.author.id));
                            metadata.insert(
                                "author_username".into(),
                                serde_json::json!(mc.author.username),
                            );
                            if let Some(ref gid) = mc.guild_id {
                                metadata.insert("guild_id".into(), serde_json::json!(gid));
                            }
                            if let Some(agent_name) = agent {
                                metadata.insert("agent".into(), serde_json::json!(agent_name));
                            }

                            let inbound = InboundMessage {
                                namespace: ns,
                                message: Message::user(content.trim()),
                                metadata,
                            };

                            if inbound_tx.send(inbound).await.is_err() {
                                break;
                            }
                        }
                    }
                    OP_HEARTBEAT_ACK => {}
                    OP_HEARTBEAT => {
                        // Server requesting heartbeat - we handle this in our heartbeat loop
                    }
                    _ => {}
                }
            }
        });

        Ok(())
    }
}

#[async_trait]
impl Channel for DiscordChannel {
    async fn receive(&self) -> Option<InboundMessage> {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut rx = self.inbound_rx.lock().await;
        tokio::select! {
            msg = rx.recv() => msg,
            _ = shutdown_rx.changed() => None,
        }
    }

    async fn start_typing(
        &self,
        metadata: &HashMap<String, serde_json::Value>,
    ) -> Option<crate::channels::TypingGuard> {
        let channel_id = metadata
            .get("channel_id")
            .and_then(|v| v.as_str())?
            .to_string();

        let discord = self.config.discord.clone();

        // Send the initial typing indicator
        let _ = discord
            .request(
                reqwest::Method::POST,
                &format!("channels/{channel_id}/typing"),
            )
            .send()
            .await;

        // Spawn a background task that re-triggers every 8 seconds.
        // The oneshot channel acts as a cancellation signal — when the
        // TypingGuard is dropped the sender is dropped, causing the
        // receiver to resolve and the loop to exit.
        let (cancel_tx, mut cancel_rx) = tokio::sync::oneshot::channel::<()>();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(8));
            // Skip the first tick (we already sent the initial one)
            interval.tick().await;

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let _ = discord
                            .request(
                                reqwest::Method::POST,
                                &format!("channels/{channel_id}/typing"),
                            )
                            .send()
                            .await;
                    }
                    _ = &mut cancel_rx => {
                        break;
                    }
                }
            }
        });

        Some(crate::channels::TypingGuard::new(cancel_tx))
    }

    async fn send(&self, response: OutboundMessage) -> Result<(), ChannelError> {
        let channel_id = response
            .metadata
            .get("channel_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChannelError::Send("missing channel_id in metadata".into()))?;

        // Send the reply via the Discord REST API
        let resp = self
            .config
            .discord
            .request(
                reqwest::Method::POST,
                &format!("channels/{channel_id}/messages"),
            )
            .json(&serde_json::json!({ "content": response.message.content }))
            .send()
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ChannelError::Send(format!("Discord API {status}: {body}")));
        }

        Ok(())
    }

    async fn send_error(&self, error: OutboundError) -> Result<(), ChannelError> {
        let channel_id = error
            .metadata
            .get("channel_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ChannelError::Send("missing channel_id in metadata".into()))?;

        let error_msg = format!("Sorry, something went wrong: {}", error.error);

        let resp = self
            .config
            .discord
            .request(
                reqwest::Method::POST,
                &format!("channels/{channel_id}/messages"),
            )
            .json(&serde_json::json!({ "content": error_msg }))
            .send()
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ChannelError::Send(format!("Discord API {status}: {body}")));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_ws_message(msg: &tungstenite::Message) -> Result<GatewayEvent, DiscordChannelError> {
    match msg {
        tungstenite::Message::Text(text) => serde_json::from_str(text)
            .map_err(|e| DiscordChannelError::Protocol(format!("parse gateway event: {e}"))),
        tungstenite::Message::Close(_) => {
            Err(DiscordChannelError::Connection("connection closed".into()))
        }
        _ => Err(DiscordChannelError::Protocol(
            "unexpected message type".into(),
        )),
    }
}

/// Strip `<@bot_id>` mentions from message content.
fn strip_mention(content: &str, bot_id: &str) -> String {
    content
        .replace(&format!("<@{bot_id}>"), "")
        .replace(&format!("<@!{bot_id}>"), "")
}

// Re-export detect_agent_mention from the agent module for backward compat
use crate::agent::detect_agent_mention;

#[derive(Debug, thiserror::Error)]
pub enum DiscordChannelError {
    #[error("connection error: {0}")]
    Connection(String),

    #[error("protocol error: {0}")]
    Protocol(String),
}

// ---------------------------------------------------------------------------
// We need to expose DiscordConfig.request() for the channel to send messages.
// It's currently `fn` (private). Let's make it pub(crate).
// ---------------------------------------------------------------------------

// Actually, DiscordConfig.request() is in tools::discord, which is a different
// module. We need to make it accessible. Let's add a pub(crate) method.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_mention_removes_bot_id() {
        let content = "<@123456> hello there";
        assert_eq!(strip_mention(content, "123456"), " hello there");
    }

    #[test]
    fn strip_mention_handles_nick_format() {
        let content = "<@!123456> what's up";
        assert_eq!(strip_mention(content, "123456"), " what's up");
    }

    #[test]
    fn strip_mention_preserves_other_mentions() {
        let content = "<@123456> hey <@789> check this";
        assert_eq!(strip_mention(content, "123456"), " hey <@789> check this");
    }

    #[test]
    fn strip_mention_no_mention() {
        let content = "just a regular message";
        assert_eq!(strip_mention(content, "123456"), "just a regular message");
    }

    #[test]
    fn config_defaults() {
        let dc = DiscordConfig::new("tok");
        let config = DiscordChannelConfig::new(dc);
        assert_eq!(config.namespace_prefix, "discord");
        assert!(matches!(config.filter, MessageFilter::MentionsOnly));
    }

    #[test]
    fn config_builder() {
        let dc = DiscordConfig::new("tok");
        let config = DiscordChannelConfig::new(dc)
            .with_filter(MessageFilter::All)
            .with_namespace_prefix("mybot");
        assert_eq!(config.namespace_prefix, "mybot");
        assert!(matches!(config.filter, MessageFilter::All));
    }

    #[test]
    fn message_create_deserialization() {
        let json = serde_json::json!({
            "id": "111",
            "channel_id": "222",
            "guild_id": "333",
            "content": "<@444> hello bot",
            "author": {
                "id": "555",
                "username": "alice",
                "bot": false
            },
            "mentions": [
                {"id": "444", "username": "mybot", "bot": true}
            ]
        });

        let mc: MessageCreate = serde_json::from_value(json).unwrap();
        assert_eq!(mc.id, "111");
        assert_eq!(mc.channel_id, "222");
        assert_eq!(mc.guild_id, Some("333".into()));
        assert_eq!(mc.author.username, "alice");
        assert!(!mc.author.bot);
        assert_eq!(mc.mentions.len(), 1);
        assert_eq!(mc.mentions[0].id, "444");
    }

    #[test]
    fn gateway_event_deserialization() {
        let json = serde_json::json!({
            "op": 0,
            "t": "MESSAGE_CREATE",
            "s": 42,
            "d": {"id": "123", "content": "hello"}
        });

        let event: GatewayEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.op, 0);
        assert_eq!(event.t.as_deref(), Some("MESSAGE_CREATE"));
        assert_eq!(event.s, Some(42));
    }

    #[test]
    fn gateway_hello_deserialization() {
        let json = serde_json::json!({
            "op": 10,
            "d": {"heartbeat_interval": 41250}
        });

        let event: GatewayEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.op, 10);
        assert_eq!(event.d["heartbeat_interval"], 41250);
    }

    #[test]
    fn namespace_from_message() {
        let ns = Namespace::new("discord")
            .child("guild123")
            .child("chan456")
            .child("user789");
        assert_eq!(ns.key(), "discord:guild123:chan456:user789");
    }

    #[test]
    fn namespace_dm_fallback() {
        let ns = Namespace::new("discord")
            .child("dm")
            .child("chan456")
            .child("user789");
        assert_eq!(ns.key(), "discord:dm:chan456:user789");
    }

    #[test]
    fn channel_creates_with_buffer() {
        let dc = DiscordConfig::new("tok");
        let config = DiscordChannelConfig::new(dc);
        let _channel = DiscordChannel::new(config);
        // Just verify construction doesn't panic
    }

    #[test]
    fn detect_agent_mention_finds_match() {
        let agents = vec!["Atlas".into(), "CodeBot".into()];
        let (agent, content) = detect_agent_mention("@Atlas what is 2+2?", &agents);
        assert_eq!(agent, Some("Atlas"));
        assert_eq!(content, " what is 2+2?");
    }

    #[test]
    fn detect_agent_mention_case_insensitive() {
        let agents = vec!["Atlas".into(), "CodeBot".into()];
        let (agent, content) = detect_agent_mention("@codebot help me", &agents);
        assert_eq!(agent, Some("CodeBot"));
        assert_eq!(content, " help me");
    }

    #[test]
    fn detect_agent_mention_no_match() {
        let agents = vec!["Atlas".into(), "CodeBot".into()];
        let (agent, content) = detect_agent_mention("hello world", &agents);
        assert!(agent.is_none());
        assert_eq!(content, "hello world");
    }

    #[test]
    fn detect_agent_mention_empty_agents() {
        let agents: Vec<String> = Vec::new();
        let (agent, content) = detect_agent_mention("@Atlas hello", &agents);
        assert!(agent.is_none());
        assert_eq!(content, "@Atlas hello");
    }

    #[test]
    fn detect_agent_mention_word_boundary() {
        let agents = vec!["At".into()];
        // "@Atlas" should NOT match agent "At" because "las" follows
        let (agent, content) = detect_agent_mention("@Atlas hello", &agents);
        assert!(agent.is_none());
        assert_eq!(content, "@Atlas hello");
    }

    #[test]
    fn config_with_agent_names() {
        let dc = DiscordConfig::new("tok");
        let config =
            DiscordChannelConfig::new(dc).with_agent_names(vec!["Atlas".into(), "CodeBot".into()]);
        assert_eq!(config.agent_names.len(), 2);
    }
}
