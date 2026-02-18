use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

// ---------------------------------------------------------------------------
// Shared Discord client config
// ---------------------------------------------------------------------------

/// Configuration for Discord bot API access. Shared across all Discord tools.
#[derive(Clone)]
pub struct DiscordConfig {
    client: Client,
    token: String,
    api_base: String,
}

impl DiscordConfig {
    /// Create a new Discord config with a bot token.
    ///
    /// The token should be the raw bot token (no "Bot " prefix — that's added
    /// automatically in request headers).
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            token: token.into(),
            api_base: "https://discord.com/api/v10".into(),
        }
    }

    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    pub fn with_api_base(mut self, base: impl Into<String>) -> Self {
        self.api_base = base.into();
        self
    }

    pub fn api_base(&self) -> &str {
        &self.api_base
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.api_base, path)
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        self.client
            .request(method, self.url(path))
            .header("Authorization", format!("Bot {}", self.token))
            .header("Content-Type", "application/json")
            .header("User-Agent", "claw-lib")
    }
}

// ---------------------------------------------------------------------------
// Discord API response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Channel {
    id: String,
    name: Option<String>,
    #[serde(rename = "type")]
    channel_type: u8,
    topic: Option<String>,
    parent_id: Option<String>,
    position: Option<i32>,
}

impl Channel {
    fn type_name(&self) -> &str {
        match self.channel_type {
            0 => "text",
            2 => "voice",
            4 => "category",
            5 => "announcement",
            13 => "stage",
            15 => "forum",
            _ => "other",
        }
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DiscordMessage {
    id: String,
    content: String,
    author: DiscordUser,
    timestamp: String,
    edited_timestamp: Option<String>,
    pinned: bool,
    #[serde(rename = "type")]
    message_type: u8,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct DiscordUser {
    id: String,
    username: String,
    discriminator: Option<String>,
    global_name: Option<String>,
}

impl DiscordUser {
    fn display_name(&self) -> &str {
        self.global_name
            .as_deref()
            .unwrap_or(&self.username)
    }
}

#[derive(Debug, Deserialize)]
struct Guild {
    id: String,
    name: String,
    owner_id: String,
    approximate_member_count: Option<u64>,
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

pub struct ListChannelsTool {
    dc: DiscordConfig,
}

impl ListChannelsTool {
    pub fn new(dc: DiscordConfig) -> Self {
        Self { dc }
    }
}

#[async_trait]
impl Tool for ListChannelsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_channels".into(),
            description: "List all channels in a Discord server (guild). Returns channel names, types, and IDs.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "guild_id": {
                        "type": "string",
                        "description": "The Discord server (guild) ID"
                    }
                },
                "required": ["guild_id"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let guild_id = input
            .get("guild_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'guild_id'".into()))?;

        let resp = self
            .dc
            .request(reqwest::Method::GET, &format!("guilds/{}/channels", guild_id))
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "Discord API {}: {}",
                status, body
            )));
        }

        let channels: Vec<Channel> = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {}", e)))?;

        if channels.is_empty() {
            return Ok("No channels found in this server.".into());
        }

        // Group by category
        let categories: Vec<&Channel> = channels
            .iter()
            .filter(|c| c.channel_type == 4)
            .collect();

        let mut lines = Vec::new();

        // Channels under categories
        for cat in &categories {
            lines.push(format!(
                "\n📁 {} (id: {})",
                cat.name.as_deref().unwrap_or("unnamed"),
                cat.id
            ));

            let children: Vec<&Channel> = channels
                .iter()
                .filter(|c| c.parent_id.as_deref() == Some(&cat.id) && c.channel_type != 4)
                .collect();

            for ch in children {
                let prefix = match ch.channel_type {
                    0 => "#",
                    2 => "🔊",
                    5 => "📢",
                    13 => "🎤",
                    15 => "💬",
                    _ => "·",
                };
                let topic_str = ch
                    .topic
                    .as_deref()
                    .filter(|t| !t.is_empty())
                    .map(|t| format!(" — {}", t))
                    .unwrap_or_default();
                lines.push(format!(
                    "  {} {} [{}] (id: {}){}",
                    prefix,
                    ch.name.as_deref().unwrap_or("unnamed"),
                    ch.type_name(),
                    ch.id,
                    topic_str
                ));
            }
        }

        // Uncategorized channels
        let uncategorized: Vec<&Channel> = channels
            .iter()
            .filter(|c| c.parent_id.is_none() && c.channel_type != 4)
            .collect();

        if !uncategorized.is_empty() {
            lines.push("\n(uncategorized)".into());
            for ch in uncategorized {
                lines.push(format!(
                    "  #{} [{}] (id: {})",
                    ch.name.as_deref().unwrap_or("unnamed"),
                    ch.type_name(),
                    ch.id,
                ));
            }
        }

        Ok(lines.join("\n").trim_start().to_string())
    }
}

pub struct GetChannelInfoTool {
    dc: DiscordConfig,
}

impl GetChannelInfoTool {
    pub fn new(dc: DiscordConfig) -> Self {
        Self { dc }
    }
}

#[async_trait]
impl Tool for GetChannelInfoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_channel_info".into(),
            description: "Get detailed information about a specific Discord channel.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The channel ID"
                    }
                },
                "required": ["channel_id"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let channel_id = input
            .get("channel_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'channel_id'".into()))?;

        let resp = self
            .dc
            .request(reqwest::Method::GET, &format!("channels/{}", channel_id))
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "Discord API {}: {}",
                status, body
            )));
        }

        let ch: Channel = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {}", e)))?;

        let topic = ch
            .topic
            .as_deref()
            .filter(|t| !t.is_empty())
            .unwrap_or("(none)");

        Ok(format!(
            "#{} (id: {})\ntype: {}\ntopic: {}",
            ch.name.as_deref().unwrap_or("unnamed"),
            ch.id,
            ch.type_name(),
            topic,
        ))
    }
}

pub struct GetMessagesTool {
    dc: DiscordConfig,
}

impl GetMessagesTool {
    pub fn new(dc: DiscordConfig) -> Self {
        Self { dc }
    }
}

#[async_trait]
impl Tool for GetMessagesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_messages".into(),
            description: "Get recent messages from a Discord channel. Returns message content, author, and timestamps.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The channel ID to read messages from"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of messages to fetch (1-100). Defaults to 20."
                    },
                    "before": {
                        "type": "string",
                        "description": "Get messages before this message ID (for pagination)"
                    }
                },
                "required": ["channel_id"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let channel_id = input
            .get("channel_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'channel_id'".into()))?;

        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(20)
            .min(100);

        let mut req = self
            .dc
            .request(reqwest::Method::GET, &format!("channels/{}/messages", channel_id))
            .query(&[("limit", &limit.to_string())]);

        if let Some(before) = input.get("before").and_then(|v| v.as_str()) {
            req = req.query(&[("before", before)]);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "Discord API {}: {}",
                status, body
            )));
        }

        let messages: Vec<DiscordMessage> = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {}", e)))?;

        if messages.is_empty() {
            return Ok("No messages found in this channel.".into());
        }

        // Discord returns newest first; reverse for chronological reading
        let mut lines = Vec::new();
        for msg in messages.iter().rev() {
            let edited = if msg.edited_timestamp.is_some() {
                " (edited)"
            } else {
                ""
            };
            let pinned = if msg.pinned { " 📌" } else { "" };
            lines.push(format!(
                "[{}] @{}{}{}: {}",
                msg.timestamp,
                msg.author.display_name(),
                pinned,
                edited,
                msg.content,
            ));
        }

        Ok(lines.join("\n"))
    }
}

pub struct SendMessageTool {
    dc: DiscordConfig,
}

impl SendMessageTool {
    pub fn new(dc: DiscordConfig) -> Self {
        Self { dc }
    }
}

#[async_trait]
impl Tool for SendMessageTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "send_message".into(),
            description: "Send a message to a Discord channel.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The channel ID to send the message to"
                    },
                    "content": {
                        "type": "string",
                        "description": "The message text (up to 2000 characters)"
                    }
                },
                "required": ["channel_id", "content"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let channel_id = input
            .get("channel_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'channel_id'".into()))?;

        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'content'".into()))?;

        if content.len() > 2000 {
            return Err(ToolError::InvalidInput(
                "message content exceeds 2000 character limit".into(),
            ));
        }

        let resp = self
            .dc
            .request(reqwest::Method::POST, &format!("channels/{}/messages", channel_id))
            .json(&serde_json::json!({ "content": content }))
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "Discord API {}: {}",
                status, body
            )));
        }

        let msg: DiscordMessage = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {}", e)))?;

        Ok(format!("Sent message {} in channel {}.", msg.id, channel_id))
    }
}

pub struct ReplyToMessageTool {
    dc: DiscordConfig,
}

impl ReplyToMessageTool {
    pub fn new(dc: DiscordConfig) -> Self {
        Self { dc }
    }
}

#[async_trait]
impl Tool for ReplyToMessageTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "reply_to_message".into(),
            description: "Reply to a specific message in a Discord channel. The reply will show as a threaded response.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The channel ID containing the message"
                    },
                    "message_id": {
                        "type": "string",
                        "description": "The ID of the message to reply to"
                    },
                    "content": {
                        "type": "string",
                        "description": "The reply text (up to 2000 characters)"
                    }
                },
                "required": ["channel_id", "message_id", "content"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let channel_id = input
            .get("channel_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'channel_id'".into()))?;

        let message_id = input
            .get("message_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'message_id'".into()))?;

        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'content'".into()))?;

        if content.len() > 2000 {
            return Err(ToolError::InvalidInput(
                "message content exceeds 2000 character limit".into(),
            ));
        }

        let resp = self
            .dc
            .request(reqwest::Method::POST, &format!("channels/{}/messages", channel_id))
            .json(&serde_json::json!({
                "content": content,
                "message_reference": {
                    "message_id": message_id
                }
            }))
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "Discord API {}: {}",
                status, body
            )));
        }

        let msg: DiscordMessage = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {}", e)))?;

        Ok(format!(
            "Replied to message {} with message {} in channel {}.",
            message_id, msg.id, channel_id
        ))
    }
}

pub struct GetGuildInfoTool {
    dc: DiscordConfig,
}

impl GetGuildInfoTool {
    pub fn new(dc: DiscordConfig) -> Self {
        Self { dc }
    }
}

#[async_trait]
impl Tool for GetGuildInfoTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_guild_info".into(),
            description: "Get information about a Discord server (guild), including name, owner, and member count.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "guild_id": {
                        "type": "string",
                        "description": "The Discord server (guild) ID"
                    }
                },
                "required": ["guild_id"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let guild_id = input
            .get("guild_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'guild_id'".into()))?;

        let resp = self
            .dc
            .request(
                reqwest::Method::GET,
                &format!("guilds/{}?with_counts=true", guild_id),
            )
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "Discord API {}: {}",
                status, body
            )));
        }

        let guild: Guild = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {}", e)))?;

        let members = guild
            .approximate_member_count
            .map(|c| format!("\nmembers: ~{}", c))
            .unwrap_or_default();

        Ok(format!(
            "{} (id: {})\nowner: {}{}",
            guild.name, guild.id, guild.owner_id, members,
        ))
    }
}

// ---------------------------------------------------------------------------
// Convenience registration
// ---------------------------------------------------------------------------

/// Register all Discord tools into a ToolRegistry.
pub fn register_tools(registry: &mut ToolRegistry, config: &DiscordConfig) {
    registry.register(Box::new(ListChannelsTool::new(config.clone())));
    registry.register(Box::new(GetChannelInfoTool::new(config.clone())));
    registry.register(Box::new(GetMessagesTool::new(config.clone())));
    registry.register(Box::new(SendMessageTool::new(config.clone())));
    registry.register(Box::new(ReplyToMessageTool::new(config.clone())));
    registry.register(Box::new(GetGuildInfoTool::new(config.clone())));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discord_config_url() {
        let config = DiscordConfig::new("tok123");
        assert_eq!(
            config.url("channels/456/messages"),
            "https://discord.com/api/v10/channels/456/messages"
        );
        assert_eq!(
            config.url("guilds/789/channels"),
            "https://discord.com/api/v10/guilds/789/channels"
        );
    }

    #[test]
    fn discord_config_custom_base() {
        let config = DiscordConfig::new("tok").with_api_base("http://localhost:9999");
        assert_eq!(
            config.url("channels/1/messages"),
            "http://localhost:9999/channels/1/messages"
        );
    }

    #[test]
    fn discord_config_accessors() {
        let config = DiscordConfig::new("tok");
        assert_eq!(config.api_base(), "https://discord.com/api/v10");
    }

    #[test]
    fn channel_type_names() {
        let ch = |t: u8| Channel {
            id: "1".into(),
            name: None,
            channel_type: t,
            topic: None,
            parent_id: None,
            position: None,
        };

        assert_eq!(ch(0).type_name(), "text");
        assert_eq!(ch(2).type_name(), "voice");
        assert_eq!(ch(4).type_name(), "category");
        assert_eq!(ch(5).type_name(), "announcement");
        assert_eq!(ch(13).type_name(), "stage");
        assert_eq!(ch(15).type_name(), "forum");
        assert_eq!(ch(99).type_name(), "other");
    }

    #[test]
    fn user_display_name_prefers_global() {
        let user = DiscordUser {
            id: "1".into(),
            username: "alice_99".into(),
            discriminator: None,
            global_name: Some("Alice".into()),
        };
        assert_eq!(user.display_name(), "Alice");
    }

    #[test]
    fn user_display_name_falls_back_to_username() {
        let user = DiscordUser {
            id: "1".into(),
            username: "alice_99".into(),
            discriminator: None,
            global_name: None,
        };
        assert_eq!(user.display_name(), "alice_99");
    }

    #[test]
    fn register_tools_adds_all_six() {
        let config = DiscordConfig::new("tok");
        let mut registry = ToolRegistry::new();
        register_tools(&mut registry, &config);

        assert_eq!(registry.len(), 6);
        assert!(registry.get("list_channels").is_some());
        assert!(registry.get("get_channel_info").is_some());
        assert!(registry.get("get_messages").is_some());
        assert!(registry.get("send_message").is_some());
        assert!(registry.get("reply_to_message").is_some());
        assert!(registry.get("get_guild_info").is_some());
    }

    #[test]
    fn tool_definitions_have_schemas() {
        let config = DiscordConfig::new("tok");
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(ListChannelsTool::new(config.clone())),
            Box::new(GetChannelInfoTool::new(config.clone())),
            Box::new(GetMessagesTool::new(config.clone())),
            Box::new(SendMessageTool::new(config.clone())),
            Box::new(ReplyToMessageTool::new(config.clone())),
            Box::new(GetGuildInfoTool::new(config)),
        ];

        for tool in &tools {
            let def = tool.definition();
            assert!(!def.name.is_empty());
            assert!(!def.description.is_empty());
            assert_eq!(def.input_schema["type"], "object");
        }
    }

    #[tokio::test]
    async fn send_message_validates_length() {
        let config = DiscordConfig::new("tok");
        let tool = SendMessageTool::new(config);

        let long_content = "x".repeat(2001);
        let err = tool
            .execute(serde_json::json!({
                "channel_id": "123",
                "content": long_content
            }))
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(err.to_string().contains("2000"));
    }

    #[tokio::test]
    async fn reply_validates_length() {
        let config = DiscordConfig::new("tok");
        let tool = ReplyToMessageTool::new(config);

        let long_content = "x".repeat(2001);
        let err = tool
            .execute(serde_json::json!({
                "channel_id": "123",
                "message_id": "456",
                "content": long_content
            }))
            .await
            .unwrap_err();

        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn send_message_missing_fields() {
        let config = DiscordConfig::new("tok");
        let tool = SendMessageTool::new(config);

        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));

        let config2 = DiscordConfig::new("tok");
        let tool2 = SendMessageTool::new(config2);
        let err2 = tool2
            .execute(serde_json::json!({"channel_id": "123"}))
            .await
            .unwrap_err();
        assert!(matches!(err2, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn list_channels_missing_guild_id() {
        let config = DiscordConfig::new("tok");
        let tool = ListChannelsTool::new(config);
        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn get_messages_missing_channel_id() {
        let config = DiscordConfig::new("tok");
        let tool = GetMessagesTool::new(config);
        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn get_channel_info_missing_channel_id() {
        let config = DiscordConfig::new("tok");
        let tool = GetChannelInfoTool::new(config);
        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn get_guild_info_missing_guild_id() {
        let config = DiscordConfig::new("tok");
        let tool = GetGuildInfoTool::new(config);
        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn reply_missing_fields() {
        let config = DiscordConfig::new("tok");
        let tool = ReplyToMessageTool::new(config);

        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));

        let config2 = DiscordConfig::new("tok");
        let tool2 = ReplyToMessageTool::new(config2);
        let err2 = tool2
            .execute(serde_json::json!({"channel_id": "1", "message_id": "2"}))
            .await
            .unwrap_err();
        assert!(matches!(err2, ToolError::InvalidInput(_)));
    }
}
