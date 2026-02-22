use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot, RwLock};

use crate::hook::Hook;
use crate::message::ToolCall;
use crate::namespace::Namespace;
use crate::store::Session;

#[cfg(feature = "discord")]
use crate::tools::discord::DiscordConfig;

/// A pending approval request sent from the hook to the UI handler.
pub struct ApprovalRequest {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
    /// The hook blocks on the receiver side of this channel until the
    /// UI handler sends `true` (approved) or `false` (denied).
    pub response_tx: oneshot::Sender<bool>,
}

/// A hook that gates tool execution on user approval.
///
/// In normal mode, every tool call blocks in `before_tool_call` until the
/// user approves or denies it via the UI. In "chaos mode" (per-session),
/// all tool calls are auto-approved.
///
/// When a Discord config is provided via `with_discord()`, approval requests
/// from Discord sessions (namespace root == "discord") are sent as Discord
/// messages and the hook polls for a yes/no reply from the same user.
pub struct ApprovalHook {
    /// Channel to send approval requests to the UI handler.
    request_tx: mpsc::Sender<ApprovalRequest>,
    /// Per-session chaos mode flag, read from session metadata.
    chaos_mode: RwLock<bool>,
    /// Optional Discord config for sending approval prompts to Discord channels.
    #[cfg(feature = "discord")]
    discord: Option<DiscordConfig>,
}

impl ApprovalHook {
    pub fn new(request_tx: mpsc::Sender<ApprovalRequest>) -> Self {
        Self {
            request_tx,
            chaos_mode: RwLock::new(false),
            #[cfg(feature = "discord")]
            discord: None,
        }
    }

    /// Enable Discord-native approvals. When a tool call originates from a
    /// Discord session, the approval prompt is sent as a Discord message
    /// instead of going through the WebSocket UI channel.
    #[cfg(feature = "discord")]
    pub fn with_discord(mut self, discord: DiscordConfig) -> Self {
        self.discord = Some(discord);
        self
    }
}

/// Send the approval request through the UI channel (WebSocket path).
async fn ui_approval(
    request_tx: &mpsc::Sender<ApprovalRequest>,
    call: &ToolCall,
) -> Result<(), String> {
    let (response_tx, response_rx) = oneshot::channel();

    let request = ApprovalRequest {
        call_id: call.id.clone(),
        tool_name: call.name.clone(),
        arguments: call.arguments.clone(),
        response_tx,
    };

    // Send the approval request to the UI handler
    if request_tx.send(request).await.is_err() {
        // Channel closed — no active connection, auto-approve
        return Ok(());
    }

    // Block until the user responds
    match response_rx.await {
        Ok(true) => Ok(()),
        Ok(false) => Err(format!(
            "Tool call '{}' was denied by the user.",
            call.name
        )),
        Err(_) => {
            // Sender dropped — connection lost, auto-approve to avoid
            // permanently blocking
            Ok(())
        }
    }
}

/// Send an approval prompt to a Discord channel and poll for a yes/no reply.
#[cfg(feature = "discord")]
async fn discord_approval(
    discord: &DiscordConfig,
    namespace: &Namespace,
    call: &ToolCall,
) -> Result<(), String> {
    let segments = namespace.segments();
    // Namespace format: discord:<guild>:<channel>:<user>
    if segments.len() < 4 {
        // Can't determine channel/user, auto-approve
        return Ok(());
    }
    let channel_id = &segments[2];
    let author_id = &segments[3];

    // Format the approval message
    let args_str = serde_json::to_string_pretty(&call.arguments)
        .unwrap_or_else(|_| call.arguments.to_string());
    let prompt_content = format!(
        "**Tool approval required**\n`{}` wants to run with:\n```json\n{}\n```\nReply **yes** or **no**.",
        call.name, args_str
    );

    // Truncate if over 2000 chars (Discord message limit)
    let prompt_content = if prompt_content.len() > 2000 {
        format!(
            "**Tool approval required**\n`{}` wants to run. Arguments too large to display.\nReply **yes** or **no**.",
            call.name
        )
    } else {
        prompt_content
    };

    // Send the prompt message
    let send_resp = discord
        .request(
            reqwest::Method::POST,
            &format!("channels/{}/messages", channel_id),
        )
        .json(&serde_json::json!({ "content": prompt_content }))
        .send()
        .await
        .map_err(|e| format!("Failed to send approval prompt: {}", e))?;

    if !send_resp.status().is_success() {
        // Can't send to Discord, auto-approve
        return Ok(());
    }

    // Parse the sent message to get its ID for polling "after"
    let sent_msg: serde_json::Value = send_resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse sent message: {}", e))?;
    let prompt_msg_id = sent_msg["id"]
        .as_str()
        .ok_or_else(|| "Missing message ID in response".to_string())?
        .to_string();

    // Poll for a reply from the same user
    let timeout = std::time::Duration::from_secs(300); // 5 minutes
    let poll_interval = std::time::Duration::from_secs(2);
    let start = std::time::Instant::now();

    loop {
        if start.elapsed() > timeout {
            return Err(format!(
                "Tool call '{}' timed out waiting for Discord approval.",
                call.name
            ));
        }

        tokio::time::sleep(poll_interval).await;

        // Get messages after our prompt
        let msgs_resp = discord
            .request(
                reqwest::Method::GET,
                &format!("channels/{}/messages", channel_id),
            )
            .query(&[("after", &prompt_msg_id), ("limit", &"10".to_string())])
            .send()
            .await;

        let msgs_resp = match msgs_resp {
            Ok(r) => r,
            Err(_) => continue, // Retry on network error
        };

        if !msgs_resp.status().is_success() {
            continue;
        }

        let messages: Vec<serde_json::Value> = match msgs_resp.json().await {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Look for a message from the same author that says yes/no
        for msg in &messages {
            let msg_author_id = msg["author"]["id"].as_str().unwrap_or("");
            if msg_author_id != author_id {
                continue;
            }

            let content = msg["content"]
                .as_str()
                .unwrap_or("")
                .trim()
                .to_lowercase();

            if content == "yes" || content == "y" || content == "approve" {
                return Ok(());
            }
            if content == "no" || content == "n" || content == "deny" {
                return Err(format!(
                    "Tool call '{}' was denied by the user.",
                    call.name
                ));
            }
        }
    }
}

#[async_trait]
impl Hook for ApprovalHook {
    async fn after_session_load(&self, _namespace: &Namespace, session: &Session) {
        let chaos = session
            .metadata
            .get("chaos_mode")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        *self.chaos_mode.write().await = chaos;
    }

    async fn before_tool_call(&self, namespace: &Namespace, call: &mut ToolCall) -> Result<(), String> {
        // If chaos mode is enabled, auto-approve everything
        if *self.chaos_mode.read().await {
            return Ok(());
        }

        // Route based on namespace origin
        #[cfg(feature = "discord")]
        if namespace.root() == "discord" {
            if let Some(ref discord) = self.discord {
                return discord_approval(discord, namespace, call).await;
            }
            // No Discord config available, auto-approve
            return Ok(());
        }

        // Web UI approval flow (existing behavior)
        ui_approval(&self.request_tx, call).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ToolCall;
    use crate::namespace::Namespace;
    use crate::store::Session;

    #[tokio::test]
    async fn chaos_mode_auto_approves() {
        let (tx, _rx) = mpsc::channel(1);
        let hook = ApprovalHook::new(tx);

        // Enable chaos mode via session metadata
        let ns = Namespace::new("test");
        let mut session = Session::new(ns.clone());
        session
            .metadata
            .insert("chaos_mode".into(), serde_json::json!(true));
        hook.after_session_load(&ns, &session).await;

        let mut call = ToolCall {
            id: "c1".into(),
            name: "exec".into(),
            arguments: serde_json::json!({"command": "ls"}),
        };

        // Should auto-approve without blocking
        let result = hook.before_tool_call(&ns, &mut call).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn approval_approved() {
        let (tx, mut rx) = mpsc::channel(1);
        let hook = ApprovalHook::new(tx);
        let ns = Namespace::new("test");

        let mut call = ToolCall {
            id: "c1".into(),
            name: "exec".into(),
            arguments: serde_json::json!({"command": "pwd"}),
        };

        // Spawn a task to approve the request
        let handle = tokio::spawn(async move {
            let req = rx.recv().await.unwrap();
            assert_eq!(req.tool_name, "exec");
            assert_eq!(req.call_id, "c1");
            req.response_tx.send(true).unwrap();
        });

        let result = hook.before_tool_call(&ns, &mut call).await;
        assert!(result.is_ok());
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn approval_denied() {
        let (tx, mut rx) = mpsc::channel(1);
        let hook = ApprovalHook::new(tx);
        let ns = Namespace::new("test");

        let mut call = ToolCall {
            id: "c1".into(),
            name: "exec".into(),
            arguments: serde_json::json!({"command": "rm -rf /"}),
        };

        // Spawn a task to deny the request
        let handle = tokio::spawn(async move {
            let req = rx.recv().await.unwrap();
            req.response_tx.send(false).unwrap();
        });

        let result = hook.before_tool_call(&ns, &mut call).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("denied"));
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn closed_channel_auto_approves() {
        let (tx, rx) = mpsc::channel(1);
        let hook = ApprovalHook::new(tx);
        let ns = Namespace::new("test");

        // Drop the receiver to simulate no active connection
        drop(rx);

        let mut call = ToolCall {
            id: "c1".into(),
            name: "exec".into(),
            arguments: serde_json::json!({"command": "ls"}),
        };

        // Should auto-approve when no listener is available
        let result = hook.before_tool_call(&ns, &mut call).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn chaos_mode_reads_from_session() {
        let (tx, _rx) = mpsc::channel(1);
        let hook = ApprovalHook::new(tx);
        let ns = Namespace::new("test");

        // Default: chaos mode off
        let session = Session::new(ns.clone());
        hook.after_session_load(&ns, &session).await;
        assert!(!*hook.chaos_mode.read().await);

        // Set chaos mode on
        let mut session2 = Session::new(ns.clone());
        session2
            .metadata
            .insert("chaos_mode".into(), serde_json::json!(true));
        hook.after_session_load(&ns, &session2).await;
        assert!(*hook.chaos_mode.read().await);

        // Clear chaos mode
        let session3 = Session::new(ns.clone());
        hook.after_session_load(&ns, &session3).await;
        assert!(!*hook.chaos_mode.read().await);
    }

    #[cfg(feature = "discord")]
    #[tokio::test]
    async fn discord_namespace_without_config_auto_approves() {
        let (tx, _rx) = mpsc::channel(1);
        let hook = ApprovalHook::new(tx);

        // Discord namespace but no DiscordConfig set — should auto-approve
        let ns = Namespace::new("discord")
            .child("guild123")
            .child("chan456")
            .child("user789");

        let mut call = ToolCall {
            id: "c1".into(),
            name: "exec".into(),
            arguments: serde_json::json!({"command": "ls"}),
        };

        let result = hook.before_tool_call(&ns, &mut call).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn non_discord_namespace_uses_ui_channel() {
        let (tx, mut rx) = mpsc::channel(1);
        let hook = ApprovalHook::new(tx);
        let ns = Namespace::new("web").child("session-123");

        let mut call = ToolCall {
            id: "c1".into(),
            name: "exec".into(),
            arguments: serde_json::json!({"command": "pwd"}),
        };

        // Spawn a task to approve via the UI channel
        let handle = tokio::spawn(async move {
            let req = rx.recv().await.unwrap();
            assert_eq!(req.tool_name, "exec");
            req.response_tx.send(true).unwrap();
        });

        let result = hook.before_tool_call(&ns, &mut call).await;
        assert!(result.is_ok());
        handle.await.unwrap();
    }
}
