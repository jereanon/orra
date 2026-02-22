use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot, RwLock};

use crate::hook::Hook;
use crate::message::ToolCall;
use crate::namespace::Namespace;
use crate::store::Session;

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
pub struct ApprovalHook {
    /// Channel to send approval requests to the UI handler.
    request_tx: mpsc::Sender<ApprovalRequest>,
    /// Per-session chaos mode flag, read from session metadata.
    chaos_mode: RwLock<bool>,
}

impl ApprovalHook {
    pub fn new(request_tx: mpsc::Sender<ApprovalRequest>) -> Self {
        Self {
            request_tx,
            chaos_mode: RwLock::new(false),
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

    async fn before_tool_call(&self, call: &mut ToolCall) -> Result<(), String> {
        // If chaos mode is enabled, auto-approve everything
        if *self.chaos_mode.read().await {
            return Ok(());
        }

        // Create a oneshot channel for the approval response
        let (response_tx, response_rx) = oneshot::channel();

        let request = ApprovalRequest {
            call_id: call.id.clone(),
            tool_name: call.name.clone(),
            arguments: call.arguments.clone(),
            response_tx,
        };

        // Send the approval request to the UI handler
        if self.request_tx.send(request).await.is_err() {
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
        let result = hook.before_tool_call(&mut call).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn approval_approved() {
        let (tx, mut rx) = mpsc::channel(1);
        let hook = ApprovalHook::new(tx);

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

        let result = hook.before_tool_call(&mut call).await;
        assert!(result.is_ok());
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn approval_denied() {
        let (tx, mut rx) = mpsc::channel(1);
        let hook = ApprovalHook::new(tx);

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

        let result = hook.before_tool_call(&mut call).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("denied"));
        handle.await.unwrap();
    }

    #[tokio::test]
    async fn closed_channel_auto_approves() {
        let (tx, rx) = mpsc::channel(1);
        let hook = ApprovalHook::new(tx);

        // Drop the receiver to simulate no active connection
        drop(rx);

        let mut call = ToolCall {
            id: "c1".into(),
            name: "exec".into(),
            arguments: serde_json::json!({"command": "ls"}),
        };

        // Should auto-approve when no listener is available
        let result = hook.before_tool_call(&mut call).await;
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
}
