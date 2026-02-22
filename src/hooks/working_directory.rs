use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::hook::Hook;
use crate::message::ToolCall;
use crate::namespace::Namespace;
use crate::store::Session;

/// A hook that injects `working_directory` from session metadata into tool calls.
///
/// When a session has `"working_directory"` in its metadata, this hook captures
/// it on session load and injects it into `exec` and `claude_code` tool calls
/// (unless the call already specifies a `working_directory`).
pub struct WorkingDirectoryHook {
    current_dir: RwLock<Option<String>>,
}

impl WorkingDirectoryHook {
    pub fn new() -> Self {
        Self {
            current_dir: RwLock::new(None),
        }
    }
}

impl Default for WorkingDirectoryHook {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool names that support the `working_directory` argument.
const DIR_AWARE_TOOLS: &[&str] = &["exec", "claude_code"];

#[async_trait]
impl Hook for WorkingDirectoryHook {
    async fn after_session_load(&self, _namespace: &Namespace, session: &Session) {
        let dir = session
            .metadata
            .get("working_directory")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        *self.current_dir.write().await = dir;
    }

    async fn before_tool_call(
        &self,
        _namespace: &Namespace,
        call: &mut ToolCall,
    ) -> Result<(), String> {
        if !DIR_AWARE_TOOLS.contains(&call.name.as_str()) {
            return Ok(());
        }

        let dir = self.current_dir.read().await;
        let dir = match dir.as_deref() {
            Some(d) => d,
            None => return Ok(()),
        };

        // Don't override if the call already specifies a working directory
        if let Some(obj) = call.arguments.as_object_mut() {
            if !obj.contains_key("working_directory") {
                obj.insert(
                    "working_directory".into(),
                    serde_json::Value::String(dir.to_string()),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ToolCall;
    use crate::namespace::Namespace;
    use crate::store::Session;

    #[tokio::test]
    async fn captures_dir_from_session_metadata() {
        let hook = WorkingDirectoryHook::new();
        let ns = Namespace::new("test");
        let mut session = Session::new(ns.clone());
        session.metadata.insert(
            "working_directory".into(),
            serde_json::json!("/home/user/project"),
        );

        hook.after_session_load(&ns, &session).await;

        let dir = hook.current_dir.read().await;
        assert_eq!(dir.as_deref(), Some("/home/user/project"));
    }

    #[tokio::test]
    async fn clears_dir_when_not_in_metadata() {
        let hook = WorkingDirectoryHook::new();
        let ns = Namespace::new("test");

        // First, set a directory
        let mut session = Session::new(ns.clone());
        session
            .metadata
            .insert("working_directory".into(), serde_json::json!("/some/path"));
        hook.after_session_load(&ns, &session).await;

        // Now load a session without the metadata
        let session2 = Session::new(ns.clone());
        hook.after_session_load(&ns, &session2).await;

        let dir = hook.current_dir.read().await;
        assert!(dir.is_none());
    }

    #[tokio::test]
    async fn injects_dir_into_exec_tool() {
        let hook = WorkingDirectoryHook::new();
        let ns = Namespace::new("test");
        let mut session = Session::new(ns.clone());
        session
            .metadata
            .insert("working_directory".into(), serde_json::json!("/my/project"));
        hook.after_session_load(&ns, &session).await;

        let mut call = ToolCall {
            id: "c1".into(),
            name: "exec".into(),
            arguments: serde_json::json!({"command": "ls"}),
        };

        let _ = hook.before_tool_call(&ns, &mut call).await;

        assert_eq!(
            call.arguments
                .get("working_directory")
                .and_then(|v| v.as_str()),
            Some("/my/project")
        );
    }

    #[tokio::test]
    async fn injects_dir_into_claude_code_tool() {
        let hook = WorkingDirectoryHook::new();
        let ns = Namespace::new("test");
        let mut session = Session::new(ns.clone());
        session
            .metadata
            .insert("working_directory".into(), serde_json::json!("/my/project"));
        hook.after_session_load(&ns, &session).await;

        let mut call = ToolCall {
            id: "c1".into(),
            name: "claude_code".into(),
            arguments: serde_json::json!({"prompt": "list files"}),
        };

        let _ = hook.before_tool_call(&ns, &mut call).await;

        assert_eq!(
            call.arguments
                .get("working_directory")
                .and_then(|v| v.as_str()),
            Some("/my/project")
        );
    }

    #[tokio::test]
    async fn does_not_override_explicit_dir() {
        let hook = WorkingDirectoryHook::new();
        let ns = Namespace::new("test");
        let mut session = Session::new(ns.clone());
        session.metadata.insert(
            "working_directory".into(),
            serde_json::json!("/session/dir"),
        );
        hook.after_session_load(&ns, &session).await;

        let mut call = ToolCall {
            id: "c1".into(),
            name: "exec".into(),
            arguments: serde_json::json!({
                "command": "ls",
                "working_directory": "/explicit/dir"
            }),
        };

        let _ = hook.before_tool_call(&ns, &mut call).await;

        assert_eq!(
            call.arguments
                .get("working_directory")
                .and_then(|v| v.as_str()),
            Some("/explicit/dir")
        );
    }

    #[tokio::test]
    async fn ignores_non_dir_aware_tools() {
        let hook = WorkingDirectoryHook::new();
        let ns = Namespace::new("test");
        let mut session = Session::new(ns.clone());
        session
            .metadata
            .insert("working_directory".into(), serde_json::json!("/my/project"));
        hook.after_session_load(&ns, &session).await;

        let mut call = ToolCall {
            id: "c1".into(),
            name: "web_search".into(),
            arguments: serde_json::json!({"query": "rust"}),
        };

        let _ = hook.before_tool_call(&ns, &mut call).await;

        assert!(call.arguments.get("working_directory").is_none());
    }

    #[tokio::test]
    async fn no_injection_when_no_dir_set() {
        let hook = WorkingDirectoryHook::new();
        let ns = Namespace::new("test");

        let mut call = ToolCall {
            id: "c1".into(),
            name: "exec".into(),
            arguments: serde_json::json!({"command": "pwd"}),
        };

        let _ = hook.before_tool_call(&ns, &mut call).await;

        assert!(call.arguments.get("working_directory").is_none());
    }
}
