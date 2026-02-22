//! Shell command execution tool with allowlist and timeout support.

use async_trait::async_trait;

use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

/// Execute shell commands with an allowlist and timeout.
pub struct ExecTool {
    allowed_commands: Vec<String>,
    timeout_secs: u64,
}

impl ExecTool {
    pub fn new(allowed_commands: Vec<String>, timeout_secs: u64) -> Self {
        Self {
            allowed_commands,
            timeout_secs,
        }
    }

    /// Check if a command is allowed. If the allowlist is empty, all commands
    /// are permitted. Otherwise, the command's first word (program name) must
    /// match an entry in the allowlist.
    fn is_allowed(&self, command: &str) -> bool {
        if self.allowed_commands.is_empty() {
            return true;
        }

        let program = command
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_lowercase();

        self.allowed_commands
            .iter()
            .any(|allowed| allowed.to_lowercase() == program)
    }
}

#[async_trait]
impl Tool for ExecTool {
    fn definition(&self) -> ToolDefinition {
        let description = if self.allowed_commands.is_empty() {
            "Execute a shell command and return its output.".to_string()
        } else {
            format!(
                "Execute a shell command and return its output. Allowed commands: {}",
                self.allowed_commands.join(", ")
            )
        };

        ToolDefinition {
            name: "exec".into(),
            description,
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": format!(
                            "Timeout in seconds (default: {}, max: 120)",
                            self.timeout_secs
                        )
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Working directory to run the command in (optional)"
                    }
                },
                "required": ["command"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'command'".into()))?;

        if !self.is_allowed(command) {
            return Err(ToolError::InvalidInput(format!(
                "command '{}' is not in the allowed list: [{}]",
                command.split_whitespace().next().unwrap_or(command),
                self.allowed_commands.join(", ")
            )));
        }

        let timeout = input
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.timeout_secs)
            .min(120);

        let working_dir = input.get("working_directory").and_then(|v| v.as_str());

        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg(command);
        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        let result =
            tokio::time::timeout(std::time::Duration::from_secs(timeout), cmd.output()).await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let exit_code = output.status.code().unwrap_or(-1);

                let mut result = String::new();

                if !stdout.is_empty() {
                    result.push_str(&stdout);
                }

                if !stderr.is_empty() {
                    if !result.is_empty() {
                        result.push('\n');
                    }
                    result.push_str("[stderr]\n");
                    result.push_str(&stderr);
                }

                if result.is_empty() {
                    result = "(no output)".into();
                }

                if exit_code != 0 {
                    result.push_str(&format!("\n[exit code: {exit_code}]"));
                }

                // Truncate very long output
                const MAX_OUTPUT: usize = 32_000;
                if result.len() > MAX_OUTPUT {
                    result.truncate(MAX_OUTPUT);
                    result.push_str("\n[output truncated]");
                }

                Ok(result)
            }
            Ok(Err(e)) => Err(ToolError::ExecutionFailed(format!(
                "failed to run command: {e}"
            ))),
            Err(_) => Err(ToolError::ExecutionFailed(format!(
                "command timed out after {timeout} seconds"
            ))),
        }
    }
}

/// Register the exec tool into a registry.
pub fn register_tool(
    registry: &mut ToolRegistry,
    allowed_commands: Vec<String>,
    timeout_secs: u64,
) {
    registry.register(Box::new(ExecTool::new(allowed_commands, timeout_secs)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_empty_permits_all() {
        let tool = ExecTool::new(vec![], 30);
        assert!(tool.is_allowed("ls -la"));
        assert!(tool.is_allowed("curl https://example.com"));
    }

    #[test]
    fn allowed_list_filters() {
        let tool = ExecTool::new(vec!["ls".into(), "cat".into()], 30);
        assert!(tool.is_allowed("ls -la"));
        assert!(tool.is_allowed("cat /etc/hosts"));
        assert!(!tool.is_allowed("rm -rf /"));
        assert!(!tool.is_allowed("curl https://example.com"));
    }

    #[test]
    fn allowed_case_insensitive() {
        let tool = ExecTool::new(vec!["LS".into()], 30);
        assert!(tool.is_allowed("ls -la"));
    }

    #[test]
    fn tool_definition_includes_allowed_list() {
        let tool = ExecTool::new(vec!["ls".into(), "cat".into()], 30);
        let def = tool.definition();
        assert_eq!(def.name, "exec");
        assert!(def.description.contains("ls, cat"));
    }

    #[test]
    fn tool_definition_no_allowed_list() {
        let tool = ExecTool::new(vec![], 30);
        let def = tool.definition();
        assert!(!def.description.contains("Allowed commands"));
    }

    #[tokio::test]
    async fn missing_command_returns_error() {
        let tool = ExecTool::new(vec![], 30);
        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn disallowed_command_returns_error() {
        let tool = ExecTool::new(vec!["ls".into()], 30);
        let err = tool
            .execute(serde_json::json!({"command": "rm -rf /"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
        assert!(err.to_string().contains("not in the allowed list"));
    }

    #[tokio::test]
    async fn echo_command_works() {
        let tool = ExecTool::new(vec![], 30);
        let result = tool
            .execute(serde_json::json!({"command": "echo hello"}))
            .await
            .unwrap();
        assert!(result.contains("hello"));
    }

    #[tokio::test]
    async fn working_directory_changes_cwd() {
        let tool = ExecTool::new(vec![], 30);
        let result = tool
            .execute(serde_json::json!({
                "command": "pwd",
                "working_directory": "/tmp"
            }))
            .await
            .unwrap();
        assert!(result.trim().ends_with("/tmp") || result.trim() == "/private/tmp");
    }

    #[tokio::test]
    async fn failing_command_shows_exit_code() {
        let tool = ExecTool::new(vec![], 30);
        let result = tool
            .execute(serde_json::json!({"command": "sh -c 'exit 42'"}))
            .await
            .unwrap();
        assert!(result.contains("exit code: 42"));
    }
}
