//! Claude Code CLI delegation tool.
//!
//! Spawns the locally installed `claude` CLI in non-interactive mode to handle
//! coding tasks. Supports session persistence so the calling agent can resume
//! previous work via session IDs.
//!
//! Requires the `claude-code` feature flag.

use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

/// Max characters returned in a single tool result. Keeps context windows
/// and downstream message limits manageable.
const MAX_OUTPUT_CHARS: usize = 8000;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Shared configuration for both claude_code tools.
#[derive(Debug, Clone)]
pub struct ClaudeCodeConfig {
    /// Tools the CLI is allowed to use without prompting.
    pub allowed_tools: Vec<String>,
    /// Maximum agentic turns per invocation.
    pub max_turns: u32,
    /// Timeout in seconds for the entire CLI process.
    pub timeout_secs: u64,
    /// If true, pass --dangerously-skip-permissions (fully autonomous).
    pub skip_permissions: bool,
    /// Working directory for spawned processes.
    pub working_directory: Option<PathBuf>,
}

impl Default for ClaudeCodeConfig {
    fn default() -> Self {
        Self {
            allowed_tools: vec!["Read".into(), "Edit".into(), "Bash".into()],
            max_turns: 10,
            timeout_secs: 300,
            skip_permissions: false,
            working_directory: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register both `claude_code` and `claude_code_resume` tools into a registry.
pub fn register_tools(registry: &mut ToolRegistry, config: ClaudeCodeConfig) {
    registry.register(Box::new(ClaudeCodeTool::new(config.clone())));
    registry.register(Box::new(ClaudeCodeResumeTool::new(config)));
}

// ---------------------------------------------------------------------------
// CLI response parsing
// ---------------------------------------------------------------------------

/// Matches the JSON structure returned by `claude -p ... --output-format json`.
#[derive(Debug, Deserialize)]
struct ClaudeOutput {
    #[serde(default)]
    result: String,
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    duration_ms: u64,
    #[serde(default)]
    is_error: bool,
}

/// Truncate a string to `max` characters, appending an indicator if clipped.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut t = s[..max].to_string();
        t.push_str("\n\n[output truncated]");
        t
    }
}

/// Build the base argument list shared by both tools.
fn base_args(config: &ClaudeCodeConfig) -> Vec<String> {
    let mut args = vec![
        "--output-format".into(),
        "json".into(),
        "--max-turns".into(),
        config.max_turns.to_string(),
    ];

    if !config.allowed_tools.is_empty() {
        args.push("--allowedTools".into());
        args.push(config.allowed_tools.join(","));
    }

    if config.skip_permissions {
        args.push("--dangerously-skip-permissions".into());
    }

    args
}

/// Run the claude CLI with the given arguments, enforcing a timeout.
async fn run_claude(
    args: &[String],
    working_dir: Option<&PathBuf>,
    timeout_secs: u64,
) -> Result<String, ToolError> {
    let mut cmd = tokio::process::Command::new("claude");
    cmd.args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    if let Some(dir) = working_dir {
        cmd.current_dir(dir);
    }

    let output = tokio::time::timeout(Duration::from_secs(timeout_secs), cmd.output())
        .await
        .map_err(|_| {
            ToolError::ExecutionFailed(format!(
                "Claude CLI timed out after {timeout_secs} seconds. Consider breaking the task \
                 into smaller pieces or increasing the timeout."
            ))
        })?
        .map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Failed to run claude CLI: {e}. Make sure the `claude` command is \
                 installed and available in PATH."
            ))
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() && stdout.is_empty() {
        return Err(ToolError::ExecutionFailed(format!(
            "Claude CLI exited with status {}.\nstderr: {}",
            output.status,
            truncate(&stderr, 2000),
        )));
    }

    Ok(stdout.to_string())
}

/// Parse the JSON output from the CLI and format a response string.
fn format_response(raw_json: &str) -> Result<String, ToolError> {
    let parsed: ClaudeOutput = serde_json::from_str(raw_json).map_err(|e| {
        ToolError::ExecutionFailed(format!(
            "Failed to parse claude CLI output: {}.\nRaw output: {}",
            e,
            truncate(raw_json, 500),
        ))
    })?;

    if parsed.is_error {
        return Err(ToolError::ExecutionFailed(format!(
            "Claude CLI reported an error: {}",
            truncate(&parsed.result, 2000),
        )));
    }

    let result = truncate(&parsed.result, MAX_OUTPUT_CHARS);

    Ok(format!(
        "[session_id: {} | completed in {:.1}s]\n\n{}",
        parsed.session_id,
        parsed.duration_ms as f64 / 1000.0,
        result,
    ))
}

// ---------------------------------------------------------------------------
// Tool 1: claude_code (new task)
// ---------------------------------------------------------------------------

/// Delegates a coding task to the locally installed Claude CLI.
pub struct ClaudeCodeTool {
    config: ClaudeCodeConfig,
}

impl ClaudeCodeTool {
    pub fn new(config: ClaudeCodeConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Tool for ClaudeCodeTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "claude_code".into(),
            description: "Delegate a coding task to the locally installed Claude CLI. \
                          It can read, write, and execute code autonomously. Returns \
                          a session_id you can use with claude_code_resume to continue."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The coding task to delegate (e.g., 'Write a Python script that...', 'Fix the bug in auth.py', 'Add tests for the parser module')"
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Optional working directory for the task. Defaults to the configured directory."
                    }
                },
                "required": ["task"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let task = input
            .get("task")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing required field 'task'".into()))?;

        if task.trim().is_empty() {
            return Err(ToolError::InvalidInput("task cannot be empty".into()));
        }

        // Determine working directory: per-call override > config default
        let work_dir = input
            .get("working_directory")
            .and_then(|v| v.as_str())
            .map(PathBuf::from)
            .or_else(|| self.config.working_directory.clone());

        let mut args = vec!["-p".into(), task.to_string()];
        args.extend(base_args(&self.config));

        let raw = run_claude(&args, work_dir.as_ref(), self.config.timeout_secs).await?;
        format_response(&raw)
    }
}

// ---------------------------------------------------------------------------
// Tool 2: claude_code_resume (continue session)
// ---------------------------------------------------------------------------

/// Resumes a previous Claude CLI session by its session ID.
pub struct ClaudeCodeResumeTool {
    config: ClaudeCodeConfig,
}

impl ClaudeCodeResumeTool {
    pub fn new(config: ClaudeCodeConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Tool for ClaudeCodeResumeTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "claude_code_resume".into(),
            description: "Resume a previous Claude CLI session using its session_id. \
                          Use this to continue work on a coding task or ask follow-up \
                          questions about the code."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session_id returned by a previous claude_code call"
                    },
                    "follow_up": {
                        "type": "string",
                        "description": "Follow-up instruction or question for the session"
                    }
                },
                "required": ["session_id", "follow_up"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let session_id = input
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing required field 'session_id'".into()))?;

        let follow_up = input
            .get("follow_up")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing required field 'follow_up'".into()))?;

        if session_id.trim().is_empty() {
            return Err(ToolError::InvalidInput("session_id cannot be empty".into()));
        }

        if follow_up.trim().is_empty() {
            return Err(ToolError::InvalidInput("follow_up cannot be empty".into()));
        }

        let mut args = vec![
            "--resume".into(),
            session_id.to_string(),
            "-p".into(),
            follow_up.to_string(),
        ];
        args.extend(base_args(&self.config));

        let work_dir = self.config.working_directory.as_ref();
        let raw = run_claude(&args, work_dir, self.config.timeout_secs).await?;
        format_response(&raw)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ClaudeCodeConfig {
        ClaudeCodeConfig {
            allowed_tools: vec!["Read".into(), "Edit".into(), "Bash".into()],
            max_turns: 10,
            timeout_secs: 120,
            skip_permissions: false,
            working_directory: None,
        }
    }

    #[test]
    fn claude_code_tool_definition() {
        let tool = ClaudeCodeTool::new(test_config());
        let def = tool.definition();
        assert_eq!(def.name, "claude_code");
        assert!(def.description.contains("coding task"));

        let schema = &def.input_schema;
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("task")));
    }

    #[test]
    fn resume_tool_definition() {
        let tool = ClaudeCodeResumeTool::new(test_config());
        let def = tool.definition();
        assert_eq!(def.name, "claude_code_resume");

        let schema = &def.input_schema;
        let required = schema["required"].as_array().unwrap();
        assert!(required.contains(&serde_json::json!("session_id")));
        assert!(required.contains(&serde_json::json!("follow_up")));
    }

    #[tokio::test]
    async fn claude_code_rejects_empty_task() {
        let tool = ClaudeCodeTool::new(test_config());
        let result = tool.execute(serde_json::json!({"task": ""})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn claude_code_rejects_missing_task() {
        let tool = ClaudeCodeTool::new(test_config());
        let result = tool.execute(serde_json::json!({})).await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn resume_rejects_empty_session_id() {
        let tool = ClaudeCodeResumeTool::new(test_config());
        let result = tool
            .execute(serde_json::json!({"session_id": "", "follow_up": "test"}))
            .await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[tokio::test]
    async fn resume_rejects_missing_follow_up() {
        let tool = ClaudeCodeResumeTool::new(test_config());
        let result = tool
            .execute(serde_json::json!({"session_id": "abc123"}))
            .await;
        assert!(matches!(result, Err(ToolError::InvalidInput(_))));
    }

    #[test]
    fn truncate_short_string_unchanged() {
        let s = "hello";
        assert_eq!(truncate(s, 100), "hello");
    }

    #[test]
    fn truncate_long_string_clipped() {
        let s = "a".repeat(200);
        let result = truncate(&s, 50);
        assert!(result.len() < 200);
        assert!(result.contains("[output truncated]"));
    }

    #[test]
    fn format_response_success() {
        let json =
            r#"{"result":"done","session_id":"sess_123","duration_ms":5000,"is_error":false}"#;
        let result = format_response(json).unwrap();
        assert!(result.contains("sess_123"));
        assert!(result.contains("5.0s"));
        assert!(result.contains("done"));
    }

    #[test]
    fn format_response_error() {
        let json = r#"{"result":"something broke","session_id":"sess_456","duration_ms":1000,"is_error":true}"#;
        let result = format_response(json);
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[test]
    fn format_response_invalid_json() {
        let result = format_response("not json");
        assert!(matches!(result, Err(ToolError::ExecutionFailed(_))));
    }

    #[test]
    fn base_args_includes_allowed_tools() {
        let config = test_config();
        let args = base_args(&config);
        assert!(args.contains(&"--allowedTools".into()));
        assert!(args.contains(&"Read,Edit,Bash".to_string()));
    }

    #[test]
    fn base_args_includes_skip_permissions() {
        let mut config = test_config();
        config.skip_permissions = true;
        let args = base_args(&config);
        assert!(args.contains(&"--dangerously-skip-permissions".into()));
    }

    #[test]
    fn base_args_no_skip_permissions_by_default() {
        let config = test_config();
        let args = base_args(&config);
        assert!(!args.contains(&"--dangerously-skip-permissions".into()));
    }

    #[test]
    fn default_config_has_sensible_values() {
        let config = ClaudeCodeConfig::default();
        assert_eq!(config.allowed_tools, vec!["Read", "Edit", "Bash"]);
        assert_eq!(config.max_turns, 10);
        assert_eq!(config.timeout_secs, 300);
        assert!(!config.skip_permissions);
        assert!(config.working_directory.is_none());
    }

    #[test]
    fn register_tools_adds_both() {
        let mut registry = ToolRegistry::new();
        register_tools(&mut registry, ClaudeCodeConfig::default());
        assert_eq!(registry.len(), 2);
        assert!(registry.get("claude_code").is_some());
        assert!(registry.get("claude_code_resume").is_some());
    }
}
