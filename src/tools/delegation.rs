use std::sync::Arc;

use async_trait::async_trait;

use crate::message::Message;
use crate::provider::{
    CompletionRequest, CompletionResponse, FinishReason, Provider, Usage,
};
use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

// ---------------------------------------------------------------------------
// Sub-agent configuration
// ---------------------------------------------------------------------------

/// Configuration for spawned sub-agents.
#[derive(Debug, Clone)]
pub struct SubAgentConfig {
    /// Maximum number of LLM turns the sub-agent can take before stopping.
    pub max_turns: usize,

    /// Default system prompt to use if the caller doesn't specify one.
    pub default_system_prompt: Option<String>,

    /// Max tokens per completion call.
    pub max_tokens: Option<u32>,

    /// Temperature for completions.
    pub temperature: Option<f32>,
}

impl Default for SubAgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 5,
            default_system_prompt: None,
            max_tokens: None,
            temperature: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Sub-agent runner
// ---------------------------------------------------------------------------

/// Minimal agent loop that runs a task to completion without session persistence.
/// Used internally by the spawn_agent tool.
pub struct SubAgentRunner {
    provider: Arc<dyn Provider>,
    tools: Arc<ToolRegistry>,
    config: SubAgentConfig,
}

impl SubAgentRunner {
    pub fn new(
        provider: Arc<dyn Provider>,
        tools: Arc<ToolRegistry>,
        config: SubAgentConfig,
    ) -> Self {
        Self {
            provider,
            tools,
            config,
        }
    }

    /// Run a sub-agent with the given task. Returns the final assistant message.
    pub async fn run(
        &self,
        task: &str,
        system_prompt: Option<&str>,
    ) -> Result<SubAgentResult, SubAgentError> {
        let sys_prompt = system_prompt
            .or(self.config.default_system_prompt.as_deref())
            .unwrap_or("You are a helpful sub-agent. Complete the given task concisely.");

        let mut messages = vec![
            Message::system(sys_prompt),
            Message::user(task),
        ];

        let mut total_usage = Usage::default();
        let tool_defs = self.tools.definitions();

        for turn in 0..self.config.max_turns {
            let request = CompletionRequest {
                messages: messages.clone(),
                tools: tool_defs.clone(),
                max_tokens: self.config.max_tokens,
                temperature: self.config.temperature,
                model: None,
            };

            let response = self
                .provider
                .complete(request)
                .await
                .map_err(|e| SubAgentError::Provider(e.to_string()))?;

            total_usage.input_tokens += response.usage.input_tokens;
            total_usage.output_tokens += response.usage.output_tokens;

            messages.push(response.message.clone());

            // If the model wants to use tools, execute them and continue the loop
            if response.finish_reason == FinishReason::ToolUse
                && !response.message.tool_calls.is_empty()
            {
                let mut results = Vec::new();

                for call in &response.message.tool_calls {
                    let result = match self.tools.get(&call.name) {
                        Some(tool) => match tool.execute(call.arguments.clone()).await {
                            Ok(output) => crate::message::ToolResult {
                                call_id: call.id.clone(),
                                content: output,
                                is_error: false,
                            },
                            Err(e) => crate::message::ToolResult {
                                call_id: call.id.clone(),
                                content: format!("Error: {}", e),
                                is_error: true,
                            },
                        },
                        None => crate::message::ToolResult {
                            call_id: call.id.clone(),
                            content: format!("Unknown tool: {}", call.name),
                            is_error: true,
                        },
                    };
                    results.push(result);
                }

                messages.push(Message::tool_result(results));
                continue;
            }

            // No tool use — we have a final answer
            return Ok(SubAgentResult {
                content: response.message.content.clone(),
                turns_used: turn + 1,
                usage: total_usage,
            });
        }

        // Exceeded max turns — return whatever we have
        let last_content = messages
            .iter()
            .rev()
            .find(|m| m.role == crate::message::Role::Assistant)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        Ok(SubAgentResult {
            content: last_content,
            turns_used: self.config.max_turns,
            usage: total_usage,
        })
    }
}

// ---------------------------------------------------------------------------
// Results and errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SubAgentResult {
    pub content: String,
    pub turns_used: usize,
    pub usage: Usage,
}

#[derive(Debug, thiserror::Error)]
pub enum SubAgentError {
    #[error("provider error: {0}")]
    Provider(String),

    #[error("tool error: {0}")]
    Tool(String),
}

// ---------------------------------------------------------------------------
// SpawnAgent tool
// ---------------------------------------------------------------------------

/// Tool that lets the main agent delegate a subtask to a temporary sub-agent.
/// The sub-agent gets its own conversation context and can use a subset of
/// available tools. This is useful for breaking complex tasks into
/// independent pieces that can be handled in isolation.
pub struct SpawnAgentTool {
    runner: Arc<SubAgentRunner>,
}

impl SpawnAgentTool {
    pub fn new(runner: Arc<SubAgentRunner>) -> Self {
        Self { runner }
    }
}

#[async_trait]
impl Tool for SpawnAgentTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "spawn_agent".into(),
            description: "Delegate a task to a sub-agent that runs independently with its own \
                          conversation context. The sub-agent can use tools and will return its \
                          final response. Use this for tasks that benefit from focused, \
                          independent reasoning."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task for the sub-agent to complete"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt to shape the sub-agent's behavior"
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
            .ok_or_else(|| ToolError::InvalidInput("missing 'task'".into()))?;

        let system_prompt = input.get("system_prompt").and_then(|v| v.as_str());

        let result = self
            .runner
            .run(task, system_prompt)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        Ok(format!(
            "[Sub-agent completed in {} turn(s), {} tokens used]\n\n{}",
            result.turns_used,
            result.usage.total_tokens(),
            result.content,
        ))
    }
}

// ---------------------------------------------------------------------------
// DelegateToAgent tool — route a task to a named agent's runtime
// ---------------------------------------------------------------------------

/// Tool that lets an agent delegate a task to another named agent.
/// Unlike `SpawnAgentTool`, this uses an existing agent's full runtime
/// (with its own system prompt, model, and tools) rather than creating
/// an ephemeral sub-agent.
pub struct DelegateToAgentTool {
    /// Map of lowercase agent names to their runtimes.
    runtimes: Arc<tokio::sync::RwLock<std::collections::HashMap<String, Arc<crate::runtime::Runtime<crate::context::CharEstimator>>>>>,
    /// Name of the agent that owns this tool (to prevent self-delegation).
    self_name: String,
}

impl DelegateToAgentTool {
    pub fn new(
        runtimes: Arc<tokio::sync::RwLock<std::collections::HashMap<String, Arc<crate::runtime::Runtime<crate::context::CharEstimator>>>>>,
        self_name: String,
    ) -> Self {
        Self { runtimes, self_name }
    }
}

#[async_trait]
impl Tool for DelegateToAgentTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "delegate_to_agent".into(),
            description: "Delegate a task to another named agent. The target agent \
                          has its own personality, system prompt, and capabilities. \
                          Use this when another agent is better suited for a task."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": "Name of the target agent (case-insensitive)"
                    },
                    "task": {
                        "type": "string",
                        "description": "The task or question to delegate to the agent"
                    }
                },
                "required": ["agent", "task"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let agent_name = input
            .get("agent")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'agent'".into()))?;

        let task = input
            .get("task")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'task'".into()))?;

        let key = agent_name.to_lowercase();

        // Prevent self-delegation loops
        if key == self.self_name.to_lowercase() {
            return Err(ToolError::ExecutionFailed(
                "cannot delegate to yourself".into(),
            ));
        }

        let runtimes = self.runtimes.read().await;
        let runtime = runtimes
            .get(&key)
            .ok_or_else(|| {
                let available: Vec<&str> = runtimes.keys().map(|k| k.as_str()).collect();
                ToolError::ExecutionFailed(format!(
                    "agent '{}' not found. Available agents: {}",
                    agent_name,
                    available.join(", ")
                ))
            })?
            .clone();
        drop(runtimes);

        // Create a temporary namespace for this delegation
        let ns = crate::namespace::Namespace::parse(&format!(
            "delegation:{}:{}",
            self.self_name.to_lowercase(),
            uuid::Uuid::new_v4()
        ));

        let result = runtime
            .run(&ns, crate::message::Message::user(task))
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("agent '{}' failed: {}", agent_name, e)))?;

        Ok(format!(
            "[Agent '{}' responded ({} turns, {} tokens)]\n\n{}",
            agent_name,
            result.turns.len(),
            result.total_usage.total_tokens(),
            result.final_message.content,
        ))
    }
}

// ---------------------------------------------------------------------------
// Convenience registration
// ---------------------------------------------------------------------------

/// Create a sub-agent runner with the given provider and tool set.
pub fn create_runner(
    provider: Arc<dyn Provider>,
    tools: Arc<ToolRegistry>,
    config: SubAgentConfig,
) -> Arc<SubAgentRunner> {
    Arc::new(SubAgentRunner::new(provider, tools, config))
}

/// Register the spawn_agent tool into a tool registry.
pub fn register_tool(registry: &mut ToolRegistry, runner: &Arc<SubAgentRunner>) {
    registry.register(Box::new(SpawnAgentTool::new(runner.clone())));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, ToolCall};
    use crate::provider::{CompletionResponse, ProviderError};

    // A mock provider that returns a fixed response
    struct FixedProvider {
        response: String,
    }

    #[async_trait]
    impl Provider for FixedProvider {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            Ok(CompletionResponse {
                message: Message::assistant(&self.response),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
                finish_reason: FinishReason::Stop,
            })
        }
    }

    // A provider that makes a tool call on the first turn, then gives a final answer
    struct ToolUsingProvider {
        call_count: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl Provider for ToolUsingProvider {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            let count = self
                .call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

            if count == 0 {
                // First call: make a tool call
                Ok(CompletionResponse {
                    message: Message::assistant_with_tool_calls(
                        "Let me look that up.",
                        vec![ToolCall {
                            id: "call_1".into(),
                            name: "echo".into(),
                            arguments: serde_json::json!({"text": "hello world"}),
                        }],
                    ),
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 8,
                    },
                    finish_reason: FinishReason::ToolUse,
                })
            } else {
                // Second call: final answer
                Ok(CompletionResponse {
                    message: Message::assistant("The answer is: hello world"),
                    usage: Usage {
                        input_tokens: 20,
                        output_tokens: 6,
                    },
                    finish_reason: FinishReason::Stop,
                })
            }
        }
    }

    // Simple echo tool for tests
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".into(),
                description: "Echoes input back".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"}
                    },
                    "required": ["text"]
                }),
            }
        }

        async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
            let text = input
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidInput("missing 'text'".into()))?;
            Ok(text.to_string())
        }
    }

    fn make_tools() -> Arc<ToolRegistry> {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        Arc::new(registry)
    }

    #[tokio::test]
    async fn sub_agent_simple_response() {
        let provider = Arc::new(FixedProvider {
            response: "42".into(),
        });
        let tools = make_tools();
        let config = SubAgentConfig::default();

        let runner = SubAgentRunner::new(provider, tools, config);
        let result = runner.run("What is the meaning of life?", None).await.unwrap();

        assert_eq!(result.content, "42");
        assert_eq!(result.turns_used, 1);
        assert_eq!(result.usage.total_tokens(), 15);
    }

    #[tokio::test]
    async fn sub_agent_with_tool_use() {
        let provider = Arc::new(ToolUsingProvider {
            call_count: std::sync::atomic::AtomicUsize::new(0),
        });
        let tools = make_tools();
        let config = SubAgentConfig::default();

        let runner = SubAgentRunner::new(provider, tools, config);
        let result = runner.run("Echo hello world", None).await.unwrap();

        assert_eq!(result.content, "The answer is: hello world");
        assert_eq!(result.turns_used, 2);
        assert_eq!(result.usage.input_tokens, 30);
        assert_eq!(result.usage.output_tokens, 14);
    }

    #[tokio::test]
    async fn sub_agent_custom_system_prompt() {
        let provider = Arc::new(FixedProvider {
            response: "Done.".into(),
        });
        let tools = make_tools();
        let config = SubAgentConfig::default();

        let runner = SubAgentRunner::new(provider, tools, config);
        let result = runner
            .run("Do the thing", Some("You are a specialist."))
            .await
            .unwrap();

        assert_eq!(result.content, "Done.");
    }

    #[tokio::test]
    async fn spawn_agent_tool_works() {
        let provider = Arc::new(FixedProvider {
            response: "Result from sub-agent".into(),
        });
        let tools = make_tools();
        let runner = Arc::new(SubAgentRunner::new(
            provider,
            tools,
            SubAgentConfig::default(),
        ));

        let tool = SpawnAgentTool::new(runner);
        let output = tool
            .execute(serde_json::json!({
                "task": "Summarize this document"
            }))
            .await
            .unwrap();

        assert!(output.contains("Result from sub-agent"));
        assert!(output.contains("1 turn(s)"));
    }

    #[tokio::test]
    async fn spawn_agent_tool_missing_task() {
        let provider = Arc::new(FixedProvider {
            response: "ok".into(),
        });
        let tools = make_tools();
        let runner = Arc::new(SubAgentRunner::new(
            provider,
            tools,
            SubAgentConfig::default(),
        ));

        let tool = SpawnAgentTool::new(runner);
        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn tool_definition_is_valid() {
        let provider: Arc<dyn Provider> = Arc::new(FixedProvider {
            response: "ok".into(),
        });
        let tools = make_tools();
        let runner = Arc::new(SubAgentRunner::new(
            provider,
            tools,
            SubAgentConfig::default(),
        ));

        let tool = SpawnAgentTool::new(runner);
        let def = tool.definition();
        assert_eq!(def.name, "spawn_agent");
        assert!(def.input_schema["required"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("task")));
    }

    #[test]
    fn sub_agent_config_defaults() {
        let config = SubAgentConfig::default();
        assert_eq!(config.max_turns, 5);
        assert!(config.default_system_prompt.is_none());
        assert!(config.max_tokens.is_none());
        assert!(config.temperature.is_none());
    }
}
