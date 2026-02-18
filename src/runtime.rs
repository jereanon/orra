use std::sync::Arc;

use crate::context::{ContextBudget, ContextWindow, Tokenizer};
use crate::message::{Message, ToolCall, ToolResult};
use crate::namespace::Namespace;
use crate::policy::PolicyRegistry;
use crate::provider::{CompletionRequest, CompletionResponse, FinishReason, Provider, ProviderError, Usage};
use crate::store::{Session, SessionStore, StoreError};
use crate::tool::ToolRegistry;

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub system_prompt: Option<String>,
    pub max_turns: usize,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub context_budget: ContextBudget,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            max_turns: 10,
            max_tokens: None,
            temperature: None,
            context_budget: ContextBudget::default(),
        }
    }
}

#[derive(Debug)]
pub struct TurnResult {
    pub response: CompletionResponse,
    pub tool_results: Vec<ToolResult>,
}

#[derive(Debug)]
pub struct RunResult {
    pub final_message: Message,
    pub turns: Vec<TurnResult>,
    pub total_usage: Usage,
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("provider error: {0}")]
    Provider(#[from] ProviderError),

    #[error("store error: {0}")]
    Store(#[from] StoreError),

    #[error("max turns ({0}) exceeded")]
    MaxTurnsExceeded(usize),
}

pub struct Runtime<T: Tokenizer> {
    provider: Arc<dyn Provider>,
    store: Arc<dyn SessionStore>,
    tools: ToolRegistry,
    policies: PolicyRegistry,
    context_window: ContextWindow<T>,
    config: RuntimeConfig,
}

impl<T: Tokenizer> Runtime<T> {
    pub fn new(
        provider: Arc<dyn Provider>,
        store: Arc<dyn SessionStore>,
        tools: ToolRegistry,
        policies: PolicyRegistry,
        tokenizer: T,
        config: RuntimeConfig,
    ) -> Self {
        let context_window = ContextWindow::new(tokenizer, config.context_budget.clone());
        Self {
            provider,
            store,
            tools,
            policies,
            context_window,
            config,
        }
    }

    pub async fn run(
        &self,
        namespace: &Namespace,
        user_message: Message,
    ) -> Result<RunResult, RuntimeError> {
        let mut session = self
            .store
            .load(namespace)
            .await?
            .unwrap_or_else(|| Session::new(namespace.clone()));

        session.push_message(user_message);

        let mut turns = Vec::new();
        let mut total_usage = Usage::default();

        for _ in 0..self.config.max_turns {
            let messages = self.build_messages(&session);
            let all_defs = self.tools.definitions();
            let policy = self.policies.resolve(namespace);
            let tool_defs = policy.filter_definitions(&all_defs);

            let request = CompletionRequest {
                messages,
                tools: tool_defs,
                max_tokens: self.config.max_tokens,
                temperature: self.config.temperature,
            };

            let response = self.provider.complete(request).await?;
            total_usage.input_tokens += response.usage.input_tokens;
            total_usage.output_tokens += response.usage.output_tokens;

            session.push_message(response.message.clone());

            if response.finish_reason == FinishReason::ToolUse && !response.message.tool_calls.is_empty() {
                let tool_results = self.execute_tool_calls(&response.message.tool_calls).await;
                let result_message = Message::tool_result(tool_results.clone());
                session.push_message(result_message);

                turns.push(TurnResult {
                    response,
                    tool_results,
                });
            } else {
                let final_message = response.message.clone();
                turns.push(TurnResult {
                    response,
                    tool_results: vec![],
                });

                self.store.save(&session).await?;

                return Ok(RunResult {
                    final_message,
                    turns,
                    total_usage,
                });
            }
        }

        // Exceeded max turns — save what we have and return an error
        self.store.save(&session).await?;
        Err(RuntimeError::MaxTurnsExceeded(self.config.max_turns))
    }

    fn build_messages(&self, session: &Session) -> Vec<Message> {
        let mut messages = Vec::new();

        if let Some(ref system_prompt) = self.config.system_prompt {
            messages.push(Message::system(system_prompt.clone()));
        }

        messages.extend(session.messages.clone());

        // Truncate if over budget
        if self.context_window.is_over_budget(&messages) {
            messages = self.context_window.truncate_to_fit(&messages);
        }

        messages
    }

    async fn execute_tool_calls(&self, tool_calls: &[ToolCall]) -> Vec<ToolResult> {
        let mut results = Vec::new();

        for call in tool_calls {
            let result = if let Some(tool) = self.tools.get(&call.name) {
                match tool.execute(call.arguments.clone()).await {
                    Ok(content) => {
                        let content = self.context_window.truncate_tool_result(&content);
                        ToolResult {
                            call_id: call.id.clone(),
                            content,
                            is_error: false,
                        }
                    }
                    Err(e) => ToolResult {
                        call_id: call.id.clone(),
                        content: e.to_string(),
                        is_error: true,
                    },
                }
            } else {
                ToolResult {
                    call_id: call.id.clone(),
                    content: format!("unknown tool: {}", call.name),
                    is_error: true,
                }
            };

            results.push(result);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CharEstimator;
    use crate::message::ToolCall;
    use crate::provider::{CompletionResponse, FinishReason, Usage};
    use crate::store::InMemoryStore;
    use crate::tool::{Tool, ToolDefinition, ToolError};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // --- Mock provider ---

    struct MockProvider {
        responses: Vec<CompletionResponse>,
        call_count: AtomicUsize,
    }

    impl MockProvider {
        fn new(responses: Vec<CompletionResponse>) -> Self {
            Self {
                responses,
                call_count: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            if idx < self.responses.len() {
                Ok(self.responses[idx].clone())
            } else {
                Err(ProviderError::Other("no more responses".into()))
            }
        }
    }

    // --- Mock tools ---

    struct UppercaseTool;

    #[async_trait]
    impl Tool for UppercaseTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "uppercase".into(),
                description: "Uppercases text".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }),
            }
        }

        async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
            let text = input["text"].as_str().unwrap();
            Ok(text.to_uppercase())
        }
    }

    fn make_runtime(
        responses: Vec<CompletionResponse>,
        tools: ToolRegistry,
        config: RuntimeConfig,
    ) -> Runtime<CharEstimator> {
        let provider = Arc::new(MockProvider::new(responses));
        let store = Arc::new(InMemoryStore::new());
        Runtime::new(provider, store, tools, PolicyRegistry::default(), CharEstimator::default(), config)
    }

    #[tokio::test]
    async fn simple_conversation_no_tools() {
        let responses = vec![CompletionResponse {
            message: Message::assistant("Hello! How can I help?"),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 8,
            },
            finish_reason: FinishReason::Stop,
        }];

        let runtime = make_runtime(responses, ToolRegistry::new(), RuntimeConfig::default());
        let ns = Namespace::new("test");
        let result = runtime.run(&ns, Message::user("Hi")).await.unwrap();

        assert_eq!(result.final_message.content, "Hello! How can I help?");
        assert_eq!(result.turns.len(), 1);
        assert_eq!(result.total_usage.total_tokens(), 18);
    }

    #[tokio::test]
    async fn conversation_with_tool_call() {
        let tool_call = ToolCall {
            id: "call_1".into(),
            name: "uppercase".into(),
            arguments: serde_json::json!({"text": "hello"}),
        };

        let responses = vec![
            // First response: assistant wants to use a tool
            CompletionResponse {
                message: Message::assistant_with_tool_calls("Let me uppercase that.", vec![tool_call]),
                usage: Usage { input_tokens: 10, output_tokens: 15 },
                finish_reason: FinishReason::ToolUse,
            },
            // Second response: after tool result, assistant gives final answer
            CompletionResponse {
                message: Message::assistant("The uppercased text is: HELLO"),
                usage: Usage { input_tokens: 30, output_tokens: 10 },
                finish_reason: FinishReason::Stop,
            },
        ];

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(UppercaseTool));

        let runtime = make_runtime(responses, tools, RuntimeConfig::default());
        let ns = Namespace::new("test");
        let result = runtime.run(&ns, Message::user("Uppercase hello")).await.unwrap();

        assert_eq!(result.final_message.content, "The uppercased text is: HELLO");
        assert_eq!(result.turns.len(), 2);
        assert_eq!(result.turns[0].tool_results.len(), 1);
        assert_eq!(result.turns[0].tool_results[0].content, "HELLO");
        assert!(!result.turns[0].tool_results[0].is_error);
        assert_eq!(result.total_usage.input_tokens, 40);
        assert_eq!(result.total_usage.output_tokens, 25);
    }

    #[tokio::test]
    async fn unknown_tool_returns_error_result() {
        let tool_call = ToolCall {
            id: "call_1".into(),
            name: "nonexistent".into(),
            arguments: serde_json::json!({}),
        };

        let responses = vec![
            CompletionResponse {
                message: Message::assistant_with_tool_calls("Using a tool.", vec![tool_call]),
                usage: Usage { input_tokens: 5, output_tokens: 5 },
                finish_reason: FinishReason::ToolUse,
            },
            CompletionResponse {
                message: Message::assistant("Sorry, that tool doesn't exist."),
                usage: Usage { input_tokens: 10, output_tokens: 8 },
                finish_reason: FinishReason::Stop,
            },
        ];

        let runtime = make_runtime(responses, ToolRegistry::new(), RuntimeConfig::default());
        let ns = Namespace::new("test");
        let result = runtime.run(&ns, Message::user("Do something")).await.unwrap();

        assert_eq!(result.turns[0].tool_results[0].content, "unknown tool: nonexistent");
        assert!(result.turns[0].tool_results[0].is_error);
    }

    #[tokio::test]
    async fn max_turns_exceeded() {
        // Provider always returns tool calls, never stops
        let tool_call = ToolCall {
            id: "call_1".into(),
            name: "uppercase".into(),
            arguments: serde_json::json!({"text": "hi"}),
        };

        let response = CompletionResponse {
            message: Message::assistant_with_tool_calls("Using tool.", vec![tool_call]),
            usage: Usage { input_tokens: 5, output_tokens: 5 },
            finish_reason: FinishReason::ToolUse,
        };

        let responses: Vec<CompletionResponse> = (0..5).map(|_| response.clone()).collect();

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(UppercaseTool));

        let config = RuntimeConfig {
            max_turns: 3,
            ..RuntimeConfig::default()
        };

        let runtime = make_runtime(responses, tools, config);
        let ns = Namespace::new("test");
        let err = runtime.run(&ns, Message::user("Loop")).await.unwrap_err();

        match err {
            RuntimeError::MaxTurnsExceeded(n) => assert_eq!(n, 3),
            _ => panic!("expected MaxTurnsExceeded"),
        }
    }

    #[tokio::test]
    async fn session_persists_across_runs() {
        let store = Arc::new(InMemoryStore::new());
        let ns = Namespace::new("test");

        // First run
        {
            let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
                message: Message::assistant("First response"),
                usage: Usage::default(),
                finish_reason: FinishReason::Stop,
            }]));

            let runtime = Runtime::new(
                provider,
                store.clone(),
                ToolRegistry::new(),
                PolicyRegistry::default(),
                CharEstimator::default(),
                RuntimeConfig::default(),
            );
            runtime.run(&ns, Message::user("First")).await.unwrap();
        }

        // Check session was saved
        let session = store.load(&ns).await.unwrap().unwrap();
        assert_eq!(session.message_count(), 2); // user + assistant

        // Second run
        {
            let provider = Arc::new(MockProvider::new(vec![CompletionResponse {
                message: Message::assistant("Second response"),
                usage: Usage::default(),
                finish_reason: FinishReason::Stop,
            }]));

            let runtime = Runtime::new(
                provider,
                store.clone(),
                ToolRegistry::new(),
                PolicyRegistry::default(),
                CharEstimator::default(),
                RuntimeConfig::default(),
            );
            runtime.run(&ns, Message::user("Second")).await.unwrap();
        }

        // Session should now have all 4 messages
        let session = store.load(&ns).await.unwrap().unwrap();
        assert_eq!(session.message_count(), 4);
    }

    #[tokio::test]
    async fn system_prompt_prepended() {
        struct InspectingProvider {
            had_system_prompt: tokio::sync::Mutex<bool>,
        }

        #[async_trait]
        impl Provider for InspectingProvider {
            async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
                if let Some(first) = request.messages.first() {
                    if first.role == crate::message::Role::System {
                        *self.had_system_prompt.lock().await = true;
                    }
                }
                Ok(CompletionResponse {
                    message: Message::assistant("Ok"),
                    usage: Usage::default(),
                    finish_reason: FinishReason::Stop,
                })
            }
        }

        let provider = Arc::new(InspectingProvider {
            had_system_prompt: tokio::sync::Mutex::new(false),
        });

        let config = RuntimeConfig {
            system_prompt: Some("You are helpful.".into()),
            ..RuntimeConfig::default()
        };

        let runtime = Runtime::new(
            provider.clone(),
            Arc::new(InMemoryStore::new()),
            ToolRegistry::new(),
            PolicyRegistry::default(),
            CharEstimator::default(),
            config,
        );

        let ns = Namespace::new("test");
        runtime.run(&ns, Message::user("Hi")).await.unwrap();

        assert!(*provider.had_system_prompt.lock().await);
    }

    #[tokio::test]
    async fn different_namespaces_have_isolated_sessions() {
        let store = Arc::new(InMemoryStore::new());

        let provider1 = Arc::new(MockProvider::new(vec![CompletionResponse {
            message: Message::assistant("Response for Alice"),
            usage: Usage::default(),
            finish_reason: FinishReason::Stop,
        }]));

        let provider2 = Arc::new(MockProvider::new(vec![CompletionResponse {
            message: Message::assistant("Response for Bob"),
            usage: Usage::default(),
            finish_reason: FinishReason::Stop,
        }]));

        let ns_alice = Namespace::new("acme").child("alice");
        let ns_bob = Namespace::new("acme").child("bob");

        let runtime1 = Runtime::new(
            provider1,
            store.clone(),
            ToolRegistry::new(),
            PolicyRegistry::default(),
            CharEstimator::default(),
            RuntimeConfig::default(),
        );
        runtime1.run(&ns_alice, Message::user("Hi from Alice")).await.unwrap();

        let runtime2 = Runtime::new(
            provider2,
            store.clone(),
            ToolRegistry::new(),
            PolicyRegistry::default(),
            CharEstimator::default(),
            RuntimeConfig::default(),
        );
        runtime2.run(&ns_bob, Message::user("Hi from Bob")).await.unwrap();

        let alice_session = store.load(&ns_alice).await.unwrap().unwrap();
        let bob_session = store.load(&ns_bob).await.unwrap().unwrap();

        assert_eq!(alice_session.messages[0].content, "Hi from Alice");
        assert_eq!(alice_session.messages[1].content, "Response for Alice");
        assert_eq!(bob_session.messages[0].content, "Hi from Bob");
        assert_eq!(bob_session.messages[1].content, "Response for Bob");
    }

    #[tokio::test]
    async fn policy_filters_tools_sent_to_provider() {
        use crate::policy::ToolPolicy;

        struct CapturingProvider {
            seen_tools: tokio::sync::Mutex<Vec<Vec<String>>>,
        }

        #[async_trait]
        impl Provider for CapturingProvider {
            async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
                let tool_names: Vec<String> = request.tools.iter().map(|t| t.name.clone()).collect();
                self.seen_tools.lock().await.push(tool_names);
                Ok(CompletionResponse {
                    message: Message::assistant("Done"),
                    usage: Usage::default(),
                    finish_reason: FinishReason::Stop,
                })
            }
        }

        let provider = Arc::new(CapturingProvider {
            seen_tools: tokio::sync::Mutex::new(Vec::new()),
        });

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(UppercaseTool));

        struct ReadTool;
        #[async_trait]
        impl Tool for ReadTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "read".into(),
                    description: "Read a file".into(),
                    input_schema: serde_json::json!({"type": "object"}),
                }
            }
            async fn execute(&self, _input: serde_json::Value) -> Result<String, ToolError> {
                Ok("file contents".into())
            }
        }
        tools.register(Box::new(ReadTool));

        // Only allow "read" for the restricted namespace
        let mut policies = PolicyRegistry::default();
        let restricted_ns = Namespace::new("restricted");
        policies.set_policy(&restricted_ns, ToolPolicy::AllowList(vec!["read".into()]));

        let runtime = Runtime::new(
            provider.clone(),
            Arc::new(InMemoryStore::new()),
            tools,
            policies,
            CharEstimator::default(),
            RuntimeConfig::default(),
        );

        // Run in the restricted namespace
        runtime.run(&restricted_ns, Message::user("Hello")).await.unwrap();

        let seen = provider.seen_tools.lock().await;
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0], vec!["read".to_string()]);
    }
}
