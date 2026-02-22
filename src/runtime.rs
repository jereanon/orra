use std::sync::Arc;

use crate::context::{ContextBudget, ContextWindow, Tokenizer};
use crate::message::{Message, ToolCall, ToolResult};
use crate::namespace::Namespace;
use crate::policy::PolicyRegistry;
use crate::provider::{
    CompletionRequest, CompletionResponse, FinishReason, Provider, ProviderError,
    StreamEvent, StreamingProvider, Usage,
};
use crate::store::{Session, SessionStore, StoreError};
use crate::hook::HookRegistry;
use crate::tool::ToolRegistry;

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub system_prompt: Option<String>,
    pub max_turns: usize,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub context_budget: ContextBudget,
    pub parallel_tool_execution: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            max_turns: 10,
            max_tokens: None,
            temperature: None,
            context_budget: ContextBudget::default(),
            parallel_tool_execution: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TurnResult {
    pub response: CompletionResponse,
    pub tool_results: Vec<ToolResult>,
}

#[derive(Debug, Clone)]
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

/// Events emitted during a streaming run of the agent loop.
#[derive(Debug)]
pub enum RuntimeStreamEvent {
    /// A chunk of text from the assistant.
    TextDelta(String),

    /// A tool call has started.
    ToolCallStarted { id: String, name: String },

    /// A tool call has completed execution.
    ToolCallCompleted { id: String, result: ToolResult },

    /// A full turn (LLM call + tool executions) has completed.
    TurnCompleted(TurnResult),

    /// The entire run is complete.
    Done(RunResult),

    /// An error occurred.
    Error(String),
}

pub struct Runtime<T: Tokenizer> {
    provider: Arc<dyn Provider>,
    streaming_provider: Option<Arc<dyn StreamingProvider>>,
    store: Arc<dyn SessionStore>,
    tools: ToolRegistry,
    policies: PolicyRegistry,
    context_window: ContextWindow<T>,
    config: RuntimeConfig,
    hooks: HookRegistry,
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
            streaming_provider: None,
            store,
            tools,
            policies,
            context_window,
            config,
            hooks: HookRegistry::default(),
        }
    }

    pub fn set_hooks(&mut self, hooks: HookRegistry) {
        self.hooks = hooks;
    }

    pub fn set_streaming_provider(&mut self, provider: Arc<dyn StreamingProvider>) {
        self.streaming_provider = Some(provider);
    }

    pub async fn run(
        &self,
        namespace: &Namespace,
        user_message: Message,
    ) -> Result<RunResult, RuntimeError> {
        self.run_with_model(namespace, user_message, None).await
    }

    /// Run the agent loop with an optional model override.
    pub async fn run_with_model(
        &self,
        namespace: &Namespace,
        user_message: Message,
        model: Option<String>,
    ) -> Result<RunResult, RuntimeError> {
        let mut session = self
            .store
            .load(namespace)
            .await?
            .unwrap_or_else(|| Session::new(namespace.clone()));

        self.hooks.dispatch_after_session_load(namespace, &session).await;

        session.push_message(user_message);

        let mut turns = Vec::new();
        let mut total_usage = Usage::default();

        for _ in 0..self.config.max_turns {
            let messages = self.build_messages(&session);
            let all_defs = self.tools.definitions();
            let policy = self.policies.resolve(namespace);
            let tool_defs = policy.filter_definitions(&all_defs);

            let mut request = CompletionRequest {
                messages,
                tools: tool_defs,
                max_tokens: self.config.max_tokens,
                temperature: self.config.temperature,
                model: model.clone(),
            };

            self.hooks.dispatch_before_provider_call(&mut request).await;

            let response = self.provider.complete(request).await?;

            self.hooks.dispatch_after_provider_call(&response).await;

            total_usage.input_tokens += response.usage.input_tokens;
            total_usage.output_tokens += response.usage.output_tokens;

            session.push_message(response.message.clone());

            if response.finish_reason == FinishReason::ToolUse && !response.message.tool_calls.is_empty() {
                let tool_results = self.execute_tool_calls(namespace, &response.message.tool_calls).await;
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

                self.hooks.dispatch_before_session_save(namespace, &mut session).await;
                self.store.save(&session).await?;

                return Ok(RunResult {
                    final_message,
                    turns,
                    total_usage,
                });
            }
        }

        // Exceeded max turns — save what we have and return an error
        self.hooks.dispatch_before_session_save(namespace, &mut session).await;
        self.store.save(&session).await?;
        Err(RuntimeError::MaxTurnsExceeded(self.config.max_turns))
    }

    /// Run the agent loop with streaming, emitting events as they happen.
    ///
    /// Requires a streaming provider to be set via `set_streaming_provider`.
    /// Returns a receiver that yields `RuntimeStreamEvent`s as the agent loop progresses.
    pub async fn run_streaming(
        &self,
        namespace: &Namespace,
        user_message: Message,
    ) -> Result<tokio::sync::mpsc::Receiver<RuntimeStreamEvent>, RuntimeError> {
        self.run_streaming_with_model(namespace, user_message, None).await
    }

    /// Run the agent loop with streaming and an optional model override.
    pub async fn run_streaming_with_model(
        &self,
        namespace: &Namespace,
        user_message: Message,
        model: Option<String>,
    ) -> Result<tokio::sync::mpsc::Receiver<RuntimeStreamEvent>, RuntimeError> {
        let streaming_provider = self
            .streaming_provider
            .as_ref()
            .ok_or_else(|| RuntimeError::Provider(ProviderError::Other(
                "no streaming provider configured".into(),
            )))?
            .clone();

        let mut session = self
            .store
            .load(namespace)
            .await?
            .unwrap_or_else(|| Session::new(namespace.clone()));

        self.hooks.dispatch_after_session_load(namespace, &session).await;
        session.push_message(user_message);

        let (tx, rx) = tokio::sync::mpsc::channel(64);

        // We need to collect the state we need and move into the spawned task.
        // Because the runtime borrows &self, we extract what we need.
        let store = self.store.clone();
        let config = self.config.clone();
        let namespace = namespace.clone();
        let messages_snapshot = self.build_messages(&session);
        let all_defs = self.tools.definitions();
        let policy = self.policies.resolve(&namespace);
        let tool_defs = policy.filter_definitions(&all_defs);

        let mut request = CompletionRequest {
            messages: messages_snapshot,
            tools: tool_defs,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            model,
        };

        self.hooks.dispatch_before_provider_call(&mut request).await;

        // Start streaming from the provider
        let mut stream_rx = streaming_provider.stream(request).await?;

        // Spawn a task to consume stream events and forward them
        tokio::spawn(async move {
            let mut text_content = String::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut tool_args_buffers: std::collections::HashMap<String, String> = std::collections::HashMap::new();
            let mut usage = Usage::default();
            let mut finish_reason = FinishReason::Stop;

            while let Some(event) = stream_rx.recv().await {
                match event {
                    StreamEvent::TextDelta(text) => {
                        text_content.push_str(&text);
                        if tx.send(RuntimeStreamEvent::TextDelta(text)).await.is_err() {
                            return;
                        }
                    }
                    StreamEvent::ToolCallStart { id, name } => {
                        tool_args_buffers.insert(id.clone(), String::new());
                        if tx.send(RuntimeStreamEvent::ToolCallStarted {
                            id: id.clone(),
                            name: name.clone(),
                        }).await.is_err() {
                            return;
                        }
                        tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments: serde_json::Value::Null,
                        });
                    }
                    StreamEvent::ToolCallDelta { id, arguments_delta } => {
                        if let Some(buf) = tool_args_buffers.get_mut(&id) {
                            buf.push_str(&arguments_delta);
                        }
                    }
                    StreamEvent::Done { usage: u, finish_reason: fr } => {
                        usage = u;
                        finish_reason = fr;
                    }
                    StreamEvent::Error(msg) => {
                        let _ = tx.send(RuntimeStreamEvent::Error(msg)).await;
                        return;
                    }
                }
            }

            // Finalize tool calls with accumulated arguments
            for tc in &mut tool_calls {
                if let Some(args_str) = tool_args_buffers.remove(&tc.id) {
                    tc.arguments = serde_json::from_str(&args_str)
                        .unwrap_or(serde_json::Value::String(args_str));
                }
            }

            // Build the assistant message
            let message = if tool_calls.is_empty() {
                Message::assistant(&text_content)
            } else {
                Message::assistant_with_tool_calls(&text_content, tool_calls.clone())
            };

            session.push_message(message.clone());

            let response = CompletionResponse {
                message,
                usage: usage.clone(),
                finish_reason: finish_reason.clone(),
            };

            let total_usage = usage;
            let mut turns = Vec::new();

            // If tool calls were made, we can't continue the agent loop in streaming mode
            // (that would require a non-streaming follow-up). Emit what we have.
            if finish_reason == FinishReason::ToolUse && !tool_calls.is_empty() {
                // Note: For a full streaming agent loop, we'd need the tool registry here.
                // For now, emit the turn result so the caller knows tools need execution.
                turns.push(TurnResult {
                    response,
                    tool_results: vec![],
                });

                let result = RunResult {
                    final_message: session.messages.last().cloned().unwrap_or_else(|| Message::assistant("")),
                    turns,
                    total_usage,
                };

                let _ = tx.send(RuntimeStreamEvent::Done(result)).await;
            } else {
                turns.push(TurnResult {
                    response,
                    tool_results: vec![],
                });

                let result = RunResult {
                    final_message: session.messages.last().cloned().unwrap_or_else(|| Message::assistant("")),
                    turns,
                    total_usage,
                };

                // Save session
                if let Err(e) = store.save(&session).await {
                    let _ = tx.send(RuntimeStreamEvent::Error(format!("save error: {e}"))).await;
                    return;
                }

                let _ = tx.send(RuntimeStreamEvent::Done(result)).await;
            }
        });

        Ok(rx)
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

    async fn execute_single_tool_call(&self, namespace: &Namespace, call: &ToolCall) -> ToolResult {
        let mut call = call.clone();

        // If any hook rejects the tool call, skip execution and return the
        // rejection reason as an error result.
        if let Err(reason) = self.hooks.dispatch_before_tool_call(namespace, &mut call).await {
            return ToolResult {
                call_id: call.id.clone(),
                content: reason,
                is_error: true,
            };
        }

        let mut result = if let Some(tool) = self.tools.get(&call.name) {
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

        self.hooks.dispatch_after_tool_call(&call, &mut result).await;
        result
    }

    async fn execute_tool_calls(&self, namespace: &Namespace, tool_calls: &[ToolCall]) -> Vec<ToolResult> {
        #[cfg(feature = "parallel-tools")]
        {
            if self.config.parallel_tool_execution && tool_calls.len() > 1 {
                let futs: Vec<_> = tool_calls
                    .iter()
                    .map(|call| self.execute_single_tool_call(namespace, call))
                    .collect();
                return futures::future::join_all(futs).await;
            }
        }

        // Sequential fallback
        let mut results = Vec::new();
        for call in tool_calls {
            results.push(self.execute_single_tool_call(namespace, call).await);
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

    // --- Parallel tool execution tests ---

    struct SlowTool {
        name: String,
        delay_ms: u64,
    }

    #[async_trait]
    impl Tool for SlowTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.clone(),
                description: format!("Sleeps for {}ms", self.delay_ms),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        async fn execute(&self, _input: serde_json::Value) -> Result<String, ToolError> {
            tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            Ok(format!("done:{}", self.name))
        }
    }

    #[cfg(feature = "parallel-tools")]
    struct FailingTool;

    #[cfg(feature = "parallel-tools")]
    #[async_trait]
    impl Tool for FailingTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "failing".into(),
                description: "Always fails".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        async fn execute(&self, _input: serde_json::Value) -> Result<String, ToolError> {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            Err(ToolError::ExecutionFailed("boom".into()))
        }
    }

    #[cfg(feature = "parallel-tools")]
    #[tokio::test]
    async fn parallel_execution_faster_than_sequential() {
        let tool_calls = vec![
            ToolCall { id: "c1".into(), name: "slow_a".into(), arguments: serde_json::json!({}) },
            ToolCall { id: "c2".into(), name: "slow_b".into(), arguments: serde_json::json!({}) },
            ToolCall { id: "c3".into(), name: "slow_c".into(), arguments: serde_json::json!({}) },
        ];

        let responses = vec![
            CompletionResponse {
                message: Message::assistant_with_tool_calls("Running tools.", tool_calls.clone()),
                usage: Usage { input_tokens: 10, output_tokens: 10 },
                finish_reason: FinishReason::ToolUse,
            },
            CompletionResponse {
                message: Message::assistant("All done."),
                usage: Usage { input_tokens: 20, output_tokens: 5 },
                finish_reason: FinishReason::Stop,
            },
        ];

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(SlowTool { name: "slow_a".into(), delay_ms: 100 }));
        tools.register(Box::new(SlowTool { name: "slow_b".into(), delay_ms: 100 }));
        tools.register(Box::new(SlowTool { name: "slow_c".into(), delay_ms: 100 }));

        let config = RuntimeConfig {
            parallel_tool_execution: true,
            ..RuntimeConfig::default()
        };

        let runtime = make_runtime(responses, tools, config);
        let ns = Namespace::new("test");

        let start = std::time::Instant::now();
        let result = runtime.run(&ns, Message::user("Go")).await.unwrap();
        let elapsed = start.elapsed();

        // Parallel: should take ~100ms, not ~300ms
        assert!(elapsed.as_millis() < 250, "took {}ms, expected <250ms", elapsed.as_millis());
        assert_eq!(result.turns[0].tool_results.len(), 3);
        assert_eq!(result.turns[0].tool_results[0].content, "done:slow_a");
        assert_eq!(result.turns[0].tool_results[1].content, "done:slow_b");
        assert_eq!(result.turns[0].tool_results[2].content, "done:slow_c");
    }

    #[tokio::test]
    async fn sequential_execution_when_disabled() {
        let tool_calls = vec![
            ToolCall { id: "c1".into(), name: "slow_a".into(), arguments: serde_json::json!({}) },
            ToolCall { id: "c2".into(), name: "slow_b".into(), arguments: serde_json::json!({}) },
        ];

        let responses = vec![
            CompletionResponse {
                message: Message::assistant_with_tool_calls("Running tools.", tool_calls.clone()),
                usage: Usage { input_tokens: 10, output_tokens: 10 },
                finish_reason: FinishReason::ToolUse,
            },
            CompletionResponse {
                message: Message::assistant("Done."),
                usage: Usage { input_tokens: 20, output_tokens: 5 },
                finish_reason: FinishReason::Stop,
            },
        ];

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(SlowTool { name: "slow_a".into(), delay_ms: 50 }));
        tools.register(Box::new(SlowTool { name: "slow_b".into(), delay_ms: 50 }));

        let config = RuntimeConfig {
            parallel_tool_execution: false,
            ..RuntimeConfig::default()
        };

        let runtime = make_runtime(responses, tools, config);
        let ns = Namespace::new("test");

        let start = std::time::Instant::now();
        let result = runtime.run(&ns, Message::user("Go")).await.unwrap();
        let elapsed = start.elapsed();

        // Sequential: should take >= 100ms (50 + 50)
        assert!(elapsed.as_millis() >= 90, "took {}ms, expected >=90ms", elapsed.as_millis());
        assert_eq!(result.turns[0].tool_results.len(), 2);
    }

    #[cfg(feature = "parallel-tools")]
    #[tokio::test]
    async fn parallel_one_error_doesnt_block_others() {
        let tool_calls = vec![
            ToolCall { id: "c1".into(), name: "slow_a".into(), arguments: serde_json::json!({}) },
            ToolCall { id: "c2".into(), name: "failing".into(), arguments: serde_json::json!({}) },
            ToolCall { id: "c3".into(), name: "slow_b".into(), arguments: serde_json::json!({}) },
        ];

        let responses = vec![
            CompletionResponse {
                message: Message::assistant_with_tool_calls("Running.", tool_calls.clone()),
                usage: Usage { input_tokens: 10, output_tokens: 10 },
                finish_reason: FinishReason::ToolUse,
            },
            CompletionResponse {
                message: Message::assistant("Handled."),
                usage: Usage { input_tokens: 20, output_tokens: 5 },
                finish_reason: FinishReason::Stop,
            },
        ];

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(SlowTool { name: "slow_a".into(), delay_ms: 50 }));
        tools.register(Box::new(FailingTool));
        tools.register(Box::new(SlowTool { name: "slow_b".into(), delay_ms: 50 }));

        let runtime = make_runtime(responses, tools, RuntimeConfig::default());
        let ns = Namespace::new("test");
        let result = runtime.run(&ns, Message::user("Go")).await.unwrap();

        let results = &result.turns[0].tool_results;
        assert_eq!(results.len(), 3);
        assert!(!results[0].is_error);
        assert_eq!(results[0].content, "done:slow_a");
        assert!(results[1].is_error);
        assert!(results[1].content.contains("boom"));
        assert!(!results[2].is_error);
        assert_eq!(results[2].content, "done:slow_b");
    }

    #[cfg(feature = "parallel-tools")]
    #[tokio::test]
    async fn parallel_results_maintain_call_id_ordering() {
        let tool_calls = vec![
            ToolCall { id: "first".into(), name: "slow_a".into(), arguments: serde_json::json!({}) },
            ToolCall { id: "second".into(), name: "slow_b".into(), arguments: serde_json::json!({}) },
        ];

        let responses = vec![
            CompletionResponse {
                message: Message::assistant_with_tool_calls("Go.", tool_calls.clone()),
                usage: Usage { input_tokens: 10, output_tokens: 10 },
                finish_reason: FinishReason::ToolUse,
            },
            CompletionResponse {
                message: Message::assistant("Done."),
                usage: Usage { input_tokens: 20, output_tokens: 5 },
                finish_reason: FinishReason::Stop,
            },
        ];

        let mut tools = ToolRegistry::new();
        // slow_a takes longer but should still be first in results
        tools.register(Box::new(SlowTool { name: "slow_a".into(), delay_ms: 80 }));
        tools.register(Box::new(SlowTool { name: "slow_b".into(), delay_ms: 10 }));

        let runtime = make_runtime(responses, tools, RuntimeConfig::default());
        let ns = Namespace::new("test");
        let result = runtime.run(&ns, Message::user("Go")).await.unwrap();

        let results = &result.turns[0].tool_results;
        assert_eq!(results[0].call_id, "first");
        assert_eq!(results[1].call_id, "second");
    }

    // --- Hook integration tests ---

    #[tokio::test]
    async fn hooks_called_during_run() {
        use crate::hook::{Hook, HookRegistry};
        use std::sync::atomic::AtomicUsize;

        struct CountHook {
            before_provider: AtomicUsize,
            after_provider: AtomicUsize,
            before_save: AtomicUsize,
            after_load: AtomicUsize,
        }

        #[async_trait]
        impl Hook for CountHook {
            async fn after_session_load(&self, _ns: &Namespace, _s: &Session) {
                self.after_load.fetch_add(1, Ordering::SeqCst);
            }
            async fn before_provider_call(&self, _req: &mut CompletionRequest) {
                self.before_provider.fetch_add(1, Ordering::SeqCst);
            }
            async fn after_provider_call(&self, _resp: &CompletionResponse) {
                self.after_provider.fetch_add(1, Ordering::SeqCst);
            }
            async fn before_session_save(&self, _ns: &Namespace, _s: &mut Session) {
                self.before_save.fetch_add(1, Ordering::SeqCst);
            }
        }

        let hook = Arc::new(CountHook {
            before_provider: AtomicUsize::new(0),
            after_provider: AtomicUsize::new(0),
            before_save: AtomicUsize::new(0),
            after_load: AtomicUsize::new(0),
        });

        let mut hooks = HookRegistry::new();
        hooks.register(hook.clone());

        let responses = vec![CompletionResponse {
            message: Message::assistant("Done"),
            usage: Usage::default(),
            finish_reason: FinishReason::Stop,
        }];

        let mut runtime = make_runtime(responses, ToolRegistry::new(), RuntimeConfig::default());
        runtime.set_hooks(hooks);

        let ns = Namespace::new("test");
        runtime.run(&ns, Message::user("Hi")).await.unwrap();

        assert_eq!(hook.after_load.load(Ordering::SeqCst), 1);
        assert_eq!(hook.before_provider.load(Ordering::SeqCst), 1);
        assert_eq!(hook.after_provider.load(Ordering::SeqCst), 1);
        assert_eq!(hook.before_save.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn hook_modifies_provider_request() {
        use crate::hook::{Hook, HookRegistry};

        struct ForceTemp;

        #[async_trait]
        impl Hook for ForceTemp {
            async fn before_provider_call(&self, request: &mut CompletionRequest) {
                request.temperature = Some(0.0);
            }
        }

        struct CapturingProvider {
            temps: tokio::sync::Mutex<Vec<Option<f32>>>,
        }

        #[async_trait]
        impl Provider for CapturingProvider {
            async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
                self.temps.lock().await.push(request.temperature);
                Ok(CompletionResponse {
                    message: Message::assistant("Ok"),
                    usage: Usage::default(),
                    finish_reason: FinishReason::Stop,
                })
            }
        }

        let provider = Arc::new(CapturingProvider {
            temps: tokio::sync::Mutex::new(Vec::new()),
        });

        let mut hooks = HookRegistry::new();
        hooks.register(Arc::new(ForceTemp));

        let config = RuntimeConfig {
            temperature: Some(0.7),
            ..RuntimeConfig::default()
        };

        let mut runtime = Runtime::new(
            provider.clone(),
            Arc::new(InMemoryStore::new()),
            ToolRegistry::new(),
            PolicyRegistry::default(),
            CharEstimator::default(),
            config,
        );
        runtime.set_hooks(hooks);

        let ns = Namespace::new("test");
        runtime.run(&ns, Message::user("Hi")).await.unwrap();

        let temps = provider.temps.lock().await;
        assert_eq!(temps[0], Some(0.0)); // Hook overrode 0.7 to 0.0
    }

    #[tokio::test]
    async fn hook_modifies_tool_result() {
        use crate::hook::{Hook, HookRegistry};

        struct RedactHook;

        #[async_trait]
        impl Hook for RedactHook {
            async fn after_tool_call(&self, _call: &ToolCall, result: &mut ToolResult) {
                result.content = result.content.replace("secret", "[REDACTED]");
            }
        }

        struct SecretTool;

        #[async_trait]
        impl Tool for SecretTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "get_secret".into(),
                    description: "Returns a secret".into(),
                    input_schema: serde_json::json!({"type": "object"}),
                }
            }
            async fn execute(&self, _input: serde_json::Value) -> Result<String, ToolError> {
                Ok("the secret is 42".into())
            }
        }

        let tool_call = ToolCall {
            id: "c1".into(),
            name: "get_secret".into(),
            arguments: serde_json::json!({}),
        };

        let responses = vec![
            CompletionResponse {
                message: Message::assistant_with_tool_calls("Getting.", vec![tool_call]),
                usage: Usage::default(),
                finish_reason: FinishReason::ToolUse,
            },
            CompletionResponse {
                message: Message::assistant("Here you go."),
                usage: Usage::default(),
                finish_reason: FinishReason::Stop,
            },
        ];

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(SecretTool));

        let mut hooks = HookRegistry::new();
        hooks.register(Arc::new(RedactHook));

        let mut runtime = make_runtime(responses, tools, RuntimeConfig::default());
        runtime.set_hooks(hooks);

        let ns = Namespace::new("test");
        let result = runtime.run(&ns, Message::user("Show secret")).await.unwrap();

        assert_eq!(result.turns[0].tool_results[0].content, "the [REDACTED] is 42");
    }

    // --- Streaming tests ---

    #[tokio::test]
    async fn run_streaming_with_mock_provider() {
        use crate::provider::StreamEvent;

        struct MockStreamProvider;

        #[async_trait]
        impl Provider for MockStreamProvider {
            async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
                Ok(CompletionResponse {
                    message: Message::assistant("fallback"),
                    usage: Usage::default(),
                    finish_reason: FinishReason::Stop,
                })
            }
        }

        #[async_trait]
        impl StreamingProvider for MockStreamProvider {
            async fn stream(
                &self,
                _request: CompletionRequest,
            ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>, ProviderError> {
                let (tx, rx) = tokio::sync::mpsc::channel(16);
                tokio::spawn(async move {
                    let _ = tx.send(StreamEvent::TextDelta("Hello".into())).await;
                    let _ = tx.send(StreamEvent::TextDelta(" world!".into())).await;
                    let _ = tx.send(StreamEvent::Done {
                        usage: Usage { input_tokens: 10, output_tokens: 5 },
                        finish_reason: FinishReason::Stop,
                    }).await;
                });
                Ok(rx)
            }
        }

        let provider = Arc::new(MockStreamProvider);
        let store = Arc::new(InMemoryStore::new());
        let mut runtime = Runtime::new(
            provider.clone(),
            store,
            ToolRegistry::new(),
            PolicyRegistry::default(),
            CharEstimator::default(),
            RuntimeConfig::default(),
        );
        runtime.set_streaming_provider(provider);

        let ns = Namespace::new("test");
        let mut rx = runtime.run_streaming(&ns, Message::user("Hi")).await.unwrap();

        let mut texts = Vec::new();
        let mut got_done = false;

        while let Some(event) = rx.recv().await {
            match event {
                RuntimeStreamEvent::TextDelta(t) => texts.push(t),
                RuntimeStreamEvent::Done(result) => {
                    assert_eq!(result.final_message.content, "Hello world!");
                    assert_eq!(result.total_usage.input_tokens, 10);
                    got_done = true;
                }
                _ => {}
            }
        }

        assert!(got_done);
        assert_eq!(texts.join(""), "Hello world!");
    }

    #[tokio::test]
    async fn run_streaming_errors_without_provider() {
        let runtime = make_runtime(vec![], ToolRegistry::new(), RuntimeConfig::default());
        let ns = Namespace::new("test");
        let err = runtime.run_streaming(&ns, Message::user("Hi")).await.unwrap_err();
        match err {
            RuntimeError::Provider(_) => {}
            other => panic!("expected Provider error, got {:?}", other),
        }
    }
}
