use std::sync::Arc;

use async_trait::async_trait;

use crate::message::{ToolCall, ToolResult};
use crate::namespace::Namespace;
use crate::provider::{CompletionRequest, CompletionResponse};
use crate::store::Session;

/// Lifecycle hooks for the agent runtime.
///
/// Hooks are called at various points during the agent loop, allowing you to
/// observe or modify behavior without changing the runtime itself. All methods
/// have default no-op implementations, so you only need to implement the ones
/// you care about.
#[async_trait]
pub trait Hook: Send + Sync {
    /// Called after a session is loaded from the store (or created fresh).
    async fn after_session_load(&self, _namespace: &Namespace, _session: &Session) {}

    /// Called before each LLM provider call. You can modify the request
    /// (e.g., adjust temperature, add/remove tools, modify messages).
    async fn before_provider_call(&self, _request: &mut CompletionRequest) {}

    /// Called after each LLM provider call returns.
    async fn after_provider_call(&self, _response: &CompletionResponse) {}

    /// Called before each individual tool is executed. You can modify the call
    /// (e.g., change arguments, rename the tool).
    async fn before_tool_call(&self, _call: &mut ToolCall) {}

    /// Called after each individual tool returns. You can modify the result
    /// (e.g., filter sensitive data, add metadata).
    async fn after_tool_call(&self, _call: &ToolCall, _result: &mut ToolResult) {}

    /// Called before the session is saved to the store. You can modify the
    /// session (e.g., prune messages, update metadata).
    async fn before_session_save(&self, _namespace: &Namespace, _session: &mut Session) {}
}

/// Registry that holds multiple hooks and dispatches lifecycle events to all of them.
pub struct HookRegistry {
    hooks: Vec<Arc<dyn Hook>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    pub fn register(&mut self, hook: Arc<dyn Hook>) {
        self.hooks.push(hook);
    }

    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    pub(crate) async fn dispatch_after_session_load(&self, namespace: &Namespace, session: &Session) {
        for hook in &self.hooks {
            hook.after_session_load(namespace, session).await;
        }
    }

    pub(crate) async fn dispatch_before_provider_call(&self, request: &mut CompletionRequest) {
        for hook in &self.hooks {
            hook.before_provider_call(request).await;
        }
    }

    pub(crate) async fn dispatch_after_provider_call(&self, response: &CompletionResponse) {
        for hook in &self.hooks {
            hook.after_provider_call(response).await;
        }
    }

    pub(crate) async fn dispatch_before_tool_call(&self, call: &mut ToolCall) {
        for hook in &self.hooks {
            hook.before_tool_call(call).await;
        }
    }

    pub(crate) async fn dispatch_after_tool_call(&self, call: &ToolCall, result: &mut ToolResult) {
        for hook in &self.hooks {
            hook.after_tool_call(call, result).await;
        }
    }

    pub(crate) async fn dispatch_before_session_save(&self, namespace: &Namespace, session: &mut Session) {
        for hook in &self.hooks {
            hook.before_session_save(namespace, session).await;
        }
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingHook {
        after_load: AtomicUsize,
        before_provider: AtomicUsize,
        after_provider: AtomicUsize,
        before_tool: AtomicUsize,
        after_tool: AtomicUsize,
        before_save: AtomicUsize,
    }

    impl CountingHook {
        fn new() -> Self {
            Self {
                after_load: AtomicUsize::new(0),
                before_provider: AtomicUsize::new(0),
                after_provider: AtomicUsize::new(0),
                before_tool: AtomicUsize::new(0),
                after_tool: AtomicUsize::new(0),
                before_save: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl Hook for CountingHook {
        async fn after_session_load(&self, _ns: &Namespace, _session: &Session) {
            self.after_load.fetch_add(1, Ordering::SeqCst);
        }
        async fn before_provider_call(&self, _request: &mut CompletionRequest) {
            self.before_provider.fetch_add(1, Ordering::SeqCst);
        }
        async fn after_provider_call(&self, _response: &CompletionResponse) {
            self.after_provider.fetch_add(1, Ordering::SeqCst);
        }
        async fn before_tool_call(&self, _call: &mut ToolCall) {
            self.before_tool.fetch_add(1, Ordering::SeqCst);
        }
        async fn after_tool_call(&self, _call: &ToolCall, _result: &mut ToolResult) {
            self.after_tool.fetch_add(1, Ordering::SeqCst);
        }
        async fn before_session_save(&self, _ns: &Namespace, _session: &mut Session) {
            self.before_save.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn dispatch_calls_all_lifecycle_points() {
        let hook = Arc::new(CountingHook::new());
        let mut registry = HookRegistry::new();
        registry.register(hook.clone());

        let ns = Namespace::new("test");
        let session = Session::new(ns.clone());

        registry.dispatch_after_session_load(&ns, &session).await;
        assert_eq!(hook.after_load.load(Ordering::SeqCst), 1);

        let mut request = CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: None,
            temperature: None,
        };
        registry.dispatch_before_provider_call(&mut request).await;
        assert_eq!(hook.before_provider.load(Ordering::SeqCst), 1);

        let response = CompletionResponse {
            message: Message::assistant("hi"),
            usage: crate::provider::Usage::default(),
            finish_reason: crate::provider::FinishReason::Stop,
        };
        registry.dispatch_after_provider_call(&response).await;
        assert_eq!(hook.after_provider.load(Ordering::SeqCst), 1);

        let mut call = ToolCall {
            id: "c1".into(),
            name: "test".into(),
            arguments: serde_json::json!({}),
        };
        registry.dispatch_before_tool_call(&mut call).await;
        assert_eq!(hook.before_tool.load(Ordering::SeqCst), 1);

        let mut result = ToolResult {
            call_id: "c1".into(),
            content: "ok".into(),
            is_error: false,
        };
        registry.dispatch_after_tool_call(&call, &mut result).await;
        assert_eq!(hook.after_tool.load(Ordering::SeqCst), 1);

        let mut session = Session::new(ns.clone());
        registry.dispatch_before_session_save(&ns, &mut session).await;
        assert_eq!(hook.before_save.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn modifying_hook_alters_request() {
        struct TempOverrideHook;

        #[async_trait]
        impl Hook for TempOverrideHook {
            async fn before_provider_call(&self, request: &mut CompletionRequest) {
                request.temperature = Some(0.0);
            }
        }

        let mut registry = HookRegistry::new();
        registry.register(Arc::new(TempOverrideHook));

        let mut request = CompletionRequest {
            messages: vec![],
            tools: vec![],
            max_tokens: None,
            temperature: Some(0.7),
        };

        registry.dispatch_before_provider_call(&mut request).await;
        assert_eq!(request.temperature, Some(0.0));
    }

    #[tokio::test]
    async fn multiple_hooks_compose_in_order() {
        struct AppendHook {
            suffix: String,
        }

        #[async_trait]
        impl Hook for AppendHook {
            async fn after_tool_call(&self, _call: &ToolCall, result: &mut ToolResult) {
                result.content.push_str(&self.suffix);
            }
        }

        let mut registry = HookRegistry::new();
        registry.register(Arc::new(AppendHook { suffix: ":first".into() }));
        registry.register(Arc::new(AppendHook { suffix: ":second".into() }));

        let call = ToolCall {
            id: "c1".into(),
            name: "test".into(),
            arguments: serde_json::json!({}),
        };
        let mut result = ToolResult {
            call_id: "c1".into(),
            content: "base".into(),
            is_error: false,
        };

        registry.dispatch_after_tool_call(&call, &mut result).await;
        assert_eq!(result.content, "base:first:second");
    }

    #[tokio::test]
    async fn empty_registry_is_noop() {
        let registry = HookRegistry::new();
        assert!(registry.is_empty());

        let ns = Namespace::new("test");
        let mut session = Session::new(ns.clone());

        // These should all succeed silently
        registry.dispatch_after_session_load(&ns, &session).await;
        registry.dispatch_before_session_save(&ns, &mut session).await;
    }
}
