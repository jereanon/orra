use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use async_trait::async_trait;

use crate::hook::Hook;
use crate::message::{ToolCall, ToolResult};
use crate::namespace::Namespace;
use crate::provider::{CompletionRequest, CompletionResponse};
use crate::store::Session;

/// A hook that logs runtime activity to stderr.
pub struct LoggingHook {
    call_start: tokio::sync::Mutex<Option<Instant>>,
    total_input_tokens: AtomicU64,
    total_output_tokens: AtomicU64,
}

impl LoggingHook {
    pub fn new() -> Self {
        Self {
            call_start: tokio::sync::Mutex::new(None),
            total_input_tokens: AtomicU64::new(0),
            total_output_tokens: AtomicU64::new(0),
        }
    }
}

impl Default for LoggingHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for LoggingHook {
    async fn after_session_load(&self, namespace: &Namespace, session: &Session) {
        let count = session.message_count();
        if count > 0 {
            eprintln!(
                "[session] loaded {} messages for {}",
                count,
                namespace.key()
            );
        } else {
            eprintln!("[session] new session for {}", namespace.key());
        }
    }

    async fn before_provider_call(&self, request: &mut CompletionRequest) {
        let tool_count = request.tools.len();
        let msg_count = request.messages.len();
        eprintln!("[llm] calling provider with {msg_count} messages, {tool_count} tools");
        *self.call_start.lock().await = Some(Instant::now());
    }

    async fn after_provider_call(&self, response: &CompletionResponse) {
        let elapsed = self
            .call_start
            .lock()
            .await
            .map(|s| s.elapsed())
            .unwrap_or_default();

        self.total_input_tokens
            .fetch_add(response.usage.input_tokens as u64, Ordering::Relaxed);
        self.total_output_tokens
            .fetch_add(response.usage.output_tokens as u64, Ordering::Relaxed);

        eprintln!(
            "[llm] response in {:.1}s — {} in / {} out tokens (session total: {} in / {} out)",
            elapsed.as_secs_f64(),
            response.usage.input_tokens,
            response.usage.output_tokens,
            self.total_input_tokens.load(Ordering::Relaxed),
            self.total_output_tokens.load(Ordering::Relaxed),
        );
    }

    async fn before_tool_call(
        &self,
        _namespace: &Namespace,
        call: &mut ToolCall,
    ) -> Result<(), String> {
        eprintln!("[tool] calling: {}", call.name);
        Ok(())
    }

    async fn after_tool_call(&self, call: &ToolCall, result: &mut ToolResult) {
        let status = if result.is_error { "error" } else { "ok" };
        let preview_len = 120;
        let preview = if result.content.len() > preview_len {
            format!("{}...", &result.content[..preview_len])
        } else {
            result.content.clone()
        };
        eprintln!(
            "[tool] {} → {} ({})",
            call.name,
            status,
            preview.replace('\n', " ")
        );
    }

    async fn before_session_save(&self, namespace: &Namespace, session: &mut Session) {
        eprintln!(
            "[session] saving {} messages for {}",
            session.message_count(),
            namespace.key()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use crate::provider::{FinishReason, Usage};

    #[tokio::test]
    async fn logging_hook_runs_without_panic() {
        let hook = LoggingHook::new();
        let ns = Namespace::new("test");
        let session = Session::new(ns.clone());

        hook.after_session_load(&ns, &session).await;

        let mut request = CompletionRequest {
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            model: None,
        };
        hook.before_provider_call(&mut request).await;

        let response = CompletionResponse {
            message: Message::assistant("hello"),
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
            finish_reason: FinishReason::Stop,
        };
        hook.after_provider_call(&response).await;

        let mut call = ToolCall {
            id: "c1".into(),
            name: "test_tool".into(),
            arguments: serde_json::json!({}),
        };
        let _ = hook.before_tool_call(&ns, &mut call).await;

        let mut result = ToolResult {
            call_id: "c1".into(),
            content: "result".into(),
            is_error: false,
        };
        hook.after_tool_call(&call, &mut result).await;

        let mut session = Session::new(ns.clone());
        session.push_message(Message::user("hi"));
        hook.before_session_save(&ns, &mut session).await;
    }
}
