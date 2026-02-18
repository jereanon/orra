use async_trait::async_trait;

use crate::message::Message;
use crate::tool::ToolDefinition;

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub message: Message,
    pub usage: Usage,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl Usage {
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FinishReason {
    Stop,
    ToolUse,
    MaxTokens,
    Other(String),
}

#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ProviderError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("authentication failed: {0}")]
    Auth(String),

    #[error("rate limited, retry after {retry_after_ms:?}ms")]
    RateLimited { retry_after_ms: Option<u64> },

    #[error("context length exceeded: {0}")]
    ContextLengthExceeded(String),

    #[error("provider error: {0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_total_tokens() {
        let usage = Usage {
            input_tokens: 100,
            output_tokens: 50,
        };
        assert_eq!(usage.total_tokens(), 150);
    }

    #[test]
    fn default_usage() {
        let usage = Usage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.total_tokens(), 0);
    }

    #[test]
    fn finish_reason_equality() {
        assert_eq!(FinishReason::Stop, FinishReason::Stop);
        assert_eq!(FinishReason::ToolUse, FinishReason::ToolUse);
        assert_ne!(FinishReason::Stop, FinishReason::ToolUse);
        assert_eq!(
            FinishReason::Other("foo".into()),
            FinishReason::Other("foo".into())
        );
    }

    struct MockProvider {
        response: CompletionResponse,
    }

    #[async_trait]
    impl Provider for MockProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn mock_provider_returns_response() {
        let provider = MockProvider {
            response: CompletionResponse {
                message: Message::assistant("Hello!"),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
                finish_reason: FinishReason::Stop,
            },
        };

        let request = CompletionRequest {
            messages: vec![Message::user("Hi")],
            tools: vec![],
            max_tokens: None,
            temperature: None,
        };

        let response = provider.complete(request).await.unwrap();
        assert_eq!(response.message.content, "Hello!");
        assert_eq!(response.usage.total_tokens(), 15);
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }

    struct ErrorProvider;

    #[async_trait]
    impl Provider for ErrorProvider {
        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse, ProviderError> {
            Err(ProviderError::RateLimited {
                retry_after_ms: Some(1000),
            })
        }
    }

    #[tokio::test]
    async fn provider_error_handling() {
        let provider = ErrorProvider;
        let request = CompletionRequest {
            messages: vec![Message::user("Hi")],
            tools: vec![],
            max_tokens: None,
            temperature: None,
        };

        let err = provider.complete(request).await.unwrap_err();
        match err {
            ProviderError::RateLimited { retry_after_ms } => {
                assert_eq!(retry_after_ms, Some(1000));
            }
            _ => panic!("expected RateLimited error"),
        }
    }
}
