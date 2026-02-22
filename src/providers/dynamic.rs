//! Dynamic provider wrapper that supports hot-swapping the underlying
//! LLM provider at runtime. This allows applications to start without
//! credentials and configure the provider later via a web UI or API.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::provider::{CompletionRequest, CompletionResponse, Provider, ProviderError};

// ---------------------------------------------------------------------------
// Placeholder provider (returns error until configured)
// ---------------------------------------------------------------------------

/// A provider that always returns an error indicating it hasn't been
/// configured yet. Used as the initial provider when no API key is provided.
pub struct PlaceholderProvider;

#[async_trait]
impl Provider for PlaceholderProvider {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        Err(ProviderError::Auth(
            "Provider not configured. Please set your API key via the web UI or config file."
                .into(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Dynamic provider (hot-swappable)
// ---------------------------------------------------------------------------

/// A `Provider` wrapper that allows the underlying provider to be swapped
/// at runtime. Delegates all `complete()` calls to whatever provider is
/// currently installed.
///
/// # Example
///
/// ```rust
/// use orra::providers::dynamic::DynamicProvider;
///
/// // Start unconfigured
/// let provider = DynamicProvider::placeholder();
/// assert!(!provider.is_configured());
///
/// // Later, swap in a real provider...
/// // provider.swap(real_provider).await;
/// ```
pub struct DynamicProvider {
    inner: Arc<RwLock<Arc<dyn Provider>>>,
    configured: AtomicBool,
}

impl DynamicProvider {
    /// Create a new DynamicProvider with a placeholder (unconfigured).
    pub fn placeholder() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Arc::new(PlaceholderProvider))),
            configured: AtomicBool::new(false),
        }
    }

    /// Create a new DynamicProvider wrapping a real provider.
    pub fn with_provider(provider: Arc<dyn Provider>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(provider)),
            configured: AtomicBool::new(true),
        }
    }

    /// Hot-swap the inner provider. All subsequent `complete()` calls
    /// will use the new provider.
    pub async fn swap(&self, new_provider: Arc<dyn Provider>) {
        let mut inner = self.inner.write().await;
        *inner = new_provider;
        self.configured.store(true, Ordering::SeqCst);
    }

    /// Returns true if a real provider has been configured.
    pub fn is_configured(&self) -> bool {
        self.configured.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Provider for DynamicProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let provider = self.inner.read().await.clone();
        provider.complete(request).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use crate::provider::{CompletionResponse, FinishReason, Usage};

    #[tokio::test]
    async fn placeholder_returns_auth_error() {
        let provider = DynamicProvider::placeholder();
        assert!(!provider.is_configured());

        let request = CompletionRequest {
            messages: vec![Message::user("Hello")],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            model: None,
        };

        let err = provider.complete(request).await.unwrap_err();
        match err {
            ProviderError::Auth(msg) => {
                assert!(msg.contains("not configured"));
            }
            _ => panic!("expected Auth error, got {err:?}"),
        }
    }

    #[tokio::test]
    async fn swap_replaces_provider() {
        let dynamic = DynamicProvider::placeholder();
        assert!(!dynamic.is_configured());

        struct MockProvider;

        #[async_trait]
        impl Provider for MockProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, ProviderError> {
                Ok(CompletionResponse {
                    message: Message::assistant("Hello from mock!"),
                    usage: Usage {
                        input_tokens: 5,
                        output_tokens: 3,
                    },
                    finish_reason: FinishReason::Stop,
                })
            }
        }

        dynamic.swap(Arc::new(MockProvider)).await;
        assert!(dynamic.is_configured());

        let request = CompletionRequest {
            messages: vec![Message::user("Hi")],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            model: None,
        };

        let response = dynamic.complete(request).await.unwrap();
        assert_eq!(response.message.content, "Hello from mock!");
    }

    #[tokio::test]
    async fn with_provider_starts_configured() {
        struct MockProvider;

        #[async_trait]
        impl Provider for MockProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, ProviderError> {
                Ok(CompletionResponse {
                    message: Message::assistant("ok"),
                    usage: Usage::default(),
                    finish_reason: FinishReason::Stop,
                })
            }
        }

        let dynamic = DynamicProvider::with_provider(Arc::new(MockProvider));
        assert!(dynamic.is_configured());

        let request = CompletionRequest {
            messages: vec![Message::user("test")],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            model: None,
        };

        let response = dynamic.complete(request).await.unwrap();
        assert_eq!(response.message.content, "ok");
    }
}
