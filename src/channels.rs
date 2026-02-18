#[cfg(feature = "discord")]
pub mod discord;

use std::collections::HashMap;

use async_trait::async_trait;

use crate::context::Tokenizer;
use crate::message::Message;
use crate::namespace::Namespace;
use crate::runtime::{RunResult, Runtime, RuntimeError};

/// An inbound message from a channel (e.g., a chat message from a user).
#[derive(Debug, Clone)]
pub struct InboundMessage {
    /// The namespace to route this message to.
    pub namespace: Namespace,
    /// The user message.
    pub message: Message,
    /// Arbitrary metadata (e.g., channel-specific context like thread ID, user info).
    pub metadata: HashMap<String, serde_json::Value>,
}

/// An outbound response to send back through a channel.
#[derive(Debug, Clone)]
pub struct OutboundMessage {
    /// The namespace this response came from.
    pub namespace: Namespace,
    /// The assistant's final response message.
    pub message: Message,
    /// Full run result including all turns and usage.
    pub run_result: RunResult,
    /// Metadata carried over from the inbound message, plus any additions.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// An error that occurred while processing a channel message.
#[derive(Debug, Clone)]
pub struct OutboundError {
    /// The namespace the error originated from.
    pub namespace: Namespace,
    /// The error that occurred.
    pub error: String,
    /// Metadata carried over from the inbound message.
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    #[error("send error: {0}")]
    Send(String),

    #[error("channel closed")]
    Closed,
}

/// A channel provides a transport-agnostic way to receive messages and send responses.
///
/// Implement this trait for different input sources: stdin, HTTP endpoints,
/// Discord bots, Slack, WebSocket connections, etc.
#[async_trait]
pub trait Channel: Send + Sync {
    /// Receive the next inbound message. Returns `None` when the channel is closed.
    async fn receive(&self) -> Option<InboundMessage>;

    /// Send a response back through the channel.
    async fn send(&self, response: OutboundMessage) -> Result<(), ChannelError>;

    /// Send an error response back through the channel.
    async fn send_error(&self, error: OutboundError) -> Result<(), ChannelError>;
}

/// Adapter that connects a `Channel` to a `Runtime`, running a receive-process-respond loop.
pub struct ChannelAdapter;

impl ChannelAdapter {
    /// Run the channel loop: receive messages, process through the runtime, send responses.
    ///
    /// Stops when the channel's `receive()` returns `None`.
    pub async fn run<T: Tokenizer>(
        channel: &dyn Channel,
        runtime: &Runtime<T>,
    ) -> Result<(), RuntimeError> {
        while let Some(inbound) = channel.receive().await {
            let namespace = inbound.namespace.clone();
            let metadata = inbound.metadata.clone();

            match runtime.run(&namespace, inbound.message).await {
                Ok(run_result) => {
                    let response = OutboundMessage {
                        namespace,
                        message: run_result.final_message.clone(),
                        run_result,
                        metadata,
                    };

                    if let Err(e) = channel.send(response).await {
                        // If we can't send back, log and continue
                        eprintln!("channel send error: {}", e);
                    }
                }
                Err(e) => {
                    let error_response = OutboundError {
                        namespace,
                        error: e.to_string(),
                        metadata,
                    };

                    if let Err(send_err) = channel.send_error(error_response).await {
                        eprintln!("channel send_error error: {}", send_err);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CharEstimator;
    use crate::policy::PolicyRegistry;
    use crate::provider::{
        CompletionRequest, CompletionResponse, FinishReason, Provider, ProviderError, Usage,
    };
    use crate::store::InMemoryStore;
    use crate::tool::ToolRegistry;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // Mock channel backed by mpsc
    struct MockChannel {
        inbound_rx: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<InboundMessage>>,
        outbound_tx: tokio::sync::mpsc::Sender<Result<OutboundMessage, OutboundError>>,
    }

    #[async_trait]
    impl Channel for MockChannel {
        async fn receive(&self) -> Option<InboundMessage> {
            self.inbound_rx.lock().await.recv().await
        }

        async fn send(&self, response: OutboundMessage) -> Result<(), ChannelError> {
            self.outbound_tx
                .send(Ok(response))
                .await
                .map_err(|_| ChannelError::Closed)
        }

        async fn send_error(&self, error: OutboundError) -> Result<(), ChannelError> {
            self.outbound_tx
                .send(Err(error))
                .await
                .map_err(|_| ChannelError::Closed)
        }
    }

    struct FixedProvider {
        responses: Vec<CompletionResponse>,
        call_count: AtomicUsize,
    }

    #[async_trait]
    impl Provider for FixedProvider {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            if idx < self.responses.len() {
                Ok(self.responses[idx].clone())
            } else {
                Err(ProviderError::Other("no more responses".into()))
            }
        }
    }

    fn setup() -> (
        tokio::sync::mpsc::Sender<InboundMessage>,
        tokio::sync::mpsc::Receiver<Result<OutboundMessage, OutboundError>>,
        MockChannel,
    ) {
        let (in_tx, in_rx) = tokio::sync::mpsc::channel(16);
        let (out_tx, out_rx) = tokio::sync::mpsc::channel(16);
        let channel = MockChannel {
            inbound_rx: tokio::sync::Mutex::new(in_rx),
            outbound_tx: out_tx,
        };
        (in_tx, out_rx, channel)
    }

    #[tokio::test]
    async fn adapter_routes_message_through_runtime() {
        let (in_tx, mut out_rx, channel) = setup();

        let provider = Arc::new(FixedProvider {
            responses: vec![CompletionResponse {
                message: Message::assistant("Hello back!"),
                usage: Usage { input_tokens: 5, output_tokens: 3 },
                finish_reason: FinishReason::Stop,
            }],
            call_count: AtomicUsize::new(0),
        });

        let runtime = Runtime::new(
            provider,
            Arc::new(InMemoryStore::new()),
            ToolRegistry::new(),
            PolicyRegistry::default(),
            CharEstimator::default(),
            Default::default(),
        );

        // Send one message then close
        in_tx
            .send(InboundMessage {
                namespace: Namespace::new("test"),
                message: Message::user("Hello!"),
                metadata: HashMap::new(),
            })
            .await
            .unwrap();
        drop(in_tx); // Close channel

        ChannelAdapter::run(&channel, &runtime).await.unwrap();

        let response = out_rx.recv().await.unwrap().unwrap();
        assert_eq!(response.message.content, "Hello back!");
        assert_eq!(response.namespace, Namespace::new("test"));
    }

    #[tokio::test]
    async fn metadata_passes_through() {
        let (in_tx, mut out_rx, channel) = setup();

        let provider = Arc::new(FixedProvider {
            responses: vec![CompletionResponse {
                message: Message::assistant("Ok"),
                usage: Usage::default(),
                finish_reason: FinishReason::Stop,
            }],
            call_count: AtomicUsize::new(0),
        });

        let runtime = Runtime::new(
            provider,
            Arc::new(InMemoryStore::new()),
            ToolRegistry::new(),
            PolicyRegistry::default(),
            CharEstimator::default(),
            Default::default(),
        );

        let mut metadata = HashMap::new();
        metadata.insert("thread_id".into(), serde_json::json!("abc123"));
        metadata.insert("user_name".into(), serde_json::json!("Alice"));

        in_tx
            .send(InboundMessage {
                namespace: Namespace::new("test"),
                message: Message::user("Hi"),
                metadata,
            })
            .await
            .unwrap();
        drop(in_tx);

        ChannelAdapter::run(&channel, &runtime).await.unwrap();

        let response = out_rx.recv().await.unwrap().unwrap();
        assert_eq!(response.metadata["thread_id"], "abc123");
        assert_eq!(response.metadata["user_name"], "Alice");
    }

    #[tokio::test]
    async fn channel_returning_none_stops_loop() {
        let (in_tx, _out_rx, channel) = setup();

        let provider = Arc::new(FixedProvider {
            responses: vec![],
            call_count: AtomicUsize::new(0),
        });

        let runtime = Runtime::new(
            provider,
            Arc::new(InMemoryStore::new()),
            ToolRegistry::new(),
            PolicyRegistry::default(),
            CharEstimator::default(),
            Default::default(),
        );

        // Immediately close - no messages
        drop(in_tx);

        // Should return immediately without error
        ChannelAdapter::run(&channel, &runtime).await.unwrap();
    }

    #[tokio::test]
    async fn runtime_error_sent_as_error_response() {
        let (in_tx, mut out_rx, channel) = setup();

        // Provider that always fails
        struct FailProvider;

        #[async_trait]
        impl Provider for FailProvider {
            async fn complete(
                &self,
                _request: CompletionRequest,
            ) -> Result<CompletionResponse, ProviderError> {
                Err(ProviderError::Other("provider down".into()))
            }
        }

        let runtime = Runtime::new(
            Arc::new(FailProvider),
            Arc::new(InMemoryStore::new()),
            ToolRegistry::new(),
            PolicyRegistry::default(),
            CharEstimator::default(),
            Default::default(),
        );

        in_tx
            .send(InboundMessage {
                namespace: Namespace::new("test"),
                message: Message::user("Hello!"),
                metadata: HashMap::new(),
            })
            .await
            .unwrap();
        drop(in_tx);

        ChannelAdapter::run(&channel, &runtime).await.unwrap();

        let response = out_rx.recv().await.unwrap();
        match response {
            Err(error) => {
                assert!(error.error.contains("provider"));
                assert_eq!(error.namespace, Namespace::new("test"));
            }
            Ok(_) => panic!("expected error response"),
        }
    }

    #[tokio::test]
    async fn multiple_messages_processed_sequentially() {
        let (in_tx, mut out_rx, channel) = setup();

        let provider = Arc::new(FixedProvider {
            responses: vec![
                CompletionResponse {
                    message: Message::assistant("Response 1"),
                    usage: Usage::default(),
                    finish_reason: FinishReason::Stop,
                },
                CompletionResponse {
                    message: Message::assistant("Response 2"),
                    usage: Usage::default(),
                    finish_reason: FinishReason::Stop,
                },
            ],
            call_count: AtomicUsize::new(0),
        });

        let runtime = Runtime::new(
            provider,
            Arc::new(InMemoryStore::new()),
            ToolRegistry::new(),
            PolicyRegistry::default(),
            CharEstimator::default(),
            Default::default(),
        );

        in_tx
            .send(InboundMessage {
                namespace: Namespace::new("user1"),
                message: Message::user("First"),
                metadata: HashMap::new(),
            })
            .await
            .unwrap();

        in_tx
            .send(InboundMessage {
                namespace: Namespace::new("user2"),
                message: Message::user("Second"),
                metadata: HashMap::new(),
            })
            .await
            .unwrap();

        drop(in_tx);

        ChannelAdapter::run(&channel, &runtime).await.unwrap();

        let r1 = out_rx.recv().await.unwrap().unwrap();
        assert_eq!(r1.message.content, "Response 1");
        assert_eq!(r1.namespace, Namespace::new("user1"));

        let r2 = out_rx.recv().await.unwrap().unwrap();
        assert_eq!(r2.message.content, "Response 2");
        assert_eq!(r2.namespace, Namespace::new("user2"));
    }
}
