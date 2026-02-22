//! Generic channel lifecycle manager.
//!
//! Provides [`ChannelManager`] which manages the lifecycle of a channel
//! connection, including hot-restart support, multi-agent routing, and
//! graceful shutdown. This abstracts the pattern used by Discord, gateway,
//! and other channel types.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use crate::channels::{Channel, ChannelAdapter};
use crate::context::Tokenizer;
use crate::routing::{Router, RoutingRule};
use crate::runtime::Runtime;

// ---------------------------------------------------------------------------
// Connection state
// ---------------------------------------------------------------------------

/// Current state of a managed channel connection.
#[derive(Debug, Clone)]
pub struct ChannelState {
    /// Whether the channel is currently connected and running.
    pub connected: bool,
    /// Application-specific metadata about the connection.
    pub metadata: HashMap<String, String>,
}

impl Default for ChannelState {
    fn default() -> Self {
        Self {
            connected: false,
            metadata: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Shutdown signaller
// ---------------------------------------------------------------------------

/// Trait for channels that support graceful shutdown via a signal.
///
/// Channels that can be shut down should implement this trait so the
/// manager can request a clean stop.
pub trait ShutdownSignal: Send + Sync {
    /// Signal the channel to shut down. Should cause `receive()` to
    /// return `None` after processing any in-flight messages.
    fn shutdown(&self);
}

// ---------------------------------------------------------------------------
// Channel manager
// ---------------------------------------------------------------------------

/// Manages the lifecycle of a channel connection, supporting:
///
/// - Hot-restart (disconnect + reconnect without restarting the process)
/// - Multi-agent routing via [`Router`] when multiple runtimes are provided
/// - Single-runtime fallback via [`ChannelAdapter`]
/// - Graceful shutdown with timeout
///
/// # Type Parameters
///
/// - `T`: The tokenizer type used by the runtimes.
///
/// # Example
///
/// ```ignore
/// let manager = ChannelManager::new(default_runtime);
/// manager.set_runtimes(runtimes_map, "atlas".into());
///
/// let channel = Arc::new(MyChannel::connect(config).await?);
/// manager.start(channel, "my-channel").await;
///
/// // Later...
/// manager.stop().await;
/// ```
pub struct ChannelManager<T: Tokenizer + 'static> {
    /// Default runtime for single-agent mode.
    default_runtime: Arc<Runtime<T>>,
    /// Named agent runtimes for multi-agent routing.
    agent_runtimes: RwLock<HashMap<String, Arc<Runtime<T>>>>,
    /// Default agent name (lowercase) for fallback routing.
    default_agent: RwLock<String>,
    /// Current connection state.
    state: RwLock<ChannelState>,
    /// The active channel (if any), stored as an Arc<dyn Channel> since
    /// we also need it for shutdown signaling.
    active_channel: RwLock<Option<Arc<dyn ShutdownSignal>>>,
    /// The background task running the adapter/router loop.
    task_handle: RwLock<Option<JoinHandle<()>>>,
    /// Shutdown timeout in seconds.
    shutdown_timeout_secs: u64,
}

impl<T: Tokenizer + 'static> ChannelManager<T> {
    /// Create a new manager with a default runtime.
    pub fn new(default_runtime: Arc<Runtime<T>>) -> Self {
        Self {
            default_runtime,
            agent_runtimes: RwLock::new(HashMap::new()),
            default_agent: RwLock::new(String::new()),
            state: RwLock::new(ChannelState::default()),
            active_channel: RwLock::new(None),
            task_handle: RwLock::new(None),
            shutdown_timeout_secs: 5,
        }
    }

    /// Set the shutdown timeout (default: 5 seconds).
    pub fn with_shutdown_timeout(mut self, secs: u64) -> Self {
        self.shutdown_timeout_secs = secs;
        self
    }

    /// Update the agent runtimes map and default agent name.
    pub async fn set_runtimes(
        &self,
        runtimes: HashMap<String, Arc<Runtime<T>>>,
        default_agent: String,
    ) {
        *self.agent_runtimes.write().await = runtimes;
        *self.default_agent.write().await = default_agent;
    }

    /// Get the current connection state.
    pub async fn state(&self) -> ChannelState {
        self.state.read().await.clone()
    }

    /// Check if the manager has a running connection.
    pub async fn is_connected(&self) -> bool {
        self.state.read().await.connected
    }

    /// Start running a channel, dispatching messages to the appropriate
    /// runtime(s). If already running, stops the previous channel first.
    ///
    /// The channel must implement both `Channel` and `ShutdownSignal`.
    /// A channel name is used to identify this channel in the router.
    ///
    /// Optional metadata can be stored in the connection state for
    /// application-specific tracking.
    pub async fn start(
        &self,
        channel: Arc<dyn Channel>,
        shutdown_signal: Arc<dyn ShutdownSignal>,
        channel_name: &str,
        metadata: HashMap<String, String>,
    ) {
        // Stop any existing connection
        self.stop().await;

        // Snapshot runtimes
        let runtimes_snapshot = self.agent_runtimes.read().await.clone();
        let default_agent_name = self.default_agent.read().await.clone();
        let has_agents = !runtimes_snapshot.is_empty();

        let channel_for_task = channel.clone();
        let runtime = self.default_runtime.clone();
        let name = channel_name.to_string();

        let handle = tokio::spawn(async move {
            if has_agents {
                let mut router = Router::new(RoutingRule::MetadataKey("agent".into()));
                router.add_channel(&name, channel_for_task);
                let default_key = default_agent_name.to_lowercase();
                if let Err(e) = router.run(&runtimes_snapshot, Some(&default_key)).await {
                    eprintln!("[channel-manager] Router error on '{}': {}", name, e);
                }
            } else {
                if let Err(e) = ChannelAdapter::run(channel.as_ref(), &runtime).await {
                    eprintln!("[channel-manager] Adapter error on '{}': {}", name, e);
                }
            }
            eprintln!("[channel-manager] Channel '{}' stopped", name);
        });

        *self.active_channel.write().await = Some(shutdown_signal);
        *self.task_handle.write().await = Some(handle);

        {
            let mut state = self.state.write().await;
            state.connected = true;
            state.metadata = metadata;
        }
    }

    /// Stop the current channel gracefully.
    pub async fn stop(&self) {
        // Signal shutdown
        if let Some(signal) = self.active_channel.write().await.take() {
            signal.shutdown();
        }

        // Wait for the task to finish (with timeout)
        if let Some(handle) = self.task_handle.write().await.take() {
            let _ = tokio::time::timeout(
                tokio::time::Duration::from_secs(self.shutdown_timeout_secs),
                handle,
            )
            .await;
        }

        {
            let mut state = self.state.write().await;
            state.connected = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::{ChannelError, InboundMessage, OutboundError, OutboundMessage};
    use crate::context::CharEstimator;
    use crate::message::Message;
    use crate::namespace::Namespace;
    use crate::policy::PolicyRegistry;
    use crate::provider::{CompletionRequest, CompletionResponse, FinishReason, Provider, ProviderError, Usage};
    use crate::store::InMemoryStore;
    use crate::tool::ToolRegistry;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockChannel {
        inbound_rx: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<InboundMessage>>,
        outbound_tx: tokio::sync::mpsc::Sender<OutboundMessage>,
        shutdown_tx: tokio::sync::watch::Sender<bool>,
    }

    impl ShutdownSignal for MockChannel {
        fn shutdown(&self) {
            let _ = self.shutdown_tx.send(true);
        }
    }

    #[async_trait]
    impl Channel for MockChannel {
        async fn receive(&self) -> Option<InboundMessage> {
            let mut shutdown_rx = self.shutdown_tx.subscribe();
            let mut rx = self.inbound_rx.lock().await;
            tokio::select! {
                msg = rx.recv() => msg,
                _ = shutdown_rx.changed() => None,
            }
        }

        async fn send(&self, response: OutboundMessage) -> Result<(), ChannelError> {
            self.outbound_tx.send(response).await.map_err(|_| ChannelError::Closed)
        }

        async fn send_error(&self, _error: OutboundError) -> Result<(), ChannelError> {
            Ok(())
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

    fn make_runtime(response: &str) -> Arc<Runtime<CharEstimator>> {
        let provider = Arc::new(FixedProvider {
            responses: vec![CompletionResponse {
                message: Message::assistant(response),
                usage: Usage::default(),
                finish_reason: FinishReason::Stop,
            }],
            call_count: AtomicUsize::new(0),
        });

        Arc::new(Runtime::new(
            provider,
            Arc::new(InMemoryStore::new()),
            ToolRegistry::new(),
            PolicyRegistry::default(),
            CharEstimator::default(),
            Default::default(),
        ))
    }

    fn make_mock_channel() -> (
        tokio::sync::mpsc::Sender<InboundMessage>,
        tokio::sync::mpsc::Receiver<OutboundMessage>,
        Arc<MockChannel>,
    ) {
        let (in_tx, in_rx) = tokio::sync::mpsc::channel(16);
        let (out_tx, out_rx) = tokio::sync::mpsc::channel(16);
        let (shutdown_tx, _) = tokio::sync::watch::channel(false);
        let channel = Arc::new(MockChannel {
            inbound_rx: tokio::sync::Mutex::new(in_rx),
            outbound_tx: out_tx,
            shutdown_tx,
        });
        (in_tx, out_rx, channel)
    }

    #[tokio::test]
    async fn manager_starts_and_stops() {
        let runtime = make_runtime("Hello!");
        let manager = ChannelManager::new(runtime);

        assert!(!manager.is_connected().await);

        let (in_tx, mut out_rx, channel) = make_mock_channel();

        let ch: Arc<dyn Channel> = channel.clone();
        let sig: Arc<dyn ShutdownSignal> = channel.clone();
        manager.start(ch, sig, "test", HashMap::new()).await;

        assert!(manager.is_connected().await);

        // Send a message
        in_tx
            .send(InboundMessage {
                namespace: Namespace::new("test"),
                message: Message::user("Hi"),
                metadata: HashMap::new(),
            })
            .await
            .unwrap();

        // Receive response
        let response = out_rx.recv().await.unwrap();
        assert_eq!(response.message.content, "Hello!");

        // Stop
        manager.stop().await;
        assert!(!manager.is_connected().await);
    }

    #[tokio::test]
    async fn manager_state_tracks_metadata() {
        let runtime = make_runtime("ok");
        let manager = ChannelManager::new(runtime);

        let (_, _, channel) = make_mock_channel();
        let ch: Arc<dyn Channel> = channel.clone();
        let sig: Arc<dyn ShutdownSignal> = channel.clone();

        let mut meta = HashMap::new();
        meta.insert("token_hint".into(), "****abcd".into());
        meta.insert("filter".into(), "dm".into());

        manager.start(ch, sig, "test", meta).await;

        let state = manager.state().await;
        assert!(state.connected);
        assert_eq!(state.metadata.get("token_hint").unwrap(), "****abcd");
        assert_eq!(state.metadata.get("filter").unwrap(), "dm");

        manager.stop().await;
    }

    #[tokio::test]
    async fn manager_restart_replaces_connection() {
        let runtime = make_runtime("response");
        let manager = ChannelManager::new(runtime);

        // Start first connection
        let (_, _, channel1) = make_mock_channel();
        let ch1: Arc<dyn Channel> = channel1.clone();
        let sig1: Arc<dyn ShutdownSignal> = channel1.clone();
        manager.start(ch1, sig1, "first", HashMap::new()).await;
        assert!(manager.is_connected().await);

        // Start second connection (should stop first)
        let (_, _, channel2) = make_mock_channel();
        let ch2: Arc<dyn Channel> = channel2.clone();
        let sig2: Arc<dyn ShutdownSignal> = channel2.clone();
        manager.start(ch2, sig2, "second", HashMap::new()).await;
        assert!(manager.is_connected().await);

        manager.stop().await;
        assert!(!manager.is_connected().await);
    }

    #[tokio::test]
    async fn manager_stop_when_not_connected_is_noop() {
        let runtime = make_runtime("ok");
        let manager = ChannelManager::new(runtime);

        // Should not panic
        manager.stop().await;
        assert!(!manager.is_connected().await);
    }
}
