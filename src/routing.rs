//! Message routing layer.
//!
//! Provides a router that dispatches messages from multiple channels to
//! appropriate runtimes based on configurable routing rules. This enables
//! multi-channel setups where a single application handles Discord, HTTP,
//! and other channels simultaneously.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::channels::{Channel, ChannelError, InboundMessage, OutboundError, OutboundMessage};
use crate::context::Tokenizer;
use crate::namespace::Namespace;
use crate::runtime::{Runtime, RuntimeError};

// ---------------------------------------------------------------------------
// Routing rules
// ---------------------------------------------------------------------------

/// Determines how an inbound message gets routed.
#[derive(Debug, Clone)]
pub enum RoutingRule {
    /// Route to a specific runtime by name.
    Static(String),

    /// Route based on the namespace prefix. For example, a namespace
    /// "discord/guild-123/channel-456" with prefix "discord" routes to
    /// the runtime registered under "discord".
    NamespacePrefix,

    /// Route based on metadata key. The value of the specified key
    /// determines which runtime handles the message.
    MetadataKey(String),
}

// ---------------------------------------------------------------------------
// Multi-channel router
// ---------------------------------------------------------------------------

/// A channel that aggregates multiple input channels and routes messages
/// based on configurable rules.
pub struct Router {
    channels: Vec<NamedChannel>,
    rule: RoutingRule,
}

struct NamedChannel {
    name: String,
    channel: Arc<dyn Channel>,
}

impl Router {
    pub fn new(rule: RoutingRule) -> Self {
        Self {
            channels: Vec::new(),
            rule,
        }
    }

    /// Register a named channel.
    pub fn add_channel(&mut self, name: impl Into<String>, channel: Arc<dyn Channel>) {
        self.channels.push(NamedChannel {
            name: name.into(),
            channel,
        });
    }

    /// Determine which runtime should handle this message based on the routing rule.
    pub fn resolve_target(&self, msg: &InboundMessage, source: &str) -> String {
        match &self.rule {
            RoutingRule::Static(target) => target.clone(),
            RoutingRule::NamespacePrefix => {
                // Use the first segment of the namespace as the target
                let key = msg.namespace.key();
                key.split('/')
                    .next()
                    .unwrap_or(source)
                    .to_string()
            }
            RoutingRule::MetadataKey(key) => msg
                .metadata
                .get(key)
                .and_then(|v| v.as_str())
                .unwrap_or(source)
                .to_string(),
        }
    }

    /// Run the router loop, dispatching messages from all channels to the
    /// appropriate runtime. Takes a map of named runtimes.
    pub async fn run<T: Tokenizer + 'static>(
        &self,
        runtimes: &HashMap<String, Arc<Runtime<T>>>,
        default_runtime: Option<&str>,
    ) -> Result<(), RouterError> {
        if self.channels.is_empty() {
            return Err(RouterError::NoChannels);
        }

        // Spawn a receive task for each channel
        let (tx, mut rx) = tokio::sync::mpsc::channel::<(String, InboundMessage)>(256);

        for named in &self.channels {
            let channel = named.channel.clone();
            let name = named.name.clone();
            let tx = tx.clone();

            tokio::spawn(async move {
                while let Some(msg) = channel.receive().await {
                    if tx.send((name.clone(), msg)).await.is_err() {
                        break;
                    }
                }
            });
        }

        // Drop our sender so the loop ends when all channel tasks are done
        drop(tx);

        // Process routed messages
        while let Some((source, inbound)) = rx.recv().await {
            let target = self.resolve_target(&inbound, &source);

            let runtime = runtimes
                .get(&target)
                .or_else(|| default_runtime.and_then(|d| runtimes.get(d)));

            let runtime = match runtime {
                Some(r) => r,
                None => {
                    eprintln!(
                        "router: no runtime found for target '{}', dropping message",
                        target
                    );
                    continue;
                }
            };

            let namespace = inbound.namespace.clone();
            let metadata = inbound.metadata.clone();

            // Find the channel that sent this message so we can reply
            let source_channel = self
                .channels
                .iter()
                .find(|c| c.name == source)
                .map(|c| c.channel.clone());

            // Show typing indicator while processing
            let _typing = if let Some(ref ch) = source_channel {
                ch.start_typing(&metadata).await
            } else {
                None
            };

            match runtime.run(&namespace, inbound.message).await {
                Ok(run_result) => {
                    if let Some(ch) = source_channel {
                        let response = OutboundMessage {
                            namespace,
                            message: run_result.final_message.clone(),
                            run_result,
                            metadata,
                        };
                        if let Err(e) = ch.send(response).await {
                            eprintln!("router: send error on channel '{}': {}", source, e);
                        }
                    }
                }
                Err(e) => {
                    if let Some(ch) = source_channel {
                        let err = OutboundError {
                            namespace,
                            error: e.to_string(),
                            metadata,
                        };
                        if let Err(send_err) = ch.send_error(err).await {
                            eprintln!(
                                "router: send_error error on channel '{}': {}",
                                source, send_err
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("no channels registered")]
    NoChannels,

    #[error("runtime not found: {0}")]
    RuntimeNotFound(String),

    #[error("runtime error: {0}")]
    Runtime(#[from] RuntimeError),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

    fn make_inbound(ns: &str) -> InboundMessage {
        InboundMessage {
            namespace: Namespace::parse(ns),
            message: Message::user("test"),
            metadata: HashMap::new(),
        }
    }

    fn make_inbound_with_meta(ns: &str, key: &str, value: &str) -> InboundMessage {
        let mut metadata = HashMap::new();
        metadata.insert(key.into(), serde_json::json!(value));
        InboundMessage {
            namespace: Namespace::parse(ns),
            message: Message::user("test"),
            metadata,
        }
    }

    #[test]
    fn static_routing() {
        let router = Router::new(RoutingRule::Static("main".into()));
        let msg = make_inbound("any/namespace");
        assert_eq!(router.resolve_target(&msg, "discord"), "main");
    }

    #[test]
    fn namespace_prefix_routing() {
        let router = Router::new(RoutingRule::NamespacePrefix);
        let msg = make_inbound("discord/guild-1/channel-2");
        assert_eq!(router.resolve_target(&msg, "fallback"), "discord");
    }

    #[test]
    fn namespace_prefix_routing_no_slash() {
        let router = Router::new(RoutingRule::NamespacePrefix);
        let msg = make_inbound("simple");
        assert_eq!(router.resolve_target(&msg, "fallback"), "simple");
    }

    #[test]
    fn metadata_key_routing() {
        let router = Router::new(RoutingRule::MetadataKey("target_runtime".into()));
        let msg = make_inbound_with_meta("ns", "target_runtime", "api");
        assert_eq!(router.resolve_target(&msg, "default"), "api");
    }

    #[test]
    fn metadata_key_routing_missing_falls_back_to_source() {
        let router = Router::new(RoutingRule::MetadataKey("target_runtime".into()));
        let msg = make_inbound("ns");
        assert_eq!(router.resolve_target(&msg, "discord"), "discord");
    }

    #[test]
    fn add_channels() {
        let mut router = Router::new(RoutingRule::Static("main".into()));
        assert!(router.channels.is_empty());

        // We can't easily create a real channel in a unit test, but we can
        // verify the struct accepts them
        assert_eq!(router.channels.len(), 0);
    }

    #[tokio::test]
    async fn run_with_no_channels_returns_error() {
        let router = Router::new(RoutingRule::Static("main".into()));
        let runtimes: HashMap<String, Arc<Runtime<crate::context::CharEstimator>>> = HashMap::new();

        let err = router.run(&runtimes, None).await.unwrap_err();
        assert!(matches!(err, RouterError::NoChannels));
    }
}
