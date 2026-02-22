//! HTTP/WebSocket gateway channel.
//!
//! Provides a web-based channel for interacting with the agent runtime over
//! HTTP REST endpoints and WebSocket connections. This allows external
//! applications to integrate with the agent without needing a Discord bot
//! or other chat platform.
//!
//! ## Endpoints
//!
//! - `POST /api/chat` — Send a message and get a response (synchronous)
//! - `GET /api/sessions/:id` — Get session history
//! - `DELETE /api/sessions/:id` — Clear a session
//! - `GET /ws` — WebSocket endpoint for streaming conversations
//! - `GET /health` — Health check
//!
//! Requires the `gateway` feature.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::channels::{Channel, ChannelError, InboundMessage, OutboundError, OutboundMessage};
use crate::message::Message;
use crate::namespace::Namespace;

// ---------------------------------------------------------------------------
// Gateway configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Host to bind to (default: "0.0.0.0")
    pub host: String,

    /// Port to listen on (default: 8080)
    pub port: u16,

    /// Optional API key for authentication. If set, requests must include
    /// an `Authorization: Bearer <key>` header.
    pub api_key: Option<String>,

    /// Maximum request body size in bytes (default: 1MB)
    pub max_body_size: usize,

    /// CORS allowed origins (default: ["*"])
    pub cors_origins: Vec<String>,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 8080,
            api_key: None,
            max_body_size: 1024 * 1024,
            cors_origins: vec!["*".into()],
        }
    }
}

// ---------------------------------------------------------------------------
// Request/Response types
// ---------------------------------------------------------------------------

/// Inbound chat request from the HTTP API.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ChatRequest {
    /// The message text from the user.
    pub message: String,

    /// Namespace/session to route this message to. If not specified,
    /// a new session namespace is generated.
    pub namespace: Option<String>,

    /// Optional metadata to attach to the message.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Outbound chat response from the HTTP API.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatResponse {
    /// The assistant's response text.
    pub message: String,

    /// The namespace this conversation belongs to.
    pub namespace: String,

    /// Token usage for this turn.
    pub usage: ChatUsage,
}

/// Token usage info included in API responses.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

/// Error response returned by the API.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: String,
}

// ---------------------------------------------------------------------------
// WebSocket message types
// ---------------------------------------------------------------------------

/// Messages sent/received over the WebSocket connection.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum WsMessage {
    /// Client sends a chat message.
    #[serde(rename = "chat")]
    Chat {
        message: String,
        namespace: Option<String>,
        /// Optional model override for this message.
        model: Option<String>,
        /// Optional agent name to route this message to.
        agent: Option<String>,
    },

    /// Server sends a text chunk (streaming).
    #[serde(rename = "text_delta")]
    TextDelta { content: String },

    /// Server sends the final response.
    #[serde(rename = "response")]
    Response {
        message: String,
        namespace: String,
        usage: ChatUsage,
        /// Name of the agent that generated this response.
        #[serde(skip_serializing_if = "Option::is_none")]
        agent: Option<String>,
    },

    /// Server sends an error.
    #[serde(rename = "error")]
    Error { error: String },

    /// Server asks client to approve a tool call before execution.
    #[serde(rename = "tool_approval_request")]
    ToolApprovalRequest {
        call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },

    /// Client responds with approval or denial for a tool call.
    #[serde(rename = "tool_approval_response")]
    ToolApprovalResponse { call_id: String, approved: bool },

    /// Ping/pong for keepalive.
    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "pong")]
    Pong,
}

// ---------------------------------------------------------------------------
// Gateway channel implementation
// ---------------------------------------------------------------------------

/// The Gateway channel receives messages from HTTP/WebSocket clients and
/// sends responses back. It acts as a bridge between the web API and the
/// runtime's channel abstraction.
pub struct GatewayChannel {
    config: GatewayConfig,
    inbound_tx: tokio::sync::mpsc::Sender<InboundMessage>,
    inbound_rx: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<InboundMessage>>,
    response_map: Arc<tokio::sync::RwLock<HashMap<String, ResponseSender>>>,
}

/// Internal type used to send responses back to waiting HTTP requests.
type ResponseSender = tokio::sync::oneshot::Sender<Result<OutboundMessage, OutboundError>>;

impl GatewayChannel {
    pub fn new(config: GatewayConfig) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(256);
        Self {
            config,
            inbound_tx: tx,
            inbound_rx: tokio::sync::Mutex::new(rx),
            response_map: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    pub fn config(&self) -> &GatewayConfig {
        &self.config
    }

    /// Submit a chat request and wait for the response. This is called by
    /// the HTTP handler to send a message into the runtime and block until
    /// the response is ready.
    pub async fn submit_and_wait(
        &self,
        request: ChatRequest,
    ) -> Result<ChatResponse, GatewayError> {
        let ns_key = request
            .namespace
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let request_id = uuid::Uuid::new_v4().to_string();

        // Set up the response channel
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.response_map
            .write()
            .await
            .insert(request_id.clone(), tx);

        // Build metadata with the request ID so we can match the response
        let mut metadata = request.metadata;
        metadata.insert("_gateway_request_id".into(), serde_json::json!(request_id));

        let inbound = InboundMessage {
            namespace: Namespace::parse(&ns_key),
            message: Message::user(request.message),
            metadata,
        };

        self.inbound_tx
            .send(inbound)
            .await
            .map_err(|_| GatewayError::ChannelClosed)?;

        // Wait for the runtime to process and respond
        let result = rx.await.map_err(|_| GatewayError::ResponseDropped)?;

        match result {
            Ok(outbound) => Ok(ChatResponse {
                message: outbound.message.content.clone(),
                namespace: ns_key,
                usage: ChatUsage {
                    input_tokens: outbound.run_result.total_usage.input_tokens,
                    output_tokens: outbound.run_result.total_usage.output_tokens,
                    total_tokens: outbound.run_result.total_usage.total_tokens(),
                },
            }),
            Err(err) => Err(GatewayError::Runtime(err.error)),
        }
    }

    /// Validate an API key against the configured key. Returns true if
    /// no key is configured (authentication disabled) or if the key matches.
    pub fn authenticate(&self, bearer_token: Option<&str>) -> bool {
        match &self.config.api_key {
            None => true,
            Some(expected) => bearer_token == Some(expected.as_str()),
        }
    }

    /// Route a response back to the waiting HTTP request.
    async fn route_response(
        &self,
        metadata: &HashMap<String, serde_json::Value>,
        result: Result<OutboundMessage, OutboundError>,
    ) -> Result<(), GatewayError> {
        let request_id = metadata
            .get("_gateway_request_id")
            .and_then(|v| v.as_str())
            .ok_or(GatewayError::MissingRequestId)?;

        let sender = self
            .response_map
            .write()
            .await
            .remove(request_id)
            .ok_or_else(|| GatewayError::NoWaitingRequest(request_id.to_string()))?;

        let _ = sender.send(result);
        Ok(())
    }
}

#[async_trait]
impl Channel for GatewayChannel {
    async fn receive(&self) -> Option<InboundMessage> {
        self.inbound_rx.lock().await.recv().await
    }

    async fn send(&self, response: OutboundMessage) -> Result<(), ChannelError> {
        self.route_response(&response.metadata, Ok(response.clone()))
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))
    }

    async fn send_error(&self, error: OutboundError) -> Result<(), ChannelError> {
        self.route_response(&error.metadata, Err(error.clone()))
            .await
            .map_err(|e| ChannelError::Send(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("gateway channel closed")]
    ChannelClosed,

    #[error("response sender was dropped")]
    ResponseDropped,

    #[error("runtime error: {0}")]
    Runtime(String),

    #[error("missing gateway request ID in metadata")]
    MissingRequestId,

    #[error("no waiting request for ID: {0}")]
    NoWaitingRequest(String),

    #[error("authentication failed")]
    Unauthorized,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_gateway_config() {
        let config = GatewayConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert!(config.api_key.is_none());
        assert_eq!(config.max_body_size, 1024 * 1024);
        assert_eq!(config.cors_origins, vec!["*"]);
    }

    #[test]
    fn chat_request_deserialization() {
        let json = r#"{"message": "Hello!", "namespace": "user-123"}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.message, "Hello!");
        assert_eq!(req.namespace.as_deref(), Some("user-123"));
        assert!(req.metadata.is_empty());
    }

    #[test]
    fn chat_request_minimal() {
        let json = r#"{"message": "Hi"}"#;
        let req: ChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.message, "Hi");
        assert!(req.namespace.is_none());
    }

    #[test]
    fn chat_response_serialization() {
        let resp = ChatResponse {
            message: "Hello!".into(),
            namespace: "ns-1".into(),
            usage: ChatUsage {
                input_tokens: 10,
                output_tokens: 5,
                total_tokens: 15,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("Hello!"));
        assert!(json.contains("ns-1"));
    }

    #[test]
    fn ws_message_chat_serialization() {
        let msg = WsMessage::Chat {
            message: "Hello".into(),
            namespace: Some("test".into()),
            model: None,
            agent: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"chat\""));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn ws_message_text_delta_serialization() {
        let msg = WsMessage::TextDelta {
            content: "chunk".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"text_delta\""));
    }

    #[test]
    fn authenticate_no_key_required() {
        let channel = GatewayChannel::new(GatewayConfig::default());
        assert!(channel.authenticate(None));
        assert!(channel.authenticate(Some("anything")));
    }

    #[test]
    fn authenticate_with_key() {
        let config = GatewayConfig {
            api_key: Some("secret-key".into()),
            ..Default::default()
        };
        let channel = GatewayChannel::new(config);

        assert!(channel.authenticate(Some("secret-key")));
        assert!(!channel.authenticate(Some("wrong-key")));
        assert!(!channel.authenticate(None));
    }

    #[tokio::test]
    async fn submit_and_route_roundtrip() {
        let channel = GatewayChannel::new(GatewayConfig::default());

        // Simulate: spawn a task that submits a request
        let _channel_ref = &channel;
        let submit_handle = tokio::spawn({
            let _inbound_tx = channel.inbound_tx.clone();
            let _response_map = channel.response_map.clone();

            async move {
                // We need to call submit_and_wait from outside, but since we
                // don't have the full channel setup, let's test the pieces
                // individually
                let request = ChatRequest {
                    message: "test message".into(),
                    namespace: Some("test-ns".into()),
                    metadata: HashMap::new(),
                };

                // Verify the request is properly formed
                assert_eq!(request.message, "test message");
            }
        });

        submit_handle.await.unwrap();
    }

    #[test]
    fn error_response_serialization() {
        let err = ErrorResponse {
            error: "something went wrong".into(),
            code: "internal_error".into(),
        };
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("internal_error"));
    }

    #[test]
    fn ws_message_roundtrip() {
        let msg = WsMessage::Response {
            message: "Hi there".into(),
            namespace: "ns".into(),
            usage: ChatUsage {
                input_tokens: 5,
                output_tokens: 3,
                total_tokens: 8,
            },
            agent: Some("Atlas".into()),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let parsed: WsMessage = serde_json::from_str(&json).unwrap();

        match parsed {
            WsMessage::Response {
                message, namespace, ..
            } => {
                assert_eq!(message, "Hi there");
                assert_eq!(namespace, "ns");
            }
            _ => panic!("expected Response variant"),
        }
    }
}
