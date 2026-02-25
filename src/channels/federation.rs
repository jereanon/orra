//! Federation protocol types.
//!
//! Shared data structures used by the federation wire protocol. These types
//! are serialised to/from JSON for communication between herald instances.
//!
//! ## Protocol Overview
//!
//! - `GET  /api/federation/agents`        → `Vec<RemoteAgentInfo>`
//! - `POST /api/federation/relay`         → `RelayRequest` / `RelayResponse`
//! - `GET  /api/federation/health`        → `HealthStatus`
//! - `GET  /api/federation/sessions`      → `Vec<FederatedSessionInfo>`
//! - `GET  /api/federation/sessions/detail` → `FederatedSessionDetail`
//! - `POST /api/federation/sessions/chat` → `SessionChatRequest` / `SessionChatResponse`
//!
//! Authentication: `Authorization: Bearer <shared_secret>`

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Agent discovery
// ---------------------------------------------------------------------------

/// Information about an agent exposed by a peer instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteAgentInfo {
    /// Agent name (e.g. "Atlas", "CodeBot").
    pub name: String,
    /// Short personality description.
    pub personality: String,
    /// LLM model the agent uses (e.g. "claude-opus-4-6").
    pub model: String,
    /// The instance that hosts this agent.
    pub instance: String,
}

// ---------------------------------------------------------------------------
// Message relay
// ---------------------------------------------------------------------------

/// A request to relay a message to a remote agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayRequest {
    /// Target agent name on the remote instance.
    pub agent: String,
    /// The user message to send.
    pub message: String,
    /// The peer that originated this request.
    pub source_peer: String,
    /// The agent that initiated the delegation (if any).
    pub source_agent: Option<String>,
    /// Namespace key for session continuity (e.g. "federation:peer:uuid").
    pub namespace: String,
    /// URL on the originating peer for tool execution callbacks.
    /// When present, the receiving peer should proxy tool calls back to this
    /// URL instead of executing them locally.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_callback_url: Option<String>,
    /// Shared secret for authenticating tool callback requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_callback_secret: Option<String>,
}

/// A response from a remote agent after processing a relayed message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayResponse {
    /// The agent's response text.
    pub message: String,
    /// The agent that produced this response.
    pub agent: String,
    /// The instance that processed the request.
    pub instance: String,
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

/// Health status returned by `GET /api/federation/health`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Instance name.
    pub instance: String,
    /// Overall status ("ok", "degraded", etc.).
    pub status: String,
    /// Number of agents available on this instance.
    pub agent_count: usize,
}

// ---------------------------------------------------------------------------
// Session federation
// ---------------------------------------------------------------------------

/// Summary of a session exposed to federated peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedSessionInfo {
    /// Namespace key (e.g. "web:abc-123").
    pub namespace: String,
    /// Optional human-readable session name.
    pub name: Option<String>,
    /// Number of messages in the session.
    pub message_count: usize,
    /// ISO-8601 creation timestamp.
    pub created_at: String,
    /// ISO-8601 last-updated timestamp.
    pub updated_at: String,
    /// The instance that owns this session.
    pub instance: String,
}

/// Full session detail including messages, for cross-instance viewing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedSessionDetail {
    /// Namespace key.
    pub namespace: String,
    /// All messages in the session.
    pub messages: Vec<FederatedMessageInfo>,
    /// ISO-8601 creation timestamp.
    pub created_at: String,
    /// ISO-8601 last-updated timestamp.
    pub updated_at: String,
    /// The instance that owns this session.
    pub instance: String,
}

/// A single message within a federated session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedMessageInfo {
    /// Role: "user", "assistant", "system".
    pub role: String,
    /// Message text content.
    pub content: String,
}

/// Request to chat within a remote session (non-streaming).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionChatRequest {
    /// Namespace of the target session.
    pub namespace: String,
    /// The user's message.
    pub message: String,
    /// Optional agent name to route to.
    pub agent: Option<String>,
    /// Optional model override.
    pub model: Option<String>,
    /// The peer instance originating this request.
    pub source_peer: String,
}

/// Response from chatting in a remote session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionChatResponse {
    /// The agent's response text.
    pub message: String,
    /// Namespace of the session.
    pub namespace: String,
    /// The agent that responded (if known).
    pub agent: Option<String>,
    /// The instance that processed the request.
    pub instance: String,
}

// ---------------------------------------------------------------------------
// Tool execution callback (federation relay)
// ---------------------------------------------------------------------------

/// Request to execute tool calls on the originating peer.
///
/// Sent by the receiving peer back to the originating peer when a relayed
/// agent invocation produces tool calls. The originating peer runs the tools
/// locally and returns results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecRequest {
    /// Namespace for session/hook context.
    pub namespace: String,
    /// Tool calls to execute.
    pub tool_calls: Vec<ToolCallInfo>,
}

/// A single tool call to be executed remotely.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInfo {
    /// Unique ID for this tool call (matches the LLM's call ID).
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Tool arguments (JSON object).
    pub arguments: serde_json::Value,
}

/// Response containing tool execution results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecResponse {
    /// Results for each tool call, in the same order as the request.
    pub results: Vec<ToolResultInfo>,
}

/// Result of a single tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultInfo {
    /// The call ID this result corresponds to.
    pub call_id: String,
    /// Output content from the tool.
    pub content: String,
    /// Whether the tool returned an error.
    pub is_error: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remote_agent_info_round_trip() {
        let info = RemoteAgentInfo {
            name: "Atlas".into(),
            personality: "friendly and helpful".into(),
            model: "claude-opus-4-6".into(),
            instance: "work-herald".into(),
        };

        let json = serde_json::to_string(&info).unwrap();
        let decoded: RemoteAgentInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Atlas");
        assert_eq!(decoded.instance, "work-herald");
    }

    #[test]
    fn relay_request_round_trip() {
        let req = RelayRequest {
            agent: "CodeBot".into(),
            message: "Help me refactor this function".into(),
            source_peer: "home-herald".into(),
            source_agent: Some("Atlas".into()),
            namespace: "federation:home-herald:abc123".into(),
            tool_callback_url: None,
            tool_callback_secret: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        let decoded: RelayRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.agent, "CodeBot");
        assert_eq!(decoded.source_peer, "home-herald");
        assert_eq!(decoded.source_agent, Some("Atlas".into()));
        assert!(decoded.tool_callback_url.is_none());
    }

    #[test]
    fn relay_response_round_trip() {
        let resp = RelayResponse {
            message: "Here's the refactored code...".into(),
            agent: "CodeBot".into(),
            instance: "work-herald".into(),
        };

        let json = serde_json::to_string(&resp).unwrap();
        let decoded: RelayResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.agent, "CodeBot");
        assert_eq!(decoded.instance, "work-herald");
    }

    #[test]
    fn health_status_round_trip() {
        let status = HealthStatus {
            instance: "my-herald".into(),
            status: "ok".into(),
            agent_count: 3,
        };

        let json = serde_json::to_string(&status).unwrap();
        let decoded: HealthStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.instance, "my-herald");
        assert_eq!(decoded.agent_count, 3);
    }

    #[test]
    fn relay_request_without_source_agent() {
        let req = RelayRequest {
            agent: "Atlas".into(),
            message: "Hello".into(),
            source_peer: "peer-1".into(),
            source_agent: None,
            namespace: "federation:peer-1:xyz".into(),
            tool_callback_url: None,
            tool_callback_secret: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        let decoded: RelayRequest = serde_json::from_str(&json).unwrap();
        assert!(decoded.source_agent.is_none());
    }

    #[test]
    fn relay_request_with_tool_callback() {
        let req = RelayRequest {
            agent: "CodeBot".into(),
            message: "Check the logs".into(),
            source_peer: "home-herald".into(),
            source_agent: Some("Atlas".into()),
            namespace: "federation:home-herald:abc123".into(),
            tool_callback_url: Some("http://home:8082/api/federation/tool-exec".into()),
            tool_callback_secret: Some("secret123".into()),
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("tool_callback_url"));
        let decoded: RelayRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.tool_callback_url.as_deref(),
            Some("http://home:8082/api/federation/tool-exec")
        );
        assert_eq!(decoded.tool_callback_secret.as_deref(), Some("secret123"));
    }

    #[test]
    fn relay_request_backward_compatible_deserialization() {
        // Old format without tool_callback fields should still deserialize
        let json = r#"{
            "agent": "Atlas",
            "message": "Hello",
            "source_peer": "peer-1",
            "namespace": "federation:peer-1:xyz"
        }"#;
        let decoded: RelayRequest = serde_json::from_str(json).unwrap();
        assert_eq!(decoded.agent, "Atlas");
        assert!(decoded.tool_callback_url.is_none());
        assert!(decoded.tool_callback_secret.is_none());
    }

    #[test]
    fn tool_exec_request_round_trip() {
        let req = ToolExecRequest {
            namespace: "federation:peer:abc".into(),
            tool_calls: vec![
                ToolCallInfo {
                    id: "call_1".into(),
                    name: "exec".into(),
                    arguments: serde_json::json!({"command": "ls"}),
                },
                ToolCallInfo {
                    id: "call_2".into(),
                    name: "read_file".into(),
                    arguments: serde_json::json!({"path": "/tmp/test"}),
                },
            ],
        };

        let json = serde_json::to_string(&req).unwrap();
        let decoded: ToolExecRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.tool_calls.len(), 2);
        assert_eq!(decoded.tool_calls[0].name, "exec");
        assert_eq!(decoded.tool_calls[1].id, "call_2");
    }

    #[test]
    fn tool_exec_response_round_trip() {
        let resp = ToolExecResponse {
            results: vec![
                ToolResultInfo {
                    call_id: "call_1".into(),
                    content: "file1.txt\nfile2.txt".into(),
                    is_error: false,
                },
                ToolResultInfo {
                    call_id: "call_2".into(),
                    content: "permission denied".into(),
                    is_error: true,
                },
            ],
        };

        let json = serde_json::to_string(&resp).unwrap();
        let decoded: ToolExecResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.results.len(), 2);
        assert!(!decoded.results[0].is_error);
        assert!(decoded.results[1].is_error);
    }
}
