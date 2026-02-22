//! Federation protocol types.
//!
//! Shared data structures used by the federation wire protocol. These types
//! are serialised to/from JSON for communication between herald instances.
//!
//! ## Protocol Overview
//!
//! - `GET  /api/federation/agents` → `Vec<RemoteAgentInfo>`
//! - `POST /api/federation/relay`  → `RelayRequest` / `RelayResponse`
//! - `GET  /api/federation/health` → `HealthStatus`
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
        };

        let json = serde_json::to_string(&req).unwrap();
        let decoded: RelayRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.agent, "CodeBot");
        assert_eq!(decoded.source_peer, "home-herald");
        assert_eq!(decoded.source_agent, Some("Atlas".into()));
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
        };

        let json = serde_json::to_string(&req).unwrap();
        let decoded: RelayRequest = serde_json::from_str(&json).unwrap();
        assert!(decoded.source_agent.is_none());
    }
}
