use serde::{Deserialize, Serialize};

// --- JSON-RPC 2.0 ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub id: u64,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    pub fn new(id: u64, method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            method: method.into(),
            params,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

// --- MCP protocol types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolResult {
    #[serde(default)]
    pub content: Vec<McpContent>,
    #[serde(default, rename = "isError")]
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
    #[serde(rename = "resource")]
    Resource { resource: serde_json::Value },
}

impl McpContent {
    pub fn as_text(&self) -> Option<&str> {
        match self {
            McpContent::Text { text } => Some(text),
            _ => None,
        }
    }
}

// --- Initialize types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ClientCapabilities,
    #[serde(rename = "clientInfo")]
    pub client_info: Implementation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCapabilities {
    #[serde(default)]
    pub roots: Option<serde_json::Value>,
    #[serde(default)]
    pub sampling: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Implementation {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    #[serde(rename = "serverInfo")]
    pub server_info: Implementation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
}

// --- List tools ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListToolsResult {
    pub tools: Vec<McpToolDefinition>,
}

// --- Call tool ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallToolParams {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_rpc_request_serialization() {
        let req = JsonRpcRequest::new(1, "initialize", Some(serde_json::json!({"key": "value"})));
        let json = serde_json::to_string(&req).unwrap();
        let parsed: JsonRpcRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, 1);
        assert_eq!(parsed.method, "initialize");
        assert_eq!(parsed.jsonrpc, "2.0");
    }

    #[test]
    fn json_rpc_response_with_result() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn json_rpc_response_with_error() {
        let json = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
        assert_eq!(err.message, "Method not found");
    }

    #[test]
    fn mcp_tool_definition_roundtrip() {
        let def = McpToolDefinition {
            name: "search".into(),
            description: Some("Search the web".into()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"query": {"type": "string"}}
            }),
        };

        let json = serde_json::to_string(&def).unwrap();
        let parsed: McpToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "search");
        assert_eq!(parsed.description, Some("Search the web".into()));
    }

    #[test]
    fn mcp_tool_definition_to_tool_definition() {
        let mcp_def = McpToolDefinition {
            name: "calculate".into(),
            description: Some("Math".into()),
            input_schema: serde_json::json!({"type": "object"}),
        };

        let tool_def = crate::tool::ToolDefinition {
            name: mcp_def.name.clone(),
            description: mcp_def.description.clone().unwrap_or_default(),
            input_schema: mcp_def.input_schema.clone(),
        };

        assert_eq!(tool_def.name, "calculate");
        assert_eq!(tool_def.description, "Math");
    }

    #[test]
    fn mcp_tool_result_text_extraction() {
        let result = McpToolResult {
            content: vec![
                McpContent::Text { text: "line 1".into() },
                McpContent::Text { text: "line 2".into() },
            ],
            is_error: false,
        };

        let texts: Vec<&str> = result.content.iter().filter_map(|c| c.as_text()).collect();
        assert_eq!(texts, vec!["line 1", "line 2"]);
    }

    #[test]
    fn mcp_tool_result_error() {
        let json = r#"{"content":[{"type":"text","text":"something broke"}],"isError":true}"#;
        let result: McpToolResult = serde_json::from_str(json).unwrap();
        assert!(result.is_error);
        assert_eq!(result.content[0].as_text(), Some("something broke"));
    }

    #[test]
    fn call_tool_params_serialization() {
        let params = CallToolParams {
            name: "search".into(),
            arguments: Some(serde_json::json!({"query": "rust"})),
        };
        let json = serde_json::to_string(&params).unwrap();
        let parsed: CallToolParams = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "search");
    }
}
