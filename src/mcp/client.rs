use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::transport::{McpTransport, TransportError};
use super::types::*;

#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("transport error: {0}")]
    Transport(#[from] TransportError),

    #[error("server error ({code}): {message}")]
    Server { code: i64, message: String },

    #[error("unexpected response: {0}")]
    UnexpectedResponse(String),
}

/// MCP protocol client that wraps a transport.
pub struct McpClient {
    transport: Arc<dyn McpTransport>,
    next_id: AtomicU64,
}

impl McpClient {
    pub fn new(transport: Arc<dyn McpTransport>) -> Self {
        Self {
            transport,
            next_id: AtomicU64::new(1),
        }
    }

    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    async fn call(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        let request = JsonRpcRequest::new(self.next_id(), method, params);
        let response = self.transport.send(request).await?;

        if let Some(error) = response.error {
            return Err(McpError::Server {
                code: error.code,
                message: error.message,
            });
        }

        response
            .result
            .ok_or_else(|| McpError::UnexpectedResponse("no result or error".into()))
    }

    /// Perform the MCP initialization handshake.
    pub async fn initialize(&self) -> Result<InitializeResult, McpError> {
        let params = InitializeParams {
            protocol_version: "2024-11-05".into(),
            capabilities: ClientCapabilities {
                roots: None,
                sampling: None,
            },
            client_info: Implementation {
                name: "orra".into(),
                version: env!("CARGO_PKG_VERSION").into(),
            },
        };

        let result = self
            .call("initialize", Some(serde_json::to_value(&params).unwrap()))
            .await?;

        let init_result: InitializeResult = serde_json::from_value(result)
            .map_err(|e| McpError::UnexpectedResponse(format!("parse init result: {e}")))?;

        // Send initialized notification (no response expected, but we use our transport)
        // In a full implementation this would be a notification (no id), but
        // for simplicity we skip it here since it doesn't affect tool discovery.

        Ok(init_result)
    }

    /// List available tools from the MCP server.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDefinition>, McpError> {
        let result = self.call("tools/list", None).await?;

        let list_result: ListToolsResult = serde_json::from_value(result)
            .map_err(|e| McpError::UnexpectedResponse(format!("parse tools list: {e}")))?;

        Ok(list_result.tools)
    }

    /// Call a tool on the MCP server.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Option<serde_json::Value>,
    ) -> Result<McpToolResult, McpError> {
        let params = CallToolParams {
            name: name.into(),
            arguments,
        };

        let result = self
            .call("tools/call", Some(serde_json::to_value(&params).unwrap()))
            .await?;

        let tool_result: McpToolResult = serde_json::from_value(result)
            .map_err(|e| McpError::UnexpectedResponse(format!("parse tool result: {e}")))?;

        Ok(tool_result)
    }

    /// Close the underlying transport.
    pub async fn close(&self) -> Result<(), McpError> {
        self.transport.close().await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::transport::MockTransport;

    #[tokio::test]
    async fn initialize_handshake() {
        let (transport, mut req_rx, resp_tx) = MockTransport::new();
        let client = McpClient::new(Arc::new(transport));

        let server_task = tokio::spawn(async move {
            let req = req_rx.recv().await.unwrap();
            assert_eq!(req.method, "initialize");

            resp_tx
                .send(JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: req.id,
                    result: Some(serde_json::json!({
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "test-server", "version": "1.0"}
                    })),
                    error: None,
                })
                .await
                .unwrap();
        });

        let result = client.initialize().await.unwrap();
        assert_eq!(result.server_info.name, "test-server");
        assert_eq!(result.protocol_version, "2024-11-05");
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn list_tools_returns_definitions() {
        let (transport, mut req_rx, resp_tx) = MockTransport::new();
        let client = McpClient::new(Arc::new(transport));

        let server_task = tokio::spawn(async move {
            let req = req_rx.recv().await.unwrap();
            assert_eq!(req.method, "tools/list");

            resp_tx.send(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: Some(serde_json::json!({
                    "tools": [
                        {
                            "name": "search",
                            "description": "Search the web",
                            "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}
                        },
                        {
                            "name": "calculate",
                            "description": "Do math",
                            "inputSchema": {"type": "object"}
                        }
                    ]
                })),
                error: None,
            }).await.unwrap();
        });

        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "search");
        assert_eq!(tools[1].name, "calculate");
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn call_tool_success() {
        let (transport, mut req_rx, resp_tx) = MockTransport::new();
        let client = McpClient::new(Arc::new(transport));

        let server_task = tokio::spawn(async move {
            let req = req_rx.recv().await.unwrap();
            assert_eq!(req.method, "tools/call");

            let params = req.params.unwrap();
            assert_eq!(params["name"], "search");
            assert_eq!(params["arguments"]["q"], "rust");

            resp_tx
                .send(JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: req.id,
                    result: Some(serde_json::json!({
                        "content": [{"type": "text", "text": "Found 42 results"}],
                        "isError": false
                    })),
                    error: None,
                })
                .await
                .unwrap();
        });

        let result = client
            .call_tool("search", Some(serde_json::json!({"q": "rust"})))
            .await
            .unwrap();

        assert!(!result.is_error);
        assert_eq!(result.content[0].as_text(), Some("Found 42 results"));
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn call_tool_error_from_server() {
        let (transport, mut req_rx, resp_tx) = MockTransport::new();
        let client = McpClient::new(Arc::new(transport));

        let server_task = tokio::spawn(async move {
            let req = req_rx.recv().await.unwrap();
            resp_tx
                .send(JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: req.id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32601,
                        message: "Method not found".into(),
                        data: None,
                    }),
                })
                .await
                .unwrap();
        });

        let err = client.call_tool("nope", None).await.unwrap_err();
        match err {
            McpError::Server { code, message } => {
                assert_eq!(code, -32601);
                assert_eq!(message, "Method not found");
            }
            other => panic!("expected Server error, got {other:?}"),
        }
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn full_init_list_call_flow() {
        let (transport, mut req_rx, resp_tx) = MockTransport::new();
        let client = McpClient::new(Arc::new(transport));

        let server_task = tokio::spawn(async move {
            // 1. Initialize
            let req = req_rx.recv().await.unwrap();
            assert_eq!(req.method, "initialize");
            resp_tx
                .send(JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: req.id,
                    result: Some(serde_json::json!({
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "test", "version": "1.0"}
                    })),
                    error: None,
                })
                .await
                .unwrap();

            // 2. List tools
            let req = req_rx.recv().await.unwrap();
            assert_eq!(req.method, "tools/list");
            resp_tx.send(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: Some(serde_json::json!({
                    "tools": [{
                        "name": "echo",
                        "description": "Echoes input",
                        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}
                    }]
                })),
                error: None,
            }).await.unwrap();

            // 3. Call tool
            let req = req_rx.recv().await.unwrap();
            assert_eq!(req.method, "tools/call");
            let params = req.params.unwrap();
            let input_text = params["arguments"]["text"].as_str().unwrap().to_string();
            resp_tx
                .send(JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: req.id,
                    result: Some(serde_json::json!({
                        "content": [{"type": "text", "text": input_text}],
                        "isError": false
                    })),
                    error: None,
                })
                .await
                .unwrap();
        });

        // Run client flow
        let init = client.initialize().await.unwrap();
        assert_eq!(init.server_info.name, "test");

        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "echo");

        let result = client
            .call_tool("echo", Some(serde_json::json!({"text": "hello"})))
            .await
            .unwrap();
        assert_eq!(result.content[0].as_text(), Some("hello"));

        server_task.await.unwrap();
    }
}
