use std::sync::Arc;

use async_trait::async_trait;

use super::client::McpClient;
use super::types::McpToolDefinition;
use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

/// A tool backed by an MCP server. Implements the `Tool` trait so it can be
/// registered in a `ToolRegistry` and used by the runtime like any other tool.
pub struct McpTool {
    client: Arc<McpClient>,
    mcp_definition: McpToolDefinition,
}

impl McpTool {
    pub fn new(client: Arc<McpClient>, definition: McpToolDefinition) -> Self {
        Self {
            client,
            mcp_definition: definition,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.mcp_definition.name.clone(),
            description: self.mcp_definition.description.clone().unwrap_or_default(),
            input_schema: self.mcp_definition.input_schema.clone(),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let arguments = if input.is_null() || input == serde_json::json!({}) {
            None
        } else {
            Some(input)
        };

        let result = self
            .client
            .call_tool(&self.mcp_definition.name, arguments)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        // Extract text content from the result
        let text: String = result
            .content
            .iter()
            .filter_map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join("\n");

        if result.is_error {
            Err(ToolError::ExecutionFailed(text))
        } else {
            Ok(text)
        }
    }
}

/// Connect to an MCP server via the given transport, perform the handshake,
/// discover tools, and register them all into the tool registry.
pub async fn register_mcp_tools(
    registry: &mut ToolRegistry,
    transport: Arc<dyn super::transport::McpTransport>,
) -> Result<Arc<McpClient>, super::client::McpError> {
    let client = Arc::new(McpClient::new(transport));

    // Initialize handshake
    client.initialize().await?;

    // Discover tools
    let tools = client.list_tools().await?;

    // Register each tool
    for tool_def in tools {
        let mcp_tool = McpTool::new(client.clone(), tool_def);
        registry.register(Box::new(mcp_tool));
    }

    Ok(client)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::transport::MockTransport;
    use crate::mcp::types::{JsonRpcResponse, McpContent, McpToolDefinition, McpToolResult};

    #[tokio::test]
    async fn mcp_tool_implements_tool_trait() {
        let (transport, mut req_rx, resp_tx) = MockTransport::new();
        let client = Arc::new(McpClient::new(Arc::new(transport)));

        let tool = McpTool::new(
            client,
            McpToolDefinition {
                name: "greet".into(),
                description: Some("Say hello".into()),
                input_schema: serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}}),
            },
        );

        // Check definition
        let def = tool.definition();
        assert_eq!(def.name, "greet");
        assert_eq!(def.description, "Say hello");

        // Run the tool
        let server_task = tokio::spawn(async move {
            let req = req_rx.recv().await.unwrap();
            assert_eq!(req.method, "tools/call");
            resp_tx.send(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: Some(serde_json::to_value(McpToolResult {
                    content: vec![McpContent::Text { text: "Hello, Alice!".into() }],
                    is_error: false,
                }).unwrap()),
                error: None,
            }).await.unwrap();
        });

        let result = tool.execute(serde_json::json!({"name": "Alice"})).await.unwrap();
        assert_eq!(result, "Hello, Alice!");
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn mcp_tool_error_result() {
        let (transport, mut req_rx, resp_tx) = MockTransport::new();
        let client = Arc::new(McpClient::new(Arc::new(transport)));

        let tool = McpTool::new(
            client,
            McpToolDefinition {
                name: "fail".into(),
                description: None,
                input_schema: serde_json::json!({"type": "object"}),
            },
        );

        let server_task = tokio::spawn(async move {
            let req = req_rx.recv().await.unwrap();
            resp_tx.send(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: Some(serde_json::to_value(McpToolResult {
                    content: vec![McpContent::Text { text: "something broke".into() }],
                    is_error: true,
                }).unwrap()),
                error: None,
            }).await.unwrap();
        });

        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn register_mcp_tools_populates_registry() {
        let (transport, mut req_rx, resp_tx) = MockTransport::new();
        let transport = Arc::new(transport);

        let server_task = tokio::spawn(async move {
            // Initialize
            let req = req_rx.recv().await.unwrap();
            resp_tx.send(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "test", "version": "1.0"}
                })),
                error: None,
            }).await.unwrap();

            // List tools
            let req = req_rx.recv().await.unwrap();
            resp_tx.send(JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: Some(serde_json::json!({
                    "tools": [
                        {"name": "tool_a", "description": "A", "inputSchema": {"type": "object"}},
                        {"name": "tool_b", "description": "B", "inputSchema": {"type": "object"}}
                    ]
                })),
                error: None,
            }).await.unwrap();
        });

        let mut registry = ToolRegistry::new();
        register_mcp_tools(&mut registry, transport).await.unwrap();

        assert_eq!(registry.len(), 2);
        assert!(registry.get("tool_a").is_some());
        assert!(registry.get("tool_b").is_some());
        server_task.await.unwrap();
    }
}
