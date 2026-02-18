use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("execution failed: {0}")]
    ExecutionFailed(String),
}

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name = tool.definition().name.clone();
        self.tools.insert(name, tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn remove(&mut self, name: &str) -> bool {
        self.tools.remove(name).is_some()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".into(),
                description: "Echoes the input back".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"}
                    },
                    "required": ["text"]
                }),
            }
        }

        async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
            let text = input
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ToolError::InvalidInput("missing 'text' field".into()))?;
            Ok(text.to_string())
        }
    }

    struct FailTool;

    #[async_trait]
    impl Tool for FailTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "fail".into(),
                description: "Always fails".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }
        }

        async fn execute(&self, _input: serde_json::Value) -> Result<String, ToolError> {
            Err(ToolError::ExecutionFailed("something went wrong".into()))
        }
    }

    #[tokio::test]
    async fn echo_tool_works() {
        let tool = EchoTool;
        let result = tool
            .execute(serde_json::json!({"text": "hello"}))
            .await
            .unwrap();
        assert_eq!(result, "hello");
    }

    #[tokio::test]
    async fn echo_tool_invalid_input() {
        let tool = EchoTool;
        let err = tool
            .execute(serde_json::json!({"wrong": "field"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn fail_tool_returns_error() {
        let tool = FailTool;
        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
    }

    #[test]
    fn registry_register_and_get() {
        let mut registry = ToolRegistry::new();
        assert!(registry.is_empty());

        registry.register(Box::new(EchoTool));
        assert_eq!(registry.len(), 1);
        assert!(registry.get("echo").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn registry_definitions() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        registry.register(Box::new(FailTool));

        let defs = registry.definitions();
        assert_eq!(defs.len(), 2);
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"echo"));
        assert!(names.contains(&"fail"));
    }

    #[test]
    fn registry_remove() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        assert_eq!(registry.len(), 1);

        assert!(registry.remove("echo"));
        assert!(registry.is_empty());
        assert!(!registry.remove("echo")); // already removed
    }

    #[test]
    fn registry_names() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));
        registry.register(Box::new(FailTool));

        let mut names = registry.names();
        names.sort();
        assert_eq!(names, vec!["echo", "fail"]);
    }

    #[test]
    fn tool_definition_serialization() {
        let def = ToolDefinition {
            name: "search".into(),
            description: "Search the web".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }),
        };

        let json = serde_json::to_string(&def).unwrap();
        let deserialized: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "search");
    }
}
