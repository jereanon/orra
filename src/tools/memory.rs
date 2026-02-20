use std::sync::Arc;

use async_trait::async_trait;

use crate::memory::{MemoryManager, MemorySearchResult};
use crate::namespace::Namespace;
use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

// ---------------------------------------------------------------------------
// Remember tool
// ---------------------------------------------------------------------------

pub struct RememberTool {
    manager: Arc<MemoryManager>,
}

impl RememberTool {
    pub fn new(manager: Arc<MemoryManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for RememberTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "remember".into(),
            description: "Store a piece of information in long-term memory for later retrieval. \
                          Use this when the user tells you something important, shares a preference, \
                          or when you learn a fact worth remembering across conversations."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "The namespace to store this memory under (e.g. user ID or channel)"
                    },
                    "content": {
                        "type": "string",
                        "description": "The information to remember"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional tags for categorization (e.g. 'preference', 'fact', 'project')"
                    }
                },
                "required": ["namespace", "content"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let ns_key = input
            .get("namespace")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'namespace'".into()))?;

        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'content'".into()))?;

        let tags: Vec<String> = input
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let ns = Namespace::parse(ns_key);

        let id = self
            .manager
            .remember(&ns, content, tags)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        Ok(format!("Stored memory {}", id))
    }
}

// ---------------------------------------------------------------------------
// Recall tool
// ---------------------------------------------------------------------------

pub struct RecallTool {
    manager: Arc<MemoryManager>,
}

impl RecallTool {
    pub fn new(manager: Arc<MemoryManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for RecallTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "recall".into(),
            description: "Search long-term memory for relevant information. Use this when you \
                          need to recall something that was previously stored, like user preferences, \
                          past conversations, or learned facts."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "The namespace to search in"
                    },
                    "query": {
                        "type": "string",
                        "description": "What to search for"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default: 5)"
                    }
                },
                "required": ["namespace", "query"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let ns_key = input
            .get("namespace")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'namespace'".into()))?;

        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'query'".into()))?;

        let limit = input
            .get("limit")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        let ns = Namespace::parse(ns_key);

        let results = self
            .manager
            .recall(&ns, query, limit)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if results.is_empty() {
            return Ok("No relevant memories found.".into());
        }

        Ok(format_results(&results))
    }
}

// ---------------------------------------------------------------------------
// Forget tool
// ---------------------------------------------------------------------------

pub struct ForgetTool {
    manager: Arc<MemoryManager>,
}

impl ForgetTool {
    pub fn new(manager: Arc<MemoryManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for ForgetTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "forget".into(),
            description: "Delete a specific memory by its ID. Use this when the user asks you \
                          to forget something or when stored information is no longer relevant."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The memory ID to delete"
                    }
                },
                "required": ["id"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let id = input
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'id'".into()))?;

        let deleted = self
            .manager
            .forget(id)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if deleted {
            Ok(format!("Deleted memory {}.", id))
        } else {
            Ok(format!("Memory {} not found.", id))
        }
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all memory tools (remember, recall, forget) into a tool registry.
pub fn register_tools(registry: &mut ToolRegistry, manager: &Arc<MemoryManager>) {
    registry.register(Box::new(RememberTool::new(manager.clone())));
    registry.register(Box::new(RecallTool::new(manager.clone())));
    registry.register(Box::new(ForgetTool::new(manager.clone())));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_results(results: &[MemorySearchResult]) -> String {
    let mut lines = Vec::new();
    for (i, r) in results.iter().enumerate() {
        let tags_str = if r.entry.tags.is_empty() {
            String::new()
        } else {
            format!(" [{}]", r.entry.tags.join(", "))
        };
        lines.push(format!(
            "{}. (score: {:.2}, id: {}){}\n   {}",
            i + 1,
            r.score,
            r.entry.id,
            tags_str,
            r.entry.content,
        ));
    }
    lines.join("\n\n")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::InMemoryMemoryStore;

    fn make_manager() -> Arc<MemoryManager> {
        let store = Arc::new(InMemoryMemoryStore::new());
        Arc::new(MemoryManager::new(store))
    }

    #[test]
    fn tool_definitions_valid() {
        let mgr = make_manager();

        let remember = RememberTool::new(mgr.clone());
        let recall = RecallTool::new(mgr.clone());
        let forget = ForgetTool::new(mgr);

        assert_eq!(remember.definition().name, "remember");
        assert_eq!(recall.definition().name, "recall");
        assert_eq!(forget.definition().name, "forget");

        // All should have required fields
        assert!(remember.definition().input_schema["required"].as_array().is_some());
        assert!(recall.definition().input_schema["required"].as_array().is_some());
        assert!(forget.definition().input_schema["required"].as_array().is_some());
    }

    #[tokio::test]
    async fn remember_and_recall_roundtrip() {
        let mgr = make_manager();

        let remember = RememberTool::new(mgr.clone());
        let recall = RecallTool::new(mgr);

        remember
            .execute(serde_json::json!({
                "namespace": "test",
                "content": "Paris is the capital of France",
                "tags": ["geography"]
            }))
            .await
            .unwrap();

        let result = recall
            .execute(serde_json::json!({
                "namespace": "test",
                "query": "capital France"
            }))
            .await
            .unwrap();

        assert!(result.contains("Paris"));
    }

    #[tokio::test]
    async fn forget_removes_memory() {
        let mgr = make_manager();

        let remember = RememberTool::new(mgr.clone());
        let forget = ForgetTool::new(mgr.clone());
        let recall = RecallTool::new(mgr);

        let stored = remember
            .execute(serde_json::json!({
                "namespace": "test",
                "content": "temporary info"
            }))
            .await
            .unwrap();

        // Extract ID from "Stored memory <id>"
        let id = stored.strip_prefix("Stored memory ").unwrap().trim();

        let deleted = forget
            .execute(serde_json::json!({ "id": id }))
            .await
            .unwrap();
        assert!(deleted.contains("Deleted"));

        let results = recall
            .execute(serde_json::json!({
                "namespace": "test",
                "query": "temporary"
            }))
            .await
            .unwrap();
        assert!(results.contains("No relevant memories"));
    }

    #[tokio::test]
    async fn missing_fields_return_errors() {
        let mgr = make_manager();

        let remember = RememberTool::new(mgr.clone());
        assert!(remember.execute(serde_json::json!({})).await.is_err());

        let recall = RecallTool::new(mgr.clone());
        assert!(recall.execute(serde_json::json!({})).await.is_err());

        let forget = ForgetTool::new(mgr);
        assert!(forget.execute(serde_json::json!({})).await.is_err());
    }

    #[test]
    fn register_tools_adds_three() {
        let mgr = make_manager();
        let mut registry = ToolRegistry::new();
        register_tools(&mut registry, &mgr);

        assert_eq!(registry.len(), 3);
        assert!(registry.get("remember").is_some());
        assert!(registry.get("recall").is_some());
        assert!(registry.get("forget").is_some());
    }
}
