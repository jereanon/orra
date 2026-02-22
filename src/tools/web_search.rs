//! Web search tool using the Brave Search API.

use async_trait::async_trait;
use serde::Deserialize;

use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

/// Search the web using the Brave Search API.
pub struct WebSearchTool {
    client: reqwest::Client,
    api_key: String,
}

impl WebSearchTool {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("agentic/0.1")
                .timeout(std::time::Duration::from_secs(15))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
            api_key: api_key.into(),
        }
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "web_search".into(),
            description: "Search the web for current information. Returns titles, URLs, and \
                          snippets from search results. Use this when you need to find up-to-date \
                          information about any topic."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results to return (1-20, default: 5)"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'query'".into()))?;

        let count = input
            .get("count")
            .and_then(|v| v.as_u64())
            .unwrap_or(5)
            .min(20);

        let resp = self
            .client
            .get("https://api.search.brave.com/res/v1/web/search")
            .header("X-Subscription-Token", &self.api_key)
            .header("Accept", "application/json")
            .query(&[("q", query), ("count", &count.to_string())])
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("search error: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "Brave Search API {status}: {body}"
            )));
        }

        let search_result: BraveSearchResponse = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {e}")))?;

        let results = search_result
            .web
            .as_ref()
            .map(|w| w.results.as_slice())
            .unwrap_or(&[]);

        if results.is_empty() {
            return Ok(format!("No results found for: {query}"));
        }

        let mut lines = Vec::new();
        for (i, result) in results.iter().enumerate() {
            lines.push(format!(
                "{}. {}\n   {}\n   {}",
                i + 1,
                result.title,
                result.url,
                result.description.as_deref().unwrap_or("(no description)")
            ));
        }

        Ok(lines.join("\n\n"))
    }
}

/// Register the web_search tool into a registry.
pub fn register_tool(registry: &mut ToolRegistry, api_key: impl Into<String>) {
    registry.register(Box::new(WebSearchTool::new(api_key)));
}

// ---------------------------------------------------------------------------
// Brave Search API types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BraveSearchResponse {
    web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResults {
    results: Vec<BraveWebResult>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResult {
    title: String,
    url: String,
    description: Option<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_definition_schema() {
        let tool = WebSearchTool::new("key");
        let def = tool.definition();
        assert_eq!(def.name, "web_search");
        assert_eq!(def.input_schema["required"][0], "query");
    }

    #[tokio::test]
    async fn missing_query_returns_error() {
        let tool = WebSearchTool::new("key");
        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }
}
