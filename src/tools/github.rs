use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

// ---------------------------------------------------------------------------
// Shared GitHub client config
// ---------------------------------------------------------------------------

/// Configuration for GitHub API access. Shared across all GitHub issue tools.
#[derive(Clone)]
pub struct GitHubConfig {
    client: Client,
    token: String,
    owner: String,
    repo: String,
}

impl GitHubConfig {
    pub fn new(
        token: impl Into<String>,
        owner: impl Into<String>,
        repo: impl Into<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            token: token.into(),
            owner: owner.into(),
            repo: repo.into(),
        }
    }

    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    pub fn owner(&self) -> &str {
        &self.owner
    }

    pub fn repo(&self) -> &str {
        &self.repo
    }

    fn api_url(&self, path: &str) -> String {
        format!(
            "https://api.github.com/repos/{}/{}/{}",
            self.owner, self.repo, path
        )
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        self.client
            .request(method, self.api_url(path))
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "orra")
            .header("X-GitHub-Api-Version", "2022-11-28")
    }
}

// ---------------------------------------------------------------------------
// GitHub API response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct GhIssue {
    number: u64,
    title: String,
    state: String,
    body: Option<String>,
    user: GhUser,
    labels: Vec<GhLabel>,
    assignees: Vec<GhUser>,
    comments: u64,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct GhUser {
    login: String,
}

#[derive(Debug, Deserialize)]
struct GhLabel {
    name: String,
}

#[derive(Debug, Deserialize)]
struct GhComment {
    user: GhUser,
    body: String,
    created_at: String,
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

pub struct ListIssuesTool {
    gh: GitHubConfig,
}

impl ListIssuesTool {
    pub fn new(gh: GitHubConfig) -> Self {
        Self { gh }
    }
}

#[async_trait]
impl Tool for ListIssuesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_issues".into(),
            description: "List issues in the repository. Returns issue number, title, state, labels, and assignees.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "Filter by state. Defaults to 'open'."
                    },
                    "labels": {
                        "type": "string",
                        "description": "Comma-separated list of label names to filter by"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of issues to return. Defaults to 10."
                    }
                }
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let state = input
            .get("state")
            .and_then(|v| v.as_str())
            .unwrap_or("open");
        let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(10);

        let mut req = self
            .gh
            .request(reqwest::Method::GET, "issues")
            .query(&[("state", state), ("per_page", &limit.to_string())]);

        if let Some(labels) = input.get("labels").and_then(|v| v.as_str()) {
            req = req.query(&[("labels", labels)]);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "GitHub API {status}: {body}"
            )));
        }

        let issues: Vec<GhIssue> = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {e}")))?;

        if issues.is_empty() {
            return Ok(format!("No {state} issues found."));
        }

        let mut lines = Vec::new();
        for issue in &issues {
            let labels: Vec<&str> = issue.labels.iter().map(|l| l.name.as_str()).collect();
            let assignees: Vec<&str> = issue.assignees.iter().map(|a| a.login.as_str()).collect();

            let mut parts = vec![format!(
                "#{} [{}] {}",
                issue.number, issue.state, issue.title
            )];
            if !labels.is_empty() {
                parts.push(format!("  labels: {}", labels.join(", ")));
            }
            if !assignees.is_empty() {
                parts.push(format!("  assigned: {}", assignees.join(", ")));
            }
            parts.push(format!(
                "  by @{}, {} comments",
                issue.user.login, issue.comments
            ));
            lines.push(parts.join("\n"));
        }

        Ok(lines.join("\n\n"))
    }
}

pub struct GetIssueTool {
    gh: GitHubConfig,
}

impl GetIssueTool {
    pub fn new(gh: GitHubConfig) -> Self {
        Self { gh }
    }
}

#[async_trait]
impl Tool for GetIssueTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_issue".into(),
            description:
                "Get details of a specific issue by number, including its body and recent comments."
                    .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The issue number"
                    }
                },
                "required": ["number"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let number = input
            .get("number")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidInput("missing 'number'".into()))?;

        let resp = self
            .gh
            .request(reqwest::Method::GET, &format!("issues/{number}"))
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "GitHub API {status}: {body}"
            )));
        }

        let issue: GhIssue = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {e}")))?;

        let labels: Vec<&str> = issue.labels.iter().map(|l| l.name.as_str()).collect();
        let assignees: Vec<&str> = issue.assignees.iter().map(|a| a.login.as_str()).collect();
        let labels_str = if labels.is_empty() {
            "none".to_string()
        } else {
            labels.join(", ")
        };
        let assignees_str = if assignees.is_empty() {
            "none".to_string()
        } else {
            assignees.join(", ")
        };

        let mut text = format!(
            "#{} [{}] {}\nby @{} on {}\nlabels: {}\nassigned: {}\n\n{}",
            issue.number,
            issue.state,
            issue.title,
            issue.user.login,
            issue.created_at,
            labels_str,
            assignees_str,
            issue.body.as_deref().unwrap_or("(no description)"),
        );

        // Fetch recent comments
        if issue.comments > 0 {
            let comments_resp = self
                .gh
                .request(reqwest::Method::GET, &format!("issues/{number}/comments"))
                .query(&[("per_page", "5")])
                .send()
                .await;

            if let Ok(resp) = comments_resp {
                if let Ok(comments) = resp.json::<Vec<GhComment>>().await {
                    text.push_str("\n\n--- Recent comments ---");
                    for c in &comments {
                        text.push_str(&format!(
                            "\n\n@{} ({})\n{}",
                            c.user.login, c.created_at, c.body
                        ));
                    }
                }
            }
        }

        Ok(text)
    }
}

pub struct CreateIssueTool {
    gh: GitHubConfig,
}

impl CreateIssueTool {
    pub fn new(gh: GitHubConfig) -> Self {
        Self { gh }
    }
}

#[async_trait]
impl Tool for CreateIssueTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "create_issue".into(),
            description: "Create a new issue in the repository.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Issue title"
                    },
                    "body": {
                        "type": "string",
                        "description": "Issue body/description (markdown)"
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels to add"
                    },
                    "assignees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "GitHub usernames to assign"
                    }
                },
                "required": ["title"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let title = input
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'title'".into()))?;

        let mut body_json = serde_json::json!({ "title": title });

        if let Some(body) = input.get("body").and_then(|v| v.as_str()) {
            body_json["body"] = serde_json::Value::String(body.to_string());
        }
        if let Some(labels) = input.get("labels") {
            body_json["labels"] = labels.clone();
        }
        if let Some(assignees) = input.get("assignees") {
            body_json["assignees"] = assignees.clone();
        }

        let resp = self
            .gh
            .request(reqwest::Method::POST, "issues")
            .json(&body_json)
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "GitHub API {status}: {body}"
            )));
        }

        let issue: GhIssue = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {e}")))?;

        Ok(format!("Created issue #{}: {}", issue.number, issue.title))
    }
}

pub struct AddCommentTool {
    gh: GitHubConfig,
}

impl AddCommentTool {
    pub fn new(gh: GitHubConfig) -> Self {
        Self { gh }
    }
}

#[async_trait]
impl Tool for AddCommentTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "add_comment".into(),
            description: "Add a comment to an existing issue.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The issue number to comment on"
                    },
                    "body": {
                        "type": "string",
                        "description": "Comment text (markdown)"
                    }
                },
                "required": ["number", "body"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let number = input
            .get("number")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidInput("missing 'number'".into()))?;

        let body = input
            .get("body")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'body'".into()))?;

        let resp = self
            .gh
            .request(reqwest::Method::POST, &format!("issues/{number}/comments"))
            .json(&serde_json::json!({ "body": body }))
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let resp_body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "GitHub API {status}: {resp_body}"
            )));
        }

        Ok(format!("Added comment to issue #{number}."))
    }
}

pub struct CloseIssueTool {
    gh: GitHubConfig,
}

impl CloseIssueTool {
    pub fn new(gh: GitHubConfig) -> Self {
        Self { gh }
    }
}

#[async_trait]
impl Tool for CloseIssueTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "close_issue".into(),
            description: "Close an issue. Optionally add a closing comment.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The issue number to close"
                    },
                    "comment": {
                        "type": "string",
                        "description": "Optional closing comment"
                    },
                    "reason": {
                        "type": "string",
                        "enum": ["completed", "not_planned"],
                        "description": "Close reason. Defaults to 'completed'."
                    }
                },
                "required": ["number"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let number = input
            .get("number")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| ToolError::InvalidInput("missing 'number'".into()))?;

        let reason = input
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("completed");

        // Add comment first if provided
        if let Some(comment) = input.get("comment").and_then(|v| v.as_str()) {
            let _ = self
                .gh
                .request(reqwest::Method::POST, &format!("issues/{number}/comments"))
                .json(&serde_json::json!({ "body": comment }))
                .send()
                .await;
        }

        let resp = self
            .gh
            .request(reqwest::Method::PATCH, &format!("issues/{number}"))
            .json(&serde_json::json!({
                "state": "closed",
                "state_reason": reason
            }))
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "GitHub API {status}: {body}"
            )));
        }

        Ok(format!("Closed issue #{number} (reason: {reason})."))
    }
}

pub struct SearchIssuesTool {
    gh: GitHubConfig,
}

impl SearchIssuesTool {
    pub fn new(gh: GitHubConfig) -> Self {
        Self { gh }
    }
}

#[async_trait]
impl Tool for SearchIssuesTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_issues".into(),
            description: "Search issues by keyword. Searches in title and body.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed"],
                        "description": "Filter by state. Omit to search all."
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

        let mut q = format!("{} repo:{}/{} is:issue", query, self.gh.owner, self.gh.repo);
        if let Some(state) = input.get("state").and_then(|v| v.as_str()) {
            q.push_str(&format!(" is:{state}"));
        }

        let resp = self
            .gh
            .client
            .get("https://api.github.com/search/issues")
            .header("Authorization", format!("Bearer {}", self.gh.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "orra")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .query(&[("q", &q), ("per_page", &"10".to_string())])
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "GitHub API {status}: {body}"
            )));
        }

        #[derive(Deserialize)]
        struct SearchResult {
            total_count: u64,
            items: Vec<GhIssue>,
        }

        let result: SearchResult = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {e}")))?;

        if result.items.is_empty() {
            return Ok(format!("No issues found for '{query}'."));
        }

        let mut lines = vec![format!("{} results:", result.total_count)];
        for issue in &result.items {
            let labels: Vec<&str> = issue.labels.iter().map(|l| l.name.as_str()).collect();
            let label_str = if labels.is_empty() {
                String::new()
            } else {
                format!(" [{}]", labels.join(", "))
            };
            lines.push(format!(
                "  #{} [{}] {}{}",
                issue.number, issue.state, issue.title, label_str
            ));
        }

        Ok(lines.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Convenience registration
// ---------------------------------------------------------------------------

/// Register all GitHub issue tools into a ToolRegistry.
pub fn register_tools(registry: &mut ToolRegistry, config: &GitHubConfig) {
    registry.register(Box::new(ListIssuesTool::new(config.clone())));
    registry.register(Box::new(GetIssueTool::new(config.clone())));
    registry.register(Box::new(CreateIssueTool::new(config.clone())));
    registry.register(Box::new(AddCommentTool::new(config.clone())));
    registry.register(Box::new(CloseIssueTool::new(config.clone())));
    registry.register(Box::new(SearchIssuesTool::new(config.clone())));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn github_config_api_url() {
        let config = GitHubConfig::new("tok", "rust-lang", "rust");
        assert_eq!(
            config.api_url("issues"),
            "https://api.github.com/repos/rust-lang/rust/issues"
        );
        assert_eq!(
            config.api_url("issues/42/comments"),
            "https://api.github.com/repos/rust-lang/rust/issues/42/comments"
        );
    }

    #[test]
    fn github_config_accessors() {
        let config = GitHubConfig::new("tok", "rust-lang", "rust");
        assert_eq!(config.owner(), "rust-lang");
        assert_eq!(config.repo(), "rust");
    }

    #[test]
    fn register_tools_adds_all_six() {
        let config = GitHubConfig::new("tok", "owner", "repo");
        let mut registry = ToolRegistry::new();
        register_tools(&mut registry, &config);

        assert_eq!(registry.len(), 6);
        assert!(registry.get("list_issues").is_some());
        assert!(registry.get("get_issue").is_some());
        assert!(registry.get("create_issue").is_some());
        assert!(registry.get("add_comment").is_some());
        assert!(registry.get("close_issue").is_some());
        assert!(registry.get("search_issues").is_some());
    }

    #[test]
    fn tool_definitions_have_schemas() {
        let config = GitHubConfig::new("tok", "owner", "repo");
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(ListIssuesTool::new(config.clone())),
            Box::new(GetIssueTool::new(config.clone())),
            Box::new(CreateIssueTool::new(config.clone())),
            Box::new(AddCommentTool::new(config.clone())),
            Box::new(CloseIssueTool::new(config.clone())),
            Box::new(SearchIssuesTool::new(config)),
        ];

        for tool in &tools {
            let def = tool.definition();
            assert!(!def.name.is_empty());
            assert!(!def.description.is_empty());
            assert_eq!(def.input_schema["type"], "object");
        }
    }
}
