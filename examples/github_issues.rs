use std::io::{self, Write};
use std::sync::Arc;

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use claw_lib::context::CharEstimator;
use claw_lib::message::Message;
use claw_lib::namespace::Namespace;
use claw_lib::policy::PolicyRegistry;
use claw_lib::providers::claude::ClaudeProvider;
use claw_lib::runtime::{Runtime, RuntimeConfig};
use claw_lib::store::InMemoryStore;
use claw_lib::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

// ---------------------------------------------------------------------------
// Shared GitHub client config
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct GitHubConfig {
    client: Client,
    token: String,
    owner: String,
    repo: String,
}

impl GitHubConfig {
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
            .header("User-Agent", "claw-lib-example")
            .header("X-GitHub-Api-Version", "2022-11-28")
    }
}

// ---------------------------------------------------------------------------
// GitHub API response types (just what we need)
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

struct ListIssuesTool {
    gh: GitHubConfig,
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
        let state = input.get("state").and_then(|v| v.as_str()).unwrap_or("open");
        let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(10);

        let mut req = self
            .gh
            .request(reqwest::Method::GET, "issues")
            .query(&[("state", state), ("per_page", &limit.to_string())]);

        if let Some(labels) = input.get("labels").and_then(|v| v.as_str()) {
            req = req.query(&[("labels", labels)]);
        }

        let resp = req.send().await.map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!("GitHub API {}: {}", status, body)));
        }

        let issues: Vec<GhIssue> = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {}", e)))?;

        if issues.is_empty() {
            return Ok(format!("No {} issues found.", state));
        }

        let mut lines = Vec::new();
        for issue in &issues {
            let labels: Vec<&str> = issue.labels.iter().map(|l| l.name.as_str()).collect();
            let assignees: Vec<&str> = issue.assignees.iter().map(|a| a.login.as_str()).collect();

            let mut parts = vec![format!("#{} [{}] {}", issue.number, issue.state, issue.title)];
            if !labels.is_empty() {
                parts.push(format!("  labels: {}", labels.join(", ")));
            }
            if !assignees.is_empty() {
                parts.push(format!("  assigned: {}", assignees.join(", ")));
            }
            parts.push(format!("  by @{}, {} comments", issue.user.login, issue.comments));
            lines.push(parts.join("\n"));
        }

        Ok(lines.join("\n\n"))
    }
}

struct GetIssueTool {
    gh: GitHubConfig,
}

#[async_trait]
impl Tool for GetIssueTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "get_issue".into(),
            description: "Get details of a specific issue by number, including its body and recent comments.".into(),
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
            .request(reqwest::Method::GET, &format!("issues/{}", number))
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!("GitHub API {}: {}", status, body)));
        }

        let issue: GhIssue = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {}", e)))?;

        let labels: Vec<&str> = issue.labels.iter().map(|l| l.name.as_str()).collect();
        let assignees: Vec<&str> = issue.assignees.iter().map(|a| a.login.as_str()).collect();
        let labels_str = if labels.is_empty() { "none".to_string() } else { labels.join(", ") };
        let assignees_str = if assignees.is_empty() { "none".to_string() } else { assignees.join(", ") };

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
                .request(reqwest::Method::GET, &format!("issues/{}/comments", number))
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

struct CreateIssueTool {
    gh: GitHubConfig,
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
            return Err(ToolError::ExecutionFailed(format!("GitHub API {}: {}", status, body)));
        }

        let issue: GhIssue = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {}", e)))?;

        Ok(format!("Created issue #{}: {}", issue.number, issue.title))
    }
}

struct AddCommentTool {
    gh: GitHubConfig,
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
            .request(reqwest::Method::POST, &format!("issues/{}/comments", number))
            .json(&serde_json::json!({ "body": body }))
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let resp_body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!(
                "GitHub API {}: {}",
                status, resp_body
            )));
        }

        Ok(format!("Added comment to issue #{}.", number))
    }
}

struct CloseIssueTool {
    gh: GitHubConfig,
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
                .request(reqwest::Method::POST, &format!("issues/{}/comments", number))
                .json(&serde_json::json!({ "body": comment }))
                .send()
                .await;
        }

        let resp = self
            .gh
            .request(reqwest::Method::PATCH, &format!("issues/{}", number))
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
            return Err(ToolError::ExecutionFailed(format!("GitHub API {}: {}", status, body)));
        }

        Ok(format!("Closed issue #{} (reason: {}).", number, reason))
    }
}

struct SearchIssuesTool {
    gh: GitHubConfig,
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
            q.push_str(&format!(" is:{}", state));
        }

        let resp = self
            .gh
            .client
            .get("https://api.github.com/search/issues")
            .header("Authorization", format!("Bearer {}", self.gh.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "claw-lib-example")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .query(&[("q", &q), ("per_page", &"10".to_string())])
            .send()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ToolError::ExecutionFailed(format!("GitHub API {}: {}", status, body)));
        }

        #[derive(Deserialize)]
        struct SearchResult {
            total_count: u64,
            items: Vec<GhIssue>,
        }

        let result: SearchResult = resp
            .json()
            .await
            .map_err(|e| ToolError::ExecutionFailed(format!("parse error: {}", e)))?;

        if result.items.is_empty() {
            return Ok(format!("No issues found for '{}'.", query));
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
// CLI
// ---------------------------------------------------------------------------

fn build_tools(gh: &GitHubConfig) -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ListIssuesTool { gh: gh.clone() }));
    tools.register(Box::new(GetIssueTool { gh: gh.clone() }));
    tools.register(Box::new(CreateIssueTool { gh: gh.clone() }));
    tools.register(Box::new(AddCommentTool { gh: gh.clone() }));
    tools.register(Box::new(CloseIssueTool { gh: gh.clone() }));
    tools.register(Box::new(SearchIssuesTool { gh: gh.clone() }));
    tools
}

fn system_prompt(owner: &str, repo: &str) -> String {
    format!(
        "You are a GitHub issues assistant for the {owner}/{repo} repository. \
         You help users browse, search, create, comment on, and close issues. \
         Always use the provided tools to interact with GitHub — never guess about issue contents. \
         Be concise. When listing issues, summarize them clearly. \
         When creating issues, write clear titles and well-formatted markdown descriptions."
    )
}

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: ANTHROPIC_API_KEY not set.");
        std::process::exit(1);
    });

    let gh_token = std::env::var("GITHUB_TOKEN").unwrap_or_else(|_| {
        eprintln!("Error: GITHUB_TOKEN not set.");
        eprintln!("Create one at https://github.com/settings/tokens with 'repo' scope.");
        std::process::exit(1);
    });

    // Parse owner/repo from args or env
    let args: Vec<String> = std::env::args().collect();
    let repo_arg = args.get(1).cloned().or_else(|| std::env::var("GITHUB_REPO").ok());

    let (owner, repo) = match repo_arg {
        Some(ref r) if r.contains('/') => {
            let parts: Vec<&str> = r.splitn(2, '/').collect();
            (parts[0].to_string(), parts[1].to_string())
        }
        _ => {
            eprintln!("Usage: github_issues <owner/repo>");
            eprintln!("  e.g. github_issues rust-lang/rust");
            eprintln!("  or set GITHUB_REPO=owner/repo");
            std::process::exit(1);
        }
    };

    let model = std::env::var("CLAW_MODEL").unwrap_or_else(|_| "claude-sonnet-4-5-20250929".into());

    let gh = GitHubConfig {
        client: Client::new(),
        token: gh_token,
        owner: owner.clone(),
        repo: repo.clone(),
    };

    let provider = Arc::new(ClaudeProvider::new(&api_key, &model));
    let store = Arc::new(InMemoryStore::new());
    let tools = build_tools(&gh);

    let config = RuntimeConfig {
        system_prompt: Some(system_prompt(&owner, &repo)),
        max_turns: 5,
        max_tokens: Some(2048),
        temperature: Some(0.3),
        ..RuntimeConfig::default()
    };

    let runtime = Runtime::new(
        provider,
        store,
        tools,
        PolicyRegistry::default(),
        CharEstimator::default(),
        config,
    );

    let ns = Namespace::new("github").child(&format!("{}/{}", owner, repo));

    println!("=== GitHub Issues Assistant ===");
    println!("Repository: {}/{}", owner, repo);
    println!("Type your request, or /quit to exit.");
    println!();

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if stdin.read_line(&mut input).unwrap() == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/quit" || input == "/exit" {
            println!("Goodbye!");
            break;
        }

        match runtime.run(&ns, Message::user(input)).await {
            Ok(result) => {
                for turn in &result.turns {
                    for tc in &turn.response.message.tool_calls {
                        println!(
                            "  [{}({})]",
                            tc.name,
                            serde_json::to_string(&tc.arguments).unwrap_or_default()
                        );
                    }
                    for tr in &turn.tool_results {
                        if tr.is_error {
                            println!("  [tool error: {}]", tr.content);
                        }
                    }
                }
                println!();
                println!("{}", result.final_message.content);
                println!(
                    "  ({} input, {} output tokens)",
                    result.total_usage.input_tokens, result.total_usage.output_tokens
                );
                println!();
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                println!();
            }
        }
    }
}
