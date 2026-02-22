//! Browser/web reading tool.
//!
//! Fetches web pages and extracts readable content, stripping navigation,
//! ads, and other non-content elements. Uses a simplified readability
//! algorithm to pull out the main article text.

use async_trait::async_trait;

use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

// ---------------------------------------------------------------------------
// Content extraction
// ---------------------------------------------------------------------------

/// Extracted content from a web page.
#[derive(Debug, Clone)]
pub struct PageContent {
    /// The page title, if found.
    pub title: Option<String>,

    /// The main text content of the page.
    pub text: String,

    /// The final URL (after redirects).
    pub url: String,
}

/// Strip HTML tags and extract readable text content from raw HTML.
/// This is a simplified readability extraction that:
/// 1. Removes script and style blocks
/// 2. Strips all HTML tags
/// 3. Decodes common HTML entities
/// 4. Collapses excess whitespace
pub fn extract_text(html: &str) -> String {
    let mut result = html.to_string();

    // Remove script blocks
    while let Some(start) = result.to_lowercase().find("<script") {
        if let Some(end) = result.to_lowercase()[start..].find("</script>") {
            result = format!(
                "{}{}",
                &result[..start],
                &result[start + end + "</script>".len()..]
            );
        } else {
            break;
        }
    }

    // Remove style blocks
    while let Some(start) = result.to_lowercase().find("<style") {
        if let Some(end) = result.to_lowercase()[start..].find("</style>") {
            result = format!(
                "{}{}",
                &result[..start],
                &result[start + end + "</style>".len()..]
            );
        } else {
            break;
        }
    }

    // Remove HTML comments
    while let Some(start) = result.find("<!--") {
        if let Some(end) = result[start..].find("-->") {
            result = format!("{}{}", &result[..start], &result[start + end + 3..]);
        } else {
            break;
        }
    }

    // Add line breaks for block elements
    let block_tags = [
        "<br", "<p", "</p>", "<div", "</div>", "<h1", "<h2", "<h3", "<h4", "<h5", "<h6", "</h1>",
        "</h2>", "</h3>", "</h4>", "</h5>", "</h6>", "<li", "</li>", "<tr", "</tr>",
    ];
    for tag in &block_tags {
        result = result.replace(tag, &format!("\n{tag}"));
    }

    // Strip all remaining tags
    let mut cleaned = String::with_capacity(result.len());
    let mut in_tag = false;
    for ch in result.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => cleaned.push(ch),
            _ => {}
        }
    }

    // Decode common HTML entities
    cleaned = cleaned
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
        .replace("&#x27;", "'")
        .replace("&#x2F;", "/");

    // Collapse whitespace: multiple spaces → single space, multiple newlines → double newline
    let mut final_text = String::with_capacity(cleaned.len());
    let mut prev_was_newline = false;
    let mut newline_count = 0;

    for line in cleaned.lines() {
        let trimmed = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if trimmed.is_empty() {
            newline_count += 1;
            if newline_count <= 2 {
                prev_was_newline = true;
            }
        } else {
            if prev_was_newline && !final_text.is_empty() {
                final_text.push('\n');
            }
            if !final_text.is_empty() {
                final_text.push('\n');
            }
            final_text.push_str(&trimmed);
            prev_was_newline = false;
            newline_count = 0;
        }
    }

    final_text.trim().to_string()
}

/// Try to extract the page title from HTML.
pub fn extract_title(html: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let start = lower.find("<title")?;
    let tag_end = lower[start..].find('>')?;
    let content_start = start + tag_end + 1;
    let end = lower[content_start..].find("</title>")?;

    let title = html[content_start..content_start + end].trim().to_string();
    if title.is_empty() {
        None
    } else {
        Some(title)
    }
}

// ---------------------------------------------------------------------------
// Browser tool
// ---------------------------------------------------------------------------

/// Tool that fetches a URL and returns the readable text content.
pub struct BrowserTool {
    client: reqwest::Client,
    max_content_length: usize,
}

impl BrowserTool {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (compatible; AgenticBot/1.0)")
            .timeout(std::time::Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::limited(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        Self {
            client,
            max_content_length: 100_000,
        }
    }

    pub fn with_max_content_length(mut self, max: usize) -> Self {
        self.max_content_length = max;
        self
    }

    /// Fetch a URL and extract readable text.
    pub async fn fetch(&self, url: &str) -> Result<PageContent, BrowserError> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| BrowserError::Request(e.to_string()))?;

        let final_url = response.url().to_string();
        let status = response.status();

        if !status.is_success() {
            return Err(BrowserError::HttpStatus {
                status: status.as_u16(),
                url: final_url,
            });
        }

        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        if !content_type.contains("text/html") && !content_type.contains("text/plain") {
            return Err(BrowserError::UnsupportedContentType(content_type));
        }

        let body = response
            .text()
            .await
            .map_err(|e| BrowserError::Request(e.to_string()))?;

        // Truncate if too long
        let body = if body.len() > self.max_content_length {
            body[..self.max_content_length].to_string()
        } else {
            body
        };

        let title = extract_title(&body);
        let text = extract_text(&body);

        Ok(PageContent {
            title,
            text,
            url: final_url,
        })
    }
}

impl Default for BrowserTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for BrowserTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_url".into(),
            description: "Fetch a web page and extract its readable text content. Strips \
                          navigation, scripts, and other non-content elements to return \
                          just the main article text."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch and read"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let url = input
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'url'".into()))?;

        let content = self
            .fetch(url)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let mut output = String::new();
        if let Some(title) = &content.title {
            output.push_str(&format!("Title: {title}\n"));
        }
        output.push_str(&format!("URL: {}\n\n", content.url));
        output.push_str(&content.text);

        // Truncate the final output to a reasonable size for the model
        if output.len() > 50_000 {
            output.truncate(50_000);
            output.push_str("\n\n[Content truncated]");
        }

        Ok(output)
    }
}

/// Register the browser tool.
pub fn register_tool(registry: &mut ToolRegistry) {
    registry.register(Box::new(BrowserTool::new()));
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum BrowserError {
    #[error("request failed: {0}")]
    Request(String),

    #[error("HTTP {status} for {url}")]
    HttpStatus { status: u16, url: String },

    #[error("unsupported content type: {0}")]
    UnsupportedContentType(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_text_basic_html() {
        let html = "<html><body><h1>Hello</h1><p>This is a test.</p></body></html>";
        let text = extract_text(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("This is a test."));
    }

    #[test]
    fn extract_text_removes_scripts() {
        let html = r#"<p>Before</p><script>alert('xss');</script><p>After</p>"#;
        let text = extract_text(html);
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
        assert!(!text.contains("alert"));
    }

    #[test]
    fn extract_text_removes_styles() {
        let html = r#"<p>Content</p><style>.foo { color: red; }</style><p>More</p>"#;
        let text = extract_text(html);
        assert!(text.contains("Content"));
        assert!(!text.contains("color"));
    }

    #[test]
    fn extract_text_decodes_entities() {
        let html = "<p>A &amp; B &lt; C &gt; D &quot;E&quot;</p>";
        let text = extract_text(html);
        assert!(text.contains("A & B < C > D \"E\""));
    }

    #[test]
    fn extract_text_collapses_whitespace() {
        let html = "<p>  Hello    world  </p>";
        let text = extract_text(html);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn extract_text_removes_comments() {
        let html = "<p>Visible</p><!-- hidden comment --><p>Also visible</p>";
        let text = extract_text(html);
        assert!(text.contains("Visible"));
        assert!(text.contains("Also visible"));
        assert!(!text.contains("hidden comment"));
    }

    #[test]
    fn extract_title_basic() {
        let html = "<html><head><title>My Page</title></head><body></body></html>";
        assert_eq!(extract_title(html), Some("My Page".into()));
    }

    #[test]
    fn extract_title_missing() {
        let html = "<html><body>No title here</body></html>";
        assert_eq!(extract_title(html), None);
    }

    #[test]
    fn extract_title_empty() {
        let html = "<title></title>";
        assert_eq!(extract_title(html), None);
    }

    #[test]
    fn tool_definition_valid() {
        let tool = BrowserTool::new();
        let def = tool.definition();
        assert_eq!(def.name, "read_url");
        assert!(def.input_schema["required"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("url")));
    }

    #[test]
    fn browser_error_display() {
        let err = BrowserError::HttpStatus {
            status: 404,
            url: "https://example.com".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("404"));
        assert!(msg.contains("example.com"));
    }

    #[test]
    fn extract_text_complex_page() {
        let html = r#"
            <!DOCTYPE html>
            <html>
            <head>
                <title>Test Article</title>
                <style>body { font-family: sans-serif; }</style>
                <script>console.log('analytics');</script>
            </head>
            <body>
                <nav>
                    <a href="/">Home</a>
                    <a href="/about">About</a>
                </nav>
                <article>
                    <h1>Breaking News</h1>
                    <p>This is the first paragraph of the article.</p>
                    <p>This is the second paragraph with <strong>bold</strong> text.</p>
                </article>
                <footer>Copyright 2025</footer>
                <script>trackPageView();</script>
            </body>
            </html>
        "#;

        let text = extract_text(html);
        assert!(text.contains("Breaking News"));
        assert!(text.contains("first paragraph"));
        assert!(text.contains("bold"));
        assert!(!text.contains("console.log"));
        assert!(!text.contains("trackPageView"));
        assert!(!text.contains("font-family"));
    }
}
