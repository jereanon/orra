use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::message::{Message, Role, ToolCall};
use crate::provider::{
    CompletionRequest, CompletionResponse, FinishReason, Provider, ProviderError, Usage,
};
use crate::tool::ToolDefinition;

const DEFAULT_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

pub struct ClaudeProvider {
    client: Client,
    api_key: String,
    model: String,
    api_url: String,
}

impl ClaudeProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            api_url: DEFAULT_API_URL.to_string(),
        }
    }

    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    fn build_api_request(&self, request: &CompletionRequest) -> ApiRequest {
        let mut system = None;
        let mut messages = Vec::new();

        for msg in &request.messages {
            match msg.role {
                Role::System => {
                    system = Some(msg.content.clone());
                }
                Role::User if !msg.tool_results.is_empty() => {
                    let mut content: Vec<ContentBlock> = msg
                        .tool_results
                        .iter()
                        .map(|tr| ContentBlock::ToolResult {
                            tool_use_id: tr.call_id.clone(),
                            content: tr.content.clone(),
                            is_error: if tr.is_error { Some(true) } else { None },
                        })
                        .collect();

                    if !msg.content.is_empty() {
                        content.push(ContentBlock::Text {
                            text: msg.content.clone(),
                        });
                    }

                    messages.push(ApiMessage {
                        role: "user".to_string(),
                        content: ApiContent::Blocks(content),
                    });
                }
                Role::User => {
                    messages.push(ApiMessage {
                        role: "user".to_string(),
                        content: ApiContent::Text(msg.content.clone()),
                    });
                }
                Role::Assistant if !msg.tool_calls.is_empty() => {
                    let mut content = Vec::new();

                    if !msg.content.is_empty() {
                        content.push(ContentBlock::Text {
                            text: msg.content.clone(),
                        });
                    }

                    for tc in &msg.tool_calls {
                        content.push(ContentBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            input: tc.arguments.clone(),
                        });
                    }

                    messages.push(ApiMessage {
                        role: "assistant".to_string(),
                        content: ApiContent::Blocks(content),
                    });
                }
                Role::Assistant => {
                    messages.push(ApiMessage {
                        role: "assistant".to_string(),
                        content: ApiContent::Text(msg.content.clone()),
                    });
                }
            }
        }

        let tools: Option<Vec<ApiTool>> = if request.tools.is_empty() {
            None
        } else {
            Some(request.tools.iter().map(|t| ApiTool::from(t.clone())).collect())
        };

        ApiRequest {
            model: self.model.clone(),
            max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            system,
            messages,
            tools,
            temperature: request.temperature,
        }
    }

    fn parse_api_response(&self, response: ApiResponse) -> CompletionResponse {
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();

        for block in &response.content {
            match block {
                ContentBlock::Text { text } => {
                    text_parts.push(text.clone());
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: input.clone(),
                    });
                }
                ContentBlock::ToolResult { .. } => {
                    // Should not appear in responses, ignore
                }
            }
        }

        let content = text_parts.join("");

        let message = if tool_calls.is_empty() {
            Message::assistant(content)
        } else {
            Message::assistant_with_tool_calls(content, tool_calls)
        };

        let finish_reason = match response.stop_reason.as_deref() {
            Some("end_turn") => FinishReason::Stop,
            Some("tool_use") => FinishReason::ToolUse,
            Some("max_tokens") => FinishReason::MaxTokens,
            Some(other) => FinishReason::Other(other.to_string()),
            None => FinishReason::Stop,
        };

        CompletionResponse {
            message,
            usage: Usage {
                input_tokens: response.usage.input_tokens,
                output_tokens: response.usage.output_tokens,
            },
            finish_reason,
        }
    }

    fn parse_api_error(status: reqwest::StatusCode, error: ApiError) -> ProviderError {
        match status.as_u16() {
            401 | 403 => ProviderError::Auth(error.error.message),
            429 => ProviderError::RateLimited {
                retry_after_ms: None,
            },
            400 if error.error.message.contains("context")
                || error.error.message.contains("token") =>
            {
                ProviderError::ContextLengthExceeded(error.error.message)
            }
            _ => ProviderError::Other(format!(
                "{}: {}",
                error.error.error_type, error.error.message
            )),
        }
    }
}

#[async_trait]
impl Provider for ClaudeProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let api_request = self.build_api_request(&request);

        let http_response = self
            .client
            .post(&self.api_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&api_request)
            .send()
            .await
            .map_err(|e| ProviderError::Other(format!("request failed: {e}")))?;

        let status = http_response.status();

        if !status.is_success() {
            let error: ApiError = http_response
                .json()
                .await
                .map_err(|e| ProviderError::Other(format!("failed to parse error: {e}")))?;
            return Err(Self::parse_api_error(status, error));
        }

        let api_response: ApiResponse = http_response
            .json()
            .await
            .map_err(|e| ProviderError::Other(format!("failed to parse response: {e}")))?;

        Ok(self.parse_api_response(api_response))
    }
}

// --- Claude API types ---

#[derive(Debug, Serialize)]
struct ApiRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ApiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct ApiMessage {
    role: String,
    content: ApiContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ApiContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Serialize)]
struct ApiTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

impl From<ToolDefinition> for ApiTool {
    fn from(def: ToolDefinition) -> Self {
        Self {
            name: def.name,
            description: def.description,
            input_schema: def.input_schema,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
    usage: ApiUsage,
}

#[derive(Debug, Deserialize)]
struct ApiUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ApiErrorDetail {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ToolResult;

    fn provider() -> ClaudeProvider {
        ClaudeProvider::new("test-key", "claude-sonnet-4-5-20250929")
    }

    #[test]
    fn build_simple_request() {
        let p = provider();
        let request = CompletionRequest {
            messages: vec![Message::user("Hello")],
            tools: vec![],
            max_tokens: Some(1024),
            temperature: Some(0.7),
        };

        let api_req = p.build_api_request(&request);
        assert_eq!(api_req.model, "claude-sonnet-4-5-20250929");
        assert_eq!(api_req.max_tokens, 1024);
        assert_eq!(api_req.temperature, Some(0.7));
        assert!(api_req.system.is_none());
        assert!(api_req.tools.is_none());
        assert_eq!(api_req.messages.len(), 1);
        assert_eq!(api_req.messages[0].role, "user");
    }

    #[test]
    fn build_request_extracts_system_prompt() {
        let p = provider();
        let request = CompletionRequest {
            messages: vec![
                Message::system("You are helpful."),
                Message::user("Hi"),
            ],
            tools: vec![],
            max_tokens: None,
            temperature: None,
        };

        let api_req = p.build_api_request(&request);
        assert_eq!(api_req.system, Some("You are helpful.".into()));
        // System message should not appear in messages array
        assert_eq!(api_req.messages.len(), 1);
        assert_eq!(api_req.messages[0].role, "user");
    }

    #[test]
    fn build_request_default_max_tokens() {
        let p = provider();
        let request = CompletionRequest {
            messages: vec![Message::user("Hi")],
            tools: vec![],
            max_tokens: None,
            temperature: None,
        };

        let api_req = p.build_api_request(&request);
        assert_eq!(api_req.max_tokens, DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn build_request_with_tools() {
        let p = provider();
        let request = CompletionRequest {
            messages: vec![Message::user("Search for rust")],
            tools: vec![ToolDefinition {
                name: "search".into(),
                description: "Search the web".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }),
            }],
            max_tokens: None,
            temperature: None,
        };

        let api_req = p.build_api_request(&request);
        let tools = api_req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search");
    }

    #[test]
    fn build_request_with_tool_results() {
        let p = provider();
        let results = vec![ToolResult {
            call_id: "toolu_123".into(),
            content: "Sunny, 72F".into(),
            is_error: false,
        }];

        let request = CompletionRequest {
            messages: vec![
                Message::user("What's the weather?"),
                Message::assistant_with_tool_calls(
                    "Let me check.",
                    vec![ToolCall {
                        id: "toolu_123".into(),
                        name: "weather".into(),
                        arguments: serde_json::json!({"city": "SF"}),
                    }],
                ),
                Message::tool_result(results),
            ],
            tools: vec![],
            max_tokens: None,
            temperature: None,
        };

        let api_req = p.build_api_request(&request);
        assert_eq!(api_req.messages.len(), 3);

        // Verify assistant message has tool_use block
        let assistant_json = serde_json::to_value(&api_req.messages[1]).unwrap();
        let content = assistant_json["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["id"], "toolu_123");

        // Verify user message has tool_result block
        let user_json = serde_json::to_value(&api_req.messages[2]).unwrap();
        let content = user_json["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "toolu_123");
    }

    #[test]
    fn build_request_tool_result_with_error() {
        let p = provider();
        let results = vec![ToolResult {
            call_id: "toolu_456".into(),
            content: "connection failed".into(),
            is_error: true,
        }];

        let request = CompletionRequest {
            messages: vec![Message::tool_result(results)],
            tools: vec![],
            max_tokens: None,
            temperature: None,
        };

        let api_req = p.build_api_request(&request);
        let msg_json = serde_json::to_value(&api_req.messages[0]).unwrap();
        let block = &msg_json["content"][0];
        assert_eq!(block["is_error"], true);
    }

    #[test]
    fn parse_text_response() {
        let p = provider();
        let api_response = ApiResponse {
            content: vec![ContentBlock::Text {
                text: "Hello!".into(),
            }],
            stop_reason: Some("end_turn".into()),
            usage: ApiUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let response = p.parse_api_response(api_response);
        assert_eq!(response.message.content, "Hello!");
        assert!(response.message.tool_calls.is_empty());
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
    }

    #[test]
    fn parse_tool_use_response() {
        let p = provider();
        let api_response = ApiResponse {
            content: vec![
                ContentBlock::Text {
                    text: "Let me search.".into(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_abc".into(),
                    name: "search".into(),
                    input: serde_json::json!({"query": "rust"}),
                },
            ],
            stop_reason: Some("tool_use".into()),
            usage: ApiUsage {
                input_tokens: 20,
                output_tokens: 15,
            },
        };

        let response = p.parse_api_response(api_response);
        assert_eq!(response.message.content, "Let me search.");
        assert_eq!(response.message.tool_calls.len(), 1);
        assert_eq!(response.message.tool_calls[0].id, "toolu_abc");
        assert_eq!(response.message.tool_calls[0].name, "search");
        assert_eq!(response.finish_reason, FinishReason::ToolUse);
    }

    #[test]
    fn parse_max_tokens_response() {
        let p = provider();
        let api_response = ApiResponse {
            content: vec![ContentBlock::Text {
                text: "Partial...".into(),
            }],
            stop_reason: Some("max_tokens".into()),
            usage: ApiUsage {
                input_tokens: 10,
                output_tokens: 100,
            },
        };

        let response = p.parse_api_response(api_response);
        assert_eq!(response.finish_reason, FinishReason::MaxTokens);
    }

    #[test]
    fn parse_multi_text_blocks() {
        let p = provider();
        let api_response = ApiResponse {
            content: vec![
                ContentBlock::Text {
                    text: "First ".into(),
                },
                ContentBlock::Text {
                    text: "Second".into(),
                },
            ],
            stop_reason: Some("end_turn".into()),
            usage: ApiUsage {
                input_tokens: 5,
                output_tokens: 5,
            },
        };

        let response = p.parse_api_response(api_response);
        assert_eq!(response.message.content, "First Second");
    }

    #[test]
    fn parse_error_auth() {
        let error = ApiError {
            error: ApiErrorDetail {
                error_type: "authentication_error".into(),
                message: "Invalid API key.".into(),
            },
        };
        let result = ClaudeProvider::parse_api_error(
            reqwest::StatusCode::UNAUTHORIZED,
            error,
        );
        assert!(matches!(result, ProviderError::Auth(_)));
    }

    #[test]
    fn parse_error_rate_limit() {
        let error = ApiError {
            error: ApiErrorDetail {
                error_type: "rate_limit_error".into(),
                message: "Rate limited".into(),
            },
        };
        let result = ClaudeProvider::parse_api_error(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            error,
        );
        assert!(matches!(
            result,
            ProviderError::RateLimited { retry_after_ms: None }
        ));
    }

    #[test]
    fn parse_error_context_length() {
        let error = ApiError {
            error: ApiErrorDetail {
                error_type: "invalid_request_error".into(),
                message: "prompt is too long: 300000 tokens > 200000 token limit".into(),
            },
        };
        let result = ClaudeProvider::parse_api_error(
            reqwest::StatusCode::BAD_REQUEST,
            error,
        );
        assert!(matches!(result, ProviderError::ContextLengthExceeded(_)));
    }

    #[test]
    fn parse_error_generic() {
        let error = ApiError {
            error: ApiErrorDetail {
                error_type: "api_error".into(),
                message: "Internal server error".into(),
            },
        };
        let result = ClaudeProvider::parse_api_error(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR,
            error,
        );
        assert!(matches!(result, ProviderError::Other(_)));
    }

    #[test]
    fn api_request_serializes_correctly() {
        let p = provider();
        let request = CompletionRequest {
            messages: vec![
                Message::system("Be concise."),
                Message::user("Hello"),
            ],
            tools: vec![ToolDefinition {
                name: "echo".into(),
                description: "Echoes input".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {"text": {"type": "string"}}}),
            }],
            max_tokens: Some(512),
            temperature: Some(0.5),
        };

        let api_req = p.build_api_request(&request);
        let json = serde_json::to_value(&api_req).unwrap();

        assert_eq!(json["model"], "claude-sonnet-4-5-20250929");
        assert_eq!(json["max_tokens"], 512);
        assert_eq!(json["system"], "Be concise.");
        assert_eq!(json["temperature"], 0.5);
        assert_eq!(json["messages"].as_array().unwrap().len(), 1);
        assert_eq!(json["tools"].as_array().unwrap().len(), 1);
        assert_eq!(json["tools"][0]["name"], "echo");
    }

    #[test]
    fn with_api_url() {
        let p = ClaudeProvider::new("key", "model")
            .with_api_url("https://custom.api.com/v1/messages");
        assert_eq!(p.api_url, "https://custom.api.com/v1/messages");
    }

    // --- Integration test with wiremock ---

    #[tokio::test]
    async fn complete_with_mock_server() {
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let response_body = serde_json::json!({
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello from Claude!"}
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 8
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "test-api-key"))
            .and(header("anthropic-version", ANTHROPIC_VERSION))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        let provider = ClaudeProvider::new("test-api-key", "claude-sonnet-4-5-20250929")
            .with_api_url(format!("{}/v1/messages", mock_server.uri()));

        let request = CompletionRequest {
            messages: vec![Message::user("Hi")],
            tools: vec![],
            max_tokens: Some(1024),
            temperature: None,
        };

        let response = provider.complete(request).await.unwrap();
        assert_eq!(response.message.content, "Hello from Claude!");
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.usage.input_tokens, 12);
        assert_eq!(response.usage.output_tokens, 8);
    }

    #[tokio::test]
    async fn complete_tool_use_roundtrip() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let response_body = serde_json::json!({
            "id": "msg_tool",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll search for that."},
                {
                    "type": "tool_use",
                    "id": "toolu_search_1",
                    "name": "search",
                    "input": {"query": "Rust programming"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 25,
                "output_tokens": 30
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        let provider = ClaudeProvider::new("key", "claude-sonnet-4-5-20250929")
            .with_api_url(format!("{}/v1/messages", mock_server.uri()));

        let request = CompletionRequest {
            messages: vec![Message::user("Search for Rust")],
            tools: vec![ToolDefinition {
                name: "search".into(),
                description: "Search".into(),
                input_schema: serde_json::json!({"type": "object"}),
            }],
            max_tokens: None,
            temperature: None,
        };

        let response = provider.complete(request).await.unwrap();
        assert_eq!(response.finish_reason, FinishReason::ToolUse);
        assert_eq!(response.message.content, "I'll search for that.");
        assert_eq!(response.message.tool_calls.len(), 1);
        assert_eq!(response.message.tool_calls[0].name, "search");
        assert_eq!(response.message.tool_calls[0].id, "toolu_search_1");
        assert_eq!(
            response.message.tool_calls[0].arguments,
            serde_json::json!({"query": "Rust programming"})
        );
    }

    #[tokio::test]
    async fn complete_handles_auth_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let error_body = serde_json::json!({
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid API key."
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(401).set_body_json(&error_body))
            .mount(&mock_server)
            .await;

        let provider = ClaudeProvider::new("bad-key", "claude-sonnet-4-5-20250929")
            .with_api_url(format!("{}/v1/messages", mock_server.uri()));

        let request = CompletionRequest {
            messages: vec![Message::user("Hi")],
            tools: vec![],
            max_tokens: None,
            temperature: None,
        };

        let err = provider.complete(request).await.unwrap_err();
        assert!(matches!(err, ProviderError::Auth(_)));
    }

    #[tokio::test]
    async fn complete_handles_rate_limit() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let error_body = serde_json::json!({
            "type": "error",
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded"
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_json(&error_body))
            .mount(&mock_server)
            .await;

        let provider = ClaudeProvider::new("key", "claude-sonnet-4-5-20250929")
            .with_api_url(format!("{}/v1/messages", mock_server.uri()));

        let request = CompletionRequest {
            messages: vec![Message::user("Hi")],
            tools: vec![],
            max_tokens: None,
            temperature: None,
        };

        let err = provider.complete(request).await.unwrap_err();
        assert!(matches!(err, ProviderError::RateLimited { .. }));
    }

    #[test]
    fn request_body_matches_api_spec() {
        let p = provider();
        let request = CompletionRequest {
            messages: vec![
                Message::system("You help with weather."),
                Message::user("What's the weather in SF?"),
                Message::assistant_with_tool_calls(
                    "I'll check.",
                    vec![ToolCall {
                        id: "toolu_01".into(),
                        name: "get_weather".into(),
                        arguments: serde_json::json!({"location": "San Francisco, CA"}),
                    }],
                ),
                Message::tool_result(vec![ToolResult {
                    call_id: "toolu_01".into(),
                    content: "72F, sunny".into(),
                    is_error: false,
                }]),
            ],
            tools: vec![ToolDefinition {
                name: "get_weather".into(),
                description: "Get the current weather".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }),
            }],
            max_tokens: Some(1024),
            temperature: None,
        };

        let api_req = p.build_api_request(&request);
        let json = serde_json::to_value(&api_req).unwrap();

        // System extracted to top level
        assert_eq!(json["system"], "You help with weather.");

        // 3 messages (no system in messages array)
        let msgs = json["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);

        // User message is plain text
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "What's the weather in SF?");

        // Assistant message has content blocks
        assert_eq!(msgs[1]["role"], "assistant");
        let assistant_content = msgs[1]["content"].as_array().unwrap();
        assert_eq!(assistant_content[0]["type"], "text");
        assert_eq!(assistant_content[1]["type"], "tool_use");
        assert_eq!(assistant_content[1]["name"], "get_weather");

        // Tool result message has content blocks
        assert_eq!(msgs[2]["role"], "user");
        let result_content = msgs[2]["content"].as_array().unwrap();
        assert_eq!(result_content[0]["type"], "tool_result");
        assert_eq!(result_content[0]["tool_use_id"], "toolu_01");
        assert_eq!(result_content[0]["content"], "72F, sunny");
        // is_error should not be present when false
        assert!(result_content[0].get("is_error").is_none());
    }
}
