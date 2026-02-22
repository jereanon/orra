use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::message::{Message, Role, ToolCall};
use crate::provider::{
    CompletionRequest, CompletionResponse, FinishReason, Provider, ProviderError, StreamEvent,
    StreamingProvider, Usage,
};
use crate::tool::ToolDefinition;

const DEFAULT_API_URL: &str = "https://api.openai.com/v1/chat/completions";
const DEFAULT_MAX_TOKENS: u32 = 4096;

pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    model: String,
    api_url: String,
}

impl OpenAIProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            api_key: api_key.into(),
            model: model.into(),
            api_url: DEFAULT_API_URL.to_string(),
        }
    }

    /// Point at a different endpoint (Azure OpenAI, local vLLM, ollama, etc.)
    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    fn build_api_request(&self, request: &CompletionRequest) -> ChatRequest {
        let mut messages = Vec::new();

        for msg in &request.messages {
            match msg.role {
                Role::System => {
                    messages.push(ChatMessage {
                        role: "system".into(),
                        content: Some(msg.content.clone()),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                Role::User if !msg.tool_results.is_empty() => {
                    // Tool results go as individual "tool" role messages
                    for tr in &msg.tool_results {
                        messages.push(ChatMessage {
                            role: "tool".into(),
                            content: Some(tr.content.clone()),
                            tool_calls: None,
                            tool_call_id: Some(tr.call_id.clone()),
                        });
                    }
                }
                Role::User => {
                    messages.push(ChatMessage {
                        role: "user".into(),
                        content: Some(msg.content.clone()),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                Role::Assistant if !msg.tool_calls.is_empty() => {
                    let tool_calls: Vec<ChatToolCall> = msg
                        .tool_calls
                        .iter()
                        .map(|tc| ChatToolCall {
                            id: tc.id.clone(),
                            r#type: "function".into(),
                            function: ChatFunctionCall {
                                name: tc.name.clone(),
                                arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
                            },
                        })
                        .collect();

                    messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: if msg.content.is_empty() {
                            None
                        } else {
                            Some(msg.content.clone())
                        },
                        tool_calls: Some(tool_calls),
                        tool_call_id: None,
                    });
                }
                Role::Assistant => {
                    messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: Some(msg.content.clone()),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
        }

        let tools: Option<Vec<ChatTool>> = if request.tools.is_empty() {
            None
        } else {
            Some(request.tools.iter().map(ChatTool::from).collect())
        };

        ChatRequest {
            model: request.model.clone().unwrap_or_else(|| self.model.clone()),
            messages,
            tools,
            max_tokens: Some(request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS)),
            temperature: request.temperature,
            stream: None,
        }
    }

    fn parse_response(&self, response: ChatResponse) -> CompletionResponse {
        let choice = response.choices.into_iter().next().unwrap_or_default();

        let mut tool_calls = Vec::new();
        if let Some(tcs) = choice.message.tool_calls {
            for tc in tcs {
                let arguments: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));

                tool_calls.push(ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments,
                });
            }
        }

        let content = choice.message.content.unwrap_or_default();
        let message = if tool_calls.is_empty() {
            Message::assistant(content)
        } else {
            Message::assistant_with_tool_calls(content, tool_calls)
        };

        let finish_reason = match choice.finish_reason.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("tool_calls") => FinishReason::ToolUse,
            Some("length") => FinishReason::MaxTokens,
            Some(other) => FinishReason::Other(other.to_string()),
            None => FinishReason::Stop,
        };

        let usage = response.usage.unwrap_or_default();

        CompletionResponse {
            message,
            usage: Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
            },
            finish_reason,
        }
    }

    fn parse_error(status: reqwest::StatusCode, body: &str) -> ProviderError {
        let error_msg = serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| body.to_string());

        match status.as_u16() {
            401 | 403 => ProviderError::Auth(error_msg),
            429 => ProviderError::RateLimited {
                retry_after_ms: None,
            },
            400 if error_msg.contains("context") || error_msg.contains("token") => {
                ProviderError::ContextLengthExceeded(error_msg)
            }
            _ => ProviderError::Other(format!("HTTP {status}: {error_msg}")),
        }
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        let api_request = self.build_api_request(&request);

        let http_response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&api_request)
            .send()
            .await
            .map_err(|e| ProviderError::Other(format!("request failed: {e}")))?;

        let status = http_response.status();

        if !status.is_success() {
            let body = http_response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".into());
            return Err(Self::parse_error(status, &body));
        }

        let api_response: ChatResponse = http_response
            .json()
            .await
            .map_err(|e| ProviderError::Other(format!("failed to parse response: {e}")))?;

        Ok(self.parse_response(api_response))
    }
}

#[async_trait]
impl StreamingProvider for OpenAIProvider {
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>, ProviderError> {
        use tokio_stream::StreamExt;

        let mut api_request = self.build_api_request(&request);
        api_request.stream = Some(true);

        let http_response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&api_request)
            .send()
            .await
            .map_err(|e| ProviderError::Other(format!("request failed: {e}")))?;

        let status = http_response.status();
        if !status.is_success() {
            let body = http_response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".into());
            return Err(Self::parse_error(status, &body));
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);

        let byte_stream = http_response.bytes_stream();
        tokio::spawn(async move {
            let mut stream = byte_stream;
            let mut buffer = String::new();
            let mut usage = Usage::default();
            let mut finish_reason = FinishReason::Stop;

            while let Some(chunk_result) = stream.next().await {
                let chunk = match chunk_result {
                    Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                    Err(e) => {
                        let _ = tx.send(StreamEvent::Error(e.to_string())).await;
                        return;
                    }
                };

                buffer.push_str(&chunk);

                // Process complete SSE lines
                while let Some(pos) = buffer.find("\n\n") {
                    let event_block = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    for line in event_block.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            let data = data.trim();
                            if data == "[DONE]" {
                                let _ = tx
                                    .send(StreamEvent::Done {
                                        usage: usage.clone(),
                                        finish_reason: finish_reason.clone(),
                                    })
                                    .await;
                                return;
                            }

                            let parsed: serde_json::Value = match serde_json::from_str(data) {
                                Ok(v) => v,
                                Err(_) => continue,
                            };

                            // Extract usage if present (some models send it)
                            if let Some(u) = parsed.get("usage") {
                                if let Some(pt) = u.get("prompt_tokens").and_then(|v| v.as_u64()) {
                                    usage.input_tokens = pt as u32;
                                }
                                if let Some(ct) =
                                    u.get("completion_tokens").and_then(|v| v.as_u64())
                                {
                                    usage.output_tokens = ct as u32;
                                }
                            }

                            if let Some(choices) = parsed.get("choices").and_then(|c| c.as_array())
                            {
                                for choice in choices {
                                    if let Some(fr) =
                                        choice.get("finish_reason").and_then(|v| v.as_str())
                                    {
                                        finish_reason = match fr {
                                            "stop" => FinishReason::Stop,
                                            "tool_calls" => FinishReason::ToolUse,
                                            "length" => FinishReason::MaxTokens,
                                            other => FinishReason::Other(other.into()),
                                        };
                                    }

                                    if let Some(delta) = choice.get("delta") {
                                        // Text content
                                        if let Some(content) =
                                            delta.get("content").and_then(|v| v.as_str())
                                        {
                                            if !content.is_empty() {
                                                let _ = tx
                                                    .send(StreamEvent::TextDelta(
                                                        content.to_string(),
                                                    ))
                                                    .await;
                                            }
                                        }

                                        // Tool calls
                                        if let Some(tool_calls) =
                                            delta.get("tool_calls").and_then(|v| v.as_array())
                                        {
                                            for tc in tool_calls {
                                                let id = tc
                                                    .get("id")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or("")
                                                    .to_string();
                                                if let Some(func) = tc.get("function") {
                                                    if let Some(name) =
                                                        func.get("name").and_then(|v| v.as_str())
                                                    {
                                                        let _ = tx
                                                            .send(StreamEvent::ToolCallStart {
                                                                id: id.clone(),
                                                                name: name.to_string(),
                                                            })
                                                            .await;
                                                    }
                                                    if let Some(args) = func
                                                        .get("arguments")
                                                        .and_then(|v| v.as_str())
                                                    {
                                                        if !args.is_empty() {
                                                            let _ = tx
                                                                .send(StreamEvent::ToolCallDelta {
                                                                    id: id.clone(),
                                                                    arguments_delta: args
                                                                        .to_string(),
                                                                })
                                                                .await;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Stream ended without [DONE] — emit what we have
            let _ = tx
                .send(StreamEvent::Done {
                    usage,
                    finish_reason,
                })
                .await;
        });

        Ok(rx)
    }
}

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ChatTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ChatToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatToolCall {
    id: String,
    r#type: String,
    function: ChatFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct ChatTool {
    r#type: String,
    function: ChatFunctionDef,
}

#[derive(Debug, Serialize)]
struct ChatFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl From<&ToolDefinition> for ChatTool {
    fn from(def: &ToolDefinition) -> Self {
        ChatTool {
            r#type: "function".into(),
            function: ChatFunctionDef {
                name: def.name.clone(),
                description: def.description.clone(),
                parameters: def.input_schema.clone(),
            },
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize, Default)]
struct ChatChoice {
    message: ChatResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct ChatResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<ChatToolCall>>,
}

#[derive(Debug, Deserialize, Default)]
struct ChatUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

    #[test]
    fn builds_basic_request() {
        let provider = OpenAIProvider::new("key", "gpt-4o");
        let request = CompletionRequest {
            messages: vec![Message::system("Be helpful."), Message::user("Hello")],
            tools: vec![],
            max_tokens: Some(1024),
            temperature: Some(0.5),
            model: None,
        };

        let api_req = provider.build_api_request(&request);
        assert_eq!(api_req.model, "gpt-4o");
        assert_eq!(api_req.messages.len(), 2);
        assert_eq!(api_req.messages[0].role, "system");
        assert_eq!(api_req.messages[1].role, "user");
        assert!(api_req.tools.is_none());
        assert_eq!(api_req.max_tokens, Some(1024));
        assert_eq!(api_req.temperature, Some(0.5));
    }

    #[test]
    fn builds_request_with_tools() {
        let provider = OpenAIProvider::new("key", "gpt-4o");
        let request = CompletionRequest {
            messages: vec![Message::user("search for cats")],
            tools: vec![ToolDefinition {
                name: "search".into(),
                description: "Search the web".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }),
            }],
            max_tokens: None,
            temperature: None,
            model: None,
        };

        let api_req = provider.build_api_request(&request);
        let tools = api_req.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].r#type, "function");
        assert_eq!(tools[0].function.name, "search");
    }

    #[test]
    fn builds_request_with_tool_results() {
        let provider = OpenAIProvider::new("key", "gpt-4o");

        let tool_result_msg = Message::tool_result(vec![crate::message::ToolResult {
            call_id: "call_123".into(),
            content: "search results here".into(),
            is_error: false,
        }]);

        let request = CompletionRequest {
            messages: vec![
                Message::user("hi"),
                Message::assistant_with_tool_calls(
                    "",
                    vec![ToolCall {
                        id: "call_123".into(),
                        name: "search".into(),
                        arguments: serde_json::json!({"q": "cats"}),
                    }],
                ),
                tool_result_msg,
            ],
            tools: vec![],
            max_tokens: None,
            temperature: None,
            model: None,
        };

        let api_req = provider.build_api_request(&request);
        // user, assistant with tool_calls, tool result
        assert_eq!(api_req.messages.len(), 3);
        assert_eq!(api_req.messages[2].role, "tool");
        assert_eq!(api_req.messages[2].tool_call_id, Some("call_123".into()));
    }

    #[test]
    fn parses_simple_response() {
        let provider = OpenAIProvider::new("key", "gpt-4o");

        let response = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatResponseMessage {
                    content: Some("Hello there!".into()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Some(ChatUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
            }),
        };

        let result = provider.parse_response(response);
        assert_eq!(result.message.content, "Hello there!");
        assert_eq!(result.finish_reason, FinishReason::Stop);
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 5);
    }

    #[test]
    fn parses_tool_call_response() {
        let provider = OpenAIProvider::new("key", "gpt-4o");

        let response = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatResponseMessage {
                    content: None,
                    tool_calls: Some(vec![ChatToolCall {
                        id: "call_abc".into(),
                        r#type: "function".into(),
                        function: ChatFunctionCall {
                            name: "search".into(),
                            arguments: "{\"query\":\"cats\"}".into(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".into()),
            }],
            usage: Some(ChatUsage {
                prompt_tokens: 15,
                completion_tokens: 10,
            }),
        };

        let result = provider.parse_response(response);
        assert_eq!(result.finish_reason, FinishReason::ToolUse);
        assert_eq!(result.message.tool_calls.len(), 1);
        assert_eq!(result.message.tool_calls[0].name, "search");
        assert_eq!(result.message.tool_calls[0].arguments["query"], "cats");
    }

    #[test]
    fn custom_api_url() {
        let provider = OpenAIProvider::new("key", "gpt-4o")
            .with_api_url("http://localhost:8080/v1/chat/completions");
        assert_eq!(
            provider.api_url,
            "http://localhost:8080/v1/chat/completions"
        );
    }

    #[test]
    fn error_parsing() {
        let auth_err = OpenAIProvider::parse_error(
            reqwest::StatusCode::UNAUTHORIZED,
            "{\"error\":{\"message\":\"invalid key\"}}",
        );
        assert!(matches!(auth_err, ProviderError::Auth(_)));

        let rate_err = OpenAIProvider::parse_error(reqwest::StatusCode::TOO_MANY_REQUESTS, "{}");
        assert!(matches!(rate_err, ProviderError::RateLimited { .. }));

        let ctx_err = OpenAIProvider::parse_error(
            reqwest::StatusCode::BAD_REQUEST,
            "{\"error\":{\"message\":\"maximum context length exceeded\"}}",
        );
        assert!(matches!(ctx_err, ProviderError::ContextLengthExceeded(_)));
    }
}
