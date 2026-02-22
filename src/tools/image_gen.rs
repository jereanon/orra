//! Image generation tool.
//!
//! Provides a tool that wraps image generation APIs (OpenAI DALL-E, Stability AI,
//! etc.) behind a common interface. The actual HTTP calls are abstracted behind
//! the `ImageProvider` trait so different backends can be plugged in.

use std::sync::Arc;

use async_trait::async_trait;

use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

// ---------------------------------------------------------------------------
// Image provider trait
// ---------------------------------------------------------------------------

/// Result of an image generation request.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// URL where the generated image can be accessed.
    pub url: String,

    /// Revised prompt (some providers rewrite the prompt for safety/quality).
    pub revised_prompt: Option<String>,
}

/// Trait for image generation backends.
#[async_trait]
pub trait ImageProvider: Send + Sync {
    /// Generate an image from a text prompt.
    async fn generate(
        &self,
        prompt: &str,
        options: &ImageOptions,
    ) -> Result<GeneratedImage, ImageGenError>;
}

/// Options for image generation.
#[derive(Debug, Clone)]
pub struct ImageOptions {
    /// Desired image size (e.g., "1024x1024", "512x512").
    pub size: String,

    /// Style hint (e.g., "vivid", "natural"). Provider-specific.
    pub style: Option<String>,

    /// Quality hint (e.g., "standard", "hd"). Provider-specific.
    pub quality: Option<String>,
}

impl Default for ImageOptions {
    fn default() -> Self {
        Self {
            size: "1024x1024".into(),
            style: None,
            quality: None,
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI DALL-E provider
// ---------------------------------------------------------------------------

/// Image generation using OpenAI's DALL-E API.
pub struct DallEProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    api_url: String,
}

impl DallEProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            model: "dall-e-3".into(),
            api_url: "https://api.openai.com/v1/images/generations".into(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }
}

#[async_trait]
impl ImageProvider for DallEProvider {
    async fn generate(
        &self,
        prompt: &str,
        options: &ImageOptions,
    ) -> Result<GeneratedImage, ImageGenError> {
        let mut body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "n": 1,
            "size": options.size,
            "response_format": "url",
        });

        if let Some(ref quality) = options.quality {
            body["quality"] = serde_json::json!(quality);
        }
        if let Some(ref style) = options.style {
            body["style"] = serde_json::json!(style);
        }

        let response = self
            .client
            .post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| ImageGenError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            return Err(ImageGenError::Api {
                status,
                message: text,
            });
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| ImageGenError::Parse(e.to_string()))?;

        let image_data = data["data"]
            .as_array()
            .and_then(|arr| arr.first())
            .ok_or_else(|| ImageGenError::Parse("no images in response".into()))?;

        let url = image_data["url"]
            .as_str()
            .ok_or_else(|| ImageGenError::Parse("missing image url".into()))?
            .to_string();

        let revised_prompt = image_data["revised_prompt"].as_str().map(|s| s.to_string());

        Ok(GeneratedImage {
            url,
            revised_prompt,
        })
    }
}

// ---------------------------------------------------------------------------
// Image generation tool
// ---------------------------------------------------------------------------

/// Agent-facing tool for generating images from text descriptions.
pub struct ImageGenTool {
    provider: Arc<dyn ImageProvider>,
}

impl ImageGenTool {
    pub fn new(provider: Arc<dyn ImageProvider>) -> Self {
        Self { provider }
    }
}

#[async_trait]
impl Tool for ImageGenTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "generate_image".into(),
            description: "Generate an image from a text description. Returns a URL to the \
                          generated image."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A detailed description of the image to generate"
                    },
                    "size": {
                        "type": "string",
                        "description": "Image size (e.g., '1024x1024', '1792x1024'). Default: 1024x1024",
                        "enum": ["1024x1024", "1792x1024", "1024x1792"]
                    },
                    "style": {
                        "type": "string",
                        "description": "Image style: 'vivid' for dramatic, 'natural' for realistic",
                        "enum": ["vivid", "natural"]
                    },
                    "quality": {
                        "type": "string",
                        "description": "Quality level: 'standard' or 'hd'",
                        "enum": ["standard", "hd"]
                    }
                },
                "required": ["prompt"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let prompt = input
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'prompt'".into()))?;

        let size = input
            .get("size")
            .and_then(|v| v.as_str())
            .unwrap_or("1024x1024")
            .to_string();

        let style = input
            .get("style")
            .and_then(|v| v.as_str())
            .map(String::from);
        let quality = input
            .get("quality")
            .and_then(|v| v.as_str())
            .map(String::from);

        let options = ImageOptions {
            size,
            style,
            quality,
        };

        let result = self
            .provider
            .generate(prompt, &options)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let mut output = format!("Generated image: {}", result.url);
        if let Some(revised) = &result.revised_prompt {
            output.push_str(&format!("\nRevised prompt: {revised}"));
        }

        Ok(output)
    }
}

/// Register the image generation tool.
pub fn register_tool(registry: &mut ToolRegistry, provider: Arc<dyn ImageProvider>) {
    registry.register(Box::new(ImageGenTool::new(provider)));
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ImageGenError {
    #[error("request failed: {0}")]
    Request(String),

    #[error("API error (status {status}): {message}")]
    Api { status: u16, message: String },

    #[error("failed to parse response: {0}")]
    Parse(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Mock image provider for testing
    struct MockImageProvider {
        url: String,
    }

    #[async_trait]
    impl ImageProvider for MockImageProvider {
        async fn generate(
            &self,
            prompt: &str,
            _options: &ImageOptions,
        ) -> Result<GeneratedImage, ImageGenError> {
            Ok(GeneratedImage {
                url: self.url.clone(),
                revised_prompt: Some(format!("A beautiful {prompt}")),
            })
        }
    }

    struct FailingImageProvider;

    #[async_trait]
    impl ImageProvider for FailingImageProvider {
        async fn generate(
            &self,
            _prompt: &str,
            _options: &ImageOptions,
        ) -> Result<GeneratedImage, ImageGenError> {
            Err(ImageGenError::Api {
                status: 429,
                message: "rate limited".into(),
            })
        }
    }

    #[tokio::test]
    async fn image_gen_tool_basic() {
        let provider = Arc::new(MockImageProvider {
            url: "https://example.com/image.png".into(),
        });
        let tool = ImageGenTool::new(provider);

        let result = tool
            .execute(serde_json::json!({
                "prompt": "a sunset over mountains"
            }))
            .await
            .unwrap();

        assert!(result.contains("https://example.com/image.png"));
        assert!(result.contains("Revised prompt"));
    }

    #[tokio::test]
    async fn image_gen_tool_with_options() {
        let provider = Arc::new(MockImageProvider {
            url: "https://example.com/img.png".into(),
        });
        let tool = ImageGenTool::new(provider);

        let result = tool
            .execute(serde_json::json!({
                "prompt": "a cat",
                "size": "1792x1024",
                "style": "vivid",
                "quality": "hd"
            }))
            .await
            .unwrap();

        assert!(result.contains("https://example.com/img.png"));
    }

    #[tokio::test]
    async fn image_gen_tool_missing_prompt() {
        let provider = Arc::new(MockImageProvider {
            url: "https://example.com/img.png".into(),
        });
        let tool = ImageGenTool::new(provider);

        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn image_gen_tool_provider_error() {
        let provider = Arc::new(FailingImageProvider);
        let tool = ImageGenTool::new(provider);

        let err = tool
            .execute(serde_json::json!({"prompt": "test"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
    }

    #[test]
    fn tool_definition_valid() {
        let provider = Arc::new(MockImageProvider { url: "test".into() });
        let tool = ImageGenTool::new(provider);
        let def = tool.definition();

        assert_eq!(def.name, "generate_image");
        assert!(def.input_schema["required"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("prompt")));
    }

    #[test]
    fn default_image_options() {
        let opts = ImageOptions::default();
        assert_eq!(opts.size, "1024x1024");
        assert!(opts.style.is_none());
        assert!(opts.quality.is_none());
    }

    #[test]
    fn image_gen_error_display() {
        let err = ImageGenError::Api {
            status: 400,
            message: "bad prompt".into(),
        };
        assert!(err.to_string().contains("400"));
        assert!(err.to_string().contains("bad prompt"));
    }
}
