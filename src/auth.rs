//! Authentication and OAuth2 support.
//!
//! Provides token management for provider authentication, including OAuth2
//! authorization code and client credentials flows. Handles token refresh
//! automatically when tokens expire.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Token types
// ---------------------------------------------------------------------------

/// An access token with optional expiration.
#[derive(Debug, Clone)]
pub struct AccessToken {
    /// The token value.
    pub token: String,

    /// Token type (usually "Bearer").
    pub token_type: String,

    /// When this token expires, if known.
    pub expires_at: Option<Instant>,

    /// Refresh token for obtaining a new access token.
    pub refresh_token: Option<String>,

    /// Scopes granted by this token.
    pub scopes: Vec<String>,
}

impl AccessToken {
    /// Check if this token has expired (with a 30-second safety margin).
    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(expires_at) => Instant::now() + Duration::from_secs(30) >= expires_at,
            None => false,
        }
    }

    /// Get the Authorization header value.
    pub fn authorization_header(&self) -> String {
        format!("{} {}", self.token_type, self.token)
    }
}

// ---------------------------------------------------------------------------
// OAuth2 configuration
// ---------------------------------------------------------------------------

/// Configuration for an OAuth2 provider.
#[derive(Debug, Clone)]
pub struct OAuth2Config {
    /// Client ID.
    pub client_id: String,

    /// Client secret.
    pub client_secret: String,

    /// Authorization endpoint URL.
    pub auth_url: String,

    /// Token endpoint URL.
    pub token_url: String,

    /// Redirect URI for authorization code flow.
    pub redirect_uri: Option<String>,

    /// Requested scopes.
    pub scopes: Vec<String>,
}

// ---------------------------------------------------------------------------
// Token provider trait
// ---------------------------------------------------------------------------

/// Trait for obtaining and refreshing access tokens.
#[async_trait]
pub trait TokenProvider: Send + Sync {
    /// Get a valid access token. May refresh automatically if expired.
    async fn get_token(&self) -> Result<AccessToken, AuthError>;

    /// Force a token refresh.
    async fn refresh(&self) -> Result<AccessToken, AuthError>;

    /// Revoke the current token.
    async fn revoke(&self) -> Result<(), AuthError>;
}

// ---------------------------------------------------------------------------
// Static token provider (API keys)
// ---------------------------------------------------------------------------

/// Simple provider for static API keys that never expire.
pub struct StaticTokenProvider {
    token: AccessToken,
}

impl StaticTokenProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            token: AccessToken {
                token: api_key.into(),
                token_type: "Bearer".into(),
                expires_at: None,
                refresh_token: None,
                scopes: Vec::new(),
            },
        }
    }

    pub fn with_token_type(mut self, token_type: impl Into<String>) -> Self {
        self.token.token_type = token_type.into();
        self
    }
}

#[async_trait]
impl TokenProvider for StaticTokenProvider {
    async fn get_token(&self) -> Result<AccessToken, AuthError> {
        Ok(self.token.clone())
    }

    async fn refresh(&self) -> Result<AccessToken, AuthError> {
        Ok(self.token.clone())
    }

    async fn revoke(&self) -> Result<(), AuthError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OAuth2 client credentials provider
// ---------------------------------------------------------------------------

/// OAuth2 client credentials flow. Suitable for server-to-server
/// authentication where no user interaction is needed.
pub struct ClientCredentialsProvider {
    config: OAuth2Config,
    client: reqwest::Client,
    cached_token: Arc<RwLock<Option<AccessToken>>>,
}

impl ClientCredentialsProvider {
    pub fn new(config: OAuth2Config) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            cached_token: Arc::new(RwLock::new(None)),
        }
    }

    async fn fetch_token(&self) -> Result<AccessToken, AuthError> {
        let mut params = HashMap::new();
        params.insert("grant_type", "client_credentials".to_string());
        params.insert("client_id", self.config.client_id.clone());
        params.insert("client_secret", self.config.client_secret.clone());

        if !self.config.scopes.is_empty() {
            params.insert("scope", self.config.scopes.join(" "));
        }

        let response = self
            .client
            .post(&self.config.token_url)
            .form(&params)
            .send()
            .await
            .map_err(|e| AuthError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(AuthError::TokenEndpoint {
                status,
                message: body,
            });
        }

        let data: TokenResponse = response
            .json()
            .await
            .map_err(|e| AuthError::Parse(e.to_string()))?;

        let expires_at = data
            .expires_in
            .map(|secs| Instant::now() + Duration::from_secs(secs));

        Ok(AccessToken {
            token: data.access_token,
            token_type: data.token_type.unwrap_or_else(|| "Bearer".into()),
            expires_at,
            refresh_token: data.refresh_token,
            scopes: data
                .scope
                .map(|s| s.split_whitespace().map(String::from).collect())
                .unwrap_or_default(),
        })
    }
}

use std::collections::HashMap;

#[async_trait]
impl TokenProvider for ClientCredentialsProvider {
    async fn get_token(&self) -> Result<AccessToken, AuthError> {
        // Check cache first
        if let Some(token) = self.cached_token.read().await.as_ref() {
            if !token.is_expired() {
                return Ok(token.clone());
            }
        }

        // Fetch a new token
        let token = self.fetch_token().await?;
        *self.cached_token.write().await = Some(token.clone());
        Ok(token)
    }

    async fn refresh(&self) -> Result<AccessToken, AuthError> {
        let token = self.fetch_token().await?;
        *self.cached_token.write().await = Some(token.clone());
        Ok(token)
    }

    async fn revoke(&self) -> Result<(), AuthError> {
        *self.cached_token.write().await = None;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// OAuth2 refresh token provider
// ---------------------------------------------------------------------------

/// OAuth2 provider that uses a refresh token to maintain access.
/// Useful when an initial authorization code flow has already been completed.
pub struct RefreshTokenProvider {
    config: OAuth2Config,
    client: reqwest::Client,
    cached_token: Arc<RwLock<Option<AccessToken>>>,
}

impl RefreshTokenProvider {
    pub fn new(config: OAuth2Config, initial_token: AccessToken) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            cached_token: Arc::new(RwLock::new(Some(initial_token))),
        }
    }

    async fn refresh_with_token(&self, refresh_token: &str) -> Result<AccessToken, AuthError> {
        let mut params = HashMap::new();
        params.insert("grant_type", "refresh_token".to_string());
        params.insert("refresh_token", refresh_token.to_string());
        params.insert("client_id", self.config.client_id.clone());
        params.insert("client_secret", self.config.client_secret.clone());

        let response = self
            .client
            .post(&self.config.token_url)
            .form(&params)
            .send()
            .await
            .map_err(|e| AuthError::Request(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body = response.text().await.unwrap_or_default();
            return Err(AuthError::TokenEndpoint {
                status,
                message: body,
            });
        }

        let data: TokenResponse = response
            .json()
            .await
            .map_err(|e| AuthError::Parse(e.to_string()))?;

        let expires_at = data
            .expires_in
            .map(|secs| Instant::now() + Duration::from_secs(secs));

        Ok(AccessToken {
            token: data.access_token,
            token_type: data.token_type.unwrap_or_else(|| "Bearer".into()),
            expires_at,
            refresh_token: data.refresh_token.or(Some(refresh_token.to_string())),
            scopes: data
                .scope
                .map(|s| s.split_whitespace().map(String::from).collect())
                .unwrap_or_default(),
        })
    }
}

#[async_trait]
impl TokenProvider for RefreshTokenProvider {
    async fn get_token(&self) -> Result<AccessToken, AuthError> {
        if let Some(token) = self.cached_token.read().await.as_ref() {
            if !token.is_expired() {
                return Ok(token.clone());
            }
        }

        self.refresh().await
    }

    async fn refresh(&self) -> Result<AccessToken, AuthError> {
        let current = self.cached_token.read().await.clone();
        let refresh_token = current
            .and_then(|t| t.refresh_token)
            .ok_or(AuthError::NoRefreshToken)?;

        let token = self.refresh_with_token(&refresh_token).await?;
        *self.cached_token.write().await = Some(token.clone());
        Ok(token)
    }

    async fn revoke(&self) -> Result<(), AuthError> {
        *self.cached_token.write().await = None;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Token response (OAuth2 standard format)
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
struct TokenResponse {
    access_token: String,
    token_type: Option<String>,
    expires_in: Option<u64>,
    refresh_token: Option<String>,
    scope: Option<String>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("request failed: {0}")]
    Request(String),

    #[error("token endpoint error (status {status}): {message}")]
    TokenEndpoint { status: u16, message: String },

    #[error("failed to parse token response: {0}")]
    Parse(String),

    #[error("no refresh token available")]
    NoRefreshToken,

    #[error("token revoked")]
    Revoked,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_token_never_expires() {
        let token = AccessToken {
            token: "sk-test-123".into(),
            token_type: "Bearer".into(),
            expires_at: None,
            refresh_token: None,
            scopes: Vec::new(),
        };
        assert!(!token.is_expired());
    }

    #[test]
    fn expired_token() {
        let token = AccessToken {
            token: "old".into(),
            token_type: "Bearer".into(),
            expires_at: Some(Instant::now() - Duration::from_secs(60)),
            refresh_token: None,
            scopes: Vec::new(),
        };
        assert!(token.is_expired());
    }

    #[test]
    fn token_expiring_within_margin() {
        // Token expires in 20 seconds, but we have a 30s safety margin
        let token = AccessToken {
            token: "soon".into(),
            token_type: "Bearer".into(),
            expires_at: Some(Instant::now() + Duration::from_secs(20)),
            refresh_token: None,
            scopes: Vec::new(),
        };
        assert!(token.is_expired());
    }

    #[test]
    fn valid_token() {
        let token = AccessToken {
            token: "fresh".into(),
            token_type: "Bearer".into(),
            expires_at: Some(Instant::now() + Duration::from_secs(3600)),
            refresh_token: None,
            scopes: Vec::new(),
        };
        assert!(!token.is_expired());
    }

    #[test]
    fn authorization_header() {
        let token = AccessToken {
            token: "abc123".into(),
            token_type: "Bearer".into(),
            expires_at: None,
            refresh_token: None,
            scopes: Vec::new(),
        };
        assert_eq!(token.authorization_header(), "Bearer abc123");
    }

    #[tokio::test]
    async fn static_provider_returns_token() {
        let provider = StaticTokenProvider::new("my-api-key");
        let token = provider.get_token().await.unwrap();
        assert_eq!(token.token, "my-api-key");
        assert_eq!(token.token_type, "Bearer");
    }

    #[tokio::test]
    async fn static_provider_refresh_returns_same() {
        let provider = StaticTokenProvider::new("key");
        let t1 = provider.get_token().await.unwrap();
        let t2 = provider.refresh().await.unwrap();
        assert_eq!(t1.token, t2.token);
    }

    #[tokio::test]
    async fn static_provider_with_custom_type() {
        let provider = StaticTokenProvider::new("key").with_token_type("Basic");
        let token = provider.get_token().await.unwrap();
        assert_eq!(token.token_type, "Basic");
        assert_eq!(token.authorization_header(), "Basic key");
    }

    #[test]
    fn oauth2_config_construction() {
        let config = OAuth2Config {
            client_id: "id".into(),
            client_secret: "secret".into(),
            auth_url: "https://auth.example.com/authorize".into(),
            token_url: "https://auth.example.com/token".into(),
            redirect_uri: Some("https://localhost/callback".into()),
            scopes: vec!["read".into(), "write".into()],
        };

        assert_eq!(config.client_id, "id");
        assert_eq!(config.scopes.len(), 2);
    }

    #[test]
    fn auth_error_display() {
        let err = AuthError::TokenEndpoint {
            status: 401,
            message: "invalid_client".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("401"));
        assert!(msg.contains("invalid_client"));
    }

    #[test]
    fn token_response_deserialization() {
        let json = r#"{
            "access_token": "abc",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "read write"
        }"#;

        let resp: TokenResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.access_token, "abc");
        assert_eq!(resp.expires_in, Some(3600));
        assert_eq!(resp.scope.as_deref(), Some("read write"));
    }

    #[test]
    fn token_response_minimal() {
        let json = r#"{"access_token": "xyz"}"#;
        let resp: TokenResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.access_token, "xyz");
        assert!(resp.token_type.is_none());
        assert!(resp.expires_in.is_none());
    }
}
