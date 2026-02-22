//! Plugin system.
//!
//! Allows extending the agent with dynamically loaded plugins that can
//! register tools, hooks, and providers. Plugins are defined by implementing
//! the `Plugin` trait and can be loaded from a plugin registry.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::hook::HookRegistry;
use crate::tool::ToolRegistry;

// ---------------------------------------------------------------------------
// Plugin trait
// ---------------------------------------------------------------------------

/// A plugin that extends the agent with additional capabilities.
///
/// Plugins are initialized once during startup and can register tools,
/// hooks, and configuration. They provide a clean way to package related
/// functionality into reusable modules.
#[async_trait]
pub trait Plugin: Send + Sync {
    /// Unique name identifying this plugin.
    fn name(&self) -> &str;

    /// Human-readable description of what this plugin does.
    fn description(&self) -> &str;

    /// Semantic version of this plugin.
    fn version(&self) -> &str;

    /// Initialize the plugin. Called once during startup before
    /// tools and hooks are registered.
    async fn init(&mut self, config: &PluginConfig) -> Result<(), PluginError>;

    /// Register any tools this plugin provides.
    fn register_tools(&self, registry: &mut ToolRegistry);

    /// Register any hooks this plugin provides.
    fn register_hooks(&self, registry: &mut HookRegistry);

    /// Shut down the plugin. Called during graceful shutdown.
    async fn shutdown(&self) -> Result<(), PluginError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Plugin configuration
// ---------------------------------------------------------------------------

/// Configuration passed to a plugin during initialization.
#[derive(Debug, Clone, Default)]
pub struct PluginConfig {
    /// Key-value settings specific to this plugin, typically loaded
    /// from the application's config file.
    pub settings: HashMap<String, serde_json::Value>,
}

impl PluginConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a string setting.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.settings.get(key).and_then(|v| v.as_str())
    }

    /// Get a boolean setting.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.settings.get(key).and_then(|v| v.as_bool())
    }

    /// Get an integer setting.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.settings.get(key).and_then(|v| v.as_i64())
    }

    /// Set a value.
    pub fn set(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.settings.insert(key.into(), value);
    }
}

// ---------------------------------------------------------------------------
// Plugin registry
// ---------------------------------------------------------------------------

/// Manages a collection of plugins, handling initialization, tool/hook
/// registration, and shutdown.
pub struct PluginRegistry {
    plugins: Vec<PluginEntry>,
    configs: HashMap<String, PluginConfig>,
}

struct PluginEntry {
    plugin: Box<dyn Plugin>,
    enabled: bool,
    initialized: bool,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            configs: HashMap::new(),
        }
    }

    /// Register a plugin. It won't be initialized until `init_all()` is called.
    pub fn register(&mut self, plugin: Box<dyn Plugin>) {
        self.plugins.push(PluginEntry {
            plugin,
            enabled: true,
            initialized: false,
        });
    }

    /// Set configuration for a plugin by name.
    pub fn set_config(&mut self, name: impl Into<String>, config: PluginConfig) {
        self.configs.insert(name.into(), config);
    }

    /// Enable or disable a plugin by name.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> bool {
        for entry in &mut self.plugins {
            if entry.plugin.name() == name {
                entry.enabled = enabled;
                return true;
            }
        }
        false
    }

    /// Initialize all enabled plugins.
    pub async fn init_all(&mut self) -> Result<Vec<String>, PluginError> {
        let mut initialized = Vec::new();

        for entry in &mut self.plugins {
            if !entry.enabled {
                continue;
            }

            let config = self
                .configs
                .get(entry.plugin.name())
                .cloned()
                .unwrap_or_default();

            entry
                .plugin
                .init(&config)
                .await
                .map_err(|e| PluginError::InitFailed {
                    plugin: entry.plugin.name().to_string(),
                    cause: e.to_string(),
                })?;

            entry.initialized = true;
            initialized.push(entry.plugin.name().to_string());
        }

        Ok(initialized)
    }

    /// Register tools from all initialized plugins.
    pub fn register_all_tools(&self, registry: &mut ToolRegistry) {
        for entry in &self.plugins {
            if entry.enabled && entry.initialized {
                entry.plugin.register_tools(registry);
            }
        }
    }

    /// Register hooks from all initialized plugins.
    pub fn register_all_hooks(&self, registry: &mut HookRegistry) {
        for entry in &self.plugins {
            if entry.enabled && entry.initialized {
                entry.plugin.register_hooks(registry);
            }
        }
    }

    /// Shut down all initialized plugins.
    pub async fn shutdown_all(&self) -> Vec<(String, Result<(), PluginError>)> {
        let mut results = Vec::new();

        for entry in self.plugins.iter().rev() {
            if entry.initialized {
                let name = entry.plugin.name().to_string();
                let result = entry.plugin.shutdown().await;
                results.push((name, result));
            }
        }

        results
    }

    /// List all registered plugins.
    pub fn list(&self) -> Vec<PluginInfo> {
        self.plugins
            .iter()
            .map(|entry| PluginInfo {
                name: entry.plugin.name().to_string(),
                description: entry.plugin.description().to_string(),
                version: entry.plugin.version().to_string(),
                enabled: entry.enabled,
                initialized: entry.initialized,
            })
            .collect()
    }

    /// Get the number of registered plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Plugin info
// ---------------------------------------------------------------------------

/// Public information about a registered plugin.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PluginInfo {
    pub name: String,
    pub description: String,
    pub version: String,
    pub enabled: bool,
    pub initialized: bool,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    #[error("plugin '{plugin}' failed to initialize: {cause}")]
    InitFailed { plugin: String, cause: String },

    #[error("plugin error: {0}")]
    Other(String),
}

// Display is derived by thiserror::Error above.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct TestPlugin {
        name: String,
        init_called: Arc<std::sync::atomic::AtomicBool>,
        shutdown_called: Arc<std::sync::atomic::AtomicBool>,
    }

    impl TestPlugin {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                init_called: Arc::new(std::sync::atomic::AtomicBool::new(false)),
                shutdown_called: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            }
        }
    }

    #[async_trait]
    impl Plugin for TestPlugin {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "A test plugin"
        }

        fn version(&self) -> &str {
            "1.0.0"
        }

        async fn init(&mut self, _config: &PluginConfig) -> Result<(), PluginError> {
            self.init_called
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }

        fn register_tools(&self, _registry: &mut ToolRegistry) {
            // No tools in test plugin
        }

        fn register_hooks(&self, _registry: &mut HookRegistry) {
            // No hooks in test plugin
        }

        async fn shutdown(&self) -> Result<(), PluginError> {
            self.shutdown_called
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn plugin_config_get_set() {
        let mut config = PluginConfig::new();
        config.set("api_key", serde_json::json!("sk-123"));
        config.set("enabled", serde_json::json!(true));
        config.set("max_retries", serde_json::json!(3));

        assert_eq!(config.get_str("api_key"), Some("sk-123"));
        assert_eq!(config.get_bool("enabled"), Some(true));
        assert_eq!(config.get_i64("max_retries"), Some(3));
        assert_eq!(config.get_str("missing"), None);
    }

    #[tokio::test]
    async fn registry_init_and_list() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(TestPlugin::new("test-1")));
        registry.register(Box::new(TestPlugin::new("test-2")));

        assert_eq!(registry.len(), 2);

        let initialized = registry.init_all().await.unwrap();
        assert_eq!(initialized.len(), 2);

        let list = registry.list();
        assert_eq!(list.len(), 2);
        assert!(list[0].initialized);
        assert!(list[1].initialized);
    }

    #[tokio::test]
    async fn disabled_plugin_not_initialized() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(TestPlugin::new("active")));
        registry.register(Box::new(TestPlugin::new("disabled")));
        registry.set_enabled("disabled", false);

        let initialized = registry.init_all().await.unwrap();
        assert_eq!(initialized, vec!["active"]);

        let list = registry.list();
        let disabled = list.iter().find(|p| p.name == "disabled").unwrap();
        assert!(!disabled.initialized);
        assert!(!disabled.enabled);
    }

    #[tokio::test]
    async fn shutdown_all_plugins() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(TestPlugin::new("p1")));
        registry.register(Box::new(TestPlugin::new("p2")));

        registry.init_all().await.unwrap();
        let results = registry.shutdown_all().await;

        assert_eq!(results.len(), 2);
        for (_, result) in &results {
            assert!(result.is_ok());
        }
    }

    #[test]
    fn empty_registry() {
        let registry = PluginRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn set_enabled_returns_false_for_unknown() {
        let mut registry = PluginRegistry::new();
        assert!(!registry.set_enabled("nonexistent", false));
    }

    #[test]
    fn plugin_info_serialization() {
        let info = PluginInfo {
            name: "test".into(),
            description: "A test".into(),
            version: "1.0.0".into(),
            enabled: true,
            initialized: false,
        };

        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("\"name\":\"test\""));
        assert!(json.contains("\"version\":\"1.0.0\""));
    }

    #[tokio::test]
    async fn plugin_with_config() {
        let mut registry = PluginRegistry::new();
        registry.register(Box::new(TestPlugin::new("configured")));

        let mut config = PluginConfig::new();
        config.set("api_key", serde_json::json!("secret"));
        registry.set_config("configured", config);

        let initialized = registry.init_all().await.unwrap();
        assert_eq!(initialized, vec!["configured"]);
    }
}
