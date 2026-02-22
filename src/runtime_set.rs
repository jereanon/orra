//! Multi-agent runtime management.
//!
//! Provides [`RuntimeSetBuilder`] for managing a collection of named runtimes,
//! one per agent. Handles the chicken-and-egg problem of inter-agent
//! delegation (tools need runtimes, runtimes need tools) by using a
//! shared `Arc<RwLock<HashMap>>` that is populated after all runtimes
//! are constructed.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::agent::AgentProfile;
use crate::context::{CharEstimator, ContextBudget, Tokenizer};
use crate::hook::HookRegistry;
use crate::policy::PolicyRegistry;
use crate::prompt::PromptBuilder;
use crate::provider::Provider;
use crate::runtime::{Runtime, RuntimeConfig};
use crate::store::SessionStore;
use crate::tool::ToolRegistry;
use crate::tools::delegation::DelegateToAgentTool;

/// A shared map of named runtimes, keyed by lowercase agent name.
pub type RuntimeMap<T> = Arc<RwLock<HashMap<String, Arc<Runtime<T>>>>>;

/// Configuration for building a runtime set.
pub struct RuntimeSetConfig {
    /// Maximum turns per agent.
    pub max_turns: usize,
    /// Maximum tokens per completion.
    pub max_tokens: Option<u32>,
    /// Temperature for completions.
    pub temperature: Option<f32>,
    /// Context budget for the context window.
    pub context_budget: ContextBudget,
    /// Whether to enable parallel tool execution.
    pub parallel_tool_execution: bool,
    /// Whether to register the `DelegateToAgentTool` for inter-agent
    /// delegation. Only useful when there are multiple agents.
    pub enable_delegation: bool,
}

impl Default for RuntimeSetConfig {
    fn default() -> Self {
        Self {
            max_turns: 10,
            max_tokens: None,
            temperature: None,
            context_budget: ContextBudget::default(),
            parallel_tool_execution: true,
            enable_delegation: true,
        }
    }
}

/// Result of building a runtime set.
pub struct RuntimeSetResult<T: Tokenizer> {
    /// The shared map of named runtimes (lowercase keys).
    pub runtimes: RuntimeMap<T>,
    /// The name of the default agent (first in the list).
    pub default_agent: String,
}

/// Builder for creating a set of named runtimes from agent profiles.
///
/// Handles the common pattern of:
/// 1. Creating a shared `RuntimeMap` upfront
/// 2. Building per-agent runtimes with shared provider/store
/// 3. Optionally registering `DelegateToAgentTool` for inter-agent delegation
/// 4. Populating the shared map after all runtimes are built
///
/// # Example
///
/// ```ignore
/// let agents = vec![
///     AgentProfile::new("Atlas").with_personality("helpful"),
///     AgentProfile::new("CodeBot").with_personality("precise coder"),
/// ];
///
/// let result = RuntimeSetBuilder::new()
///     .provider(my_provider)
///     .store(my_store)
///     .agents(agents)
///     .tool_factory(|_agent| {
///         let mut registry = ToolRegistry::new();
///         // register tools...
///         registry
///     })
///     .build()
///     .await;
///
/// // result.runtimes contains both agents
/// // result.default_agent is "Atlas"
/// ```
#[allow(clippy::type_complexity)]
pub struct RuntimeSetBuilder<F>
where
    F: FnMut(&AgentProfile) -> ToolRegistry,
{
    agents: Vec<AgentProfile>,
    provider: Option<Arc<dyn Provider>>,
    store: Option<Arc<dyn SessionStore>>,
    tool_factory: Option<F>,
    prompt_builder: Option<PromptBuilder>,
    hook_factory: Option<Box<dyn FnMut(&AgentProfile) -> HookRegistry>>,
    config: RuntimeSetConfig,
}

impl RuntimeSetBuilder<fn(&AgentProfile) -> ToolRegistry> {
    /// Create a new builder with no configuration.
    pub fn new() -> RuntimeSetBuilder<fn(&AgentProfile) -> ToolRegistry> {
        RuntimeSetBuilder {
            agents: Vec::new(),
            provider: None,
            store: None,
            tool_factory: None,
            prompt_builder: None,
            hook_factory: None,
            config: RuntimeSetConfig::default(),
        }
    }
}

impl<F> RuntimeSetBuilder<F>
where
    F: FnMut(&AgentProfile) -> ToolRegistry,
{
    /// Set the agent profiles.
    pub fn agents(mut self, agents: Vec<AgentProfile>) -> Self {
        self.agents = agents;
        self
    }

    /// Set the shared provider for all agents.
    pub fn provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the shared session store for all agents.
    pub fn store(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Set a factory function that creates a `ToolRegistry` for each agent.
    pub fn tool_factory<G>(self, factory: G) -> RuntimeSetBuilder<G>
    where
        G: FnMut(&AgentProfile) -> ToolRegistry,
    {
        RuntimeSetBuilder {
            agents: self.agents,
            provider: self.provider,
            store: self.store,
            tool_factory: Some(factory),
            prompt_builder: self.prompt_builder,
            hook_factory: self.hook_factory,
            config: self.config,
        }
    }

    /// Set a prompt builder for generating per-agent system prompts.
    pub fn prompt_builder(mut self, builder: PromptBuilder) -> Self {
        self.prompt_builder = Some(builder);
        self
    }

    /// Set a factory function that creates a `HookRegistry` for each agent.
    pub fn hook_factory(
        mut self,
        factory: impl FnMut(&AgentProfile) -> HookRegistry + 'static,
    ) -> Self {
        self.hook_factory = Some(Box::new(factory));
        self
    }

    /// Set the runtime configuration.
    pub fn config(mut self, config: RuntimeSetConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the runtime set.
    ///
    /// Creates a `Runtime` for each agent profile, optionally registers
    /// `DelegateToAgentTool` for inter-agent delegation, and returns
    /// the shared runtime map.
    ///
    /// # Panics
    ///
    /// Panics if `provider` or `store` are not set, or if `agents` is empty.
    pub async fn build(mut self) -> RuntimeSetResult<CharEstimator> {
        let provider = self.provider.expect("provider is required");
        let store = self.store.expect("store is required");
        assert!(!self.agents.is_empty(), "at least one agent is required");

        let default_agent = self.agents[0].name.clone();
        let prompt_builder = self.prompt_builder.unwrap_or_default();

        // Create the shared map upfront so DelegateToAgentTool can reference it
        let runtimes: RuntimeMap<CharEstimator> = Arc::new(RwLock::new(HashMap::new()));
        let mut runtimes_map = HashMap::new();

        let enable_delegation = self.config.enable_delegation && self.agents.len() > 1;

        for agent in &self.agents {
            let system_prompt = prompt_builder.build(agent);

            let runtime_config = RuntimeConfig {
                system_prompt: Some(system_prompt),
                max_turns: self.config.max_turns,
                max_tokens: self.config.max_tokens,
                temperature: self.config.temperature,
                context_budget: self.config.context_budget.clone(),
                parallel_tool_execution: self.config.parallel_tool_execution,
            };

            let mut tools = if let Some(ref mut factory) = self.tool_factory {
                factory(agent)
            } else {
                ToolRegistry::new()
            };

            // Register inter-agent delegation if enabled
            if enable_delegation {
                tools.register(Box::new(DelegateToAgentTool::new(
                    runtimes.clone(),
                    agent.name.clone(),
                )));
            }

            let hooks = if let Some(ref mut factory) = self.hook_factory {
                factory(agent)
            } else {
                HookRegistry::default()
            };

            let mut rt = Runtime::new(
                provider.clone(),
                store.clone(),
                tools,
                PolicyRegistry::default(),
                CharEstimator::default(),
                runtime_config,
            );
            rt.set_hooks(hooks);

            let key = agent.name.to_lowercase();
            runtimes_map.insert(key, Arc::new(rt));
        }

        // Populate the shared map now that all runtimes are built
        *runtimes.write().await = runtimes_map;

        RuntimeSetResult {
            runtimes,
            default_agent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use crate::provider::{
        CompletionRequest, CompletionResponse, FinishReason, ProviderError, Usage,
    };
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct FixedProvider {
        call_count: AtomicUsize,
    }

    #[async_trait]
    impl Provider for FixedProvider {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, ProviderError> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(CompletionResponse {
                message: Message::assistant("ok"),
                usage: Usage::default(),
                finish_reason: FinishReason::Stop,
            })
        }
    }

    #[tokio::test]
    async fn build_single_agent() {
        let provider = Arc::new(FixedProvider {
            call_count: AtomicUsize::new(0),
        });
        let store = Arc::new(crate::store::InMemoryStore::new());

        let result = RuntimeSetBuilder::new()
            .agents(vec![AgentProfile::new("Atlas")])
            .provider(provider)
            .store(store)
            .config(RuntimeSetConfig {
                enable_delegation: false,
                ..Default::default()
            })
            .build()
            .await;

        assert_eq!(result.default_agent, "Atlas");
        let map = result.runtimes.read().await;
        assert_eq!(map.len(), 1);
        assert!(map.contains_key("atlas"));
    }

    #[tokio::test]
    async fn build_multiple_agents() {
        let provider = Arc::new(FixedProvider {
            call_count: AtomicUsize::new(0),
        });
        let store = Arc::new(crate::store::InMemoryStore::new());

        let result = RuntimeSetBuilder::new()
            .agents(vec![
                AgentProfile::new("Atlas").with_personality("helpful"),
                AgentProfile::new("CodeBot").with_personality("precise"),
            ])
            .provider(provider)
            .store(store)
            .build()
            .await;

        assert_eq!(result.default_agent, "Atlas");
        let map = result.runtimes.read().await;
        assert_eq!(map.len(), 2);
        assert!(map.contains_key("atlas"));
        assert!(map.contains_key("codebot"));
    }

    #[tokio::test]
    async fn build_with_tool_factory() {
        let provider = Arc::new(FixedProvider {
            call_count: AtomicUsize::new(0),
        });
        let store = Arc::new(crate::store::InMemoryStore::new());

        let result = RuntimeSetBuilder::new()
            .agents(vec![AgentProfile::new("Atlas")])
            .provider(provider)
            .store(store)
            .tool_factory(|_agent| {
                // Could register tools here
                ToolRegistry::new()
            })
            .config(RuntimeSetConfig {
                enable_delegation: false,
                ..Default::default()
            })
            .build()
            .await;

        let map = result.runtimes.read().await;
        assert_eq!(map.len(), 1);
    }

    #[tokio::test]
    async fn build_with_custom_prompt() {
        let provider = Arc::new(FixedProvider {
            call_count: AtomicUsize::new(0),
        });
        let store = Arc::new(crate::store::InMemoryStore::new());

        let mut prompt_builder = PromptBuilder::new();
        prompt_builder.set_channel_context("a Discord server");

        let result = RuntimeSetBuilder::new()
            .agents(vec![AgentProfile::new("Atlas")])
            .provider(provider)
            .store(store)
            .prompt_builder(prompt_builder)
            .config(RuntimeSetConfig {
                enable_delegation: false,
                ..Default::default()
            })
            .build()
            .await;

        let map = result.runtimes.read().await;
        assert_eq!(map.len(), 1);
    }

    #[tokio::test]
    async fn default_agent_is_first() {
        let provider = Arc::new(FixedProvider {
            call_count: AtomicUsize::new(0),
        });
        let store = Arc::new(crate::store::InMemoryStore::new());

        let result = RuntimeSetBuilder::new()
            .agents(vec![
                AgentProfile::new("SecondBot"),
                AgentProfile::new("FirstBot"),
            ])
            .provider(provider)
            .store(store)
            .config(RuntimeSetConfig {
                enable_delegation: false,
                ..Default::default()
            })
            .build()
            .await;

        // Default is the first in the list, not alphabetical
        assert_eq!(result.default_agent, "SecondBot");
    }

    #[tokio::test]
    #[should_panic(expected = "at least one agent")]
    async fn build_with_no_agents_panics() {
        let provider = Arc::new(FixedProvider {
            call_count: AtomicUsize::new(0),
        });
        let store = Arc::new(crate::store::InMemoryStore::new());

        RuntimeSetBuilder::new()
            .agents(vec![])
            .provider(provider)
            .store(store)
            .build()
            .await;
    }
}
