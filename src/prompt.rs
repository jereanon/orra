//! System prompt building utilities.
//!
//! Provides a [`PromptBuilder`] for constructing system prompts from
//! agent profiles and tool descriptions. Applications can use the
//! default builder or implement custom prompt generation.

use crate::agent::AgentProfile;
use crate::tool::ToolRegistry;

/// A capability description for inclusion in the system prompt.
///
/// Each entry describes a tool or group of tools the agent has access to,
/// so the LLM knows what it can do.
#[derive(Debug, Clone)]
pub struct CapabilityDescription {
    /// Short name for the capability (e.g. "Web search").
    pub name: String,
    /// Longer description of what the capability does.
    pub description: String,
}

impl CapabilityDescription {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }
}

/// Builds a system prompt for an agent.
///
/// The default implementation generates a prompt from the agent's name,
/// personality, and capability descriptions. Applications can extend this
/// with custom sections.
pub struct PromptBuilder {
    /// Capability descriptions to include (tool/feature descriptions).
    capabilities: Vec<CapabilityDescription>,
    /// Additional context lines appended after the main prompt.
    extra_sections: Vec<String>,
    /// Override the channel context (default: "a chat interface").
    channel_context: String,
}

impl Default for PromptBuilder {
    fn default() -> Self {
        Self {
            capabilities: Vec::new(),
            extra_sections: Vec::new(),
            channel_context: "a chat interface".into(),
        }
    }
}

impl PromptBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a capability description.
    pub fn add_capability(&mut self, cap: CapabilityDescription) -> &mut Self {
        self.capabilities.push(cap);
        self
    }

    /// Add multiple capability descriptions at once.
    pub fn add_capabilities(
        &mut self,
        caps: impl IntoIterator<Item = CapabilityDescription>,
    ) -> &mut Self {
        self.capabilities.extend(caps);
        self
    }

    /// Add a free-form section to the end of the prompt.
    pub fn add_section(&mut self, section: impl Into<String>) -> &mut Self {
        self.extra_sections.push(section.into());
        self
    }

    /// Set the channel context description (e.g. "a Discord server", "a web chat").
    pub fn set_channel_context(&mut self, context: impl Into<String>) -> &mut Self {
        self.channel_context = context.into();
        self
    }

    /// Auto-detect capabilities from a tool registry.
    ///
    /// Creates a summary line for each registered tool based on its
    /// definition name and description.
    pub fn add_capabilities_from_registry(&mut self, registry: &ToolRegistry) -> &mut Self {
        for def in registry.definitions() {
            self.capabilities.push(CapabilityDescription {
                name: def.name.clone(),
                description: def.description.clone(),
            });
        }
        self
    }

    /// Build the system prompt for an agent profile.
    ///
    /// If the agent has a custom `system_prompt` set, that is returned
    /// verbatim. Otherwise, generates a prompt from the agent's name,
    /// personality, capabilities, and extra sections.
    pub fn build(&self, agent: &AgentProfile) -> String {
        // Custom system prompt takes precedence
        if let Some(ref custom) = agent.system_prompt {
            if !custom.is_empty() {
                return custom.clone();
            }
        }

        let name = &agent.name;
        let personality = &agent.personality;
        let now = chrono::Local::now().format("%A, %B %d, %Y");

        let tools_section = if self.capabilities.is_empty() {
            String::new()
        } else {
            let lines: Vec<String> = self
                .capabilities
                .iter()
                .map(|cap| cap.description.clone())
                .collect();
            format!(
                "\n\nYou have access to the following capabilities:\n- {}",
                lines.join("\n- ")
            )
        };

        let extras = if self.extra_sections.is_empty() {
            String::new()
        } else {
            format!("\n\n{}", self.extra_sections.join("\n\n"))
        };

        format!(
            "You are {name}, an AI assistant. Your personality is: {personality}.\n\
             Today is {now}.\n\
             \n\
             You are chatting in {channel_context}. Keep your responses concise and \
             conversational. Use markdown formatting when appropriate.\n\
             \n\
             When a user asks you something, respond directly. If you need more \
             information, use your tools to look it up before answering. If you're \
             unsure about something, say so rather than making things up.{tools_section}{extras}",
            channel_context = self.channel_context,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_prompt_includes_name_and_personality() {
        let builder = PromptBuilder::new();
        let agent = AgentProfile::new("Atlas").with_personality("friendly and helpful");

        let prompt = builder.build(&agent);
        assert!(prompt.contains("Atlas"));
        assert!(prompt.contains("friendly and helpful"));
        assert!(prompt.contains("AI assistant"));
    }

    #[test]
    fn custom_system_prompt_overrides() {
        let builder = PromptBuilder::new();
        let agent = AgentProfile::new("Atlas").with_system_prompt("You are a pirate.");

        let prompt = builder.build(&agent);
        assert_eq!(prompt, "You are a pirate.");
    }

    #[test]
    fn empty_custom_prompt_falls_back_to_auto() {
        let builder = PromptBuilder::new();
        let agent = AgentProfile::new("Atlas").with_system_prompt("");

        let prompt = builder.build(&agent);
        assert!(prompt.contains("Atlas"));
    }

    #[test]
    fn capabilities_included_in_prompt() {
        let mut builder = PromptBuilder::new();
        builder.add_capability(CapabilityDescription::new(
            "Web search",
            "You can search the web to find current information.",
        ));
        builder.add_capability(CapabilityDescription::new(
            "Memory",
            "You can remember and recall information.",
        ));

        let agent = AgentProfile::new("Atlas");
        let prompt = builder.build(&agent);

        assert!(prompt.contains("search the web"));
        assert!(prompt.contains("remember and recall"));
        assert!(prompt.contains("capabilities"));
    }

    #[test]
    fn channel_context_customizable() {
        let mut builder = PromptBuilder::new();
        builder.set_channel_context("a Discord server");

        let agent = AgentProfile::new("Atlas");
        let prompt = builder.build(&agent);
        assert!(prompt.contains("a Discord server"));
    }

    #[test]
    fn extra_sections_appended() {
        let mut builder = PromptBuilder::new();
        builder.add_section("Always end with a haiku.");

        let agent = AgentProfile::new("Atlas");
        let prompt = builder.build(&agent);
        assert!(prompt.contains("Always end with a haiku."));
    }

    #[test]
    fn no_capabilities_no_section() {
        let builder = PromptBuilder::new();
        let agent = AgentProfile::new("Atlas");
        let prompt = builder.build(&agent);
        assert!(!prompt.contains("capabilities"));
    }

    #[test]
    fn capabilities_from_registry() {
        use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

        struct DummyTool;

        #[async_trait::async_trait]
        impl Tool for DummyTool {
            fn definition(&self) -> ToolDefinition {
                ToolDefinition {
                    name: "dummy".into(),
                    description: "A dummy tool for testing".into(),
                    input_schema: serde_json::json!({"type": "object"}),
                }
            }
            async fn execute(&self, _input: serde_json::Value) -> Result<String, ToolError> {
                Ok("ok".into())
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(Box::new(DummyTool));

        let mut builder = PromptBuilder::new();
        builder.add_capabilities_from_registry(&registry);

        let agent = AgentProfile::new("Test");
        let prompt = builder.build(&agent);
        assert!(prompt.contains("dummy tool for testing"));
    }
}
