//! Agent profile configuration for multi-agent systems.
//!
//! Provides [`AgentProfile`] for describing named agents with distinct
//! personalities and system prompts, and a helper to resolve legacy
//! single-agent configs into the multi-agent format.

use serde::{Deserialize, Serialize};

/// A named agent profile for multi-agent setups.
///
/// Each agent has a unique name, a personality description, an optional
/// custom system prompt, and an optional per-agent model override.
///
/// # TOML example
///
/// ```toml
/// [[agents]]
/// name = "Atlas"
/// personality = "friendly, helpful, and concise"
///
/// [[agents]]
/// name = "CodeBot"
/// personality = "a precise coding assistant"
/// model = "claude-sonnet-4-20250514"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProfile {
    /// Unique name for this agent.
    pub name: String,

    /// A short personality description used in prompt generation.
    #[serde(default = "default_personality")]
    pub personality: String,

    /// If set, overrides the auto-generated system prompt entirely.
    pub system_prompt: Option<String>,

    /// Per-agent model override (uses the runtime default if `None`).
    pub model: Option<String>,
}

impl AgentProfile {
    /// Create a new agent profile with just a name (uses default personality).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            personality: default_personality(),
            system_prompt: None,
            model: None,
        }
    }

    /// Set the personality.
    pub fn with_personality(mut self, personality: impl Into<String>) -> Self {
        self.personality = personality.into();
        self
    }

    /// Set a custom system prompt override.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set a per-agent model override.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

fn default_personality() -> String {
    "friendly, helpful, and concise".into()
}

/// Default agent name used when none is configured.
pub fn default_agent_name() -> String {
    "Atlas".into()
}

/// Resolve a list of agent profiles from potentially legacy config.
///
/// If `agents` is non-empty, returns it. Otherwise, creates a single
/// profile from the legacy `name`/`personality`/`system_prompt` values.
///
/// This allows backward compatibility with single-agent config formats:
///
/// ```toml
/// # Legacy format (single agent):
/// [agent]
/// name = "Atlas"
/// personality = "friendly"
///
/// # New format (multi-agent):
/// [[agents]]
/// name = "Atlas"
/// personality = "friendly"
///
/// [[agents]]
/// name = "CodeBot"
/// personality = "precise coder"
/// ```
pub fn resolve_agents(
    agents: &[AgentProfile],
    legacy_name: Option<&str>,
    legacy_personality: Option<&str>,
    legacy_system_prompt: Option<&str>,
) -> Vec<AgentProfile> {
    if !agents.is_empty() {
        return agents.to_vec();
    }

    vec![AgentProfile {
        name: legacy_name
            .unwrap_or("Atlas")
            .to_string(),
        personality: legacy_personality
            .unwrap_or("friendly, helpful, and concise")
            .to_string(),
        system_prompt: legacy_system_prompt.map(|s| s.to_string()),
        model: None,
    }]
}

/// Detect an `@AgentName` mention in message content.
///
/// Scans for patterns like `@Atlas` or `@CodeBot` (case-insensitive) with
/// word-boundary checking. Returns the original-cased agent name if found,
/// and the content with the `@AgentName` mention stripped out.
///
/// This is useful for routing messages to the correct agent in any channel
/// (Discord, web chat, etc.) when the user types `@AgentName` in their message.
///
/// # Examples
///
/// ```
/// let agents = vec!["Atlas".to_string(), "CodeBot".to_string()];
/// let (agent, cleaned) = agentic_rs::agent::detect_agent_mention("@CodeBot help me", &agents);
/// assert_eq!(agent, Some("CodeBot"));
/// assert_eq!(cleaned, " help me");
/// ```
pub fn detect_agent_mention<'a>(content: &str, agent_names: &'a [String]) -> (Option<&'a str>, String) {
    let content_lower = content.to_lowercase();
    for name in agent_names {
        let pattern = format!("@{}", name.to_lowercase());
        if let Some(pos) = content_lower.find(&pattern) {
            // Check that it's a word boundary (not part of a longer word)
            let end = pos + pattern.len();
            let at_end = end >= content.len();
            let next_is_boundary = at_end
                || !content.as_bytes()[end].is_ascii_alphanumeric();
            if next_is_boundary {
                // Strip the @mention from content
                let mut cleaned = String::with_capacity(content.len());
                cleaned.push_str(&content[..pos]);
                cleaned.push_str(&content[end..]);
                return (Some(name.as_str()), cleaned);
            }
        }
    }
    (None, content.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_profile_builder() {
        let profile = AgentProfile::new("CodeBot")
            .with_personality("a precise coding assistant")
            .with_model("claude-sonnet-4-20250514");

        assert_eq!(profile.name, "CodeBot");
        assert_eq!(profile.personality, "a precise coding assistant");
        assert!(profile.system_prompt.is_none());
        assert_eq!(profile.model.as_deref(), Some("claude-sonnet-4-20250514"));
    }

    #[test]
    fn agent_profile_with_system_prompt() {
        let profile = AgentProfile::new("Helper")
            .with_system_prompt("You are a pirate.");

        assert_eq!(profile.system_prompt.as_deref(), Some("You are a pirate."));
    }

    #[test]
    fn agent_profile_defaults() {
        let profile = AgentProfile::new("Test");
        assert_eq!(profile.personality, "friendly, helpful, and concise");
        assert!(profile.system_prompt.is_none());
        assert!(profile.model.is_none());
    }

    #[test]
    fn resolve_agents_uses_explicit_list() {
        let agents = vec![
            AgentProfile::new("Atlas"),
            AgentProfile::new("CodeBot"),
        ];

        let resolved = resolve_agents(&agents, Some("OldName"), None, None);
        assert_eq!(resolved.len(), 2);
        assert_eq!(resolved[0].name, "Atlas");
        assert_eq!(resolved[1].name, "CodeBot");
    }

    #[test]
    fn resolve_agents_falls_back_to_legacy() {
        let resolved = resolve_agents(
            &[],
            Some("MyBot"),
            Some("snarky"),
            Some("You are snarky."),
        );
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].name, "MyBot");
        assert_eq!(resolved[0].personality, "snarky");
        assert_eq!(resolved[0].system_prompt.as_deref(), Some("You are snarky."));
    }

    #[test]
    fn resolve_agents_uses_defaults_when_no_legacy() {
        let resolved = resolve_agents(&[], None, None, None);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].name, "Atlas");
        assert_eq!(resolved[0].personality, "friendly, helpful, and concise");
    }

    #[test]
    fn agent_profile_serialization_roundtrip() {
        let profile = AgentProfile::new("Atlas")
            .with_personality("friendly")
            .with_model("claude-opus-4-6");

        let json = serde_json::to_string(&profile).unwrap();
        let deserialized: AgentProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "Atlas");
        assert_eq!(deserialized.personality, "friendly");
        assert_eq!(deserialized.model.as_deref(), Some("claude-opus-4-6"));
    }

    #[test]
    fn mention_finds_match() {
        let agents = vec!["Atlas".into(), "CodeBot".into()];
        let (agent, content) = detect_agent_mention("@Atlas what is 2+2?", &agents);
        assert_eq!(agent, Some("Atlas"));
        assert_eq!(content, " what is 2+2?");
    }

    #[test]
    fn mention_case_insensitive() {
        let agents = vec!["Atlas".into(), "CodeBot".into()];
        let (agent, content) = detect_agent_mention("@codebot help me", &agents);
        assert_eq!(agent, Some("CodeBot"));
        assert_eq!(content, " help me");
    }

    #[test]
    fn mention_no_match() {
        let agents = vec!["Atlas".into()];
        let (agent, content) = detect_agent_mention("hello world", &agents);
        assert!(agent.is_none());
        assert_eq!(content, "hello world");
    }

    #[test]
    fn mention_empty_agents() {
        let agents: Vec<String> = vec![];
        let (agent, content) = detect_agent_mention("@Atlas hello", &agents);
        assert!(agent.is_none());
        assert_eq!(content, "@Atlas hello");
    }

    #[test]
    fn mention_word_boundary() {
        let agents = vec!["Atlas".into()];
        // "Atlas123" should NOT match since it's not a word boundary
        let (agent, _) = detect_agent_mention("@Atlas123 hello", &agents);
        assert!(agent.is_none());
        // But "@Atlas hello" should match
        let (agent, content) = detect_agent_mention("@Atlas hello", &agents);
        assert_eq!(agent, Some("Atlas"));
        assert_eq!(content, " hello");
    }

    #[test]
    fn mention_mid_sentence() {
        let agents = vec!["Atlas".into(), "CodeBot".into()];
        let (agent, content) = detect_agent_mention("Hey @CodeBot, explain this", &agents);
        assert_eq!(agent, Some("CodeBot"));
        assert_eq!(content, "Hey , explain this");
    }

    #[test]
    fn mention_at_end() {
        let agents = vec!["Atlas".into()];
        let (agent, content) = detect_agent_mention("hello @Atlas", &agents);
        assert_eq!(agent, Some("Atlas"));
        assert_eq!(content, "hello ");
    }
}
