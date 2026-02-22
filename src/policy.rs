use std::collections::HashMap;

use crate::namespace::Namespace;
use crate::tool::ToolDefinition;

#[derive(Debug, Clone, Default)]
pub enum ToolPolicy {
    #[default]
    AllowAll,
    DenyAll,
    AllowList(Vec<String>),
    DenyList(Vec<String>),
}

impl ToolPolicy {
    pub fn is_allowed(&self, tool_name: &str) -> bool {
        match self {
            ToolPolicy::AllowAll => true,
            ToolPolicy::DenyAll => false,
            ToolPolicy::AllowList(names) => names.iter().any(|n| n == tool_name),
            ToolPolicy::DenyList(names) => !names.iter().any(|n| n == tool_name),
        }
    }

    pub fn filter_definitions(&self, definitions: &[ToolDefinition]) -> Vec<ToolDefinition> {
        definitions
            .iter()
            .filter(|d| self.is_allowed(&d.name))
            .cloned()
            .collect()
    }
}

pub struct PolicyRegistry {
    policies: HashMap<String, ToolPolicy>,
    default_policy: ToolPolicy,
}

impl PolicyRegistry {
    pub fn new(default_policy: ToolPolicy) -> Self {
        Self {
            policies: HashMap::new(),
            default_policy,
        }
    }

    pub fn set_policy(&mut self, namespace: &Namespace, policy: ToolPolicy) {
        self.policies.insert(namespace.key(), policy);
    }

    pub fn remove_policy(&mut self, namespace: &Namespace) -> bool {
        self.policies.remove(&namespace.key()).is_some()
    }

    /// Resolve the effective policy for a namespace by walking up the hierarchy.
    /// The most specific (deepest) policy wins. Falls back to default if none found.
    pub fn resolve(&self, namespace: &Namespace) -> &ToolPolicy {
        // Check this namespace first
        if let Some(policy) = self.policies.get(&namespace.key()) {
            return policy;
        }

        // Walk up parents
        let mut current = namespace.parent();
        while let Some(ns) = current {
            if let Some(policy) = self.policies.get(&ns.key()) {
                return policy;
            }
            current = ns.parent();
        }

        &self.default_policy
    }
}

impl Default for PolicyRegistry {
    fn default() -> Self {
        Self::new(ToolPolicy::AllowAll)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_all_policy() {
        let policy = ToolPolicy::AllowAll;
        assert!(policy.is_allowed("anything"));
        assert!(policy.is_allowed("search"));
    }

    #[test]
    fn deny_all_policy() {
        let policy = ToolPolicy::DenyAll;
        assert!(!policy.is_allowed("anything"));
    }

    #[test]
    fn allow_list_policy() {
        let policy = ToolPolicy::AllowList(vec!["search".into(), "read".into()]);
        assert!(policy.is_allowed("search"));
        assert!(policy.is_allowed("read"));
        assert!(!policy.is_allowed("delete"));
    }

    #[test]
    fn deny_list_policy() {
        let policy = ToolPolicy::DenyList(vec!["delete".into(), "admin".into()]);
        assert!(policy.is_allowed("search"));
        assert!(policy.is_allowed("read"));
        assert!(!policy.is_allowed("delete"));
        assert!(!policy.is_allowed("admin"));
    }

    #[test]
    fn filter_definitions() {
        let defs = vec![
            ToolDefinition {
                name: "search".into(),
                description: "Search".into(),
                input_schema: serde_json::json!({}),
            },
            ToolDefinition {
                name: "delete".into(),
                description: "Delete".into(),
                input_schema: serde_json::json!({}),
            },
            ToolDefinition {
                name: "read".into(),
                description: "Read".into(),
                input_schema: serde_json::json!({}),
            },
        ];

        let policy = ToolPolicy::AllowList(vec!["search".into(), "read".into()]);
        let filtered = policy.filter_definitions(&defs);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|d| d.name != "delete"));
    }

    #[test]
    fn policy_registry_default() {
        let registry = PolicyRegistry::default();
        let ns = Namespace::new("anything");
        assert!(matches!(registry.resolve(&ns), ToolPolicy::AllowAll));
    }

    #[test]
    fn policy_registry_exact_match() {
        let mut registry = PolicyRegistry::default();
        let ns = Namespace::new("acme").child("alice");
        registry.set_policy(&ns, ToolPolicy::DenyAll);

        assert!(matches!(registry.resolve(&ns), ToolPolicy::DenyAll));
    }

    #[test]
    fn policy_registry_inherits_from_parent() {
        let mut registry = PolicyRegistry::default();
        let parent = Namespace::new("acme");
        let child = parent.child("alice");
        let grandchild = child.child("support");

        registry.set_policy(&parent, ToolPolicy::DenyList(vec!["admin".into()]));

        // Child and grandchild inherit from parent
        match registry.resolve(&child) {
            ToolPolicy::DenyList(names) => assert_eq!(names, &["admin"]),
            _ => panic!("expected DenyList"),
        }

        match registry.resolve(&grandchild) {
            ToolPolicy::DenyList(names) => assert_eq!(names, &["admin"]),
            _ => panic!("expected DenyList"),
        }
    }

    #[test]
    fn policy_registry_child_overrides_parent() {
        let mut registry = PolicyRegistry::default();
        let parent = Namespace::new("acme");
        let child = parent.child("alice");

        registry.set_policy(&parent, ToolPolicy::DenyAll);
        registry.set_policy(&child, ToolPolicy::AllowAll);

        // Parent has DenyAll
        assert!(matches!(registry.resolve(&parent), ToolPolicy::DenyAll));
        // Child overrides to AllowAll
        assert!(matches!(registry.resolve(&child), ToolPolicy::AllowAll));
    }

    #[test]
    fn policy_registry_remove() {
        let mut registry = PolicyRegistry::default();
        let ns = Namespace::new("acme");
        registry.set_policy(&ns, ToolPolicy::DenyAll);
        assert!(matches!(registry.resolve(&ns), ToolPolicy::DenyAll));

        assert!(registry.remove_policy(&ns));
        assert!(matches!(registry.resolve(&ns), ToolPolicy::AllowAll));

        assert!(!registry.remove_policy(&ns)); // already removed
    }

    #[test]
    fn policy_registry_unrelated_namespaces_isolated() {
        let mut registry = PolicyRegistry::default();
        let acme = Namespace::new("acme");
        let other = Namespace::new("other");

        registry.set_policy(&acme, ToolPolicy::DenyAll);

        assert!(matches!(registry.resolve(&acme), ToolPolicy::DenyAll));
        assert!(matches!(registry.resolve(&other), ToolPolicy::AllowAll));
    }
}
