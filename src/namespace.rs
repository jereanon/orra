use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Namespace {
    segments: Vec<String>,
}

impl Namespace {
    pub fn new(root: impl Into<String>) -> Self {
        Self {
            segments: vec![root.into()],
        }
    }

    pub fn child(&self, segment: impl Into<String>) -> Self {
        let mut segments = self.segments.clone();
        segments.push(segment.into());
        Self { segments }
    }

    pub fn parent(&self) -> Option<Self> {
        if self.segments.len() <= 1 {
            return None;
        }
        let mut segments = self.segments.clone();
        segments.pop();
        Some(Self { segments })
    }

    pub fn root(&self) -> &str {
        &self.segments[0]
    }

    pub fn depth(&self) -> usize {
        self.segments.len()
    }

    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    pub fn is_ancestor_of(&self, other: &Namespace) -> bool {
        if self.segments.len() >= other.segments.len() {
            return false;
        }
        other.segments.starts_with(&self.segments)
    }

    pub fn is_descendant_of(&self, other: &Namespace) -> bool {
        other.is_ancestor_of(self)
    }

    pub fn parse(key: &str) -> Self {
        let segments: Vec<String> = key.split(':').map(|s| s.to_string()).collect();
        assert!(!segments.is_empty(), "namespace key cannot be empty");
        Self { segments }
    }

    pub fn key(&self) -> String {
        self.segments.join(":")
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.key())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_root_namespace() {
        let ns = Namespace::new("tenant:acme");
        assert_eq!(ns.root(), "tenant:acme");
        assert_eq!(ns.depth(), 1);
        assert_eq!(ns.key(), "tenant:acme");
    }

    #[test]
    fn create_child_namespace() {
        let root = Namespace::new("acme");
        let child = root.child("user:alice");
        assert_eq!(child.depth(), 2);
        assert_eq!(child.key(), "acme:user:alice");
    }

    #[test]
    fn nested_children() {
        let ns = Namespace::new("acme").child("alice").child("support");
        assert_eq!(ns.depth(), 3);
        assert_eq!(ns.key(), "acme:alice:support");
    }

    #[test]
    fn parent_of_child() {
        let ns = Namespace::new("acme").child("alice").child("support");
        let parent = ns.parent().unwrap();
        assert_eq!(parent.key(), "acme:alice");
        let grandparent = parent.parent().unwrap();
        assert_eq!(grandparent.key(), "acme");
        assert!(grandparent.parent().is_none());
    }

    #[test]
    fn root_has_no_parent() {
        let ns = Namespace::new("acme");
        assert!(ns.parent().is_none());
    }

    #[test]
    fn ancestor_checks() {
        let root = Namespace::new("acme");
        let child = root.child("alice");
        let grandchild = child.child("support");

        assert!(root.is_ancestor_of(&child));
        assert!(root.is_ancestor_of(&grandchild));
        assert!(child.is_ancestor_of(&grandchild));

        assert!(!child.is_ancestor_of(&root));
        assert!(!root.is_ancestor_of(&root));
        assert!(!grandchild.is_ancestor_of(&child));
    }

    #[test]
    fn descendant_checks() {
        let root = Namespace::new("acme");
        let child = root.child("alice");
        let grandchild = child.child("support");

        assert!(child.is_descendant_of(&root));
        assert!(grandchild.is_descendant_of(&root));
        assert!(grandchild.is_descendant_of(&child));

        assert!(!root.is_descendant_of(&child));
    }

    #[test]
    fn parse_namespace_key() {
        let ns = Namespace::parse("acme:alice:support");
        assert_eq!(ns.depth(), 3);
        assert_eq!(ns.segments(), &["acme", "alice", "support"]);
    }

    #[test]
    fn display_trait() {
        let ns = Namespace::new("acme").child("alice");
        assert_eq!(format!("{ns}"), "acme:alice");
    }

    #[test]
    fn equality_and_hashing() {
        use std::collections::HashSet;

        let a = Namespace::new("acme").child("alice");
        let b = Namespace::parse("acme:alice");
        assert_eq!(a, b);

        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }

    #[test]
    fn serialization_roundtrip() {
        let ns = Namespace::new("acme").child("alice");
        let json = serde_json::to_string(&ns).unwrap();
        let deserialized: Namespace = serde_json::from_str(&json).unwrap();
        assert_eq!(ns, deserialized);
    }
}
