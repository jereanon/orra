use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::message::Message;
use crate::namespace::Namespace;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub namespace: Namespace,
    pub messages: Vec<Message>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Session {
    pub fn new(namespace: Namespace) -> Self {
        let now = Utc::now();
        Self {
            namespace,
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    pub fn push_message(&mut self, message: Message) {
        self.messages.push(message);
        self.updated_at = Utc::now();
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("session not found: {0}")]
    NotFound(String),

    #[error("storage error: {0}")]
    Storage(String),
}

#[async_trait]
pub trait SessionStore: Send + Sync {
    async fn load(&self, namespace: &Namespace) -> Result<Option<Session>, StoreError>;
    async fn save(&self, session: &Session) -> Result<(), StoreError>;
    async fn delete(&self, namespace: &Namespace) -> Result<bool, StoreError>;
    async fn list(&self, prefix: Option<&Namespace>) -> Result<Vec<Namespace>, StoreError>;
}

#[derive(Clone)]
pub struct InMemoryStore {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SessionStore for InMemoryStore {
    async fn load(&self, namespace: &Namespace) -> Result<Option<Session>, StoreError> {
        let sessions = self.sessions.read().await;
        Ok(sessions.get(&namespace.key()).cloned())
    }

    async fn save(&self, session: &Session) -> Result<(), StoreError> {
        let mut sessions = self.sessions.write().await;
        sessions.insert(session.namespace.key(), session.clone());
        Ok(())
    }

    async fn delete(&self, namespace: &Namespace) -> Result<bool, StoreError> {
        let mut sessions = self.sessions.write().await;
        Ok(sessions.remove(&namespace.key()).is_some())
    }

    async fn list(&self, prefix: Option<&Namespace>) -> Result<Vec<Namespace>, StoreError> {
        let sessions = self.sessions.read().await;
        let namespaces: Vec<Namespace> = sessions
            .keys()
            .filter(|key| {
                if let Some(prefix) = prefix {
                    let prefix_key = prefix.key();
                    key.starts_with(&prefix_key) && (key.len() == prefix_key.len() || key[prefix_key.len()..].starts_with(':'))
                } else {
                    true
                }
            })
            .map(|key| Namespace::parse(key))
            .collect();
        Ok(namespaces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_creation() {
        let ns = Namespace::new("acme").child("alice");
        let session = Session::new(ns.clone());
        assert_eq!(session.namespace, ns);
        assert_eq!(session.message_count(), 0);
        assert!(session.messages.is_empty());
    }

    #[test]
    fn session_push_message() {
        let ns = Namespace::new("acme");
        let mut session = Session::new(ns);
        let before = session.updated_at;

        std::thread::sleep(std::time::Duration::from_millis(10));
        session.push_message(Message::user("Hello"));

        assert_eq!(session.message_count(), 1);
        assert!(session.updated_at >= before);
    }

    #[test]
    fn session_metadata() {
        let ns = Namespace::new("acme");
        let mut session = Session::new(ns);
        session
            .metadata
            .insert("model".into(), serde_json::json!("claude-4"));
        assert_eq!(session.metadata["model"], "claude-4");
    }

    #[tokio::test]
    async fn in_memory_store_save_and_load() {
        let store = InMemoryStore::new();
        let ns = Namespace::new("acme").child("alice");

        // Not found initially
        let loaded = store.load(&ns).await.unwrap();
        assert!(loaded.is_none());

        // Save
        let mut session = Session::new(ns.clone());
        session.push_message(Message::user("Hello"));
        store.save(&session).await.unwrap();

        // Load
        let loaded = store.load(&ns).await.unwrap().unwrap();
        assert_eq!(loaded.message_count(), 1);
        assert_eq!(loaded.messages[0].content, "Hello");
    }

    #[tokio::test]
    async fn in_memory_store_delete() {
        let store = InMemoryStore::new();
        let ns = Namespace::new("acme");

        // Delete nonexistent
        let deleted = store.delete(&ns).await.unwrap();
        assert!(!deleted);

        // Save then delete
        store.save(&Session::new(ns.clone())).await.unwrap();
        let deleted = store.delete(&ns).await.unwrap();
        assert!(deleted);

        let loaded = store.load(&ns).await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn in_memory_store_list_all() {
        let store = InMemoryStore::new();

        store.save(&Session::new(Namespace::new("acme").child("alice"))).await.unwrap();
        store.save(&Session::new(Namespace::new("acme").child("bob"))).await.unwrap();
        store.save(&Session::new(Namespace::new("other"))).await.unwrap();

        let all = store.list(None).await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn in_memory_store_list_with_prefix() {
        let store = InMemoryStore::new();

        store.save(&Session::new(Namespace::new("acme").child("alice"))).await.unwrap();
        store.save(&Session::new(Namespace::new("acme").child("bob"))).await.unwrap();
        store.save(&Session::new(Namespace::new("other"))).await.unwrap();

        let acme = Namespace::new("acme");
        let filtered = store.list(Some(&acme)).await.unwrap();
        assert_eq!(filtered.len(), 2);

        for ns in &filtered {
            assert!(ns.key().starts_with("acme:"));
        }
    }

    #[tokio::test]
    async fn in_memory_store_list_prefix_no_partial_match() {
        let store = InMemoryStore::new();

        store.save(&Session::new(Namespace::parse("acme:alice"))).await.unwrap();
        store.save(&Session::new(Namespace::parse("acmeother:bob"))).await.unwrap();

        let acme = Namespace::new("acme");
        let filtered = store.list(Some(&acme)).await.unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].key(), "acme:alice");
    }

    #[tokio::test]
    async fn in_memory_store_overwrite() {
        let store = InMemoryStore::new();
        let ns = Namespace::new("acme");

        let mut session = Session::new(ns.clone());
        session.push_message(Message::user("First"));
        store.save(&session).await.unwrap();

        session.push_message(Message::user("Second"));
        store.save(&session).await.unwrap();

        let loaded = store.load(&ns).await.unwrap().unwrap();
        assert_eq!(loaded.message_count(), 2);
    }

    #[test]
    fn session_serialization_roundtrip() {
        let ns = Namespace::new("acme").child("alice");
        let mut session = Session::new(ns);
        session.push_message(Message::user("Hello"));
        session.metadata.insert("key".into(), serde_json::json!("value"));

        let json = serde_json::to_string(&session).unwrap();
        let deserialized: Session = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.namespace, session.namespace);
        assert_eq!(deserialized.message_count(), 1);
        assert_eq!(deserialized.metadata["key"], "value");
    }
}
