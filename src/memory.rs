use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::namespace::Namespace;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A chunk of knowledge stored in memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub namespace: Namespace,
    pub content: String,
    pub tags: Vec<String>,
    pub embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl MemoryEntry {
    pub fn new(namespace: Namespace, content: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            namespace,
            content: content.into(),
            tags: Vec::new(),
            embedding: None,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// A search result from the memory store with a relevance score.
#[derive(Debug, Clone)]
pub struct MemorySearchResult {
    pub entry: MemoryEntry,
    pub score: f32,
}

// ---------------------------------------------------------------------------
// Embedding provider trait
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("embedding provider error: {0}")]
    Provider(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// Generates vector embeddings for text. Plug in OpenAI, a local model, etc.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Return the dimensionality of the embedding vectors this provider produces.
    fn dimensions(&self) -> usize;

    /// Embed a single piece of text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;

    /// Embed multiple texts in a single batch call (default impl calls embed() in a loop).
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Memory store trait
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("storage error: {0}")]
    Storage(String),

    #[error("embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
}

#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// Store a memory entry.
    async fn store(&self, entry: MemoryEntry) -> Result<String, MemoryError>;

    /// Retrieve a memory entry by ID.
    async fn get(&self, id: &str) -> Result<Option<MemoryEntry>, MemoryError>;

    /// Delete a memory entry by ID.
    async fn delete(&self, id: &str) -> Result<bool, MemoryError>;

    /// Search memories by semantic similarity to a query embedding.
    /// Returns results sorted by relevance (highest score first).
    async fn search(
        &self,
        namespace: &Namespace,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<MemorySearchResult>, MemoryError>;

    /// Full-text keyword search as a fallback when embeddings aren't available.
    async fn search_text(
        &self,
        namespace: &Namespace,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemorySearchResult>, MemoryError>;

    /// List all memories under a given namespace.
    async fn list(
        &self,
        namespace: &Namespace,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryError>;
}

// ---------------------------------------------------------------------------
// In-memory implementation
// ---------------------------------------------------------------------------

pub struct InMemoryMemoryStore {
    entries: Arc<RwLock<HashMap<String, MemoryEntry>>>,
}

impl InMemoryMemoryStore {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MemoryStore for InMemoryMemoryStore {
    async fn store(&self, entry: MemoryEntry) -> Result<String, MemoryError> {
        let id = entry.id.clone();
        let mut entries = self.entries.write().await;
        entries.insert(id.clone(), entry);
        Ok(id)
    }

    async fn get(&self, id: &str) -> Result<Option<MemoryEntry>, MemoryError> {
        let entries = self.entries.read().await;
        Ok(entries.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<bool, MemoryError> {
        let mut entries = self.entries.write().await;
        Ok(entries.remove(id).is_some())
    }

    async fn search(
        &self,
        namespace: &Namespace,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<MemorySearchResult>, MemoryError> {
        let entries = self.entries.read().await;

        let mut scored: Vec<MemorySearchResult> = entries
            .values()
            .filter(|e| e.namespace == *namespace || namespace.is_ancestor_of(&e.namespace))
            .filter_map(|entry| {
                let embedding = entry.embedding.as_ref()?;
                let score = cosine_similarity(query_embedding, embedding);
                Some(MemorySearchResult {
                    entry: entry.clone(),
                    score,
                })
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        Ok(scored)
    }

    async fn search_text(
        &self,
        namespace: &Namespace,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemorySearchResult>, MemoryError> {
        let entries = self.entries.read().await;
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<MemorySearchResult> = entries
            .values()
            .filter(|e| e.namespace == *namespace || namespace.is_ancestor_of(&e.namespace))
            .filter_map(|entry| {
                let content_lower = entry.content.to_lowercase();
                let tag_text: String = entry.tags.join(" ").to_lowercase();

                // Simple TF scoring: count how many query terms appear
                let hits: usize = query_terms
                    .iter()
                    .filter(|term| content_lower.contains(**term) || tag_text.contains(**term))
                    .count();

                if hits == 0 {
                    return None;
                }

                let score = hits as f32 / query_terms.len() as f32;
                Some(MemorySearchResult {
                    entry: entry.clone(),
                    score,
                })
            })
            .collect();

        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        Ok(scored)
    }

    async fn list(
        &self,
        namespace: &Namespace,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryError> {
        let entries = self.entries.read().await;

        let mut matched: Vec<&MemoryEntry> = entries
            .values()
            .filter(|e| e.namespace == *namespace || namespace.is_ancestor_of(&e.namespace))
            .collect();

        matched.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        matched.truncate(limit);

        Ok(matched.into_iter().cloned().collect())
    }
}

// ---------------------------------------------------------------------------
// Memory manager
// ---------------------------------------------------------------------------

/// High-level memory manager that combines a store with an embedding provider.
/// Handles automatic embedding generation on store and query operations.
pub struct MemoryManager {
    store: Arc<dyn MemoryStore>,
    embedder: Option<Arc<dyn EmbeddingProvider>>,
}

impl MemoryManager {
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        Self {
            store,
            embedder: None,
        }
    }

    pub fn with_embedder(mut self, embedder: Arc<dyn EmbeddingProvider>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Store a piece of knowledge. If an embedding provider is configured,
    /// the content is automatically embedded before storage.
    pub async fn remember(
        &self,
        namespace: &Namespace,
        content: impl Into<String>,
        tags: Vec<String>,
    ) -> Result<String, MemoryError> {
        let content = content.into();
        let mut entry = MemoryEntry::new(namespace.clone(), &content).with_tags(tags);

        if let Some(ref embedder) = self.embedder {
            let embedding = embedder.embed(&content).await?;
            entry = entry.with_embedding(embedding);
        }

        self.store.store(entry).await
    }

    /// Search for relevant memories. Uses semantic search if embeddings are
    /// available, falls back to keyword search otherwise.
    pub async fn recall(
        &self,
        namespace: &Namespace,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemorySearchResult>, MemoryError> {
        if let Some(ref embedder) = self.embedder {
            let query_embedding = embedder.embed(query).await?;
            self.store.search(namespace, &query_embedding, limit).await
        } else {
            self.store.search_text(namespace, query, limit).await
        }
    }

    /// Delete a memory by ID.
    pub async fn forget(&self, id: &str) -> Result<bool, MemoryError> {
        self.store.delete(id).await
    }

    /// List recent memories for a namespace.
    pub async fn list(
        &self,
        namespace: &Namespace,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryError> {
        self.store.list(namespace, limit).await
    }

    /// Get the underlying store.
    pub fn store(&self) -> &Arc<dyn MemoryStore> {
        &self.store
    }
}

// ---------------------------------------------------------------------------
// Cosine similarity
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
    (dot / denom) as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_empty_vectors() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn cosine_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn memory_entry_builder() {
        let ns = Namespace::new("test");
        let entry = MemoryEntry::new(ns.clone(), "fact about cats")
            .with_tags(vec!["animals".into(), "cats".into()])
            .with_embedding(vec![0.1, 0.2, 0.3])
            .with_metadata("source", serde_json::json!("user"));

        assert_eq!(entry.content, "fact about cats");
        assert_eq!(entry.tags.len(), 2);
        assert!(entry.embedding.is_some());
        assert_eq!(entry.metadata["source"], "user");
    }

    #[tokio::test]
    async fn in_memory_store_roundtrip() {
        let store = InMemoryMemoryStore::new();
        let ns = Namespace::new("test");

        let entry = MemoryEntry::new(ns.clone(), "cats are great");
        let id = store.store(entry).await.unwrap();

        let loaded = store.get(&id).await.unwrap().unwrap();
        assert_eq!(loaded.content, "cats are great");
    }

    #[tokio::test]
    async fn in_memory_store_delete() {
        let store = InMemoryMemoryStore::new();
        let ns = Namespace::new("test");

        let entry = MemoryEntry::new(ns, "temporary");
        let id = store.store(entry).await.unwrap();

        assert!(store.delete(&id).await.unwrap());
        assert!(!store.delete(&id).await.unwrap());
        assert!(store.get(&id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn in_memory_store_text_search() {
        let store = InMemoryMemoryStore::new();
        let ns = Namespace::new("test");

        store
            .store(MemoryEntry::new(ns.clone(), "cats are fluffy animals"))
            .await
            .unwrap();
        store
            .store(MemoryEntry::new(ns.clone(), "dogs are loyal friends"))
            .await
            .unwrap();
        store
            .store(MemoryEntry::new(ns.clone(), "python is a programming language"))
            .await
            .unwrap();

        let results = store.search_text(&ns, "fluffy cats", 10).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].entry.content.contains("cats"));
    }

    #[tokio::test]
    async fn in_memory_store_vector_search() {
        let store = InMemoryMemoryStore::new();
        let ns = Namespace::new("test");

        store
            .store(
                MemoryEntry::new(ns.clone(), "cats are fluffy")
                    .with_embedding(vec![0.9, 0.1, 0.0]),
            )
            .await
            .unwrap();
        store
            .store(
                MemoryEntry::new(ns.clone(), "dogs are loyal")
                    .with_embedding(vec![0.1, 0.9, 0.0]),
            )
            .await
            .unwrap();
        store
            .store(
                MemoryEntry::new(ns.clone(), "python is great")
                    .with_embedding(vec![0.0, 0.0, 1.0]),
            )
            .await
            .unwrap();

        // Query close to "cats"
        let results = store
            .search(&ns, &[0.8, 0.2, 0.0], 2)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].entry.content.contains("cats"));
        assert!(results[0].score > results[1].score);
    }

    #[tokio::test]
    async fn in_memory_store_namespace_filtering() {
        let store = InMemoryMemoryStore::new();

        let ns_alice = Namespace::new("users").child("alice");
        let ns_bob = Namespace::new("users").child("bob");

        store
            .store(MemoryEntry::new(ns_alice.clone(), "alice's memory"))
            .await
            .unwrap();
        store
            .store(MemoryEntry::new(ns_bob.clone(), "bob's memory"))
            .await
            .unwrap();

        let alice_results = store.search_text(&ns_alice, "memory", 10).await.unwrap();
        assert_eq!(alice_results.len(), 1);
        assert!(alice_results[0].entry.content.contains("alice"));
    }

    #[tokio::test]
    async fn in_memory_store_ancestor_namespace_sees_children() {
        let store = InMemoryMemoryStore::new();

        let parent = Namespace::new("org");
        let child = parent.child("team");

        store
            .store(MemoryEntry::new(child.clone(), "team memory"))
            .await
            .unwrap();

        // Parent namespace should be able to see children's memories
        let results = store.search_text(&parent, "memory", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn in_memory_store_list_ordered_by_recency() {
        let store = InMemoryMemoryStore::new();
        let ns = Namespace::new("test");

        store
            .store(MemoryEntry::new(ns.clone(), "first"))
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        store
            .store(MemoryEntry::new(ns.clone(), "second"))
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        store
            .store(MemoryEntry::new(ns.clone(), "third"))
            .await
            .unwrap();

        let entries = store.list(&ns, 10).await.unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].content, "third");
        assert_eq!(entries[2].content, "first");
    }

    #[tokio::test]
    async fn in_memory_store_list_respects_limit() {
        let store = InMemoryMemoryStore::new();
        let ns = Namespace::new("test");

        for i in 0..10 {
            store
                .store(MemoryEntry::new(ns.clone(), format!("entry {}", i)))
                .await
                .unwrap();
        }

        let entries = store.list(&ns, 3).await.unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[tokio::test]
    async fn memory_manager_remember_and_recall_text() {
        let store = Arc::new(InMemoryMemoryStore::new());
        let manager = MemoryManager::new(store);

        let ns = Namespace::new("test");
        manager
            .remember(&ns, "the capital of France is Paris", vec!["geography".into()])
            .await
            .unwrap();
        manager
            .remember(&ns, "rust is a systems programming language", vec!["programming".into()])
            .await
            .unwrap();

        let results = manager.recall(&ns, "France capital", 5).await.unwrap();
        assert!(!results.is_empty());
        assert!(results[0].entry.content.contains("Paris"));
    }

    #[tokio::test]
    async fn memory_manager_forget() {
        let store = Arc::new(InMemoryMemoryStore::new());
        let manager = MemoryManager::new(store);

        let ns = Namespace::new("test");
        let id = manager
            .remember(&ns, "temporary fact", vec![])
            .await
            .unwrap();

        assert!(manager.forget(&id).await.unwrap());

        let results = manager.recall(&ns, "temporary", 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn memory_entry_serialization_roundtrip() {
        let entry = MemoryEntry::new(Namespace::new("test"), "some knowledge")
            .with_tags(vec!["tag1".into()])
            .with_embedding(vec![0.1, 0.2, 0.3]);

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: MemoryEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.content, entry.content);
        assert_eq!(deserialized.tags, entry.tags);
        assert_eq!(deserialized.embedding, entry.embedding);
    }
}
