use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A document stored in the document store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for this document.
    pub id: String,
    /// Human-readable title.
    pub title: String,
    /// The full text content of the document.
    pub content: String,
    /// Arbitrary key-value metadata (source path, author, date, etc.).
    pub metadata: HashMap<String, String>,
}

/// A chunk of a document returned from a search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The document this result came from.
    pub document_id: String,
    /// Title of the source document.
    pub title: String,
    /// The relevant text chunk.
    pub chunk: String,
    /// Relevance score (higher is better, scale is implementation-defined).
    pub score: f64,
}

// ---------------------------------------------------------------------------
// DocumentStore trait
// ---------------------------------------------------------------------------

/// Backend for storing and searching documents.
///
/// Implementations might use a simple in-memory TF-IDF index, a vector
/// database like Qdrant or pgvector, or even a full-text search engine.
/// The library ships with `InMemoryDocumentStore` for prototyping.
#[async_trait]
pub trait DocumentStore: Send + Sync {
    /// Add or replace a document in the store.
    async fn upsert(&self, doc: Document) -> Result<(), DocumentStoreError>;

    /// Remove a document by ID.
    async fn remove(&self, id: &str) -> Result<bool, DocumentStoreError>;

    /// Get a document by ID.
    async fn get(&self, id: &str) -> Result<Option<Document>, DocumentStoreError>;

    /// List all documents (just id + title + metadata, no content).
    async fn list(&self) -> Result<Vec<DocumentSummary>, DocumentStoreError>;

    /// Search for documents matching a text query. Returns the top `limit`
    /// results ranked by relevance.
    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, DocumentStoreError>;
}

/// Lightweight summary returned by `list()` — avoids loading full content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSummary {
    pub id: String,
    pub title: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, thiserror::Error)]
pub enum DocumentStoreError {
    #[error("document not found: {0}")]
    NotFound(String),

    #[error("store error: {0}")]
    Internal(String),
}

// ---------------------------------------------------------------------------
// InMemoryDocumentStore — TF-IDF search for prototyping
// ---------------------------------------------------------------------------

/// A simple in-memory document store with TF-IDF text search.
///
/// Good enough for prototyping and small document sets (hundreds of docs).
/// For production workloads with large corpora, use a vector database or
/// full-text search engine behind the `DocumentStore` trait.
pub struct InMemoryDocumentStore {
    docs: Arc<RwLock<HashMap<String, Document>>>,
}

impl InMemoryDocumentStore {
    pub fn new() -> Self {
        Self {
            docs: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryDocumentStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentStore for InMemoryDocumentStore {
    async fn upsert(&self, doc: Document) -> Result<(), DocumentStoreError> {
        let mut docs = self.docs.write().await;
        docs.insert(doc.id.clone(), doc);
        Ok(())
    }

    async fn remove(&self, id: &str) -> Result<bool, DocumentStoreError> {
        let mut docs = self.docs.write().await;
        Ok(docs.remove(id).is_some())
    }

    async fn get(&self, id: &str) -> Result<Option<Document>, DocumentStoreError> {
        let docs = self.docs.read().await;
        Ok(docs.get(id).cloned())
    }

    async fn list(&self) -> Result<Vec<DocumentSummary>, DocumentStoreError> {
        let docs = self.docs.read().await;
        let mut summaries: Vec<DocumentSummary> = docs
            .values()
            .map(|d| DocumentSummary {
                id: d.id.clone(),
                title: d.title.clone(),
                metadata: d.metadata.clone(),
            })
            .collect();
        summaries.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(summaries)
    }

    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, DocumentStoreError> {
        let docs = self.docs.read().await;
        if docs.is_empty() {
            return Ok(Vec::new());
        }

        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        let doc_count = docs.len() as f64;

        // Build document frequency for each query term
        let mut df: HashMap<&str, usize> = HashMap::new();
        for term in &query_terms {
            let count = docs
                .values()
                .filter(|d| {
                    let text = format!("{} {}", d.title, d.content).to_lowercase();
                    text.contains(term.as_str())
                })
                .count();
            df.insert(term, count);
        }

        // Score each document using TF-IDF
        let mut scored: Vec<(String, f64, String)> = Vec::new();
        for doc in docs.values() {
            let text = format!("{} {}", doc.title, doc.content).to_lowercase();
            let doc_terms = tokenize(&text);
            let doc_len = doc_terms.len() as f64;
            if doc_len == 0.0 {
                continue;
            }

            let mut score = 0.0;
            for term in &query_terms {
                let tf = doc_terms.iter().filter(|t| t == &term).count() as f64 / doc_len;
                let doc_freq = *df.get(term.as_str()).unwrap_or(&0) as f64;
                if doc_freq > 0.0 {
                    let idf = (doc_count / doc_freq).ln() + 1.0;
                    score += tf * idf;
                }
            }

            if score > 0.0 {
                // Extract a relevant chunk around the first match
                let chunk = extract_chunk(&doc.content, &query_terms, 500);
                scored.push((doc.id.clone(), score, chunk));
            }
        }

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        let results = scored
            .into_iter()
            .map(|(id, score, chunk)| {
                let doc = docs.get(&id).unwrap();
                SearchResult {
                    document_id: id,
                    title: doc.title.clone(),
                    chunk,
                    score,
                }
            })
            .collect();

        Ok(results)
    }
}

/// Split text into lowercase terms, filtering out very short words.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_string())
        .collect()
}

/// Extract a chunk of text around the first occurrence of any query term.
/// Returns up to `max_chars` characters centered on the match.
fn extract_chunk(content: &str, query_terms: &[String], max_chars: usize) -> String {
    if content.len() <= max_chars {
        return content.to_string();
    }

    let lower = content.to_lowercase();

    // Find the earliest match position
    let match_pos = query_terms
        .iter()
        .filter_map(|term| lower.find(term.as_str()))
        .min()
        .unwrap_or(0);

    let half = max_chars / 2;
    let start = match_pos.saturating_sub(half);
    let end = (start + max_chars).min(content.len());
    let start = if end == content.len() {
        end.saturating_sub(max_chars)
    } else {
        start
    };

    // Snap to word boundaries
    let start = if start > 0 {
        content[start..]
            .find(char::is_whitespace)
            .map(|i| start + i + 1)
            .unwrap_or(start)
    } else {
        0
    };

    let end = if end < content.len() {
        content[..end].rfind(char::is_whitespace).unwrap_or(end)
    } else {
        end
    };

    let mut chunk = content[start..end].to_string();
    if start > 0 {
        chunk.insert_str(0, "...");
    }
    if end < content.len() {
        chunk.push_str("...");
    }

    chunk
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

pub struct SearchDocumentsTool {
    store: Arc<dyn DocumentStore>,
}

impl SearchDocumentsTool {
    pub fn new(store: Arc<dyn DocumentStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for SearchDocumentsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_documents".into(),
            description: "Search documents by keyword or phrase. Returns the most relevant chunks ranked by relevance.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query — keywords or a natural language question"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results to return. Defaults to 5."
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'query'".into()))?;

        let limit = input.get("limit").and_then(|v| v.as_u64()).unwrap_or(5) as usize;

        let results = self
            .store
            .search(query, limit)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if results.is_empty() {
            return Ok(format!("No documents found matching '{query}'."));
        }

        let mut lines = Vec::new();
        for (i, r) in results.iter().enumerate() {
            lines.push(format!(
                "{}. [{}] {} (score: {:.3})\n{}",
                i + 1,
                r.document_id,
                r.title,
                r.score,
                r.chunk,
            ));
        }

        Ok(lines.join("\n\n"))
    }
}

pub struct ReadDocumentTool {
    store: Arc<dyn DocumentStore>,
}

impl ReadDocumentTool {
    pub fn new(store: Arc<dyn DocumentStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for ReadDocumentTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_document".into(),
            description: "Read the full content of a document by its ID.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The document ID"
                    }
                },
                "required": ["id"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let id = input
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'id'".into()))?;

        let doc = self
            .store
            .get(id)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?
            .ok_or_else(|| ToolError::ExecutionFailed(format!("document '{id}' not found")))?;

        let mut meta_lines: Vec<String> = doc
            .metadata
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect();
        meta_lines.sort();

        let header = if meta_lines.is_empty() {
            format!("# {}\n(id: {})", doc.title, doc.id)
        } else {
            format!(
                "# {}\n(id: {})\n{}",
                doc.title,
                doc.id,
                meta_lines.join("\n")
            )
        };

        Ok(format!("{}\n\n{}", header, doc.content))
    }
}

pub struct ListDocumentsTool {
    store: Arc<dyn DocumentStore>,
}

impl ListDocumentsTool {
    pub fn new(store: Arc<dyn DocumentStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for ListDocumentsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_documents".into(),
            description: "List all available documents with their IDs, titles, and metadata."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    async fn execute(&self, _input: serde_json::Value) -> Result<String, ToolError> {
        let summaries = self
            .store
            .list()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if summaries.is_empty() {
            return Ok("No documents in the store.".into());
        }

        let mut lines = Vec::new();
        for s in &summaries {
            let meta = if s.metadata.is_empty() {
                String::new()
            } else {
                let pairs: Vec<String> =
                    s.metadata.iter().map(|(k, v)| format!("{k}={v}")).collect();
                format!(" ({})", pairs.join(", "))
            };
            lines.push(format!("- [{}] {}{}", s.id, s.title, meta));
        }

        Ok(lines.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// Convenience registration
// ---------------------------------------------------------------------------

/// Register all document retrieval tools into a ToolRegistry.
pub fn register_tools(registry: &mut ToolRegistry, store: Arc<dyn DocumentStore>) {
    registry.register(Box::new(SearchDocumentsTool::new(Arc::clone(&store))));
    registry.register(Box::new(ReadDocumentTool::new(Arc::clone(&store))));
    registry.register(Box::new(ListDocumentsTool::new(store)));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_docs() -> Vec<Document> {
        vec![
            Document {
                id: "handbook".into(),
                title: "Employee Handbook".into(),
                content: "All employees are entitled to 15 days of paid vacation per year. \
                          Vacation requests must be submitted at least two weeks in advance. \
                          Unused vacation days do not roll over to the next year."
                    .into(),
                metadata: HashMap::from([
                    ("source".into(), "hr/handbook.pdf".into()),
                    ("updated".into(), "2024-01-15".into()),
                ]),
            },
            Document {
                id: "security".into(),
                title: "Security Policy".into(),
                content: "All company laptops must use full-disk encryption. Passwords must \
                          be at least 12 characters and rotated every 90 days. Two-factor \
                          authentication is required for all internal services."
                    .into(),
                metadata: HashMap::from([("source".into(), "it/security-policy.pdf".into())]),
            },
            Document {
                id: "onboarding".into(),
                title: "Onboarding Guide".into(),
                content: "Welcome to the company! Your first week will include orientation \
                          sessions, IT setup, and team introductions. Please bring your ID \
                          and completed tax forms on your first day."
                    .into(),
                metadata: HashMap::new(),
            },
        ]
    }

    async fn seeded_store() -> InMemoryDocumentStore {
        let store = InMemoryDocumentStore::new();
        for doc in sample_docs() {
            store.upsert(doc).await.unwrap();
        }
        store
    }

    #[tokio::test]
    async fn upsert_and_get() {
        let store = seeded_store().await;
        let doc = store.get("handbook").await.unwrap().unwrap();
        assert_eq!(doc.title, "Employee Handbook");
        assert!(doc.content.contains("vacation"));
    }

    #[tokio::test]
    async fn get_nonexistent_returns_none() {
        let store = seeded_store().await;
        assert!(store.get("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn upsert_replaces_existing() {
        let store = seeded_store().await;
        let updated = Document {
            id: "handbook".into(),
            title: "Employee Handbook v2".into(),
            content: "Updated vacation policy.".into(),
            metadata: HashMap::new(),
        };
        store.upsert(updated).await.unwrap();
        let doc = store.get("handbook").await.unwrap().unwrap();
        assert_eq!(doc.title, "Employee Handbook v2");
    }

    #[tokio::test]
    async fn remove_document() {
        let store = seeded_store().await;
        assert!(store.remove("handbook").await.unwrap());
        assert!(store.get("handbook").await.unwrap().is_none());
        assert!(!store.remove("handbook").await.unwrap());
    }

    #[tokio::test]
    async fn list_returns_sorted_summaries() {
        let store = seeded_store().await;
        let summaries = store.list().await.unwrap();
        assert_eq!(summaries.len(), 3);
        assert_eq!(summaries[0].id, "handbook");
        assert_eq!(summaries[1].id, "onboarding");
        assert_eq!(summaries[2].id, "security");
        // Summaries shouldn't carry content (checked via type — DocumentSummary has no content field)
        assert_eq!(summaries[0].title, "Employee Handbook");
    }

    #[tokio::test]
    async fn search_finds_relevant_docs() {
        let store = seeded_store().await;
        let results = store.search("vacation days", 5).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].document_id, "handbook");
        assert!(results[0].score > 0.0);
    }

    #[tokio::test]
    async fn search_ranks_correctly() {
        let store = seeded_store().await;
        let results = store.search("password encryption", 5).await.unwrap();
        assert!(!results.is_empty());
        // Security policy should rank highest for password/encryption queries
        assert_eq!(results[0].document_id, "security");
    }

    #[tokio::test]
    async fn search_empty_query_returns_empty() {
        let store = seeded_store().await;
        let results = store.search("", 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_no_matches() {
        let store = seeded_store().await;
        let results = store.search("quantum physics", 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_respects_limit() {
        let store = seeded_store().await;
        let results = store.search("the company", 1).await.unwrap();
        assert!(results.len() <= 1);
    }

    #[tokio::test]
    async fn search_empty_store() {
        let store = InMemoryDocumentStore::new();
        let results = store.search("anything", 5).await.unwrap();
        assert!(results.is_empty());
    }

    // -- Tool tests --

    #[tokio::test]
    async fn search_tool_returns_results() {
        let store = Arc::new(seeded_store().await);
        let tool = SearchDocumentsTool::new(store);
        let result = tool
            .execute(serde_json::json!({"query": "vacation"}))
            .await
            .unwrap();
        assert!(result.contains("Employee Handbook"));
        assert!(result.contains("handbook"));
    }

    #[tokio::test]
    async fn search_tool_no_results() {
        let store = Arc::new(seeded_store().await);
        let tool = SearchDocumentsTool::new(store);
        let result = tool
            .execute(serde_json::json!({"query": "quantum entanglement"}))
            .await
            .unwrap();
        assert!(result.contains("No documents found"));
    }

    #[tokio::test]
    async fn search_tool_missing_query() {
        let store = Arc::new(seeded_store().await);
        let tool = SearchDocumentsTool::new(store);
        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn read_tool_returns_full_document() {
        let store = Arc::new(seeded_store().await);
        let tool = ReadDocumentTool::new(store);
        let result = tool
            .execute(serde_json::json!({"id": "security"}))
            .await
            .unwrap();
        assert!(result.contains("Security Policy"));
        assert!(result.contains("full-disk encryption"));
        assert!(result.contains("source: it/security-policy.pdf"));
    }

    #[tokio::test]
    async fn read_tool_not_found() {
        let store = Arc::new(seeded_store().await);
        let tool = ReadDocumentTool::new(store);
        let err = tool
            .execute(serde_json::json!({"id": "nonexistent"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::ExecutionFailed(_)));
    }

    #[tokio::test]
    async fn read_tool_missing_id() {
        let store = Arc::new(seeded_store().await);
        let tool = ReadDocumentTool::new(store);
        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn list_tool_returns_all() {
        let store = Arc::new(seeded_store().await);
        let tool = ListDocumentsTool::new(store);
        let result = tool.execute(serde_json::json!({})).await.unwrap();
        assert!(result.contains("[handbook]"));
        assert!(result.contains("[security]"));
        assert!(result.contains("[onboarding]"));
        assert!(result.contains("Employee Handbook"));
    }

    #[tokio::test]
    async fn list_tool_empty_store() {
        let store = Arc::new(InMemoryDocumentStore::new());
        let tool = ListDocumentsTool::new(store);
        let result = tool.execute(serde_json::json!({})).await.unwrap();
        assert!(result.contains("No documents"));
    }

    #[tokio::test]
    async fn register_tools_adds_all_three() {
        let store = Arc::new(InMemoryDocumentStore::new());
        let mut registry = ToolRegistry::new();
        register_tools(&mut registry, store);

        assert_eq!(registry.len(), 3);
        assert!(registry.get("search_documents").is_some());
        assert!(registry.get("read_document").is_some());
        assert!(registry.get("list_documents").is_some());
    }

    #[tokio::test]
    async fn tool_definitions_have_schemas() {
        let store: Arc<dyn DocumentStore> = Arc::new(InMemoryDocumentStore::new());
        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(SearchDocumentsTool::new(Arc::clone(&store))),
            Box::new(ReadDocumentTool::new(Arc::clone(&store))),
            Box::new(ListDocumentsTool::new(store)),
        ];

        for tool in &tools {
            let def = tool.definition();
            assert!(!def.name.is_empty());
            assert!(!def.description.is_empty());
            assert_eq!(def.input_schema["type"], "object");
        }
    }

    // -- Utility function tests --

    #[test]
    fn tokenize_splits_and_lowercases() {
        let tokens = tokenize("Hello, World! This is a TEST.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        // Single-char words filtered out
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn extract_chunk_short_content() {
        let content = "Short text about vacation.";
        let chunk = extract_chunk(content, &["vacation".into()], 500);
        assert_eq!(chunk, content);
    }

    #[test]
    fn extract_chunk_centers_on_match() {
        let content = "A ".repeat(200) + "vacation policy is important" + &" B".repeat(200);
        let chunk = extract_chunk(&content, &["vacation".into()], 100);
        assert!(chunk.contains("vacation"));
        assert!(chunk.len() <= 120); // some slack for word boundary snapping + ellipsis
    }
}
