//! Project and workspace management.
//!
//! Provides a way to organize sessions, tools, and configuration into
//! logical projects. Each project can have its own system prompt, tool
//! set, and memory namespace, making it easy to run multiple distinct
//! agent configurations from a single application.

use std::collections::HashMap;
use std::sync::Arc;

use crate::namespace::Namespace;

// ---------------------------------------------------------------------------
// Project definition
// ---------------------------------------------------------------------------

/// A project defines a self-contained agent configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Project {
    /// Unique project identifier.
    pub id: String,

    /// Human-readable project name.
    pub name: String,

    /// Project description.
    pub description: String,

    /// Custom system prompt for this project.
    pub system_prompt: Option<String>,

    /// Tool names that should be available in this project.
    /// If empty, all tools are available.
    pub enabled_tools: Vec<String>,

    /// Additional key-value metadata.
    pub metadata: HashMap<String, serde_json::Value>,

    /// Whether this project is currently active.
    pub active: bool,

    /// Creation timestamp (ISO 8601).
    pub created_at: String,

    /// Last modified timestamp (ISO 8601).
    pub updated_at: String,
}

impl Project {
    /// Create a new project with the given ID and name.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            system_prompt: None,
            enabled_tools: Vec::new(),
            metadata: HashMap::new(),
            active: true,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    /// Get the namespace prefix for this project.
    pub fn namespace(&self) -> Namespace {
        Namespace::parse(&format!("project/{}", self.id))
    }

    /// Build a session namespace scoped to this project and a user.
    pub fn user_namespace(&self, user_id: &str) -> Namespace {
        Namespace::parse(&format!("project/{}/user/{}", self.id, user_id))
    }
}

// ---------------------------------------------------------------------------
// Project store trait
// ---------------------------------------------------------------------------

/// Persistence layer for projects.
#[async_trait::async_trait]
pub trait ProjectStore: Send + Sync {
    /// Save or update a project.
    async fn save(&self, project: &Project) -> Result<(), ProjectError>;

    /// Load a project by ID.
    async fn load(&self, id: &str) -> Result<Option<Project>, ProjectError>;

    /// Delete a project by ID.
    async fn delete(&self, id: &str) -> Result<bool, ProjectError>;

    /// List all projects.
    async fn list(&self) -> Result<Vec<Project>, ProjectError>;
}

// ---------------------------------------------------------------------------
// In-memory project store
// ---------------------------------------------------------------------------

/// Simple in-memory project store for development and testing.
pub struct InMemoryProjectStore {
    projects: tokio::sync::RwLock<HashMap<String, Project>>,
}

impl InMemoryProjectStore {
    pub fn new() -> Self {
        Self {
            projects: tokio::sync::RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryProjectStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ProjectStore for InMemoryProjectStore {
    async fn save(&self, project: &Project) -> Result<(), ProjectError> {
        self.projects
            .write()
            .await
            .insert(project.id.clone(), project.clone());
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<Project>, ProjectError> {
        Ok(self.projects.read().await.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<bool, ProjectError> {
        Ok(self.projects.write().await.remove(id).is_some())
    }

    async fn list(&self) -> Result<Vec<Project>, ProjectError> {
        Ok(self.projects.read().await.values().cloned().collect())
    }
}

// ---------------------------------------------------------------------------
// Project manager
// ---------------------------------------------------------------------------

/// Manages the lifecycle of projects.
pub struct ProjectManager {
    store: Arc<dyn ProjectStore>,
}

impl ProjectManager {
    pub fn new(store: Arc<dyn ProjectStore>) -> Self {
        Self { store }
    }

    /// Create a new project.
    pub async fn create(
        &self,
        id: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<Project, ProjectError> {
        let id = id.into();

        // Check for duplicates
        if self.store.load(&id).await?.is_some() {
            return Err(ProjectError::AlreadyExists(id));
        }

        let project = Project::new(id, name);
        self.store.save(&project).await?;
        Ok(project)
    }

    /// Get a project by ID.
    pub async fn get(&self, id: &str) -> Result<Option<Project>, ProjectError> {
        self.store.load(id).await
    }

    /// Update an existing project.
    pub async fn update(&self, project: &mut Project) -> Result<(), ProjectError> {
        project.updated_at = chrono::Utc::now().to_rfc3339();
        self.store.save(project).await
    }

    /// Delete a project.
    pub async fn delete(&self, id: &str) -> Result<bool, ProjectError> {
        self.store.delete(id).await
    }

    /// List all projects.
    pub async fn list(&self) -> Result<Vec<Project>, ProjectError> {
        self.store.list().await
    }

    /// List only active projects.
    pub async fn list_active(&self) -> Result<Vec<Project>, ProjectError> {
        let all = self.store.list().await?;
        Ok(all.into_iter().filter(|p| p.active).collect())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    #[error("project already exists: {0}")]
    AlreadyExists(String),

    #[error("project not found: {0}")]
    NotFound(String),

    #[error("storage error: {0}")]
    Storage(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_creation() {
        let project = Project::new("my-bot", "My Bot");
        assert_eq!(project.id, "my-bot");
        assert_eq!(project.name, "My Bot");
        assert!(project.active);
        assert!(project.enabled_tools.is_empty());
    }

    #[test]
    fn project_namespace() {
        let project = Project::new("proj-1", "Test");
        assert_eq!(project.namespace().key(), "project/proj-1");
    }

    #[test]
    fn project_user_namespace() {
        let project = Project::new("proj-1", "Test");
        let ns = project.user_namespace("user-42");
        assert_eq!(ns.key(), "project/proj-1/user/user-42");
    }

    #[test]
    fn project_serialization() {
        let project = Project::new("test", "Test Project");
        let json = serde_json::to_string(&project).unwrap();
        let deser: Project = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.id, "test");
        assert_eq!(deser.name, "Test Project");
    }

    #[tokio::test]
    async fn in_memory_store_crud() {
        let store = InMemoryProjectStore::new();

        let project = Project::new("p1", "Project 1");
        store.save(&project).await.unwrap();

        let loaded = store.load("p1").await.unwrap().unwrap();
        assert_eq!(loaded.name, "Project 1");

        assert!(store.load("nonexistent").await.unwrap().is_none());

        assert!(store.delete("p1").await.unwrap());
        assert!(!store.delete("p1").await.unwrap());
    }

    #[tokio::test]
    async fn in_memory_store_list() {
        let store = InMemoryProjectStore::new();
        store.save(&Project::new("a", "A")).await.unwrap();
        store.save(&Project::new("b", "B")).await.unwrap();

        let list = store.list().await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn manager_create_and_get() {
        let store = Arc::new(InMemoryProjectStore::new());
        let manager = ProjectManager::new(store);

        let project = manager.create("test", "Test").await.unwrap();
        assert_eq!(project.id, "test");

        let loaded = manager.get("test").await.unwrap().unwrap();
        assert_eq!(loaded.name, "Test");
    }

    #[tokio::test]
    async fn manager_duplicate_create_fails() {
        let store = Arc::new(InMemoryProjectStore::new());
        let manager = ProjectManager::new(store);

        manager.create("dup", "First").await.unwrap();
        let err = manager.create("dup", "Second").await.unwrap_err();
        assert!(matches!(err, ProjectError::AlreadyExists(_)));
    }

    #[tokio::test]
    async fn manager_update() {
        let store = Arc::new(InMemoryProjectStore::new());
        let manager = ProjectManager::new(store);

        let mut project = manager.create("upd", "Original").await.unwrap();
        let original_updated = project.updated_at.clone();

        // Small delay to ensure timestamp changes
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        project.name = "Updated".into();
        manager.update(&mut project).await.unwrap();

        let loaded = manager.get("upd").await.unwrap().unwrap();
        assert_eq!(loaded.name, "Updated");
        assert_ne!(loaded.updated_at, original_updated);
    }

    #[tokio::test]
    async fn manager_list_active() {
        let store = Arc::new(InMemoryProjectStore::new());
        let manager = ProjectManager::new(store);

        manager.create("active", "Active").await.unwrap();
        let mut inactive = manager.create("inactive", "Inactive").await.unwrap();
        inactive.active = false;
        manager.update(&mut inactive).await.unwrap();

        let active = manager.list_active().await.unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, "active");
    }

    #[tokio::test]
    async fn manager_delete() {
        let store = Arc::new(InMemoryProjectStore::new());
        let manager = ProjectManager::new(store);

        manager.create("del", "Delete Me").await.unwrap();
        assert!(manager.delete("del").await.unwrap());
        assert!(manager.get("del").await.unwrap().is_none());
    }
}
