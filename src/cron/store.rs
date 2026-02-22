use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use super::types::CronJob;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum CronStoreError {
    #[error("cron job not found: {0}")]
    NotFound(String),

    #[error("storage error: {0}")]
    Storage(String),
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait CronStore: Send + Sync {
    async fn save(&self, job: &CronJob) -> Result<(), CronStoreError>;
    async fn load(&self, id: &str) -> Result<Option<CronJob>, CronStoreError>;
    async fn delete(&self, id: &str) -> Result<bool, CronStoreError>;
    async fn list(&self) -> Result<Vec<CronJob>, CronStoreError>;
}

// ---------------------------------------------------------------------------
// In-memory store
// ---------------------------------------------------------------------------

pub struct InMemoryCronStore {
    jobs: Arc<RwLock<HashMap<String, CronJob>>>,
}

impl InMemoryCronStore {
    pub fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryCronStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CronStore for InMemoryCronStore {
    async fn save(&self, job: &CronJob) -> Result<(), CronStoreError> {
        self.jobs.write().await.insert(job.id.clone(), job.clone());
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<CronJob>, CronStoreError> {
        Ok(self.jobs.read().await.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<bool, CronStoreError> {
        Ok(self.jobs.write().await.remove(id).is_some())
    }

    async fn list(&self) -> Result<Vec<CronJob>, CronStoreError> {
        Ok(self.jobs.read().await.values().cloned().collect())
    }
}

// ---------------------------------------------------------------------------
// File-backed store (JSON)
// ---------------------------------------------------------------------------

/// Persists cron jobs to a single JSON file with atomic writes.
pub struct FileCronStore {
    path: PathBuf,
    cache: Arc<RwLock<HashMap<String, CronJob>>>,
}

impl FileCronStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Load all jobs from disk into the in-memory cache.
    pub async fn load_from_disk(&self) -> Result<(), CronStoreError> {
        let path = self.path.clone();
        let data = match tokio::fs::read_to_string(&path).await {
            Ok(s) => s,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(CronStoreError::Storage(format!("read error: {e}"))),
        };

        let jobs: Vec<CronJob> = serde_json::from_str(&data)
            .map_err(|e| CronStoreError::Storage(format!("deserialize error: {e}")))?;

        let mut cache = self.cache.write().await;
        cache.clear();
        for job in jobs {
            cache.insert(job.id.clone(), job);
        }

        Ok(())
    }

    /// Flush the in-memory cache to disk (atomic write).
    async fn flush(&self) -> Result<(), CronStoreError> {
        let cache = self.cache.read().await;
        let jobs: Vec<&CronJob> = cache.values().collect();

        let json = serde_json::to_string_pretty(&jobs)
            .map_err(|e| CronStoreError::Storage(format!("serialize error: {e}")))?;

        // Ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| CronStoreError::Storage(format!("mkdir error: {e}")))?;
        }

        // Atomic write: write to tmp, then rename
        let tmp_path = self.path.with_extension("tmp");
        tokio::fs::write(&tmp_path, &json)
            .await
            .map_err(|e| CronStoreError::Storage(format!("write error: {e}")))?;
        tokio::fs::rename(&tmp_path, &self.path)
            .await
            .map_err(|e| CronStoreError::Storage(format!("rename error: {e}")))?;

        Ok(())
    }
}

#[async_trait]
impl CronStore for FileCronStore {
    async fn save(&self, job: &CronJob) -> Result<(), CronStoreError> {
        self.cache.write().await.insert(job.id.clone(), job.clone());
        self.flush().await
    }

    async fn load(&self, id: &str) -> Result<Option<CronJob>, CronStoreError> {
        Ok(self.cache.read().await.get(id).cloned())
    }

    async fn delete(&self, id: &str) -> Result<bool, CronStoreError> {
        let removed = self.cache.write().await.remove(id).is_some();
        if removed {
            self.flush().await?;
        }
        Ok(removed)
    }

    async fn list(&self) -> Result<Vec<CronJob>, CronStoreError> {
        Ok(self.cache.read().await.values().cloned().collect())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cron::types::*;

    fn make_job(name: &str) -> CronJob {
        CronJob::new(
            name,
            CronScheduleType::Every { interval_ms: 60000 },
            CronPayload::SystemEvent {
                message: "tick".into(),
            },
            "test",
        )
    }

    #[tokio::test]
    async fn in_memory_crud() {
        let store = InMemoryCronStore::new();

        let job = make_job("test1");
        let id = job.id.clone();

        store.save(&job).await.unwrap();
        assert!(store.load(&id).await.unwrap().is_some());

        let all = store.list().await.unwrap();
        assert_eq!(all.len(), 1);

        assert!(store.delete(&id).await.unwrap());
        assert!(store.load(&id).await.unwrap().is_none());
        assert!(!store.delete(&id).await.unwrap());
    }

    #[tokio::test]
    async fn file_store_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cron_jobs.json");
        let store = FileCronStore::new(&path);

        let job1 = make_job("job1");
        let job2 = make_job("job2");
        let id1 = job1.id.clone();

        store.save(&job1).await.unwrap();
        store.save(&job2).await.unwrap();

        // Reload from disk
        let store2 = FileCronStore::new(&path);
        store2.load_from_disk().await.unwrap();
        let all = store2.list().await.unwrap();
        assert_eq!(all.len(), 2);

        assert!(store2.delete(&id1).await.unwrap());
        let all = store2.list().await.unwrap();
        assert_eq!(all.len(), 1);
    }

    #[tokio::test]
    async fn file_store_empty_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nonexistent.json");
        let store = FileCronStore::new(&path);
        store.load_from_disk().await.unwrap();
        let all = store.list().await.unwrap();
        assert!(all.is_empty());
    }
}
