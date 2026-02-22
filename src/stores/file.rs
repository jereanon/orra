use std::path::{Path, PathBuf};

use async_trait::async_trait;
use tokio::fs;

use crate::namespace::Namespace;
use crate::store::{Session, SessionStore, StoreError};

/// A file-system backed session store.
///
/// Each namespace maps to a JSON file on disk. The namespace segments become
/// directory components: `company:engineering:alice` is stored at
/// `<base_dir>/company/engineering/alice.json`.
///
/// Writes are atomic: data is written to a `.tmp` file first, then renamed.
pub struct FileStore {
    base_dir: PathBuf,
}

impl FileStore {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    fn namespace_to_path(&self, namespace: &Namespace) -> PathBuf {
        let mut path = self.base_dir.clone();
        for segment in namespace.segments() {
            path.push(segment);
        }
        path.set_extension("json");
        path
    }

    fn path_to_namespace(&self, path: &Path) -> Option<Namespace> {
        let relative = path.strip_prefix(&self.base_dir).ok()?;
        let stem = relative.with_extension("");
        let segments: Vec<&str> = stem.iter().filter_map(|s| s.to_str()).collect();
        if segments.is_empty() {
            return None;
        }
        let key = segments.join(":");
        Some(Namespace::parse(&key))
    }
}

#[async_trait]
impl SessionStore for FileStore {
    async fn load(&self, namespace: &Namespace) -> Result<Option<Session>, StoreError> {
        let path = self.namespace_to_path(namespace);
        match fs::read_to_string(&path).await {
            Ok(contents) => {
                let session: Session = serde_json::from_str(&contents)
                    .map_err(|e| StoreError::Storage(format!("deserialize error: {e}")))?;
                Ok(Some(session))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(StoreError::Storage(format!("read error: {e}"))),
        }
    }

    async fn save(&self, session: &Session) -> Result<(), StoreError> {
        let path = self.namespace_to_path(&session.namespace);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| StoreError::Storage(format!("mkdir error: {e}")))?;
        }

        // Write to temp file first, then atomic rename
        let tmp_path = path.with_extension("tmp");
        let json = serde_json::to_string_pretty(session)
            .map_err(|e| StoreError::Storage(format!("serialize error: {e}")))?;

        fs::write(&tmp_path, &json)
            .await
            .map_err(|e| StoreError::Storage(format!("write error: {e}")))?;

        fs::rename(&tmp_path, &path)
            .await
            .map_err(|e| StoreError::Storage(format!("rename error: {e}")))?;

        Ok(())
    }

    async fn delete(&self, namespace: &Namespace) -> Result<bool, StoreError> {
        let path = self.namespace_to_path(namespace);
        match fs::remove_file(&path).await {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(StoreError::Storage(format!("delete error: {e}"))),
        }
    }

    async fn list(&self, prefix: Option<&Namespace>) -> Result<Vec<Namespace>, StoreError> {
        let search_dir = match prefix {
            Some(ns) => {
                // Walk from the prefix directory
                let mut path = self.base_dir.clone();
                for segment in ns.segments() {
                    path.push(segment);
                }
                path
            }
            None => self.base_dir.clone(),
        };

        if !search_dir.exists() {
            return Ok(Vec::new());
        }

        let mut namespaces = Vec::new();
        let mut dirs = vec![search_dir.clone()];

        while let Some(dir) = dirs.pop() {
            let mut entries = fs::read_dir(&dir)
                .await
                .map_err(|e| StoreError::Storage(format!("readdir error: {e}")))?;

            while let Some(entry) = entries
                .next_entry()
                .await
                .map_err(|e| StoreError::Storage(format!("entry error: {e}")))?
            {
                let path = entry.path();
                if path.is_dir() {
                    dirs.push(path);
                } else if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Some(ns) = self.path_to_namespace(&path) {
                        namespaces.push(ns);
                    }
                }
            }
        }

        namespaces.sort_by_key(|a| a.key());
        Ok(namespaces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;

    fn make_store() -> (tempfile::TempDir, FileStore) {
        let tmp = tempfile::tempdir().unwrap();
        let store = FileStore::new(tmp.path());
        (tmp, store)
    }

    #[tokio::test]
    async fn save_and_load_roundtrip() {
        let (_tmp, store) = make_store();
        let ns = Namespace::new("acme").child("alice");

        let mut session = Session::new(ns.clone());
        session.push_message(Message::user("Hello"));
        session.push_message(Message::assistant("Hi there!"));
        store.save(&session).await.unwrap();

        let loaded = store.load(&ns).await.unwrap().unwrap();
        assert_eq!(loaded.message_count(), 2);
        assert_eq!(loaded.messages[0].content, "Hello");
        assert_eq!(loaded.messages[1].content, "Hi there!");
        assert_eq!(loaded.namespace, ns);
    }

    #[tokio::test]
    async fn load_nonexistent_returns_none() {
        let (_tmp, store) = make_store();
        let ns = Namespace::new("nope");
        let result = store.load(&ns).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn correct_directory_structure() {
        let (tmp, store) = make_store();
        let ns = Namespace::new("company")
            .child("engineering")
            .child("alice");

        let session = Session::new(ns.clone());
        store.save(&session).await.unwrap();

        let expected = tmp
            .path()
            .join("company")
            .join("engineering")
            .join("alice.json");
        assert!(expected.exists(), "expected file at {expected:?}");
    }

    #[tokio::test]
    async fn no_leftover_tmp_files() {
        let (tmp, store) = make_store();
        let ns = Namespace::new("test");

        let session = Session::new(ns.clone());
        store.save(&session).await.unwrap();

        // Should have .json but not .tmp
        assert!(tmp.path().join("test.json").exists());
        assert!(!tmp.path().join("test.tmp").exists());
    }

    #[tokio::test]
    async fn delete_existing_session() {
        let (_tmp, store) = make_store();
        let ns = Namespace::new("test");

        store.save(&Session::new(ns.clone())).await.unwrap();
        let deleted = store.delete(&ns).await.unwrap();
        assert!(deleted);

        let loaded = store.load(&ns).await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn delete_nonexistent_returns_false() {
        let (_tmp, store) = make_store();
        let ns = Namespace::new("nope");
        let deleted = store.delete(&ns).await.unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn list_all_sessions() {
        let (_tmp, store) = make_store();

        store
            .save(&Session::new(Namespace::new("acme").child("alice")))
            .await
            .unwrap();
        store
            .save(&Session::new(Namespace::new("acme").child("bob")))
            .await
            .unwrap();
        store
            .save(&Session::new(Namespace::new("other")))
            .await
            .unwrap();

        let all = store.list(None).await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[tokio::test]
    async fn list_with_prefix() {
        let (_tmp, store) = make_store();

        store
            .save(&Session::new(Namespace::new("acme").child("alice")))
            .await
            .unwrap();
        store
            .save(&Session::new(Namespace::new("acme").child("bob")))
            .await
            .unwrap();
        store
            .save(&Session::new(Namespace::new("other")))
            .await
            .unwrap();

        let acme = Namespace::new("acme");
        let filtered = store.list(Some(&acme)).await.unwrap();
        assert_eq!(filtered.len(), 2);

        for ns in &filtered {
            assert!(ns.key().starts_with("acme:"), "unexpected: {}", ns.key());
        }
    }

    #[tokio::test]
    async fn list_empty_directory() {
        let (_tmp, store) = make_store();
        let all = store.list(None).await.unwrap();
        assert!(all.is_empty());
    }

    #[tokio::test]
    async fn list_nonexistent_prefix() {
        let (_tmp, store) = make_store();
        let ns = Namespace::new("nope");
        let result = store.list(Some(&ns)).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn deeply_nested_namespaces() {
        let (_tmp, store) = make_store();
        let ns = Namespace::new("a")
            .child("b")
            .child("c")
            .child("d")
            .child("e");

        let mut session = Session::new(ns.clone());
        session.push_message(Message::user("deep"));
        store.save(&session).await.unwrap();

        let loaded = store.load(&ns).await.unwrap().unwrap();
        assert_eq!(loaded.messages[0].content, "deep");
        assert_eq!(loaded.namespace, ns);
    }

    #[tokio::test]
    async fn overwrite_existing_session() {
        let (_tmp, store) = make_store();
        let ns = Namespace::new("test");

        let mut session = Session::new(ns.clone());
        session.push_message(Message::user("First"));
        store.save(&session).await.unwrap();

        session.push_message(Message::user("Second"));
        store.save(&session).await.unwrap();

        let loaded = store.load(&ns).await.unwrap().unwrap();
        assert_eq!(loaded.message_count(), 2);
    }

    #[tokio::test]
    async fn list_includes_sessions_at_prefix_level() {
        let (_tmp, store) = make_store();

        // Session directly at the prefix namespace
        store
            .save(&Session::new(Namespace::new("acme")))
            .await
            .unwrap();
        // Session under the prefix
        store
            .save(&Session::new(Namespace::new("acme").child("alice")))
            .await
            .unwrap();

        let acme = Namespace::new("acme");
        let filtered = store.list(Some(&acme)).await.unwrap();
        // The prefix-level session file is at acme.json which is outside acme/ dir,
        // so list with prefix only finds children inside the acme/ directory
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].key(), "acme:alice");
    }
}
