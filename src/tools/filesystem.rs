//! Filesystem tools with optional sandbox boundary enforcement.
//!
//! Provides `read_file`, `write_file`, `edit_file`, and `list_dir` tools
//! that give agents safe, scoped file access without raw shell commands.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;

use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Shared configuration for all filesystem tools.
#[derive(Debug, Clone)]
pub struct FilesystemConfig {
    /// If set, all paths are resolved relative to (and confined within) this
    /// directory. If `None`, paths resolve against the current working
    /// directory with no boundary enforcement.
    pub base_dir: Option<PathBuf>,
    /// Maximum file size in bytes for `read_file` (default: 1 MB).
    pub max_read_size: u64,
    /// Maximum content size in bytes for `write_file` / `edit_file` (default: 5 MB).
    pub max_write_size: usize,
    /// Path suffixes / filenames that `write_file` and `edit_file` refuse to
    /// touch (e.g. `[".env", "secrets.toml"]`).
    pub protected_paths: Vec<String>,
    /// Directory names to skip during recursive `list_dir`.
    pub skip_dirs: Vec<String>,
}

impl Default for FilesystemConfig {
    fn default() -> Self {
        Self {
            base_dir: None,
            max_read_size: 1_048_576,  // 1 MB
            max_write_size: 5_242_880, // 5 MB
            protected_paths: Vec::new(),
            skip_dirs: vec![
                "node_modules".into(),
                "target".into(),
                ".git".into(),
                "__pycache__".into(),
                "venv".into(),
                ".venv".into(),
                ".next".into(),
                "dist".into(),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Path validation helpers
// ---------------------------------------------------------------------------

/// Resolve `.` and `..` components without touching the filesystem.
/// This prevents directory-traversal via non-existent intermediate paths.
fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::ParentDir => {
                // Don't pop beyond root / prefix
                if !out.pop() {
                    out.push(comp);
                }
            }
            Component::CurDir => {} // skip
            other => out.push(other),
        }
    }
    out
}

/// Resolve `path_str` to an absolute, canonicalized path and — when
/// `base_dir` is set — verify the result stays inside the sandbox.
fn validate_path(path_str: &str, base_dir: Option<&Path>) -> Result<PathBuf, ToolError> {
    let raw = Path::new(path_str);

    // Step 1 — build the absolute, lexically-normalized path.
    let abs = if raw.is_absolute() {
        normalize_lexical(raw)
    } else if let Some(base) = base_dir {
        normalize_lexical(&base.join(raw))
    } else {
        let cwd = std::env::current_dir()
            .map_err(|e| ToolError::ExecutionFailed(format!("cannot determine cwd: {e}")))?;
        normalize_lexical(&cwd.join(raw))
    };

    // Step 2 — sandbox containment check (if configured).
    if let Some(base) = base_dir {
        let canon_base = base
            .canonicalize()
            .map_err(|e| ToolError::InvalidInput(format!("base_dir cannot be resolved: {e}")))?;

        // For existing paths, fully canonicalize to chase symlinks.
        // For non-existing paths, walk up to the nearest existing ancestor,
        // canonicalize *that*, then re-append the tail — this catches symlink
        // escapes above not-yet-created files.
        let canon = if abs.exists() {
            abs.canonicalize().map_err(|e| {
                ToolError::ExecutionFailed(format!("cannot resolve path: {e}"))
            })?
        } else {
            let mut ancestor = abs.clone();
            let mut tail = PathBuf::new();
            while !ancestor.exists() {
                if let Some(name) = ancestor.file_name() {
                    tail = PathBuf::from(name).join(&tail);
                }
                if !ancestor.pop() {
                    break;
                }
            }
            let canon_ancestor = ancestor.canonicalize().map_err(|e| {
                ToolError::ExecutionFailed(format!("cannot resolve path: {e}"))
            })?;
            canon_ancestor.join(tail)
        };

        if !canon.starts_with(&canon_base) {
            return Err(ToolError::InvalidInput(format!(
                "path escapes sandbox: {}",
                path_str
            )));
        }
    }

    Ok(abs)
}

/// Check whether `path` matches any protected-path pattern.
fn check_protected(path: &Path, protected: &[String]) -> Result<(), ToolError> {
    for pattern in protected {
        let matches = path
            .file_name()
            .map(|f| f == pattern.as_str())
            .unwrap_or(false)
            || path.ends_with(pattern);

        if matches {
            return Err(ToolError::InvalidInput(format!(
                "path '{}' is protected and cannot be written to",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Human-readable file size.
fn format_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1}GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1}MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes}B")
    }
}

// ---------------------------------------------------------------------------
// ReadFileTool
// ---------------------------------------------------------------------------

pub struct ReadFileTool {
    config: Arc<FilesystemConfig>,
}

impl ReadFileTool {
    pub fn new(config: Arc<FilesystemConfig>) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".into(),
            description: "Read the contents of a file. Returns the text with line numbers. \
                          Use offset and limit for large files."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed, default: 1)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to return (default: all)"
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'path'".into()))?;

        let offset = input
            .get("offset")
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            .max(1) as usize;

        let limit = input.get("limit").and_then(|v| v.as_u64()).map(|v| v as usize);

        let path = validate_path(path_str, self.config.base_dir.as_deref())?;

        // Check metadata
        let meta = tokio::fs::metadata(&path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ToolError::ExecutionFailed(format!("file not found: {}", path.display()))
            } else {
                ToolError::ExecutionFailed(format!("cannot access file: {e}"))
            }
        })?;

        if meta.is_dir() {
            return Err(ToolError::ExecutionFailed(
                "path is a directory, use list_dir instead".into(),
            ));
        }

        if meta.len() > self.config.max_read_size {
            return Err(ToolError::ExecutionFailed(format!(
                "file too large ({}, max {}). Use offset/limit for partial reads.",
                format_size(meta.len()),
                format_size(self.config.max_read_size)
            )));
        }

        let content = tokio::fs::read_to_string(&path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to read file: {e}"))
        })?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        // Apply offset (1-indexed) and limit
        let start = (offset - 1).min(total_lines);
        let end = if let Some(lim) = limit {
            (start + lim).min(total_lines)
        } else {
            total_lines
        };

        let selected = &lines[start..end];

        let mut result = format!(
            "[{} | {} lines | showing {}-{}]\n",
            path.display(),
            total_lines,
            start + 1,
            end
        );

        for (i, line) in selected.iter().enumerate() {
            let line_num = start + i + 1;
            result.push_str(&format!("{:>6}| {}\n", line_num, line));
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// WriteFileTool
// ---------------------------------------------------------------------------

pub struct WriteFileTool {
    config: Arc<FilesystemConfig>,
}

impl WriteFileTool {
    pub fn new(config: Arc<FilesystemConfig>) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".into(),
            description: "Write content to a file. Creates parent directories if needed. \
                          Overwrites existing content."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to write to (creates file and parent directories if they don't exist)"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file (overwrites existing content)"
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'path'".into()))?;

        let content = input
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'content'".into()))?;

        if content.len() > self.config.max_write_size {
            return Err(ToolError::InvalidInput(format!(
                "content too large ({}, max {})",
                format_size(content.len() as u64),
                format_size(self.config.max_write_size as u64)
            )));
        }

        let path = validate_path(path_str, self.config.base_dir.as_deref())?;
        check_protected(&path, &self.config.protected_paths)?;

        // Create parent directories
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                ToolError::ExecutionFailed(format!("failed to create directories: {e}"))
            })?;
        }

        let bytes = content.len();
        tokio::fs::write(&path, content).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to write file: {e}"))
        })?;

        Ok(format!("Wrote {} bytes to {}", bytes, path.display()))
    }
}

// ---------------------------------------------------------------------------
// EditFileTool
// ---------------------------------------------------------------------------

pub struct EditFileTool {
    config: Arc<FilesystemConfig>,
}

impl EditFileTool {
    pub fn new(config: Arc<FilesystemConfig>) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit_file".into(),
            description: "Edit a file by replacing an exact string with a new string. \
                          The old_string must match exactly (including whitespace and indentation)."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find (must match exactly, including whitespace)"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "If true, replace all occurrences (default: false, replaces first only)"
                    }
                },
                "required": ["path", "old_string", "new_string"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'path'".into()))?;

        let old_string = input
            .get("old_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'old_string'".into()))?;

        let new_string = input
            .get("new_string")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'new_string'".into()))?;

        let replace_all = input
            .get("replace_all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let path = validate_path(path_str, self.config.base_dir.as_deref())?;
        check_protected(&path, &self.config.protected_paths)?;

        let content = tokio::fs::read_to_string(&path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ToolError::ExecutionFailed(format!("file not found: {}", path.display()))
            } else {
                ToolError::ExecutionFailed(format!("failed to read file: {e}"))
            }
        })?;

        if !content.contains(old_string) {
            return Err(ToolError::ExecutionFailed(format!(
                "could not find the specified text in {}. Make sure old_string matches exactly.",
                path.display()
            )));
        }

        let (new_content, count) = if replace_all {
            let count = content.matches(old_string).count();
            (content.replace(old_string, new_string), count)
        } else {
            (content.replacen(old_string, new_string, 1), 1)
        };

        if new_content.len() > self.config.max_write_size {
            return Err(ToolError::InvalidInput(format!(
                "resulting file too large ({}, max {})",
                format_size(new_content.len() as u64),
                format_size(self.config.max_write_size as u64)
            )));
        }

        tokio::fs::write(&path, &new_content).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("failed to write file: {e}"))
        })?;

        Ok(format!(
            "Replaced {} occurrence(s) in {}",
            count,
            path.display()
        ))
    }
}

// ---------------------------------------------------------------------------
// ListDirTool
// ---------------------------------------------------------------------------

pub struct ListDirTool {
    config: Arc<FilesystemConfig>,
}

impl ListDirTool {
    pub fn new(config: Arc<FilesystemConfig>) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Tool for ListDirTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_dir".into(),
            description: "List files and directories at a given path. \
                          Use recursive=true to traverse subdirectories."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory to list (default: current directory)"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List contents recursively (default: false)"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth for recursive listing (default: 3)"
                    }
                }
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let path_str = input
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        let recursive = input
            .get("recursive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let max_depth = input
            .get("max_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;

        let path = validate_path(path_str, self.config.base_dir.as_deref())?;

        let meta = tokio::fs::metadata(&path).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ToolError::ExecutionFailed(format!("directory not found: {}", path.display()))
            } else {
                ToolError::ExecutionFailed(format!("cannot access path: {e}"))
            }
        })?;

        if !meta.is_dir() {
            return Err(ToolError::ExecutionFailed(format!(
                "not a directory: {}",
                path.display()
            )));
        }

        const MAX_ENTRIES: usize = 500;
        let mut entries: Vec<(String, bool, u64)> = Vec::new(); // (display_name, is_dir, size)
        let mut truncated = false;

        collect_entries(
            &path,
            &path,
            recursive,
            max_depth,
            0,
            &self.config.skip_dirs,
            MAX_ENTRIES,
            &mut entries,
            &mut truncated,
        )
        .await?;

        // Sort: directories first, then alphabetically
        entries.sort_by(|a, b| match (a.1, b.1) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.0.to_lowercase().cmp(&b.0.to_lowercase()),
        });

        let total = entries.len();
        let mut result = format!("[{} | {} entries]\n", path.display(), total);

        for (name, is_dir, size) in &entries {
            if *is_dir {
                result.push_str(&format!("{name}/\n"));
            } else {
                result.push_str(&format!("{name} ({})\n", format_size(*size)));
            }
        }

        if truncated {
            result.push_str(&format!("[truncated: {MAX_ENTRIES} entries shown]\n"));
        }

        Ok(result)
    }
}

/// Recursively collect directory entries.
#[allow(clippy::too_many_arguments)]
async fn collect_entries(
    root: &Path,
    dir: &Path,
    recursive: bool,
    max_depth: usize,
    depth: usize,
    skip_dirs: &[String],
    max_entries: usize,
    entries: &mut Vec<(String, bool, u64)>,
    truncated: &mut bool,
) -> Result<(), ToolError> {
    let mut read_dir = tokio::fs::read_dir(dir).await.map_err(|e| {
        ToolError::ExecutionFailed(format!("failed to read directory: {e}"))
    })?;

    while let Some(entry) = read_dir.next_entry().await.map_err(|e| {
        ToolError::ExecutionFailed(format!("failed to read entry: {e}"))
    })? {
        if entries.len() >= max_entries {
            *truncated = true;
            return Ok(());
        }

        let name = entry.file_name().to_string_lossy().to_string();
        let meta = match entry.metadata().await {
            Ok(m) => m,
            Err(_) => continue, // skip unreadable entries
        };

        let is_dir = meta.is_dir();

        // Skip configured directories
        if is_dir && skip_dirs.iter().any(|s| s == &name) {
            continue;
        }

        // Build relative display path
        let rel = entry
            .path()
            .strip_prefix(root)
            .unwrap_or(&entry.path())
            .to_string_lossy()
            .to_string();

        entries.push((rel, is_dir, meta.len()));

        if is_dir && recursive && depth < max_depth {
            Box::pin(collect_entries(
                root,
                &entry.path(),
                recursive,
                max_depth,
                depth + 1,
                skip_dirs,
                max_entries,
                entries,
                truncated,
            ))
            .await?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all filesystem tools into a tool registry.
pub fn register_tools(registry: &mut ToolRegistry, config: FilesystemConfig) {
    let config = Arc::new(config);
    registry.register(Box::new(ReadFileTool::new(config.clone())));
    registry.register(Box::new(WriteFileTool::new(config.clone())));
    registry.register(Box::new(EditFileTool::new(config.clone())));
    registry.register(Box::new(ListDirTool::new(config)));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- normalize_lexical --------------------------------------------------

    #[test]
    fn normalize_resolves_dot_dot() {
        let p = normalize_lexical(Path::new("/a/b/../c"));
        assert_eq!(p, PathBuf::from("/a/c"));
    }

    #[test]
    fn normalize_strips_dot() {
        let p = normalize_lexical(Path::new("/a/./b"));
        assert_eq!(p, PathBuf::from("/a/b"));
    }

    #[test]
    fn normalize_cannot_escape_root() {
        let p = normalize_lexical(Path::new("/a/../../.."));
        assert_eq!(p, PathBuf::from("/"));
    }

    #[test]
    fn normalize_relative_dot_dot() {
        let p = normalize_lexical(Path::new("a/b/../c.txt"));
        assert_eq!(p, PathBuf::from("a/c.txt"));
    }

    // -- validate_path (sandbox) --------------------------------------------

    #[test]
    fn validate_rejects_traversal() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let err = validate_path("../../etc/passwd", Some(base)).unwrap_err();
        assert!(err.to_string().contains("escapes sandbox"));
    }

    #[test]
    fn validate_rejects_nonexistent_parent_traversal() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let err = validate_path("foo/../../outside/file.txt", Some(base)).unwrap_err();
        assert!(err.to_string().contains("escapes sandbox"));
    }

    #[test]
    fn validate_allows_nested_within_sandbox() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let result = validate_path("subdir/file.txt", Some(base));
        assert!(result.is_ok());
        assert!(result.unwrap().starts_with(base));
    }

    #[test]
    fn validate_allows_dot_dot_within_sandbox() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path();
        let result = validate_path("a/b/../c.txt", Some(base));
        assert!(result.is_ok());
        let resolved = result.unwrap();
        assert!(resolved.starts_with(base));
        assert!(resolved.ends_with("a/c.txt"));
    }

    #[test]
    fn validate_no_sandbox_resolves_relative() {
        let result = validate_path("some_file.txt", None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_absolute());
    }

    // -- check_protected ----------------------------------------------------

    #[test]
    fn protected_rejects_matching_filename() {
        let path = Path::new("/home/user/.env");
        let err = check_protected(path, &[".env".into()]).unwrap_err();
        assert!(err.to_string().contains("protected"));
    }

    #[test]
    fn protected_allows_unmatched_path() {
        let path = Path::new("/home/user/config.toml");
        assert!(check_protected(path, &[".env".into()]).is_ok());
    }

    // -- format_size --------------------------------------------------------

    #[test]
    fn format_size_bytes() {
        assert_eq!(format_size(512), "512B");
    }

    #[test]
    fn format_size_kb() {
        assert_eq!(format_size(2048), "2.0KB");
    }

    #[test]
    fn format_size_mb() {
        assert_eq!(format_size(3_145_728), "3.0MB");
    }

    // -- ReadFileTool -------------------------------------------------------

    #[tokio::test]
    async fn read_file_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("test.txt");
        std::fs::write(&file, "line one\nline two\nline three\n").unwrap();

        let tool = ReadFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let result = tool
            .execute(serde_json::json!({"path": "test.txt"}))
            .await
            .unwrap();

        assert!(result.contains("3 lines"));
        assert!(result.contains("     1| line one"));
        assert!(result.contains("     2| line two"));
        assert!(result.contains("     3| line three"));
    }

    #[tokio::test]
    async fn read_file_with_offset_and_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let content: String = (1..=10).map(|i| format!("line {i}\n")).collect();
        let file = tmp.path().join("ten.txt");
        std::fs::write(&file, &content).unwrap();

        let tool = ReadFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let result = tool
            .execute(serde_json::json!({"path": "ten.txt", "offset": 3, "limit": 2}))
            .await
            .unwrap();

        assert!(result.contains("showing 3-4"));
        assert!(result.contains("     3| line 3"));
        assert!(result.contains("     4| line 4"));
        assert!(!result.contains("line 5"));
    }

    #[tokio::test]
    async fn read_file_missing_path() {
        let tool = ReadFileTool::new(Arc::new(FilesystemConfig::default()));
        let err = tool.execute(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn read_file_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = ReadFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let err = tool
            .execute(serde_json::json!({"path": "nope.txt"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[tokio::test]
    async fn read_file_too_large() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("big.txt");
        std::fs::write(&file, "x".repeat(200)).unwrap();

        let tool = ReadFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            max_read_size: 100,
            ..Default::default()
        }));

        let err = tool
            .execute(serde_json::json!({"path": "big.txt"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("too large"));
    }

    #[tokio::test]
    async fn read_file_directory_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("subdir")).unwrap();

        let tool = ReadFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let err = tool
            .execute(serde_json::json!({"path": "subdir"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("directory"));
    }

    // -- WriteFileTool ------------------------------------------------------

    #[tokio::test]
    async fn write_file_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let result = tool
            .execute(serde_json::json!({"path": "out.txt", "content": "hello world"}))
            .await
            .unwrap();

        assert!(result.contains("11 bytes"));
        assert_eq!(std::fs::read_to_string(tmp.path().join("out.txt")).unwrap(), "hello world");
    }

    #[tokio::test]
    async fn write_file_creates_parent_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        tool.execute(serde_json::json!({
            "path": "a/b/c.txt",
            "content": "nested"
        }))
        .await
        .unwrap();

        assert_eq!(
            std::fs::read_to_string(tmp.path().join("a/b/c.txt")).unwrap(),
            "nested"
        );
    }

    #[tokio::test]
    async fn write_file_overwrites() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("over.txt");
        std::fs::write(&file, "old").unwrap();

        let tool = WriteFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        tool.execute(serde_json::json!({"path": "over.txt", "content": "new"}))
            .await
            .unwrap();

        assert_eq!(std::fs::read_to_string(&file).unwrap(), "new");
    }

    #[tokio::test]
    async fn write_file_protected_path() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            protected_paths: vec![".env".into()],
            ..Default::default()
        }));

        let err = tool
            .execute(serde_json::json!({"path": ".env", "content": "SECRET=bad"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("protected"));
    }

    #[tokio::test]
    async fn write_file_too_large() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            max_write_size: 10,
            ..Default::default()
        }));

        let err = tool
            .execute(serde_json::json!({"path": "big.txt", "content": "x".repeat(100)}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("too large"));
    }

    #[tokio::test]
    async fn write_file_sandbox_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let tool = WriteFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let err = tool
            .execute(serde_json::json!({"path": "../../evil.txt", "content": "pwned"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("escapes sandbox"));
    }

    // -- EditFileTool -------------------------------------------------------

    #[tokio::test]
    async fn edit_file_single_replacement() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("edit.txt");
        std::fs::write(&file, "foo bar foo baz").unwrap();

        let tool = EditFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let result = tool
            .execute(serde_json::json!({
                "path": "edit.txt",
                "old_string": "foo",
                "new_string": "qux"
            }))
            .await
            .unwrap();

        assert!(result.contains("1 occurrence"));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "qux bar foo baz");
    }

    #[tokio::test]
    async fn edit_file_replace_all() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("edit2.txt");
        std::fs::write(&file, "aaa bbb aaa ccc aaa").unwrap();

        let tool = EditFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let result = tool
            .execute(serde_json::json!({
                "path": "edit2.txt",
                "old_string": "aaa",
                "new_string": "zzz",
                "replace_all": true
            }))
            .await
            .unwrap();

        assert!(result.contains("3 occurrence"));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "zzz bbb zzz ccc zzz");
    }

    #[tokio::test]
    async fn edit_file_string_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("noedit.txt");
        std::fs::write(&file, "hello world").unwrap();

        let tool = EditFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let err = tool
            .execute(serde_json::json!({
                "path": "noedit.txt",
                "old_string": "missing",
                "new_string": "replacement"
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("could not find"));
    }

    #[tokio::test]
    async fn edit_file_protected_path() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("secrets.toml");
        std::fs::write(&file, "key = \"value\"").unwrap();

        let tool = EditFileTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            protected_paths: vec!["secrets.toml".into()],
            ..Default::default()
        }));

        let err = tool
            .execute(serde_json::json!({
                "path": "secrets.toml",
                "old_string": "value",
                "new_string": "new_value"
            }))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("protected"));
    }

    // -- ListDirTool --------------------------------------------------------

    #[tokio::test]
    async fn list_dir_basic() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("a.txt"), "a").unwrap();
        std::fs::write(tmp.path().join("b.txt"), "bb").unwrap();
        std::fs::create_dir(tmp.path().join("subdir")).unwrap();

        let tool = ListDirTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let result = tool.execute(serde_json::json!({})).await.unwrap();

        assert!(result.contains("3 entries"));
        assert!(result.contains("subdir/"));
        assert!(result.contains("a.txt"));
        assert!(result.contains("b.txt"));
    }

    #[tokio::test]
    async fn list_dir_recursive() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("sub")).unwrap();
        std::fs::write(tmp.path().join("sub/nested.txt"), "n").unwrap();
        std::fs::write(tmp.path().join("top.txt"), "t").unwrap();

        let tool = ListDirTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let result = tool
            .execute(serde_json::json!({"recursive": true}))
            .await
            .unwrap();

        assert!(result.contains("sub/"));
        assert!(result.contains("sub/nested.txt"));
        assert!(result.contains("top.txt"));
    }

    #[tokio::test]
    async fn list_dir_skips_configured_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir(tmp.path().join("node_modules")).unwrap();
        std::fs::write(tmp.path().join("node_modules/pkg.json"), "{}").unwrap();
        std::fs::write(tmp.path().join("index.js"), "").unwrap();

        let tool = ListDirTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let result = tool
            .execute(serde_json::json!({"recursive": true}))
            .await
            .unwrap();

        assert!(!result.contains("node_modules"));
        assert!(result.contains("index.js"));
    }

    #[tokio::test]
    async fn list_dir_not_a_directory() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("file.txt"), "x").unwrap();

        let tool = ListDirTool::new(Arc::new(FilesystemConfig {
            base_dir: Some(tmp.path().to_path_buf()),
            ..Default::default()
        }));

        let err = tool
            .execute(serde_json::json!({"path": "file.txt"}))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not a directory"));
    }

    // -- Registration -------------------------------------------------------

    #[test]
    fn register_tools_adds_four() {
        let mut registry = ToolRegistry::new();
        register_tools(&mut registry, FilesystemConfig::default());
        assert_eq!(registry.len(), 4);
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("write_file").is_some());
        assert!(registry.get("edit_file").is_some());
        assert!(registry.get("list_dir").is_some());
    }

    // -- Config defaults ----------------------------------------------------

    #[test]
    fn default_config_has_sensible_values() {
        let cfg = FilesystemConfig::default();
        assert!(cfg.base_dir.is_none());
        assert_eq!(cfg.max_read_size, 1_048_576);
        assert_eq!(cfg.max_write_size, 5_242_880);
        assert!(cfg.protected_paths.is_empty());
        assert!(cfg.skip_dirs.contains(&"node_modules".to_string()));
        assert!(cfg.skip_dirs.contains(&"target".to_string()));
        assert!(cfg.skip_dirs.contains(&".git".to_string()));
    }
}
