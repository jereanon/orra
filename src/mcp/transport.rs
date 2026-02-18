use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};

use super::types::{JsonRpcRequest, JsonRpcResponse};

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("io error: {0}")]
    Io(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("transport closed")]
    Closed,
}

/// Transport layer for communicating with an MCP server.
#[async_trait]
pub trait McpTransport: Send + Sync {
    async fn send(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse, TransportError>;
    async fn close(&self) -> Result<(), TransportError>;
}

/// Stdio transport: communicates with a child process via stdin/stdout JSON lines.
pub struct StdioTransport {
    stdin: tokio::sync::Mutex<tokio::process::ChildStdin>,
    stdout: tokio::sync::Mutex<BufReader<tokio::process::ChildStdout>>,
    child: tokio::sync::Mutex<Child>,
}

impl StdioTransport {
    /// Spawn a child process and create a transport for communicating with it.
    pub async fn spawn(program: &str, args: &[&str]) -> Result<Self, TransportError> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| TransportError::Io(format!("failed to spawn {}: {}", program, e)))?;

        let stdin = child.stdin.take()
            .ok_or_else(|| TransportError::Io("no stdin".into()))?;
        let stdout = child.stdout.take()
            .ok_or_else(|| TransportError::Io("no stdout".into()))?;

        Ok(Self {
            stdin: tokio::sync::Mutex::new(stdin),
            stdout: tokio::sync::Mutex::new(BufReader::new(stdout)),
            child: tokio::sync::Mutex::new(child),
        })
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn send(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse, TransportError> {
        let mut json = serde_json::to_string(&request)
            .map_err(|e| TransportError::Serialization(e.to_string()))?;
        json.push('\n');

        let mut stdin = self.stdin.lock().await;
        stdin.write_all(json.as_bytes()).await
            .map_err(|e| TransportError::Io(e.to_string()))?;
        stdin.flush().await
            .map_err(|e| TransportError::Io(e.to_string()))?;

        let mut stdout = self.stdout.lock().await;
        let mut line = String::new();
        let bytes = stdout.read_line(&mut line).await
            .map_err(|e| TransportError::Io(e.to_string()))?;

        if bytes == 0 {
            return Err(TransportError::Closed);
        }

        let response: JsonRpcResponse = serde_json::from_str(line.trim())
            .map_err(|e| TransportError::Serialization(format!("parse response: {}", e)))?;

        Ok(response)
    }

    async fn close(&self) -> Result<(), TransportError> {
        let mut child = self.child.lock().await;
        let _ = child.kill().await;
        Ok(())
    }
}

/// In-memory transport for testing. Uses mpsc channels to simulate a server.
#[cfg(test)]
pub(crate) struct MockTransport {
    tx: tokio::sync::mpsc::Sender<JsonRpcRequest>,
    rx: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<JsonRpcResponse>>,
}

#[cfg(test)]
impl MockTransport {
    pub fn new() -> (
        Self,
        tokio::sync::mpsc::Receiver<JsonRpcRequest>,
        tokio::sync::mpsc::Sender<JsonRpcResponse>,
    ) {
        let (req_tx, req_rx) = tokio::sync::mpsc::channel(16);
        let (resp_tx, resp_rx) = tokio::sync::mpsc::channel(16);
        let transport = Self {
            tx: req_tx,
            rx: tokio::sync::Mutex::new(resp_rx),
        };
        (transport, req_rx, resp_tx)
    }
}

#[cfg(test)]
#[async_trait]
impl McpTransport for MockTransport {
    async fn send(&self, request: JsonRpcRequest) -> Result<JsonRpcResponse, TransportError> {
        self.tx.send(request).await
            .map_err(|_| TransportError::Closed)?;
        let mut rx = self.rx.lock().await;
        rx.recv().await.ok_or(TransportError::Closed)
    }

    async fn close(&self) -> Result<(), TransportError> {
        Ok(())
    }
}
