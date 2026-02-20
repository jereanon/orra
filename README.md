# agentic-rs

[![CI](https://github.com/agentic-rs/agentic/actions/workflows/ci.yml/badge.svg)](https://github.com/agentic-rs/agentic/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/agentic-rs.svg)](https://crates.io/crates/agentic-rs)
[![docs.rs](https://docs.rs/agentic-rs/badge.svg)](https://docs.rs/agentic-rs)

A Rust library for building AI-powered agents. Handles context management, session isolation, tool execution, and the agent loop so you can focus on your domain logic.

## Why this exists

Most AI/LLM libraries focus on model abstraction or prompt chaining. What they don't give you is a good answer to: "How do I run this for multiple users without their conversations bleeding into each other? How do I control which tools each user can access? How do I keep context from blowing past the token limit?"

agentic is built around those problems. It's designed for applications where AI is a feature, not the whole product — think project management tools, internal dashboards, customer support platforms, developer tooling. Places where you already have domain-specific data and operations, and you want users to interact with them through natural language.

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
agentic-rs = { version = "0.1", features = ["claude"] }
```

Basic usage:

```rust
use std::sync::Arc;
use agentic_rs::providers::claude::ClaudeProvider;
use agentic_rs::runtime::{Runtime, RuntimeConfig};
use agentic_rs::store::InMemoryStore;
use agentic_rs::tool::ToolRegistry;
use agentic_rs::policy::PolicyRegistry;
use agentic_rs::context::CharEstimator;

let provider = Arc::new(ClaudeProvider::new("your-api-key", "claude-sonnet-4-5-20250929"));
let store = Arc::new(InMemoryStore::new());

let runtime = Runtime::new(
    provider,
    store,
    ToolRegistry::new(),
    PolicyRegistry::default(),
    CharEstimator::default(),
    RuntimeConfig {
        system_prompt: Some("You are a helpful assistant.".into()),
        ..RuntimeConfig::default()
    },
);

let result = runtime.send_message("Hello!", "user-123").await?;
println!("{}", result);
```

## Feature flags

| Feature | Description |
|---------|-------------|
| `claude` | Claude provider (Anthropic API) |
| `openai` | OpenAI-compatible provider (GPT-4o, local LLMs, etc.) |
| `discord` | Discord channel and bot tools |
| `mcp` | Model Context Protocol client |
| `documents` | Document knowledge store with TF-IDF search |
| `github` | GitHub issue and PR management tools |
| `parallel-tools` | Concurrent tool execution |
| `file-store` | File-based session persistence |
| `gateway` | HTTP/WebSocket gateway channel |

## Core concepts

**Namespaced sessions.** Every conversation lives under a hierarchical namespace like `tenant:acme:user:bob`. Sessions are fully isolated. Policies flow down from parent namespaces so you can set access control at the org level.

**Token-aware context management.** Tracks token budgets, auto-truncates history when it gets too long (keeping system prompt and recent messages), and caps tool result sizes.

**Pluggable everything.** The `Provider` trait wraps any LLM. The `Tool` trait exposes operations. The `SessionStore` trait handles persistence. Implement what you need, use the defaults for the rest.

## Architecture

```
                    +-----------+
                    |  Runtime  |  orchestrates the agent loop
                    +-----+-----+
                          |
          +-------+-------+-------+-------+
          |       |       |       |       |
       Provider  Tools   Store  Hooks  Policies
       (LLM)   (ops)  (persist) (events) (access)
```

## Modules

**Core:**
- `runtime` — Agent loop: load session, build context, call LLM, execute tools, persist
- `provider` — `Provider` and `StreamingProvider` traits for LLM backends
- `tool` — `Tool` trait and `ToolRegistry`
- `store` — `SessionStore` trait with `InMemoryStore`
- `context` — Token budgets and automatic truncation
- `namespace` — Hierarchical keys for session isolation
- `message` — Message types with roles, tool calls, tool results
- `policy` — Per-namespace tool allow/deny lists
- `hook` — Pre/post message, tool call, and error hooks

**Providers:**
- `providers::claude` — Anthropic Claude API with streaming
- `providers::openai` — OpenAI Chat Completions API (works with any compatible endpoint)

**Channels:**
- `channels::discord` — Discord Gateway WebSocket channel
- `channels::gateway` — HTTP REST + WebSocket gateway

**Tools:**
- `tools::discord` — Discord bot tools (channels, messages, guild info)
- `tools::github` — GitHub issue/PR tools
- `tools::documents` — Document search and retrieval
- `tools::browser` — Web page reading with HTML content extraction
- `tools::image_gen` — Image generation (DALL-E)
- `tools::delegation` — Sub-agent spawning for complex subtasks
- `tools::memory` — Remember/recall/forget for long-term context

**Infrastructure:**
- `mcp` — Model Context Protocol client for external tool servers
- `memory` — Long-term memory with keyword and semantic search
- `scheduler` — Cron-based task scheduling
- `routing` — Message routing across channels and runtimes
- `metrics` — Counters, gauges, histograms with pluggable sinks
- `auth` — OAuth2 token management
- `plugin` — Plugin system with lifecycle management
- `project` — Multi-project management
- `voice` — TTS and STT traits

## Using with OpenAI

```rust
use agentic_rs::providers::openai::OpenAIProvider;

// OpenAI
let provider = OpenAIProvider::new("your-key", "gpt-4o");

// Local server (Ollama, vLLM, etc.)
let provider = OpenAIProvider::new("not-needed", "llama3")
    .with_api_url("http://localhost:11434/v1");
```

## MCP integration

Connect to external tool servers via the Model Context Protocol:

```rust
use agentic_rs::mcp::transport::StdioTransport;
use agentic_rs::mcp::register_mcp_tools;

let transport = Arc::new(
    StdioTransport::spawn("npx", &["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
        .await?
);

let mut tools = ToolRegistry::new();
register_mcp_tools(&mut tools, transport).await?;
// All server tools are now available in the registry
```

## Examples

See the [examples/](examples/) directory:

- **[kanban](examples/kanban.rs)** — CLI kanban board assistant with multi-user namespacing and role-based tool policies
- **[discord_bot](examples/discord_bot.rs)** — Live Discord bot using `DiscordChannel` + `ChannelAdapter`
- **[github_issues](examples/github_issues.rs)** — CLI for managing GitHub issues through natural language

```bash
# Kanban board
ANTHROPIC_API_KEY=... cargo run --features claude --example kanban

# Discord bot
ANTHROPIC_API_KEY=... DISCORD_TOKEN=... cargo run --features claude,discord --example discord_bot

# GitHub issues
ANTHROPIC_API_KEY=... GITHUB_TOKEN=... cargo run --features claude,github --example github_issues -- owner/repo
```

## License

MIT
