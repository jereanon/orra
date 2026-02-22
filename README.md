# orra

A Rust library for building AI agents. Handles sessions, context management, tool execution, hooks, and the agent loop so you can focus on your domain logic.

## Why this exists

Most AI/LLM libraries focus on model abstraction or prompt chaining. What they don't give you is a good answer to: "How do I run this for multiple users without their conversations bleeding into each other? How do I control which tools each user can access? How do I keep context from blowing past the token limit?"

agentic is built around those problems. It's designed for applications where AI is a feature, not the whole product — think project management tools, internal dashboards, customer support platforms, developer tooling. Places where you already have domain-specific data and operations, and you want users to interact with them through natural language.

## Quick start

```toml
[dependencies]
orra = { version = "0.0.2", features = ["claude"] }
```

```rust
use std::sync::Arc;
use orra::context::CharEstimator;
use orra::message::Message;
use orra::namespace::Namespace;
use orra::policy::PolicyRegistry;
use orra::providers::claude::ClaudeProvider;
use orra::runtime::{Runtime, RuntimeConfig};
use orra::store::InMemoryStore;
use orra::tool::ToolRegistry;

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

let ns = Namespace::new("user").child("bob");
let result = runtime.run(&ns, Message::user("Hello!")).await?;
println!("{}", result.final_message.content);
```

## Feature flags

| Feature | What it enables |
|---------|-----------------|
| `claude` | Anthropic Claude API with streaming |
| `openai` | OpenAI-compatible API (GPT-4o, Ollama, vLLM, etc.) |
| `discord` | Discord gateway channel and bot tools |
| `mcp` | Model Context Protocol client for external tool servers |
| `documents` | Document knowledge store with TF-IDF search |
| `github` | GitHub issue and PR tools |
| `claude-code` | Claude Code CLI delegation tool |
| `web-fetch` | Web page fetching with HTML-to-text extraction |
| `web-search` | Brave Search API tool |
| `parallel-tools` | Concurrent tool execution |
| `file-store` | File-based session persistence |
| `gateway` | HTTP/WebSocket gateway channel |

## Core concepts

**Namespaced sessions.** Every conversation lives under a namespace like `tenant:acme:user:bob`. Sessions are fully isolated. Policies cascade from parent namespaces so you can set access control at the org level.

**Token-aware context management.** The runtime tracks token budgets and auto-truncates old messages when context gets too long, keeping the system prompt and recent messages intact.

**Hooks.** The `Hook` trait lets you intercept any point in the agent lifecycle — before/after LLM calls, before/after tool execution, session load/save. The library ships with three built-in hooks:

- `hooks::logging` — logs runtime activity and tracks token usage
- `hooks::approval` — gates tool execution on user approval (with per-session "chaos mode" to auto-approve)
- `hooks::working_directory` — injects a working directory into `exec` and `claude_code` tool calls from session metadata

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
       (LLM)   (ops)  (persist) (lifecycle) (access)
```

## Modules

**Runtime:**
- `runtime` — Agent loop: load session, build context, call LLM, execute tools, save
- `provider` — `Provider` and `StreamingProvider` traits
- `tool` — `Tool` trait and `ToolRegistry`
- `store` — `SessionStore` trait with `InMemoryStore`
- `context` — Token budgets and automatic context truncation
- `namespace` — Hierarchical session keys
- `message` — Messages, tool calls, tool results
- `policy` — Per-namespace tool allow/deny lists
- `hook` — Lifecycle hook trait and `HookRegistry`
- `hooks` — Built-in hook implementations (logging, approval, working directory)

**Providers:**
- `providers::claude` — Anthropic Claude API with streaming
- `providers::openai` — OpenAI Chat Completions (works with any compatible endpoint)
- `providers::dynamic` — Hot-swappable provider wrapper

**Tools:**
- `tools::exec` — Shell command execution with allowlist and timeout
- `tools::web_fetch` — Web page fetching with HTML-to-text extraction
- `tools::web_search` — Brave Search API
- `tools::browser` — Web page reading with readability extraction
- `tools::discord` — Discord bot tools (channels, messages, guild info)
- `tools::github` — GitHub issue/PR management
- `tools::documents` — Document search and retrieval (TF-IDF)
- `tools::image_gen` — DALL-E image generation
- `tools::delegation` — Sub-agent spawning for complex subtasks
- `tools::memory` — Remember/recall/forget for persistent context
- `tools::claude_code` — Delegate coding tasks to the Claude Code CLI
- `tools::cron` — AI-managed scheduled tasks

**Channels:**
- `channels::discord` — Discord Gateway WebSocket
- `channels::gateway` — HTTP REST + WebSocket gateway

**Infrastructure:**
- `mcp` — Model Context Protocol client for external tool servers
- `memory` — Long-term memory with keyword search
- `scheduler` — Cron-based task scheduling
- `routing` — Message routing across channels and runtimes
- `metrics` — Counters, gauges, histograms with pluggable sinks
- `auth` — OAuth2 token management
- `voice` — TTS and STT traits

## Hooks

Register hooks to observe or modify behavior at any lifecycle point:

```rust
use std::sync::Arc;
use orra::hook::HookRegistry;
use orra::hooks::logging::LoggingHook;
use orra::hooks::working_directory::WorkingDirectoryHook;

let mut hooks = HookRegistry::new();
hooks.register(Arc::new(LoggingHook::new()));
hooks.register(Arc::new(WorkingDirectoryHook::new()));

let mut runtime = Runtime::new(/* ... */);
runtime.set_hooks(hooks);
```

The approval hook requires a channel to communicate with your UI:

```rust
use orra::hooks::approval::{ApprovalHook, ApprovalRequest};

let (tx, rx) = tokio::sync::mpsc::channel::<ApprovalRequest>(32);
hooks.register(Arc::new(ApprovalHook::new(tx)));

// In your WebSocket/UI handler, receive from `rx` and send back
// true (approved) or false (denied) via the oneshot channel.
```

## Using with OpenAI

```rust
use orra::providers::openai::OpenAIProvider;

// OpenAI
let provider = OpenAIProvider::new("your-key", "gpt-4o");

// Local server (Ollama, vLLM, etc.)
let provider = OpenAIProvider::new("not-needed", "llama3")
    .with_api_url("http://localhost:11434/v1");
```

## MCP integration

Connect to external tool servers via the Model Context Protocol:

```rust
use orra::mcp::transport::StdioTransport;
use orra::mcp::register_mcp_tools;

let transport = Arc::new(
    StdioTransport::spawn("npx", &["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
        .await?
);

let mut tools = ToolRegistry::new();
register_mcp_tools(&mut tools, transport).await?;
```

## Examples

See the [examples/](examples/) directory:

- **[kanban](examples/kanban.rs)** — CLI kanban board assistant with multi-user namespacing and role-based tool policies
- **[discord_bot](examples/discord_bot.rs)** — Discord bot using the Discord channel and tools
- **[github_issues](examples/github_issues.rs)** — Manage GitHub issues through natural language

```bash
ANTHROPIC_API_KEY=... cargo run --features claude --example kanban
ANTHROPIC_API_KEY=... DISCORD_TOKEN=... cargo run --features claude,discord --example discord_bot
ANTHROPIC_API_KEY=... GITHUB_TOKEN=... cargo run --features claude,github --example github_issues -- owner/repo
```

## License

MIT
