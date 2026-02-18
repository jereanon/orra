# claw-lib

A Rust library for adding AI assistant capabilities to any application. You bring the tools and resources — claw-lib handles context management, session isolation, and the agent execution loop.

## Why this exists

Most AI/LLM libraries focus on model abstraction or prompt chaining. What they don't give you is a good answer to: "How do I run this for multiple users without their conversations bleeding into each other? How do I control which tools each user can access? How do I keep context from blowing past the token limit?"

claw-lib is built around those problems. It's designed for applications where AI is a feature, not the whole product — think project management tools, internal dashboards, customer support platforms, developer tooling. Places where you already have domain-specific data and operations, and you want users to interact with them through natural language.

## What it does

**Context namespacing.** Every conversation lives under a hierarchical namespace like `company:engineering:alice` or `tenant:acme:user:bob:channel:support`. Sessions are fully isolated. Namespaces support parent-child relationships, so you can set policies at the tenant level and have them flow down to all users.

**Token-aware context management.** The library tracks token budgets, automatically truncates conversation history when it gets too long (keeping the system prompt and most recent messages), and caps tool result sizes. You plug in a tokenizer — or use the built-in character estimator for quick prototyping.

**Tool policies.** Allow-list or deny-list tools per namespace. Set a policy on `company:viewer` and every namespace under it inherits read-only access. Override it for specific users when needed. Policies resolve by walking up the namespace tree.

**Pluggable everything.** The `Provider` trait is how you connect an LLM. The `Tool` trait is how you expose operations. The `SessionStore` trait is how you persist conversations. Implement what you need, use the defaults for the rest.

## Architecture

```
                    +-----------+
                    |  Runtime  |  orchestrates the agent loop
                    +-----+-----+
                          |
          +---------------+---------------+
          |               |               |
    +-----+-----+  +-----+-----+  +------+------+
    |  Provider  |  |   Tools   |  |    Store    |
    | (LLM API)  |  | (your ops)|  | (sessions)  |
    +-----+-----+  +-----+-----+  +------+------+
          |               |               |
    Claude, OpenAI,   anything you     in-memory,
    local, etc.       implement        filesystem,
                                       Redis, etc.
```

**Modules:**

- **`namespace`** — Hierarchical keys for isolating contexts (`Namespace::new("acme").child("alice")`)
- **`message`** — Message types with roles, tool calls, and tool results
- **`tool`** — `Tool` trait and `ToolRegistry` for registering executable tools
- **`provider`** — `Provider` trait that wraps any LLM API
- **`context`** — Token budgets, automatic truncation, tool result size limits
- **`store`** — `SessionStore` trait with a built-in `InMemoryStore`
- **`policy`** — Per-namespace tool visibility with hierarchical inheritance
- **`runtime`** — The agent loop: load session, build context, call LLM, execute tools, persist
- **`providers::claude`** — Ready-to-use Claude API implementation (feature-gated)
- **`tools::github`** — GitHub Issues tools: list, search, get, create, comment, close (feature-gated)
- **`tools::documents`** — Document retrieval tools with pluggable search backend (feature-gated)

## Built-in tools

The library ships with optional tool sets you can drop into your application. Each lives behind a feature flag so you only pull in the dependencies you need.

**Document Retrieval** (`documents` feature) — Three tools for searching and reading documents: `search_documents`, `read_document`, `list_documents`. You provide a `DocumentStore` implementation — the library includes `InMemoryDocumentStore` with TF-IDF search for prototyping, and you can swap in a vector database or full-text search engine for production.

```rust
use std::sync::Arc;
use claw_lib::tools::documents::{InMemoryDocumentStore, Document, register_tools};

let store = Arc::new(InMemoryDocumentStore::new());
// Load your documents into the store...
let mut tools = ToolRegistry::new();
register_tools(&mut tools, store);
```

**GitHub Issues** (`github` feature) — Six tools for managing GitHub issues via the REST API. Configure with a token and repo, register into your `ToolRegistry`, done.

```rust
use claw_lib::tools::github::GitHubConfig;

let gh = GitHubConfig::new("your-github-token", "owner", "repo");
let mut tools = ToolRegistry::new();
claw_lib::tools::github::register_tools(&mut tools, &gh);
```

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
claw-lib = { path = "." }  # or from your registry
```

Basic usage:

```rust
use std::sync::Arc;
use claw_lib::context::CharEstimator;
use claw_lib::message::Message;
use claw_lib::namespace::Namespace;
use claw_lib::policy::PolicyRegistry;
use claw_lib::providers::claude::ClaudeProvider;
use claw_lib::runtime::{Runtime, RuntimeConfig};
use claw_lib::store::InMemoryStore;
use claw_lib::tool::ToolRegistry;

let provider = Arc::new(ClaudeProvider::new("your-api-key", "claude-sonnet-4-5-20250929"));
let store = Arc::new(InMemoryStore::new());
let tools = ToolRegistry::new(); // register your tools here

let runtime = Runtime::new(
    provider,
    store,
    tools,
    PolicyRegistry::default(),
    CharEstimator::default(),
    RuntimeConfig {
        system_prompt: Some("You are a helpful assistant.".into()),
        ..RuntimeConfig::default()
    },
);

let ns = Namespace::new("myapp").child("user-123");
let result = runtime.run(&ns, Message::user("Hello!")).await?;
println!("{}", result.final_message.content);
```

## Examples

### [Kanban board assistant](examples/kanban.rs)

A CLI project management assistant with an in-memory kanban board. Shows multi-user namespacing, tool policies (viewer vs. member roles), and session persistence across user switches.

```bash
ANTHROPIC_API_KEY=... cargo run --features claude --example kanban
```

### [GitHub issues assistant](examples/github_issues.rs)

A CLI for managing GitHub issues through natural language. Uses the built-in GitHub tools from the library.

```bash
ANTHROPIC_API_KEY=... GITHUB_TOKEN=... cargo run --features claude,github --example github_issues -- owner/repo
```

## Status

Early development. The core abstractions are in place and tested, but the API will evolve. Notable things not yet implemented:

- Streaming responses
- LLM-based context compaction (summarizing old messages instead of dropping them)
- Filesystem session store
- OpenAI-compatible provider
- Async tool execution (parallel tool calls)

## License

MIT
