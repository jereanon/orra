use std::collections::BTreeMap;
use std::io::{self, Write};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use agentic_rs::context::CharEstimator;
use agentic_rs::message::Message;
use agentic_rs::namespace::Namespace;
use agentic_rs::policy::{PolicyRegistry, ToolPolicy};
use agentic_rs::providers::claude::ClaudeProvider;
use agentic_rs::runtime::{Runtime, RuntimeConfig};
use agentic_rs::store::{InMemoryStore, SessionStore};
use agentic_rs::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

// ---------------------------------------------------------------------------
// Kanban board — shared mutable state behind Arc<RwLock<..>>
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Task {
    id: u32,
    title: String,
    assignee: Option<String>,
}

#[derive(Debug, Clone)]
struct Board {
    columns: BTreeMap<String, Vec<Task>>,
    next_id: u32,
}

impl Board {
    fn new() -> Self {
        let mut columns = BTreeMap::new();
        columns.insert("backlog".into(), vec![
            Task { id: 1, title: "Design login page".into(), assignee: None },
            Task { id: 2, title: "Set up CI pipeline".into(), assignee: None },
            Task { id: 3, title: "Write API docs".into(), assignee: None },
        ]);
        columns.insert("in-progress".into(), vec![]);
        columns.insert("done".into(), vec![]);
        Self { columns, next_id: 4 }
    }
}

type SharedBoard = Arc<RwLock<Board>>;

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

struct ListColumnsTool {
    board: SharedBoard,
}

#[async_trait]
impl Tool for ListColumnsTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_columns".into(),
            description: "List all columns on the kanban board and the number of tasks in each."
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {},
            }),
        }
    }

    async fn execute(&self, _input: serde_json::Value) -> Result<String, ToolError> {
        let board = self.board.read().await;
        let mut lines = Vec::new();
        for (name, tasks) in &board.columns {
            lines.push(format!("- {} ({} tasks)", name, tasks.len()));
        }
        Ok(lines.join("\n"))
    }
}

struct ListTasksTool {
    board: SharedBoard,
}

#[async_trait]
impl Tool for ListTasksTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_tasks".into(),
            description:
                "List tasks in a specific column. If no column is provided, lists all tasks."
                    .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name (e.g. 'backlog', 'in-progress', 'done'). Omit to list all."
                    }
                }
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let board = self.board.read().await;
        let column = input.get("column").and_then(|v| v.as_str());

        let mut lines = Vec::new();
        for (col_name, tasks) in &board.columns {
            if let Some(filter) = column {
                if col_name != filter {
                    continue;
                }
            }
            lines.push(format!("[{}]", col_name));
            if tasks.is_empty() {
                lines.push("  (empty)".into());
            }
            for task in tasks {
                let assignee = task
                    .assignee
                    .as_deref()
                    .map(|a| format!(" (assigned to {})", a))
                    .unwrap_or_default();
                lines.push(format!("  #{}: {}{}", task.id, task.title, assignee));
            }
        }

        if lines.is_empty() {
            return Err(ToolError::InvalidInput(format!(
                "Column '{}' not found. Use list_columns to see available columns.",
                column.unwrap_or("?")
            )));
        }

        Ok(lines.join("\n"))
    }
}

struct CreateTaskTool {
    board: SharedBoard,
}

#[async_trait]
impl Tool for CreateTaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "create_task".into(),
            description: "Create a new task in the backlog (or a specified column).".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title of the new task"
                    },
                    "column": {
                        "type": "string",
                        "description": "Column to add the task to. Defaults to 'backlog'."
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Optional person to assign the task to"
                    }
                },
                "required": ["title"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let title = input
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'title'".into()))?;

        let column = input
            .get("column")
            .and_then(|v| v.as_str())
            .unwrap_or("backlog");

        let assignee = input
            .get("assignee")
            .and_then(|v| v.as_str())
            .map(String::from);

        let mut board = self.board.write().await;

        if !board.columns.contains_key(column) {
            return Err(ToolError::InvalidInput(format!("column '{}' not found", column)));
        }

        let id = board.next_id;
        board.next_id += 1;

        board.columns.get_mut(column).unwrap().push(Task {
            id,
            title: title.to_string(),
            assignee,
        });

        Ok(format!("Created task #{} '{}' in '{}'", id, title, column))
    }
}

struct MoveTaskTool {
    board: SharedBoard,
}

#[async_trait]
impl Tool for MoveTaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "move_task".into(),
            description: "Move a task from one column to another. Identify the task by its ID or title.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "ID number of the task to move"
                    },
                    "task_title": {
                        "type": "string",
                        "description": "Title of the task to move (used if task_id not provided)"
                    },
                    "to": {
                        "type": "string",
                        "description": "Destination column name"
                    }
                },
                "required": ["to"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let dest = input
            .get("to")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'to' column".into()))?;

        let task_id = input.get("task_id").and_then(|v| v.as_u64()).map(|v| v as u32);
        let task_title = input.get("task_title").and_then(|v| v.as_str());

        if task_id.is_none() && task_title.is_none() {
            return Err(ToolError::InvalidInput(
                "provide either 'task_id' or 'task_title'".into(),
            ));
        }

        let mut board = self.board.write().await;

        if !board.columns.contains_key(dest) {
            return Err(ToolError::InvalidInput(format!(
                "destination column '{}' not found",
                dest
            )));
        }

        // Find and remove the task from its current column
        let mut found_task: Option<Task> = None;
        let mut source_col = String::new();

        for (col_name, tasks) in board.columns.iter_mut() {
            let pos = tasks.iter().position(|t| {
                if let Some(id) = task_id {
                    t.id == id
                } else if let Some(title) = task_title {
                    t.title.to_lowercase().contains(&title.to_lowercase())
                } else {
                    false
                }
            });

            if let Some(idx) = pos {
                found_task = Some(tasks.remove(idx));
                source_col = col_name.clone();
                break;
            }
        }

        let task = found_task.ok_or_else(|| {
            ToolError::ExecutionFailed(format!(
                "task not found (id={:?}, title={:?})",
                task_id, task_title
            ))
        })?;

        let task_desc = format!("#{} '{}'", task.id, task.title);
        board.columns.get_mut(dest).unwrap().push(task);

        Ok(format!("Moved {} from '{}' to '{}'", task_desc, source_col, dest))
    }
}

struct AssignTaskTool {
    board: SharedBoard,
}

#[async_trait]
impl Tool for AssignTaskTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "assign_task".into(),
            description: "Assign or reassign a task to a person.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "ID number of the task"
                    },
                    "assignee": {
                        "type": "string",
                        "description": "Name of the person to assign the task to"
                    }
                },
                "required": ["task_id", "assignee"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let task_id = input
            .get("task_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or_else(|| ToolError::InvalidInput("missing 'task_id'".into()))?;

        let assignee = input
            .get("assignee")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'assignee'".into()))?;

        let mut board = self.board.write().await;

        for tasks in board.columns.values_mut() {
            if let Some(task) = tasks.iter_mut().find(|t| t.id == task_id) {
                let old = task.assignee.replace(assignee.to_string());
                return Ok(format!(
                    "Assigned #{} '{}' to {} (was: {})",
                    task.id,
                    task.title,
                    assignee,
                    old.unwrap_or_else(|| "unassigned".into())
                ));
            }
        }

        Err(ToolError::ExecutionFailed(format!(
            "task #{} not found",
            task_id
        )))
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn build_tools(board: &SharedBoard) -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ListColumnsTool { board: board.clone() }));
    tools.register(Box::new(ListTasksTool { board: board.clone() }));
    tools.register(Box::new(CreateTaskTool { board: board.clone() }));
    tools.register(Box::new(MoveTaskTool { board: board.clone() }));
    tools.register(Box::new(AssignTaskTool { board: board.clone() }));
    tools
}

fn build_policies() -> PolicyRegistry {
    // Viewers can only read, not mutate
    let mut policies = PolicyRegistry::default();
    let viewer_ns = Namespace::parse("company:viewer");
    policies.set_policy(
        &viewer_ns,
        ToolPolicy::AllowList(vec!["list_columns".into(), "list_tasks".into()]),
    );
    policies
}

const SYSTEM_PROMPT: &str = "\
You are a project management assistant for a small development team. \
You help manage a kanban board with tasks across columns. \
Be concise and helpful. When the user asks about tasks, use the provided tools. \
Always use the tools rather than guessing about the board state.";

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: ANTHROPIC_API_KEY environment variable not set.");
        eprintln!("Set it with: export ANTHROPIC_API_KEY=your-key-here");
        std::process::exit(1);
    });

    let model = std::env::var("AGENTIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-5-20250929".into());

    let board = Arc::new(RwLock::new(Board::new()));
    let store = Arc::new(InMemoryStore::new());
    let provider = Arc::new(ClaudeProvider::new(&api_key, &model));

    let tools = build_tools(&board);
    let policies = build_policies();

    let config = RuntimeConfig {
        system_prompt: Some(SYSTEM_PROMPT.into()),
        max_turns: 5,
        max_tokens: Some(1024),
        temperature: Some(0.3),
        ..RuntimeConfig::default()
    };

    let runtime = Runtime::new(
        provider,
        store.clone(),
        tools,
        policies,
        CharEstimator::default(),
        config,
    );

    // Parse --user flag, default to "alice"
    let args: Vec<String> = std::env::args().collect();
    let mut current_user = "alice".to_string();
    let mut current_role = "member".to_string();

    for i in 0..args.len() {
        if args[i] == "--user" {
            if let Some(name) = args.get(i + 1) {
                current_user = name.clone();
            }
        }
        if args[i] == "--role" {
            if let Some(role) = args.get(i + 1) {
                current_role = role.clone();
            }
        }
    }

    println!("=== Kanban PM Assistant ===");
    println!("User: {} (role: {})", current_user, current_role);
    println!("Board: 3 tasks in backlog, 0 in-progress, 0 done");
    println!();
    println!("Commands:");
    println!("  /switch <user> [role]  — Switch to a different user");
    println!("  /board                 — Print the raw board state");
    println!("  /quit                  — Exit");
    println!();

    let stdin = io::stdin();
    loop {
        print!("[{}]> ", current_user);
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if stdin.read_line(&mut input).unwrap() == 0 {
            break; // EOF
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Handle slash commands
        if input == "/quit" || input == "/exit" {
            println!("Goodbye!");
            break;
        }

        if input == "/board" {
            let board = board.read().await;
            for (col, tasks) in &board.columns {
                println!("[{}]", col);
                if tasks.is_empty() {
                    println!("  (empty)");
                }
                for task in tasks {
                    let a = task
                        .assignee
                        .as_deref()
                        .map(|a| format!(" -> {}", a))
                        .unwrap_or_default();
                    println!("  #{}: {}{}", task.id, task.title, a);
                }
            }
            println!();
            continue;
        }

        if input.starts_with("/switch") {
            let parts: Vec<&str> = input.split_whitespace().collect();
            if parts.len() >= 2 {
                current_user = parts[1].to_string();
                current_role = parts.get(2).unwrap_or(&"member").to_string();
                println!("Switched to user: {} (role: {})", current_user, current_role);

                // Show if session exists
                let ns = Namespace::new("company")
                    .child(&current_role)
                    .child(&current_user);
                if let Ok(Some(session)) = store.load(&ns).await {
                    println!("  (restored session with {} messages)", session.message_count());
                } else {
                    println!("  (new session)");
                }
            } else {
                println!("Usage: /switch <username> [role]");
            }
            println!();
            continue;
        }

        // Build namespace: company:<role>:<user>
        let ns = Namespace::new("company")
            .child(&current_role)
            .child(&current_user);

        match runtime.run(&ns, Message::user(input)).await {
            Ok(result) => {
                // Show tool usage
                for turn in &result.turns {
                    for tr in &turn.tool_results {
                        if tr.is_error {
                            println!("  [tool error: {}]", tr.content);
                        }
                    }
                    for tc in &turn.response.message.tool_calls {
                        println!(
                            "  [tool: {}({})]",
                            tc.name,
                            serde_json::to_string(&tc.arguments).unwrap_or_default()
                        );
                    }
                }

                println!();
                println!("{}", result.final_message.content);
                println!(
                    "  ({} input, {} output tokens)",
                    result.total_usage.input_tokens, result.total_usage.output_tokens
                );
                println!();
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                println!();
            }
        }
    }
}
