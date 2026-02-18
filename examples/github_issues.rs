use std::io::{self, Write};
use std::sync::Arc;

use claw_lib::context::CharEstimator;
use claw_lib::message::Message;
use claw_lib::namespace::Namespace;
use claw_lib::policy::PolicyRegistry;
use claw_lib::providers::claude::ClaudeProvider;
use claw_lib::runtime::{Runtime, RuntimeConfig};
use claw_lib::store::InMemoryStore;
use claw_lib::tool::ToolRegistry;
use claw_lib::tools::github::GitHubConfig;

fn system_prompt(owner: &str, repo: &str) -> String {
    format!(
        "You are a GitHub issues assistant for the {owner}/{repo} repository. \
         You help users browse, search, create, comment on, and close issues. \
         Always use the provided tools to interact with GitHub — never guess about issue contents. \
         Be concise. When listing issues, summarize them clearly. \
         When creating issues, write clear titles and well-formatted markdown descriptions."
    )
}

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: ANTHROPIC_API_KEY not set.");
        std::process::exit(1);
    });

    let gh_token = std::env::var("GITHUB_TOKEN").unwrap_or_else(|_| {
        eprintln!("Error: GITHUB_TOKEN not set.");
        eprintln!("Create one at https://github.com/settings/tokens with 'repo' scope.");
        std::process::exit(1);
    });

    // Parse owner/repo from args or env
    let args: Vec<String> = std::env::args().collect();
    let repo_arg = args.get(1).cloned().or_else(|| std::env::var("GITHUB_REPO").ok());

    let (owner, repo) = match repo_arg {
        Some(ref r) if r.contains('/') => {
            let parts: Vec<&str> = r.splitn(2, '/').collect();
            (parts[0].to_string(), parts[1].to_string())
        }
        _ => {
            eprintln!("Usage: github_issues <owner/repo>");
            eprintln!("  e.g. github_issues rust-lang/rust");
            eprintln!("  or set GITHUB_REPO=owner/repo");
            std::process::exit(1);
        }
    };

    let model = std::env::var("CLAW_MODEL").unwrap_or_else(|_| "claude-sonnet-4-5-20250929".into());

    let gh = GitHubConfig::new(&gh_token, &owner, &repo);

    let provider = Arc::new(ClaudeProvider::new(&api_key, &model));
    let store = Arc::new(InMemoryStore::new());

    let mut tools = ToolRegistry::new();
    claw_lib::tools::github::register_tools(&mut tools, &gh);

    let config = RuntimeConfig {
        system_prompt: Some(system_prompt(&owner, &repo)),
        max_turns: 5,
        max_tokens: Some(2048),
        temperature: Some(0.3),
        ..RuntimeConfig::default()
    };

    let runtime = Runtime::new(
        provider,
        store,
        tools,
        PolicyRegistry::default(),
        CharEstimator::default(),
        config,
    );

    let ns = Namespace::new("github").child(&format!("{}/{}", owner, repo));

    println!("=== GitHub Issues Assistant ===");
    println!("Repository: {}/{}", owner, repo);
    println!("Type your request, or /quit to exit.");
    println!();

    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if stdin.read_line(&mut input).unwrap() == 0 {
            break;
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/quit" || input == "/exit" {
            println!("Goodbye!");
            break;
        }

        match runtime.run(&ns, Message::user(input)).await {
            Ok(result) => {
                for turn in &result.turns {
                    for tc in &turn.response.message.tool_calls {
                        println!(
                            "  [{}({})]",
                            tc.name,
                            serde_json::to_string(&tc.arguments).unwrap_or_default()
                        );
                    }
                    for tr in &turn.tool_results {
                        if tr.is_error {
                            println!("  [tool error: {}]", tr.content);
                        }
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
