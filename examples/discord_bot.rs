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
use claw_lib::tools::discord::DiscordConfig;

fn system_prompt(bot_name: &str) -> String {
    format!(
        "You are {bot_name}, a helpful Discord bot. You can read messages, send messages, \
         list channels, and reply to conversations in Discord servers. \
         When asked about what's happening in a channel, use get_messages to read recent history. \
         When asked to say something, use send_message or reply_to_message. \
         Be concise and conversational — you're chatting, not writing essays. \
         Use Discord markdown formatting when appropriate."
    )
}

/// Interactive CLI mode — type messages and see how the bot would respond.
///
/// This is useful for testing your bot's behavior before connecting it to
/// a live Discord server. For a real bot, you'd replace this with a
/// Gateway WebSocket listener (using serenity, twilight, etc.) that
/// feeds incoming messages into `runtime.run()`.
#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_else(|_| {
        eprintln!("Error: ANTHROPIC_API_KEY not set.");
        std::process::exit(1);
    });

    let discord_token = std::env::var("DISCORD_TOKEN").unwrap_or_else(|_| {
        eprintln!("Error: DISCORD_TOKEN not set.");
        eprintln!("Create a bot at https://discord.com/developers/applications");
        std::process::exit(1);
    });

    let guild_id = std::env::var("DISCORD_GUILD_ID").unwrap_or_else(|_| {
        eprintln!("Error: DISCORD_GUILD_ID not set.");
        eprintln!("Right-click your server in Discord → Copy Server ID");
        std::process::exit(1);
    });

    let model = std::env::var("CLAW_MODEL").unwrap_or_else(|_| "claude-sonnet-4-5-20250929".into());
    let bot_name = std::env::var("BOT_NAME").unwrap_or_else(|_| "ClawBot".into());

    let dc = DiscordConfig::new(&discord_token);

    let provider = Arc::new(ClaudeProvider::new(&api_key, &model));
    let store = Arc::new(InMemoryStore::new());

    let mut tools = ToolRegistry::new();
    claw_lib::tools::discord::register_tools(&mut tools, &dc);

    let config = RuntimeConfig {
        system_prompt: Some(system_prompt(&bot_name)),
        max_turns: 5,
        max_tokens: Some(1024),
        temperature: Some(0.7),
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

    let ns = Namespace::new("discord").child(&guild_id);

    println!("=== {} — Discord Bot (Interactive Mode) ===", bot_name);
    println!("Guild: {}", guild_id);
    println!();
    println!("Talk to the bot as if you're a user in the Discord server.");
    println!("The bot can use Discord tools to read/send messages in your server.");
    println!("Type /quit to exit.");
    println!();

    let stdin = io::stdin();
    loop {
        print!("you> ");
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
                // Show tool calls for visibility
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
                println!("bot> {}", result.final_message.content);
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
