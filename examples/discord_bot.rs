use std::sync::Arc;

use orra::channels::discord::{DiscordChannel, DiscordChannelConfig, MessageFilter};
use orra::channels::ChannelAdapter;
use orra::context::CharEstimator;
use orra::policy::PolicyRegistry;
use orra::providers::claude::ClaudeProvider;
use orra::runtime::{Runtime, RuntimeConfig};
use orra::store::InMemoryStore;
use orra::tool::ToolRegistry;
use orra::tools::discord::DiscordConfig;

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

    let model =
        std::env::var("AGENTIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-5-20250929".into());
    let bot_name = std::env::var("BOT_NAME").unwrap_or_else(|_| "AgenticBot".into());

    // Pass --all to process all messages, default is mentions only
    let filter = if std::env::args().any(|a| a == "--all") {
        MessageFilter::All
    } else {
        MessageFilter::MentionsOnly
    };

    let dc = DiscordConfig::new(&discord_token);

    let provider = Arc::new(ClaudeProvider::new(&api_key, &model));
    let store = Arc::new(InMemoryStore::new());

    let mut tools = ToolRegistry::new();
    orra::tools::discord::register_tools(&mut tools, &dc);

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

    // Set up the Discord Gateway channel
    let channel_config = DiscordChannelConfig::new(dc).with_filter(filter);
    let channel = DiscordChannel::new(channel_config);

    println!("=== {} — Discord Bot ===", bot_name);
    println!("Connecting to Discord Gateway...");

    match channel.connect().await {
        Ok(()) => println!("Connected! Listening for messages..."),
        Err(e) => {
            eprintln!("Failed to connect: {}", e);
            std::process::exit(1);
        }
    }

    println!("Press Ctrl+C to stop.");
    println!();

    // Run the channel adapter loop — receives Discord messages, processes
    // them through the runtime, and sends responses back to Discord.
    if let Err(e) = ChannelAdapter::run(&channel, &runtime).await {
        eprintln!("Runtime error: {}", e);
    }
}
