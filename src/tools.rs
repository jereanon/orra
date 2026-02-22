#[cfg(feature = "discord")]
pub mod discord;

#[cfg(feature = "documents")]
pub mod documents;

#[cfg(feature = "github")]
pub mod github;

#[cfg(feature = "browser")]
pub mod browser;

#[cfg(feature = "claude-code")]
pub mod claude_code;

pub mod cron;
pub mod delegation;
pub mod exec;
#[cfg(feature = "image-gen")]
pub mod image_gen;
pub mod memory;

#[cfg(feature = "web-fetch")]
pub mod web_fetch;

#[cfg(feature = "web-search")]
pub mod web_search;
