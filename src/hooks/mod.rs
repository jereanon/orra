//! Built-in hook implementations for common agent lifecycle patterns.
//!
//! These hooks can be registered with a [`HookRegistry`](crate::hook::HookRegistry)
//! to add logging, tool approval, working directory injection, and more.

pub mod approval;
pub mod logging;
pub mod working_directory;
