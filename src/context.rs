use crate::message::Message;

pub trait Tokenizer: Send + Sync {
    fn count_tokens(&self, text: &str) -> usize;
}

/// Simple character-based estimator for when no real tokenizer is available.
/// Roughly 4 characters per token is a common heuristic.
pub struct CharEstimator {
    pub chars_per_token: usize,
}

impl Default for CharEstimator {
    fn default() -> Self {
        Self { chars_per_token: 4 }
    }
}

impl Tokenizer for CharEstimator {
    fn count_tokens(&self, text: &str) -> usize {
        (text.len() + self.chars_per_token - 1) / self.chars_per_token
    }
}

#[derive(Debug, Clone)]
pub struct ContextBudget {
    pub max_tokens: usize,
    pub reserved_for_output: usize,
    pub tool_result_share: f32,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            max_tokens: 200_000,
            reserved_for_output: 4_096,
            tool_result_share: 0.3,
        }
    }
}

impl ContextBudget {
    pub fn available_for_input(&self) -> usize {
        self.max_tokens.saturating_sub(self.reserved_for_output)
    }

    pub fn max_tool_result_tokens(&self) -> usize {
        (self.available_for_input() as f32 * self.tool_result_share) as usize
    }
}

pub struct ContextWindow<T: Tokenizer> {
    tokenizer: T,
    budget: ContextBudget,
}

impl<T: Tokenizer> ContextWindow<T> {
    pub fn new(tokenizer: T, budget: ContextBudget) -> Self {
        Self { tokenizer, budget }
    }

    pub fn budget(&self) -> &ContextBudget {
        &self.budget
    }

    pub fn estimate_message_tokens(&self, message: &Message) -> usize {
        let mut tokens = self.tokenizer.count_tokens(&message.content);

        for tc in &message.tool_calls {
            tokens += self.tokenizer.count_tokens(&tc.name);
            tokens += self.tokenizer.count_tokens(&tc.arguments.to_string());
            tokens += 5; // overhead per tool call
        }

        for tr in &message.tool_results {
            tokens += self.tokenizer.count_tokens(&tr.content);
            tokens += 5; // overhead per tool result
        }

        tokens += 4; // role + message framing overhead

        tokens
    }

    pub fn estimate_total_tokens(&self, messages: &[Message]) -> usize {
        messages.iter().map(|m| self.estimate_message_tokens(m)).sum()
    }

    pub fn is_over_budget(&self, messages: &[Message]) -> bool {
        self.estimate_total_tokens(messages) > self.budget.available_for_input()
    }

    pub fn tokens_remaining(&self, messages: &[Message]) -> usize {
        let used = self.estimate_total_tokens(messages);
        self.budget.available_for_input().saturating_sub(used)
    }

    pub fn truncate_to_fit(&self, messages: &[Message]) -> Vec<Message> {
        let budget = self.budget.available_for_input();
        let total = self.estimate_total_tokens(messages);

        if total <= budget {
            return messages.to_vec();
        }

        // Always keep the first message (system prompt) and the last message (current user input).
        // Drop oldest non-system messages from the front until we fit.
        if messages.len() <= 2 {
            return messages.to_vec();
        }

        let first = &messages[0];
        let last = &messages[messages.len() - 1];
        let fixed_cost = self.estimate_message_tokens(first) + self.estimate_message_tokens(last);

        let middle = &messages[1..messages.len() - 1];
        let mut kept: Vec<&Message> = Vec::new();
        let mut used = fixed_cost;

        // Walk backward through the middle to keep the most recent messages
        for msg in middle.iter().rev() {
            let msg_tokens = self.estimate_message_tokens(msg);
            if used + msg_tokens <= budget {
                kept.push(msg);
                used += msg_tokens;
            } else {
                break;
            }
        }

        kept.reverse();

        let mut result = vec![first.clone()];
        result.extend(kept.into_iter().cloned());
        result.push(last.clone());
        result
    }

    pub fn truncate_tool_result(&self, content: &str) -> String {
        let max_tokens = self.budget.max_tool_result_tokens();
        let current_tokens = self.tokenizer.count_tokens(content);

        if current_tokens <= max_tokens {
            return content.to_string();
        }

        // Estimate how many characters we can keep.
        // Use a rough ratio based on the tokenizer.
        let ratio = max_tokens as f32 / current_tokens as f32;
        let keep_chars = (content.len() as f32 * ratio) as usize;

        // Try to cut at a newline boundary for cleaner output
        let truncated = &content[..keep_chars.min(content.len())];
        let cut_point = truncated.rfind('\n').unwrap_or(keep_chars.min(content.len()));

        format!(
            "{}\n\n[truncated — original was {} tokens, limit is {}]",
            &content[..cut_point],
            current_tokens,
            max_tokens
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_window() -> ContextWindow<CharEstimator> {
        ContextWindow::new(
            CharEstimator::default(),
            ContextBudget {
                max_tokens: 100,
                reserved_for_output: 20,
                tool_result_share: 0.3,
            },
        )
    }

    #[test]
    fn char_estimator_basic() {
        let est = CharEstimator::default();
        assert_eq!(est.count_tokens("hello"), 2); // 5 chars / 4 = 2 (ceil)
        assert_eq!(est.count_tokens(""), 0);
        assert_eq!(est.count_tokens("a"), 1);
        assert_eq!(est.count_tokens("abcd"), 1);
        assert_eq!(est.count_tokens("abcde"), 2);
    }

    #[test]
    fn budget_available_for_input() {
        let budget = ContextBudget {
            max_tokens: 200_000,
            reserved_for_output: 4_096,
            tool_result_share: 0.3,
        };
        assert_eq!(budget.available_for_input(), 195_904);
    }

    #[test]
    fn budget_max_tool_result_tokens() {
        let budget = ContextBudget {
            max_tokens: 100,
            reserved_for_output: 20,
            tool_result_share: 0.3,
        };
        assert_eq!(budget.max_tool_result_tokens(), 24); // 80 * 0.3
    }

    #[test]
    fn estimate_simple_message_tokens() {
        let window = test_window();
        let msg = Message::user("Hello world"); // 11 chars -> 3 tokens + 4 overhead = 7
        let tokens = window.estimate_message_tokens(&msg);
        assert_eq!(tokens, 7);
    }

    #[test]
    fn estimate_total_tokens() {
        let window = test_window();
        let messages = vec![
            Message::system("You are helpful."),
            Message::user("Hi"),
        ];
        let total = window.estimate_total_tokens(&messages);
        // system: ceil(16/4) + 4 = 8, user: ceil(2/4) + 4 = 5
        assert_eq!(total, 13);
    }

    #[test]
    fn is_over_budget() {
        let window = test_window(); // 80 tokens available

        let small = vec![Message::user("Hi")];
        assert!(!window.is_over_budget(&small));

        // Create enough messages to exceed 80 tokens
        let large: Vec<Message> = (0..20)
            .map(|i| Message::user(format!("This is message number {} with some content", i)))
            .collect();
        assert!(window.is_over_budget(&large));
    }

    #[test]
    fn tokens_remaining() {
        let window = test_window(); // 80 tokens available
        let messages = vec![Message::user("Hi")]; // ~5 tokens
        let remaining = window.tokens_remaining(&messages);
        assert_eq!(remaining, 75);
    }

    #[test]
    fn truncate_to_fit_no_truncation_needed() {
        let window = test_window();
        let messages = vec![
            Message::system("Be helpful."),
            Message::user("Hello"),
        ];
        let result = window.truncate_to_fit(&messages);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn truncate_to_fit_drops_old_middle_messages() {
        let window = ContextWindow::new(
            CharEstimator::default(),
            ContextBudget {
                max_tokens: 50,
                reserved_for_output: 10,
                tool_result_share: 0.3,
            },
        );

        // Budget: 40 tokens available
        let messages = vec![
            Message::system("System"),                   // ~6 tokens
            Message::user("Old message one"),            // ~8 tokens
            Message::user("Old message two"),            // ~8 tokens
            Message::assistant("Old response"),          // ~7 tokens
            Message::user("Recent message"),             // ~8 tokens
            Message::user("What should I do?"),          // ~9 tokens
        ];

        let result = window.truncate_to_fit(&messages);

        // Should keep first (system), last (current), and as many recent middle as fit
        assert!(result.len() < messages.len());
        assert_eq!(result[0].content, "System");
        assert_eq!(result.last().unwrap().content, "What should I do?");

        // Verify it fits now
        assert!(!window.is_over_budget(&result));
    }

    #[test]
    fn truncate_to_fit_preserves_two_messages() {
        let window = ContextWindow::new(
            CharEstimator::default(),
            ContextBudget {
                max_tokens: 5, // very tight
                reserved_for_output: 0,
                tool_result_share: 0.3,
            },
        );

        let messages = vec![
            Message::system("S"),
            Message::user("U"),
        ];

        // Even if over budget, we keep both when there are only 2
        let result = window.truncate_to_fit(&messages);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn truncate_tool_result_no_truncation() {
        let window = test_window();
        let short = "Small result.";
        assert_eq!(window.truncate_tool_result(short), short);
    }

    #[test]
    fn truncate_tool_result_long_content() {
        let window = ContextWindow::new(
            CharEstimator::default(),
            ContextBudget {
                max_tokens: 30,
                reserved_for_output: 10,
                tool_result_share: 0.5, // 10 tokens max for tool results
            },
        );

        // 200 chars -> 50 tokens, well over the 10-token limit
        let long_content = "Line one\nLine two\nLine three\nLine four\n".repeat(5);
        let result = window.truncate_tool_result(&long_content);
        assert!(result.contains("[truncated"));
        assert!(result.len() < long_content.len());
    }
}
