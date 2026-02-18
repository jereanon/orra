use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub call_id: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_results: Vec<ToolResult>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            tool_calls: vec![],
            tool_results: vec![],
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            tool_calls: vec![],
            tool_results: vec![],
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls: vec![],
            tool_results: vec![],
        }
    }

    pub fn assistant_with_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            tool_calls,
            tool_results: vec![],
        }
    }

    pub fn tool_result(results: Vec<ToolResult>) -> Self {
        Self {
            role: Role::User,
            content: String::new(),
            tool_calls: vec![],
            tool_results: results,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_system_message() {
        let msg = Message::system("You are a helpful assistant.");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.content, "You are a helpful assistant.");
        assert!(msg.tool_calls.is_empty());
    }

    #[test]
    fn create_user_message() {
        let msg = Message::user("Hello!");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "Hello!");
    }

    #[test]
    fn create_assistant_message() {
        let msg = Message::assistant("Hi there!");
        assert_eq!(msg.role, Role::Assistant);
        assert!(msg.tool_calls.is_empty());
    }

    #[test]
    fn assistant_with_tool_calls() {
        let calls = vec![ToolCall {
            id: "call_1".into(),
            name: "search".into(),
            arguments: serde_json::json!({"query": "weather"}),
        }];
        let msg = Message::assistant_with_tool_calls("Let me search.", calls.clone());
        assert_eq!(msg.tool_calls.len(), 1);
        assert_eq!(msg.tool_calls[0].name, "search");
    }

    #[test]
    fn tool_result_message() {
        let results = vec![ToolResult {
            call_id: "call_1".into(),
            content: "Sunny, 72F".into(),
            is_error: false,
        }];
        let msg = Message::tool_result(results);
        assert_eq!(msg.tool_results.len(), 1);
        assert!(!msg.tool_results[0].is_error);
    }

    #[test]
    fn serialization_skips_empty_vecs() {
        let msg = Message::user("Hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("tool_calls"));
        assert!(!json.contains("tool_results"));
    }

    #[test]
    fn serialization_roundtrip() {
        let msg = Message::assistant_with_tool_calls(
            "Searching...",
            vec![ToolCall {
                id: "c1".into(),
                name: "search".into(),
                arguments: serde_json::json!({"q": "rust"}),
            }],
        );
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, deserialized);
    }
}
