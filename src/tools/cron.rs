use std::sync::Arc;

use async_trait::async_trait;

use crate::cron::service::CronService;
use crate::cron::types::*;
use crate::tool::{Tool, ToolDefinition, ToolError, ToolRegistry};

/// Tool that lets the AI create, list, pause, resume, and delete cron jobs.
pub struct CronTool {
    service: Arc<CronService>,
}

impl CronTool {
    pub fn new(service: Arc<CronService>) -> Self {
        Self { service }
    }
}

#[async_trait]
impl Tool for CronTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "cron".into(),
            description: "Manage scheduled/recurring tasks (cron jobs). You can create timed reminders, \
                          recurring checks, scheduled reports, and more. The job will run automatically \
                          at the specified time(s) and deliver results back to the conversation.\n\n\
                          Actions: create, list, get, pause, resume, delete\n\n\
                          Schedule types:\n\
                          - \"at\": One-shot, fires at a specific ISO-8601 datetime (e.g. \"2025-06-15T09:00:00Z\")\n\
                          - \"every\": Repeating interval in milliseconds (e.g. 3600000 for every hour)\n\
                          - \"cron\": Standard 5-field cron expression (minute hour day-of-month month day-of-week)\n\n\
                          Payload types:\n\
                          - \"agent_turn\": Run a fresh AI turn with the given prompt (best for reminders, reports)\n\
                          - \"system_event\": Inject a message into the session (best for notifications)"
                .into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "get", "pause", "resume", "delete"],
                        "description": "The action to perform"
                    },
                    "id": {
                        "type": "string",
                        "description": "Job ID (required for get, pause, resume, delete)"
                    },
                    "name": {
                        "type": "string",
                        "description": "Human-readable name for the job (required for create)"
                    },
                    "schedule": {
                        "type": "object",
                        "description": "Schedule configuration (required for create)",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["at", "every", "cron"],
                                "description": "Schedule type"
                            },
                            "datetime": {
                                "type": "string",
                                "description": "ISO-8601 datetime for 'at' type"
                            },
                            "interval_ms": {
                                "type": "integer",
                                "description": "Interval in milliseconds for 'every' type"
                            },
                            "expression": {
                                "type": "string",
                                "description": "Cron expression for 'cron' type (e.g. '0 9 * * 1-5')"
                            }
                        }
                    },
                    "payload": {
                        "type": "object",
                        "description": "What to do when the job fires (required for create)",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["agent_turn", "system_event"],
                                "description": "Payload type"
                            },
                            "prompt": {
                                "type": "string",
                                "description": "Prompt for 'agent_turn' type"
                            },
                            "message": {
                                "type": "string",
                                "description": "Message for 'system_event' type"
                            }
                        }
                    },
                    "namespace": {
                        "type": "string",
                        "description": "The namespace/channel this job belongs to (required for create)"
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "Maximum number of agent turns for this job (default: uses runtime setting, typically 10)"
                    }
                },
                "required": ["action"]
            }),
        }
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String, ToolError> {
        let action = input
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'action'".into()))?;

        match action {
            "create" => self.create_job(&input).await,
            "list" => self.list_jobs().await,
            "get" => self.get_job(&input).await,
            "pause" => self.pause_job(&input).await,
            "resume" => self.resume_job(&input).await,
            "delete" => self.delete_job(&input).await,
            _ => Err(ToolError::InvalidInput(format!(
                "unknown action: '{action}'. Use: create, list, get, pause, resume, delete"
            ))),
        }
    }
}

impl CronTool {
    async fn create_job(&self, input: &serde_json::Value) -> Result<String, ToolError> {
        let name = input
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'name' for create".into()))?;

        let namespace = input
            .get("namespace")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'namespace' for create".into()))?;

        let schedule = parse_schedule(input)?;
        let payload = parse_payload(input)?;

        let mut job = CronJob::new(name, schedule, payload, namespace);
        job.max_turns = input
            .get("max_turns")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);
        let job = self
            .service
            .add_job(job)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        let next_run = job
            .next_run
            .map(|t| t.to_rfc3339())
            .unwrap_or_else(|| "none".into());

        Ok(format!(
            "Created cron job '{}' (id: {}). Next run: {}",
            job.name, job.id, next_run
        ))
    }

    async fn list_jobs(&self) -> Result<String, ToolError> {
        let jobs = self
            .service
            .list_jobs()
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if jobs.is_empty() {
            return Ok("No cron jobs configured.".into());
        }

        let mut lines = Vec::new();
        for job in &jobs {
            let next = job
                .next_run
                .map(|t| t.to_rfc3339())
                .unwrap_or_else(|| "none".into());
            let schedule_desc = match &job.schedule {
                CronScheduleType::At { datetime } => format!("at {}", datetime.to_rfc3339()),
                CronScheduleType::Every { interval_ms } => format!("every {interval_ms}ms"),
                CronScheduleType::Cron { expression } => format!("cron: {expression}"),
            };
            lines.push(format!(
                "- {} (id: {}) [{:?}] schedule: {} | next: {} | runs: {}",
                job.name, job.id, job.status, schedule_desc, next, job.run_count
            ));
        }

        Ok(lines.join("\n"))
    }

    async fn get_job(&self, input: &serde_json::Value) -> Result<String, ToolError> {
        let id = input
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'id' for get".into()))?;

        let job = self
            .service
            .get_job(id)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        match job {
            Some(j) => {
                let json = serde_json::to_string_pretty(&j)
                    .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
                Ok(json)
            }
            None => Ok(format!("Job not found: {id}")),
        }
    }

    async fn pause_job(&self, input: &serde_json::Value) -> Result<String, ToolError> {
        let id = input
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'id' for pause".into()))?;

        let paused = self
            .service
            .pause_job(id)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if paused {
            Ok(format!("Paused job {id}."))
        } else {
            Ok(format!("Job not found: {id}"))
        }
    }

    async fn resume_job(&self, input: &serde_json::Value) -> Result<String, ToolError> {
        let id = input
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'id' for resume".into()))?;

        let resumed = self
            .service
            .resume_job(id)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if resumed {
            Ok(format!("Resumed job {id}."))
        } else {
            Ok(format!("Job not found: {id}"))
        }
    }

    async fn delete_job(&self, input: &serde_json::Value) -> Result<String, ToolError> {
        let id = input
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidInput("missing 'id' for delete".into()))?;

        let deleted = self
            .service
            .delete_job(id)
            .await
            .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;

        if deleted {
            Ok(format!("Deleted job {id}."))
        } else {
            Ok(format!("Job not found: {id}"))
        }
    }
}

// ---------------------------------------------------------------------------
// Input parsing helpers
// ---------------------------------------------------------------------------

fn parse_schedule(input: &serde_json::Value) -> Result<CronScheduleType, ToolError> {
    let sched = input
        .get("schedule")
        .ok_or_else(|| ToolError::InvalidInput("missing 'schedule' for create".into()))?;

    let stype = sched
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::InvalidInput("missing 'schedule.type'".into()))?;

    match stype {
        "at" => {
            let dt_str = sched
                .get("datetime")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ToolError::InvalidInput("missing 'schedule.datetime' for 'at' type".into())
                })?;
            let dt: chrono::DateTime<chrono::Utc> = dt_str.parse().map_err(|e| {
                ToolError::InvalidInput(format!("invalid datetime '{dt_str}': {e}"))
            })?;
            Ok(CronScheduleType::At { datetime: dt })
        }
        "every" => {
            let ms = sched
                .get("interval_ms")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    ToolError::InvalidInput(
                        "missing 'schedule.interval_ms' for 'every' type".into(),
                    )
                })?;
            if ms == 0 {
                return Err(ToolError::InvalidInput(
                    "interval_ms must be greater than 0".into(),
                ));
            }
            Ok(CronScheduleType::Every { interval_ms: ms })
        }
        "cron" => {
            let expr = sched
                .get("expression")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ToolError::InvalidInput("missing 'schedule.expression' for 'cron' type".into())
                })?;
            // Validate the expression
            crate::scheduler::CronSchedule::parse(expr).map_err(|e| {
                ToolError::InvalidInput(format!("invalid cron expression '{expr}': {e}"))
            })?;
            Ok(CronScheduleType::Cron {
                expression: expr.into(),
            })
        }
        _ => Err(ToolError::InvalidInput(format!(
            "unknown schedule type '{stype}'. Use: at, every, cron"
        ))),
    }
}

fn parse_payload(input: &serde_json::Value) -> Result<CronPayload, ToolError> {
    let payload = input
        .get("payload")
        .ok_or_else(|| ToolError::InvalidInput("missing 'payload' for create".into()))?;

    let ptype = payload
        .get("type")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolError::InvalidInput("missing 'payload.type'".into()))?;

    match ptype {
        "agent_turn" => {
            let prompt = payload
                .get("prompt")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ToolError::InvalidInput("missing 'payload.prompt' for 'agent_turn' type".into())
                })?;
            Ok(CronPayload::AgentTurn {
                prompt: prompt.into(),
            })
        }
        "system_event" => {
            let message = payload
                .get("message")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ToolError::InvalidInput(
                        "missing 'payload.message' for 'system_event' type".into(),
                    )
                })?;
            Ok(CronPayload::SystemEvent {
                message: message.into(),
            })
        }
        _ => Err(ToolError::InvalidInput(format!(
            "unknown payload type '{ptype}'. Use: agent_turn, system_event"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register the cron tool into a tool registry.
pub fn register_tool(registry: &mut ToolRegistry, service: &Arc<CronService>) {
    registry.register(Box::new(CronTool::new(service.clone())));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cron::store::InMemoryCronStore;

    fn make_service() -> Arc<CronService> {
        let store = Arc::new(InMemoryCronStore::new());
        Arc::new(CronService::new(store))
    }

    #[test]
    fn tool_definition_valid() {
        let svc = make_service();
        let tool = CronTool::new(svc);
        let def = tool.definition();
        assert_eq!(def.name, "cron");
        assert!(def.input_schema["properties"]["action"].is_object());
    }

    #[tokio::test]
    async fn create_and_list() {
        let svc = make_service();
        let tool = CronTool::new(svc);

        let result = tool
            .execute(serde_json::json!({
                "action": "create",
                "name": "morning reminder",
                "namespace": "test",
                "schedule": {
                    "type": "cron",
                    "expression": "0 9 * * *"
                },
                "payload": {
                    "type": "agent_turn",
                    "prompt": "Good morning! Here's your daily briefing."
                }
            }))
            .await
            .unwrap();

        assert!(result.contains("Created cron job"));
        assert!(result.contains("morning reminder"));

        let list = tool
            .execute(serde_json::json!({ "action": "list" }))
            .await
            .unwrap();
        assert!(list.contains("morning reminder"));
    }

    #[tokio::test]
    async fn create_at_schedule() {
        let svc = make_service();
        let tool = CronTool::new(svc);

        let future = (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339();
        let result = tool
            .execute(serde_json::json!({
                "action": "create",
                "name": "one-shot",
                "namespace": "test",
                "schedule": {
                    "type": "at",
                    "datetime": future
                },
                "payload": {
                    "type": "system_event",
                    "message": "Time's up!"
                }
            }))
            .await
            .unwrap();

        assert!(result.contains("Created"));
    }

    #[tokio::test]
    async fn pause_resume_delete() {
        let svc = make_service();
        let tool = CronTool::new(svc);

        let result = tool
            .execute(serde_json::json!({
                "action": "create",
                "name": "temp",
                "namespace": "test",
                "schedule": { "type": "every", "interval_ms": 60000 },
                "payload": { "type": "system_event", "message": "tick" }
            }))
            .await
            .unwrap();

        // Extract ID
        let id = result
            .split("id: ")
            .nth(1)
            .unwrap()
            .split(')')
            .next()
            .unwrap();

        // Pause
        let r = tool
            .execute(serde_json::json!({ "action": "pause", "id": id }))
            .await
            .unwrap();
        assert!(r.contains("Paused"));

        // Resume
        let r = tool
            .execute(serde_json::json!({ "action": "resume", "id": id }))
            .await
            .unwrap();
        assert!(r.contains("Resumed"));

        // Delete
        let r = tool
            .execute(serde_json::json!({ "action": "delete", "id": id }))
            .await
            .unwrap();
        assert!(r.contains("Deleted"));
    }

    #[tokio::test]
    async fn invalid_action() {
        let svc = make_service();
        let tool = CronTool::new(svc);

        let err = tool
            .execute(serde_json::json!({ "action": "explode" }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[tokio::test]
    async fn invalid_cron_expression() {
        let svc = make_service();
        let tool = CronTool::new(svc);

        let err = tool
            .execute(serde_json::json!({
                "action": "create",
                "name": "bad",
                "namespace": "test",
                "schedule": { "type": "cron", "expression": "not a cron" },
                "payload": { "type": "agent_turn", "prompt": "x" }
            }))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidInput(_)));
    }

    #[test]
    fn register_tool_adds_one() {
        let svc = make_service();
        let mut registry = ToolRegistry::new();
        register_tool(&mut registry, &svc);
        assert_eq!(registry.len(), 1);
        assert!(registry.get("cron").is_some());
    }
}
