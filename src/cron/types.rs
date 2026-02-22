use chrono::{DateTime, Timelike, Utc};
use serde::{Deserialize, Serialize};

/// How the cron job should be scheduled.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CronScheduleType {
    /// Fire once at a specific ISO-8601 datetime.
    At { datetime: DateTime<Utc> },
    /// Fire on a repeating interval (milliseconds).
    Every { interval_ms: u64 },
    /// Standard 5-field cron expression (minute hour dom month dow).
    Cron { expression: String },
}

/// What happens when the job fires.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CronPayload {
    /// Inject a message into an existing session as a system event.
    SystemEvent { message: String },
    /// Run a fresh, isolated agent turn with this prompt.
    AgentTurn { prompt: String },
}

/// How the result should be delivered.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CronDelivery {
    /// Post the result to the originating channel/session.
    #[default]
    Announce,
    /// Do nothing with the result (fire-and-forget).
    Silent,
}

/// Runtime status of a cron job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CronJobStatus {
    Active,
    Paused,
    Completed,
}

/// A single cron job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJob {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name (set by the AI or user).
    pub name: String,
    /// Schedule configuration.
    pub schedule: CronScheduleType,
    /// What to do when the job fires.
    pub payload: CronPayload,
    /// How to deliver the result.
    #[serde(default)]
    pub delivery: CronDelivery,
    /// The namespace (e.g. channel or user) this job belongs to.
    pub namespace: String,
    /// Optional model override. If set, the cron job uses this model
    /// instead of the default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Current status.
    pub status: CronJobStatus,
    /// When the job was created.
    pub created_at: DateTime<Utc>,
    /// When the job last fired.
    pub last_run: Option<DateTime<Utc>>,
    /// When the job should next fire (computed).
    pub next_run: Option<DateTime<Utc>>,
    /// How many times the job has fired.
    pub run_count: u64,
}

impl CronJob {
    pub fn new(
        name: impl Into<String>,
        schedule: CronScheduleType,
        payload: CronPayload,
        namespace: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        let id = uuid::Uuid::new_v4().to_string();
        let mut job = Self {
            id,
            name: name.into(),
            schedule,
            payload,
            delivery: CronDelivery::default(),
            namespace: namespace.into(),
            model: None,
            status: CronJobStatus::Active,
            created_at: now,
            last_run: None,
            next_run: None,
            run_count: 0,
        };
        job.next_run = job.compute_next_run(now);
        job
    }

    /// Compute the next fire time from a given reference point.
    pub fn compute_next_run(&self, from: DateTime<Utc>) -> Option<DateTime<Utc>> {
        match &self.schedule {
            CronScheduleType::At { datetime } => {
                if *datetime > from {
                    Some(*datetime)
                } else {
                    None // already passed
                }
            }
            CronScheduleType::Every { interval_ms } => {
                if *interval_ms == 0 {
                    return None;
                }
                let interval = chrono::Duration::milliseconds(*interval_ms as i64);
                match self.last_run {
                    Some(last) => {
                        let mut next = last + interval;
                        // If we're behind, fast-forward
                        while next <= from {
                            next += interval;
                        }
                        Some(next)
                    }
                    None => Some(from + interval),
                }
            }
            CronScheduleType::Cron { expression } => compute_next_cron_time(expression, from),
        }
    }

    /// Returns true if this job should fire at the given time.
    pub fn should_fire(&self, now: DateTime<Utc>) -> bool {
        if self.status != CronJobStatus::Active {
            return false;
        }
        match self.next_run {
            Some(next) => now >= next,
            None => false,
        }
    }

    /// Mark the job as having just fired and recompute next_run.
    pub fn mark_fired(&mut self, now: DateTime<Utc>) {
        self.last_run = Some(now);
        self.run_count += 1;
        self.next_run = self.compute_next_run(now);

        // One-shot "at" jobs auto-complete
        if matches!(self.schedule, CronScheduleType::At { .. }) {
            self.status = CronJobStatus::Completed;
            self.next_run = None;
        }
    }
}

/// Compute the next time a cron expression matches after `from`.
///
/// Brute-force: scan minute-by-minute up to 366 days out.
fn compute_next_cron_time(expression: &str, from: DateTime<Utc>) -> Option<DateTime<Utc>> {
    use crate::scheduler::CronSchedule;

    let schedule = CronSchedule::parse(expression).ok()?;

    // Start from the next minute boundary
    let start = from.with_nanosecond(0)?;
    let start = start + chrono::Duration::minutes(1);
    let start = start.with_nanosecond(0)?;

    let max_checks = 366 * 24 * 60; // one year of minutes
    let mut candidate = start;

    for _ in 0..max_checks {
        if schedule.matches(&candidate) {
            return Some(candidate);
        }
        candidate += chrono::Duration::minutes(1);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn at_schedule_future() {
        let future = Utc::now() + chrono::Duration::hours(1);
        let job = CronJob::new(
            "test",
            CronScheduleType::At { datetime: future },
            CronPayload::AgentTurn {
                prompt: "hello".into(),
            },
            "test-ns",
        );
        assert_eq!(job.status, CronJobStatus::Active);
        assert!(job.next_run.is_some());
        assert!(job.should_fire(future));
    }

    #[test]
    fn at_schedule_past() {
        let past = Utc::now() - chrono::Duration::hours(1);
        let job = CronJob::new(
            "test",
            CronScheduleType::At { datetime: past },
            CronPayload::AgentTurn {
                prompt: "hello".into(),
            },
            "test-ns",
        );
        assert!(job.next_run.is_none());
    }

    #[test]
    fn every_schedule() {
        let job = CronJob::new(
            "test",
            CronScheduleType::Every { interval_ms: 60000 },
            CronPayload::SystemEvent {
                message: "tick".into(),
            },
            "test-ns",
        );
        assert!(job.next_run.is_some());
    }

    #[test]
    fn cron_schedule_next_run() {
        // Every hour at minute 0
        let now = Utc.with_ymd_and_hms(2025, 6, 15, 10, 30, 0).unwrap();
        let next = compute_next_cron_time("0 * * * *", now);
        assert!(next.is_some());
        let next = next.unwrap();
        assert_eq!(next.format("%H:%M").to_string(), "11:00");
    }

    #[test]
    fn mark_fired_at_completes() {
        let future = Utc::now() + chrono::Duration::hours(1);
        let mut job = CronJob::new(
            "once",
            CronScheduleType::At { datetime: future },
            CronPayload::AgentTurn {
                prompt: "go".into(),
            },
            "ns",
        );
        job.mark_fired(future);
        assert_eq!(job.status, CronJobStatus::Completed);
        assert!(job.next_run.is_none());
        assert_eq!(job.run_count, 1);
    }

    #[test]
    fn mark_fired_every_reschedules() {
        let mut job = CronJob::new(
            "repeat",
            CronScheduleType::Every { interval_ms: 60000 },
            CronPayload::SystemEvent {
                message: "tick".into(),
            },
            "ns",
        );
        let now = Utc::now();
        job.mark_fired(now);
        assert_eq!(job.status, CronJobStatus::Active);
        assert!(job.next_run.is_some());
        assert!(job.next_run.unwrap() > now);
        assert_eq!(job.run_count, 1);
    }

    #[test]
    fn paused_job_does_not_fire() {
        let mut job = CronJob::new(
            "paused",
            CronScheduleType::Every { interval_ms: 1 },
            CronPayload::SystemEvent {
                message: "nope".into(),
            },
            "ns",
        );
        job.status = CronJobStatus::Paused;
        assert!(!job.should_fire(Utc::now() + chrono::Duration::hours(1)));
    }

    #[test]
    fn serialization_roundtrip() {
        let job = CronJob::new(
            "test",
            CronScheduleType::Cron {
                expression: "*/5 * * * *".into(),
            },
            CronPayload::AgentTurn {
                prompt: "check weather".into(),
            },
            "discord:123",
        );
        let json = serde_json::to_string_pretty(&job).unwrap();
        let deser: CronJob = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.id, job.id);
        assert_eq!(deser.name, "test");
    }
}
