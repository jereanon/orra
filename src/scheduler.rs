use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Cron expression parsing (simplified)
// ---------------------------------------------------------------------------

/// Represents a parsed cron schedule.
///
/// Supports standard 5-field cron syntax:
///   `minute hour day-of-month month day-of-week`
///
/// Special values:
///
/// - `*` = any value
/// - `*/N` = every N units
/// - `N` = specific value
/// - `N,M` = multiple values
/// - `N-M` = range
#[derive(Debug, Clone)]
pub struct CronSchedule {
    minutes: FieldSpec,
    hours: FieldSpec,
    days_of_month: FieldSpec,
    months: FieldSpec,
    days_of_week: FieldSpec,
    raw: String,
}

#[derive(Debug, Clone)]
enum FieldSpec {
    Any,
    Every(u32),
    Values(Vec<u32>),
}

impl FieldSpec {
    fn matches(&self, value: u32) -> bool {
        match self {
            FieldSpec::Any => true,
            FieldSpec::Every(step) => value % step == 0,
            FieldSpec::Values(vals) => vals.contains(&value),
        }
    }

    fn parse(field: &str) -> Result<Self, SchedulerError> {
        if field == "*" {
            return Ok(FieldSpec::Any);
        }

        if let Some(step) = field.strip_prefix("*/") {
            let n: u32 = step
                .parse()
                .map_err(|_| SchedulerError::InvalidCron(format!("bad step: {field}")))?;
            if n == 0 {
                return Err(SchedulerError::InvalidCron("step cannot be 0".into()));
            }
            return Ok(FieldSpec::Every(n));
        }

        // Could be a comma-separated list, a range, or a single value
        let mut values = Vec::new();

        for part in field.split(',') {
            if let Some((start, end)) = part.split_once('-') {
                let s: u32 = start
                    .parse()
                    .map_err(|_| SchedulerError::InvalidCron(format!("bad range start: {part}")))?;
                let e: u32 = end
                    .parse()
                    .map_err(|_| SchedulerError::InvalidCron(format!("bad range end: {part}")))?;
                if s > e {
                    return Err(SchedulerError::InvalidCron(format!(
                        "range start > end: {part}"
                    )));
                }
                for v in s..=e {
                    values.push(v);
                }
            } else {
                let v: u32 = part
                    .parse()
                    .map_err(|_| SchedulerError::InvalidCron(format!("bad value: {part}")))?;
                values.push(v);
            }
        }

        values.sort();
        values.dedup();
        Ok(FieldSpec::Values(values))
    }
}

impl CronSchedule {
    /// Parse a standard 5-field cron expression.
    pub fn parse(expr: &str) -> Result<Self, SchedulerError> {
        let fields: Vec<&str> = expr.split_whitespace().collect();
        if fields.len() != 5 {
            return Err(SchedulerError::InvalidCron(format!(
                "expected 5 fields, got {}",
                fields.len()
            )));
        }

        Ok(Self {
            minutes: FieldSpec::parse(fields[0])?,
            hours: FieldSpec::parse(fields[1])?,
            days_of_month: FieldSpec::parse(fields[2])?,
            months: FieldSpec::parse(fields[3])?,
            days_of_week: FieldSpec::parse(fields[4])?,
            raw: expr.to_string(),
        })
    }

    /// Check if a given datetime matches this cron schedule.
    pub fn matches(&self, dt: &DateTime<Utc>) -> bool {
        let minute = dt.format("%M").to_string().parse::<u32>().unwrap_or(0);
        let hour = dt.format("%H").to_string().parse::<u32>().unwrap_or(0);
        let day = dt.format("%d").to_string().parse::<u32>().unwrap_or(1);
        let month = dt.format("%m").to_string().parse::<u32>().unwrap_or(1);
        // chrono: Monday = 1, Sunday = 7. Cron: Sunday = 0, Saturday = 6
        let weekday = dt.format("%u").to_string().parse::<u32>().unwrap_or(1);
        let cron_weekday = if weekday == 7 { 0 } else { weekday };

        self.minutes.matches(minute)
            && self.hours.matches(hour)
            && self.days_of_month.matches(day)
            && self.months.matches(month)
            && self.days_of_week.matches(cron_weekday)
    }

    /// Returns the raw cron expression string.
    pub fn expression(&self) -> &str {
        &self.raw
    }
}

// ---------------------------------------------------------------------------
// Scheduled job
// ---------------------------------------------------------------------------

/// A callback that gets invoked when a scheduled job fires.
pub type JobCallback = Arc<dyn Fn() -> tokio::task::JoinHandle<()> + Send + Sync>;

/// A single scheduled job.
pub struct ScheduledJob {
    pub id: String,
    pub name: String,
    pub schedule: CronSchedule,
    pub callback: JobCallback,
    pub enabled: bool,
    pub last_run: Option<DateTime<Utc>>,
    pub run_count: u64,
}

impl std::fmt::Debug for ScheduledJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScheduledJob")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("schedule", &self.schedule.raw)
            .field("enabled", &self.enabled)
            .field("last_run", &self.last_run)
            .field("run_count", &self.run_count)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// The scheduler manages a set of cron-based jobs and runs them on schedule.
///
/// Call `start()` to begin the tick loop in the background. The scheduler
/// checks every 30 seconds whether any jobs need to fire.
pub struct Scheduler {
    jobs: Arc<RwLock<HashMap<String, ScheduledJob>>>,
    running: Arc<tokio::sync::watch::Sender<bool>>,
}

impl Scheduler {
    pub fn new() -> Self {
        let (tx, _) = tokio::sync::watch::channel(false);
        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            running: Arc::new(tx),
        }
    }

    /// Add a new scheduled job. Returns the job ID.
    pub async fn add_job(
        &self,
        name: impl Into<String>,
        cron_expr: &str,
        callback: JobCallback,
    ) -> Result<String, SchedulerError> {
        let schedule = CronSchedule::parse(cron_expr)?;
        let id = uuid::Uuid::new_v4().to_string();

        let job = ScheduledJob {
            id: id.clone(),
            name: name.into(),
            schedule,
            callback,
            enabled: true,
            last_run: None,
            run_count: 0,
        };

        self.jobs.write().await.insert(id.clone(), job);
        Ok(id)
    }

    /// Remove a job by ID.
    pub async fn remove_job(&self, id: &str) -> bool {
        self.jobs.write().await.remove(id).is_some()
    }

    /// Enable or disable a job.
    pub async fn set_enabled(&self, id: &str, enabled: bool) -> bool {
        if let Some(job) = self.jobs.write().await.get_mut(id) {
            job.enabled = enabled;
            true
        } else {
            false
        }
    }

    /// List all registered jobs.
    pub async fn list_jobs(&self) -> Vec<JobInfo> {
        self.jobs
            .read()
            .await
            .values()
            .map(|j| JobInfo {
                id: j.id.clone(),
                name: j.name.clone(),
                schedule: j.schedule.raw.clone(),
                enabled: j.enabled,
                last_run: j.last_run,
                run_count: j.run_count,
            })
            .collect()
    }

    /// Start the scheduler's background tick loop. Returns a handle that
    /// stops the scheduler when dropped.
    pub fn start(&self) -> SchedulerHandle {
        let _ = self.running.send(true);
        let jobs = self.jobs.clone();
        let mut rx = self.running.subscribe();

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let now = Utc::now();
                        let mut jobs_lock = jobs.write().await;

                        for job in jobs_lock.values_mut() {
                            if !job.enabled {
                                continue;
                            }

                            // Skip if we already ran this job in the same minute
                            if let Some(last) = job.last_run {
                                if last.format("%Y-%m-%d %H:%M").to_string()
                                    == now.format("%Y-%m-%d %H:%M").to_string()
                                {
                                    continue;
                                }
                            }

                            if job.schedule.matches(&now) {
                                job.last_run = Some(now);
                                job.run_count += 1;
                                drop((job.callback)());
                            }
                        }
                    }
                    _ = rx.changed() => {
                        if !*rx.borrow() {
                            break;
                        }
                    }
                }
            }
        });

        SchedulerHandle {
            stop_signal: self.running.clone(),
            task: Some(handle),
        }
    }

    /// Run a single tick manually (useful for testing). Checks all jobs
    /// against the provided timestamp.
    pub async fn tick(&self, now: &DateTime<Utc>) {
        let mut jobs_lock = self.jobs.write().await;

        for job in jobs_lock.values_mut() {
            if !job.enabled {
                continue;
            }

            if job.schedule.matches(now) {
                job.last_run = Some(*now);
                job.run_count += 1;
                drop((job.callback)());
            }
        }
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Handle and info types
// ---------------------------------------------------------------------------

/// Handle that keeps the scheduler running. Dropping it stops the tick loop.
pub struct SchedulerHandle {
    stop_signal: Arc<tokio::sync::watch::Sender<bool>>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl SchedulerHandle {
    /// Stop the scheduler.
    pub fn stop(&self) {
        let _ = self.stop_signal.send(false);
    }
}

impl Drop for SchedulerHandle {
    fn drop(&mut self) {
        let _ = self.stop_signal.send(false);
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

/// Serializable info about a job (returned by list operations).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct JobInfo {
    pub id: String,
    pub name: String,
    pub schedule: String,
    pub enabled: bool,
    pub last_run: Option<DateTime<Utc>>,
    pub run_count: u64,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("invalid cron expression: {0}")]
    InvalidCron(String),

    #[error("job not found: {0}")]
    NotFound(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn parse_every_minute() {
        let sched = CronSchedule::parse("* * * * *").unwrap();
        let dt = Utc.with_ymd_and_hms(2025, 6, 15, 10, 30, 0).unwrap();
        assert!(sched.matches(&dt));
    }

    #[test]
    fn parse_specific_minute() {
        let sched = CronSchedule::parse("30 * * * *").unwrap();
        let dt_match = Utc.with_ymd_and_hms(2025, 6, 15, 10, 30, 0).unwrap();
        let dt_no_match = Utc.with_ymd_and_hms(2025, 6, 15, 10, 15, 0).unwrap();
        assert!(sched.matches(&dt_match));
        assert!(!sched.matches(&dt_no_match));
    }

    #[test]
    fn parse_every_5_minutes() {
        let sched = CronSchedule::parse("*/5 * * * *").unwrap();
        let dt0 = Utc.with_ymd_and_hms(2025, 6, 15, 10, 0, 0).unwrap();
        let dt5 = Utc.with_ymd_and_hms(2025, 6, 15, 10, 5, 0).unwrap();
        let dt3 = Utc.with_ymd_and_hms(2025, 6, 15, 10, 3, 0).unwrap();
        assert!(sched.matches(&dt0));
        assert!(sched.matches(&dt5));
        assert!(!sched.matches(&dt3));
    }

    #[test]
    fn parse_range() {
        let sched = CronSchedule::parse("0 9-17 * * *").unwrap();
        let dt_9am = Utc.with_ymd_and_hms(2025, 6, 15, 9, 0, 0).unwrap();
        let dt_5pm = Utc.with_ymd_and_hms(2025, 6, 15, 17, 0, 0).unwrap();
        let dt_8am = Utc.with_ymd_and_hms(2025, 6, 15, 8, 0, 0).unwrap();
        assert!(sched.matches(&dt_9am));
        assert!(sched.matches(&dt_5pm));
        assert!(!sched.matches(&dt_8am));
    }

    #[test]
    fn parse_comma_list() {
        let sched = CronSchedule::parse("0 8,12,18 * * *").unwrap();
        let dt_8 = Utc.with_ymd_and_hms(2025, 6, 15, 8, 0, 0).unwrap();
        let dt_12 = Utc.with_ymd_and_hms(2025, 6, 15, 12, 0, 0).unwrap();
        let dt_10 = Utc.with_ymd_and_hms(2025, 6, 15, 10, 0, 0).unwrap();
        assert!(sched.matches(&dt_8));
        assert!(sched.matches(&dt_12));
        assert!(!sched.matches(&dt_10));
    }

    #[test]
    fn parse_weekday() {
        // Monday through Friday (1-5 in cron)
        let sched = CronSchedule::parse("0 9 * * 1-5").unwrap();
        // June 16, 2025 is a Monday
        let monday = Utc.with_ymd_and_hms(2025, 6, 16, 9, 0, 0).unwrap();
        // June 15, 2025 is a Sunday
        let sunday = Utc.with_ymd_and_hms(2025, 6, 15, 9, 0, 0).unwrap();
        assert!(sched.matches(&monday));
        assert!(!sched.matches(&sunday));
    }

    #[test]
    fn invalid_cron_too_few_fields() {
        assert!(CronSchedule::parse("* * *").is_err());
    }

    #[test]
    fn invalid_cron_bad_value() {
        assert!(CronSchedule::parse("abc * * * *").is_err());
    }

    #[test]
    fn invalid_cron_zero_step() {
        assert!(CronSchedule::parse("*/0 * * * *").is_err());
    }

    #[test]
    fn expression_roundtrip() {
        let expr = "0 */2 * * 1-5";
        let sched = CronSchedule::parse(expr).unwrap();
        assert_eq!(sched.expression(), expr);
    }

    #[tokio::test]
    async fn scheduler_add_and_list() {
        let scheduler = Scheduler::new();

        let cb: JobCallback = Arc::new(|| tokio::spawn(async {}));
        let id = scheduler
            .add_job("test job", "* * * * *", cb)
            .await
            .unwrap();

        let jobs = scheduler.list_jobs().await;
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].name, "test job");
        assert_eq!(jobs[0].id, id);
    }

    #[tokio::test]
    async fn scheduler_remove_job() {
        let scheduler = Scheduler::new();
        let cb: JobCallback = Arc::new(|| tokio::spawn(async {}));
        let id = scheduler.add_job("temp", "* * * * *", cb).await.unwrap();

        assert!(scheduler.remove_job(&id).await);
        assert!(!scheduler.remove_job(&id).await);
        assert!(scheduler.list_jobs().await.is_empty());
    }

    #[tokio::test]
    async fn scheduler_enable_disable() {
        let scheduler = Scheduler::new();
        let cb: JobCallback = Arc::new(|| tokio::spawn(async {}));
        let id = scheduler.add_job("toggle", "* * * * *", cb).await.unwrap();

        assert!(scheduler.set_enabled(&id, false).await);

        let jobs = scheduler.list_jobs().await;
        assert!(!jobs[0].enabled);
    }

    #[tokio::test]
    async fn scheduler_tick_fires_matching_job() {
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let scheduler = Scheduler::new();
        let cb: JobCallback = Arc::new(move || {
            let c = counter_clone.clone();
            tokio::spawn(async move {
                c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            })
        });

        scheduler
            .add_job("counter", "30 10 * * *", cb)
            .await
            .unwrap();

        // This time matches: 10:30
        let matching = Utc.with_ymd_and_hms(2025, 6, 15, 10, 30, 0).unwrap();
        scheduler.tick(&matching).await;

        // Give the spawned task a moment
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 1);

        // This time doesn't match
        let not_matching = Utc.with_ymd_and_hms(2025, 6, 15, 11, 0, 0).unwrap();
        scheduler.tick(&not_matching).await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn scheduler_disabled_job_doesnt_fire() {
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let scheduler = Scheduler::new();
        let cb: JobCallback = Arc::new(move || {
            let c = counter_clone.clone();
            tokio::spawn(async move {
                c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            })
        });

        let id = scheduler
            .add_job("disabled", "* * * * *", cb)
            .await
            .unwrap();
        scheduler.set_enabled(&id, false).await;

        let now = Utc::now();
        scheduler.tick(&now).await;

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn scheduler_start_and_stop() {
        let scheduler = Scheduler::new();
        let handle = scheduler.start();

        // Just make sure it doesn't panic
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        handle.stop();
    }

    #[test]
    fn job_info_serialization() {
        let info = JobInfo {
            id: "test-id".into(),
            name: "test job".into(),
            schedule: "0 * * * *".into(),
            enabled: true,
            last_run: None,
            run_count: 0,
        };

        let json = serde_json::to_string(&info).unwrap();
        let deser: JobInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.id, "test-id");
        assert_eq!(deser.name, "test job");
    }
}
