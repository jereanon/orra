use std::sync::Arc;

use chrono::Utc;
use tokio::sync::RwLock;

use super::store::{CronStore, CronStoreError};
use super::types::{CronJob, CronJobStatus};

/// Callback invoked when a cron job fires.
///
/// Receives the job and returns a future that resolves when execution completes.
pub type CronCallback =
    Arc<dyn Fn(CronJob) -> tokio::task::JoinHandle<()> + Send + Sync>;

/// The cron service manages scheduled jobs, persists them, and fires them.
pub struct CronService {
    store: Arc<dyn CronStore>,
    callback: Arc<RwLock<Option<CronCallback>>>,
    running: Arc<tokio::sync::watch::Sender<bool>>,
}

impl CronService {
    pub fn new(store: Arc<dyn CronStore>) -> Self {
        let (tx, _) = tokio::sync::watch::channel(false);
        Self {
            store,
            callback: Arc::new(RwLock::new(None)),
            running: Arc::new(tx),
        }
    }

    /// Set the callback that gets invoked when a job fires.
    pub async fn set_callback(&self, cb: CronCallback) {
        *self.callback.write().await = Some(cb);
    }

    /// Add a new cron job. Returns the created job.
    pub async fn add_job(&self, job: CronJob) -> Result<CronJob, CronStoreError> {
        self.store.save(&job).await?;
        eprintln!("[cron] Added job '{}' (id: {})", job.name, job.id);
        Ok(job)
    }

    /// Get a job by ID.
    pub async fn get_job(&self, id: &str) -> Result<Option<CronJob>, CronStoreError> {
        self.store.load(id).await
    }

    /// List all jobs.
    pub async fn list_jobs(&self) -> Result<Vec<CronJob>, CronStoreError> {
        self.store.list().await
    }

    /// Delete a job by ID.
    pub async fn delete_job(&self, id: &str) -> Result<bool, CronStoreError> {
        let deleted = self.store.delete(id).await?;
        if deleted {
            eprintln!("[cron] Deleted job {}", id);
        }
        Ok(deleted)
    }

    /// Pause a job.
    pub async fn pause_job(&self, id: &str) -> Result<bool, CronStoreError> {
        if let Some(mut job) = self.store.load(id).await? {
            job.status = CronJobStatus::Paused;
            self.store.save(&job).await?;
            eprintln!("[cron] Paused job '{}' (id: {})", job.name, id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Resume a paused job.
    pub async fn resume_job(&self, id: &str) -> Result<bool, CronStoreError> {
        if let Some(mut job) = self.store.load(id).await? {
            job.status = CronJobStatus::Active;
            let now = Utc::now();
            job.next_run = job.compute_next_run(now);
            self.store.save(&job).await?;
            eprintln!("[cron] Resumed job '{}' (id: {})", job.name, id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Start the background tick loop. Returns a handle to stop it.
    pub fn start(&self) -> CronServiceHandle {
        let _ = self.running.send(true);
        let store = self.store.clone();
        let callback = self.callback.clone();
        let mut rx = self.running.subscribe();

        let handle = tokio::spawn(async move {
            // Tick every 30 seconds
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let now = Utc::now();
                        let jobs = match store.list().await {
                            Ok(j) => j,
                            Err(e) => {
                                eprintln!("[cron] Error listing jobs: {}", e);
                                continue;
                            }
                        };

                        let cb = callback.read().await;

                        for mut job in jobs {
                            if !job.should_fire(now) {
                                continue;
                            }

                            eprintln!("[cron] Firing job '{}' (id: {})", job.name, job.id);
                            job.mark_fired(now);

                            // Persist updated state
                            if let Err(e) = store.save(&job).await {
                                eprintln!("[cron] Error saving job state: {}", e);
                            }

                            // Invoke the callback
                            if let Some(ref cb) = *cb {
                                let _ = cb(job);
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

        CronServiceHandle {
            stop_signal: self.running.clone(),
            task: Some(handle),
        }
    }

    /// Run a single tick manually (for testing).
    pub async fn tick(&self) {
        let now = Utc::now();
        let jobs = match self.store.list().await {
            Ok(j) => j,
            Err(_) => return,
        };

        let cb = self.callback.read().await;

        for mut job in jobs {
            if !job.should_fire(now) {
                continue;
            }

            job.mark_fired(now);
            if let Err(e) = self.store.save(&job).await {
                eprintln!("[cron] Error saving job state: {}", e);
            }

            if let Some(ref cb) = *cb {
                let _ = cb(job);
            }
        }
    }
}

/// Handle that keeps the cron service tick loop running.
pub struct CronServiceHandle {
    stop_signal: Arc<tokio::sync::watch::Sender<bool>>,
    task: Option<tokio::task::JoinHandle<()>>,
}

impl CronServiceHandle {
    pub fn stop(&self) {
        let _ = self.stop_signal.send(false);
    }
}

impl Drop for CronServiceHandle {
    fn drop(&mut self) {
        let _ = self.stop_signal.send(false);
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cron::store::InMemoryCronStore;
    use crate::cron::types::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn make_service() -> CronService {
        let store = Arc::new(InMemoryCronStore::new());
        CronService::new(store)
    }

    #[tokio::test]
    async fn add_and_list_jobs() {
        let svc = make_service();

        let job = CronJob::new(
            "test",
            CronScheduleType::Every { interval_ms: 60000 },
            CronPayload::SystemEvent { message: "hi".into() },
            "ns",
        );
        svc.add_job(job).await.unwrap();

        let jobs = svc.list_jobs().await.unwrap();
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].name, "test");
    }

    #[tokio::test]
    async fn delete_job() {
        let svc = make_service();

        let job = CronJob::new(
            "temp",
            CronScheduleType::Every { interval_ms: 60000 },
            CronPayload::SystemEvent { message: "bye".into() },
            "ns",
        );
        let id = job.id.clone();
        svc.add_job(job).await.unwrap();

        assert!(svc.delete_job(&id).await.unwrap());
        assert!(!svc.delete_job(&id).await.unwrap());
        assert!(svc.list_jobs().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn pause_and_resume() {
        let svc = make_service();

        let job = CronJob::new(
            "toggle",
            CronScheduleType::Every { interval_ms: 60000 },
            CronPayload::SystemEvent { message: "x".into() },
            "ns",
        );
        let id = job.id.clone();
        svc.add_job(job).await.unwrap();

        assert!(svc.pause_job(&id).await.unwrap());
        let j = svc.get_job(&id).await.unwrap().unwrap();
        assert_eq!(j.status, CronJobStatus::Paused);

        assert!(svc.resume_job(&id).await.unwrap());
        let j = svc.get_job(&id).await.unwrap().unwrap();
        assert_eq!(j.status, CronJobStatus::Active);
    }

    #[tokio::test]
    async fn start_and_stop() {
        let svc = make_service();
        let handle = svc.start();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        handle.stop();
    }
}
