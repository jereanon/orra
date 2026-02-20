//! Metrics and observability.
//!
//! Provides lightweight metrics collection for monitoring agent performance.
//! Tracks things like request counts, latencies, token usage, and error rates.
//! Designed to work with any external metrics system through the `MetricsSink`
//! trait.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::RwLock;

// ---------------------------------------------------------------------------
// Metric types
// ---------------------------------------------------------------------------

/// A single metric data point.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricPoint {
    pub name: String,
    pub value: f64,
    pub tags: HashMap<String, String>,
    pub timestamp_ms: u64,
}

/// Types of metrics we can record.
#[derive(Debug, Clone)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
}

// ---------------------------------------------------------------------------
// Metrics sink trait
// ---------------------------------------------------------------------------

/// Trait for exporting metrics to external systems (Prometheus, StatsD,
/// CloudWatch, etc.).
#[async_trait]
pub trait MetricsSink: Send + Sync {
    /// Record a set of metric points.
    async fn record(&self, points: &[MetricPoint]) -> Result<(), MetricsError>;

    /// Flush any buffered metrics.
    async fn flush(&self) -> Result<(), MetricsError>;
}

// ---------------------------------------------------------------------------
// In-memory metrics collector
// ---------------------------------------------------------------------------

/// Collects metrics in memory. Useful for development, testing, and as
/// a base that can be periodically flushed to an external sink.
pub struct MetricsCollector {
    counters: Arc<RwLock<HashMap<String, AtomicU64>>>,
    gauges: Arc<RwLock<HashMap<String, f64>>>,
    histograms: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    sinks: Vec<Arc<dyn MetricsSink>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
            sinks: Vec::new(),
        }
    }

    /// Add a sink for exporting metrics.
    pub fn add_sink(&mut self, sink: Arc<dyn MetricsSink>) {
        self.sinks.push(sink);
    }

    /// Increment a counter by 1.
    pub async fn increment(&self, name: &str) {
        self.increment_by(name, 1).await;
    }

    /// Increment a counter by a specific amount.
    pub async fn increment_by(&self, name: &str, amount: u64) {
        let counters = self.counters.read().await;
        if let Some(counter) = counters.get(name) {
            counter.fetch_add(amount, Ordering::Relaxed);
            return;
        }
        drop(counters);

        let mut counters = self.counters.write().await;
        counters
            .entry(name.to_string())
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(amount, Ordering::Relaxed);
    }

    /// Set a gauge to a specific value.
    pub async fn gauge(&self, name: &str, value: f64) {
        self.gauges.write().await.insert(name.to_string(), value);
    }

    /// Record a value in a histogram (for latency distributions, etc.).
    pub async fn histogram(&self, name: &str, value: f64) {
        self.histograms
            .write()
            .await
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    /// Get the current value of a counter.
    pub async fn get_counter(&self, name: &str) -> u64 {
        self.counters
            .read()
            .await
            .get(name)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    /// Get the current value of a gauge.
    pub async fn get_gauge(&self, name: &str) -> Option<f64> {
        self.gauges.read().await.get(name).copied()
    }

    /// Get histogram values.
    pub async fn get_histogram(&self, name: &str) -> Vec<f64> {
        self.histograms
            .read()
            .await
            .get(name)
            .cloned()
            .unwrap_or_default()
    }

    /// Get a snapshot of all metrics.
    pub async fn snapshot(&self) -> MetricsSnapshot {
        let counters: HashMap<String, u64> = self
            .counters
            .read()
            .await
            .iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
            .collect();

        let gauges = self.gauges.read().await.clone();
        let histograms = self.histograms.read().await.clone();

        MetricsSnapshot {
            counters,
            gauges,
            histograms,
        }
    }

    /// Flush metrics to all registered sinks.
    pub async fn flush(&self) -> Result<(), MetricsError> {
        for sink in &self.sinks {
            sink.flush().await?;
        }
        Ok(())
    }

    /// Reset all metrics to zero/empty.
    pub async fn reset(&self) {
        self.counters.write().await.clear();
        self.gauges.write().await.clear();
        self.histograms.write().await.clear();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of all collected metrics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
    pub histograms: HashMap<String, Vec<f64>>,
}

impl MetricsSnapshot {
    /// Calculate summary statistics for a histogram.
    pub fn histogram_stats(&self, name: &str) -> Option<HistogramStats> {
        let values = self.histograms.get(name)?;
        if values.is_empty() {
            return None;
        }

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = sorted.len();
        let sum: f64 = sorted.iter().sum();
        let mean = sum / count as f64;
        let min = sorted[0];
        let max = sorted[count - 1];
        let p50 = sorted[count / 2];
        let p95 = sorted[(count as f64 * 0.95) as usize];
        let p99 = sorted[((count as f64 * 0.99) as usize).min(count - 1)];

        Some(HistogramStats {
            count,
            sum,
            mean,
            min,
            max,
            p50,
            p95,
            p99,
        })
    }
}

/// Summary statistics for a histogram metric.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HistogramStats {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

// ---------------------------------------------------------------------------
// Timer utility
// ---------------------------------------------------------------------------

/// Utility for timing operations. Records the elapsed time to a histogram
/// metric when dropped.
pub struct Timer {
    name: String,
    start: Instant,
    collector: Arc<MetricsCollector>,
}

impl Timer {
    pub fn new(name: impl Into<String>, collector: Arc<MetricsCollector>) -> Self {
        Self {
            name: name.into(),
            start: Instant::now(),
            collector,
        }
    }

    /// Get the elapsed time so far without stopping the timer.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and record the elapsed time as milliseconds.
    pub async fn stop(self) {
        let elapsed_ms = self.start.elapsed().as_secs_f64() * 1000.0;
        self.collector.histogram(&self.name, elapsed_ms).await;
    }
}

// ---------------------------------------------------------------------------
// Predefined metric names
// ---------------------------------------------------------------------------

/// Standard metric names used throughout the framework.
pub mod names {
    pub const REQUESTS_TOTAL: &str = "agent.requests.total";
    pub const REQUESTS_ERRORS: &str = "agent.requests.errors";
    pub const PROVIDER_CALLS: &str = "agent.provider.calls";
    pub const PROVIDER_LATENCY_MS: &str = "agent.provider.latency_ms";
    pub const TOOL_CALLS: &str = "agent.tool.calls";
    pub const TOOL_ERRORS: &str = "agent.tool.errors";
    pub const TOOL_LATENCY_MS: &str = "agent.tool.latency_ms";
    pub const TOKENS_INPUT: &str = "agent.tokens.input";
    pub const TOKENS_OUTPUT: &str = "agent.tokens.output";
    pub const ACTIVE_SESSIONS: &str = "agent.sessions.active";
    pub const MEMORY_ENTRIES: &str = "agent.memory.entries";
}

// ---------------------------------------------------------------------------
// Logging sink (simple built-in sink)
// ---------------------------------------------------------------------------

/// A simple sink that logs metrics to stderr. Useful for development.
pub struct LoggingSink;

#[async_trait]
impl MetricsSink for LoggingSink {
    async fn record(&self, points: &[MetricPoint]) -> Result<(), MetricsError> {
        for point in points {
            eprintln!(
                "[metrics] {} = {} {:?}",
                point.name, point.value, point.tags
            );
        }
        Ok(())
    }

    async fn flush(&self) -> Result<(), MetricsError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[error("failed to record metrics: {0}")]
    RecordFailed(String),

    #[error("failed to flush metrics: {0}")]
    FlushFailed(String),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn counter_increment() {
        let collector = MetricsCollector::new();
        collector.increment("test.counter").await;
        collector.increment("test.counter").await;
        collector.increment("test.counter").await;

        assert_eq!(collector.get_counter("test.counter").await, 3);
    }

    #[tokio::test]
    async fn counter_increment_by() {
        let collector = MetricsCollector::new();
        collector.increment_by("test.counter", 10).await;
        collector.increment_by("test.counter", 5).await;

        assert_eq!(collector.get_counter("test.counter").await, 15);
    }

    #[tokio::test]
    async fn gauge_set_and_get() {
        let collector = MetricsCollector::new();
        collector.gauge("test.gauge", 42.5).await;

        assert_eq!(collector.get_gauge("test.gauge").await, Some(42.5));
        assert_eq!(collector.get_gauge("nonexistent").await, None);
    }

    #[tokio::test]
    async fn gauge_overwrite() {
        let collector = MetricsCollector::new();
        collector.gauge("test.gauge", 1.0).await;
        collector.gauge("test.gauge", 2.0).await;

        assert_eq!(collector.get_gauge("test.gauge").await, Some(2.0));
    }

    #[tokio::test]
    async fn histogram_records() {
        let collector = MetricsCollector::new();
        collector.histogram("test.latency", 10.0).await;
        collector.histogram("test.latency", 20.0).await;
        collector.histogram("test.latency", 30.0).await;

        let values = collector.get_histogram("test.latency").await;
        assert_eq!(values, vec![10.0, 20.0, 30.0]);
    }

    #[tokio::test]
    async fn snapshot_captures_all() {
        let collector = MetricsCollector::new();
        collector.increment_by("requests", 100).await;
        collector.gauge("active", 5.0).await;
        collector.histogram("latency", 15.0).await;

        let snap = collector.snapshot().await;
        assert_eq!(snap.counters["requests"], 100);
        assert_eq!(snap.gauges["active"], 5.0);
        assert_eq!(snap.histograms["latency"], vec![15.0]);
    }

    #[tokio::test]
    async fn histogram_stats() {
        let collector = MetricsCollector::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
            collector.histogram("latency", v).await;
        }

        let snap = collector.snapshot().await;
        let stats = snap.histogram_stats("latency").unwrap();

        assert_eq!(stats.count, 10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
        assert!((stats.mean - 5.5).abs() < 0.01);
        assert_eq!(stats.p50, 6.0);
    }

    #[tokio::test]
    async fn histogram_stats_empty() {
        let snap = MetricsSnapshot {
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
        };

        assert!(snap.histogram_stats("nonexistent").is_none());
    }

    #[tokio::test]
    async fn reset_clears_everything() {
        let collector = MetricsCollector::new();
        collector.increment("c").await;
        collector.gauge("g", 1.0).await;
        collector.histogram("h", 1.0).await;

        collector.reset().await;

        assert_eq!(collector.get_counter("c").await, 0);
        assert_eq!(collector.get_gauge("g").await, None);
        assert!(collector.get_histogram("h").await.is_empty());
    }

    #[tokio::test]
    async fn nonexistent_counter_returns_zero() {
        let collector = MetricsCollector::new();
        assert_eq!(collector.get_counter("nope").await, 0);
    }

    #[tokio::test]
    async fn timer_records_elapsed() {
        let collector = Arc::new(MetricsCollector::new());
        let timer = Timer::new("op.latency", collector.clone());

        // Wait a tiny bit so we get a non-zero measurement
        tokio::time::sleep(Duration::from_millis(10)).await;
        timer.stop().await;

        let values = collector.get_histogram("op.latency").await;
        assert_eq!(values.len(), 1);
        assert!(values[0] > 0.0);
    }

    #[test]
    fn metric_names_defined() {
        assert_eq!(names::REQUESTS_TOTAL, "agent.requests.total");
        assert_eq!(names::PROVIDER_CALLS, "agent.provider.calls");
        assert_eq!(names::TOKENS_INPUT, "agent.tokens.input");
    }

    #[test]
    fn snapshot_serialization() {
        let snap = MetricsSnapshot {
            counters: {
                let mut m = HashMap::new();
                m.insert("req".into(), 42);
                m
            },
            gauges: HashMap::new(),
            histograms: HashMap::new(),
        };

        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("42"));
    }

    #[tokio::test]
    async fn logging_sink_doesnt_error() {
        let sink = LoggingSink;
        let points = vec![MetricPoint {
            name: "test".into(),
            value: 1.0,
            tags: HashMap::new(),
            timestamp_ms: 0,
        }];

        sink.record(&points).await.unwrap();
        sink.flush().await.unwrap();
    }
}
