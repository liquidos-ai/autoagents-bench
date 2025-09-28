mod trip;

use anyhow::{Context, Result, anyhow};
use autoagents::async_trait;
use autoagents::core::agent::memory::SlidingWindowMemory;
use autoagents::core::agent::prebuilt::executor::{ReActAgent, ReActAgentOutput};
use autoagents::core::agent::task::Task;
use autoagents::core::agent::{AgentBuilder, AgentOutputT, DirectAgent};
use autoagents::core::tool::{ToolCallError, ToolInputT, ToolRuntime, ToolT};
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use autoagents_derive::{AgentHooks, AgentOutput, ToolInput, agent, tool};
use futures::stream::{FuturesUnordered, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::trip::compute_average_trip_duration;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct TripDataProcessingArgs {}

#[tool(
    name = "TripDataProcessingTool",
    description = "Calculate average trip duration from the TLC trip_data.parquet dataset",
    input = TripDataProcessingArgs,
)]
pub struct TripDataProcessorTool {}

impl Default for TripDataProcessorTool {
    fn default() -> Self {
        Self {}
    }
}

#[async_trait]
impl ToolRuntime for TripDataProcessorTool {
    async fn execute(&self, _args: Value) -> Result<Value, ToolCallError> {
        let path =
            std::env::var("TRIP_DATA_PATH").unwrap_or_else(|_| "trip_data.parquet".to_string());

        // Offload CPU-heavy work to a blocking thread
        let aggregate =
            tokio::task::spawn_blocking(move || compute_average_trip_duration(Path::new(&path)))
                .await
                .map_err(|err| ToolCallError::RuntimeError(Box::new(err)))? // Join error
                .map_err(|err| ToolCallError::RuntimeError(Box::new(err)))?; // Your compute error

        // println!("Aggregate: {:?}", aggregate);

        let summary = format!(
            "Average trip duration across {} trips is {:.2} minutes.",
            aggregate.row_count, aggregate.average_trip_duration_minutes
        );

        Ok(Value::String(summary))
    }
}

/// Math agent output with Value and Explanation
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct SimpleAgentOutput {
    #[output(description = "The Average Trip Duration")]
    value: f64,
}

impl From<ReActAgentOutput> for SimpleAgentOutput {
    fn from(output: ReActAgentOutput) -> Self {
        let resp = output.response;
        if output.done && !resp.trim().is_empty() {
            // Try to parse as structured JSON first
            if let Ok(value) = serde_json::from_str::<SimpleAgentOutput>(&resp) {
                return value;
            }
        }
        // For streaming chunks or unparseable content, create a default response
        SimpleAgentOutput { value: 0.0 }
    }
}

#[agent(
    name = "data_processing_agent",
    description = "You are a helpful assistant",
    tools = [TripDataProcessorTool],
    output = SimpleAgentOutput,
)]
#[derive(Default, Clone, AgentHooks)]
pub struct SimpleAgent {}

#[derive(Clone, Debug, Deserialize)]
struct BenchmarkConfig {
    total_requests: usize,
    concurrency: usize,
    prompt_template: String,
    model: String,
}

#[derive(Debug)]
struct BenchmarkResult {
    name: &'static str,
    total_requests: usize,
    concurrency: usize,
    total_duration: Duration,
    throughput_rps: f64,
    average_latency_ms: f64,
    p95_latency_ms: f64,
    average_queue_ms: f64,
    p95_queue_ms: f64,
    average_call_ms: f64,
    p95_call_ms: f64,
    average_processing_ms: f64,
    p95_processing_ms: f64,
    total_success: usize,
    total_failure: usize,
}

#[derive(Debug)]
struct TimingBreakdown {
    total: Duration,
    queue_wait: Duration,
    call: Duration,
    status: bool,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    async_main().await
}

async fn async_main() -> Result<()> {
    let config_path = std::env::var("BENCH_CONFIG").unwrap_or_else(|_| "benchmark.yaml".into());
    let config = load_config(Path::new(&config_path))
        .with_context(|| format!("failed to load benchmark config from {config_path}"))?;

    if config.concurrency == 0 || config.total_requests == 0 {
        return Err(anyhow!(
            "Benchmark requires at least one request and concurrency > 0"
        ));
    }

    println!(
        "Preparing benchmark: {} requests with concurrency {}",
        config.total_requests, config.concurrency
    );

    let autoagents = bench_autoagents(&config).await?;

    println!("\n=== Results ===");
    autoagents.print();
    persist_result(&autoagents).context("failed to persist benchmark results")?;

    Ok(())
}

async fn bench_autoagents(config: &BenchmarkConfig) -> Result<BenchmarkResult> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .context("OPENAI_API_KEY environment variable is required for AutoAgents benchmark")?;

    let llm = LLMBuilder::<OpenAI>::new()
        .api_key(api_key)
        .model(config.model.clone())
        .temperature(0.2)
        .build()
        .context("failed to build AutoAgents LLM client")?;

    let mut tasks = FuturesUnordered::new();
    for req_id in 0..config.total_requests {
        let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));
        let agent_handle = AgentBuilder::<_, DirectAgent>::new(ReActAgent::new(SimpleAgent {}))
            .llm(llm.clone())
            .memory(sliding_window_memory)
            .build()
            .await?;
        let prompt = build_prompt(&config.prompt_template, req_id);

        tasks.push(tokio::spawn(async move {
            let submitted = Instant::now();
            let dequeued = Instant::now();
            let queue_wait = dequeued.duration_since(submitted);

            let call_start = Instant::now();
            let result = agent_handle.agent.run(Task::new(&prompt)).await.unwrap();

            // println!("Results: {:.4}", result.value);

            let call_duration = call_start.elapsed();
            let total_duration = submitted.elapsed();

            let target = 25.70;
            let rounded = (result.value * 100.0).round() / 100.0;
            let status = (rounded - target).abs() < f64::EPSILON;

            // println!("Status: {}", status);

            TimingBreakdown {
                total: total_duration,
                queue_wait,
                call: call_duration,
                status,
            }
        }));
    }

    let mut breakdowns = Vec::with_capacity(config.total_requests);

    let start = Instant::now();

    while let Some(next) = tasks.next().await {
        match next {
            Ok(inner) => breakdowns.push(inner),
            Err(join_err) => return Err(anyhow!("Task join error: {}", join_err)),
        }
    }

    let elapsed = start.elapsed();
    Ok(summarize("AutoAgents", config, breakdowns, elapsed))
}

fn load_config(path: &Path) -> Result<BenchmarkConfig> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read config file {}", path.display()))?;
    let config: BenchmarkConfig = serde_yaml::from_str(&raw)
        .with_context(|| format!("failed to parse YAML config at {}", path.display()))?;
    Ok(config)
}

fn build_prompt(template: &str, request_id: usize) -> String {
    template.replace("{i}", &request_id.to_string())
}

fn summarize(
    name: &'static str,
    config: &BenchmarkConfig,
    breakdowns: Vec<TimingBreakdown>,
    elapsed: Duration,
) -> BenchmarkResult {
    let total_requests = breakdowns.len();
    if total_requests == 0 {
        return BenchmarkResult {
            name,
            total_requests: 0,
            concurrency: config.concurrency,
            total_duration: elapsed,
            throughput_rps: 0.0,
            average_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            average_queue_ms: 0.0,
            p95_queue_ms: 0.0,
            average_call_ms: 0.0,
            p95_call_ms: 0.0,
            average_processing_ms: 0.0,
            p95_processing_ms: 0.0,
            total_success: 0,
            total_failure: 0,
        };
    }

    let total_secs = elapsed.as_secs_f64().max(f64::EPSILON);
    let throughput_rps = total_requests as f64 / total_secs;

    let mut total_latencies: Vec<Duration> = breakdowns.iter().map(|b| b.total).collect();
    let mut queue_waits: Vec<Duration> = breakdowns.iter().map(|b| b.queue_wait).collect();
    let mut call_latencies: Vec<Duration> = breakdowns.iter().map(|b| b.call).collect();
    let mut processing_latencies: Vec<Duration> = breakdowns
        .iter()
        .map(|b| b.total.saturating_sub(b.call))
        .collect();

    total_latencies.sort_unstable();
    queue_waits.sort_unstable();
    call_latencies.sort_unstable();
    processing_latencies.sort_unstable();

    let divisor = total_requests as f64;
    let sum_totals = total_latencies.iter().map(|d| d.as_secs_f64()).sum::<f64>();
    let sum_queue = queue_waits.iter().map(|d| d.as_secs_f64()).sum::<f64>();
    let sum_call = call_latencies.iter().map(|d| d.as_secs_f64()).sum::<f64>();
    let sum_processing = processing_latencies
        .iter()
        .map(|d| d.as_secs_f64())
        .sum::<f64>();

    let total_success = breakdowns.iter().filter(|b| b.status == true).count();
    let total_failure = total_requests - total_success;

    BenchmarkResult {
        name,
        total_requests,
        concurrency: config.concurrency,
        total_duration: elapsed,
        throughput_rps,
        average_latency_ms: (sum_totals / divisor) * 1_000.0,
        p95_latency_ms: percentile_ms(&total_latencies, 0.95),
        average_queue_ms: (sum_queue / divisor) * 1_000.0,
        p95_queue_ms: percentile_ms(&queue_waits, 0.95),
        average_call_ms: (sum_call / divisor) * 1_000.0,
        p95_call_ms: percentile_ms(&call_latencies, 0.95),
        average_processing_ms: (sum_processing / divisor) * 1_000.0,
        p95_processing_ms: percentile_ms(&processing_latencies, 0.95),
        total_success,
        total_failure,
    }
}

fn percentile_ms(durations: &[Duration], percentile: f64) -> f64 {
    if durations.is_empty() {
        return 0.0;
    }

    let capped = percentile.clamp(0.0, 1.0);
    let rank = (capped * durations.len() as f64).ceil().max(1.0) as usize - 1;
    let idx = rank.min(durations.len() - 1);
    durations[idx].as_secs_f64() * 1_000.0
}

fn persist_result(result: &BenchmarkResult) -> Result<()> {
    let output_path = std::env::var("BENCH_RESULTS_PATH")
        .unwrap_or_else(|_| "benchmark_results.json".to_string());
    let path = Path::new(&output_path);

    let mut data = fs::read_to_string(path)
        .ok()
        .and_then(|content| serde_json::from_str::<Map<String, Value>>(&content).ok())
        .unwrap_or_default();

    data.insert(
        result.name.to_string(),
        json!({
            "name": result.name,
            "total_requests": result.total_requests,
            "concurrency": result.concurrency,
            "total_duration": result.total_duration.as_secs_f64(),
            "throughput_rps": result.throughput_rps,
            "average_latency_ms": result.average_latency_ms,
            "p95_latency_ms": result.p95_latency_ms,
            "average_queue_ms": result.average_queue_ms,
            "p95_queue_ms": result.p95_queue_ms,
            "average_call_ms": result.average_call_ms,
            "p95_call_ms": result.p95_call_ms,
            "average_processing_ms": result.average_processing_ms,
            "p95_processing_ms": result.p95_processing_ms,
            "total_success": result.total_success,
            "total_failure": result.total_failure,
        }),
    );

    let output_value = Value::Object(data);
    let serialized = serde_json::to_string_pretty(&output_value)
        .context("failed to serialise benchmark results to JSON")?;
    fs::write(path, serialized)
        .with_context(|| format!("failed to write benchmark results to {}", path.display()))?;

    Ok(())
}

impl BenchmarkResult {
    fn print(&self) {
        println!("--- {} ---", self.name);
        println!("requests      : {}", self.total_requests);
        println!("concurrency   : {}", self.concurrency);
        println!("total time    : {:.3} s", self.total_duration.as_secs_f64());
        println!("throughput    : {:.2} req/s", self.throughput_rps);
        println!("avg latency   : {:.2} ms", self.average_latency_ms);
        println!("p95 latency   : {:.2} ms", self.p95_latency_ms);
        println!(
            "queue wait    : avg {:.2} ms | p95 {:.2} ms",
            self.average_queue_ms, self.p95_queue_ms
        );
        println!(
            "llm latency   : avg {:.2} ms | p95 {:.2} ms",
            self.average_call_ms, self.p95_call_ms
        );
        println!(
            "framework ovh : avg {:.2} ms | p95 {:.2} ms",
            self.average_processing_ms, self.p95_processing_ms
        );
    }
}
