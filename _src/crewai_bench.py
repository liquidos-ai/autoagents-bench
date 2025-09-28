import asyncio
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List
from crewai import Agent, Task, Crew

from crewai.tools.base_tool import BaseTool
import yaml
from langchain_openai import ChatOpenAI

from .trip_tool import (
    TripDataEmptyError,
    TripDataNotFoundError,
    format_average_trip_duration,
)


@dataclass
class BenchmarkConfig:
    total_requests: int
    concurrency: int
    model: str
    prompt_template: str


@dataclass
class TimingBreakdown:
    total: float
    queue_wait: float
    call: float


@dataclass
class BenchmarkResult:
    name: str
    total_requests: int
    concurrency: int
    total_duration: float
    throughput_rps: float
    average_latency_ms: float
    p95_latency_ms: float
    average_queue_ms: float
    p95_queue_ms: float
    average_call_ms: float
    p95_call_ms: float
    average_processing_ms: float
    p95_processing_ms: float

    def print(self) -> None:
        print(f"--- {self.name} ---")
        print(f"requests      : {self.total_requests}")
        print(f"concurrency   : {self.concurrency}")
        print(f"total time    : {self.total_duration:.3f} s")
        print(f"throughput    : {self.throughput_rps:.2f} req/s")
        print(f"avg latency   : {self.average_latency_ms:.2f} ms")
        print(f"p95 latency   : {self.p95_latency_ms:.2f} ms")
        print(
            f"queue wait    : avg {self.average_queue_ms:.2f} ms | p95 {self.p95_queue_ms:.2f} ms"
        )
        print(
            f"llm latency   : avg {self.average_call_ms:.2f} ms | p95 {self.p95_call_ms:.2f} ms"
        )
        print(
            f"framework ovh : avg {self.average_processing_ms:.2f} ms | p95 {self.p95_processing_ms:.2f} ms"
        )


def persist_result(result: BenchmarkResult) -> None:
    output_path = Path(os.getenv("BENCH_RESULTS_PATH", "benchmark_results.json"))
    payload = asdict(result)
    payload["total_duration"] = float(payload.get("total_duration", 0.0))

    data = {}
    if output_path.exists():
        try:
            loaded = json.loads(output_path.read_text())
            if isinstance(loaded, dict):
                data = loaded
        except json.JSONDecodeError:
            pass

    data[result.name] = payload
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True))


class TripDataAverageDurationTool(BaseTool):
    name: str = "trip_data_average_duration"
    description: str = "Compute the average TLC trip duration in minutes from the trip_data.parquet dataset."

    def _run(self, *_, **__) -> str:
        try:
            return format_average_trip_duration()
        except (TripDataNotFoundError, TripDataEmptyError) as exc:
            return f"Unable to compute trip duration: {exc}"
        except Exception as exc:  # pragma: no cover - defensive guard
            return f"Unexpected error while computing trip duration: {exc}"

    async def _arun(
        self, *args, **kwargs
    ) -> str:  # pragma: no cover - used for async interface
        return self._run(*args, **kwargs)


def percentile(data: List[float], perc: float) -> float:
    if not data:
        return 0.0
    percentage = max(0.0, min(1.0, perc))
    rank = max(0, min(len(data) - 1, math.ceil(percentage * len(data)) - 1))
    return data[rank]


def load_config() -> BenchmarkConfig:
    config_path = Path(os.getenv("BENCH_CONFIG", "benchmark.yaml"))
    try:
        raw = config_path.read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Benchmark config not found at {config_path}") from exc

    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("Benchmark config must be a YAML mapping")

    try:
        total_requests = int(data["total_requests"])
        concurrency = int(data["concurrency"])
        model = str(data.get("model", "gpt-4o-mini"))
        prompt_template = str(
            data.get(
                "prompt_template",
                "Calculate the average trip duration in minutes using the available tool.",
            )
        )
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc}") from exc

    return BenchmarkConfig(
        total_requests=total_requests,
        concurrency=concurrency,
        model=model,
        prompt_template=prompt_template,
    )


def build_agent(model: str, tool: BaseTool) -> Agent:
    llm = ChatOpenAI(model=model, temperature=0.2)
    return Agent(
        role="Python Data Analyst",
        goal="Analyse TLC trip data using Python to produce clear aggregates",
        backstory=(
            "You are an experienced NYC TLC data analyst with strong Python skills who "
            "relies on helper tools to compute accurate metrics."
        ),
        allow_code_execution=False,
        allow_delegation=False,
        llm=llm,
        max_iter=3,
        verbose=False,
        tools=[tool],
    )


async def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    if config.concurrency < 1:
        raise ValueError("concurrency must be greater than zero")
    if config.total_requests < 1:
        raise ValueError("total_requests must be greater than zero")

    print(
        f"Preparing CrewAI benchmark: {config.total_requests} requests with concurrency {config.concurrency}"
    )

    breakdowns: List[TimingBreakdown] = []

    async def worker(request_id: int) -> None:
        prompt = config.prompt_template.format(i=request_id)
        submitted = time.perf_counter()
        dequeued = time.perf_counter()
        queue_wait = dequeued - submitted
        call_started = time.perf_counter()

        def kickoff() -> None:
            tool = TripDataAverageDurationTool()
            agent = build_agent(config.model, tool)
            task = Task(
                description=(
                    f"{prompt}\nUse the provided trip data tool or Python code execution "
                    "to compute the average trip duration in minutes."
                ),
                agent=agent,
                expected_output="Average trip duration summary in minutes.",
            )
            crew = Crew(agents=[agent], tasks=[task])
            result = crew.kickoff()
            # print(result)

        await asyncio.to_thread(kickoff)
        call_duration = time.perf_counter() - call_started
        total_duration = time.perf_counter() - submitted
        breakdowns.append(
            TimingBreakdown(
                total=total_duration,
                queue_wait=queue_wait,
                call=call_duration,
            )
        )

    tasks = [asyncio.create_task(worker(i)) for i in range(config.total_requests)]

    overall_started = time.perf_counter()
    await asyncio.gather(*tasks)
    total_duration = time.perf_counter() - overall_started

    totals = sorted(b.total for b in breakdowns)
    queue_waits = sorted(b.queue_wait for b in breakdowns)
    call_latencies = sorted(b.call for b in breakdowns)
    processing_latencies = sorted(max(0.0, b.total - b.call) for b in breakdowns)

    divisor = len(breakdowns) or 1
    avg_latency_ms = (sum(totals) / divisor) * 1_000.0
    avg_queue_ms = (sum(queue_waits) / divisor) * 1_000.0
    avg_call_ms = (sum(call_latencies) / divisor) * 1_000.0
    avg_processing_ms = (sum(processing_latencies) / divisor) * 1_000.0

    throughput_rps = len(breakdowns) / max(total_duration, 1e-9)

    return BenchmarkResult(
        name="CrewAI",
        total_requests=len(breakdowns),
        concurrency=config.concurrency,
        total_duration=total_duration,
        throughput_rps=throughput_rps,
        average_latency_ms=avg_latency_ms,
        p95_latency_ms=percentile(totals, 0.95) * 1_000.0,
        average_queue_ms=avg_queue_ms,
        p95_queue_ms=percentile(queue_waits, 0.95) * 1_000.0,
        average_call_ms=avg_call_ms,
        p95_call_ms=percentile(call_latencies, 0.95) * 1_000.0,
        average_processing_ms=avg_processing_ms,
        p95_processing_ms=percentile(processing_latencies, 0.95) * 1_000.0,
    )


async def run_crewai() -> None:
    config = load_config()
    result = await run_benchmark(config)

    print("\n=== CrewAI Results ===")
    result.print()
    persist_result(result)
