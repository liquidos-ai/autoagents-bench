import asyncio
import json
import math
from numbers import Number
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.store.base import Result
from pydantic import BaseModel, Field

from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import yaml

from .trip_tool import (
    TripDataEmptyError,
    TripDataNotFoundError,
    format_average_trip_duration,
)


FINAL_AGGREGATION_RESPONSE: float = 25.7


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
    status: bool


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
    total_success: int
    total_failure: int

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
        print(f"success count  : {self.total_success}")
        print(f"failure count  : {self.total_failure}")


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


@tool("trip_data_average_duration", return_direct=False)
def trip_data_average_duration_tool() -> str:
    """Summarise the average TLC trip duration from the parquet dataset."""

    try:
        return format_average_trip_duration()
    except (TripDataNotFoundError, TripDataEmptyError) as exc:
        return (
            f"Unable to compute trip duration: {exc}"  # Agent can relay precise issue.
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        return f"Unexpected error while computing trip duration: {exc}"


def percentile(data: List[float], perc: float) -> float:
    if not data:
        return 0.0
    percentage = max(0.0, min(1.0, perc))
    rank = max(0, min(len(data) - 1, math.ceil(percentage * len(data)) - 1))
    return data[rank]


# 1. Define your structured output schema
class TripDurationResult(BaseModel):
    average_trip_duration_minutes: float = Field(
        ..., description="Average trip duration in minutes"
    )
    row_count: int = Field(..., description="Number of trips in dataset")


class StructuredResponseSchema(BaseModel):
    """Always use this tool to structure your response to the user."""

    answer: float = Field(description="The Average Trip Duration in Minutes")


async def run_langgraph_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    if config.concurrency < 1:
        raise ValueError("concurrency must be greater than zero")
    if config.total_requests < 1:
        raise ValueError("total_requests must be greater than zero")

    print(
        f"Preparing LangGraph benchmark: {config.total_requests} requests with concurrency {config.concurrency}"
    )

    llm = ChatOpenAI(model=config.model)
    agent = create_react_agent(
        llm,
        prompt="You are helpful assistant, Provide the answer to the user's question in structured format",
        tools=[trip_data_average_duration_tool],
        response_format=StructuredResponseSchema,
    )

    breakdowns: List[TimingBreakdown] = []

    async def worker(request_id: int) -> None:
        prompt = config.prompt_template.format(i=request_id)
        submitted = time.perf_counter()
        dequeued = time.perf_counter()
        queue_wait = dequeued - submitted
        call_started = time.perf_counter()
        result = await agent.ainvoke({"messages": [("user", prompt)]})

        response = result["structured_response"]
        status = response.answer == FINAL_AGGREGATION_RESPONSE

        call_duration = time.perf_counter() - call_started
        total_duration = time.perf_counter() - submitted
        breakdowns.append(
            TimingBreakdown(
                total=total_duration,
                queue_wait=queue_wait,
                call=call_duration,
                status=status,
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
    success_count = sum(1 for b in breakdowns if b.status)
    failure_count = len(breakdowns) - success_count

    return BenchmarkResult(
        name="LangGraph",
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
        total_success=success_count,
        total_failure=failure_count,
    )


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
                "Calculate the average trip duration in minutes using the provided tools.",
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


async def run_langgraph() -> None:
    config = load_config()
    result = await run_langgraph_benchmark(config)

    print("\n=== LangGraph Results ===")
    result.print()
    persist_result(result)
