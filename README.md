## AutoAgents Benchmark

Concurrent completion benchmarks for the `autoagents` framework alongside LangGraph and CrewAI agents.

All runners read their workload settings from `benchmark.yaml` (or a path provided via `BENCH_CONFIG`). Update that file to change request count, concurrency, model, or prompt template once and share it across languages.

All runners require an `OPENAI_API_KEY` that can call the configured models.

### Rust benchmark (AutoAgents)

```shell
export OPENAI_API_KEY=sk-your-key

cargo run --release
```

### Python benchmark (LangGraph, CrewAI)

```shell
export OPENAI_API_KEY=sk-your-key

# Using uv (recommended) or your preferred Python runner
uv run main.py
```


### Note

Python Files are in `_src` folder and Rust in `src`
