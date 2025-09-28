# Repository Guidelines

## Project Structure & Module Organization
The Rust benchmark lives in `src/` (`main.rs`, `trip.rs`) and drives AutoAgents workloads, while Python runners sit in `_src/` with `main.py` as the async entrypoint. Shared configuration is stored in `benchmark.yaml`, generated metrics accumulate in `benchmark_results*.json`, and comparison plots render into `plots/`. Treat large data inputs such as `trip_data.parquet` as read-only artifacts and keep sensitive config out of version control.

## Build, Test, and Development Commands
`cargo run --release` executes the AutoAgents benchmark defined by the current `benchmark.yaml`. `cargo check` offers a fast type pass before commits. `uv run main.py` launches the LangGraph workflow (uncomment CrewAI in `main.py` when needed). `uv run python plot_benchmarks.py` refreshes charts in `plots/`. Override defaults with `BENCH_CONFIG` or `BENCH_RESULTS_PATH` to point at alternate config/result files.

## Coding Style & Naming Conventions
Run `cargo fmt` to maintain standard Rust formatting and keep four-space indents with module-level doc comments when adding flows. Follow `cargo clippy -- -D warnings` so lints never regress performance-critical loops. Python modules stay snake_case under `_src`, prefer explicit coroutine names (`run_crewai`, `run_langgraph`), and rely on type hints plus descriptive docstrings. Load runtime secrets (e.g., `OPENAI_API_KEY`) from environment variables rather than code.

## Testing Guidelines
Unit coverage is currently sparse; add Rust tests beside each module and Python tests under `tests/` named `test_*.py` as you extend functionality. Run `cargo test` for Rust validation and `uv run python -m pytest` for Python suites. When a benchmark needs live APIs, mock external calls where practical and call out benchmark parameters in the PR description.

## Commit & Pull Request Guidelines
Commits follow short, imperative subjects (`Add check for success`) with optional bodies for dataset or API context. Pull requests should summarize benchmark deltas, reference issues, and attach relevant output or plots. Note any `OPENAI_API_KEY` prerequisites, confirm the commands above succeed locally, and include screenshots when plots change.
