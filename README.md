# Kaggle-Math

Nemotron Nano, vLLM-first AIMO3 solver system for AWS iteration, Kaggle-offline packaging, and bounded `autoresearch`.

## Architecture

The repo is organized around four domains:

- `src/runtime/`: vLLM launcher and OpenAI-compatible client, plus a mock runtime for tests
- `src/solver/`: routing, prompts, branch search, memory compression, selector, and final answer guarantee
- `src/research/`: eval runner and budget ledger
- `scripts/`: AWS setup, vLLM launch, eval prep, smoke runs, and Kaggle bundle packaging

## Core guarantees

- Nemotron Nano is the primary runtime target
- vLLM is the only real inference backend
- the solver always emits one integer in `0..99999`
- the research loop is bounded to a single editable policy file
- GPU-hour usage is tracked against a budget ledger

## Quick start

For local dry runs and tests:

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
pytest
python scripts/run_smoke.py
```

The default backend is `mock`, so tests and smoke runs do not require a GPU.

## AWS H100 path

```bash
bash scripts/setup_aws_ami.sh
export AIMO3_BACKEND=vllm
export AIMO3_MODEL_PATH=./models/nemotron-nano
bash scripts/launch_vllm_server.sh
python scripts/run_smoke.py
```

If you prefer automatic server launch from the app path, set:

```bash
export AIMO3_LAUNCH_SERVER=1
python -m src.kaggle_entry "What is 123 + 456?"
```

## Public interfaces

- `solve_one(problem_text, runtime, config) -> FinalAnswer`
- `runtime.chat_batch(messages, sampling_plan, tool_schema=None) -> list[ModelTurn]`
- `compress_branch(branch_trace, config) -> BranchState`
- `select_final(branches, selector_config) -> SelectionResult`
- `run_eval(eval_path, runtime, config) -> EvalSummary`
- `build_kaggle_bundle(...) -> BundleManifest`

## Kaggle packaging flow

```bash
python scripts/package_kaggle_bundle.py --zip
```

This creates `output/kaggle_bundle/` and optionally `output/kaggle_bundle.zip`. The generated notebook expects:

- code bundle dataset at `/kaggle/input/aimo3-bundle`
- model dataset at `/kaggle/input/aimo3-model`

The notebook uses the same `solve_one` path as AWS and launches a local vLLM server against the attached model mount.
