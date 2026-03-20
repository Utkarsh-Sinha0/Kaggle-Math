# Kaggle-Math

Cloud-first AIMO3 math solver baseline built for GitHub source control, AWS iteration, and Kaggle packaging.

## What is included

- Two prompt families: `reasoning_first` and `code_first`
- Subject router with configurable sample budgets
- Candidate generation backend with a `transformers` path and a test-friendly `mock` path
- Python execution with timeout protection
- Branch memory compression
- Heuristic selector with majority fallback
- Strict integer answer extraction with deterministic fallback
- Experiment logging to `logs/experiment_log.jsonl`
- Minimal Kaggle-compatible entrypoint

## Quick start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
pytest
python scripts/run_smoke.py
```

The smoke script uses the `mock` backend by default. Switch to the real model backend on AWS by editing `configs/runtime.yaml` and `configs/model.yaml`.

## AWS H100 path

1. Install runtime dependencies with `pip install -e .[runtime,dev]`
2. Set `configs/runtime.yaml` to `backend: "transformers"`
3. Optionally set `HF_TOKEN` if the model download requires authentication
4. Run `python scripts/run_smoke.py` or call `solve_one` directly

The `transformers` backend loads models through `AutoTokenizer` and `AutoModelForCausalLM`, uses `bfloat16`, `device_map: auto`, and `attn_implementation: sdpa` by default, and can be pointed at a local weight mount via `AIMO3_MODEL_PATH`.

## Kaggle packaging flow

Create a Kaggle-ready code bundle with:

```bash
python scripts/package_kaggle_bundle.py --zip
```

This writes `output/kaggle_bundle/` and a matching zip archive. The generated notebook expects:

- a code bundle dataset mounted at `/kaggle/input/aimo3-bundle`
- a model weights dataset mounted at `/kaggle/input/aimo3-model`

The notebook sets `AIMO3_KAGGLE=1`, forces the `transformers` backend, points the runtime at the local model mount, and runs the same `solve_one` path as AWS.
