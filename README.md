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

