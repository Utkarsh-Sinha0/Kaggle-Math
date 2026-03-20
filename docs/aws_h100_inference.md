# AWS H100 Inference Notes

## Intended environment

- Instance: AWS `p5.4xlarge`
- GPU: 1x H100 80GB
- Python: 3.11 or 3.12

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[runtime,dev]
```

## Configure

- Set `AIMO3_BACKEND=vllm`
- Point `AIMO3_MODEL_PATH` at the local Nemotron Nano weights
- Keep the parser plugin file `nano_v3_reasoning_parser.py` next to the repo root
- Use the official vLLM reasoning parser and tool-call parser from the model card

## Run

```bash
bash scripts/launch_vllm_server.sh
python scripts/run_smoke.py
python -m src.kaggle_entry "What is 17 squared?"
```

## Notes

- The runtime is vLLM-first and uses the local OpenAI-compatible endpoint.
- `AIMO3_LAUNCH_SERVER=1` can be used to let the code path start the local server automatically.
- Budget usage should be recorded via the budget ledger before long eval or autoresearch runs.
