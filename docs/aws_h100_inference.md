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

- Set `configs/runtime.yaml` to `backend: "transformers"`
- Keep `dtype: "bfloat16"`
- Keep `device_map: "auto"`
- Keep `attn_implementation: "sdpa"` unless you have validated another path
- Set `HF_TOKEN` if the model host requires it

## Run

```bash
python scripts/run_smoke.py
python -m src.kaggle_entry "What is 17 squared?"
```

## Notes

- The backend prefers `local_model_path` or `AIMO3_MODEL_PATH` when present, which is useful for pre-downloaded weights on EBS or Kaggle datasets.
- The model loader uses `AutoTokenizer` and `AutoModelForCausalLM` directly rather than the high-level pipeline so generation behavior stays explicit.

