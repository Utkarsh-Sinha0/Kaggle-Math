#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${AIMO3_MODEL_PATH:-./models/nemotron-nano}"
PORT="${AIMO3_PORT:-8000}"
MAX_MODEL_LEN="${AIMO3_MAX_MODEL_LEN:-262144}"
SERVED_MODEL_NAME="${AIMO3_SERVED_MODEL_NAME:-aimo3-nano}"

if [[ "${MAX_MODEL_LEN}" -gt 262144 ]]; then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --max-num-seqs 8 \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization 0.92 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin nano_v3_reasoning_parser.py \
  --reasoning-parser nano_v3

