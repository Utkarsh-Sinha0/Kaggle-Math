#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[runtime,dev]
python -m pip install huggingface_hub

if [[ ! -f nano_v3_reasoning_parser.py ]]; then
  wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py
fi

if [[ ! -d models/nemotron-nano ]]; then
  huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --local-dir ./models/nemotron-nano --local-dir-use-symlinks False
fi

echo "AWS AMI setup complete."

