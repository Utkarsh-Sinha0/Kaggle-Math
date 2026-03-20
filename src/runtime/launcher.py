from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from src.models import SolverConfig


def resolve_model_source(config: SolverConfig) -> str:
    return str(config.model.get("local_model_path") or config.model["model_id"])


def build_vllm_command(config: SolverConfig) -> list[str]:
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        resolve_model_source(config),
        "--served-model-name",
        str(config.model.get("served_model_name", "aimo3-nano")),
        "--host",
        "127.0.0.1",
        "--port",
        str(_port_from_base_url(config.runtime.get("api_base_url", "http://127.0.0.1:8000/v1"))),
        "--max-num-seqs",
        str(config.model.get("max_num_seqs", 8)),
        "--tensor-parallel-size",
        str(config.model.get("tensor_parallel_size", 1)),
        "--max-model-len",
        str(config.model.get("max_model_len", 262144)),
        "--gpu-memory-utilization",
        str(config.model.get("gpu_memory_utilization", 0.92)),
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        str(config.model.get("tool_call_parser", "qwen3_coder")),
        "--reasoning-parser-plugin",
        str(config.model.get("reasoning_parser_plugin", "./nano_v3_reasoning_parser.py")),
        "--reasoning-parser",
        str(config.model.get("reasoning_parser", "nano_v3")),
    ]


def build_vllm_env(config: SolverConfig) -> dict[str, str]:
    env = os.environ.copy()
    if int(config.model.get("max_model_len", 262144)) > 262144:
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    return env


def launch_vllm_server(config: SolverConfig, cwd: str | None = None) -> subprocess.Popen:
    working_dir = cwd or config.project_root
    return subprocess.Popen(
        build_vllm_command(config),
        cwd=working_dir,
        env=build_vllm_env(config),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _port_from_base_url(base_url: str) -> int:
    try:
        return int(base_url.split(":")[-1].split("/")[0])
    except Exception:
        return 8000


def resolve_parser_path(config: SolverConfig) -> Path:
    parser_path = Path(str(config.model.get("reasoning_parser_plugin", "./nano_v3_reasoning_parser.py")))
    if parser_path.is_absolute():
        return parser_path
    return Path(config.project_root) / parser_path


def wait_for_vllm_server(base_url: str, timeout_seconds: int = 60) -> None:
    models_url = base_url.rstrip("/")[:-3] + "/models" if base_url.endswith("/v1") else base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"Timed out waiting for vLLM server at {models_url}")
