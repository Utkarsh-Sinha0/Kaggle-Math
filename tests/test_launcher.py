from pathlib import Path

from src.config import load_config_bundle
from src.runtime.launcher import build_vllm_command


def test_build_vllm_command_uses_reasoning_parser() -> None:
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    command = build_vllm_command(config)
    assert "--reasoning-parser" in command
    assert "nano_v3" in command
    assert "--enable-auto-tool-choice" in command

