from pathlib import Path

from src.config import load_config_bundle


def test_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("AIMO3_MODEL_PATH", "/tmp/model")
    monkeypatch.setenv("AIMO3_BACKEND", "vllm")
    monkeypatch.setenv("AIMO3_API_BASE_URL", "http://127.0.0.1:9000/v1")
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    assert config.model["local_model_path"] == "/tmp/model"
    assert config.model["local_files_only"] is True
    assert config.runtime["backend"] == "vllm"
    assert config.runtime["api_base_url"] == "http://127.0.0.1:9000/v1"

