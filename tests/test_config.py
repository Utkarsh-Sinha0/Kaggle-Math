from pathlib import Path

from src.config import load_config_bundle


def test_environment_overrides(monkeypatch) -> None:
    monkeypatch.setenv("AIMO3_MODEL_PATH", "/tmp/model")
    monkeypatch.setenv("AIMO3_BACKEND", "transformers")
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    assert config.model["local_model_path"] == "/tmp/model"
    assert config.model["local_files_only"] is True
    assert config.runtime["backend"] == "transformers"

