import json
from pathlib import Path

import yaml

from scripts.package_kaggle_bundle import build_kaggle_bundle


def test_build_bundle_rewrites_configs_and_notebook(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "bundle"
    manifest = build_kaggle_bundle(
        repo_root=repo_root,
        output_dir=output_dir,
        bundle_mount="/kaggle/input/aimo3-bundle",
        model_mount="/kaggle/input/aimo3-model",
        create_zip=False,
    )

    model_config = yaml.safe_load((output_dir / "configs" / "model.yaml").read_text(encoding="utf-8"))
    runtime_config = yaml.safe_load((output_dir / "configs" / "runtime.yaml").read_text(encoding="utf-8"))
    notebook = json.loads((output_dir / "notebooks" / "final_kaggle.ipynb").read_text(encoding="utf-8"))

    assert manifest.runtime_backend == "vllm"
    assert model_config["local_model_path"] == "/kaggle/input/aimo3-model"
    assert runtime_config["backend"] == "vllm"
    assert any("AIMO3_BACKEND" in line and "vllm" in line for line in notebook["cells"][1]["source"])

