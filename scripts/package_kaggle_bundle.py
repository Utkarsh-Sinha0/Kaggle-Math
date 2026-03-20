from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


NOTEBOOK_TEMPLATE = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AIMO3 Kaggle Submission Baseline\n",
                "\n",
                "This notebook expects two Kaggle datasets:\n",
                "- a code bundle dataset mounted at `bundle_mount`\n",
                "- a model weights dataset mounted at `model_mount`\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "BUNDLE_ROOT = Path(\"{bundle_mount}\")\n",
                "MODEL_ROOT = Path(\"{model_mount}\")\n",
                "os.environ[\"AIMO3_KAGGLE\"] = \"1\"\n",
                "os.environ[\"AIMO3_BACKEND\"] = \"transformers\"\n",
                "os.environ[\"AIMO3_MODEL_PATH\"] = str(MODEL_ROOT)\n",
                "sys.path.insert(0, str(BUNDLE_ROOT))\n",
                "\n",
                "from src.config import load_config_bundle\n",
                "from src.solve_one import solve_one\n",
                "\n",
                "config = load_config_bundle(BUNDLE_ROOT / \"configs\")\n",
                "problem_text = \"What is 2 + 3 + 5?\"\n",
                "result = solve_one(problem_text, config, data_slice_id=\"kaggle_dry_run\")\n",
                "print(result.answer_int)\n",
            ],
        },
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Kaggle bundle for the AIMO3 solver baseline")
    parser.add_argument("--output-dir", default="output/kaggle_bundle")
    parser.add_argument("--bundle-mount", default="/kaggle/input/aimo3-bundle")
    parser.add_argument("--model-mount", default="/kaggle/input/aimo3-model")
    parser.add_argument("--zip", action="store_true", dest="create_zip")
    return parser.parse_args()


def _copy_tree(repo_root: Path, output_dir: Path) -> None:
    for name in ["src", "configs", "prompts", "docs"]:
        shutil.copytree(repo_root / name, output_dir / name, dirs_exist_ok=True)
    for name in ["README.md", "pyproject.toml"]:
        shutil.copy2(repo_root / name, output_dir / name)


def _rewrite_bundle_configs(output_dir: Path, model_mount: str) -> None:
    model_config_path = output_dir / "configs" / "model.yaml"
    runtime_config_path = output_dir / "configs" / "runtime.yaml"

    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8")) or {}
    model_config["local_model_path"] = model_mount
    model_config["local_files_only"] = True
    model_config_path.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")

    runtime_config = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8")) or {}
    runtime_config["backend"] = "transformers"
    runtime_config_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")


def _write_notebook(output_dir: Path, bundle_mount: str, model_mount: str) -> None:
    notebook = json.loads(json.dumps(NOTEBOOK_TEMPLATE))
    notebook["cells"][1]["source"] = [line.format(bundle_mount=bundle_mount, model_mount=model_mount) for line in notebook["cells"][1]["source"]]
    (output_dir / "notebooks").mkdir(parents=True, exist_ok=True)
    (output_dir / "notebooks" / "kaggle_submission_baseline.ipynb").write_text(
        json.dumps(notebook, indent=2),
        encoding="utf-8",
    )


def _write_manifest(output_dir: Path, bundle_mount: str, model_mount: str) -> None:
    manifest = {
        "bundle_mount": bundle_mount,
        "model_mount": model_mount,
        "included_paths": ["src", "configs", "prompts", "docs", "README.md", "pyproject.toml"],
        "entrypoint": "python -m src.kaggle_entry --config-dir ./configs",
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_bundle(repo_root: Path, output_dir: Path, bundle_mount: str, model_mount: str, create_zip: bool) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _copy_tree(repo_root, output_dir)
    _rewrite_bundle_configs(output_dir, model_mount)
    _write_notebook(output_dir, bundle_mount, model_mount)
    _write_manifest(output_dir, bundle_mount, model_mount)

    if create_zip:
        shutil.make_archive(str(output_dir), "zip", output_dir)
    return output_dir


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    build_bundle(repo_root, output_dir, args.bundle_mount, args.model_mount, args.create_zip)
    print(f"Created Kaggle bundle at {output_dir}")


if __name__ == "__main__":
    main()
