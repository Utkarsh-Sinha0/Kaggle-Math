from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
import stat

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import BundleManifest


NOTEBOOK_TEMPLATE = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# AIMO3 Nemotron Nano Kaggle Submission\n",
                "\n",
                "This notebook expects an attached code bundle dataset and a separate model dataset.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import subprocess\n",
                "import sys\n",
                "import time\n",
                "from pathlib import Path\n",
                "\n",
                "BUNDLE_ROOT = Path(\"{bundle_mount}\")\n",
                "MODEL_ROOT = Path(\"{model_mount}\")\n",
                "os.environ[\"AIMO3_KAGGLE\"] = \"1\"\n",
                "os.environ[\"AIMO3_BACKEND\"] = \"vllm\"\n",
                "os.environ[\"AIMO3_MODEL_PATH\"] = str(MODEL_ROOT)\n",
                "os.environ[\"AIMO3_LAUNCH_SERVER\"] = \"1\"\n",
                "sys.path.insert(0, str(BUNDLE_ROOT))\n",
                "\n",
                "from src.config import load_config_bundle\n",
                "from src.runtime.factory import create_runtime\n",
                "from src.runtime.launcher import launch_vllm_server\n",
                "from src.solver.solve import solve_one\n",
                "\n",
                "config = load_config_bundle(BUNDLE_ROOT / \"configs\")\n",
                "server = launch_vllm_server(config, cwd=str(BUNDLE_ROOT))\n",
                "time.sleep(20)\n",
                "runtime = create_runtime(config)\n",
                "problem_text = \"What is 2 + 3 + 5?\"\n",
                "result = solve_one(problem_text, runtime, config, data_slice_id=\"kaggle_dry_run\")\n",
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
    parser = argparse.ArgumentParser(description="Build a Kaggle bundle for the AIMO3 Nemotron Nano solver")
    parser.add_argument("--output-dir", default="output/kaggle_bundle")
    parser.add_argument("--bundle-mount", default="/kaggle/input/aimo3-bundle")
    parser.add_argument("--model-mount", default="/kaggle/input/aimo3-model")
    parser.add_argument("--zip", action="store_true", dest="create_zip")
    return parser.parse_args()


def _copy_tree(repo_root: Path, output_dir: Path) -> None:
    for name in ["src", "configs", "prompts", "docs"]:
        shutil.copytree(repo_root / name, output_dir / name, dirs_exist_ok=True)
    for name in ["README.md", "pyproject.toml", "program.md"]:
        shutil.copy2(repo_root / name, output_dir / name)
    for script_name in ["launch_vllm_server.sh", "setup_aws_ami.sh"]:
        shutil.copy2(repo_root / "scripts" / script_name, output_dir / "scripts" / script_name)
    parser_path = repo_root / "nano_v3_reasoning_parser.py"
    if parser_path.exists():
        shutil.copy2(parser_path, output_dir / "nano_v3_reasoning_parser.py")


def _rewrite_bundle_configs(output_dir: Path, model_mount: str) -> None:
    model_config_path = output_dir / "configs" / "model.yaml"
    runtime_config_path = output_dir / "configs" / "runtime.yaml"

    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8")) or {}
    model_config["local_model_path"] = model_mount
    model_config["local_files_only"] = True
    model_config_path.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")

    runtime_config = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8")) or {}
    runtime_config["backend"] = "vllm"
    runtime_config["launch_server"] = True
    runtime_config_path.write_text(yaml.safe_dump(runtime_config, sort_keys=False), encoding="utf-8")


def _write_notebook(output_dir: Path, bundle_mount: str, model_mount: str) -> Path:
    notebook = json.loads(json.dumps(NOTEBOOK_TEMPLATE))
    notebook["cells"][1]["source"] = [line.format(bundle_mount=bundle_mount, model_mount=model_mount) for line in notebook["cells"][1]["source"]]
    notebook_dir = output_dir / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebook_dir / "final_kaggle.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return notebook_path


def _write_manifest(output_dir: Path, notebook_path: Path, bundle_mount: str, model_mount: str) -> BundleManifest:
    manifest = BundleManifest(
        output_dir=str(output_dir),
        bundle_mount=bundle_mount,
        model_mount=model_mount,
        notebook_path=str(notebook_path),
        included_paths=["src", "configs", "prompts", "docs", "scripts", "README.md", "pyproject.toml", "program.md"],
        runtime_backend="vllm",
    )
    (output_dir / "bundle_manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return manifest


def build_kaggle_bundle(repo_root: Path, output_dir: Path, bundle_mount: str, model_mount: str, create_zip: bool) -> BundleManifest:
    if output_dir.exists():
        shutil.rmtree(output_dir, onerror=_handle_remove_readonly)
    (output_dir / "scripts").mkdir(parents=True, exist_ok=True)
    _copy_tree(repo_root, output_dir)
    _rewrite_bundle_configs(output_dir, model_mount)
    notebook_path = _write_notebook(output_dir, bundle_mount, model_mount)
    manifest = _write_manifest(output_dir, notebook_path, bundle_mount, model_mount)
    if create_zip:
        shutil.make_archive(str(output_dir), "zip", output_dir)
    return manifest


def _handle_remove_readonly(func, path, exc_info) -> None:
    del exc_info
    os.chmod(path, stat.S_IWRITE)
    func(path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = build_kaggle_bundle(repo_root, Path(args.output_dir), args.bundle_mount, args.model_mount, args.create_zip)
    print(f"Created Kaggle bundle at {manifest.output_dir}")


if __name__ == "__main__":
    main()
