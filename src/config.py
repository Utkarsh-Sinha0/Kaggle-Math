from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from src.models import SolverConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config_bundle(config_dir: str | Path) -> SolverConfig:
    config_dir = Path(config_dir)
    project_root = str(config_dir.parent.resolve())
    bundle = SolverConfig(
        model=_load_yaml(config_dir / "model.yaml"),
        runtime=_load_yaml(config_dir / "runtime.yaml"),
        solver=_load_yaml(config_dir / "solver.yaml"),
        logging=_load_yaml(config_dir / "logging.yaml"),
        research=_load_yaml(config_dir / "research_policy.yaml"),
        project_root=project_root,
    )
    apply_environment_overrides(bundle)
    return bundle


def apply_environment_overrides(bundle: SolverConfig) -> None:
    model_path = os.getenv("AIMO3_MODEL_PATH")
    if model_path:
        bundle.model["local_model_path"] = model_path
        bundle.model["local_files_only"] = True

    backend = os.getenv("AIMO3_BACKEND")
    if backend:
        bundle.runtime["backend"] = backend

    api_base_url = os.getenv("AIMO3_API_BASE_URL")
    if api_base_url:
        bundle.runtime["api_base_url"] = api_base_url

    if os.getenv("AIMO3_LAUNCH_SERVER") == "1":
        bundle.runtime["launch_server"] = True

    if os.getenv("AIMO3_KAGGLE") == "1":
        bundle.model["local_files_only"] = True
        bundle.runtime["backend"] = "vllm"
        bundle.runtime["launch_server"] = True


def config_hash(bundle: SolverConfig) -> str:
    payload = {
        "model": bundle.model,
        "runtime": bundle.runtime,
        "solver": bundle.solver,
        "logging": bundle.logging,
        "research": bundle.research,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]
