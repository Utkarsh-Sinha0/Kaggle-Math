from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from src.models import ConfigBundle


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config_bundle(config_dir: str | Path) -> ConfigBundle:
    config_dir = Path(config_dir)
    project_root = str(config_dir.parent.resolve())
    return ConfigBundle(
        model=_load_yaml(config_dir / "model.yaml"),
        router=_load_yaml(config_dir / "router.yaml"),
        runtime=_load_yaml(config_dir / "runtime.yaml"),
        logging=_load_yaml(config_dir / "logging.yaml"),
        project_root=project_root,
    )


def config_hash(bundle: ConfigBundle) -> str:
    payload = {
        "model": bundle.model,
        "router": bundle.router,
        "runtime": bundle.runtime,
        "logging": bundle.logging,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]
