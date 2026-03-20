from __future__ import annotations

from pathlib import Path

from src.models import BundleManifest, SolverConfig


def build_kaggle_bundle(config: SolverConfig, model_mount: str, bundle_mount: str) -> BundleManifest:
    from scripts.package_kaggle_bundle import build_kaggle_bundle as build_bundle_script

    repo_root = Path(config.project_root)
    output_dir = repo_root / config.runtime.get("kaggle_bundle_dir", "output/kaggle_bundle")
    return build_bundle_script(repo_root, output_dir, bundle_mount, model_mount, create_zip=False)
