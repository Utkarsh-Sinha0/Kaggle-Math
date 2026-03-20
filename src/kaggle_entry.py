from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import load_config_bundle
from src.solve_one import solve_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Kaggle-facing AIMO3 entrypoint")
    parser.add_argument("problem_text", nargs="?", default=None)
    parser.add_argument("--config-dir", default=None, help="Path to a config directory. Defaults to repo configs/")
    parser.add_argument("--problem-file", default=None, help="Read problem text from a file")
    parser.add_argument("--data-slice-id", default="kaggle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_dir = Path(args.config_dir) if args.config_dir else root / "configs"
    config = load_config_bundle(config_dir)

    if args.problem_file:
        problem_text = Path(args.problem_file).read_text(encoding="utf-8").strip()
    elif args.problem_text:
        problem_text = args.problem_text
    else:
        problem_text = sys.stdin.read().strip()

    result = solve_one(problem_text, config, data_slice_id=args.data_slice_id)
    print(result.answer_int)


if __name__ == "__main__":
    main()
