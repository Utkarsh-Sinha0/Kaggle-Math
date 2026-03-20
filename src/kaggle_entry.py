from __future__ import annotations

import sys
from pathlib import Path

from src.config import load_config_bundle
from src.solve_one import solve_one


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config_bundle(root / "configs")
    if len(sys.argv) > 1:
        problem_text = sys.argv[1]
    else:
        problem_text = sys.stdin.read().strip()

    result = solve_one(problem_text, config, data_slice_id="kaggle")
    print(result.answer_int)


if __name__ == "__main__":
    main()
