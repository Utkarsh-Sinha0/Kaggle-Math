import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config_bundle
from src.solve_one import solve_one


def main() -> None:
    config = load_config_bundle(ROOT / "configs")
    result = solve_one("What is 2 + 3 + 5?", config)
    print(f"answer={result.answer_int} valid={result.valid} fallback={result.fallback_used}")


if __name__ == "__main__":
    main()
