import json
from pathlib import Path

from src.config import load_config_bundle
from src.research.eval import run_eval
from src.runtime.factory import create_runtime


def test_run_eval_returns_summary(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval_small.jsonl"
    eval_path.write_text(json.dumps({"problem": "What is 2 + 3 + 5?", "answer": 10}) + "\n", encoding="utf-8")
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    config.logging["budget_ledger_path"] = str(tmp_path / "budget_ledger.json")
    runtime = create_runtime(config)
    summary = run_eval(eval_path, runtime, config)
    assert summary.total_examples == 1
    assert summary.selector_at_n >= 0.0
