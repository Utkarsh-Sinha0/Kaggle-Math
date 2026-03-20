from pathlib import Path

from src.config import load_config_bundle
from src.research.budget import load_budget_ledger, record_budget_usage


def test_budget_ledger_records_usage(tmp_path: Path) -> None:
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    config.logging["budget_ledger_path"] = str(tmp_path / "budget_ledger.json")
    ledger = record_budget_usage(config, hours=1.25, estimated_cost_usd=4.5, reason="test")
    assert ledger.consumed_hours == 1.25
    assert load_budget_ledger(config).entries[0]["reason"] == "test"

