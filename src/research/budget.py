from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.models import BudgetLedger, SolverConfig


def _ledger_path(config: SolverConfig) -> Path:
    return Path(config.project_root) / config.logging.get("budget_ledger_path", "logs/budget_ledger.json")


def load_budget_ledger(config: SolverConfig) -> BudgetLedger:
    path = _ledger_path(config)
    if not path.exists():
        return BudgetLedger(
            budget_hours_limit=float(config.runtime.get("budget_hours_limit", 45.0)),
            consumed_hours=0.0,
            estimated_cost_usd=0.0,
            blocked=False,
            entries=[],
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return BudgetLedger(**data)


def save_budget_ledger(config: SolverConfig, ledger: BudgetLedger) -> None:
    path = _ledger_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(ledger), indent=2), encoding="utf-8")


def ensure_budget_headroom(config: SolverConfig, planned_hours: float) -> None:
    ledger = load_budget_ledger(config)
    if ledger.consumed_hours + planned_hours > ledger.budget_hours_limit:
        raise RuntimeError(
            f"Budget headroom exceeded: planned {planned_hours:.2f}h with {ledger.consumed_hours:.2f}h already consumed "
            f"against a {ledger.budget_hours_limit:.2f}h limit."
        )


def record_budget_usage(config: SolverConfig, hours: float, estimated_cost_usd: float, reason: str) -> BudgetLedger:
    ledger = load_budget_ledger(config)
    ledger.consumed_hours += hours
    ledger.estimated_cost_usd += estimated_cost_usd
    ledger.blocked = ledger.consumed_hours >= ledger.budget_hours_limit
    ledger.entries.append(
        {
            "reason": reason,
            "hours": round(hours, 6),
            "estimated_cost_usd": round(estimated_cost_usd, 6),
            "consumed_hours": round(ledger.consumed_hours, 6),
        }
    )
    save_budget_ledger(config, ledger)
    return ledger

