from __future__ import annotations

import json
from pathlib import Path

from src.models import EvalSummary, SolverConfig
from src.research.budget import record_budget_usage
from src.solver.solve import solve_one


def _iter_eval_items(eval_path: Path):
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def run_eval(eval_path: str | Path, runtime, config: SolverConfig) -> EvalSummary:
    path = Path(eval_path)
    items = list(_iter_eval_items(path))
    if not items:
        return EvalSummary(
            eval_path=str(path),
            total_examples=0,
            answered_examples=0,
            pass_at_1=0.0,
            majority_at_n=0.0,
            selector_at_n=0.0,
            invalid_answer_rate=0.0,
            timeout_rate=0.0,
            average_runtime_seconds=0.0,
            estimated_cost_usd=0.0,
        )

    pass_at_1 = 0.0
    majority_at_n = 0.0
    selector_at_n = 0.0
    invalid = 0
    timeout = 0
    runtime_seconds = 0.0

    for index, item in enumerate(items):
        problem_text = item.get("problem") or item.get("question") or item.get("prompt") or ""
        expected = item.get("answer")
        result = solve_one(problem_text, runtime, config, data_slice_id=path.stem)
        runtime_seconds += float(result.provenance.get("runtime_seconds", 0.0))
        if result.valid:
            if result.provenance.get("selected_answer") == expected:
                selector_at_n += 1.0
            if result.provenance.get("majority_answer") == expected:
                majority_at_n += 1.0
            branch_answers = result.provenance.get("branch_answers", [])
            if branch_answers and branch_answers[0] == expected:
                pass_at_1 += 1.0
        else:
            invalid += 1
        if result.provenance.get("timed_out"):
            timeout += 1

    average_runtime = runtime_seconds / len(items)
    estimated_cost = float(config.runtime.get("instance_hourly_cost_usd", 0.0)) * (runtime_seconds / 3600.0)
    record_budget_usage(config, runtime_seconds / 3600.0, estimated_cost, f"eval:{path.name}")
    return EvalSummary(
        eval_path=str(path),
        total_examples=len(items),
        answered_examples=len(items) - invalid,
        pass_at_1=pass_at_1 / len(items),
        majority_at_n=majority_at_n / len(items),
        selector_at_n=selector_at_n / len(items),
        invalid_answer_rate=invalid / len(items),
        timeout_rate=timeout / len(items),
        average_runtime_seconds=average_runtime,
        estimated_cost_usd=estimated_cost,
    )
