from __future__ import annotations

from collections import Counter

from src.models import ExecResult, ModelTurn, SolverConfig
from src.solver.parsing import deterministic_fallback, extract_answer_from_text, extract_ints


def extract_candidate_answer(candidate: ModelTurn, config: SolverConfig) -> int | None:
    return extract_answer_from_text(candidate.content, config)


def extract_exec_answer(exec_result: ExecResult, config: SolverConfig) -> int | None:
    return extract_answer_from_text(f"{exec_result.stdout}\n{exec_result.stderr}", config)


def majority_answer(values: list[int | None]) -> int | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]
