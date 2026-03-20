from __future__ import annotations

from collections import Counter

from src.models import Candidate, ExecResult, MemoryState, SelectionResult


def _score_candidate(
    candidate: Candidate,
    memory: MemoryState,
    exec_result: ExecResult,
    answer_counts: Counter,
    selector_config: dict,
) -> float:
    score = candidate.confidence
    if candidate.extracted_answer is not None:
        score += answer_counts[candidate.extracted_answer] * float(selector_config.get("answer_agreement_bonus", 2.0))
    if exec_result.success:
        score += float(selector_config.get("exec_success_bonus", 1.5))
    if candidate.code:
        score += float(selector_config.get("code_bonus", 0.5))
    if memory.key_facts:
        score += float(selector_config.get("reasoning_bonus", 0.25))
    return score


def select_candidate(
    problem: str,
    candidates: list[Candidate],
    memory_states: list[MemoryState],
    exec_results: list[ExecResult],
    selector_config: dict,
) -> SelectionResult:
    del problem
    answers = [candidate.extracted_answer for candidate in candidates if candidate.extracted_answer is not None]
    answer_counts = Counter(answers)
    scoreboard: list[tuple[int, float]] = []

    best_idx = 0
    best_score = float("-inf")
    for idx, candidate in enumerate(candidates):
        exec_result = exec_results[idx] if idx < len(exec_results) else ExecResult(False, "", "missing exec result", 1)
        memory = memory_states[idx] if idx < len(memory_states) else MemoryState("general", "final integer answer", [], [], [], [])
        score = _score_candidate(candidate, memory, exec_result, answer_counts, selector_config)
        scoreboard.append((candidate.candidate_id, score))
        if score > best_score:
            best_idx = idx
            best_score = score

    selected = candidates[best_idx]
    mode = "selector"
    if selected.extracted_answer is None and answers:
        majority_answer = answer_counts.most_common(1)[0][0]
        majority_candidate = next(candidate for candidate in candidates if candidate.extracted_answer == majority_answer)
        selected = majority_candidate
        mode = "majority-fallback"

    return SelectionResult(
        selected_candidate=selected,
        mode=mode,
        selected_answer=selected.extracted_answer,
        scoreboard=scoreboard,
    )

