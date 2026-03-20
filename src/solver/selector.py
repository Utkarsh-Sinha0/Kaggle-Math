from __future__ import annotations

from collections import Counter

from src.models import BranchState, SelectionResult


def _build_critique(branches: list[BranchState]) -> str:
    if not branches:
        return "No branches available."
    best = max(branches, key=lambda branch: branch.score)
    weakest = min(branches, key=lambda branch: branch.score)
    weak_issue = weakest.contradictions[0] if weakest.contradictions else "weak support"
    return f"Branch {best.branch_id} is strongest; weakest contradiction is branch {weakest.branch_id}: {weak_issue}."


def select_final(branches: list[BranchState], selector_config: dict) -> SelectionResult:
    answers = [branch.candidate_answer for branch in branches if branch.candidate_answer is not None]
    counts = Counter(answers)
    scoreboard: list[tuple[int, float]] = []

    for branch in branches:
        score = 0.0
        if branch.candidate_answer is not None:
            score += counts[branch.candidate_answer] * float(selector_config.get("consensus_weight", 2.5))
        if branch.exec_result and branch.exec_result.success:
            score += float(selector_config.get("code_success_weight", 1.75))
        if branch.contradictions:
            score -= len(branch.contradictions) * float(selector_config.get("contradiction_penalty", 1.25))
        if branch.critique_summary:
            score += float(selector_config.get("critique_weight", 1.25))
        if branch.proven_facts:
            score += float(selector_config.get("reasoning_signal_weight", 0.5))
        branch.score = score
        scoreboard.append((branch.branch_id, score))

    selected = max(branches, key=lambda branch: branch.score)
    mode = "selector"
    if selected.candidate_answer is None and counts:
        majority_answer = counts.most_common(1)[0][0]
        selected = next(branch for branch in branches if branch.candidate_answer == majority_answer)
        mode = "majority-fallback"

    critique = _build_critique(branches)
    return SelectionResult(
        selected_branch=selected,
        mode=mode,
        selected_answer=selected.candidate_answer,
        critique=critique,
        scoreboard=scoreboard,
    )

