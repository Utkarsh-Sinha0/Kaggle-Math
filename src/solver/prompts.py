from __future__ import annotations

from pathlib import Path

from src.models import BranchState, SolverConfig
from src.solver.routing import RouteDecision


def _prompt_text(config: SolverConfig, filename: str) -> str:
    return (Path(config.project_root) / "prompts" / filename).read_text(encoding="utf-8").strip()


def build_initial_messages(problem_text: str, route: RouteDecision, config: SolverConfig) -> list[dict[str, str]]:
    system_prompt = _prompt_text(config, "system_solver.txt")
    user_prompt = (
        f"Problem subject guess: {route.subject}\n"
        f"Use Python tools: {'yes' if route.use_tools else 'only if clearly needed'}\n\n"
        f"Problem:\n{problem_text.strip()}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_followup_messages(problem_text: str, branch: BranchState, config: SolverConfig) -> list[dict[str, str]]:
    followup_prompt = _prompt_text(config, "followup_solver.txt")
    memory_blob = "\n".join(
        [
            f"Proven facts: {', '.join(branch.proven_facts) if branch.proven_facts else 'none'}",
            f"Dead ends: {', '.join(branch.dead_ends) if branch.dead_ends else 'none'}",
            f"Code observations: {', '.join(branch.code_observations) if branch.code_observations else 'none'}",
            f"Next hints: {', '.join(branch.next_step_hints) if branch.next_step_hints else 'none'}",
            f"Current answer: {branch.candidate_answer if branch.candidate_answer is not None else 'unknown'}",
        ]
    )
    user_prompt = f"Problem:\n{problem_text.strip()}\n\nCompressed branch memory:\n{memory_blob}"
    return [
        {"role": "system", "content": followup_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_critique_prompt(problem_text: str, branches: list[BranchState], config: SolverConfig) -> list[dict[str, str]]:
    critique_prompt = _prompt_text(config, "critique.txt")
    branch_lines = []
    for branch in branches:
        branch_lines.append(
            f"Branch {branch.branch_id}: answer={branch.candidate_answer}, "
            f"exec_success={branch.exec_result.success if branch.exec_result else False}, "
            f"contradictions={branch.contradictions or ['none']}"
        )
    return [
        {"role": "system", "content": critique_prompt},
        {"role": "user", "content": f"Problem:\n{problem_text}\n\nCandidates:\n" + "\n".join(branch_lines)},
    ]

