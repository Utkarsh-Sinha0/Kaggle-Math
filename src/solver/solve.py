from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from src.config import config_hash
from src.models import BranchState, ExecResult, ExperimentRecord, FinalAnswer, ProblemInput, SamplingPlan, SolverConfig
from src.python_exec import execute_python
from src.solver.memory import compress_branch
from src.solver.parsing import (
    branch_has_valid_answer,
    deterministic_fallback,
    extract_answer_from_text,
    extract_code_from_turn,
)
from src.solver.prompts import build_followup_messages, build_initial_messages
from src.solver.routing import route_problem
from src.solver.selector import select_final


def _git_commit_hash(project_root: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except OSError:
        pass
    return "uncommitted"


def _estimate_cost(config: SolverConfig, runtime_seconds: float) -> float:
    hourly_rate = float(config.runtime.get("instance_hourly_cost_usd", 0.0))
    return hourly_rate * (runtime_seconds / 3600.0)


def _write_experiment_record(config: SolverConfig, record: ExperimentRecord) -> None:
    log_path = Path(config.project_root) / config.logging.get("experiment_log_path", "logs/experiment_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def _execute_branch(branch: BranchState, config: SolverConfig) -> BranchState:
    if branch.code:
        branch.exec_result = execute_python(branch.code, int(config.runtime.get("python_timeout_seconds", 3)))
        if branch.exec_result.stdout.strip():
            exec_answer = extract_answer_from_text(branch.exec_result.stdout, config)
            branch.exec_result.extracted_answer = exec_answer
            if branch.candidate_answer is None:
                branch.candidate_answer = exec_answer
        if branch.exec_result.extracted_answer is not None and branch.candidate_answer is not None:
            if branch.exec_result.extracted_answer != branch.candidate_answer:
                branch.contradictions.append("Code result disagrees with prose answer")
    else:
        branch.exec_result = ExecResult(success=False, stdout="", stderr="no code provided", return_code=1)
    return compress_branch(branch, config)


def _make_branch(
    branch_id: int,
    problem_text: str,
    subject: str,
    messages: list[dict[str, str]],
    turn,
    resample_round: int,
    config: SolverConfig,
) -> BranchState:
    candidate_answer = extract_answer_from_text(turn.content, config)
    code = extract_code_from_turn(turn)
    branch = BranchState(
        branch_id=branch_id,
        problem_text=problem_text,
        subject=subject,
        messages=messages,
        model_turn=turn,
        code=code,
        candidate_answer=candidate_answer,
        resample_round=resample_round,
    )
    return _execute_branch(branch, config)


def _tool_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "python_exec",
                "description": "Run a small Python snippet to verify a math candidate.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]


def solve_one(problem_text: str, runtime, config: SolverConfig, data_slice_id: str = "unspecified") -> FinalAnswer:
    start = time.perf_counter()
    problem = ProblemInput(problem_text=problem_text, data_slice_id=data_slice_id)
    route = route_problem(problem.problem_text, config)
    messages = build_initial_messages(problem.problem_text, route, config)
    initial_plan = SamplingPlan(
        sample_count=route.sample_count,
        max_tokens=int(config.runtime.get("default_max_tokens", 4096)),
        temperature=float(config.runtime.get("default_temperature", 1.0)),
        top_p=float(config.runtime.get("default_top_p", 1.0)),
        enable_thinking=bool(config.runtime.get("enable_thinking", True)),
        reasoning_budget=int(config.runtime.get("default_reasoning_budget", 2048)),
        prompt_mode="initial",
    )
    turns = runtime.chat_batch(messages, initial_plan, _tool_schema() if route.use_tools else None)
    branches = [
        _make_branch(idx, problem.problem_text, route.subject, messages, turn, 0, config)
        for idx, turn in enumerate(turns)
    ]

    followup_rounds = int(config.solver.get("max_resample_rounds", 1))
    next_branch_id = len(branches)
    for _ in range(followup_rounds):
        needing_followup = [branch for branch in branches if branch.needs_followup][: int(config.solver.get("follow_up_sample_count", 2))]
        if not needing_followup:
            break
        for branch in needing_followup:
            followup_messages = build_followup_messages(problem.problem_text, branch, config)
            followup_plan = SamplingPlan(
                sample_count=1,
                max_tokens=int(config.runtime.get("default_max_tokens", 4096)),
                temperature=float(config.runtime.get("tool_temperature", 0.6) if route.use_tools else config.runtime.get("default_temperature", 1.0)),
                top_p=float(config.runtime.get("tool_top_p", 0.95) if route.use_tools else config.runtime.get("default_top_p", 1.0)),
                enable_thinking=bool(config.runtime.get("enable_thinking", True)),
                reasoning_budget=int(config.runtime.get("default_reasoning_budget", 2048)),
                prompt_mode="followup",
            )
            followup_turn = runtime.chat_batch(followup_messages, followup_plan, _tool_schema() if route.use_tools else None)[0]
            new_branch = _make_branch(next_branch_id, problem.problem_text, route.subject, followup_messages, followup_turn, branch.resample_round + 1, config)
            next_branch_id += 1
            branches.append(new_branch)

    top_branches = sorted(branches, key=lambda branch: (branch.candidate_answer is not None, branch.exec_result.success if branch.exec_result else False), reverse=True)[
        : int(config.solver.get("max_branches_for_critique", 3))
    ]
    for branch in top_branches:
        if branch.exec_result and branch.exec_result.success:
            branch.critique_summary = "Supported by successful python verification."
        elif branch.candidate_answer is not None:
            branch.critique_summary = "Has a direct candidate answer but weaker verification."
        else:
            branch.critique_summary = "Needs stronger evidence."

    selection = select_final(branches, config.solver.get("selector", {}))
    selected_answer = selection.selected_answer
    fallback_used = False
    if selected_answer is None:
        selected_answer = deterministic_fallback(problem.problem_text, config)
        fallback_used = True

    runtime_seconds = time.perf_counter() - start
    final_answer = FinalAnswer(
        answer_int=selected_answer,
        source_mode=selection.mode,
        valid=True,
        fallback_used=fallback_used,
        provenance={
            "selected_branch_id": selection.selected_branch.branch_id,
            "selected_answer": selection.selected_answer,
            "critique": selection.critique,
            "scoreboard": selection.scoreboard,
            "branch_answers": [branch.candidate_answer for branch in branches],
        },
    )

    initial_answers = [branch.candidate_answer for branch in branches[: initial_plan.sample_count]]
    majority_initial = None
    filtered = [answer for answer in initial_answers if answer is not None]
    if filtered:
        majority_initial = max(set(filtered), key=filtered.count)

    invalid_count = sum(1 for branch in branches if not branch_has_valid_answer(branch))
    timeout_count = sum(1 for branch in branches if branch.exec_result and branch.exec_result.timed_out)
    record = ExperimentRecord(
        config_hash=config_hash(config),
        commit_hash=_git_commit_hash(config.project_root),
        data_slice_id=problem.data_slice_id,
        runtime_backend=str(config.runtime.get("backend", "mock")),
        sample_count=len(branches),
        reasoning_budget=initial_plan.reasoning_budget,
        runtime_seconds=runtime_seconds,
        pass_at_1=1.0 if initial_answers and initial_answers[0] == final_answer.answer_int else 0.0,
        majority_at_n=1.0 if majority_initial == final_answer.answer_int else 0.0,
        selector_at_n=1.0 if selection.selected_answer == final_answer.answer_int else 0.0,
        invalid_answer_rate=invalid_count / max(len(branches), 1),
        timeout_rate=timeout_count / max(len(branches), 1),
        crash_rate=0.0,
        estimated_cost_usd=_estimate_cost(config, runtime_seconds),
        notes=str(config.runtime.get("notes", "")),
    )
    _write_experiment_record(config, record)
    final_answer.provenance.update(
        {
            "runtime_seconds": runtime_seconds,
            "majority_answer": majority_initial,
            "selected_answer": selection.selected_answer,
            "timed_out": timeout_count > 0,
            "estimated_cost_usd": record.estimated_cost_usd,
        }
    )
    return final_answer
