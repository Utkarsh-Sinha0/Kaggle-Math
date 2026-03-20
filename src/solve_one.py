from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from src.answer_extract import extract_exec_answer, extract_final_answer
from src.config import config_hash
from src.memory import build_memory_state
from src.models import ConfigBundle, ExecResult, ExperimentRecord, FinalAnswer, ProblemInput
from src.prompt_builder import build_prompt
from src.python_exec import execute_python
from src.router import route_problem
from src.sampler import generate_candidates
from src.selector import select_candidate


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


def _write_experiment_record(config: ConfigBundle, record: ExperimentRecord) -> None:
    log_path = Path(config.project_root) / config.logging.get("experiment_log_path", "logs/experiment_log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record)) + "\n")


def solve_one(problem_text: str, config_bundle: ConfigBundle, data_slice_id: str = "unspecified") -> FinalAnswer:
    start = time.perf_counter()
    problem = ProblemInput(problem_text=problem_text, data_slice_id=data_slice_id)
    route = route_problem(problem.problem_text, config_bundle)
    prompt = build_prompt(problem.problem_text, route.prompt_family, None, config_bundle)
    candidates = generate_candidates(
        problem=problem.problem_text,
        prompt_family=route.prompt_family,
        sample_budget=route.sample_budget,
        memory_state=None,
        runtime_config=config_bundle,
        prompt=prompt,
    )

    exec_results: list[ExecResult] = []
    memory_states = []
    timeout_count = 0
    invalid_count = 0
    for candidate in candidates:
        exec_result = execute_python(candidate.code, int(config_bundle.runtime.get("python_timeout_seconds", 2)))
        exec_result.extracted_answer = extract_exec_answer(exec_result)
        if exec_result.timed_out:
            timeout_count += 1
        if candidate.extracted_answer is None and exec_result.extracted_answer is None:
            invalid_count += 1
        exec_results.append(exec_result)
        memory_states.append(build_memory_state(route.subject, candidate, exec_result))

    selection = select_candidate(
        problem.problem_text,
        candidates,
        memory_states,
        exec_results,
        config_bundle.runtime.get("selector", {}),
    )
    final_answer = extract_final_answer(
        problem.problem_text,
        selection.selected_answer,
        candidates,
        exec_results,
        config_bundle.runtime,
        selection.mode,
    )

    runtime_seconds = time.perf_counter() - start
    record = ExperimentRecord(
        config_hash=config_hash(config_bundle),
        commit_hash=_git_commit_hash(config_bundle.project_root),
        data_slice_id=problem.data_slice_id,
        runtime_seconds=runtime_seconds,
        pass_at_1=1.0 if candidates and candidates[0].extracted_answer == final_answer.answer_int else 0.0,
        majority_at_n=1.0 if final_answer.provenance.get("majority_answer") == final_answer.answer_int else 0.0,
        selector_at_n=1.0 if selection.selected_answer == final_answer.answer_int else 0.0,
        invalid_answer_rate=invalid_count / max(len(candidates), 1),
        timeout_rate=timeout_count / max(len(candidates), 1),
        crash_rate=0.0,
        notes=config_bundle.runtime.get("notes", ""),
    )
    _write_experiment_record(config_bundle, record)
    return final_answer
