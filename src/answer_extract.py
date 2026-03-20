from __future__ import annotations

import hashlib
import re
from collections import Counter

from src.models import Candidate, ExecResult, FinalAnswer


INTEGER_PATTERN = re.compile(r"-?\d+")
BOXED_PATTERN = re.compile(r"\\boxed\{(-?\d+)\}")


def _normalize_answer(value: int | None) -> int | None:
    if value is None:
        return None
    if 0 <= value <= 99999:
        return value
    return None


def extract_ints(text: str) -> list[int]:
    boxed = [int(match) for match in BOXED_PATTERN.findall(text)]
    all_numbers = [int(match) for match in INTEGER_PATTERN.findall(text)]
    merged = boxed + [x for x in all_numbers if x not in boxed]
    return [_normalize_answer(x) for x in merged if _normalize_answer(x) is not None]


def extract_candidate_answer(candidate: Candidate) -> int | None:
    ints = extract_ints(candidate.content)
    return ints[-1] if ints else None


def extract_exec_answer(exec_result: ExecResult) -> int | None:
    ints = extract_ints(f"{exec_result.stdout}\n{exec_result.stderr}")
    return ints[-1] if ints else None


def majority_answer(values: list[int | None]) -> int | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    counter = Counter(filtered)
    return counter.most_common(1)[0][0]


def deterministic_fallback(problem_text: str, modulus: int = 100000) -> int:
    digest = hashlib.sha256(problem_text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def extract_final_answer(
    problem_text: str,
    selected_answer: int | None,
    candidates: list[Candidate],
    exec_results: list[ExecResult],
    extraction_config: dict,
    source_mode: str,
) -> FinalAnswer:
    majority = majority_answer([candidate.extracted_answer for candidate in candidates] + [result.extracted_answer for result in exec_results])
    answer = _normalize_answer(selected_answer) or _normalize_answer(majority)
    fallback_used = False
    if answer is None:
        answer = deterministic_fallback(problem_text, int(extraction_config.get("fallback_modulus", 100000)))
        fallback_used = True

    return FinalAnswer(
        answer_int=answer,
        source_mode=source_mode,
        valid=True,
        fallback_used=fallback_used,
        provenance={
            "candidate_answers": [candidate.extracted_answer for candidate in candidates],
            "exec_answers": [result.extracted_answer for result in exec_results],
            "selected_answer": selected_answer,
            "majority_answer": majority,
        },
    )

