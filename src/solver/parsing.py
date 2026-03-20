from __future__ import annotations

import hashlib
import json
import re

from src.models import BranchState, ModelTurn, SolverConfig


INTEGER_PATTERN = re.compile(r"-?\d+")
BOXED_PATTERN = re.compile(r"\\boxed\{(-?\d+)\}")


def extract_first_code_block(text: str) -> str | None:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() or None
    return None


def extract_ints(text: str, min_value: int = 0, max_value: int = 99999) -> list[int]:
    boxed = [int(match) for match in BOXED_PATTERN.findall(text)]
    ordered = boxed + [int(match) for match in INTEGER_PATTERN.findall(text) if int(match) not in boxed]
    return [value for value in ordered if min_value <= value <= max_value]


def extract_answer_from_text(text: str, config: SolverConfig) -> int | None:
    answer_range = config.solver.get("answer_range", {"min": 0, "max": 99999})
    ints = extract_ints(text, answer_range.get("min", 0), answer_range.get("max", 99999))
    return ints[-1] if ints else None


def extract_code_from_turn(turn: ModelTurn) -> str | None:
    if turn.tool_calls:
        for tool_call in turn.tool_calls:
            try:
                function = tool_call.get("function", {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    parsed = json.loads(arguments)
                else:
                    parsed = arguments or {}
                code = parsed.get("code")
                if code:
                    return str(code)
            except Exception:
                continue
    return extract_first_code_block(turn.content)


def deterministic_fallback(problem_text: str, config: SolverConfig) -> int:
    modulus = int(config.runtime.get("fallback_modulus", 100000))
    digest = hashlib.sha256(problem_text.encode("utf-8")).hexdigest()
    answer_range = config.solver.get("answer_range", {"min": 0, "max": 99999})
    span = answer_range.get("max", 99999) - answer_range.get("min", 0) + 1
    return answer_range.get("min", 0) + (int(digest[:12], 16) % min(modulus, span))


def branch_has_valid_answer(branch: BranchState) -> bool:
    return branch.candidate_answer is not None

