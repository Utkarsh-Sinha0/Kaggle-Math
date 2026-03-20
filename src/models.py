from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProblemInput:
    problem_text: str
    data_slice_id: str = "unspecified"


@dataclass(slots=True)
class Candidate:
    candidate_id: int
    prompt_family: str
    content: str
    raw_output: str
    extracted_answer: int | None = None
    confidence: float = 0.0
    code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecResult:
    success: bool
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False
    extracted_answer: int | None = None


@dataclass(slots=True)
class MemoryState:
    subject: str
    target_quantity: str
    key_facts: list[str]
    failed_attempt_tags: list[str]
    code_observations: list[str]
    candidate_answers: list[int]


@dataclass(slots=True)
class SelectionResult:
    selected_candidate: Candidate
    mode: str
    selected_answer: int | None
    scoreboard: list[tuple[int, float]]


@dataclass(slots=True)
class FinalAnswer:
    answer_int: int
    source_mode: str
    valid: bool
    fallback_used: bool
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentRecord:
    config_hash: str
    commit_hash: str
    data_slice_id: str
    runtime_seconds: float
    pass_at_1: float
    majority_at_n: float
    selector_at_n: float
    invalid_answer_rate: float
    timeout_rate: float
    crash_rate: float
    notes: str = ""


@dataclass(slots=True)
class ConfigBundle:
    model: dict[str, Any]
    router: dict[str, Any]
    runtime: dict[str, Any]
    logging: dict[str, Any]
    project_root: str

