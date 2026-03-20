from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ProblemInput:
    problem_text: str
    data_slice_id: str = "unspecified"


@dataclass(slots=True)
class SolverConfig:
    model: dict[str, Any]
    runtime: dict[str, Any]
    solver: dict[str, Any]
    logging: dict[str, Any]
    research: dict[str, Any]
    project_root: str


@dataclass(slots=True)
class SamplingPlan:
    sample_count: int
    max_tokens: int
    temperature: float
    top_p: float
    enable_thinking: bool
    reasoning_budget: int | None
    prompt_mode: str


@dataclass(slots=True)
class ModelTurn:
    content: str
    finish_reason: str
    prompt_mode: str
    reasoning_enabled: bool
    reasoning_budget: int | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)
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
class BranchState:
    branch_id: int
    problem_text: str
    subject: str
    messages: list[dict[str, str]]
    model_turn: ModelTurn
    code: str | None = None
    exec_result: ExecResult | None = None
    candidate_answer: int | None = None
    proven_facts: list[str] = field(default_factory=list)
    dead_ends: list[str] = field(default_factory=list)
    code_observations: list[str] = field(default_factory=list)
    next_step_hints: list[str] = field(default_factory=list)
    critique_summary: str = ""
    contradictions: list[str] = field(default_factory=list)
    score: float = 0.0
    resample_round: int = 0
    needs_followup: bool = False


@dataclass(slots=True)
class SelectionResult:
    selected_branch: BranchState
    mode: str
    selected_answer: int | None
    critique: str
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
    runtime_backend: str
    sample_count: int
    reasoning_budget: int | None
    runtime_seconds: float
    pass_at_1: float
    majority_at_n: float
    selector_at_n: float
    invalid_answer_rate: float
    timeout_rate: float
    crash_rate: float
    estimated_cost_usd: float
    notes: str = ""


@dataclass(slots=True)
class EvalSummary:
    eval_path: str
    total_examples: int
    answered_examples: int
    pass_at_1: float
    majority_at_n: float
    selector_at_n: float
    invalid_answer_rate: float
    timeout_rate: float
    average_runtime_seconds: float
    estimated_cost_usd: float


@dataclass(slots=True)
class BundleManifest:
    output_dir: str
    bundle_mount: str
    model_mount: str
    notebook_path: str
    included_paths: list[str]
    runtime_backend: str


@dataclass(slots=True)
class BudgetLedger:
    budget_hours_limit: float
    consumed_hours: float
    estimated_cost_usd: float
    blocked: bool
    entries: list[dict[str, Any]] = field(default_factory=list)
