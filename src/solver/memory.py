from __future__ import annotations

from src.models import BranchState, SolverConfig


def compress_branch(branch_trace: list[BranchState] | BranchState, config: SolverConfig) -> BranchState:
    branch = branch_trace[-1] if isinstance(branch_trace, list) else branch_trace
    memory_cfg = config.solver.get("memory", {})
    max_facts = int(memory_cfg.get("max_facts", 5))
    max_dead_ends = int(memory_cfg.get("max_dead_ends", 4))
    max_hints = int(memory_cfg.get("max_hints", 3))

    proven_facts = list(branch.proven_facts)
    dead_ends = list(branch.dead_ends)
    code_observations = list(branch.code_observations)
    hints = list(branch.next_step_hints)

    if branch.candidate_answer is not None:
        proven_facts.append(f"Current branch proposes integer {branch.candidate_answer}")
    if branch.exec_result and branch.exec_result.extracted_answer is not None:
        proven_facts.append(f"Python execution returned {branch.exec_result.extracted_answer}")
    if branch.exec_result and branch.exec_result.timed_out:
        dead_ends.append("Python execution timed out")
    if branch.exec_result and branch.exec_result.stderr:
        dead_ends.append("Python execution produced an error")
    if branch.code:
        code_observations.append("Branch used a python check")
    if branch.contradictions:
        hints.append("Resolve contradictions before trusting the current answer")
    if branch.candidate_answer is None:
        hints.append("Try a cleaner verification path or exact computation")
    elif branch.exec_result and branch.exec_result.extracted_answer != branch.candidate_answer and branch.exec_result.extracted_answer is not None:
        hints.append("Reconcile the prose answer with the code result")

    branch.proven_facts = proven_facts[:max_facts]
    branch.dead_ends = dead_ends[:max_dead_ends]
    branch.code_observations = code_observations[:max_facts]
    branch.next_step_hints = hints[:max_hints]
    branch.needs_followup = branch.candidate_answer is None or bool(branch.contradictions)
    return branch

