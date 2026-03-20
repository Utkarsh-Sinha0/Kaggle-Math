from __future__ import annotations

from src.models import Candidate, ExecResult, MemoryState


def build_memory_state(subject: str, candidate: Candidate, exec_result: ExecResult | None) -> MemoryState:
    facts = []
    if candidate.extracted_answer is not None:
        facts.append(f"candidate proposes {candidate.extracted_answer}")
    if exec_result and exec_result.extracted_answer is not None:
        facts.append(f"python execution produced {exec_result.extracted_answer}")

    failed_tags = []
    if exec_result and exec_result.timed_out:
        failed_tags.append("timeout")
    if exec_result and exec_result.stderr:
        failed_tags.append("runtime-error")
    if candidate.extracted_answer is None:
        failed_tags.append("no-answer")

    code_observations = []
    if candidate.code:
        code_observations.append("candidate included python")
    if exec_result and exec_result.success:
        code_observations.append("python executed cleanly")

    answers = [x for x in [candidate.extracted_answer, exec_result.extracted_answer if exec_result else None] if x is not None]

    return MemoryState(
        subject=subject,
        target_quantity="final integer answer",
        key_facts=facts[:5],
        failed_attempt_tags=failed_tags[:5],
        code_observations=code_observations[:5],
        candidate_answers=answers[:5],
    )

