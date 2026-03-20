from src.models import Candidate, ExecResult, MemoryState
from src.selector import select_candidate


def test_selector_prefers_exec_success_and_agreement() -> None:
    candidates = [
        Candidate(candidate_id=0, prompt_family="reasoning_first", content="Answer: 11", raw_output="Answer: 11", extracted_answer=11, confidence=0.7),
        Candidate(candidate_id=1, prompt_family="code_first", content="Answer: 11", raw_output="Answer: 11", extracted_answer=11, confidence=0.6, code="print(11)"),
    ]
    exec_results = [
        ExecResult(success=False, stdout="", stderr="runtime error", return_code=1),
        ExecResult(success=True, stdout="11", stderr="", return_code=0, extracted_answer=11),
    ]
    memory_states = [
        MemoryState("algebra", "final integer answer", ["candidate proposes 11"], ["runtime-error"], [], [11]),
        MemoryState("algebra", "final integer answer", ["candidate proposes 11"], [], ["python executed cleanly"], [11]),
    ]
    selected = select_candidate("problem", candidates, memory_states, exec_results, {"answer_agreement_bonus": 2.0, "exec_success_bonus": 1.5})
    assert selected.selected_candidate.candidate_id == 1

