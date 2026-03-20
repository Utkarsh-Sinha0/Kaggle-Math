from src.models import BranchState, ExecResult, ModelTurn
from src.solver.selector import select_final


def test_selector_prefers_code_verified_branch() -> None:
    branches = [
        BranchState(
            branch_id=0,
            problem_text="problem",
            subject="algebra",
            messages=[],
            model_turn=ModelTurn(content="Final answer: 11", finish_reason="stop", prompt_mode="initial", reasoning_enabled=True, reasoning_budget=64),
            candidate_answer=11,
            exec_result=ExecResult(success=False, stdout="", stderr="runtime error", return_code=1),
        ),
        BranchState(
            branch_id=1,
            problem_text="problem",
            subject="algebra",
            messages=[],
            model_turn=ModelTurn(content="```python\nprint(11)\n```\nFinal answer: 11", finish_reason="stop", prompt_mode="initial", reasoning_enabled=True, reasoning_budget=64),
            candidate_answer=11,
            exec_result=ExecResult(success=True, stdout="11", stderr="", return_code=0, extracted_answer=11),
            critique_summary="Supported by successful python verification.",
        ),
    ]
    result = select_final(branches, {"consensus_weight": 2.5, "code_success_weight": 1.75, "critique_weight": 1.25})
    assert result.selected_branch.branch_id == 1

