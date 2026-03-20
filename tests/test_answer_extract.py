from src.answer_extract import deterministic_fallback, extract_ints, majority_answer


def test_extract_ints_prefers_valid_integers() -> None:
    text = "We compute 123 and then \\boxed{456} as the final answer."
    assert extract_ints(text) == [456, 123]


def test_majority_answer() -> None:
    assert majority_answer([1, None, 2, 2, 3]) == 2


def test_deterministic_fallback_is_bounded() -> None:
    value = deterministic_fallback("example problem")
    assert 0 <= value <= 99999

