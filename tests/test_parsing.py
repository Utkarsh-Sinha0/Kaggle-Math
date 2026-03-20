from pathlib import Path

from src.config import load_config_bundle
from src.solver.parsing import deterministic_fallback, extract_answer_from_text, extract_ints


def test_extract_ints_prefers_boxed_answer() -> None:
    text = "Try 123 first and then \\boxed{456}."
    assert extract_ints(text) == [456, 123]


def test_extract_answer_from_text() -> None:
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    assert extract_answer_from_text("Reasoning...\nFinal answer: 579", config) == 579


def test_deterministic_fallback_is_bounded() -> None:
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    value = deterministic_fallback("example problem", config)
    assert 0 <= value <= 99999

