from pathlib import Path

from src.config import load_config_bundle
from src.router import route_problem


def test_router_geometry() -> None:
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    decision = route_problem("A triangle has area 6 and perimeter 12.", config)
    assert decision.subject == "geometry"
    assert decision.prompt_family == "reasoning_first"


def test_router_number_theory() -> None:
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    decision = route_problem("Find the gcd of two integers modulo 7.", config)
    assert decision.subject == "number_theory"
    assert decision.prompt_family == "code_first"

