from pathlib import Path

from src.config import load_config_bundle
from src.solver.routing import route_problem


def test_route_problem_geometry() -> None:
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    route = route_problem("A triangle has area 6 and perimeter 12.", config)
    assert route.subject == "geometry"
    assert route.use_tools is False


def test_route_problem_number_theory_prefers_tools() -> None:
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    route = route_problem("Find the gcd of two integers modulo 7.", config)
    assert route.subject == "number_theory"
    assert route.use_tools is True

