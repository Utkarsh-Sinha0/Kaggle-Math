from __future__ import annotations

import re
from dataclasses import dataclass

from src.models import SolverConfig


@dataclass(slots=True)
class RouteDecision:
    subject: str
    use_tools: bool
    sample_count: int


def route_problem(problem_text: str, config: SolverConfig) -> RouteDecision:
    lowered = re.sub(r"[^a-z0-9\s]", " ", problem_text.lower())
    rules = config.solver.get("subject_rules", {})
    best_subject = "general"
    best_score = -1
    use_tools = False

    for subject, rule in rules.items():
        score = sum(1 for keyword in rule.get("keywords", []) if keyword.lower() in lowered)
        if score > best_score:
            best_score = score
            best_subject = subject
            use_tools = bool(rule.get("use_tools", False))

    tool_keywords = config.solver.get("tool_keywords", [])
    if any(keyword in lowered for keyword in tool_keywords):
        use_tools = True

    sample_count = int(config.solver.get("default_sample_count", 4))
    if use_tools:
        sample_count = min(sample_count + 1, int(config.solver.get("max_sample_count", 8)))

    return RouteDecision(subject=best_subject, use_tools=use_tools, sample_count=sample_count)

