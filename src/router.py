from __future__ import annotations

import re
from dataclasses import dataclass

from src.models import ConfigBundle


@dataclass(slots=True)
class RouteDecision:
    subject: str
    prompt_family: str
    sample_budget: int


def route_problem(problem_text: str, config: ConfigBundle) -> RouteDecision:
    lowered = problem_text.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    scores: dict[str, int] = {}
    subjects = config.router.get("subjects", {})

    for subject, rule in subjects.items():
        keywords = rule.get("keywords", [])
        score = sum(1 for keyword in keywords if keyword.lower() in cleaned)
        scores[subject] = score

    best_subject = max(scores, key=scores.get, default="general")
    if scores.get(best_subject, 0) == 0:
        return RouteDecision(
            subject="general",
            prompt_family=config.router.get("default_prompt_family", "reasoning_first"),
            sample_budget=int(config.router.get("default_sample_budget", 3)),
        )

    rule = subjects[best_subject]
    return RouteDecision(
        subject=best_subject,
        prompt_family=rule.get("prompt_family", config.router.get("default_prompt_family", "reasoning_first")),
        sample_budget=int(rule.get("sample_budget", config.router.get("default_sample_budget", 3))),
    )

