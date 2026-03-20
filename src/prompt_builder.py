from __future__ import annotations

from pathlib import Path

from src.models import ConfigBundle, MemoryState


def render_memory(memory: MemoryState | None) -> str:
    if memory is None:
        return "No prior branch memory."
    lines = [
        f"Subject: {memory.subject}",
        f"Target quantity: {memory.target_quantity}",
        f"Key facts: {', '.join(memory.key_facts) if memory.key_facts else 'none'}",
        f"Failed attempts: {', '.join(memory.failed_attempt_tags) if memory.failed_attempt_tags else 'none'}",
        f"Code observations: {', '.join(memory.code_observations) if memory.code_observations else 'none'}",
        f"Candidate answers: {', '.join(str(x) for x in memory.candidate_answers) if memory.candidate_answers else 'none'}",
    ]
    return "\n".join(lines)


def build_prompt(problem_text: str, prompt_family: str, memory: MemoryState | None, config: ConfigBundle) -> str:
    root = Path(config.project_root)
    template_path = root / "prompts" / f"{prompt_family}.txt"
    template = template_path.read_text(encoding="utf-8")
    return template.format(problem=problem_text.strip(), memory=render_memory(memory))

