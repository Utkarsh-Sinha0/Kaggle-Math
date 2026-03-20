from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from src.answer_extract import extract_candidate_answer
from src.models import Candidate, ConfigBundle, MemoryState
from src.python_exec import extract_first_code_block


class GeneratorBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, sample_budget: int, config: ConfigBundle) -> list[str]:
        raise NotImplementedError


class MockGeneratorBackend(GeneratorBackend):
    def generate(self, prompt: str, sample_budget: int, config: ConfigBundle) -> list[str]:
        del config
        problem = prompt.lower()
        outputs = []
        base_number = int(hashlib.sha256(problem.encode("utf-8")).hexdigest()[:6], 16) % 97
        for idx in range(sample_budget):
            guess = (base_number + idx) % 100
            if any(token in problem for token in ["2 + 3 + 5", "2+3+5"]):
                guess = 10
            if idx == 0:
                outputs.append(f"Reasoning suggests the answer is {guess}.")
            else:
                outputs.append(
                    "Let's verify by Python.\n"
                    "```python\n"
                    f"print({guess})\n"
                    "```\n"
                    f"Final answer: {guess}"
                )
        return outputs


class TransformersGeneratorBackend(GeneratorBackend):
    def __init__(self) -> None:
        self._pipeline = None
        self._loaded_model_id: str | None = None

    def _ensure_loaded(self, config: ConfigBundle) -> None:
        model_id = config.model["model_id"]
        if self._pipeline is not None and self._loaded_model_id == model_id:
            return

        from transformers import pipeline

        self._pipeline = pipeline(
            "text-generation",
            model=model_id,
            device_map=config.model.get("device_map", "auto"),
            model_kwargs={"torch_dtype": config.model.get("dtype", "auto")},
        )
        self._loaded_model_id = model_id

    def generate(self, prompt: str, sample_budget: int, config: ConfigBundle) -> list[str]:
        self._ensure_loaded(config)
        outputs = self._pipeline(
            [prompt] * sample_budget,
            max_new_tokens=int(config.model.get("max_new_tokens", 512)),
            temperature=float(config.model.get("temperature", 0.2)),
            top_p=float(config.model.get("top_p", 0.95)),
            do_sample=bool(config.model.get("do_sample", True)),
            return_full_text=False,
        )
        return [item[0]["generated_text"] for item in outputs]


def get_backend(name: str) -> GeneratorBackend:
    if name == "transformers":
        return TransformersGeneratorBackend()
    return MockGeneratorBackend()


def generate_candidates(
    problem: str,
    prompt_family: str,
    sample_budget: int,
    memory_state: MemoryState | None,
    runtime_config: ConfigBundle,
    prompt: str,
) -> list[Candidate]:
    del problem, memory_state
    backend_name = runtime_config.runtime.get("backend", "mock")
    backend = get_backend(backend_name)
    raw_outputs = backend.generate(prompt, sample_budget, runtime_config)
    candidates: list[Candidate] = []
    for idx, raw in enumerate(raw_outputs):
        candidate = Candidate(
            candidate_id=idx,
            prompt_family=prompt_family,
            content=raw.strip(),
            raw_output=raw,
            confidence=max(0.0, 1.0 - (idx * 0.1)),
            code=extract_first_code_block(raw),
        )
        candidate.extracted_answer = extract_candidate_answer(candidate)
        candidates.append(candidate)
    return candidates

