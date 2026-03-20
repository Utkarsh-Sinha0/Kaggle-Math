from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path

from src.answer_extract import extract_candidate_answer
from src.models import Candidate, ConfigBundle, MemoryState
from src.python_exec import extract_first_code_block


def _resolve_model_source(config: ConfigBundle) -> tuple[str, bool]:
    local_model_path = config.model.get("local_model_path")
    if local_model_path:
        return str(local_model_path), True
    env_model_path = os.getenv("AIMO3_MODEL_PATH")
    if env_model_path:
        return env_model_path, True
    return str(config.model["model_id"]), bool(config.model.get("local_files_only", False))


def _resolve_torch_dtype(dtype_name: str):
    import torch

    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_name, torch.bfloat16)


def _find_model_device(model) -> str:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return str(device)
    device_map = getattr(model, "hf_device_map", None) or {}
    for mapped_device in device_map.values():
        if isinstance(mapped_device, str) and mapped_device not in {"cpu", "disk", "meta"}:
            return mapped_device
    return "cpu"


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
        self._tokenizer = None
        self._model = None
        self._loaded_signature: tuple[str, str, str, bool] | None = None

    def _ensure_loaded(self, config: ConfigBundle) -> None:
        model_source, local_files_only = _resolve_model_source(config)
        dtype_name = str(config.model.get("dtype", "bfloat16"))
        attn_implementation = str(config.model.get("attn_implementation", "sdpa"))
        signature = (model_source, dtype_name, attn_implementation, local_files_only)
        if self._model is not None and self._loaded_signature == signature:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformers runtime dependencies are missing. Install with `pip install -e .[runtime]` on AWS."
            ) from exc

        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=bool(config.model.get("trust_remote_code", False)),
            local_files_only=local_files_only,
            token=token,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = str(config.model.get("padding_side", "left"))

        model_kwargs = {
            "device_map": config.model.get("device_map", "auto"),
            "trust_remote_code": bool(config.model.get("trust_remote_code", False)),
            "local_files_only": local_files_only,
            "token": token,
            "low_cpu_mem_usage": True,
        }
        torch_dtype = _resolve_torch_dtype(dtype_name)
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._loaded_signature = signature

    def generate(self, prompt: str, sample_budget: int, config: ConfigBundle) -> list[str]:
        self._ensure_loaded(config)
        import torch

        assert self._tokenizer is not None
        assert self._model is not None

        prompts = [prompt] * sample_budget
        tokenized = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        model_device = _find_model_device(self._model)
        tokenized = {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in tokenized.items()
        }

        prompt_length = tokenized["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = self._model.generate(
                **tokenized,
                max_new_tokens=int(config.model.get("max_new_tokens", 512)),
                temperature=float(config.model.get("temperature", 0.2)),
                top_p=float(config.model.get("top_p", 0.95)),
                do_sample=bool(config.model.get("do_sample", True)),
                repetition_penalty=float(config.model.get("repetition_penalty", 1.0)),
                use_cache=bool(config.model.get("use_cache", True)),
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated_only = outputs[:, prompt_length:]
        return self._tokenizer.batch_decode(generated_only, skip_special_tokens=True)


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
