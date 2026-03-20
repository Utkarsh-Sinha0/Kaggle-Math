from __future__ import annotations

import json
import time
from typing import Any

from src.models import ModelTurn, SamplingPlan, SolverConfig
from src.runtime.base import RuntimeBackend, RuntimeErrorWithHint
from src.runtime.launcher import resolve_model_source


class VLLMRuntime(RuntimeBackend):
    def __init__(self, config: SolverConfig) -> None:
        self.config = config
        self._client = None
        self._tokenizer = None

    def _client_instance(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeErrorWithHint(
                    "Missing OpenAI client dependency. Install the runtime extras with `pip install -e .[runtime]`."
                ) from exc
            self._client = OpenAI(
                base_url=self.config.runtime.get("api_base_url", "http://127.0.0.1:8000/v1"),
                api_key=self.config.runtime.get("api_key", "EMPTY"),
            )
        return self._client

    def _tokenizer_instance(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise RuntimeErrorWithHint(
                    "Missing transformers tokenizer dependency. Install the runtime extras with `pip install -e .[runtime]`."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(
                resolve_model_source(self.config),
                trust_remote_code=bool(self.config.model.get("trust_remote_code", True)),
                local_files_only=bool(self.config.model.get("local_files_only", False)),
            )
        return self._tokenizer

    def chat_batch(
        self,
        messages: list[dict[str, str]],
        sampling_plan: SamplingPlan,
        tool_schema: list[dict] | None = None,
    ) -> list[ModelTurn]:
        if sampling_plan.reasoning_budget:
            return self._chat_batch_with_budget(messages, sampling_plan, tool_schema)
        return self._chat_batch_plain(messages, sampling_plan, tool_schema)

    def _chat_batch_plain(
        self,
        messages: list[dict[str, str]],
        sampling_plan: SamplingPlan,
        tool_schema: list[dict] | None,
    ) -> list[ModelTurn]:
        client = self._client_instance()
        payload: dict[str, Any] = {
            "model": self.config.model.get("served_model_name", "aimo3-nano"),
            "messages": messages,
            "n": sampling_plan.sample_count,
            "max_tokens": sampling_plan.max_tokens,
            "temperature": sampling_plan.temperature,
            "top_p": sampling_plan.top_p,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": sampling_plan.enable_thinking},
            },
        }
        if tool_schema:
            payload["tools"] = tool_schema
            payload["tool_choice"] = "auto"
        response = client.chat.completions.create(**payload)
        turns: list[ModelTurn] = []
        for choice in response.choices:
            message = choice.message
            turns.append(
                ModelTurn(
                    content=message.content or "",
                    finish_reason=choice.finish_reason or "stop",
                    prompt_mode=sampling_plan.prompt_mode,
                    reasoning_enabled=sampling_plan.enable_thinking,
                    reasoning_budget=sampling_plan.reasoning_budget,
                    tool_calls=[tool.model_dump() for tool in (message.tool_calls or [])],
                    raw_response=choice.model_dump(),
                    metadata={"id": response.id, "created": response.created},
                )
            )
        return turns

    def _chat_batch_with_budget(
        self,
        messages: list[dict[str, str]],
        sampling_plan: SamplingPlan,
        tool_schema: list[dict] | None,
    ) -> list[ModelTurn]:
        del tool_schema
        client = self._client_instance()
        tokenizer = self._tokenizer_instance()
        turns: list[ModelTurn] = []

        for _ in range(sampling_plan.sample_count):
            first = client.chat.completions.create(
                model=self.config.model.get("served_model_name", "aimo3-nano"),
                messages=messages,
                max_tokens=sampling_plan.reasoning_budget,
                temperature=sampling_plan.temperature,
                top_p=sampling_plan.top_p,
            )
            reasoning_content = first.choices[0].message.content or ""
            if "</think>" not in reasoning_content:
                reasoning_content = f"{reasoning_content.rstrip()}\n</think>\n\n"

            reasoning_tokens = len(tokenizer.encode(reasoning_content, add_special_tokens=False))
            remaining_tokens = max(1, sampling_plan.max_tokens - reasoning_tokens)
            appended_messages = list(messages) + [{"role": "assistant", "content": reasoning_content}]
            prompt = tokenizer.apply_chat_template(
                appended_messages,
                tokenize=False,
                continue_final_message=True,
            )
            second = client.completions.create(
                model=self.config.model.get("served_model_name", "aimo3-nano"),
                prompt=prompt,
                max_tokens=remaining_tokens,
                temperature=sampling_plan.temperature,
                top_p=sampling_plan.top_p,
            )
            final_text = second.choices[0].text or ""
            content = f"{reasoning_content}{final_text}".strip()
            turns.append(
                ModelTurn(
                    content=content,
                    finish_reason=second.choices[0].finish_reason or "stop",
                    prompt_mode=sampling_plan.prompt_mode,
                    reasoning_enabled=sampling_plan.enable_thinking,
                    reasoning_budget=sampling_plan.reasoning_budget,
                    raw_response={
                        "reasoning": first.model_dump(mode="json"),
                        "completion": second.model_dump(mode="json"),
                    },
                    metadata={"budget_control": True, "timestamp": time.time()},
                )
            )
        return turns

