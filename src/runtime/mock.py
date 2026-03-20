from __future__ import annotations

import hashlib

from src.models import ModelTurn, SamplingPlan
from src.runtime.base import RuntimeBackend


class MockRuntime(RuntimeBackend):
    def chat_batch(
        self,
        messages: list[dict[str, str]],
        sampling_plan: SamplingPlan,
        tool_schema: list[dict] | None = None,
    ) -> list[ModelTurn]:
        del tool_schema
        prompt = "\n".join(message.get("content", "") for message in messages)
        lowered = prompt.lower()
        seed = int(hashlib.sha256(lowered.encode("utf-8")).hexdigest()[:8], 16) % 100
        outputs: list[ModelTurn] = []
        for idx in range(sampling_plan.sample_count):
            answer = (seed + idx) % 100
            if "2 + 3 + 5" in lowered or "2+3+5" in lowered:
                answer = 10
            elif "123 + 456" in lowered:
                answer = 579
            if idx % 2 == 0:
                content = (
                    f"<think>\nI should reason carefully and look for an integer answer.\n"
                    f"The candidate answer is {answer}.\n</think>\n\n"
                    f"Final answer: {answer}"
                )
            else:
                content = (
                    "<think>\nA short brute force check will help.\n</think>\n\n"
                    "```python\n"
                    f"print({answer})\n"
                    "```\n"
                    f"Final answer: {answer}"
                )
            outputs.append(
                ModelTurn(
                    content=content,
                    finish_reason="stop",
                    prompt_mode=sampling_plan.prompt_mode,
                    reasoning_enabled=sampling_plan.enable_thinking,
                    reasoning_budget=sampling_plan.reasoning_budget,
                    raw_response={"mock": True, "index": idx},
                    metadata={"temperature": sampling_plan.temperature, "top_p": sampling_plan.top_p},
                )
            )
        return outputs

