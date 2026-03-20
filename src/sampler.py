from __future__ import annotations

from src.models import SamplingPlan, SolverConfig
from src.runtime.factory import create_runtime


def generate_candidates(
    messages: list[dict[str, str]],
    sampling_plan: SamplingPlan,
    config: SolverConfig,
    tool_schema: list[dict] | None = None,
):
    runtime = create_runtime(config)
    try:
        return runtime.chat_batch(messages, sampling_plan, tool_schema=tool_schema)
    finally:
        runtime.close()
