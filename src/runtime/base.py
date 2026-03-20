from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import ModelTurn, SamplingPlan, SolverConfig


class RuntimeBackend(ABC):
    @abstractmethod
    def chat_batch(
        self,
        messages: list[dict[str, str]],
        sampling_plan: SamplingPlan,
        tool_schema: list[dict] | None = None,
    ) -> list[ModelTurn]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class RuntimeErrorWithHint(RuntimeError):
    """Raised when a runtime backend cannot serve requests with a helpful next step."""

