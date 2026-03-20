from pathlib import Path

from src.config import load_config_bundle
from src.models import SamplingPlan
from src.runtime.factory import create_runtime


def test_mock_runtime_returns_requested_sample_count() -> None:
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    runtime = create_runtime(config)
    turns = runtime.chat_batch(
        [{"role": "user", "content": "What is 2 + 3 + 5?"}],
        SamplingPlan(
            sample_count=4,
            max_tokens=128,
            temperature=1.0,
            top_p=1.0,
            enable_thinking=True,
            reasoning_budget=64,
            prompt_mode="initial",
        ),
    )
    assert len(turns) == 4
    assert all(turn.reasoning_enabled for turn in turns)

