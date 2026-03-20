from pathlib import Path

from src.config import load_config_bundle
from src.runtime.factory import create_runtime
from src.solver.solve import solve_one


def test_solve_one_returns_valid_integer_and_logs(tmp_path: Path) -> None:
    config = load_config_bundle(Path(__file__).resolve().parents[1] / "configs")
    config.logging["experiment_log_path"] = str(tmp_path / "experiment_log.jsonl")
    runtime = create_runtime(config)
    result = solve_one("What is 2 + 3 + 5?", runtime, config, data_slice_id="eval_small")
    assert result.valid is True
    assert result.answer_int == 10
    assert (tmp_path / "experiment_log.jsonl").exists()

