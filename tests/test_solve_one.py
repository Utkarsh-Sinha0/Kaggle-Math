from pathlib import Path

from src.config import load_config_bundle
from src.solve_one import solve_one


def test_solve_one_returns_valid_integer_and_logs(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config_bundle(project_root / "configs")
    config.logging["experiment_log_path"] = str(tmp_path / "experiment_log.jsonl")
    result = solve_one("What is 2 + 3 + 5?", config, data_slice_id="eval_small")
    assert result.valid is True
    assert result.answer_int == 10
    assert (tmp_path / "experiment_log.jsonl").exists()
