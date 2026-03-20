from __future__ import annotations

from src.runtime.factory import create_runtime
from src.solver.solve import solve_one as solve_with_runtime


def solve_one(problem_text: str, config_bundle, data_slice_id: str = "unspecified"):
    runtime = create_runtime(config_bundle)
    try:
        return solve_with_runtime(problem_text, runtime, config_bundle, data_slice_id=data_slice_id)
    finally:
        runtime.close()
