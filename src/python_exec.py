from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from src.models import ExecResult


def extract_first_code_block(text: str) -> str | None:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip() or None


def execute_python(code: str | None, timeout_s: int) -> ExecResult:
    if not code:
        return ExecResult(success=False, stdout="", stderr="no code provided", return_code=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "snippet.py"
        script_path.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            return ExecResult(
                success=completed.returncode == 0,
                stdout=completed.stdout.strip(),
                stderr=completed.stderr.strip(),
                return_code=completed.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                success=False,
                stdout=(exc.stdout or "").strip() if exc.stdout else "",
                stderr="execution timed out",
                return_code=124,
                timed_out=True,
            )

