from src.python_exec import execute_python, extract_first_code_block


def test_extract_first_code_block() -> None:
    text = "```python\nprint(42)\n```"
    assert extract_first_code_block(text) == "print(42)"


def test_execute_python_success() -> None:
    result = execute_python("print(42)", timeout_s=1)
    assert result.success is True
    assert result.stdout == "42"


def test_execute_python_timeout() -> None:
    result = execute_python("while True:\n    pass", timeout_s=1)
    assert result.success is False
    assert result.timed_out is True

