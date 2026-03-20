from __future__ import annotations

import subprocess

from src.models import SolverConfig
from src.runtime.base import RuntimeBackend
from src.runtime.launcher import launch_vllm_server, wait_for_vllm_server
from src.runtime.mock import MockRuntime
from src.runtime.vllm import VLLMRuntime


class ManagedRuntime(RuntimeBackend):
    def __init__(self, runtime: RuntimeBackend, process: subprocess.Popen | None = None) -> None:
        self.runtime = runtime
        self.process = process

    def chat_batch(self, messages, sampling_plan, tool_schema=None):
        return self.runtime.chat_batch(messages, sampling_plan, tool_schema=tool_schema)

    def close(self) -> None:
        try:
            self.runtime.close()
        finally:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)


def create_runtime(config: SolverConfig) -> RuntimeBackend:
    if config.runtime.get("backend") == "vllm":
        process = None
        if bool(config.runtime.get("launch_server", False)):
            process = launch_vllm_server(config)
            wait_for_vllm_server(str(config.runtime.get("api_base_url", "http://127.0.0.1:8000/v1")))
        return ManagedRuntime(VLLMRuntime(config), process=process)
    return MockRuntime()
