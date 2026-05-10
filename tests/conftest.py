"""Shared test fixtures: stub Slurm + isolated state dir."""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pytest

from autolease.config import PoolConfig


@dataclass
class StubResponse:
    """Mimics subprocess.CompletedProcess for SlurmConfig.run."""
    args: list
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class StubSlurmConfig:
    """Drop-in replacement for SlurmConfig.run that records calls and
    returns canned responses based on a router function."""

    def __init__(self, router: Optional[Callable[[str], StubResponse]] = None,
                 ssh_host: str = "stub-host", shell: str = "bash"):
        self.ssh_host = ssh_host
        self.shell = shell
        self.ssh_opts = ()
        self.calls: list[str] = []
        self._router = router or (lambda cmd: StubResponse([cmd]))

    def run(self, cmd: str, timeout: int = 30) -> StubResponse:
        self.calls.append(cmd)
        resp = self._router(cmd)
        if resp is None:
            return StubResponse([cmd])
        return resp


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Each test gets a fresh state_dir under a tmp_path."""
    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg_data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg_config"))
    return state


@pytest.fixture
def cfg(isolated_state):
    """A minimal PoolConfig pointed at an isolated state dir."""
    return PoolConfig(
        ssh_host="stub-host",
        shell="bash",
        state_dir=str(isolated_state),
    )
