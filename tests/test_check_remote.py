"""Tests for _check_remote — the most failure-prone function.
Each test runs against a stub Slurm whose output we control. Would have
caught: bash-vs-fish if/then/elif fragility, false LOST when SSH errors
or returns garbage, mtime defensive fallback for user scripts that
detach background work."""

import pytest

from autolease.queue import Job, JobQueue, _now
from conftest import StubResponse, StubSlurmConfig


@pytest.fixture
def queue_with_stub(cfg, monkeypatch):
    """JobQueue wired to a stub Slurm."""
    q = JobQueue(cfg)
    stub_cfg = StubSlurmConfig(ssh_host=cfg.ssh_host, shell=cfg.shell)
    q.slurm.cfg = stub_cfg
    return q, stub_cfg


def _make_job(jid: int = 1, **kw) -> Job:
    defaults = dict(
        id=jid, project="p", command="c", state="running",
        num_gpus=1, lease_job_id=12345, remote_pid=99999,
        step_name=f"autolease-job-{jid}",
    )
    defaults.update(kw)
    return Job(**defaults)


class TestCheckRemoteSlurmPath:
    def test_running_when_step_in_squeue(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1)
        # Stub: any /bin/sh -c command we run returns __AL_RUN__
        stub._router = lambda cmd: StubResponse([cmd], 0, "__AL_RUN__\n", "")
        assert q._check_remote(job) == ("running", None)

    def test_done_when_exit_code_present(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1)
        stub._router = lambda cmd: StubResponse([cmd], 0, "__AL_DONE__:0\n", "")
        assert q._check_remote(job) == ("done", 0)

    def test_done_with_nonzero_exit(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1)
        stub._router = lambda cmd: StubResponse([cmd], 0, "__AL_DONE__:127\n", "")
        assert q._check_remote(job) == ("done", 127)

    def test_done_with_garbage_exit_code_returns_done_none(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1)
        stub._router = lambda cmd: StubResponse([cmd], 0, "__AL_DONE__:notanint\n", "")
        assert q._check_remote(job) == ("done", None)

    def test_lost_when_step_gone_and_files_quiet(self, queue_with_stub, monkeypatch):
        q, stub = queue_with_stub
        job = _make_job(1)
        # mtime is from 2 hours ago — well past mtime_threshold (60s)
        import time
        old_mtime = int(time.time()) - 7200
        stub._router = lambda cmd: StubResponse([cmd], 0, f"__AL_NOSTEP__:{old_mtime}\n", "")
        assert q._check_remote(job) == ("lost", None)

    def test_running_when_step_gone_but_files_recent(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1)
        # File was just written → user script may have detached background work
        import time
        recent = int(time.time()) - 5
        stub._router = lambda cmd: StubResponse([cmd], 0, f"__AL_NOSTEP__:{recent}\n", "")
        assert q._check_remote(job) == ("running", None)

    def test_unknown_on_ssh_failure(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1)
        # SSH command failed (returncode != 0) → DO NOT change state
        stub._router = lambda cmd: StubResponse([cmd], 255, "", "ssh: connection refused")
        assert q._check_remote(job) == ("unknown", None)

    def test_unknown_on_garbage_output(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1)
        # MOTD or arbitrary noise — no marker found → UNKNOWN, no state change
        stub._router = lambda cmd: StubResponse([cmd], 0, "Welcome to the cluster!\n", "")
        assert q._check_remote(job) == ("unknown", None)

    def test_marker_with_motd_noise_still_recognized(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1)
        # Real-world: SSH banners precede the actual command output
        out = (
            "* Welcome to the cluster\n"
            "* Reminder: report bugs to admin\n"
            "__AL_RUN__\n"
        )
        stub._router = lambda cmd: StubResponse([cmd], 0, out, "")
        assert q._check_remote(job) == ("running", None)


class TestCheckRemoteLegacyPath:
    """Legacy PID-based check (pre-step_name jobs)."""

    def test_running_when_pid_alive(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1, step_name=None)  # legacy
        stub._router = lambda cmd: StubResponse([cmd], 0, "__AL_RUN__\n", "")
        assert q._check_remote(job) == ("running", None)

    def test_done_when_pidgone_with_exit_code(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1, step_name=None)
        stub._router = lambda cmd: StubResponse([cmd], 0, "__AL_DONE__:0\n", "")
        assert q._check_remote(job) == ("done", 0)

    def test_lost_when_pidgone_quiet(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1, step_name=None)
        import time
        old = int(time.time()) - 7200
        stub._router = lambda cmd: StubResponse([cmd], 0, f"__AL_PIDGONE__:{old}\n", "")
        assert q._check_remote(job) == ("lost", None)

    def test_running_when_pidgone_recent_writes(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1, step_name=None)
        import time
        recent = int(time.time()) - 5
        stub._router = lambda cmd: StubResponse([cmd], 0, f"__AL_PIDGONE__:{recent}\n", "")
        assert q._check_remote(job) == ("running", None)

    def test_no_pid_returns_unknown(self, queue_with_stub):
        q, stub = queue_with_stub
        job = _make_job(1, step_name=None, remote_pid=None)
        assert q._check_remote(job) == ("unknown", None)
