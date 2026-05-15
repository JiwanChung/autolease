"""gpu-procs / gpu-clean: detection and cleanup of orphaned GPU processes.

A GPU process is an "orphan" if it's still holding GPU memory on a lease's
compute node but Slurm doesn't track it in any live step (i.e. its step ended
but the process survived — typically a backgrounded child or one that ignored
SIGTERM). Detection is via `scontrol listpids` cross-referenced with
`nvidia-smi --query-compute-apps=pid`.
"""

from autolease.pool import Pool, GpuProc
from autolease.slurm import Lease


def _running_lease(jid=100, num_gpus=2):
    return Lease(
        job_id=jid, partition="p", qos="q", gpu_type="RTX3090",
        num_gpus=num_gpus, node=f"n{jid}", state="RUNNING",
    )


def _cpu_lease(jid=200):
    return Lease(
        job_id=jid, partition="cpu_p", qos="q", gpu_type="cpu",
        num_gpus=0, node=f"n{jid}", state="RUNNING",
    )


class TestParseGpuInspect:
    def test_full_parse_with_live_step_and_pid(self):
        """Both squeue (live steps + names) and scontrol listpids parsed,
        then cross-referenced via _classify_pid."""
        out = (
            "__AL_STEPS_BEGIN__\n"
            "1451430.0    autolease-job-99\n"
            "1451430.batch    sleep\n"
            "__AL_STEPS_END__\n"
            "__AL_PIDS_BEGIN__\n"
            "PID      JOBID    STEPID   LOCALID  GLOBALID\n"
            "12345    1451430  0        0        0\n"
            "__AL_PIDS_END__\n"
            "__AL_PROC__12345|GPU-aaa|512|python train.py\n"
        )
        live, pid2step, name_by_id, procs = Pool._parse_gpu_inspect(out)
        assert live == {"0", "batch"}
        assert pid2step == {12345: "0"}
        assert name_by_id == {"0": "autolease-job-99", "batch": "sleep"}
        assert len(procs) == 1
        # Step "0" is live and not stale → "live"
        assert Pool._classify_pid(12345, live, pid2step,
                                  step_name_by_id=name_by_id) == "live"

    def test_pid_in_dead_step_is_orphan(self):
        """scontrol still shows PID under step '0' but '0' is no longer
        in squeue → orphan."""
        out = (
            "__AL_STEPS_BEGIN__\n"
            "1451430.batch    sleep\n"
            "__AL_STEPS_END__\n"
            "__AL_PIDS_BEGIN__\n"
            "PID      JOBID    STEPID\n"
            "99999    1451430  0\n"
            "__AL_PIDS_END__\n"
            "__AL_PROC__99999|GPU-bbb|2048|python stuck.py\n"
        )
        live, pid2step, _, _ = Pool._parse_gpu_inspect(out)
        assert "0" not in live
        assert pid2step[99999] == "0"
        assert Pool._classify_pid(99999, live, pid2step) == "orphan"

    def test_pid_in_step_with_stale_autolease_job_is_orphan(self):
        """Regression for user-reported case: Slurm still considers the
        step alive (slurmstepd hasn't reaped yet) but autolease's records
        say the owning job is done/failed → must be classified orphan,
        not live."""
        out = (
            "__AL_STEPS_BEGIN__\n"
            "1451430.0    autolease-job-7\n"     # squeue thinks step 0 is alive
            "1451430.batch    sleep\n"
            "__AL_STEPS_END__\n"
            "__AL_PIDS_BEGIN__\n"
            "PID      JOBID    STEPID\n"
            "88888    1451430  0\n"
            "__AL_PIDS_END__\n"
            "__AL_PROC__88888|GPU-c|4096|python\n"
        )
        live, pid2step, name_by_id, _ = Pool._parse_gpu_inspect(out)
        # Without stale info: would be 'live' (step is in squeue)
        assert Pool._classify_pid(88888, live, pid2step,
                                  step_name_by_id=name_by_id) == "live"
        # With stale info (autolease says job 7 is done): orphan
        assert Pool._classify_pid(
            88888, live, pid2step,
            step_name_by_id=name_by_id,
            stale_step_names={"autolease-job-7"},
        ) == "orphan"

    def test_pid_in_batch_step_is_live(self):
        """The lease's sleep wrapper is in 'batch' — never kill it."""
        out = (
            "__AL_STEPS_BEGIN__\n"
            "1451430.batch    sleep\n"
            "__AL_STEPS_END__\n"
            "__AL_PIDS_BEGIN__\n"
            "PID      JOBID    STEPID\n"
            "5555     1451430  batch\n"
            "__AL_PIDS_END__\n"
            "__AL_PROC__5555|GPU-x|128|sleep infinity\n"
        )
        live, pid2step, _, _ = Pool._parse_gpu_inspect(out)
        assert Pool._classify_pid(5555, live, pid2step) == "live"

    def test_pid_in_extern_step_is_orphan(self):
        """Adopted PIDs in 'extern' are typically survivors of dead steps
        — treat as orphan so they're killable. (Real running work doesn't
        normally land in extern.)"""
        out = (
            "__AL_STEPS_BEGIN__\n"
            "1451430.batch sleep\n"
            "__AL_STEPS_END__\n"
            "__AL_PIDS_BEGIN__\n"
            "PID JOBID STEPID\n"
            "777 1451430 extern\n"
            "__AL_PIDS_END__\n"
            "__AL_PROC__777|GPU-y|999|adopted\n"
        )
        live, pid2step, _, _ = Pool._parse_gpu_inspect(out)
        assert Pool._classify_pid(777, live, pid2step) == "orphan"

    def test_pid_not_in_scontrol_is_orphan(self):
        out = (
            "__AL_STEPS_BEGIN__\n"
            "1451430.batch sleep\n"
            "__AL_STEPS_END__\n"
            "__AL_PIDS_BEGIN__\n"
            "PID JOBID STEPID\n"
            "100 1451430 batch\n"
            "__AL_PIDS_END__\n"
            "__AL_PROC__777|GPU-y|999|escaped\n"
        )
        live, pid2step, _, _ = Pool._parse_gpu_inspect(out)
        assert Pool._classify_pid(777, live, pid2step) == "orphan"

    def test_missing_sections_yield_unknown(self):
        out = "__AL_PROC__12345|GPU-aaa|512|python train.py\n"
        live, pid2step, _, _ = Pool._parse_gpu_inspect(out)
        assert live is None
        assert pid2step is None
        assert Pool._classify_pid(12345, live, pid2step) == "unknown"

    def test_tolerates_motd_and_noise(self):
        out = (
            "MOTD: welcome to the cluster\n"
            "**post-quantum warning**\n"
            "__AL_STEPS_BEGIN__\n"
            "STEPID NAME\n"
            "1451430.0 autolease-job-1\n"
            "__AL_STEPS_END__\n"
            "__AL_PIDS_BEGIN__\n"
            "PID  JOBID  STEPID\n"
            "55  1451430  0\n"
            "__AL_PIDS_END__\n"
            "stray nvidia-smi warning\n"
            "__AL_PROC__55|GPU-z|256|x\n"
            "__AL_PROC__66|GPU-z|512|y\n"
        )
        live, pid2step, name_by_id, procs = Pool._parse_gpu_inspect(out)
        assert live == {"0"}
        assert pid2step == {55: "0"}
        assert name_by_id == {"0": "autolease-job-1"}
        assert {p["pid"] for p in procs} == {55, 66}
        assert Pool._classify_pid(55, live, pid2step,
                                  step_name_by_id=name_by_id) == "live"
        assert Pool._classify_pid(66, live, pid2step) == "orphan"


class TestListGpuProcsClassification:
    """End-to-end classification through Pool.list_gpu_procs with a stub on
    slurm.cfg.run (we now bypass run_on_lease and force /bin/sh ourselves)."""

    def _patch_run(self, pool, stdout, stderr="", returncode=0):
        class Resp:
            def __init__(self):
                self.returncode = returncode
                self.stdout = stdout
                self.stderr = stderr
                self.args = ()
        pool.slurm.cfg.run = lambda cmd, timeout=30: Resp()  # type: ignore

    def test_classifies_live_and_orphan(self, cfg):
        """End-to-end: PID 100 is in live step '0', PID 200 has no cgroup
        entry → orphan (escaped); PID 300 is in dead step '1' → orphan."""
        pool = Pool(cfg)
        self._patch_run(pool, (
            "__AL_STEPS_BEGIN__\n"
            "100.0 autolease-job-1\n"
            "__AL_STEPS_END__\n"
            "__AL_PIDS_BEGIN__\n"
            "PID JOBID STEPID\n"
            "100 100 0\n"
            "300 100 1\n"
            "__AL_PIDS_END__\n"
            "__AL_PROC__100|GPU-a|512|python train.py\n"
            "__AL_PROC__200|GPU-a|1024|python escaped.py\n"
            "__AL_PROC__300|GPU-a|2048|python stuck.py\n"
        ))
        procs, debug = pool.list_gpu_procs(_running_lease())
        by_pid = {p.pid: p for p in procs}
        assert by_pid[100].status == "live"
        assert by_pid[100].step == "0"
        assert by_pid[100].step_name == "autolease-job-1"
        assert by_pid[200].status == "orphan"
        assert by_pid[200].step is None
        assert by_pid[300].status == "orphan"
        assert by_pid[300].step == "1"
        assert debug == ""

    def test_stale_step_names_marks_live_step_as_orphan(self, cfg):
        """Cross-reference: when caller passes stale_step_names, PIDs whose
        step name matches a stale autolease job become orphan even if
        Slurm still lists the step as alive."""
        pool = Pool(cfg)
        self._patch_run(pool, (
            "__AL_STEPS_BEGIN__\n"
            "100.0 autolease-job-7\n"  # squeue says step 0 is alive
            "__AL_STEPS_END__\n"
            "__AL_PIDS_BEGIN__\n"
            "PID JOBID STEPID\n"
            "555 100 0\n"
            "__AL_PIDS_END__\n"
            "__AL_PROC__555|GPU-a|2048|python\n"
        ))
        # Without stale info: live
        procs, _ = pool.list_gpu_procs(_running_lease())
        assert procs[0].status == "live"
        # With stale info (autolease's job 7 is done): orphan
        procs, _ = pool.list_gpu_procs(
            _running_lease(), stale_step_names={"autolease-job-7"},
        )
        assert procs[0].status == "orphan"
        assert procs[0].step_name == "autolease-job-7"

    def test_cpu_lease_returns_empty(self, cfg):
        pool = Pool(cfg)
        called = []
        pool.slurm.cfg.run = lambda cmd, timeout=30: called.append(cmd)  # type: ignore
        procs, debug = pool.list_gpu_procs(_cpu_lease())
        assert procs == []
        assert "CPU-only" in debug
        assert called == []

    def test_pending_lease_returns_empty(self, cfg):
        pool = Pool(cfg)
        pending = _running_lease()
        pending.state = "PENDING"
        called = []
        pool.slurm.cfg.run = lambda cmd, timeout=30: called.append(cmd)  # type: ignore
        procs, debug = pool.list_gpu_procs(pending)
        assert procs == []
        assert "PENDING" in debug
        assert called == []

    def test_unknown_when_sections_missing(self, cfg):
        """If neither STEPS nor PIDS section appeared, classifier yields
        'unknown' so we never act on uncertain data."""
        pool = Pool(cfg)
        self._patch_run(pool, "__AL_PROC__500|GPU-x|256|python\n")
        procs, debug = pool.list_gpu_procs(_running_lease())
        assert len(procs) == 1
        assert procs[0].status == "unknown"

    def test_empty_scan_surfaces_stderr_for_debug(self, cfg):
        """When 0 procs found, debug message includes stderr so the user
        can see what went wrong (shell mismatch, permission denied, etc.)."""
        pool = Pool(cfg)
        self._patch_run(pool, "", stderr="srun: error: no such jobid")
        procs, debug = pool.list_gpu_procs(_running_lease())
        assert procs == []
        assert "no such jobid" in debug


class TestSrunCommandUsesSh:
    """Regression: the inspect command goes through /bin/sh, not config.shell.
    Without this, fish/zsh login shells silently fail on bash-style scripts."""

    def test_uses_sh_not_config_shell(self, cfg, monkeypatch):
        cfg.shell = "fish"  # config.shell is fish
        pool = Pool(cfg)
        captured = []

        class Resp:
            returncode = 0
            stdout = ""
            stderr = ""
            args = ()

        def fake_run(cmd, timeout=30):
            captured.append(cmd)
            return Resp()

        pool.slurm.cfg.run = fake_run  # type: ignore
        pool.list_gpu_procs(_running_lease())
        assert len(captured) == 1
        assert "/bin/sh -c" in captured[0]
        # Must NOT use config.shell here
        assert " fish " not in captured[0]
        assert "srun --jobid=100" in captured[0]
        assert "--gres=gpu:2" in captured[0]
        assert "--overlap" in captured[0]
        # Inspect script must query Slurm with the lease's job id
        assert "squeue -s -j 100" in captured[0]
        assert "scontrol listpids 100" in captured[0]


class TestKillGpuProcsReportsRemaining:
    def test_freed_vs_still_holding(self, cfg):
        """After SIGTERM/SIGKILL, parse nvidia-smi to see which PIDs remain."""
        pool = Pool(cfg)

        class Resp:
            returncode = 0
            # PID 300 freed, PID 400 still holding
            stdout = "__AL_AFTER__\n400\n"
            stderr = ""
            args = ()

        pool.slurm.cfg.run = lambda cmd, timeout=30: Resp()  # type: ignore
        result = pool.kill_gpu_procs(_running_lease(), [300, 400])
        assert result == {300: True, 400: False}
