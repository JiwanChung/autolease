"""Async job queue and dispatcher."""

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .slurm import Slurm, SlurmConfig, Lease
from .config import PoolConfig, GPU_VRAM
from .sync import sync as rsync_project, get_remote_dir


JOB_SCHEMA_VERSION = 2  # bump on every dataclass change


@dataclass
class Job:
    id: int
    project: str
    command: str
    state: str  # queued, running, done, failed
    num_gpus: int = 1
    min_vram: int = 0  # GB, 0 = any
    gpu_type: Optional[str] = None  # None = any
    priority: int = 0  # higher = more important
    remote_cwd: Optional[str] = None  # remote dir to cd into before running
    exit_code: Optional[int] = None
    lease_job_id: Optional[int] = None
    remote_pid: Optional[int] = None  # legacy: login-node wrapper PID (kept
                                      # for in-flight jobs launched before
                                      # the wrapperless refactor)
    step_name: Optional[str] = None  # srun --job-name marker, used as the
                                     # canonical liveness handle via squeue
    node: Optional[str] = None
    submitted: Optional[str] = None
    started: Optional[str] = None
    finished: Optional[str] = None
    schema_version: int = JOB_SCHEMA_VERSION


def _migrate_job_dict(d: dict) -> dict:
    """Bring an on-disk job JSON dict up to the current schema.
    Cheap and idempotent — just sets defaults for missing fields. Bump
    JOB_SCHEMA_VERSION when adding a field that needs a non-default migration."""
    v = d.get("schema_version", 0)
    if v < 1:
        d.setdefault("priority", 0)
        d.setdefault("remote_cwd", None)
    if v < 2:
        d.setdefault("step_name", None)
    d["schema_version"] = JOB_SCHEMA_VERSION
    return d


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _detect_project() -> str:
    """Auto-detect project name from git root or cwd."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return os.path.basename(r.stdout.strip())
    except Exception:
        pass
    return os.path.basename(os.getcwd())


class JobQueue:
    def __init__(self, config: PoolConfig):
        self.config = config
        self.slurm = Slurm(SlurmConfig(ssh_host=config.ssh_host, shell=config.shell))
        self._jobs_dir = os.path.join(config.state_path, "jobs")
        self._counter_file = os.path.join(config.state_path, "next_job_id")
        self._log_file = os.path.join(config.state_path, "events.log")

    def _log_event(self, msg: str):
        """Append a timestamped event to the log file."""
        os.makedirs(os.path.dirname(self._log_file), exist_ok=True)
        with open(self._log_file, "a") as f:
            f.write(f"[{_now()}] {msg}\n")

    def _ensure_dirs(self):
        os.makedirs(self._jobs_dir, exist_ok=True)

    def _next_id(self) -> int:
        self._ensure_dirs()
        if os.path.exists(self._counter_file):
            with open(self._counter_file) as f:
                n = int(f.read().strip())
        else:
            n = 1
        with open(self._counter_file, "w") as f:
            f.write(str(n + 1))
        return n

    def _job_path(self, job_id: int) -> str:
        return os.path.join(self._jobs_dir, f"{job_id}.json")

    def _job_history_path(self, job_id: int) -> str:
        return os.path.join(self._jobs_dir, f"{job_id}.history")

    def _log_job_history(self, job_id: int, event: str, **details) -> None:
        """Append a timestamped event to a job's per-job history log.
        Cheap (one file append, no SSH). Used to reconstruct what happened
        to a job when debugging — captures state transitions, dispatches,
        preemptions, recoveries, cancels."""
        self._ensure_dirs()
        line = f"[{_now()}] {event}"
        if details:
            line += " " + " ".join(f"{k}={v}" for k, v in details.items())
        try:
            with open(self._job_history_path(job_id), "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _read_job_history(self, job_id: int) -> list[str]:
        p = self._job_history_path(job_id)
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return [line.rstrip("\n") for line in f]

    def _save_job(self, job: Job):
        self._ensure_dirs()
        tmp = self._job_path(job.id) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(job), f, indent=2)
        os.replace(tmp, self._job_path(job.id))

    def _load_job(self, job_id: int) -> Optional[Job]:
        p = self._job_path(job_id)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            d = json.load(f)
        return Job(**_migrate_job_dict(d))

    def _all_jobs(self) -> list[Job]:
        self._ensure_dirs()
        jobs = []
        for fname in os.listdir(self._jobs_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self._jobs_dir, fname)) as f:
                        d = json.load(f)
                    jobs.append(Job(**_migrate_job_dict(d)))
                except (json.JSONDecodeError, TypeError):
                    continue
        jobs.sort(key=lambda j: j.id)
        return jobs

    # ── Remote execution ──

    def _remote_job_dir(self, job_id: int) -> str:
        return f"~/.autolease/jobs/{job_id}"

    def _step_name(self, job: Job) -> str:
        """Slurm step name used as the canonical liveness handle."""
        return f"autolease-job-{job.id}"

    def _launch_remote(self, job: Job, lease: Lease) -> Optional[tuple[int, str]]:
        """Write script to remote, launch via nohup srun.
        The login-node `nohup bash` is intentionally minimal — it only exists
        to detach from SSH and to stage tee for chronological log interleaving.
        Liveness is tracked via the srun job-step's name (squeue --steps), not
        the wrapper PID. Returns (login_pid, step_name) for backward-compat
        and future cancel/kill operations."""
        rdir = self._remote_job_dir(job.id)
        sh = self.slurm.cfg.shell
        step_name = self._step_name(job)

        # The srun --job-name is the canonical handle: it identifies our step
        # within the lease via `squeue --steps --jobid={lease}`. Login-node
        # processes can come and go (network blips, ControlMaster wedging,
        # login-node reboots) — Slurm always knows whether the step is alive
        # on the compute node.
        # num_gpus == 0 → CPU job, no --gres (runs as a CPU-only step inside
        # whatever lease it landed on, GPU or CPU).
        if job.num_gpus > 0:
            srun = (
                f"srun --jobid={lease.job_id} --gres=gpu:{job.num_gpus}"
                f" --overlap --job-name={step_name}"
            )
        else:
            srun = (
                f"srun --jobid={lease.job_id}"
                f" --overlap --job-name={step_name}"
            )

        # Step 1: write run.sh (the user's command, on the compute node).
        cd_prefix = f"cd {job.remote_cwd} && " if job.remote_cwd else ""
        # Step 2: write launch.sh (the login-node bash script that actually
        # invokes srun + tee + exit-code capture). Externalised so it's
        # readable on disk for debugging instead of being a Python f-string
        # of nested bash with quoting hell.
        launch_script = (
            f"#!/bin/bash\n"
            f"# Auto-generated by autolease for job {job.id}\n"
            f"# srun job-step name: {step_name}\n"
            f"# To debug: cat {rdir}/launch.sh\n"
            f"{srun} {sh} {rdir}/run.sh \\\n"
            f"  > >(tee -a {rdir}/stdout >> {rdir}/combined) \\\n"
            f"  2> >(tee -a {rdir}/stderr >> {rdir}/combined)\n"
            f"__EC=$?\n"
            f"# wait for tee subshells to flush before recording exit code\n"
            f"wait\n"
            f"echo $__EC > {rdir}/exit_code\n"
        )
        # Single SSH: setup + launch. CRITICAL: statements must be separated
        # by newlines (not `&&`). With `&&`, the trailing `&` backgrounds the
        # WHOLE pipeline in a subshell that holds the SSH channel's FDs,
        # which makes SSH block until the launched job finishes. With
        # newlines, `&` applies only to the immediately preceding command
        # (the nohup), so SSH disconnects right after `echo $!`.
        # We also use `< /dev/null` on the nohup to fully detach stdin.
        cmd = (
            f"mkdir -p {rdir}\n"
            f"cat > {rdir}/run.sh << '__AUTOLEASE_RUN_EOF__'\n"
            f"{cd_prefix}{job.command}\n"
            f"__AUTOLEASE_RUN_EOF__\n"
            f"cat > {rdir}/launch.sh << '__AUTOLEASE_LAUNCH_EOF__'\n"
            f"{launch_script}"
            f"__AUTOLEASE_LAUNCH_EOF__\n"
            f"chmod +x {rdir}/run.sh {rdir}/launch.sh\n"
            f"nohup bash {rdir}/launch.sh > /dev/null 2>&1 < /dev/null &\n"
            f"echo $!"
        )
        r = self.slurm.cfg.run(cmd, timeout=10)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        try:
            # Last line is the echoed PID
            login_pid = int(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return None
        return (login_pid, step_name)

    def _check_remote(self, job: Job) -> tuple[str, Optional[int]]:
        """Check remote job state in a single SSH call.
        Slurm is the source of truth: we look up our srun job-step by name
        within the lease. Login-node processes (the wrapper bash, ssh
        ControlMaster, etc.) can come and go without affecting the verdict.

        Returns (state, exit_code). state is one of:
          - 'running': step is alive in Slurm (`squeue -s -j LEASE` lists
                       it), OR exit_code not yet written but output files
                       written in last 60s (handles user scripts that
                       detached background work; the wrapper may be mid-flush)
          - 'done':    exit_code file present (canonical signal — Slurm step
                       may already be gone or still finalizing)
          - 'lost':    no exit_code file AND step not in Slurm AND output
                       files quiet for 60s+
          - 'unknown': SSH failed or output unrecognized — DO NOT change state

        Backward compat: if the Job has no step_name (legacy in-flight job),
        falls back to PID-based checking via _check_remote_legacy.
        """
        if not job.step_name:
            return self._check_remote_legacy(job)
        if not job.lease_job_id:
            return "unknown", None

        rdir = self._remote_job_dir(job.id)
        step_name = job.step_name
        lease_id = job.lease_job_id

        # One SSH round-trip:
        #   1. exit_code file present? -> __AL_DONE__:<code>
        #   2. step in squeue?         -> __AL_RUN__
        #   3. otherwise mtime of combined/stdout, caller decides
        # squeue --steps --jobid=N --noheader -o "%j" lists step names.
        # We grep -Fx (fixed-string, exact line) for our name.
        inner = (
            f"if [ -s {rdir}/exit_code ]; then "
            f"  echo __AL_DONE__:`cat {rdir}/exit_code`; "
            f"elif squeue -s -j {lease_id} --noheader -o '%j' 2>/dev/null "
            f"     | grep -Fxq {shlex.quote(step_name)}; then "
            f"  echo __AL_RUN__; "
            f"else "
            f"  echo __AL_NOSTEP__:`stat -c %Y {rdir}/combined 2>/dev/null "
            f"    || stat -c %Y {rdir}/stdout 2>/dev/null || echo 0`; "
            f"fi"
        )
        cmd = f"/bin/sh -c {shlex.quote(inner)}"
        try:
            r = self.slurm.cfg.run(cmd, timeout=10)
        except Exception:
            return "unknown", None
        if r.returncode != 0:
            return "unknown", None
        for line in r.stdout.splitlines():
            line = line.strip()
            if line == "__AL_RUN__":
                return "running", None
            if line.startswith("__AL_DONE__:"):
                code_str = line.split(":", 1)[1].strip()
                try:
                    return "done", int(code_str)
                except ValueError:
                    return "done", None
            if line.startswith("__AL_NOSTEP__:"):
                # Slurm doesn't know about our step and there's no exit_code.
                # Could be: (a) Slurm racing — step just registered/just
                # ended, (b) output flushing, (c) actually lost.
                mtime_str = line.split(":", 1)[1].strip()
                try:
                    mtime = int(mtime_str)
                except ValueError:
                    mtime = 0
                if mtime > 0:
                    import time as _time
                    if _time.time() - mtime < self.config.mtime_threshold:
                        return "running", None
                return "lost", None
        return "unknown", None

    def _check_remote_legacy(self, job: Job) -> tuple[str, Optional[int]]:
        """Legacy PID-based check for jobs launched before the step_name
        refactor. Same logic as the old _check_remote: use kill -0 + mtime
        defensive fallback. Once these in-flight jobs finish, this path
        won't be exercised again."""
        pid = job.remote_pid
        if not pid or pid <= 0:
            return "unknown", None
        rdir = self._remote_job_dir(job.id)
        inner = (
            f"kill -0 {pid} 2>/dev/null && echo __AL_RUN__ || "
            f"( test -s {rdir}/exit_code && "
            f"  echo __AL_DONE__:`cat {rdir}/exit_code` || "
            f"  echo __AL_PIDGONE__:`stat -c %Y {rdir}/combined 2>/dev/null "
            f"  || stat -c %Y {rdir}/stdout 2>/dev/null || echo 0` )"
        )
        cmd = f"/bin/sh -c {shlex.quote(inner)}"
        try:
            r = self.slurm.cfg.run(cmd, timeout=10)
        except Exception:
            return "unknown", None
        if r.returncode != 0:
            return "unknown", None
        for line in r.stdout.splitlines():
            line = line.strip()
            if line == "__AL_RUN__":
                return "running", None
            if line.startswith("__AL_DONE__:"):
                code_str = line.split(":", 1)[1].strip()
                try:
                    return "done", int(code_str)
                except ValueError:
                    return "done", None
            if line.startswith("__AL_PIDGONE__:"):
                mtime_str = line.split(":", 1)[1].strip()
                try:
                    mtime = int(mtime_str)
                except ValueError:
                    mtime = 0
                if mtime > 0:
                    import time as _time
                    if _time.time() - mtime < self.config.mtime_threshold:
                        return "running", None
                return "lost", None
        return "unknown", None

    def read_log(self, job_id: int, stream: str = "stdout",
                 tail: Optional[int] = None,
                 byte_offset: int = 0) -> str:
        """Read stdout, stderr, or combined from remote.
        byte_offset: skip first N bytes (for incremental reads)."""
        rdir = self._remote_job_dir(job_id)
        if tail:
            cmd = f"tail -n {tail} {rdir}/{stream} 2>/dev/null"
        elif byte_offset > 0:
            cmd = f"tail -c +{byte_offset + 1} {rdir}/{stream} 2>/dev/null"
        else:
            cmd = f"cat {rdir}/{stream} 2>/dev/null"
        r = self.slurm.cfg.run(cmd, timeout=10)
        return r.stdout

    def _kill_remote(self, job: Job):
        """Kill a running remote job. Uses scancel on the Slurm step (the
        canonical kill path — survives login-node weirdness, terminates the
        compute-node work cleanly), then SIGKILLs the login-node wrapper as
        a backup. For legacy jobs (no step_name), only the PID kill applies."""
        if job.step_name and job.lease_job_id:
            # Find the step ID by name within our lease, then scancel it.
            # `scancel` doesn't support --name filtering across all jobs in a
            # way that works for steps, so we look up the step ID first.
            inner = (
                f"squeue -s -j {job.lease_job_id} --noheader -o '%i %j' 2>/dev/null"
                f" | awk '$2 == {shlex.quote(job.step_name)} {{print $1; exit}}'"
            )
            cmd = f"/bin/sh -c {shlex.quote(inner)}"
            try:
                r = self.slurm.cfg.run(cmd, timeout=10)
                step_id = r.stdout.strip()
                if step_id:
                    self.slurm.cfg.run(f"scancel {step_id}", timeout=10)
            except Exception:
                pass
        if job.remote_pid:
            try:
                self.slurm.cfg.run(
                    f"kill {job.remote_pid} 2>/dev/null;"
                    f" kill -9 {job.remote_pid} 2>/dev/null",
                    timeout=10,
                )
            except Exception:
                pass

    # ── Queue operations ──

    def _wrap_with_env(self, command: str, env: Optional[str] = None) -> str:
        """Wrap command with env activation if configured."""
        env_name = env or self.config.env
        if not env_name:
            return command
        activate = self.config.env_activate.replace("{env}", env_name)
        return f"{activate} {command}"

    def submit(self, command: str, project: Optional[str] = None,
               num_gpus: int = 1, min_vram: int = 0,
               gpu_type: Optional[str] = None,
               priority: int = 0,
               env: Optional[str] = None,
               no_sync: bool = False) -> Job:
        """Submit a new job to the queue. Auto-syncs code files first."""
        if project is None:
            project = _detect_project()

        # Wrap command with env activation
        wrapped_command = self._wrap_with_env(command, env)

        # Auto-sync code files to cluster
        remote_cwd = None
        if not no_sync:
            try:
                r = rsync_project(self.config)
                remote_cwd = get_remote_dir(self.config)
                if r is not None and r.returncode != 0:
                    self._log_event(f"SYNC_WARN job for {project}: rsync failed: {r.stderr.strip()[:100]}")
            except Exception as e:
                self._log_event(f"SYNC_WARN job for {project}: {e}")

        job = Job(
            id=self._next_id(),
            project=project,
            command=wrapped_command,
            state="queued",
            num_gpus=num_gpus,
            min_vram=min_vram,
            gpu_type=gpu_type,
            priority=priority,
            remote_cwd=remote_cwd,
            submitted=_now(),
        )
        self._save_job(job)
        self._log_event(f"SUBMIT job {job.id} project={project} priority={priority} gpus={num_gpus} cwd={remote_cwd}")
        self._log_job_history(
            job.id, "SUBMIT", project=project, priority=priority,
            gpus=num_gpus, cwd=remote_cwd,
        )
        self.dispatch()
        return job

    def get(self, job_id: int, refresh: bool = True,
            dispatch: bool = True) -> Optional[Job]:
        """Get a job. By default refreshes running state (1 SSH) and dispatches
        pending jobs. Pass refresh=False / dispatch=False to skip SSH calls."""
        job = self._load_job(job_id)
        if refresh and job and job.state == "running":
            self._refresh_running(job)
        if dispatch:
            self.dispatch()
            job = self._load_job(job_id)  # re-read in case dispatch changed it
        return job

    def _refresh_running(self, job: Job):
        """Update a running job's state from remote (1 SSH call).
        On 'running' or 'unknown' (SSH error / unrecognized output), leave
        the job state unchanged — never mark a job as failed unless we
        positively confirmed the PID is gone with no exit_code file."""
        remote_state, exit_code = self._check_remote(job)
        if remote_state == "done":
            job.exit_code = exit_code
            job.state = "done"
            job.finished = _now()
            self._save_job(job)
            self._log_job_history(job.id, "DONE", exit_code=exit_code)
        elif remote_state == "lost":
            job.state = "failed"
            job.finished = _now()
            self._save_job(job)
            self._log_job_history(
                job.id, "LOST",
                detail="step gone, no exit_code, output quiet",
            )
        # 'running' or 'unknown' → no change

    def cancel(self, job_id: int) -> bool:
        """Cancel a queued or running job."""
        job = self._load_job(job_id)
        if job is None:
            return False
        if job.state == "running":
            self._kill_remote(job)
        if job.state in ("queued", "running"):
            job.state = "failed"
            job.exit_code = -1
            job.finished = _now()
            self._save_job(job)
            self._log_job_history(job_id, "CANCEL")
            return True
        return False

    def list_jobs(self, project: Optional[str] = None,
                  active_only: bool = False,
                  refresh: bool = True,
                  dispatch: bool = True) -> list[Job]:
        """List jobs, optionally filtered by project.
        refresh=True: check remote state of each running job (1 SSH per job).
        dispatch=True: also dispatch queued jobs (extra SSH if queue non-empty)."""
        jobs = self._all_jobs()
        if refresh:
            for j in jobs:
                if j.state == "running":
                    self._refresh_running(j)
        if dispatch:
            self.dispatch()
            jobs = self._all_jobs()
        if project:
            jobs = [j for j in jobs if j.project == project]
        if active_only:
            jobs = [j for j in jobs if j.state in ("queued", "running")]
        return jobs

    # ── Dispatcher ──

    def _lease_matches(self, lease: Lease, job: Job) -> bool:
        """Check if a lease can host this job.
        CPU job (num_gpus == 0) accepts any RUNNING lease (CPU lease preferred,
        GPU lease piggyback allowed). GPU job requires a GPU lease that fits
        the count / type / vram constraints."""
        if lease.state != "RUNNING":
            return False

        job_is_cpu = job.num_gpus == 0
        lease_is_cpu = lease.num_gpus == 0

        if not job_is_cpu and lease_is_cpu:
            return False  # GPU job can't run on a CPU lease

        # If user pinned a gpu_type, honor it. A CPU lease can never satisfy
        # an explicit gpu_type request.
        if job.gpu_type:
            if lease_is_cpu:
                return False
            if lease.gpu_type.lower() != job.gpu_type.lower():
                return False

        if job_is_cpu:
            return True  # count / vram filters don't apply to CPU jobs

        if lease.num_gpus < job.num_gpus:
            return False
        if job.min_vram > 0:
            vram = GPU_VRAM.get(lease.gpu_type, 0)
            if vram < job.min_vram:
                return False
        return True

    def _lease_is_busy(self, lease: Lease, running_jobs: list[Job],
                       for_cpu_job: bool = False) -> bool:
        """Lease can host at most one CPU job + at most one GPU job at a time.
        CPU and GPU work coexist via --overlap (they claim disjoint resources).
        for_cpu_job=True: busy iff another CPU job is on the lease.
        for_cpu_job=False: busy iff another GPU job is on the lease."""
        for j in running_jobs:
            if j.lease_job_id != lease.job_id:
                continue
            if (j.num_gpus == 0) == for_cpu_job:
                return True
        return False

    def _preempt(self, victim: Job) -> None:
        """Kill a running job and re-queue it."""
        self._kill_remote(victim)
        old_node = victim.node
        victim.state = "queued"
        victim.remote_pid = None
        victim.step_name = None
        victim.lease_job_id = None
        victim.node = None
        victim.started = None
        self._save_job(victim)
        self._log_event(
            f"PREEMPT job {victim.id} (priority={victim.priority}, "
            f"project={victim.project}) on {old_node} — re-queued"
        )
        self._log_job_history(victim.id, "PREEMPT", node=old_node)

    def dispatch(self, leases: Optional[list] = None,
                 skip_running_refresh: bool = False):
        """Try to dispatch queued jobs to free lease slots.
        Higher priority jobs dispatch first. If no free slot,
        a higher-priority job can preempt a lower-priority running job.

        leases: already-refreshed leases (skips the pool.refresh() SSH call)
        skip_running_refresh: skip _refresh_running per job (saves 1 SSH per job)
        """
        # Fast path: no queued jobs, no work to do → skip all SSH calls
        queued_quick = [j for j in self._all_jobs() if j.state == "queued"]
        if not queued_quick:
            return

        if leases is None:
            from .pool import Pool
            pool = Pool(self.config)
            leases = pool.refresh()

        all_jobs = self._all_jobs()
        running = [j for j in all_jobs if j.state == "running"]

        if not skip_running_refresh:
            # Refresh running jobs first — free up slots for finished ones
            for j in running:
                self._refresh_running(j)
            running = [j for j in self._all_jobs() if j.state == "running"]

        queued = [j for j in self._all_jobs() if j.state == "queued"]
        if not queued:
            return

        # Sort queued: highest priority first, then by submit time (id)
        queued.sort(key=lambda j: (-j.priority, j.id))

        # Round-robin within same priority tier
        from collections import OrderedDict
        rr_queue: list[Job] = []
        # Group by priority tier
        tiers: dict[int, list[Job]] = {}
        for j in queued:
            tiers.setdefault(j.priority, []).append(j)

        for pri in sorted(tiers.keys(), reverse=True):
            tier_jobs = tiers[pri]
            # Round-robin within this tier
            by_project: OrderedDict[str, list[Job]] = OrderedDict()
            for j in tier_jobs:
                by_project.setdefault(j.project, []).append(j)
            while any(by_project.values()):
                for proj in list(by_project.keys()):
                    if by_project[proj]:
                        rr_queue.append(by_project[proj].pop(0))
                    else:
                        del by_project[proj]

        # Sort leases best-fit ascending: prefer the smallest LEASE (in GPU
        # count) that still satisfies the job, then the smallest VRAM. So a
        # 1-GPU job lands on a 1-GPU lease before a 2-GPU lease; a 2-GPU job
        # lands on a 2-GPU lease before a 4-GPU lease. Without this, big
        # leases get fragmented (one lease slot occupied → entire lease
        # marked busy by _lease_is_busy) and small jobs waste capacity.
        # CPU jobs (num_gpus == 0) need a different order: CPU leases first
        # (their natural home), then smallest GPU lease as piggyback overflow.
        def _lease_order_for(job: Job):
            if job.num_gpus == 0:
                return lambda l: (
                    0 if l.num_gpus == 0 else 1,
                    l.num_gpus,
                    GPU_VRAM.get(l.gpu_type, 0),
                )
            return lambda l: (l.num_gpus, GPU_VRAM.get(l.gpu_type, 0))

        for job in rr_queue:
            is_cpu_job = job.num_gpus == 0
            leases_by_fit = sorted(leases, key=_lease_order_for(job))
            # Try free lease first
            launched = False
            for lease in leases_by_fit:
                if not self._lease_matches(lease, job):
                    continue
                if self._lease_is_busy(lease, running, for_cpu_job=is_cpu_job):
                    continue
                result = self._launch_remote(job, lease)
                if result:
                    pid, step_name = result
                    job.state = "running"
                    job.remote_pid = pid
                    job.step_name = step_name
                    job.lease_job_id = lease.job_id
                    job.node = lease.node
                    job.started = _now()
                    self._save_job(job)
                    self._log_event(
                        f"DISPATCH job {job.id} (priority={job.priority}) "
                        f"-> {lease.node} ({lease.gpu_type}) step={step_name}"
                    )
                    self._log_job_history(
                        job.id, "DISPATCH", lease=lease.job_id, step=step_name,
                        node=lease.node, gpu=lease.gpu_type, pid=pid,
                    )
                    running.append(job)
                    launched = True
                    break
                else:
                    job.state = "failed"
                    job.finished = _now()
                    self._save_job(job)
                    self._log_job_history(
                        job.id, "LAUNCH_FAILED", lease=lease.job_id,
                    )
                    launched = True
                    break

            if launched:
                continue

            # No free lease — try preemption
            # Find lowest-priority running job on a matching lease.
            # Only same-kind jobs are preemptable: a CPU job can preempt a
            # CPU job; a GPU job preempts a GPU job. Cross-kind preemption
            # would needlessly evict work that doesn't compete for resources.
            candidates = []
            for lease in leases_by_fit:
                if not self._lease_matches(lease, job):
                    continue
                for rj in running:
                    if rj.lease_job_id != lease.job_id:
                        continue
                    if (rj.num_gpus == 0) != is_cpu_job:
                        continue
                    if rj.priority < job.priority:
                        candidates.append((rj, lease))

            if not candidates:
                continue  # no preemptable job, stays queued

            # Preempt the lowest-priority one
            candidates.sort(key=lambda x: (x[0].priority, -x[0].id))
            victim, lease = candidates[0]

            self._preempt(victim)
            running = [r for r in running if r.id != victim.id]

            # Launch on freed lease
            result = self._launch_remote(job, lease)
            if result:
                pid, step_name = result
                job.state = "running"
                job.remote_pid = pid
                job.step_name = step_name
                job.lease_job_id = lease.job_id
                job.node = lease.node
                job.started = _now()
                self._save_job(job)
                self._log_event(
                    f"DISPATCH job {job.id} (priority={job.priority}) "
                    f"-> {lease.node} ({lease.gpu_type}) step={step_name} "
                    f"[preempted job {victim.id}]"
                )
                self._log_job_history(
                    job.id, "DISPATCH", lease=lease.job_id, step=step_name,
                    node=lease.node, gpu=lease.gpu_type, pid=pid,
                    preempted=victim.id,
                )
                running.append(job)
