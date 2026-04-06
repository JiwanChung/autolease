"""Async job queue and dispatcher."""

import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .slurm import Slurm, SlurmConfig, Lease
from .config import PoolConfig, GPU_VRAM


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
    exit_code: Optional[int] = None
    lease_job_id: Optional[int] = None
    remote_pid: Optional[int] = None
    node: Optional[str] = None
    submitted: Optional[str] = None
    started: Optional[str] = None
    finished: Optional[str] = None


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
        d.setdefault("priority", 0)  # backward compat
        return Job(**d)

    def _all_jobs(self) -> list[Job]:
        self._ensure_dirs()
        jobs = []
        for fname in os.listdir(self._jobs_dir):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(self._jobs_dir, fname)) as f:
                        d = json.load(f)
                    d.setdefault("priority", 0)
                    jobs.append(Job(**d))
                except (json.JSONDecodeError, TypeError):
                    continue
        jobs.sort(key=lambda j: j.id)
        return jobs

    # ── Remote execution ──

    def _remote_job_dir(self, job_id: int) -> str:
        return f"~/.autolease/jobs/{job_id}"

    def _launch_remote(self, job: Job, lease: Lease) -> Optional[int]:
        """Write script to remote, launch via nohup srun, return remote PID."""
        rdir = self._remote_job_dir(job.id)
        sh = self.slurm.cfg.shell
        srun = f"srun --jobid={lease.job_id} --gres=gpu:{job.num_gpus} --overlap"

        # Step 1: write the job script to remote
        setup = (
            f"mkdir -p {rdir} && "
            f"cat > {rdir}/run.sh << '__AUTOLEASE_SCRIPT__'\n"
            f"{job.command}\n"
            f"__AUTOLEASE_SCRIPT__\n"
            f"chmod +x {rdir}/run.sh"
        )
        r = self.slurm.cfg.run(setup, timeout=10)
        if r.returncode != 0:
            return None

        # Step 2: launch with nohup, using configured shell
        launch = (
            f"nohup bash -c '"
            f"{srun} {sh} {rdir}/run.sh > {rdir}/stdout 2> {rdir}/stderr;"
            f" echo $? > {rdir}/exit_code"
            f"' > /dev/null 2>&1 & echo $!"
        )
        r = self.slurm.cfg.run(launch, timeout=10)
        if r.returncode != 0 or not r.stdout.strip():
            return None
        try:
            return int(r.stdout.strip())
        except ValueError:
            return None

    def _check_remote(self, job: Job) -> str:
        """Check remote job state. Returns 'running', 'done', or 'lost'."""
        rdir = self._remote_job_dir(job.id)
        if job.remote_pid:
            r = self.slurm.cfg.run(
                f"kill -0 {job.remote_pid} 2>/dev/null && echo alive || echo dead",
                timeout=10,
            )
            alive = r.stdout.strip() == "alive"
            if alive:
                return "running"

        # Process dead — check for exit code
        r = self.slurm.cfg.run(f"cat {rdir}/exit_code 2>/dev/null", timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return "done"
        return "lost"

    def _read_remote_exit_code(self, job: Job) -> Optional[int]:
        rdir = self._remote_job_dir(job.id)
        r = self.slurm.cfg.run(f"cat {rdir}/exit_code 2>/dev/null", timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            try:
                return int(r.stdout.strip())
            except ValueError:
                pass
        return None

    def read_log(self, job_id: int, stream: str = "stdout",
                 tail: Optional[int] = None) -> str:
        """Read stdout or stderr from remote."""
        rdir = self._remote_job_dir(job_id)
        if tail:
            cmd = f"tail -n {tail} {rdir}/{stream} 2>/dev/null"
        else:
            cmd = f"cat {rdir}/{stream} 2>/dev/null"
        r = self.slurm.cfg.run(cmd, timeout=10)
        return r.stdout

    def _kill_remote(self, job: Job):
        """Kill a running remote job."""
        if job.remote_pid:
            self.slurm.cfg.run(
                f"kill {job.remote_pid} 2>/dev/null; kill -9 {job.remote_pid} 2>/dev/null",
                timeout=10,
            )

    # ── Queue operations ──

    def submit(self, command: str, project: Optional[str] = None,
               num_gpus: int = 1, min_vram: int = 0,
               gpu_type: Optional[str] = None,
               priority: int = 0) -> Job:
        """Submit a new job to the queue. Returns the job."""
        if project is None:
            project = _detect_project()
        job = Job(
            id=self._next_id(),
            project=project,
            command=command,
            state="queued",
            num_gpus=num_gpus,
            min_vram=min_vram,
            gpu_type=gpu_type,
            priority=priority,
            submitted=_now(),
        )
        self._save_job(job)
        self._log_event(f"SUBMIT job {job.id} project={project} priority={priority} gpus={num_gpus}")
        self.dispatch()
        return job

    def get(self, job_id: int) -> Optional[Job]:
        """Get a job, refreshing its state if running. Also dispatches pending jobs."""
        job = self._load_job(job_id)
        if job and job.state == "running":
            self._refresh_running(job)
        self.dispatch()
        return self._load_job(job_id)  # re-read in case dispatch changed it

    def _refresh_running(self, job: Job):
        """Update a running job's state from remote."""
        remote_state = self._check_remote(job)
        if remote_state == "done":
            job.exit_code = self._read_remote_exit_code(job)
            job.state = "done"
            job.finished = _now()
            self._save_job(job)
        elif remote_state == "lost":
            job.state = "failed"
            job.finished = _now()
            self._save_job(job)

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
            return True
        return False

    def list_jobs(self, project: Optional[str] = None,
                  active_only: bool = False) -> list[Job]:
        """List jobs, optionally filtered by project. Also dispatches pending."""
        jobs = self._all_jobs()
        # Refresh running jobs
        for j in jobs:
            if j.state == "running":
                self._refresh_running(j)
        self.dispatch()
        # Re-read after dispatch
        jobs = self._all_jobs()
        if project:
            jobs = [j for j in jobs if j.project == project]
        if active_only:
            jobs = [j for j in jobs if j.state in ("queued", "running")]
        return jobs

    # ── Dispatcher ──

    def _lease_matches(self, lease: Lease, job: Job) -> bool:
        """Check if a lease satisfies a job's GPU requirements."""
        if lease.state != "RUNNING":
            return False
        if job.gpu_type and lease.gpu_type.lower() != job.gpu_type.lower():
            return False
        if lease.num_gpus < job.num_gpus:
            return False
        if job.min_vram > 0:
            vram = GPU_VRAM.get(lease.gpu_type, 0)
            if vram < job.min_vram:
                return False
        return True

    def _lease_is_busy(self, lease: Lease, running_jobs: list[Job]) -> bool:
        """Check if a lease already has a job running on it."""
        return any(j.lease_job_id == lease.job_id for j in running_jobs)

    def _preempt(self, victim: Job) -> None:
        """Kill a running job and re-queue it."""
        self._kill_remote(victim)
        old_node = victim.node
        victim.state = "queued"
        victim.remote_pid = None
        victim.lease_job_id = None
        victim.node = None
        victim.started = None
        self._save_job(victim)
        self._log_event(
            f"PREEMPT job {victim.id} (priority={victim.priority}, "
            f"project={victim.project}) on {old_node} — re-queued"
        )

    def dispatch(self):
        """Try to dispatch queued jobs to free lease slots.
        Higher priority jobs dispatch first. If no free slot,
        a higher-priority job can preempt a lower-priority running job."""
        from .pool import Pool
        pool = Pool(self.config)
        leases = pool.refresh()

        all_jobs = self._all_jobs()
        running = [j for j in all_jobs if j.state == "running"]

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

        # Sort leases by VRAM ascending — fill small GPUs first
        leases_by_vram = sorted(leases, key=lambda l: GPU_VRAM.get(l.gpu_type, 0))

        for job in rr_queue:
            # Try free lease first
            launched = False
            for lease in leases_by_vram:
                if not self._lease_matches(lease, job):
                    continue
                if self._lease_is_busy(lease, running):
                    continue
                pid = self._launch_remote(job, lease)
                if pid:
                    job.state = "running"
                    job.remote_pid = pid
                    job.lease_job_id = lease.job_id
                    job.node = lease.node
                    job.started = _now()
                    self._save_job(job)
                    self._log_event(
                        f"DISPATCH job {job.id} (priority={job.priority}) "
                        f"-> {lease.node} ({lease.gpu_type})"
                    )
                    running.append(job)
                    launched = True
                    break
                else:
                    job.state = "failed"
                    job.finished = _now()
                    self._save_job(job)
                    launched = True
                    break

            if launched:
                continue

            # No free lease — try preemption
            # Find lowest-priority running job on a matching lease
            candidates = []
            for lease in leases_by_vram:
                if not self._lease_matches(lease, job):
                    continue
                for rj in running:
                    if rj.lease_job_id == lease.job_id and rj.priority < job.priority:
                        candidates.append((rj, lease))

            if not candidates:
                continue  # no preemptable job, stays queued

            # Preempt the lowest-priority one
            candidates.sort(key=lambda x: (x[0].priority, -x[0].id))
            victim, lease = candidates[0]

            self._preempt(victim)
            running = [r for r in running if r.id != victim.id]

            # Launch on freed lease
            pid = self._launch_remote(job, lease)
            if pid:
                job.state = "running"
                job.remote_pid = pid
                job.lease_job_id = lease.job_id
                job.node = lease.node
                job.started = _now()
                self._save_job(job)
                self._log_event(
                    f"DISPATCH job {job.id} (priority={job.priority}) "
                    f"-> {lease.node} ({lease.gpu_type}) "
                    f"[preempted job {victim.id}]"
                )
                running.append(job)
