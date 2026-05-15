"""Pool manager — holds and tracks GPU leases."""

import json
import os
import shlex
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from .slurm import Slurm, SlurmConfig, Lease
from .config import PoolConfig, LeaseSpec


@dataclass
class GpuProc:
    """A GPU-holding process observed on a lease's compute node."""
    lease_job_id: int
    node: Optional[str]
    pid: int
    gpu_uuid: str
    memory_mb: int
    cmdline: str
    status: str  # 'live' | 'orphan' | 'unknown'
    step: Optional[str] = None       # slurm step id ('0', 'batch', 'extern', or None)
    step_name: Optional[str] = None  # squeue's %j for the step (e.g. 'autolease-job-7')


def _parse_slurm_time(s: str) -> Optional[datetime]:
    """Parse Slurm EndTime like '2026-04-06T15:30:00'."""
    if not s or s in ("Unknown", "N/A", "None"):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


class Pool:
    def __init__(self, config: PoolConfig):
        self.config = config
        self.slurm = Slurm(SlurmConfig(ssh_host=config.ssh_host, shell=config.shell))
        self._state_file = os.path.join(config.state_path, "state.json")

    # ── State persistence ──

    def _ensure_state_dir(self):
        os.makedirs(os.path.dirname(self._state_file), exist_ok=True)

    def _load_state(self) -> dict:
        if not os.path.exists(self._state_file):
            return {"leases": [], "bad_nodes": [], "cancelled": []}
        with open(self._state_file) as f:
            data = json.load(f)
        if isinstance(data, dict) and "leases" in data:
            data.setdefault("bad_nodes", [])
            data.setdefault("cancelled", [])
            return data
        return {"leases": [], "bad_nodes": [], "cancelled": []}

    def _save_state(self, leases: list[Lease], bad_nodes: Optional[list[str]] = None):
        self._ensure_state_dir()
        state = self._load_state()
        if bad_nodes is not None:
            state["bad_nodes"] = bad_nodes
        state["leases"] = [l.to_dict() for l in leases]
        tmp = self._state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self._state_file)

    def _get_leases(self) -> list[Lease]:
        state = self._load_state()
        return [Lease.from_dict(d) for d in state.get("leases", [])]

    def _get_bad_nodes(self) -> list[str]:
        state = self._load_state()
        return state.get("bad_nodes", [])

    def _all_excludes(self) -> str:
        """Combine config excludes + dynamically discovered bad nodes."""
        nodes = set(self.config.exclude_nodes) | set(self._get_bad_nodes())
        return ",".join(sorted(nodes)) if nodes else ""

    def _add_bad_node(self, node: str):
        bad = set(self._get_bad_nodes())
        bad.add(node)
        self._save_state(self._get_leases(), sorted(bad))

    # ── Core operations ──

    def up(self, spec: LeaseSpec) -> Lease:
        """Acquire a single lease. Returns the new lease."""
        exclude = spec.exclude or self._all_excludes()
        # CPU leases shouldn't carry GPU-node excludes (different node pool)
        if spec.num_gpus == 0:
            exclude = spec.exclude
        job_id = self.slurm.submit_holder(
            partition=spec.partition,
            qos=spec.qos,
            num_gpus=spec.num_gpus,
            cpus_per_task=spec.cpus_per_task,
            time=spec.time,
            exclude=exclude,
        )
        lease = Lease(
            job_id=job_id,
            partition=spec.partition,
            qos=spec.qos,
            gpu_type=spec.gpu_type,
            num_gpus=spec.num_gpus,
            state="PENDING",
            time_limit=spec.time,
        )
        existing = self._get_leases()
        existing.append(lease)
        self._save_state(existing)
        return lease

    def _acquire_replacement(self, old_lease: Lease) -> Optional[Lease]:
        """Submit a replacement lease with the same spec as old_lease."""
        exclude = self._all_excludes()
        time_limit = old_lease.time_limit or "4:00:00"
        try:
            job_id = self.slurm.submit_holder(
                partition=old_lease.partition,
                qos=old_lease.qos,
                num_gpus=old_lease.num_gpus,
                time=time_limit,
                exclude=exclude,
            )
            return Lease(
                job_id=job_id,
                partition=old_lease.partition,
                qos=old_lease.qos,
                gpu_type=old_lease.gpu_type,
                num_gpus=old_lease.num_gpus,
                state="PENDING",
                time_limit=time_limit,
            )
        except RuntimeError:
            return None

    def _get_cancelled(self) -> set[int]:
        state = self._load_state()
        return set(state.get("cancelled", []))

    def _add_cancelled(self, job_ids: list[int]):
        state = self._load_state()
        cancelled = set(state.get("cancelled", []))
        cancelled.update(job_ids)
        state["cancelled"] = list(cancelled)
        tmp = self._state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self._state_file)

    def release(self, job_id: int):
        """Cancel and remove a single lease."""
        self.slurm.cancel_job(job_id)
        leases = [l for l in self._get_leases() if l.job_id != job_id]
        self._save_state(leases)
        self._add_cancelled([job_id])

    def down(self):
        """Cancel all held leases."""
        leases = self._get_leases()
        ids = []
        for lease in leases:
            try:
                self.slurm.cancel_job(lease.job_id)
                ids.append(lease.job_id)
            except RuntimeError:
                pass
        self._save_state([])
        self._add_cancelled(ids)
        return len(leases)

    def refresh(self) -> list[Lease]:
        """Refresh lease states from Slurm in a single squeue call.
        Adopts orphaned autolease jobs. Sets self.lost_leases."""
        from .config import PARTITION_INFO
        old_leases = {l.job_id: l for l in self._get_leases()}
        self.lost_leases = []

        # Single SSH call: get all autolease jobs from squeue
        cancelled = self._get_cancelled()
        all_squeue = self.slurm.my_jobs("autolease")
        slurm_jobs = {sj["job_id"]: sj for sj in all_squeue if sj["job_id"] not in cancelled}

        # Clean cancelled IDs that have left squeue
        squeue_ids = {sj["job_id"] for sj in all_squeue}
        remaining = cancelled & squeue_ids
        if remaining != cancelled:
            state = self._load_state()
            state["cancelled"] = list(remaining)
            tmp = self._state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, self._state_file)

        # Detect lost leases (were running, no longer in squeue)
        for jid, old in old_leases.items():
            if jid not in slurm_jobs and old.state == "RUNNING":
                self.lost_leases.append(old)

        # Build alive list from squeue (source of truth)
        alive = []
        for jid, sj in slurm_jobs.items():
            if jid in old_leases:
                # Known lease — update state
                lease = old_leases[jid]
                lease.state = sj["state"]
                lease.node = sj["node"]
                lease.end_time = sj.get("end_time")
                lease.time_limit = sj.get("timelimit") or lease.time_limit
                lease.qos = sj.get("qos") or lease.qos
            else:
                # Orphan — adopt it
                gres = (sj.get("gres") or "").strip()
                if gres and gres not in ("(null)", "N/A") and "gpu" in gres.lower():
                    # GPU lease. Parse GRES like "gpu:RTX4090:2" or "gpu:2"
                    num_gpus = 1
                    gpu_type = "unknown"
                    gparts = gres.split(":")
                    try:
                        num_gpus = int(gparts[-1])
                    except ValueError:
                        pass
                    if len(gparts) >= 3:
                        gpu_type = gparts[1]
                    if gpu_type == "unknown":
                        pinfo = PARTITION_INFO.get(sj["partition"])
                        if pinfo:
                            gpu_type = pinfo[1]
                else:
                    # No GPU gres → CPU lease
                    num_gpus = 0
                    gpu_type = "cpu"
                lease = Lease(
                    job_id=jid,
                    partition=sj["partition"],
                    qos=sj.get("qos", ""),
                    gpu_type=gpu_type,
                    num_gpus=num_gpus,
                    node=sj["node"],
                    state=sj["state"],
                    end_time=sj.get("end_time"),
                    time_limit=sj.get("timelimit"),
                )
            alive.append(lease)

        self._save_state(alive)
        return alive

    def status(self) -> list[Lease]:
        """Get current lease states (refreshed)."""
        return self.refresh()

    def check_lease(self, lease: Lease, timeout: int = 15) -> bool:
        """Quick health-check a running lease.
        GPU lease: nvidia-smi. CPU lease: hostname (just verify srun works)."""
        if lease.state != "RUNNING":
            return False
        if lease.num_gpus == 0:
            try:
                r = self.slurm.run_on_lease(
                    job_id=lease.job_id,
                    command="hostname",
                    num_gpus=0,
                    timeout=timeout,
                )
                return r.returncode == 0 and r.stdout.strip() != ""
            except Exception:
                return False
        try:
            r = self.slurm.run_on_lease(
                job_id=lease.job_id,
                command="nvidia-smi --query-gpu=name --format=csv,noheader",
                num_gpus=1,
                timeout=timeout,
            )
            return r.returncode == 0 and r.stdout.strip() != ""
        except Exception:
            return False

    def test_lease(self, lease: Lease, timeout: int = 60) -> dict:
        """Thorough GPU test: nvidia-smi details + CUDA compute capability.
        Returns {ok: bool, nvidia_smi: {...}, cuda: {...}, errors: [...]}"""
        result = {"ok": False, "nvidia_smi": {}, "cuda": {}, "errors": []}
        if lease.state != "RUNNING":
            result["errors"].append(f"lease not running (state={lease.state})")
            return result

        # Test 1: nvidia-smi
        try:
            r = self.slurm.run_on_lease(
                job_id=lease.job_id,
                command=(
                    "nvidia-smi --query-gpu=name,memory.total,memory.free,"
                    "driver_version,temperature.gpu,utilization.gpu"
                    " --format=csv,noheader,nounits"
                ),
                num_gpus=lease.num_gpus,
                timeout=timeout,
            )
            if r.returncode != 0:
                # Filter SSH warnings from error
                err = "\n".join(
                    l for l in r.stderr.strip().splitlines()
                    if not l.startswith("**")
                )
                result["errors"].append(f"nvidia-smi failed: {err[:200]}" if err else
                                        f"nvidia-smi failed (rc={r.returncode}): {r.stdout.strip()[:200]}")
            else:
                gpus = []
                for line in r.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        gpus.append({
                            "name": parts[0],
                            "mem_total_mb": parts[1],
                            "mem_free_mb": parts[2],
                            "driver": parts[3],
                            "temp_c": parts[4],
                            "util_pct": parts[5],
                        })
                result["nvidia_smi"] = {"gpus": gpus, "count": len(gpus)}
        except Exception as e:
            result["errors"].append(f"nvidia-smi exception: {e}")

        # Test 2: torch CUDA
        # Write script inline via the srun command itself.
        # Uses echo to create the file on the compute node, then runs it.
        # Test 2: CUDA compute test (no torch needed)
        # Uses nvidia-cuda-mps or a simple deviceQuery-style check
        cuda_cmd = (
            "nvidia-smi -L; "
            "echo '---'; "
            "nvidia-smi --query-gpu=compute_cap --format=csv,noheader"
        )
        try:
            r = self.slurm.run_on_lease(
                job_id=lease.job_id,
                command=cuda_cmd,
                num_gpus=lease.num_gpus,
                timeout=timeout,
            )
            if r.returncode != 0:
                err = "\n".join(
                    l for l in r.stderr.strip().splitlines()
                    if not l.startswith("**") and "post-quantum" not in l
                ).strip()
                result["cuda"] = {"ok": False, "error": err if err else "unknown"}
            else:
                lines = r.stdout.strip().splitlines()
                sep = next((i for i, l in enumerate(lines) if l.strip() == "---"), len(lines))
                gpu_list = [l.strip() for l in lines[:sep] if l.strip()]
                compute_caps = [l.strip() for l in lines[sep+1:] if l.strip()]
                result["cuda"] = {
                    "ok": len(gpu_list) > 0,
                    "gpu_list": gpu_list,
                    "compute_caps": compute_caps,
                }
        except Exception as e:
            result["cuda"] = {"ok": False, "error": str(e)}

        # Overall OK: nvidia-smi found GPUs, no critical errors
        has_gpus = result["nvidia_smi"].get("count", 0) > 0
        result["ok"] = has_gpus and not result["errors"]
        return result

    # ── GPU process inspection / cleanup ──

    def _build_gpu_inspect_script(self, lease_job_id: int) -> str:
        """Shell script that captures three things on the compute node:
        (1) currently-live step IDs for this lease via `squeue -s -j LEASE`,
        (2) all PIDs Slurm tracks in the lease's cgroup with their step ID
            via `scontrol listpids LEASE`,
        (3) GPU-holding processes from `nvidia-smi --query-compute-apps`,
        each filtered to our UID.

        Output is bracketed with sentinel markers so the parser is robust
        to MOTD / nvidia-smi warnings."""
        return (
            'echo __AL_STEPS_BEGIN__; '
            f'squeue -s -j {lease_job_id} --noheader -o "%i %j" 2>/dev/null '
            f'|| true; '
            'echo __AL_STEPS_END__; '
            'echo __AL_PIDS_BEGIN__; '
            f'scontrol listpids {lease_job_id} 2>/dev/null || true; '
            'echo __AL_PIDS_END__; '
            'MY_UID=$(id -u); '
            'nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory '
            '--format=csv,noheader,nounits 2>/dev/null | '
            'while IFS=, read -r pid gpu mem; do '
            '  pid=$(echo $pid | tr -d " "); '
            '  [ -z "$pid" ] && continue; '
            '  uid=$(stat -c %u /proc/$pid 2>/dev/null); '
            '  [ "$uid" != "$MY_UID" ] && continue; '
            '  cmd=$(tr "\\0" " " < /proc/$pid/cmdline 2>/dev/null | head -c 120); '
            '  echo "__AL_PROC__$pid|$gpu|$mem|$cmd"; '
            'done'
        )

    @staticmethod
    def _parse_gpu_inspect(stdout: str
                           ) -> tuple[Optional[set[str]],
                                      Optional[dict[int, str]],
                                      dict[str, str],
                                      list[dict]]:
        """Parse output of _build_gpu_inspect_script. Returns
        (live_steps, pid_to_step, step_name_by_id, procs).

        - live_steps: set of step IDs (strings — squeue's `%i` is e.g.
          "1451430.0", we keep just the part after the dot). None if the
          STEPS section never appeared (scan failed before squeue ran).
        - pid_to_step: {pid: step_id_string}. None if the PIDS section
          never appeared. Empty dict if section appeared with no rows.
        - step_name_by_id: {step_id: step_name} from squeue's `%i %j`,
          used to cross-reference against autolease's own job records.
        - procs: GPU processes from nvidia-smi.

        Step IDs are normalized: "1451430.0" -> "0", "1451430.batch" ->
        "batch". This matches the STEPID column from scontrol listpids."""
        live_steps: set[str] = set()
        step_name_by_id: dict[str, str] = {}
        pid_to_step: dict[int, str] = {}
        procs: list[dict] = []
        steps_seen = False
        pids_seen = False
        section = None  # 'steps' | 'pids' | None
        for line in stdout.splitlines():
            line = line.rstrip("\r")
            if line == "__AL_STEPS_BEGIN__":
                section = "steps"
                steps_seen = True
                continue
            if line == "__AL_STEPS_END__":
                section = None
                continue
            if line == "__AL_PIDS_BEGIN__":
                section = "pids"
                pids_seen = True
                continue
            if line == "__AL_PIDS_END__":
                section = None
                continue
            if section == "steps":
                stripped = line.strip()
                if not stripped or stripped.upper().startswith("STEPID"):
                    continue
                parts = stripped.split(None, 1)
                if not parts:
                    continue
                step_field = parts[0]
                step_id = (step_field.split(".", 1)[1]
                           if "." in step_field else step_field)
                live_steps.add(step_id)
                if len(parts) >= 2:
                    step_name_by_id[step_id] = parts[1].strip()
                continue
            if section == "pids":
                # scontrol listpids columns: PID JOBID STEPID LOCALID GLOBALID
                stripped = line.strip()
                if not stripped or stripped.upper().startswith("PID"):
                    continue
                parts = stripped.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    try:
                        pid = int(parts[0])
                    except ValueError:
                        continue
                    pid_to_step[pid] = parts[2]
                continue
            if line.startswith("__AL_PROC__"):
                payload = line[len("__AL_PROC__"):]
                fields = payload.split("|", 3)
                if len(fields) < 4:
                    continue
                pid_s, gpu, mem_s, cmd = fields
                try:
                    pid = int(pid_s)
                    mem = int(mem_s)
                except ValueError:
                    continue
                procs.append({
                    "pid": pid, "gpu_uuid": gpu.strip(),
                    "memory_mb": mem, "cmdline": cmd.strip(),
                })
        return (live_steps if steps_seen else None,
                pid_to_step if pids_seen else None,
                step_name_by_id,
                procs)

    @staticmethod
    def _classify_pid(pid: int,
                      live_steps: Optional[set[str]],
                      pid_to_step: Optional[dict[int, str]],
                      step_name_by_id: Optional[dict[str, str]] = None,
                      stale_step_names: Optional[set[str]] = None) -> str:
        """Decide live / orphan / unknown for a single GPU PID.

        - 'live': PID's step is in `squeue -s` and the step's name is NOT
                  in stale_step_names. Or PID is in lease's batch step.
        - 'orphan': PID's step is gone from squeue (step ended but PID
                    survived); OR step's name matches an autolease job
                    that's marked done/failed (Slurm hasn't reaped the
                    step yet but autolease knows the work is over); OR
                    PID isn't in scontrol listpids at all (escaped cgroup);
                    OR PID is in 'extern' (adopted, typically not real work).
        - 'unknown': we couldn't gather either lookup table — don't act.

        stale_step_names: step names of autolease jobs whose state is
        done/failed. Caller supplies these from the local job records;
        we treat any PID running under such a step as orphan even if
        Slurm still lists the step as alive.
        """
        if live_steps is None or pid_to_step is None:
            return "unknown"
        step = pid_to_step.get(pid)
        if step is None:
            return "orphan"
        if step == "batch":
            return "live"  # the lease's sleep wrapper — never kill
        if step == "extern":
            # Adopted PIDs (e.g. survivors re-parented after their step
            # died). Real running work doesn't normally land here.
            return "orphan"
        # Numeric step. Cross-reference name against autolease records.
        if step_name_by_id and stale_step_names:
            name = step_name_by_id.get(step)
            if name and name in stale_step_names:
                return "orphan"
        if step in live_steps:
            return "live"
        return "orphan"

    def _run_on_lease_sh(self, lease: Lease, script: str,
                         timeout: int) -> tuple[int, str, str]:
        """Run a POSIX /bin/sh script inside a lease via srun --overlap.
        Bypasses Slurm.run_on_lease (which uses config.shell + heredocs —
        breaks on fish/zsh login shells with bash-style scripts).
        Returns (returncode, stdout, stderr)."""
        gres = (f" --gres=gpu:{lease.num_gpus}" if lease.num_gpus > 0 else "")
        srun = f"srun --jobid={lease.job_id}{gres} --overlap"
        cmd = f"{srun} /bin/sh -c {shlex.quote(script)}"
        try:
            r = self.slurm.cfg.run(cmd, timeout=timeout)
        except Exception as e:
            return -1, "", f"ssh exception: {e}"
        return r.returncode, r.stdout, r.stderr

    def list_gpu_procs(self, lease: Lease, timeout: int = 30,
                       stale_step_names: Optional[set[str]] = None
                       ) -> tuple[list[GpuProc], str]:
        """Inspect GPU-holding processes on a lease's compute node.
        One SSH call. Classifies each PID as:
          - 'live':    PID is in scontrol listpids (part of an active step)
          - 'orphan':  PID exists but Slurm doesn't track it (step ended)
          - 'unknown': can't classify (scontrol listpids didn't run)

        Returns (procs, debug_message). debug_message is "" on success;
        otherwise contains stderr / hint when 0 procs were found, so callers
        can surface it to the user."""
        if lease.state != "RUNNING":
            return [], f"lease state={lease.state}"
        if lease.num_gpus == 0:
            return [], "CPU-only lease, no GPUs to inspect"
        rc, stdout, stderr = self._run_on_lease_sh(
            lease, self._build_gpu_inspect_script(lease.job_id),
            timeout=timeout,
        )
        if rc != 0:
            stderr_short = (stderr or "").strip()[:200]
            return [], f"srun rc={rc}: {stderr_short or '(no stderr)'}"
        live_steps, pid_to_step, step_name_by_id, procs = \
            self._parse_gpu_inspect(stdout)
        out: list[GpuProc] = []
        for p in procs:
            status = self._classify_pid(
                p["pid"], live_steps, pid_to_step,
                step_name_by_id=step_name_by_id,
                stale_step_names=stale_step_names,
            )
            step = (pid_to_step.get(p["pid"])
                    if pid_to_step is not None else None)
            step_name = step_name_by_id.get(step) if step else None
            out.append(GpuProc(
                lease_job_id=lease.job_id,
                node=lease.node,
                pid=p["pid"],
                gpu_uuid=p["gpu_uuid"],
                memory_mb=p["memory_mb"],
                cmdline=p["cmdline"],
                status=status,
                step=step,
                step_name=step_name,
            ))
        if not out:
            stderr_short = (stderr or "").strip()[:200]
            hint = (
                "nvidia-smi returned no compute apps. "
                "Either no processes are using the GPU (lease is genuinely "
                "free) or nvidia-smi was restricted by cgroups. "
                f"stderr: {stderr_short or '(empty)'}"
            )
            return [], hint
        return out, ""

    def kill_gpu_procs(self, lease: Lease, pids: list[int],
                       timeout: int = 30) -> dict[int, bool]:
        """SIGTERM then SIGKILL the given PIDs inside the lease's compute node.
        Returns {pid: True/False} for each PID (True = no longer holding GPU
        memory after kill). Caller is responsible for filtering to 'orphan' /
        'unknown' PIDs — this method blindly kills whatever it's given."""
        if lease.state != "RUNNING" or not pids:
            return {pid: False for pid in pids}
        pid_args = " ".join(str(p) for p in pids)
        script = (
            f"for p in {pid_args}; do kill $p 2>/dev/null; done; "
            f"sleep 2; "
            f"for p in {pid_args}; do kill -9 $p 2>/dev/null; done; "
            f"sleep 1; "
            f"echo __AL_AFTER__; "
            f"nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits "
            f"2>/dev/null | tr -d ' '"
        )
        rc, stdout, _stderr = self._run_on_lease_sh(lease, script, timeout=timeout)
        if rc != 0:
            return {pid: False for pid in pids}
        after = set()
        in_after = False
        for line in stdout.splitlines():
            line = line.strip()
            if line == "__AL_AFTER__":
                in_after = True
                continue
            if in_after and line.isdigit():
                after.add(int(line))
        return {pid: (pid not in after) for pid in pids}

    def wait_and_check(self, lease: Lease, poll_interval: int = 5,
                       max_wait: int = 120) -> bool:
        """Wait for a lease to start running, then health-check it."""
        waited = 0
        while waited < max_wait:
            info = self.slurm.job_info(lease.job_id)
            state = info.get("state", "GONE")
            if state == "RUNNING":
                lease.state = "RUNNING"
                lease.node = info.get("node")
                lease.end_time = info.get("end_time")
                return self.check_lease(lease)
            if state in ("COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "GONE"):
                return False
            time.sleep(poll_interval)
            waited += poll_interval
        return False

    # ── Bad-node detection & re-leasing ──

    def check_and_replace(self) -> list[dict]:
        """Health-check all running leases. Replace bad ones.
        Returns a list of action dicts for reporting."""
        leases = self.refresh()
        actions = []
        replacements = []

        for lease in leases:
            if lease.state != "RUNNING":
                actions.append({
                    "job_id": lease.job_id, "node": lease.node,
                    "action": "skip", "reason": lease.state,
                })
                continue

            ok = self.check_lease(lease)
            if ok:
                actions.append({
                    "job_id": lease.job_id, "node": lease.node,
                    "action": "ok",
                })
                continue

            # Bad node detected
            node = lease.node or "unknown"
            actions.append({
                "job_id": lease.job_id, "node": node,
                "action": "bad", "reason": "health check failed",
            })

            # Add to dynamic bad-node list
            if node != "unknown":
                self._add_bad_node(node)

            # Cancel the bad lease
            try:
                self.slurm.cancel_job(lease.job_id)
            except RuntimeError:
                pass

            # Acquire replacement
            replacement = self._acquire_replacement(lease)
            if replacement:
                replacements.append(replacement)
                actions.append({
                    "job_id": replacement.job_id, "node": None,
                    "action": "replacement", "reason": f"replacing {lease.job_id}",
                })

        # Update state: remove bad leases, add replacements
        if replacements:
            current = self.refresh()  # re-read to get clean state
            current.extend(replacements)
            self._save_state(current)

        return actions

    def bad_nodes(self) -> list[str]:
        """Return all known bad nodes (config + dynamic)."""
        return sorted(set(self.config.exclude_nodes) | set(self._get_bad_nodes()))

    def clear_bad_nodes(self):
        """Reset the dynamic bad-node list (keeps config excludes)."""
        self._save_state(self._get_leases(), [])

    # ── Lease renewal ──

    def remaining_minutes(self, lease: Lease) -> Optional[float]:
        """Estimate minutes remaining on a lease."""
        if not lease.end_time:
            return None
        end = _parse_slurm_time(lease.end_time)
        if end is None:
            return None
        now = datetime.now()
        delta = (end - now).total_seconds() / 60.0
        return max(0.0, delta)

    def renew(self, threshold_minutes: float = 30.0) -> list[dict]:
        """Check all running leases. If any are within threshold_minutes of
        expiry, submit a replacement and cancel the old one after the
        replacement starts running.
        Returns a list of action dicts for reporting."""
        leases = self.refresh()
        actions = []

        for lease in leases:
            if lease.state != "RUNNING":
                continue

            remaining = self.remaining_minutes(lease)
            if remaining is None:
                actions.append({
                    "job_id": lease.job_id, "action": "skip",
                    "reason": "cannot determine remaining time",
                })
                continue

            if remaining > threshold_minutes:
                actions.append({
                    "job_id": lease.job_id, "action": "ok",
                    "remaining_min": round(remaining, 1),
                })
                continue

            # Needs renewal
            actions.append({
                "job_id": lease.job_id, "action": "renewing",
                "remaining_min": round(remaining, 1),
            })

            replacement = self._acquire_replacement(lease)
            if replacement is None:
                actions.append({
                    "job_id": lease.job_id, "action": "renew_failed",
                    "reason": "could not submit replacement",
                })
                continue

            # Wait for replacement to start (up to 60s), then cancel old
            ok = self.wait_and_check(replacement, poll_interval=3, max_wait=60)
            if ok:
                try:
                    self.slurm.cancel_job(lease.job_id)
                except RuntimeError:
                    pass
                actions.append({
                    "job_id": replacement.job_id,
                    "node": replacement.node,
                    "action": "renewed",
                    "reason": f"replaced {lease.job_id}",
                })
            else:
                # Replacement didn't start in time; keep both for now
                actions.append({
                    "job_id": replacement.job_id,
                    "action": "renewal_pending",
                    "reason": "replacement not running yet, keeping both",
                })

            # Save replacement into state
            current = self._get_leases()
            current.append(replacement)
            self._save_state(current)

        return actions

    # ── Finding & running ──

    def find_running_lease(self, gpu_type: Optional[str] = None,
                           min_gpus: int = 1) -> Optional[Lease]:
        """Find a running lease matching requirements.
        min_gpus > 0: prefers smallest GPU lease that fits.
        min_gpus == 0: prefers CPU leases first, then smallest GPU lease (piggyback)."""
        from .config import GPU_VRAM
        if min_gpus == 0:
            leases = sorted(
                self.refresh(),
                key=lambda l: (0 if l.num_gpus == 0 else 1,
                               GPU_VRAM.get(l.gpu_type, 0)),
            )
        else:
            leases = sorted(self.refresh(),
                            key=lambda l: GPU_VRAM.get(l.gpu_type, 0))
        for lease in leases:
            if lease.state != "RUNNING":
                continue
            if gpu_type and lease.gpu_type.lower() != gpu_type.lower():
                continue
            if min_gpus > 0 and lease.num_gpus < min_gpus:
                continue
            return lease
        return None

    def run_on(self, command: str, gpu_type: Optional[str] = None,
               num_gpus: int = 1, timeout: int = 600) -> tuple[int, str, str]:
        """Find a suitable lease and run a command on it.
        Returns (returncode, stdout, stderr)."""
        lease = self.find_running_lease(gpu_type=gpu_type, min_gpus=num_gpus)
        if lease is None:
            available = self.refresh()
            running = [l for l in available if l.state == "RUNNING"]
            pending = [l for l in available if l.state == "PENDING"]
            msg = f"No running lease found"
            if gpu_type:
                msg += f" for gpu_type={gpu_type}"
            msg += f". Have {len(running)} running, {len(pending)} pending."
            raise RuntimeError(msg)

        r = self.slurm.run_on_lease(
            job_id=lease.job_id,
            command=command,
            num_gpus=num_gpus,
            timeout=timeout,
        )
        return r.returncode, r.stdout, r.stderr
