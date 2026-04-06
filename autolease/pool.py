"""Pool manager — holds and tracks GPU leases."""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional

from .slurm import Slurm, SlurmConfig, Lease
from .config import PoolConfig, LeaseSpec


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
            return {"leases": [], "bad_nodes": []}
        with open(self._state_file) as f:
            data = json.load(f)
        # Migrate old format
        if isinstance(data, dict) and "leases" in data:
            data.setdefault("bad_nodes", [])
            return data
        return {"leases": [], "bad_nodes": []}

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
        job_id = self.slurm.submit_holder(
            partition=spec.partition,
            qos=spec.qos,
            num_gpus=spec.num_gpus,
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

    def release(self, job_id: int):
        """Cancel and remove a single lease."""
        self.slurm.cancel_job(job_id)
        leases = [l for l in self._get_leases() if l.job_id != job_id]
        self._save_state(leases)

    def down(self):
        """Cancel all held leases."""
        leases = self._get_leases()
        for lease in leases:
            try:
                self.slurm.cancel_job(lease.job_id)
            except RuntimeError:
                pass  # already gone
        self._save_state([])
        return len(leases)

    def refresh(self) -> list[Lease]:
        """Refresh lease states from Slurm. Adopt orphaned autolease jobs.
        Remove dead ones. Sets self.lost_leases for callers to check."""
        from .config import PARTITION_INFO
        leases = self._get_leases()
        known_ids = {l.job_id for l in leases}
        alive = []
        self.lost_leases = []

        # Adopt orphaned autolease jobs from squeue
        for sj in self.slurm.my_jobs("autolease"):
            if sj["job_id"] not in known_ids:
                # Parse gpu count from gres (e.g. "gpu:4" or "gpu:RTX3090:4")
                gres = sj.get("gres", "")
                num_gpus = 1
                if gres:
                    try:
                        num_gpus = int(gres.split(":")[-1])
                    except ValueError:
                        pass
                # Determine GPU type from partition info
                pinfo = PARTITION_INFO.get(sj["partition"])
                gpu_type = pinfo[1] if pinfo else "unknown"
                # Determine QoS from job info
                info = self.slurm.job_info(sj["job_id"])
                qos = ""
                if info.get("state") != "GONE":
                    # Parse QoS from scontrol output
                    r = self.slurm.cfg.run(
                        f"scontrol show job {sj['job_id']} --oneliner 2>/dev/null"
                    )
                    if r.returncode == 0:
                        for token in r.stdout.strip().split():
                            if token.startswith("QOS="):
                                qos = token.split("=", 1)[1]
                                break
                leases.append(Lease(
                    job_id=sj["job_id"],
                    partition=sj["partition"],
                    qos=qos,
                    gpu_type=gpu_type,
                    num_gpus=num_gpus,
                    node=sj.get("node") or None,
                    state=sj["state"],
                ))
                known_ids.add(sj["job_id"])

        # Now refresh all leases from Slurm
        for lease in leases:
            info = self.slurm.job_info(lease.job_id)
            state = info.get("state", "GONE")
            if state in ("COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "GONE", "UNKNOWN"):
                if lease.state == "RUNNING":
                    self.lost_leases.append(lease)
                continue
            lease.state = state
            lease.node = info.get("node")
            lease.end_time = info.get("end_time")
            lease.time_limit = info.get("time_limit") or lease.time_limit
            alive.append(lease)
        self._save_state(alive)
        return alive

    def status(self) -> list[Lease]:
        """Get current lease states (refreshed)."""
        return self.refresh()

    def check_lease(self, lease: Lease, timeout: int = 15) -> bool:
        """Quick health-check a running lease (nvidia-smi only)."""
        if lease.state != "RUNNING":
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
        """Find a running lease matching requirements. Prefers smallest GPU first."""
        from .config import GPU_VRAM
        leases = sorted(self.refresh(),
                        key=lambda l: GPU_VRAM.get(l.gpu_type, 0))
        for lease in leases:
            if lease.state != "RUNNING":
                continue
            if gpu_type and lease.gpu_type.lower() != gpu_type.lower():
                continue
            if lease.num_gpus < min_gpus:
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
