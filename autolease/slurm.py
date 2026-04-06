"""Low-level Slurm command wrappers. Runs commands via SSH or locally."""

import json
import subprocess
import shlex
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SlurmConfig:
    ssh_host: Optional[str] = None  # None = local
    ssh_opts: tuple = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")

    def run(self, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
        if self.ssh_host:
            full = ["ssh", *self.ssh_opts, self.ssh_host, cmd]
        else:
            full = ["bash", "-c", cmd]
        return subprocess.run(full, capture_output=True, text=True, timeout=timeout)


@dataclass
class NodeGPU:
    node: str
    gpu_type: str
    gpu_count: int
    mem_mb: int
    state: str


@dataclass
class Lease:
    job_id: int
    partition: str
    qos: str
    gpu_type: str
    num_gpus: int
    node: Optional[str] = None
    state: str = "PENDING"  # PENDING, RUNNING, GONE
    end_time: Optional[str] = None  # ISO timestamp from Slurm
    time_limit: Optional[str] = None  # e.g. "4:00:00"

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "partition": self.partition,
            "qos": self.qos,
            "gpu_type": self.gpu_type,
            "num_gpus": self.num_gpus,
            "node": self.node,
            "state": self.state,
            "end_time": self.end_time,
            "time_limit": self.time_limit,
        }

    @classmethod
    def from_dict(cls, d):
        # Handle state files that don't have new fields yet
        return cls(
            job_id=d["job_id"],
            partition=d["partition"],
            qos=d["qos"],
            gpu_type=d["gpu_type"],
            num_gpus=d["num_gpus"],
            node=d.get("node"),
            state=d.get("state", "PENDING"),
            end_time=d.get("end_time"),
            time_limit=d.get("time_limit"),
        )


class Slurm:
    def __init__(self, cfg: SlurmConfig):
        self.cfg = cfg

    def sinfo_gpus(self) -> list[NodeGPU]:
        r = self.cfg.run('sinfo -N -o "%N|%G|%m|%T" --noheader')
        if r.returncode != 0:
            raise RuntimeError(f"sinfo failed: {r.stderr}")
        nodes = []
        for line in r.stdout.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            name, gres, mem, state = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
            if not gres or gres == "(null)":
                continue
            # gres like gpu:RTX4090:6
            gparts = gres.split(":")
            if len(gparts) >= 3:
                gpu_type = gparts[1]
                gpu_count = int(gparts[2])
            else:
                continue
            nodes.append(NodeGPU(name, gpu_type, gpu_count, int(mem), state))
        return nodes

    def node_gpu_availability(self) -> list[dict]:
        """Get per-node GPU availability (total, allocated, free).
        Returns list of {node, partition, gpu_type, total, alloc, free, state}."""
        # Get partition membership
        r = self.cfg.run('sinfo -N -o "%N|%P|%G|%t" --noheader')
        if r.returncode != 0:
            return []
        node_partition = {}
        for line in r.stdout.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            node, part, gres, state = parts[0].strip(), parts[1].rstrip("*"), parts[2], parts[3]
            if gres == "(null)":
                continue
            node_partition[node] = {"partition": part, "state": state}

        # Get detailed alloc info from scontrol
        r = self.cfg.run('scontrol show node --oneliner', timeout=15)
        if r.returncode != 0:
            return []
        results = []
        for line in r.stdout.strip().splitlines():
            info = {}
            for token in line.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    info[k] = v
            node = info.get("NodeName", "")
            gres = info.get("Gres", "")
            alloc_tres = info.get("AllocTRES", "")
            state_raw = info.get("State", "")
            if not gres or gres == "(null)":
                continue
            # Parse total GPUs from Gres field (e.g. gpu:RTX3090:7)
            gparts = gres.split(":")
            if len(gparts) < 3:
                continue
            gpu_type = gparts[1]
            total = int(gparts[2])
            # Parse allocated GPUs from AllocTRES (e.g. gres/gpu=5)
            alloc = 0
            for item in alloc_tres.split(","):
                if item.startswith("gres/gpu="):
                    try:
                        alloc = int(item.split("=")[1])
                    except ValueError:
                        pass
                    break
            pinfo = node_partition.get(node, {})
            results.append({
                "node": node,
                "partition": pinfo.get("partition", "?"),
                "gpu_type": gpu_type,
                "total": total,
                "alloc": alloc,
                "free": total - alloc,
                "state": pinfo.get("state", state_raw),
            })
        results.sort(key=lambda x: (x["partition"], x["node"]))
        return results

    def partition_availability(self) -> dict[str, dict]:
        """Get per-partition GPU availability.
        Returns {partition: {gpu_type, total_gpus, idle_gpus, nodes_idle, nodes_mixed, nodes_alloc}}"""
        r = self.cfg.run('sinfo -o "%P|%G|%D|%t" --noheader')
        if r.returncode != 0:
            return {}
        partitions: dict[str, dict] = {}
        for line in r.stdout.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) < 4:
                continue
            part = parts[0].rstrip("*")
            gres = parts[1]
            num_nodes = int(parts[2])
            state = parts[3]
            if gres == "(null)":
                continue
            gparts = gres.split(":")
            if len(gparts) < 3:
                continue
            gpu_type = gparts[1]
            gpus_per_node = int(gparts[2])

            if part not in partitions:
                partitions[part] = {
                    "gpu_type": gpu_type,
                    "total_gpus": 0,
                    "idle_gpus": 0,
                    "nodes_idle": 0,
                    "nodes_mixed": 0,
                    "nodes_alloc": 0,
                }
            p = partitions[part]
            p["total_gpus"] += gpus_per_node * num_nodes
            if state == "idle":
                p["idle_gpus"] += gpus_per_node * num_nodes
                p["nodes_idle"] += num_nodes
            elif state.startswith("mix"):
                # Mixed = at least 1 GPU free per node (conservative estimate)
                p["idle_gpus"] += num_nodes  # at least 1 free per mixed node
                p["nodes_mixed"] += num_nodes
            elif state.startswith("alloc"):
                p["nodes_alloc"] += num_nodes
        return partitions

    def gpu_usage_by_qos(self) -> dict[str, int]:
        """Count how many GPUs the current user has allocated per QoS."""
        r = self.cfg.run(
            'squeue -u $(whoami) -o "%q|%b" --noheader --states=RUNNING,PENDING'
        )
        if r.returncode != 0:
            return {}
        usage: dict[str, int] = {}
        for line in r.stdout.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            qos = parts[0].strip()
            gres = parts[1].strip()
            # Parse gres like "gpu:2" or "gpu:RTX3090:2"
            gpus = 0
            if gres.startswith("gpu"):
                try:
                    gpus = int(gres.split(":")[-1])
                except ValueError:
                    pass
            usage[qos] = usage.get(qos, 0) + gpus
        return usage

    def submit_holder(self, partition: str, qos: str, num_gpus: int,
                      time: Optional[str] = None, exclude: str = "",
                      job_name: str = "autolease") -> int:
        """Submit a sleep job to hold GPUs. Returns job ID."""
        cmd = (
            f"sbatch --parsable"
            f" --partition={shlex.quote(partition)}"
            f" --qos={shlex.quote(qos)}"
            f" --gres=gpu:{num_gpus}"
            f" --job-name={shlex.quote(job_name)}"
            f" --output=/dev/null --error=/dev/null"
        )
        if time:
            cmd += f" --time={shlex.quote(time)}"
        if exclude:
            cmd += f" --exclude={shlex.quote(exclude)}"
        cmd += " --wrap 'sleep infinity'"
        r = self.cfg.run(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"sbatch failed: {r.stderr.strip()}")
        return int(r.stdout.strip().split(";")[0])

    def cancel_job(self, job_id: int):
        r = self.cfg.run(f"scancel {job_id}")
        if r.returncode != 0:
            raise RuntimeError(f"scancel {job_id} failed: {r.stderr.strip()}")

    def job_info(self, job_id: int) -> dict:
        """Get job state, node, and timing info via scontrol."""
        r = self.cfg.run(
            f"scontrol show job {job_id} --oneliner 2>/dev/null"
        )
        if r.returncode != 0 or not r.stdout.strip():
            return {"state": "GONE"}
        info = {}
        for token in r.stdout.strip().split():
            if "=" in token:
                k, v = token.split("=", 1)
                info[k] = v
        return {
            "state": info.get("JobState", "UNKNOWN"),
            "node": info.get("NodeList", None),
            "partition": info.get("Partition", ""),
            "num_gpus": info.get("NumCPUs", ""),
            "gres": info.get("Gres", ""),
            "end_time": info.get("EndTime", None),
            "start_time": info.get("StartTime", None),
            "time_limit": info.get("TimeLimit", None),
            "run_time": info.get("RunTime", None),
        }

    def run_on_lease(self, job_id: int, command: str, num_gpus: int = 1,
                     timeout: int = 600) -> subprocess.CompletedProcess:
        """Run a command inside a held allocation via srun --jobid."""
        srun_prefix = f"srun --jobid={job_id} --gres=gpu:{num_gpus} --overlap"
        # Pipe the user command through stdin to avoid quote-escaping hell
        # across SSH + bash -c layers.
        script = f"{srun_prefix} bash <<'__AUTOLEASE_EOF__'\n{command}\n__AUTOLEASE_EOF__"
        if self.cfg.ssh_host:
            full = ["ssh", *self.cfg.ssh_opts, self.cfg.ssh_host, script]
        else:
            full = ["bash", "-c", script]
        return subprocess.run(full, capture_output=True, text=True, timeout=timeout)

    def my_jobs(self, name_prefix: str = "autolease") -> list[dict]:
        """List the user's current jobs matching a name prefix."""
        r = self.cfg.run(
            f'squeue -u $(whoami) -o "%i|%j|%P|%T|%N|%b|%l" --noheader'
        )
        if r.returncode != 0:
            return []
        jobs = []
        for line in r.stdout.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) < 7:
                continue
            jid, name, part, state, node, gres, timelim = parts[:7]
            if name.startswith(name_prefix):
                jobs.append({
                    "job_id": int(jid),
                    "name": name,
                    "partition": part,
                    "state": state,
                    "node": node,
                    "gres": gres,
                    "timelimit": timelim,
                })
        return jobs
