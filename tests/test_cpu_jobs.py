"""CPU-only support: `-n 0` jobs / leases.

CPU jobs are signalled by `num_gpus == 0`. CPU leases acquire allocations
without `--gres=gpu` and carry `num_gpus=0, gpu_type="cpu"`. CPU jobs prefer
CPU leases but may piggyback on GPU leases; GPU jobs never land on CPU leases.
A lease can host at most one CPU job + at most one GPU job at the same time.
"""

from dataclasses import dataclass

from autolease.config import GPU_VRAM
from autolease.queue import Job, JobQueue
from autolease.slurm import Lease, Slurm, SlurmConfig


@dataclass
class _Resp:
    args: list
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class _Stub:
    """Local stub that records sbatch/srun commands."""
    def __init__(self, stdout: str = ""):
        self.calls: list[str] = []
        self.ssh_host = "stub"
        self.shell = "bash"
        self.ssh_opts = ()
        self._stdout = stdout

    def run(self, cmd: str, timeout: int = 30):
        self.calls.append(cmd)
        return _Resp([cmd], stdout=self._stdout)


def _gpu_lease(jid, num_gpus, gpu_type="RTX3090"):
    return Lease(
        job_id=jid, partition="p", qos="q", gpu_type=gpu_type,
        num_gpus=num_gpus, node=f"n{jid}", state="RUNNING",
    )


def _cpu_lease(jid):
    return Lease(
        job_id=jid, partition="cpu_part", qos="q", gpu_type="cpu",
        num_gpus=0, node=f"n{jid}", state="RUNNING",
    )


class TestLeaseMatching:
    def test_cpu_job_matches_cpu_lease(self, cfg):
        q = JobQueue(cfg)
        job = Job(id=1, project="p", command="c", state="queued", num_gpus=0)
        assert q._lease_matches(_cpu_lease(10), job) is True

    def test_cpu_job_matches_gpu_lease_piggyback(self, cfg):
        q = JobQueue(cfg)
        job = Job(id=1, project="p", command="c", state="queued", num_gpus=0)
        assert q._lease_matches(_gpu_lease(10, 1), job) is True

    def test_gpu_job_rejects_cpu_lease(self, cfg):
        q = JobQueue(cfg)
        job = Job(id=1, project="p", command="c", state="queued", num_gpus=1)
        assert q._lease_matches(_cpu_lease(10), job) is False

    def test_cpu_job_with_explicit_gpu_type_rejects_cpu_lease(self, cfg):
        """`-n 0 -g a100` is unusual but valid: pin the lease to a specific
        node by GPU type, run CPU-only work there. A CPU lease can't satisfy
        an explicit gpu_type filter."""
        q = JobQueue(cfg)
        job = Job(id=1, project="p", command="c", state="queued",
                  num_gpus=0, gpu_type="A100")
        assert q._lease_matches(_cpu_lease(10), job) is False
        assert q._lease_matches(_gpu_lease(11, 1, "A100"), job) is True
        assert q._lease_matches(_gpu_lease(12, 1, "RTX3090"), job) is False

    def test_gpu_job_count_check_unchanged(self, cfg):
        """Regression: GPU job count filter still applies."""
        q = JobQueue(cfg)
        job = Job(id=1, project="p", command="c", state="queued", num_gpus=2)
        assert q._lease_matches(_gpu_lease(10, 1), job) is False
        assert q._lease_matches(_gpu_lease(11, 4), job) is True


class TestLeaseBusy:
    def test_cpu_and_gpu_job_coexist_on_same_lease(self, cfg):
        """CPU + GPU work on the same lease via --overlap (disjoint resources)."""
        q = JobQueue(cfg)
        lease = _gpu_lease(10, 1)
        gpu_running = Job(id=1, project="p", command="c", state="running",
                          num_gpus=1, lease_job_id=10)
        # Dispatching a CPU job: lease is NOT busy for CPU jobs
        assert q._lease_is_busy(lease, [gpu_running], for_cpu_job=True) is False
        # Dispatching a GPU job: lease IS busy
        assert q._lease_is_busy(lease, [gpu_running], for_cpu_job=False) is True

    def test_two_cpu_jobs_on_same_lease_busy(self, cfg):
        q = JobQueue(cfg)
        lease = _gpu_lease(10, 1)
        cpu_running = Job(id=1, project="p", command="c", state="running",
                          num_gpus=0, lease_job_id=10)
        assert q._lease_is_busy(lease, [cpu_running], for_cpu_job=True) is True
        assert q._lease_is_busy(lease, [cpu_running], for_cpu_job=False) is False


class TestLeaseOrdering:
    def test_cpu_job_prefers_cpu_lease_first(self, cfg):
        """CPU job with both CPU and GPU leases available → pick CPU lease."""
        leases = [_gpu_lease(101, 1), _cpu_lease(102), _gpu_lease(103, 2)]
        # Replicate dispatcher's per-job sort for CPU jobs
        sorted_leases = sorted(
            leases,
            key=lambda l: (0 if l.num_gpus == 0 else 1, l.num_gpus,
                           GPU_VRAM.get(l.gpu_type, 0)),
        )
        assert sorted_leases[0].num_gpus == 0
        assert sorted_leases[0].job_id == 102

    def test_cpu_job_falls_back_to_smallest_gpu(self, cfg):
        """No CPU lease available → CPU job picks smallest GPU lease."""
        leases = [_gpu_lease(101, 4), _gpu_lease(102, 1), _gpu_lease(103, 2)]
        sorted_leases = sorted(
            leases,
            key=lambda l: (0 if l.num_gpus == 0 else 1, l.num_gpus,
                           GPU_VRAM.get(l.gpu_type, 0)),
        )
        assert sorted_leases[0].num_gpus == 1


class TestSubmitHolder:
    def test_cpu_lease_sbatch_omits_gres(self, cfg):
        """`submit_holder(num_gpus=0)` builds sbatch without --gres=gpu."""
        stub = _Stub(stdout="42")
        s = Slurm(stub)
        s.submit_holder(partition="cpu_part", qos="normal", num_gpus=0,
                        cpus_per_task=4)
        sbatch_cmd = stub.calls[0]
        assert "--gres=gpu" not in sbatch_cmd
        assert "--cpus-per-task=4" in sbatch_cmd
        assert "--partition=cpu_part" in sbatch_cmd

    def test_gpu_lease_sbatch_keeps_gres(self, cfg):
        """Regression: GPU lease path still includes --gres=gpu:N."""
        stub = _Stub(stdout="42")
        s = Slurm(stub)
        s.submit_holder(partition="gpu_part", qos="normal", num_gpus=2)
        sbatch_cmd = stub.calls[0]
        assert "--gres=gpu:2" in sbatch_cmd


class TestLaunchRemoteSrun:
    def test_launch_remote_cpu_omits_gres(self, cfg):
        """_launch_remote builds the srun line without --gres for CPU jobs."""
        captured = {}

        def router(cmd, timeout=30):
            captured["cmd"] = cmd
            return _Resp([cmd], stdout="12345\n", returncode=0)

        q = JobQueue(cfg)
        q.slurm.cfg.run = router  # type: ignore
        job = Job(id=7, project="p", command="echo hi", state="queued",
                  num_gpus=0)
        lease = _cpu_lease(99)
        result = q._launch_remote(job, lease)
        assert result is not None
        sent = captured["cmd"]
        # The srun inside the generated launch.sh must NOT carry --gres
        assert "--gres=gpu" not in sent
        # Sanity: srun line is still there with --jobid + step name
        assert "srun --jobid=99" in sent
        assert "--job-name=autolease-job-7" in sent

    def test_launch_remote_gpu_keeps_gres(self, cfg):
        """Regression: GPU job path still includes --gres=gpu:N."""
        captured = {}

        def router(cmd, timeout=30):
            captured["cmd"] = cmd
            return _Resp([cmd], stdout="12345\n", returncode=0)

        q = JobQueue(cfg)
        q.slurm.cfg.run = router  # type: ignore
        job = Job(id=8, project="p", command="echo hi", state="queued",
                  num_gpus=2)
        lease = _gpu_lease(100, 4)
        result = q._launch_remote(job, lease)
        assert result is not None
        assert "--gres=gpu:2" in captured["cmd"]
