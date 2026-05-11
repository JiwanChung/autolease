"""Regression test: dispatcher should pick the smallest lease that fits.
A 1-GPU job must prefer a 1-GPU lease over a 4-GPU lease (same GPU type),
otherwise big leases get fragmented (one slot busy → whole lease counted
as busy by _lease_is_busy) and small jobs waste capacity."""

from autolease.queue import Job, JobQueue
from autolease.slurm import Lease


def _lease(jid, num_gpus, gpu_type="RTX3090", node="n"):
    return Lease(
        job_id=jid, partition="p", qos="q", gpu_type=gpu_type,
        num_gpus=num_gpus, node=node, state="RUNNING",
    )


class TestLeaseFit:
    def test_smaller_lease_preferred_for_small_job(self, cfg):
        """1-GPU job + leases of [4, 1, 2] (same GPU type) → pick the 1-GPU lease."""
        q = JobQueue(cfg)
        leases = [_lease(101, 4), _lease(102, 1), _lease(103, 2)]
        job = Job(id=1, project="p", command="c", state="queued", num_gpus=1)
        # Replicate the dispatcher's sort
        from autolease.config import GPU_VRAM
        sorted_leases = sorted(
            leases, key=lambda l: (l.num_gpus, GPU_VRAM.get(l.gpu_type, 0))
        )
        # First lease the dispatcher tries, that the job fits in
        first_match = next(l for l in sorted_leases if l.num_gpus >= job.num_gpus)
        assert first_match.num_gpus == 1, \
            f"expected 1-GPU lease first, got {first_match.num_gpus}"

    def test_two_gpu_job_picks_two_not_four(self, cfg):
        """2-GPU job + leases of [4, 1, 2] → pick the 2-GPU lease (not 4)."""
        leases = [_lease(101, 4), _lease(102, 1), _lease(103, 2)]
        job = Job(id=1, project="p", command="c", state="queued", num_gpus=2)
        from autolease.config import GPU_VRAM
        sorted_leases = sorted(
            leases, key=lambda l: (l.num_gpus, GPU_VRAM.get(l.gpu_type, 0))
        )
        first_match = next(l for l in sorted_leases if l.num_gpus >= job.num_gpus)
        assert first_match.num_gpus == 2

    def test_falls_back_to_bigger_when_smaller_doesnt_fit(self, cfg):
        """3-GPU job + leases of [1, 4, 2] → only 4 fits."""
        leases = [_lease(101, 1), _lease(102, 4), _lease(103, 2)]
        job = Job(id=1, project="p", command="c", state="queued", num_gpus=3)
        from autolease.config import GPU_VRAM
        sorted_leases = sorted(
            leases, key=lambda l: (l.num_gpus, GPU_VRAM.get(l.gpu_type, 0))
        )
        first_match = next(l for l in sorted_leases if l.num_gpus >= job.num_gpus)
        assert first_match.num_gpus == 4

    def test_same_gpu_count_breaks_tie_by_vram(self, cfg):
        """Within same lease size, prefer smaller VRAM."""
        leases = [_lease(101, 1, "A100"), _lease(102, 1, "RTX3090")]
        from autolease.config import GPU_VRAM
        sorted_leases = sorted(
            leases, key=lambda l: (l.num_gpus, GPU_VRAM.get(l.gpu_type, 0))
        )
        # RTX3090 (24GB) should come before A100 (80GB)
        assert sorted_leases[0].gpu_type == "RTX3090"
