"""Tests for low-level Slurm output parsing.
These would have caught: (a) the gres/gpu:N format bug that made
gpu_usage_by_qos always return 0 GPUs; (b) the GPU count regression
when squeue's %b format changes between Slurm versions."""

from autolease.slurm import _parse_gpu_count


class TestParseGpuCount:
    def test_plain_gpu_colon_n(self):
        assert _parse_gpu_count("gpu:2") == 2

    def test_gpu_with_type(self):
        assert _parse_gpu_count("gpu:RTX3090:2") == 2
        assert _parse_gpu_count("gpu:A100:1") == 1

    def test_gres_gpu_colon_n(self):
        # The format that caused the false-base_qos bug
        assert _parse_gpu_count("gres/gpu:4") == 4

    def test_gres_gpu_with_type_colon(self):
        assert _parse_gpu_count("gres/gpu:RTX3090:4") == 4

    def test_gres_gpu_equals_n(self):
        assert _parse_gpu_count("gres/gpu=4") == 4

    def test_gres_gpu_with_type_equals(self):
        assert _parse_gpu_count("gres/gpu:RTX3090=4") == 4

    def test_comma_separated_tres(self):
        # squeue can emit a full TRES string
        assert _parse_gpu_count("billing=1,cpu=1,gres/gpu:2,mem=8G") == 2
        assert _parse_gpu_count("billing=1,cpu=1,gres/gpu=2,mem=8G") == 2

    def test_empty_or_na(self):
        assert _parse_gpu_count("") == 0
        assert _parse_gpu_count("N/A") == 0

    def test_no_gpu_in_tres(self):
        assert _parse_gpu_count("cpu=1,mem=8G") == 0

    def test_multiple_gpu_entries_sum(self):
        # If somehow squeue lists multiple gres/gpu items, sum them
        assert _parse_gpu_count("gres/gpu=1,gres/gpu:RTX3090=2") == 3
