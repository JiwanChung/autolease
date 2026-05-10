"""Tests for per-job history log."""

import os

from autolease.queue import JobQueue


class TestHistoryLog:
    def test_log_creates_file(self, cfg):
        q = JobQueue(cfg)
        q._log_job_history(1, "SUBMIT", project="test", priority=0)
        path = q._job_history_path(1)
        assert os.path.exists(path)

    def test_log_appends_in_order(self, cfg):
        q = JobQueue(cfg)
        q._log_job_history(1, "SUBMIT")
        q._log_job_history(1, "DISPATCH", node="node13")
        q._log_job_history(1, "DONE", exit_code=0)
        lines = q._read_job_history(1)
        assert len(lines) == 3
        assert "SUBMIT" in lines[0]
        assert "DISPATCH" in lines[1]
        assert "node=node13" in lines[1]
        assert "DONE" in lines[2]
        assert "exit_code=0" in lines[2]

    def test_read_empty_history(self, cfg):
        q = JobQueue(cfg)
        assert q._read_job_history(99999) == []

    def test_history_is_separate_per_job(self, cfg):
        q = JobQueue(cfg)
        q._log_job_history(1, "SUBMIT")
        q._log_job_history(2, "SUBMIT")
        q._log_job_history(1, "DONE")
        assert len(q._read_job_history(1)) == 2
        assert len(q._read_job_history(2)) == 1
