"""Tests for Job schema versioning + on-disk migration.
Without these, adding a field to Job risks the silent 'TypeError: unexpected
keyword argument' that we hit when an old TUI process saw new JSON files."""

from autolease.queue import Job, _migrate_job_dict, JOB_SCHEMA_VERSION


def _v0_dict():
    """A dict shaped like the very first on-disk format (no priority, no
    remote_cwd, no step_name, no schema_version)."""
    return {
        "id": 1, "project": "p", "command": "c", "state": "queued",
        "num_gpus": 1, "min_vram": 0, "gpu_type": None,
        "exit_code": None, "lease_job_id": None, "remote_pid": None,
        "node": None, "submitted": None, "started": None, "finished": None,
    }


class TestMigration:
    def test_v0_dict_loads_into_current_job(self):
        d = _migrate_job_dict(_v0_dict())
        # After migration, must be a complete dict for current dataclass
        j = Job(**d)
        assert j.priority == 0
        assert j.remote_cwd is None
        assert j.step_name is None
        assert j.schema_version == JOB_SCHEMA_VERSION

    def test_v1_dict_loads_into_current_job(self):
        # v1 had priority + remote_cwd but no step_name
        d = _v0_dict()
        d["priority"] = 5
        d["remote_cwd"] = "~/proj"
        d["schema_version"] = 1
        d = _migrate_job_dict(d)
        j = Job(**d)
        assert j.priority == 5
        assert j.remote_cwd == "~/proj"
        assert j.step_name is None
        assert j.schema_version == JOB_SCHEMA_VERSION

    def test_current_dict_is_idempotent(self):
        # Re-migrating an up-to-date dict shouldn't change anything
        d = _v0_dict()
        d1 = _migrate_job_dict(d)
        d2 = _migrate_job_dict(dict(d1))
        assert d1 == d2

    def test_migration_preserves_existing_fields(self):
        d = _v0_dict()
        d["state"] = "running"
        d["lease_job_id"] = 42
        d["remote_pid"] = 12345
        m = _migrate_job_dict(d)
        assert m["state"] == "running"
        assert m["lease_job_id"] == 42
        assert m["remote_pid"] == 12345

    def test_new_job_has_current_schema_version(self):
        j = Job(id=1, project="p", command="c", state="queued")
        assert j.schema_version == JOB_SCHEMA_VERSION
