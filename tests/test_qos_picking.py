"""Tests for pick_qos.
Would have caught: hardcoded base_qos return when partition not in
PARTITION_INFO, AllowQos=ALL handling, fallback when usage hits limit."""

import pytest

from autolease.config import (
    PARTITION_INFO, QOS_GPU_LIMITS, pick_qos,
)


@pytest.fixture(autouse=True)
def reset_globals():
    """Each test gets a clean global state."""
    PARTITION_INFO.clear()
    QOS_GPU_LIMITS.clear()
    yield
    PARTITION_INFO.clear()
    QOS_GPU_LIMITS.clear()


class TestPickQos:
    def test_picks_first_when_room(self):
        PARTITION_INFO["p"] = (["base_qos", "big_qos"], "RTX4090")
        QOS_GPU_LIMITS["base_qos"] = 8
        # 0 used + 1 requested ≤ 8 → base_qos
        assert pick_qos("p", 1, {}) == "base_qos"

    def test_falls_back_when_first_full(self):
        # The bug we hit: base_qos at 8/8 used to keep being picked
        PARTITION_INFO["p"] = (["base_qos", "big_qos"], "RTX4090")
        QOS_GPU_LIMITS["base_qos"] = 8
        # 8 used + 1 requested > 8 → fall back to big_qos
        assert pick_qos("p", 1, {"base_qos": 8}) == "big_qos"

    def test_unlimited_qos_always_picked(self):
        PARTITION_INFO["p"] = (["big_qos"], "RTX4090")
        # No limit configured → always pick (limit=0 means unlimited)
        assert pick_qos("p", 100, {"big_qos": 999}) == "big_qos"

    def test_partition_not_in_info_walks_configured_qos(self):
        # After the fix: when partition is unknown, walk QOS_GPU_LIMITS
        # ordered by limited-first
        QOS_GPU_LIMITS["base_qos"] = 8
        QOS_GPU_LIMITS["big_qos"] = 0  # unlimited
        # No PARTITION_INFO entry for "x" → fallback to configured
        result = pick_qos("x", 1, {})
        assert result == "base_qos"  # limited tier preferred

    def test_partition_not_in_info_falls_back_to_unlimited(self):
        QOS_GPU_LIMITS["base_qos"] = 8
        QOS_GPU_LIMITS["big_qos"] = 0
        # base_qos full → big_qos
        result = pick_qos("x", 1, {"base_qos": 8})
        assert result == "big_qos"

    def test_partition_not_in_info_no_config_returns_default(self):
        # No partition info, no QoS rules: defensive default
        assert pick_qos("x", 1, {}) == "base_qos"

    def test_partition_with_only_unlimited_qos(self):
        PARTITION_INFO["p"] = (["a100_qos"], "A100")
        # No limit → always pick
        assert pick_qos("p", 1, {}) == "a100_qos"

    def test_request_exactly_at_limit_succeeds(self):
        PARTITION_INFO["p"] = (["base_qos"], "RTX4090")
        QOS_GPU_LIMITS["base_qos"] = 8
        # 7 used + 1 requested = 8 ≤ 8 → still room
        assert pick_qos("p", 1, {"base_qos": 7}) == "base_qos"

    def test_request_exceeds_limit_falls_back(self):
        PARTITION_INFO["p"] = (["base_qos", "big_qos"], "RTX4090")
        QOS_GPU_LIMITS["base_qos"] = 8
        # 0 used + 9 requested > 8 → big_qos even with no current usage
        assert pick_qos("p", 9, {}) == "big_qos"
