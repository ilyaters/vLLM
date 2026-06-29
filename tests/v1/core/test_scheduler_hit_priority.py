# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for LMCache hit-priority admission with reduced watermark.

Tests that requests with external computed tokens (LMCache-hit) get a
reduced watermark (25% of full), allowing earlier admission while
preserving deadlock prevention. Requests without external tokens get
the full watermark.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import RequestStatus


def _make_manager(num_blocks: int, watermark: float) -> KVCacheManager:
    """Create a KVCacheManager for testing with controlled watermark.

    Args:
        num_blocks: number of GPU blocks in the pool.
        watermark: watermark fraction (0.0–1.0).

    Returns:
        A KVCacheManager instance with the specified configuration.
    """
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=4096,
        scheduler_block_size=16,
        hash_block_size=16,
        enable_caching=True,
        watermark=watermark,
    )
    return manager


class TestHitPriorityAdmission:
    """Test that requests with external computed tokens get reduced watermark."""

    def test_hit_request_reduced_watermark(self):
        """Request with num_external_computed_tokens > 0 gets 25% watermark."""
        # watermark=0.2, num_blocks=100 → watermark_blocks=20
        # hit → reduced to 20//4=5
        # free=15, required=8 → 8+5=13 < 15 → admitted
        manager = _make_manager(num_blocks=100, watermark=0.2)
        manager.block_pool.get_num_free_blocks = MagicMock(return_value=15)

        result = manager.allocate_slots(
            request=MagicMock(
                request_id="r1",
                status=RequestStatus.WAITING,
                num_tokens=100,
                num_prompt_tokens=100,
                num_computed_tokens=0,
            ),
            num_new_tokens=8,
            num_new_computed_tokens=0,
            num_external_computed_tokens=256,  # LMCache hit
            has_scheduled_reqs=True,
        )

        assert result is not None  # admitted with reduced watermark

    def test_miss_request_full_watermark(self):
        """Request with num_external_computed_tokens=0 gets full watermark."""
        manager = _make_manager(num_blocks=100, watermark=0.2)
        manager.block_pool.get_num_free_blocks = MagicMock(return_value=15)

        result = manager.allocate_slots(
            request=MagicMock(
                request_id="r1",
                status=RequestStatus.WAITING,
                num_tokens=100,
                num_prompt_tokens=100,
                num_computed_tokens=0,
            ),
            num_new_tokens=8,
            num_new_computed_tokens=0,
            num_external_computed_tokens=0,  # no hit
            has_scheduled_reqs=True,
        )

        assert result is None  # blocked by full watermark (8+20=28 > 15)

    def test_watermark_zero_no_regression(self):
        """When watermark=0, behavior unchanged."""
        manager = _make_manager(num_blocks=100, watermark=0.0)
        manager.block_pool.get_num_free_blocks = MagicMock(return_value=50)

        result = manager.allocate_slots(
            request=MagicMock(
                request_id="r1",
                status=RequestStatus.WAITING,
                num_tokens=100,
                num_prompt_tokens=100,
                num_computed_tokens=0,
            ),
            num_new_tokens=10,
            num_new_computed_tokens=0,
            num_external_computed_tokens=0,
            has_scheduled_reqs=True,
        )

        assert result is not None

    def test_hit_still_blocked_when_very_low_free(self):
        """Hit-request is still blocked when free is critically low
        (deadlock prevention)."""
        # watermark=0.2, num_blocks=100 → watermark_blocks=20
        # hit → reduced to 5
        # free=3, required=8 → 8+5=13 > 3 → blocked (prevents deadlock)
        manager = _make_manager(num_blocks=100, watermark=0.2)
        manager.block_pool.get_num_free_blocks = MagicMock(return_value=3)

        result = manager.allocate_slots(
            request=MagicMock(
                request_id="r1",
                status=RequestStatus.WAITING,
                num_tokens=100,
                num_prompt_tokens=100,
                num_computed_tokens=0,
            ),
            num_new_tokens=8,
            num_new_computed_tokens=0,
            num_external_computed_tokens=256,  # LMCache hit
            has_scheduled_reqs=True,
        )

        assert result is None  # blocked even with hit (deadlock prevention)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
