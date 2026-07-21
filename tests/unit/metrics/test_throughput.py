"""
Unit tests for src/metrics/throughput.py

Tests:
- tokens only accumulate after start
- elapsed time calculation
- zero elapsed time is rejected
- negative token count is rejected
- double start and early finish fail
"""

import pytest
import time
import torch
from unittest.mock import patch

from metrics.throughput import ThroughputMetrics, ThroughputTracker


# ============================================================================
# ThroughputMetrics tests
# ============================================================================


class TestThroughputMetrics:
    def test_create_valid_metrics(self):
        metrics = ThroughputMetrics(
            measured_tokens=1000,
            measured_seconds=2.0,
            tokens_per_second=500.0,
            warmup_optimizer_steps=3,
        )
        assert metrics.measured_tokens == 1000
        assert metrics.measured_seconds == 2.0
        assert metrics.tokens_per_second == 500.0
        assert metrics.warmup_optimizer_steps == 3

    def test_frozen_dataclass(self):
        metrics = ThroughputMetrics(
            measured_tokens=100,
            measured_seconds=1.0,
            tokens_per_second=100.0,
            warmup_optimizer_steps=0,
        )
        with pytest.raises(Exception):
            metrics.measured_tokens = 999


# ============================================================================
# ThroughputTracker tests
# ============================================================================


class TestThroughputTracker:
    def test_tokens_only_accumulate_after_start(self):
        """Tokens recorded before start() should be ignored."""
        device = torch.device("cpu")
        tracker = ThroughputTracker(device, warmup_optimizer_steps=0)

        # Record tokens before start - should be ignored
        tracker.record_tokens(100)
        assert tracker._measured_tokens == 0

        # Start and record again
        tracker.start()
        tracker.record_tokens(200)
        assert tracker._measured_tokens == 200

    def test_negative_token_count_rejected(self):
        """Negative token counts should raise ValueError."""
        device = torch.device("cpu")
        tracker = ThroughputTracker(device, warmup_optimizer_steps=0)
        tracker.start()

        with pytest.raises(ValueError, match="non-negative"):
            tracker.record_tokens(-1)

    def test_double_start_fails(self):
        """Calling start() twice should raise RuntimeError."""
        device = torch.device("cpu")
        tracker = ThroughputTracker(device, warmup_optimizer_steps=0)
        tracker.start()

        with pytest.raises(RuntimeError, match="called twice"):
            tracker.start()

    def test_finish_before_start_fails(self):
        """Calling finish() before start() should raise RuntimeError."""
        device = torch.device("cpu")
        tracker = ThroughputTracker(device, warmup_optimizer_steps=0)

        with pytest.raises(RuntimeError, match="called before start"):
            tracker.finish()

    def test_elapsed_time_calculation(self):
        """Elapsed time should be calculated correctly."""
        device = torch.device("cpu")
        tracker = ThroughputTracker(device, warmup_optimizer_steps=0)

        with patch("time.perf_counter") as mock_time:
            mock_time.side_effect = [100.0, 102.0]  # start at 100, finish at 102

            tracker.start()
            tracker.record_tokens(1000)
            metrics = tracker.finish()

            assert metrics.measured_tokens == 1000
            assert metrics.measured_seconds == pytest.approx(2.0, rel=1e-5)
            assert metrics.tokens_per_second == pytest.approx(500.0, rel=1e-5)

    def test_zero_elapsed_time_rejected(self):
        """Zero elapsed time should raise an error."""
        device = torch.device("cpu")
        tracker = ThroughputTracker(device, warmup_optimizer_steps=0)

        with patch("time.perf_counter") as mock_time:
            mock_time.return_value = 100.0  # Same time for start and finish

            tracker.start()
            tracker.record_tokens(1000)

            with pytest.raises((RuntimeError, ValueError)):
                tracker.finish()

    def test_warmup_steps_tracking(self):
        """Optimizer step counter should track warmup completion."""
        device = torch.device("cpu")
        tracker = ThroughputTracker(device, warmup_optimizer_steps=3)

        tracker.start()
        tracker.record_tokens(100)
        tracker.step()
        tracker.step()
        tracker.step()

        assert tracker._optimizer_steps == 3

    def test_multiple_token_accumulations(self):
        """Multiple record_tokens calls should accumulate."""
        device = torch.device("cpu")
        tracker = ThroughputTracker(device, warmup_optimizer_steps=0)
        tracker.start()

        tracker.record_tokens(100)
        tracker.record_tokens(200)
        tracker.record_tokens(300)

        assert tracker._measured_tokens == 600

    def test_cuda_synchronization_on_start(self):
        """start() should synchronize CUDA when using CUDA device."""
        cuda_device = torch.device("cuda")

        with patch("torch.cuda.synchronize") as mock_sync:
            tracker = ThroughputTracker(cuda_device, warmup_optimizer_steps=0)
            tracker.start()
            mock_sync.assert_called_with(cuda_device)

    def test_cuda_synchronization_on_finish(self):
        """finish() should synchronize CUDA when using CUDA device."""
        cuda_device = torch.device("cuda")

        with patch("torch.cuda.synchronize") as mock_sync:
            tracker = ThroughputTracker(cuda_device, warmup_optimizer_steps=0)
            tracker.start()
            tracker.record_tokens(100)
            tracker.finish()
            mock_sync.assert_called()

    def test_tokens_per_second_calculation(self):
        """tokens_per_second should be measured_tokens / measured_seconds."""
        device = torch.device("cpu")
        tracker = ThroughputTracker(device, warmup_optimizer_steps=0)

        with patch("time.perf_counter") as mock_time:
            mock_time.side_effect = [0.0, 5.0]

            tracker.start()
            tracker.record_tokens(5000)
            metrics = tracker.finish()

            assert metrics.tokens_per_second == pytest.approx(1000.0, rel=1e-5)
