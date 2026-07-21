"""
Unit tests for src/metrics/memory.py

Tests with mocked CUDA API:
- reset calls correct device API
- bytes to GiB conversion
- allocated and reserved values not confused
- non-CUDA build explicitly fails
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from metrics.memory import MemoryMetrics, CudaMemoryTracker, is_cuda_oom


# ============================================================================
# MemoryMetrics tests
# ============================================================================


class TestMemoryMetrics:
    def test_create_valid_metrics(self):
        metrics = MemoryMetrics(
            peak_allocated_bytes=1024 * 1024 * 1024,  # 1 GiB
            peak_reserved_bytes=2 * 1024 * 1024 * 1024,  # 2 GiB
            peak_allocated_gb=1.0,
            peak_reserved_gb=2.0,
        )
        assert metrics.peak_allocated_bytes == 1024 ** 3
        assert metrics.peak_reserved_bytes == 2 * 1024 ** 3
        assert metrics.peak_allocated_gb == 1.0
        assert metrics.peak_reserved_gb == 2.0

    def test_bytes_to_gib_conversion(self):
        bytes_per_gb = 1024 ** 3
        metrics = MemoryMetrics(
            peak_allocated_bytes=int(1.5 * bytes_per_gb),
            peak_reserved_bytes=int(2.5 * bytes_per_gb),
            peak_allocated_gb=1.5,
            peak_reserved_gb=2.5,
        )
        assert metrics.peak_allocated_gb == pytest.approx(1.5, rel=1e-5)
        assert metrics.peak_reserved_gb == pytest.approx(2.5, rel=1e-5)

    def test_allocated_and_reserved_not_confused(self):
        """Ensure allocated and reserved values are distinct fields."""
        metrics = MemoryMetrics(
            peak_allocated_bytes=1000,
            peak_reserved_bytes=2000,
            peak_allocated_gb=1000 / (1024 ** 3),
            peak_reserved_gb=2000 / (1024 ** 3),
        )
        assert metrics.peak_allocated_bytes < metrics.peak_reserved_bytes
        assert metrics.peak_allocated_gb < metrics.peak_reserved_gb

    def test_frozen_dataclass(self):
        metrics = MemoryMetrics(
            peak_allocated_bytes=1024,
            peak_reserved_bytes=2048,
            peak_allocated_gb=1024 / (1024 ** 3),
            peak_reserved_gb=2048 / (1024 ** 3),
        )
        with pytest.raises(Exception):  # Frozen dataclass raises AttributeError or similar
            metrics.peak_allocated_bytes = 9999


# ============================================================================
# CudaMemoryTracker tests
# ============================================================================


class TestCudaMemoryTracker:
    def test_non_cuda_build_fails(self):
        """Non-CUDA device should raise ValueError."""
        cpu_device = torch.device("cpu")
        with pytest.raises(ValueError, match="requires a CUDA device"):
            CudaMemoryTracker(cpu_device)

    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.reset_peak_memory_stats")
    @patch("torch.cuda.max_memory_allocated")
    @patch("torch.cuda.max_memory_reserved")
    def test_reset_calls_correct_device_api(
        self, mock_max_reserved, mock_max_allocated, mock_reset, mock_sync
    ):
        """reset() should synchronize and reset peak stats."""
        cuda_device = torch.device("cuda")
        tracker = CudaMemoryTracker(cuda_device)

        mock_max_allocated.return_value = 1024 * 1024 * 512  # 512 MB
        mock_max_reserved.return_value = 1024 * 1024 * 1024  # 1 GB

        tracker.reset()

        mock_sync.assert_called_with(cuda_device)
        mock_reset.assert_called_with(cuda_device)

    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.reset_peak_memory_stats")
    def test_reset_does_not_call_empty_cache(self, mock_reset, mock_sync):
        """reset() should NOT call torch.cuda.empty_cache()."""
        cuda_device = torch.device("cuda")
        tracker = CudaMemoryTracker(cuda_device)

        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            tracker.reset()
            mock_empty_cache.assert_not_called()

    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.max_memory_allocated")
    @patch("torch.cuda.max_memory_reserved")
    def test_snapshot_returns_correct_values(
        self, mock_max_reserved, mock_max_allocated, mock_sync
    ):
        """snapshot() should return bytes and GiB values."""
        cuda_device = torch.device("cuda")
        tracker = CudaMemoryTracker(cuda_device)

        allocated_bytes = 512 * 1024 * 1024  # 512 MB
        reserved_bytes = 1024 * 1024 * 1024  # 1 GB

        mock_max_allocated.return_value = allocated_bytes
        mock_max_reserved.return_value = reserved_bytes

        metrics = tracker.snapshot()

        assert metrics.peak_allocated_bytes == allocated_bytes
        assert metrics.peak_reserved_bytes == reserved_bytes
        assert metrics.peak_allocated_gb == pytest.approx(0.5, rel=1e-5)
        assert metrics.peak_reserved_gb == pytest.approx(1.0, rel=1e-5)

        mock_sync.assert_called()

    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.max_memory_allocated")
    @patch("torch.cuda.max_memory_reserved")
    def test_snapshot_synchronizes_cuda(
        self, mock_max_reserved, mock_max_allocated, mock_sync
    ):
        """snapshot() should synchronize CUDA before reading stats."""
        cuda_device = torch.device("cuda")
        tracker = CudaMemoryTracker(cuda_device)

        mock_max_allocated.return_value = 1024
        mock_max_reserved.return_value = 2048

        tracker.snapshot()

        mock_sync.assert_called_with(cuda_device)


# ============================================================================
# is_cuda_oom tests
# ============================================================================


class TestIsCudaOom:
    def test_cuda_oom_error(self):
        error = RuntimeError("CUDA out of memory: tried to allocate 2.00 GiB")
        assert is_cuda_oom(error) is True

    def test_out_of_memory_error(self):
        error = RuntimeError("Out of memory")
        assert is_cuda_oom(error) is True

    def test_alloc_error(self):
        error = RuntimeError("Error in alloc")
        assert is_cuda_oom(error) is True

    def test_not_enough_memory_error(self):
        error = RuntimeError("Not enough memory for operation")
        assert is_cuda_oom(error) is True

    def test_non_oom_runtime_error(self):
        error = RuntimeError("Some unrelated error")
        assert is_cuda_oom(error) is False

    def test_non_runtime_error(self):
        error = ValueError("Value error")
        assert is_cuda_oom(error) is False

    def test_type_error(self):
        error = TypeError("Type error")
        assert is_cuda_oom(error) is False

    def test_cuda_error(self):
        error = RuntimeError("CUDA error: invalid argument")
        assert is_cuda_oom(error) is True

    def test_case_insensitive(self):
        error = RuntimeError("cuda OUT OF MEMORY")
        assert is_cuda_oom(error) is True

    def test_none_error(self):
        assert is_cuda_oom(None) is False
