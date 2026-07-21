'''
This module provides GPU memory measurement utilities for training experiments.
'''

import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryMetrics:
    """Immutable container for CUDA memory statistics."""
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    peak_allocated_gb: float
    peak_reserved_gb: float


class CudaMemoryTracker:
    """
    Tracks peak CUDA memory allocation and reservation during training.

    Responsibilities:
    - require a CUDA device
    - store the target device
    - reset peak statistics at measurement start
    - snapshot peak values at measurement end
    """

    def __init__(self, device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError(
                f"CudaMemoryTracker requires a CUDA device, got {device.type}. "
                "Use CPU components for unit tests only."
            )
        self._device = device

    def reset(self) -> None:
        """
        Synchronize CUDA and reset peak memory statistics.

        Establishes the start of the measured training region.
        Does NOT call torch.cuda.empty_cache() to avoid changing runtime behavior.
        """
        torch.cuda.synchronize(self._device)
        torch.cuda.reset_peak_memory_stats(self._device)

    def snapshot(self) -> MemoryMetrics:
        """
        Synchronize CUDA and read peak memory statistics.

        Returns both bytes and GiB values for allocated and reserved memory.
        """
        torch.cuda.synchronize(self._device)

        peak_allocated = torch.cuda.max_memory_allocated(self._device)
        peak_reserved = torch.cuda.max_memory_reserved(self._device)

        bytes_per_gb = 1024 ** 3

        return MemoryMetrics(
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            peak_allocated_gb=peak_allocated / bytes_per_gb,
            peak_reserved_gb=peak_reserved / bytes_per_gb,
        )


def is_cuda_oom(error: BaseException) -> bool:
    """
    Return True only for recognized CUDA out-of-memory failures.

    Avoids classifying arbitrary RuntimeError instances as OOM.
    Supports stable OOM result generation for batch-feasibility sweeps.
    """
    if not isinstance(error, RuntimeError):
        return False

    error_msg = str(error).lower()

    oom_indicators = [
        "out of memory",
        "cuda out of memory",
        "cuda error",
        "not enough memory",
        "alloc",
    ]

    return any(indicator in error_msg for indicator in oom_indicators)
