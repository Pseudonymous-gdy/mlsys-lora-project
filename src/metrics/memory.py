'''
This module provides GPU memory measurement utilities for training experiments.
'''

from dataclasses import dataclass

import torch


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

        allocated_bytes = torch.cuda.max_memory_allocated(
            self._device
        )
        reserved_bytes = torch.cuda.max_memory_reserved(
            self._device
        )

        gib = 1024**3

        return MemoryMetrics(
            peak_allocated_bytes=allocated_bytes,
            peak_reserved_bytes=reserved_bytes,
            peak_allocated_gb=allocated_bytes / gib,
            peak_reserved_gb=reserved_bytes / gib,
        )


def is_cuda_oom(error: BaseException) -> bool:
    """
    Return True only for recognized CUDA out-of-memory failures.

    Avoids classifying arbitrary RuntimeError instances as OOM.
    Supports stable OOM result generation for batch-feasibility sweeps.

    Explicitly returns False for:
    - CUDA error: illegal memory access
    - CUDA error: device-side assert triggered
    - invalid allocator configuration
    - failed to allocate output filename
    - plain CPU / filesystem / application memory errors
    """
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True

    if not isinstance(error, RuntimeError):
        return False

    message = str(error).lower()

    # Positive indicators — require "cuda" AND an OOM signal to avoid
    # matching CPU, filesystem, or generic application memory errors.
    oom_signals = (
        "out of memory",
        "out of memory on device",
        "cublas_status_alloc_failed",
        "not enough memory",
    )

    has_cuda = "cuda" in message
    has_oom = any(signal in message for signal in oom_signals)

    return has_cuda and has_oom
