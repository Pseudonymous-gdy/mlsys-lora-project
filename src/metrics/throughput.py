'''
This module provides training throughput measurement utilities.

Throughput is measured in non-padding tokens per second, excluding
configured warmup optimizer steps.
'''

import time
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class ThroughputMetrics:
    """Immutable container for training throughput statistics."""
    measured_tokens: int
    measured_seconds: float
    tokens_per_second: float
    warmup_optimizer_steps: int


class ThroughputTracker:
    """
    Tracks non-padding token throughput during training.

    State:
    - started: whether measurement has begun
    - start_time: perf_counter timestamp when measurement started
    - measured_tokens: accumulated non-padding tokens after warmup
    - warmup_optimizer_steps: number of optimizer steps to skip for timing
    """

    def __init__(
        self,
        device: torch.device,
        warmup_optimizer_steps: int,
    ) -> None:
        self._device = device
        self._warmup_optimizer_steps = warmup_optimizer_steps

        self._started = False
        self._start_time: float | None = None
        self._measured_tokens = 0
        self._optimizer_steps = 0

    def start(self) -> None:
        """
        Begin throughput measurement.

        Synchronizes CUDA when using CUDA device.
        Records time.perf_counter() timestamp.
        Rejects a second start call.
        """
        if self._started:
            raise RuntimeError(
                "ThroughputTracker.start() called twice. "
                "Measurement has already begun."
            )

        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

        self._start_time = time.perf_counter()
        self._started = True

    def record_tokens(self, non_padding_tokens: int) -> None:
        """
        Accumulate non-padding token counts.

        Rejects negative counts.
        Does nothing before measurement starts.
        """
        if non_padding_tokens < 0:
            raise ValueError(
                f"Non-padding token count must be non-negative, "
                f"got {non_padding_tokens}"
            )

        if not self._started:
            return

        self._measured_tokens += non_padding_tokens

    def step(self) -> None:
        """
        Increment the optimizer step counter.

        Called after each optimizer.step() to track warmup completion.
        """
        self._optimizer_steps += 1

    def finish(self) -> ThroughputMetrics:
        """
        Finalize throughput measurement.

        Synchronizes CUDA, calculates elapsed time, and returns metrics.
        Rejects finish-before-start and zero elapsed time.
        """
        if not self._started:
            raise RuntimeError(
                "ThroughputTracker.finish() called before start(). "
                "Call start() before finishing measurement."
            )

        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

        end_time = time.perf_counter()
        elapsed = end_time - self._start_time

        if elapsed <= 0:
            raise ValueError(
                f"Elapsed time must be positive, got {elapsed:.6f}s. "
                "Check timing logic and CUDA synchronization."
            )

        tokens_per_second = self._measured_tokens / elapsed

        return ThroughputMetrics(
            measured_tokens=self._measured_tokens,
            measured_seconds=elapsed,
            tokens_per_second=tokens_per_second,
            warmup_optimizer_steps=self._warmup_optimizer_steps,
        )
