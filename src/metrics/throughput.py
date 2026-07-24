"""
Training throughput measurement utilities.

Throughput is measured in non-padding tokens per second. Configured
warmup optimizer steps are excluded from both measured tokens and
measured wall-clock time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ThroughputMetrics:
    """Immutable container for training throughput statistics."""

    measured_tokens: int
    measured_seconds: float
    tokens_per_second: float
    warmup_optimizer_steps: int


class ThroughputTracker:
    """
    Track non-padding-token throughput during training.

    Expected call order:

        tracker.start()

        after every completed optimizer step:
            tracker.step()
            tracker.record_tokens(step_tokens)

        metrics = tracker.finish()

    Warmup semantics:

    - the first ``warmup_optimizer_steps`` optimizer steps are trained
      normally;
    - their tokens are not included in measured_tokens;
    - the throughput timer begins immediately after the final warmup
      optimizer step;
    - only subsequent optimizer steps contribute measured tokens.
    """

    def __init__(
        self,
        device: torch.device,
        warmup_optimizer_steps: int,
    ) -> None:
        if warmup_optimizer_steps < 0:
            raise ValueError(
                "warmup_optimizer_steps must be non-negative, "
                f"got {warmup_optimizer_steps}"
            )

        self._device = device
        self._warmup_optimizer_steps = (
            warmup_optimizer_steps
        )

        self._started = False
        self._start_time: float | None = None
        self._measured_tokens = 0
        self._optimizer_steps = 0

    def _synchronize(self) -> None:
        """Synchronize pending CUDA work before timing boundaries."""
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def _begin_measurement(self) -> None:
        """Start the post-warmup throughput measurement region."""
        if self._start_time is not None:
            raise RuntimeError(
                "Throughput measurement has already begun."
            )

        self._synchronize()
        self._start_time = time.perf_counter()

    def start(self) -> None:
        """
        Initialize throughput tracking.

        When warmup is zero, measurement starts immediately. When
        warmup is positive, the timer starts after the final warmup
        optimizer step completes.
        """
        if self._started:
            raise RuntimeError(
                "ThroughputTracker.start() called twice. "
                "Measurement has already begun."
            )

        self._started = True

        if self._warmup_optimizer_steps == 0:
            self._begin_measurement()

    def step(self) -> None:
        """
        Record completion of one optimizer step.

        This method must be called after optimizer.step() succeeds and
        before record_tokens() is called for the same optimizer step.
        """
        if not self._started:
            raise RuntimeError(
                "ThroughputTracker.step() called before start()."
            )

        self._optimizer_steps += 1

        if (
            self._optimizer_steps
            == self._warmup_optimizer_steps
            and self._start_time is None
        ):
            self._begin_measurement()

    def record_tokens(
        self,
        non_padding_tokens: int,
    ) -> None:
        """
        Record tokens committed by one completed optimizer step.

        Tokens from configured warmup optimizer steps are excluded.
        Tokens recorded before start() are ignored to preserve the
        existing tracker contract.
        """
        if non_padding_tokens < 0:
            raise ValueError(
                "Non-padding token count must be non-negative, "
                f"got {non_padding_tokens}"
            )

        if not self._started:
            return

        if (
            self._warmup_optimizer_steps > 0
            and self._optimizer_steps
            <= self._warmup_optimizer_steps
        ):
            return

        self._measured_tokens += non_padding_tokens

    def finish(self) -> ThroughputMetrics:
        """
        Finish post-warmup throughput measurement.

        Raises when tracking never started, warmup never completed, no
        post-warmup tokens were measured, or elapsed time is invalid.
        """
        if not self._started:
            raise RuntimeError(
                "ThroughputTracker.finish() called before start(). "
                "Call start() before finishing measurement."
            )

        if self._start_time is None:
            raise RuntimeError(
                "Training finished before throughput warmup "
                "completed."
            )

        if self._measured_tokens <= 0:
            raise RuntimeError(
                "No post-warmup tokens were measured."
            )

        self._synchronize()

        end_time = time.perf_counter()
        elapsed = end_time - self._start_time

        if elapsed <= 0:
            raise ValueError(
                "Elapsed time must be positive, "
                f"got {elapsed:.6f}s. "
                "Check timing logic and CUDA synchronization."
            )

        return ThroughputMetrics(
            measured_tokens=self._measured_tokens,
            measured_seconds=elapsed,
            tokens_per_second=(
                self._measured_tokens / elapsed
            ),
            warmup_optimizer_steps=(
                self._warmup_optimizer_steps
            ),
        )