'''
Training engine.

Defines the core training loop that handles gradient accumulation, token-budget
stopping, CUDA memory tracking, throughput measurement, and dataset cycling.
'''

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

from src.training.results import StopReason, TrainingResult

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from src.metrics.memory import CudaMemoryTracker
    from src.metrics.throughput import ThroughputTracker
    from src.training.config import ExperimentConfig


# ============================================================================
# Training engine
# ============================================================================


class TrainerEngine:
    """
    Core training loop for the MLSys LoRA project.

    Responsibilities:
    - gradient accumulation
    - token-budget stopping
    - max-steps stopping
    - CUDA memory tracking
    - throughput measurement
    - dataset cycling
    - proper loss division
    """

    def __init__(
        self,
        config: ExperimentConfig,
        device: torch.device,
        memory_tracker: CudaMemoryTracker,
    ) -> None:
        self.config = config
        self.device = device
        self.memory_tracker = memory_tracker

    def _prepare_model_inputs(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], int]:
        """
        Extract num_non_padding_tokens and prepare inputs for the model.

        Returns:
            Tuple of (inputs_dict, num_non_padding_tokens)
        """
        num_non_padding_tokens = int(batch.pop("num_non_padding_tokens", 0))
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        return inputs, num_non_padding_tokens

    @contextmanager
    def _autocast_context(self):
        """Return bf16 autocast context or no-op context based on config."""
        if self.config.training.precision == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        else:
            yield

    def _should_stop(
        self,
        optimizer_steps: int,
        trained_non_padding_tokens: int,
    ) -> StopReason | None:
        """
        Check if training should stop based on max_steps or token budget.

        Returns:
            StopReason if training should stop, None otherwise.
        """
        max_steps = self.config.training.max_steps
        if max_steps is not None and optimizer_steps >= max_steps:
            return StopReason.MAX_STEPS

        token_budget = self.config.training.training_token_budget
        if token_budget is not None and trained_non_padding_tokens >= token_budget:
            return StopReason.TOKEN_BUDGET

        return None

    def _perform_optimizer_step(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        grad_accum_steps: int,
        current_step: int,
    ) -> None:
        """
        Perform optimizer step with optional gradient clipping.

        Only called when gradient accumulation is complete.
        """
        # Gradient clipping
        max_grad_norm = self.config.training.max_grad_norm
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()

    def train(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
        throughput_tracker: ThroughputTracker,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> TrainingResult:
        """
        Execute the full training loop.

        Algorithm:
        1. Initialize counters and trackers
        2. Loop until stop condition:
           a. Fetch batch from dataloader (cycle if exhausted)
           b. Prepare inputs, extract token count
           c. Forward pass with autocast
           d. Compute loss, scale by grad_accum_steps
           e. Backward pass
           f. Track throughput
           g. Accumulate tokens and micro-steps
           h. Perform optimizer step if accumulation complete
           i. Check stop conditions
        3. Return TrainingResult with all metrics
        """
        config = self.config.training
        grad_accum_steps = config.gradient_accumulation_steps

        # Initialize counters
        optimizer_steps = 0
        micro_steps = 0
        trained_non_padding_tokens = 0
        total_loss = 0.0
        loss_count = 0

        # Start tracking
        self.memory_tracker.reset()
        start_time = time.time()
        throughput_tracker.start()

        # Create dataloader iterator
        data_iter = iter(train_loader)

        while True:
            # Check stop conditions
            stop_reason = self._should_stop(optimizer_steps, trained_non_padding_tokens)
            if stop_reason is not None:
                break

            # Fetch batch (cycle dataloader if exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                # Dataset exhausted, recreate iterator
                data_iter = iter(train_loader)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Empty dataset
                    stop_reason = StopReason.DATA_EXHAUSTED
                    break

            # Prepare inputs
            inputs, num_tokens = self._prepare_model_inputs(batch)

            # Forward pass
            with self._autocast_context():
                outputs = model(**inputs, use_cache=False)
                loss = outputs.loss / grad_accum_steps

            # Backward pass
            loss.backward()

            # Track throughput
            throughput_tracker.record_tokens(num_tokens)

            # Accumulate counters
            total_loss += float(loss.item()) * grad_accum_steps
            loss_count += 1
            micro_steps += 1
            trained_non_padding_tokens += num_tokens

            # Optimizer step
            if micro_steps % grad_accum_steps == 0:
                self._perform_optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    grad_accum_steps=grad_accum_steps,
                    current_step=optimizer_steps,
                )
                optimizer_steps += 1

        # Finish tracking
        measured_time = time.time() - start_time
        self.memory_tracker.snapshot()

        # Compute final metrics
        final_loss = float(total_loss / loss_count) if loss_count > 0 else 0.0
        mean_loss = final_loss  # Same as final_loss for this implementation

        throughput_tracker.finish()
        throughput_metrics = throughput_tracker.get_metrics()

        memory_metrics = self.memory_tracker.get_metrics()

        return TrainingResult(
            optimizer_steps=optimizer_steps,
            micro_steps=micro_steps,
            trained_non_padding_tokens=trained_non_padding_tokens,
            final_loss=final_loss,
            mean_loss=mean_loss,
            training_time_seconds=measured_time,
            measured_time_seconds=measured_time,
            tokens_per_second=throughput_metrics.tokens_per_second,
            peak_memory_gb=memory_metrics.peak_memory_gb,
            peak_reserved_memory_gb=memory_metrics.peak_reserved_memory_gb,
            stop_reason=stop_reason.value if stop_reason else StopReason.DATA_EXHAUSTED.value,
        )
