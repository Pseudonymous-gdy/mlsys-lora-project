'''
Training engine.

Defines the core training loop that handles gradient accumulation, token-budget
stopping, CUDA memory tracking, throughput measurement, and dataset cycling.
'''

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from training.results import StopReason, TrainingResult

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

    from metrics.memory import CudaMemoryTracker
    from metrics.throughput import ThroughputTracker
    from training.config import ExperimentConfig


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
        batch: dict[str, torch.Tensor | int],
    ) -> tuple[dict[str, torch.Tensor], int]:
        """
        Extract num_non_padding_tokens and prepare inputs for the model.

        Returns:
            Tuple of (inputs_dict, num_non_padding_tokens)
        """
        inputs: dict[str, torch.Tensor] = {}

        for key, value in batch.items():
            if key == "num_non_padding_tokens":
                continue

            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Model input '{key}' must be a torch.Tensor."
                )

            inputs[key] = value.to(self.device)
        
        if "num_non_padding_tokens" not in batch:
            raise KeyError(
                "Training batch is missing required metadata "
                "'num_non_padding_tokens'."
            )

        raw_token_count = batch["num_non_padding_tokens"]

        if isinstance(raw_token_count, torch.Tensor):
            if raw_token_count.numel() != 1:
                raise ValueError(
                    "'num_non_padding_tokens' must contain "
                    "exactly one value."
                )
            num_non_padding_tokens = int(
                raw_token_count.item()
            )
        elif isinstance(raw_token_count, int):
            num_non_padding_tokens = raw_token_count
        else:
            raise TypeError(
                "'num_non_padding_tokens' must be an int "
                "or a one-element tensor."
            )

        if num_non_padding_tokens <= 0:
            raise ValueError(
                "'num_non_padding_tokens' must be positive."
            )

        required_keys = ("input_ids", "labels")
        missing_keys = [
            key for key in required_keys if key not in batch
        ]

        if missing_keys:
            raise KeyError(
                f"Training batch is missing required inputs: "
                f"{missing_keys}"
            )

        inputs = {
            key: value.to(self.device)
            for key, value in batch.items()
            if key != "num_non_padding_tokens"
        }

        return inputs, num_non_padding_tokens

    @contextmanager
    def _autocast_context(self):
        """Return bf16 autocast context or no-op context based on config."""
        if self.config.training.precision == "bf16":
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
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
    ) -> None:
        """
        Perform optimizer step with optional gradient clipping.

        Only called when gradient accumulation is complete.
        """
        gradient_clip_norm = (
            self.config.training.gradient_clip_norm
        )

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                gradient_clip_norm,
            )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad(set_to_none=True)
    
    def _validate_loss(
        self,
        loss: torch.Tensor | None,
    ) -> torch.Tensor:
        if loss is None:
            raise RuntimeError(
                "Model output does not contain a loss."
            )

        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                "Model loss must be a torch.Tensor."
            )

        if loss.numel() != 1:
            raise ValueError(
                "Model loss must be scalar."
            )

        if not torch.isfinite(loss.detach()).item():
            value = loss.detach().item()
            raise FloatingPointError(
                f"Non-finite training loss detected: {value}"
            )

        return loss

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
        2. Loop until stop condition (checked after optimizer step):
           a. Fetch batch from dataloader (cycle if exhausted)
           b. Prepare inputs, extract token count
           c. Forward pass with autocast
           d. Compute loss, scale by grad_accum_steps
           e. Backward pass
           f. Accumulate window counters
           g. If accumulation window complete, perform optimizer step
              and submit window tokens to throughput
           h. Check stop conditions
        3. Return TrainingResult with all metrics
        """
        config = self.config.training
        grad_accum_steps = config.gradient_accumulation_steps

        # Initialize counters
        optimizer_steps = 0
        micro_steps = 0
        micro_steps_in_window = 0

        trained_non_padding_tokens = 0
        window_non_padding_tokens = 0

        loss_sum = 0.0
        loss_count = 0
        final_loss: float | None = None

        # Start training
        model.train()
        optimizer.zero_grad(set_to_none=True)

        self.memory_tracker.reset()
        throughput_tracker.start()

        training_start_time = time.perf_counter()

        # Create dataloader iterator
        data_iter = iter(train_loader)

        stop_reason: StopReason | None = None

        while stop_reason is None:
            # Fetch batch (cycle dataloader if exhausted)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)

                try:
                    batch = next(data_iter)
                except StopIteration as exc:
                    raise ValueError(
                        "Training dataloader produced no batches."
                    ) from exc

            # Prepare inputs
            inputs, num_tokens = (
                self._prepare_model_inputs(batch)
            )

            # Forward pass
            with self._autocast_context():
                outputs = model(
                    **inputs,
                    use_cache=False,
                )
                loss = self._validate_loss(outputs.loss)

            raw_loss_value = float(loss.detach().item())

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            micro_steps += 1
            micro_steps_in_window += 1
            window_non_padding_tokens += num_tokens

            loss_sum += raw_loss_value
            loss_count += 1
            final_loss = raw_loss_value

            # Continue accumulating if window not complete
            if micro_steps_in_window < grad_accum_steps:
                continue

            # Complete optimizer step
            self._perform_optimizer_step(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            optimizer_steps += 1

            trained_non_padding_tokens += (
                window_non_padding_tokens
            )

            throughput_tracker.step()
            throughput_tracker.record_tokens(
                window_non_padding_tokens
            )

            window_non_padding_tokens = 0
            micro_steps_in_window = 0

            # Check stop conditions (only after optimizer step)
            stop_reason = self._should_stop(
                optimizer_steps,
                trained_non_padding_tokens,
            )
        
        training_time_seconds = (
            time.perf_counter() - training_start_time
        )

        # Compute final metrics
        if final_loss is None or loss_count == 0:
            raise RuntimeError(
                "Training completed without a valid loss."
            )

        mean_loss = loss_sum / loss_count

        throughput_metrics = throughput_tracker.finish()

        memory_metrics = self.memory_tracker.snapshot()

        if stop_reason is None:
            raise RuntimeError(
                "Training exited without a stop reason."
            )

        return TrainingResult(
            optimizer_steps=optimizer_steps,
            micro_steps=micro_steps,
            trained_non_padding_tokens=(
                trained_non_padding_tokens
            ),
            final_loss=final_loss,
            mean_loss=mean_loss,
            training_time_seconds=training_time_seconds,
            measured_time_seconds=(
                throughput_metrics.measured_seconds
            ),
            tokens_per_second=(
                throughput_metrics.tokens_per_second
            ),
            peak_memory_gb=memory_metrics.peak_allocated_gb,
            peak_reserved_memory_gb=(
                memory_metrics.peak_reserved_gb
            ),
            stop_reason=stop_reason.value,
        )
