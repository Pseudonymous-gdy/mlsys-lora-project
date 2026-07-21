"""Shared pytest fixtures for the MLSys LoRA project."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture
def tiny_causal_lm_factory(
) -> Callable[..., LlamaForCausalLM]:
    """Create deterministic, local-only tiny causal LM instances.

    The returned model is a real Hugging Face causal language model.
    It supports:

    - prepare_inputs_for_generation
    - PEFT TaskType.CAUSAL_LM
    - save_pretrained / from_pretrained
    - q_proj, k_proj, v_proj and o_proj LoRA targets

    No model files are downloaded.
    """

    def _factory(
        *,
        seed: int = 42,
        vocab_size: int = 128,
        max_position_embeddings: int = 64,
    ) -> LlamaForCausalLM:
        torch.manual_seed(seed)

        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=max_position_embeddings,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            attention_dropout=0.0,
            hidden_act="silu",
            tie_word_embeddings=False,
            use_cache=False,
        )

        model = LlamaForCausalLM(config)
        model.config.use_cache = False
        return model

    return _factory