"""
Contract test: Generated configs validation.

Validates:
- All 33 generated configs are valid
- Config fields have correct types
- Config values are within expected ranges
"""

# import sys
from pathlib import Path

# from unittest.mock import MagicMock
import yaml

# Mock transformers to avoid version conflict during import
# sys.modules.setdefault('transformers', MagicMock())
from training.config import (
    load_experiment_config,
    validate_experiment_config,
)

# ============================================================================
# Config paths
# ============================================================================


GENERATED_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs" / "generated"

ALL_GENERATED_CONFIGS = [
    # Batch sweep configs
    "batch_full_ft_mb1.yaml",
    "batch_full_ft_mb16.yaml",
    "batch_full_ft_mb2.yaml",
    "batch_full_ft_mb4.yaml",
    "batch_full_ft_mb8.yaml",
    "batch_lora_mb1.yaml",
    "batch_lora_mb16.yaml",
    "batch_lora_mb2.yaml",
    "batch_lora_mb4.yaml",
    "batch_lora_mb8.yaml",
    # Main configs
    "main_full_ft.yaml",
    "main_lora.yaml",
    # Rank sweep configs
    "rank_lora_r16.yaml",
    "rank_lora_r32.yaml",
    "rank_lora_r4.yaml",
    "rank_lora_r8.yaml",
    # Seed sweep configs
    "seed_full_ft_r0_s11.yaml",
    "seed_full_ft_r0_s22.yaml",
    "seed_full_ft_r0_s33.yaml",
    "seed_lora_r16_s11.yaml",
    "seed_lora_r16_s22.yaml",
    "seed_lora_r16_s33.yaml",
    "seed_lora_r8_s11.yaml",
    "seed_lora_r8_s22.yaml",
    "seed_lora_r8_s33.yaml",
    # Sequence length sweep configs
    "seq_full_ft_l1024.yaml",
    "seq_full_ft_l256.yaml",
    "seq_full_ft_l512.yaml",
    "seq_lora_l1024.yaml",
    "seq_lora_l256.yaml",
    "seq_lora_l512.yaml",
    # Smoke test configs
    "smoke_full_ft.yaml",
    "smoke_lora.yaml",
]


# ============================================================================
# Tests
# ============================================================================

def test_data_config_exposes_dataset_revision() -> None:
    config = load_experiment_config(
        Path("configs/generated/smoke_lora.yaml")
    )

    assert config.data.dataset_revision
    assert not hasattr(config.data, "revision")

class TestGeneratedConfigs:
    def test_all_config_files_exist(self):
        """All 33 generated config files should exist."""
        for config_name in ALL_GENERATED_CONFIGS:
            config_path = GENERATED_CONFIGS_DIR / config_name
            assert config_path.exists(), f"Missing config: {config_name}"

    def test_all_configs_are_valid_yaml(self):
        """All config files should be valid YAML."""
        for config_name in ALL_GENERATED_CONFIGS:
            config_path = GENERATED_CONFIGS_DIR / config_name
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            assert data is not None, f"Empty config: {config_name}"
            assert isinstance(data, dict), f"Config is not a mapping: {config_name}"

    def test_all_configs_load_successfully(self):
        """All config files should load without errors."""
        for config_name in ALL_GENERATED_CONFIGS:
            config_path = GENERATED_CONFIGS_DIR / config_name
            config = load_experiment_config(config_path)
            assert config is not None

    def test_all_configs_validate(self):
        """All config files should pass validation."""
        for config_name in ALL_GENERATED_CONFIGS:
            config_path = GENERATED_CONFIGS_DIR / config_name
            config = load_experiment_config(config_path)
            validate_experiment_config(config)

    def test_batch_configs_have_correct_micro_batch_sizes(self):
        """Batch sweep configs should have correct micro_batch_size values."""
        batch_configs = [
            ("batch_full_ft_mb1.yaml", 1),
            ("batch_full_ft_mb2.yaml", 2),
            ("batch_full_ft_mb4.yaml", 4),
            ("batch_full_ft_mb8.yaml", 8),
            ("batch_full_ft_mb16.yaml", 16),
            ("batch_lora_mb1.yaml", 1),
            ("batch_lora_mb2.yaml", 2),
            ("batch_lora_mb4.yaml", 4),
            ("batch_lora_mb8.yaml", 8),
            ("batch_lora_mb16.yaml", 16),
        ]

        for config_name, expected_mb in batch_configs:
            config_path = GENERATED_CONFIGS_DIR / config_name
            config = load_experiment_config(config_path)
            assert config.training.micro_batch_size == expected_mb

    def test_rank_configs_have_correct_ranks(self):
        """Rank sweep configs should have correct LoRA rank values."""
        rank_configs = [
            ("rank_lora_r4.yaml", 4),
            ("rank_lora_r8.yaml", 8),
            ("rank_lora_r16.yaml", 16),
            ("rank_lora_r32.yaml", 32),
        ]

        for config_name, expected_rank in rank_configs:
            config_path = GENERATED_CONFIGS_DIR / config_name
            config = load_experiment_config(config_path)
            assert config.method.rank == expected_rank

    def test_seed_configs_have_correct_seeds(self):
        """Seed sweep configs should have correct seed values."""
        seed_configs = [
            ("seed_full_ft_r0_s11.yaml", 11),
            ("seed_full_ft_r0_s22.yaml", 22),
            ("seed_full_ft_r0_s33.yaml", 33),
            ("seed_lora_r8_s11.yaml", 11),
            ("seed_lora_r8_s22.yaml", 22),
            ("seed_lora_r8_s33.yaml", 33),
            ("seed_lora_r16_s11.yaml", 11),
            ("seed_lora_r16_s22.yaml", 22),
            ("seed_lora_r16_s33.yaml", 33),
        ]

        for config_name, expected_seed in seed_configs:
            config_path = GENERATED_CONFIGS_DIR / config_name
            config = load_experiment_config(config_path)
            assert config.training.seed == expected_seed

    def test_seq_configs_have_correct_max_lengths(self):
        """Sequence length configs should have correct max_length values."""
        seq_configs = [
            ("seq_full_ft_l256.yaml", 256),
            ("seq_full_ft_l512.yaml", 512),
            ("seq_full_ft_l1024.yaml", 1024),
            ("seq_lora_l256.yaml", 256),
            ("seq_lora_l512.yaml", 512),
            ("seq_lora_l1024.yaml", 1024),
        ]

        for config_name, expected_length in seq_configs:
            config_path = GENERATED_CONFIGS_DIR / config_name
            config = load_experiment_config(config_path)
            assert config.data.max_length == expected_length

    def test_full_ft_configs_have_correct_method(self):
        """Full FT configs should have method='full_ft'."""
        full_ft_configs = [
            "batch_full_ft_mb1.yaml",
            "batch_full_ft_mb2.yaml",
            "batch_full_ft_mb4.yaml",
            "batch_full_ft_mb8.yaml",
            "batch_full_ft_mb16.yaml",
            "main_full_ft.yaml",
            "seed_full_ft_r0_s11.yaml",
            "seed_full_ft_r0_s22.yaml",
            "seed_full_ft_r0_s33.yaml",
            "seq_full_ft_l256.yaml",
            "seq_full_ft_l512.yaml",
            "seq_full_ft_l1024.yaml",
            "smoke_full_ft.yaml",
        ]

        for config_name in full_ft_configs:
            config_path = GENERATED_CONFIGS_DIR / config_name
            config = load_experiment_config(config_path)
            assert config.method.name == "full_ft"

    def test_lora_configs_have_correct_method(self):
        """LoRA configs should have method='lora'."""
        lora_configs = [
            "batch_lora_mb1.yaml",
            "batch_lora_mb2.yaml",
            "batch_lora_mb4.yaml",
            "batch_lora_mb8.yaml",
            "batch_lora_mb16.yaml",
            "main_lora.yaml",
            "rank_lora_r4.yaml",
            "rank_lora_r8.yaml",
            "rank_lora_r16.yaml",
            "rank_lora_r32.yaml",
            "seed_lora_r8_s11.yaml",
            "seed_lora_r8_s22.yaml",
            "seed_lora_r8_s33.yaml",
            "seed_lora_r16_s11.yaml",
            "seed_lora_r16_s22.yaml",
            "seed_lora_r16_s33.yaml",
            "seq_lora_l256.yaml",
            "seq_lora_l512.yaml",
            "seq_lora_l1024.yaml",
            "smoke_lora.yaml",
        ]

        for config_name in lora_configs:
            config_path = GENERATED_CONFIGS_DIR / config_name
            config = load_experiment_config(config_path)
            assert config.method.name == "lora"

    def test_smoke_configs_have_small_values(self):
        """Smoke test configs should have small batch sizes and steps."""
        smoke_configs = ["smoke_full_ft.yaml", "smoke_lora.yaml"]

        for config_name in smoke_configs:
            config_path = GENERATED_CONFIGS_DIR / config_name
            config = load_experiment_config(config_path)

            # Smoke tests should have small values for quick execution
            assert config.training.micro_batch_size <= 2
            assert config.training.max_steps is not None
            assert config.training.max_steps <= 30

    def test_config_count_is_correct(self):
        """There should be exactly 39 generated config files."""
        assert len(ALL_GENERATED_CONFIGS) == 33

        existing_configs = list(GENERATED_CONFIGS_DIR.glob("*.yaml"))
        assert len(existing_configs) == 39, f"Expected 39 configs, found {len(existing_configs)}"
