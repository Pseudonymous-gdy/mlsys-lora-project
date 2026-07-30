from pathlib import Path

import yaml

from analysis.generate_configs import build_configs, generate_config_files

ROOT = Path(__file__).resolve().parents[1]


def test_full_experiment_matrix_is_complete_and_unique(tmp_path):
    base = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    sweeps = yaml.safe_load((ROOT / "configs/sweeps.yaml").read_text(encoding="utf-8"))
    configs = build_configs(base, sweeps)
    assert len(base["data"]["revision"]) == 40
    assert len(base["model"]["revision"]) == 40
    assert len(base["model"]["fallback_revision"]) == 40
    assert len(configs) == 63
    names = [config["experiment"]["name"] for config in configs]
    assert len(set(names)) == len(names)
    counts = {}
    tuning_grid = {"full_ft": set(), "lora": set()}
    tuning_seeds = {"full_ft": set(), "lora": set()}
    for config in configs:
        sweep = config["experiment"]["sweep"]
        counts[sweep] = counts.get(sweep, 0) + 1
        micro = config["training"]["micro_batch_size"]
        accumulation = config["training"]["gradient_accumulation_steps"]
        assert micro * accumulation == config["training"]["effective_batch_size"]
        method = config["method"]["name"]
        if method == "full_ft":
            assert config["method"]["rank"] is None
            assert config["method"]["alpha"] is None
            assert config["method"]["dropout"] is None
        # The tuning sweep exists to justify the defaults below, so it is the
        # only sweep allowed to vary the learning rate per method.
        if sweep == "hyperparameter_tuning":
            tuning_grid[method].add(config["training"]["learning_rate"])
            tuning_seeds[method].add(config["training"]["seed"])
            assert config["evaluation"]["split"] == "validation"
            assert config["output"]["save_final_checkpoint"] is False
        else:
            assert config["evaluation"].get("split", "test") == "test"
            expected = 0.00002 if method == "full_ft" else 0.0002
            assert config["training"]["learning_rate"] == expected
    assert counts == {
        "smoke": 2,
        "main": 2,
        "max_batch": 10,
        "rank": 4,
        "sequence_length": 6,
        "final_seeds": 15,
        "hyperparameter_tuning": 24,
    }
    assert tuning_grid == {
        "full_ft": {0.00002, 0.00005, 0.0001, 0.0002},
        "lora": {0.00002, 0.00005, 0.0001, 0.0002},
    }
    assert tuning_seeds == {"full_ft": {11, 22, 33}, "lora": {11, 22, 33}}

    paths = generate_config_files(
        ROOT / "configs/base.yaml", ROOT / "configs/sweeps.yaml", tmp_path
    )
    assert len(paths) == 63
    assert all(path.is_file() for path in paths)
