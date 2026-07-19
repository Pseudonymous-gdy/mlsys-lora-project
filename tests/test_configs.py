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
    assert len(configs) == 33
    names = [config["experiment"]["name"] for config in configs]
    assert len(set(names)) == len(names)
    counts = {}
    for config in configs:
        sweep = config["experiment"]["sweep"]
        counts[sweep] = counts.get(sweep, 0) + 1
        micro = config["training"]["micro_batch_size"]
        accumulation = config["training"]["gradient_accumulation_steps"]
        assert micro * accumulation == config["training"]["effective_batch_size"]
        if config["method"]["name"] == "full_ft":
            assert config["method"]["rank"] is None
            assert config["method"]["alpha"] is None
            assert config["method"]["dropout"] is None
            assert config["training"]["learning_rate"] == 0.00002
        else:
            assert config["training"]["learning_rate"] == 0.0002
    assert counts == {
        "smoke": 2,
        "main": 2,
        "max_batch": 10,
        "rank": 4,
        "sequence_length": 6,
        "final_seeds": 9,
    }

    paths = generate_config_files(
        ROOT / "configs/base.yaml", ROOT / "configs/sweeps.yaml", tmp_path
    )
    assert len(paths) == 33
    assert all(path.is_file() for path in paths)
