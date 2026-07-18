"""Generate deterministic per-run YAML files from the experiment matrix."""

from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def set_dotted(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = config
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set {path}: {part} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def _axis_combinations(axes: Mapping[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    names = list(axes)
    values = [list(axes[name]) for name in names]
    if any(not axis for axis in values):
        raise ValueError("sweep axes must not be empty")
    if not names:
        yield {}
        return
    for combination in itertools.product(*values):
        yield dict(zip(names, combination))


def build_configs(
    base: Mapping[str, Any], specification: Mapping[str, Any]
) -> list[dict[str, Any]]:
    paths = dict(specification.get("variable_paths", {}))
    generated: list[dict[str, Any]] = []
    names: set[str] = set()
    for sweep_name, sweep in specification["sweeps"].items():
        matrix = list(sweep.get("matrix", [{}]))
        axes = dict(sweep.get("axes", {}))
        for row in matrix:
            for axis_values in _axis_combinations(axes):
                variables = {**dict(row), **axis_values, "sweep": sweep_name}
                run_name = str(sweep["run_name"]).format(**variables)
                if run_name in names:
                    raise ValueError(f"duplicate generated run name: {run_name}")
                names.add(run_name)
                config = deep_merge(base, sweep.get("overrides", {}))
                for variable, value in variables.items():
                    if variable in paths:
                        set_dotted(config, paths[variable], value)
                config.setdefault("experiment", {})["name"] = run_name
                config["experiment"]["sweep"] = sweep_name

                micro_batch = int(config["training"]["micro_batch_size"])
                effective_batch = int(config["training"]["effective_batch_size"])
                if effective_batch % micro_batch != 0:
                    raise ValueError(
                        f"{run_name}: effective batch {effective_batch} is not divisible "
                        f"by micro batch {micro_batch}"
                    )
                config["training"]["gradient_accumulation_steps"] = (
                    effective_batch // micro_batch
                )
                if config["method"]["name"] == "full_ft":
                    config["method"]["rank"] = None
                    config["method"]["alpha"] = None
                    config["method"]["dropout"] = None
                generated.append(config)
    return generated


def generate_config_files(
    base_path: Path, sweeps_path: Path, output_dir: Path
) -> list[Path]:
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    with sweeps_path.open("r", encoding="utf-8") as handle:
        specification = yaml.safe_load(handle)
    configs = build_configs(base, specification)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for config in configs:
        path = output_dir / f"{config['experiment']['name']}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--sweeps", type=Path, default=Path("configs/sweeps.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("configs/generated"))
    args = parser.parse_args()
    paths = generate_config_files(args.base, args.sweeps, args.output_dir)
    print(f"Generated {len(paths)} configs in {args.output_dir}")


if __name__ == "__main__":
    main()
