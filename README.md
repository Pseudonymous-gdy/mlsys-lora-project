# MLSys LoRA Project

Team members:

- Daoyuan GUO
- Zhaokai LIANG

Project archetype:

- Archetype 2.3: Large-model adaptation slice
- Topic: LoRA-based adaptation and systems measurement

Planned focus:

- Compare LoRA with full fine-tuning.
- Measure peak GPU memory, training throughput, epoch time, checkpoint size, and evaluation quality.
- Use a small language model and a manageable dataset to keep the project feasible on HPC resources.

Large files such as datasets, model weights, checkpoints, and logs are not tracked by Git.

## Zhaokai's local workflow

The CPU-side pipeline can be developed and verified on macOS. Do not install
the CUDA-pinned cluster requirements on a Mac.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt

# Unit and integration tests (no model weights required)
python -m pytest -q
ruff check src tests scripts

# Real GSM8K + Qwen tokenizer smoke test
HF_HOME="$HOME/Library/Caches/mlsys-lora-project/huggingface" \
  python scripts/smoke_data.py --samples 50

# Materialize all per-run experiment configurations
python scripts/generate_sweep_configs.py
```

The local pipeline prepares data, scores existing model completions, generates
experiment configs, aggregates result JSON, and plots results. It does **not**
invent model predictions or GPU measurements. Full FT/LoRA training, A40 memory,
and throughput measurements must come from the cluster.

A one-example base-model smoke test was also completed with
`Qwen/Qwen3.5-0.8B` on Apple MPS. This validates the real generation/evaluation
interface only; its answer is not a Full FT or LoRA result.

Dataset and model revisions are pinned in `configs/base.yaml`. This guarantees
that local and HPC runs request the same upstream artifacts rather than whatever
version happens to be newest later.

See [docs/experiment_protocol.md](docs/experiment_protocol.md) for the data
contract, fairness controls, result schema, and handoff procedure.

## Collaboration rule

Develop on a feature branch and open a Pull Request for cross-review. Do not
push core code directly to `main`, and do not commit datasets, model weights,
checkpoints, caches, logs, secrets, or local environments.
