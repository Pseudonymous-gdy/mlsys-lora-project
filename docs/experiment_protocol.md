# GSM8K Data, Evaluation, and Analysis Protocol

This document defines the CPU-side contract shared by Full FT and LoRA. Any
change to these controls must be applied to both methods and recorded in the
final report.

## 1. What is fixed

- Dataset: `openai/gsm8k`, configuration `main`.
- Official train size: 7,473; official test size: 1,319.
- Validation: 500 examples deterministically carved from official train with
  seed 42, leaving 6,973 training examples.
- The official test set is never used for training, early stopping, or tuning.
- Prompt format, tokenizer, example order, maximum length, padding policy,
  effective batch size, precision, training token budget, and evaluation code
  must match between Full FT and LoRA for a comparison.
- The input pipeline returns `input_ids`, `attention_mask`, `labels`, and
  `num_non_padding_tokens`. Daoyuan's training loop should remove the token-count
  field before calling the model and use it for throughput accounting.
- Full FT and LoRA must use the same attention backend. A local Qwen3.5 smoke
  test successfully used the PyTorch fallback because optional fast-attention
  packages were absent; the selected A40 backend must be recorded in run metadata.

## 2. Prompt and supervision

`src/data/gsm8k.py` renders the model's chat template with thinking disabled and
adds an assistant generation boundary. The reference answer is never included
in an evaluation prompt.

During training, prompt labels are `-100`; only the assistant answer contributes
to causal-language-model loss. Every target ends with the tokenizer EOS token.
If an example is too long, the code truncates reasoning tokens while preserving:

1. the complete question prompt;
2. the final `#### <answer>` suffix; and
3. EOS.

If even these mandatory parts cannot fit, the example is deterministically
filtered and the retained/truncated counts must be reported.

Local validation with `Qwen/Qwen3.5-0.8B` tokenizer and Transformers 5.12.1:

| Max length | Retained train | Truncated train | Retained validation | Truncated validation |
|---:|---:|---:|---:|---:|
| 256 | 6,972 | 2,015 | 500 | 131 |
| 512 | 6,973 | 10 | 500 | 0 |
| 1024 | 6,973 | 0 | 500 | 0 |

The one-example difference at length 256 must be mentioned when interpreting the
sequence-length sweep. Full FT and LoRA at the same length still use identical
retained examples.

## 3. Exact-match evaluation

The evaluator is not another language model. It is a deterministic parser and
comparator:

1. extract the official number after GSM8K's `####` marker;
2. extract the model's final numeric answer, preferring `####`, `\boxed{}`, and
   explicit “final answer” phrases before a last-number fallback;
3. normalize commas, trailing zeros, signs, fractions, and percentages;
4. compare normalized strings exactly.

Unparseable generations count as incorrect. Greedy decoding (`do_sample: false`)
is the default so evaluation is repeatable. Raw predictions should be retained
for error analysis but must not be confused with aggregate experiment results.

## 4. Generated experiment matrix

`configs/base.yaml` and `configs/sweeps.yaml` generate 33 run configs:

- 2 smoke tests (Full FT and LoRA-16, 30 steps);
- 2 main comparisons (Full FT and LoRA-16);
- 10 maximum-batch runs (2 methods × micro-batch 1/2/4/8/16);
- 4 LoRA ranks (4/8/16/32);
- 6 sequence-length runs (2 methods × 256/512/1024);
- 9 final-seed runs (Full FT, LoRA-8, LoRA-16 × seeds 11/22/33).

The base protocol uses effective batch size 16 and a one-million non-padding
training-token budget for non-smoke experiments. The generated defaults use
learning rate 2e-5 for Full FT and 2e-4 for LoRA. These research choices are
explicit and reviewable; both teammates should approve them before starting the
expensive cluster runs.

## 5. Training result contract

Each completed training run must emit one JSON object containing at least:

```json
{
  "method": "lora",
  "rank": 16,
  "max_length": 512,
  "micro_batch_size": 4,
  "peak_memory_gb": 0.0,
  "tokens_per_second": 0.0,
  "training_time_seconds": 0.0,
  "exact_match": 0.0,
  "trainable_parameters": 1,
  "checkpoint_size_mb": 0.0,
  "seed": 42,
  "sweep": "main",
  "status": "completed"
}
```

The training entry point should also copy `experiment.sweep` from its YAML into
the result as `"sweep"`. Aggregation uses this field to prevent, for example, a
single-seed rank sweep from being mixed with the final three-seed validation.

An OOM attempt may use `"status": "oom"`; aggregation skips it while the batch
sweep logic can still use it to establish the feasibility boundary. Placeholder
numbers or unit-test fixtures must never be stored in `results/`.

## 6. Analysis commands after real runs exist

```bash
PYTHONPATH=src python -m analysis.aggregate results \
  --csv reports/aggregate.csv --json reports/aggregate.json

PYTHONPATH=src python -m analysis.plot_results reports/aggregate.csv \
  --sweep main --output reports/tradeoff.png

PYTHONPATH=src python -m analysis.plot_results reports/aggregate.csv \
  --sweep rank --kind rank --output reports/rank_sweep.png

PYTHONPATH=src python -m analysis.plot_results reports/aggregate.csv \
  --sweep sequence_length --kind sequence --output reports/sequence_sweep.png
```

Aggregation validates the schema, computes mean and sample standard deviation,
and marks configurations that are not dominated on memory (lower), throughput
(higher), and exact match (higher). It also writes
`reports/batch_feasibility.csv`, containing the largest successful micro-batch
and the first observed OOM for each method/length combination.

## 7. Handoff to Daoyuan

Daoyuan should import `prepare_gsm8k_datasets` and `CausalLMCollator` rather than
reimplement tokenization. Evaluation should call `generate_predictions` and save
its JSONL records. This ensures Full FT and LoRA share exactly one data and
evaluation path. All core changes go through a Pull Request and cross-review.
