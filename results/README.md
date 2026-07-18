# Results directory

Store only real experiment summaries produced by training/evaluation here.

- Do not place unit-test fixtures, invented metrics, model weights, checkpoints,
  caches, or logs in this directory.
- A completed result JSON must follow the schema in
  `docs/experiment_protocol.md`.
- OOM attempts should be explicitly marked with `"status": "oom"`.
- Raw generation JSONL can be kept outside Git for error analysis; commit only
  compact, auditable summaries needed to reproduce tables and figures.
