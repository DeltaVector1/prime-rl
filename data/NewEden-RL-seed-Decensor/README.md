# NewEden RL Seed Decensor

This directory is the local cache location for Decensor training data:

```toml
decensor_dataset_name = "data/NewEden-RL-seed-Decensor/rl.jsonl"
```

The `rl.jsonl` file is intentionally not committed here. It is about 125 MB,
is ignored by the repository `*.jsonl` rule, and requires Git LFS or an
external artifact store. Restore it from the Hugging Face dataset source used
for the run before launching training.
