# Data

Use `raw/` for immutable source data, `interim/` for deterministic transformations, and `processed/` for experiment-ready examples. These directories are created locally when needed and ignored by Git.

Before use, record the source URL, license, acquisition date, exact dataset revision, checksum/fingerprint, official split, schema, preprocessing command/code revision, filtering rules, and resulting sample IDs. Raw data must never be edited in place. Splits must be frozen before model comparison.

It is forbidden to tune, select prompts, debug, or choose checkpoints on test data; leak target/supporting evidence beyond the declared protocol; silently remove hard or failed examples; or change preprocessing between models. Every exclusion must be deterministic, counted, and justified before the confirmatory run.
