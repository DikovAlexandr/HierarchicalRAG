# Experiments

Each experiment must have a versioned configuration in `configs/` and a unique ID such as `E2-frozen-hrm-hotpotqa-s1`. A configuration should pin dataset/split, data revision, retriever and top-k, model and revision, context serialization, decoding budget, seed, metrics, hardware, and output path.

Raw outputs belong in `results/runs/<experiment-id>/` and are ignored by Git. Add validated aggregate values to `results/summary.csv`; store the command and configuration path with the run metadata.
