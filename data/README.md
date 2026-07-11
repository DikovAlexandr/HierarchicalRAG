# Data

- `raw/`: immutable downloaded or externally supplied data.
- `interim/`: normalized or indexed intermediate data.
- `processed/`: experiment-ready examples and deterministic slices.

Data contents are ignored by Git. Record source URL, license, version/checksum, acquisition date, preprocessing command, and resulting schema in experiment metadata or a dataset-specific README without committing the dataset itself.
