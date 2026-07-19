# Artifacts

Store local model checkpoints under `checkpoints/` and retrieval indexes under `indexes/`; both locations are ignored by Git.

An artifact may be used in a reported experiment only when its source or build command, upstream revision, checksum, creation configuration, and compatible code/config versions are recorded. Do not overwrite an artifact used by an existing run. Never commit weights or indexes, and do not delete the only reproducible copy until the corresponding result has been independently verified or the artifact can be rebuilt deterministically.
