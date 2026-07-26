# Results

Local `runs/<experiment-id>/` directories contain immutable predictions, retrieved contexts/IDs, logs, resolved configs, environment records, timing, failures, and statistical-analysis inputs. They are ignored by Git but must be retained or reproducibly archived. `summary.csv` is the reviewed, tracked index of aggregate results.

Add a summary row only after metric validation and after the experiment ID resolves to its exact command, commit, configuration, and raw outputs. Comparative rows must include sample count, seed set, uncertainty interval, paired test, p-value or adjusted p-value where applicable, and effect size. Record failed and negative runs; never delete or overwrite inconvenient evidence.

Results from exploratory runs must be labeled and cannot support final claims until repeated under a predefined confirmatory configuration.

Tracked `reviews/*.audit.json` files contain deterministic audits of immutable raw artifacts when a technical failure prevented normal finalization. An audit may recover only values derivable from checksummed raw records; unavailable runtime fields remain explicit. It does not change the original run status or authorize a rerun.
