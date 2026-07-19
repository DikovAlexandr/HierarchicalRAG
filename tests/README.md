# Tests

Before reporting results, test answer normalization, EM/F1, retrieval recall, context serialization/order/truncation, deterministic distractor injection, seed handling, split boundaries, leakage protections, configuration completeness, and statistical-analysis code. Use trusted benchmark examples or independently calculated fixtures.

Add small integration tests for dataset, retriever, and model adapters without downloading large artifacts during the default test run. A metric or analysis implementation that changed must be revalidated before old and new results are compared.
