# Tests

No tests exist yet because the package has no implemented behavior. Running
pytest reports no tests collected (exit code 5); this is not a passing suite.

When implementation starts, use pytest for:

- Deterministic unit tests for individual stages using small synthetic inputs.
- Regression tests against approved notebook outputs, with numeric tolerances.
- Integration tests connecting stages and eventually the full pipeline.

Keep tests requiring datasets, model weights, or GPUs opt-in. Keep large
research inputs outside Git. Define units, shapes, coordinate conventions,
and rejected-frame behavior before writing the corresponding tests.
