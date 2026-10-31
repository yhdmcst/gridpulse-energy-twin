# Contributing to GridPulse

Contributions should preserve the distinction between historical offline simulation and production deployment.
Recovery feasibility, performance trade-offs, and out-of-envelope failures must remain visible.

## Development

Use Python 3.12 and an isolated virtual environment:

```bash
python -m pip install -r requirements-training.txt
python scripts/prepare_release.py
python -m pytest
python scripts/check_app.py
python scripts/verify_envelope_results.py
```

Neural-network tests require PyTorch; the dashboard and V8 evidence audit do not.
Two optional historical-evidence tests skip when their earlier experiment outputs are absent.

## Pull Requests

1. Keep changes focused and add tests for changed behavior.
2. Preserve time-series causality and state when information becomes available.
3. Separate validation-period selection from evaluation-period reporting.
4. Do not overwrite frozen experiment evidence or reformat frozen sources.
5. Do not commit restored `outputs/`, `data/processed/`, virtual environments, credentials, or local caches.
6. The existing `artifacts/` archives are intentional release inputs, not accidental generated files.
7. Record new data sources and third-party material in `THIRD_PARTY_NOTICES.md`.
8. Update conclusions and documentation when a result or selection rule changes.

Earlier report, submission, and presentation scripts may require the original research workspace.
See [release scope](docs/release-scope.md) for the portable workflows supported by this public copy.

Report reproducible bugs with the issue template. Do not include personal information, unpublished competition materials, or credentials.
