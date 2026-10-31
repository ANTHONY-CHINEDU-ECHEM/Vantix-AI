# Contributing to Vantix AI

## Setup

```bash
make dev                 # locked development dependencies, editable install
pip install pre-commit && pre-commit install
```

## Workflow

1. Branch from `main`; keep pull requests focused and under about 400 changed lines.
2. Run `make ci` before pushing. It mirrors the main CI checks: lint, types, security,
   README policy, tests with the 90 percent branch coverage gate, and the model quality gate.
3. To test against real backends, start PostgreSQL, Qdrant and OpenSearch (`docker compose up`)
   and run `make integration` with the `VANTIX_TEST_*` variables set.
4. Changing dependencies: edit `pyproject.toml`, run `make lock` (requires `uv`) and commit
   both lockfiles.
5. Changing ranking behaviour: include before and after numbers from `make quality`. Lowering
   a threshold in `scripts/quality_thresholds.json` requires an explicit justification.
6. Database changes must stay backward compatible with the currently deployed version, because
   migrations run before new pods roll out.

## Releasing

1. Update the version in `pyproject.toml`, `src/vantix/__init__.py` and `deploy/helm/vantix/Chart.yaml`
   (`version` and `appVersion`), and add a `CHANGELOG.md` entry.
2. Merge to `main` and wait for CD to deploy the commit to staging.
3. Tag the merged commit: `git tag v1.2.0 && git push origin v1.2.0`.
   The release workflow verifies versions, reruns CI, promotes the tested image, publishes
   artifacts and deploys to production after approval.
