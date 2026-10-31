# Changelog

All notable changes to Vantix AI are documented here. The format follows Keep a Changelog and
the project uses semantic versioning.

## [Unreleased]

### Added
* Complete CI/CD: lint, types, security, README policy, multi version tests with a branch
  coverage gate, integration tests against PostgreSQL, Qdrant and OpenSearch, model quality gate,
  container build, scan and smoke test, Helm validation, signed multi architecture images,
  staging and production deployments with automatic rollback, nightly compatibility and load tests.
* Helm chart with API, training worker and migration job.
* `vantix migrate`, `vantix worker` and `vantix health` commands.
* `DELETE /v1/users/{user_id}/interactions` for right to erasure.
* Hot reload of models persisted by another process; database driven training in worker mode.
* Hash verified lockfiles for runtime and development dependencies.

### Fixed
* Event batches accepted while the ingestion queue was full were marked processed, so retries
  were dropped as duplicates (data loss).
* Malformed event batches returned 500 instead of 422.
* ALS fold in recomputed the item Gram matrix on every request.
* Interaction iteration used OFFSET pagination (quadratic); now keyset pagination.
* Interaction inserts issued one query per event; now bulk.
* Races on concurrent first writes of items, tenant state and schema creation.
* Caller supplied request ids are validated before being logged.
* README diagrams no longer have any fill or edge label backgrounds.

## [1.0.0]

### Added
* Initial hybrid recommender and RAG engine.
