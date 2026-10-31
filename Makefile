.PHONY: help install dev lock test integration lint format typecheck security readme quality \
        smoke load docker up pin ci clean

PYTHON ?= python
IMAGE ?= vantix:local

help:
	@grep -E '^[a-z_]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install runtime dependencies from the lock and the package
	$(PYTHON) -m pip install --require-hashes --no-deps -r requirements/runtime.lock
	$(PYTHON) -m pip install --no-deps .

dev: ## Install development dependencies from the lock and the package in editable mode
	$(PYTHON) -m pip install -r requirements/dev.lock
	$(PYTHON) -m pip install --no-deps -e .

lock: ## Regenerate requirements lockfiles from pyproject.toml (requires uv)
	uv pip compile pyproject.toml --extra postgres --universal --python-version 3.11 \
	  --generate-hashes -q -o requirements/runtime.lock --custom-compile-command "make lock"
	uv pip compile pyproject.toml --extra postgres --extra dev --universal --python-version 3.11 \
	  -q -o requirements/dev.lock --custom-compile-command "make lock"

test: ## Unit, service and API tests with the coverage gate
	pytest --cov=vantix --cov-report=term

integration: ## Full suite against real backends: set VANTIX_TEST_DATABASE_URL, VANTIX_TEST_QDRANT_URL, VANTIX_TEST_OPENSEARCH_URL
	pytest --cov-fail-under=0

lint: ## Ruff lint and format check
	ruff check src tests examples scripts
	ruff format --check src tests examples scripts

format: ## Apply formatting and safe lint fixes
	ruff format src tests examples scripts
	ruff check --fix src tests examples scripts

typecheck: ## Static type checking
	mypy src

security: ## Bandit and dependency vulnerability audit
	bandit -c pyproject.toml -r src -q
	pip-audit --strict --disable-pip -r requirements/runtime.lock

readme: ## README house style and Mermaid rendering checks
	$(PYTHON) scripts/check_readme.py README.md
	cd scripts/mermaid && npm ci --no-audit --no-fund --silent && node validate.mjs ../../README.md

quality: ## Model quality gate
	$(PYTHON) scripts/quality_gate.py reports

smoke: ## Smoke test a running instance: make smoke URL=http://localhost:8080 KEY=demo_key
	$(PYTHON) scripts/smoke_test.py $(URL) $(KEY) $(SECRET)

load: ## Load test a running instance: make load URL=http://localhost:8080 KEY=demo_key
	$(PYTHON) scripts/load_test.py $(URL) $(KEY)

docker: ## Build the production image
	docker build -t $(IMAGE) --build-arg VERSION=$$($(PYTHON) -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])") \
	  --build-arg REVISION=$$(git rev-parse HEAD 2>/dev/null || echo local) .

up: ## Start the production shaped local stack
	docker compose up --build

pin: ## Pin every GitHub Action to a commit SHA
	$(PYTHON) scripts/pin_actions.py

ci: lint typecheck security readme test quality ## Run the main CI checks locally

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml junit*.xml reports dist build
