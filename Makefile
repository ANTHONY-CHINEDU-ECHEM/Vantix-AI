.PHONY: install dev test lint format typecheck demo serve evaluate docker up sample

install:
	pip install .

dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests examples
	ruff format --check src tests examples

format:
	ruff format src tests examples
	ruff check --fix src tests examples

typecheck:
	mypy src

sample:
	python examples/generate_sample_data.py

demo:
	vantix demo

serve:
	vantix serve

evaluate:
	vantix evaluate retrieval examples/data/eval_queries.jsonl
	vantix evaluate recommender

docker:
	docker build -t vantix:latest .

up:
	docker compose up --build
