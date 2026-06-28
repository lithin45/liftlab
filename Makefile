# LiftLab developer entry points
# Local targets run via `uv` (fast, used by CI). `up`/`down` use Docker Compose
# for the full one-command stack.

.DEFAULT_GOAL := help
.PHONY: help install lock data data-force simulate test eval demo lint format clean up down build-image

UV ?= uv

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync the local virtualenv (main + dev groups).
	$(UV) sync

lock: ## Re-resolve and write uv.lock.
	$(UV) lock

data: ## Build the data layer: fetch/generate population -> DuckDB -> dbt fact tables.
	$(UV) run liftlab build

data-force: ## Rebuild the data layer from scratch (re-fetch/regenerate; needed when switching data.source).
	$(UV) run liftlab build --force-data

simulate: ## Run the synthetic experiment simulation (Phase 2+).
	$(UV) run liftlab simulate

test: ## Run the test suite.
	$(UV) run pytest

eval: ## Run the Monte-Carlo validation gates (coverage, A/A FPR, CUPED, SRM).
	$(UV) run liftlab eval

demo: ## Build data, run a single experiment, and print a decision card.
	$(UV) run liftlab demo

lint: ## Lint with ruff.
	$(UV) run ruff check .

format: ## Format with ruff.
	$(UV) run ruff format .

clean: ## Remove build artifacts, caches, the warehouse, and the raw population (keeps .gitkeep).
	rm -rf dbt/target dbt/logs dbt/dbt_packages logs runs .pytest_cache .ruff_cache
	rm -f data/warehouse/*.duckdb data/warehouse/*.duckdb.wal
	rm -f data/raw/*.csv data/raw/MANIFEST.json

up: ## Bring up the full stack (pipeline + Streamlit UI) via Docker Compose.
	docker compose up --build

down: ## Tear down the Docker Compose stack and volumes.
	docker compose down -v

build-image: ## Build the Docker image only.
	docker compose build
