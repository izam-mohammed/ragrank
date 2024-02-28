GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Running code formatter: black and isort
	@echo "(isort) Ordering imports..."
	@poetry run isort .
	@echo "(black) Formatting codebase..."
	@poetry run black .
	@echo "(ruff) Linting development project..."
	@poetry run ruff .

clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	@cd $(GIT_ROOT)/docs && make clean
	@cd $(GIT_ROOT) || exit 1
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

test: ## Run tests
	@echo "Running tests..."
	@poetry run pytest tests/ $(shell if [ -n "$(k)" ]; then echo "-k $(k)"; fi)


