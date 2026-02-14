GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

PYTHON_EXEC ?= uv run
RUFF_CMD ?= ruff
PYTEST_CMD ?= pytest
UV_CMD ?= uv

SRC_DIR ?= src
DOCS_DIR ?= docs
TESTS_DIR ?= tests

CORE_TEST_DIR ?= tests/unit_tests
INTEGRATION_TEST_DIR ?= tests/integration_tests

PACKAGE_NAME ?= ragrank

.PHONY: help format lint clean test code_coverage install_deps dependency_check build_dist build_docs

help: ## Show all Makefile targets
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

format: ## Running code formatter: ruff
	@echo "(ruff) Formatting the project..."
	@$(PYTHON_EXEC) $(RUFF_CMD) check --select I --fix .
	@$(PYTHON_EXEC) $(RUFF_CMD) format $(SRC_DIR) $(DOCS_DIR) $(TESTS_DIR)

lint: ## Running the linter: ruff
	@echo "(ruff) Linting the project ..."
	@$(PYTHON_EXEC) $(RUFF_CMD) check $(SRC_DIR)

lint-test: ## Running the linter: ruff
	@echo "(ruff) Linting the project with test files ..."
	@$(PYTHON_EXEC) $(RUFF_CMD) check $(SRC_DIR) $(TESTS_DIR)

clean: ## Clean all generated files
	@echo "Cleaning all temporary files..."
	@git clean -xdf

test: ## Run all tests (requires OPENAI_API_KEY)
	@echo "(pytest) Running all tests..."
	@$(PYTHON_EXEC) $(PYTEST_CMD) -v $(TESTS_DIR)

test-offline: ## Run tests that don't require OpenAI API key
	@echo "(pytest) Running offline tests..."
	@$(PYTHON_EXEC) $(PYTEST_CMD) -v -m "not openai" $(TESTS_DIR)

test-openai: ## Run tests that require OpenAI API key
	@echo "(pytest) Running OpenAI tests..."
	@$(PYTHON_EXEC) $(PYTEST_CMD) -v -m "openai" $(TESTS_DIR)

test-core: # Run the unit tests
	@echo "(pytest) Running test on core files"
	@$(PYTHON_EXEC) $(PYTEST_CMD) -v $(CORE_TEST_DIR)

test-integration: # Run the integration tests
	@echo "(pytest) Running test on integrations"
	@$(PYTHON_EXEC) $(PYTEST_CMD) -v $(INTEGRATION_TEST_DIR)

code_coverage: ## Run code coverage analysis
	@echo "Running code coverage analysis..."
	@$(PYTHON_EXEC) $(PYTEST_CMD) --cov=$(PACKAGE_NAME) $(TESTS_DIR)/

install_deps: ## Install dependencies
	@echo "Installing project dependencies..."
	@$(UV_CMD) sync
	@if [ "$(INSTALL_DEV)" = true ]; then \
		echo "Installing development dependencies..."; \
		$(UV_CMD) sync --group dev; \
	fi
	@if [ "$(INSTALL_DOCS)" = true ]; then \
		echo "Installing documentation dependencies..."; \
		$(UV_CMD) sync --group docs; \
	fi

dependency_check: ## Check for outdated dependencies
	@echo "Checking for outdated dependencies..."
	@$(UV_CMD) pip list --outdated

build_dist: ## Build distribution packages
	@echo "Building distribution packages..."
	@$(UV_CMD) build

build_docs: ## Build the documentation
	@$(UV_CMD) export --group docs --no-hashes -o $(DOCS_DIR)/requirements.txt
	@echo "Building the documentation ..."
	@$(PYTHON_EXEC) sphinx-build -M html $(DOCS_DIR)/docs $(DOCS_DIR)/docs/_build/
	@echo "Building the API reference..."
	@$(PYTHON_EXEC) sphinx-build -M html $(DOCS_DIR)/api_reference $(DOCS_DIR)/api_reference/_build/
