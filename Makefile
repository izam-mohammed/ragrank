GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

PYTHON_EXEC ?= poetry run
RUFF_CMD ?= ruff
PYTEST_CMD ?= pytest
POETRY_CMD ?= poetry

SRC_DIR ?= src
DOCS_DIR ?= docs
TESTS_DIR ?= tests

PACKAGE_NAME ?= ragrank

.PHONY: help format lint clean test code_coverage install_deps dependency_check build_dist build_docs

help: ## Show all Makefile targets
	@echo "Usage: make [target]"
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

format: ## Running code formatter: ruff
	@echo "(ruff) Formatting the project..."
	@$(PYTHON_EXEC) $(RUFF_CMD) format $(SRC_DIR) $(DOCS_DIR) $(TESTS_DIR)

lint: ## Running the linter: ruff
	@echo "(ruff) Linting the project ..."
	@$(PYTHON_EXEC) $(RUFF_CMD) check $(SRC_DIR) 

clean: ## Clean all generated files
	@echo "Cleaning all temporary files..."
	@git clean -xdf

test: ## Run tests
	@echo "(pytest) Running tests..."
	@$(PYTHON_EXEC) $(PYTEST_CMD) -v $(TESTS_DIR)

code_coverage: ## Run code coverage analysis
	@echo "Running code coverage analysis..."
	@$(PYTHON_EXEC) $(PYTEST_CMD) --cov=$(PACKAGE_NAME) $(TESTS_DIR)/

install_deps: ## Install dependencies
	@echo "Installing project dependencies..."
	@$(POETRY_CMD) install
	@if [ "$(INSTALL_DEV)" = true ]; then \
		echo "Installing development dependencies..."; \
		$(POETRY_CMD) install --with dev; \
	fi
	@if [ "$(INSTALL_DOCS)" = true ]; then \
		echo "Installing documentation dependencies..."; \
		$(POETRY_CMD) install --with docs; \
	fi

dependency_check: ## Check for outdated dependencies
	@echo "Checking for outdated dependencies..."
	@$(POETRY_CMD) show --outdated

build_dist: ## Build distribution packages
	@echo "Building distribution packages..."
	@$(POETRY_CMD) build

build_docs: ## Build the documentation 
	@$(POETRY_CMD) export --with docs -f requirements.txt --without-hashes --output $(DOCS_DIR)/requirements.txt 
	@echo "Building the documentation ..."
	@$(PYTHON_EXEC) sphinx-build -M html $(DOCS_DIR)/docs $(DOCS_DIR)/docs/_build/
	@echo "Building the API reference..."
	@$(PYTHON_EXEC) sphinx-build -M html $(DOCS_DIR)/api_reference $(DOCS_DIR)/api_reference/_build/