.PHONY: test test-fast test-cov test-all clean lint help

# Default target
help:
	@echo "Machine-POI Test Commands"
	@echo "========================="
	@echo "make test       - Run all fast tests (excludes slow/integration)"
	@echo "make test-all   - Run ALL tests including slow/integration"
	@echo "make test-cov   - Run tests with coverage report"
	@echo "make test-file  - Run specific test file (FILE=tests/test_xxx.py)"
	@echo "make test-match - Run tests matching pattern (MATCH=pattern)"
	@echo "make clean      - Remove cache and temp files"
	@echo "make lint       - Run code linting (if available)"

# Run fast tests only (exclude slow and integration)
test:
	python -m pytest -m "not slow and not integration" -v

# Run specific test file
test-file:
	python -m pytest $(FILE) -v

# Run tests matching a pattern
test-match:
	python -m pytest -k "$(MATCH)" -v

# Run all tests including slow ones
test-all:
	python -m pytest -v

# Run with coverage
test-cov:
	python -m pytest -m "not slow and not integration" --cov=src --cov-report=term-missing --cov-report=html

# Run tests in parallel (faster)
test-parallel:
	python -m pytest -m "not slow and not integration" -n auto

# Run only injection mode tests
test-injection:
	python -m pytest tests/test_llm_wrapper.py::TestActivationHookInjection -v

# Run only steering vector tests
test-vectors:
	python -m pytest tests/test_steering_vectors.py -v

# Run only knowledge base tests
test-kb:
	python -m pytest tests/test_knowledge_base.py -v

# Clean up
clean:
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .coverage
	rm -rf htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Lint (optional, if you have ruff/flake8)
lint:
	@if command -v ruff &> /dev/null; then \
		ruff check src tests; \
	elif command -v flake8 &> /dev/null; then \
		flake8 src tests; \
	else \
		echo "No linter found. Install ruff or flake8."; \
	fi
