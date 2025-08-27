# Makefile for Self-Sycophancy Code Review Experiment

.PHONY: help build build-force run-pilot run-full clean docker-clean analyze

# Default target
help:
	@echo "Self-Sycophancy Code Review Experiment"
	@echo "======================================"
	@echo ""
	@echo "Available targets:"
	@echo "  make build        - Build Docker image (only if needed)"
	@echo "  make build-force  - Force rebuild Docker image"
	@echo "  make run-pilot    - Run pilot experiment (5 samples)"
	@echo "  make run-full     - Run full experiment"
	@echo "  make analyze      - Analyze latest results"
	@echo "  make clean        - Clean results and logs"
	@echo "  make docker-clean - Clean Docker images and cache"
	@echo ""

# Build Docker image (with caching)
build:
	@echo "Checking if Docker build is needed..."
	@./docker/build.sh

# Force rebuild Docker image
build-force:
	@echo "Force rebuilding Docker image..."
	@./docker/build.sh --force

# Run pilot experiment
run-pilot: build
	@echo "Running pilot experiment..."
	python run_experiment.py --pilot

# Run full experiment
run-full: build
	@echo "Running full experiment..."
	python run_experiment.py --samples 50

# Analyze results
analyze:
	@echo "Analyzing latest results..."
	@LATEST_DIR=$$(ls -td results/logs/*/ 2>/dev/null | head -1); \
	if [ -n "$$LATEST_DIR" ]; then \
		echo "Analyzing $$LATEST_DIR"; \
		python -m src.analysis.statistics --results-dir "$$LATEST_DIR"; \
	else \
		echo "No results found to analyze"; \
	fi

# Clean results and logs
clean:
	@echo "Cleaning results and logs..."
	rm -rf results/logs/*
	rm -rf results/reports/*
	rm -rf results/generated_prs/*
	@echo "Clean complete"

# Clean Docker images and cache
docker-clean:
	@echo "Cleaning Docker images and cache..."
	docker rmi sycophancy-experiment:latest 2>/dev/null || true
	docker system prune -f
	@echo "Docker clean complete"

# Check if Docker is running
check-docker:
	@docker info >/dev/null 2>&1 || (echo "Error: Docker is not running" && exit 1)