# NYC Itinerary Ranking - Build Automation

.PHONY: all test clean install demo benchmark thesis data help

# Default target
all: install test

# Install the package in development mode
install:
	pip install -e .

# Run all tests
test:
	python -m pytest tests/ -v

# Clean build artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf build dist *.egg-info
	rm -rf htmlcov .coverage
	rm -rf .mypy_cache

# Run the Flask demo
demo:
	python demo/demo_nyc.py

# Run the Streamlit demo
streamlit:
	streamlit run demo/streamlit_demo.py

# Run benchmarks
benchmark:
	python benchmarks/benchmark_runner.py --scenarios 10

# Generate NYC data
data:
	python src/prepare_nyc_data.py

# Compile thesis
thesis:
	cd thesis && pdflatex thesis_final.tex
	cd thesis && biber thesis_final
	cd thesis && pdflatex thesis_final.tex
	cd thesis && pdflatex thesis_final.tex

# Run linting
lint:
	flake8 src/ tests/ --max-line-length=120
	black --check src/ tests/

# Format code
format:
	black src/ tests/

# Type checking
typecheck:
	mypy src/ --ignore-missing-imports

# Generate documentation
docs:
	cd docs && sphinx-build -b html . _build

# Run user study
user-study:
	cd user_study && python run_study.py

# Create distribution
dist: clean
	python setup.py sdist bdist_wheel

# Upload to PyPI (test)
upload-test: dist
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Help target
help:
	@echo "NYC Itinerary Ranking - Available targets:"
	@echo "  make install     - Install package in development mode"
	@echo "  make test        - Run all tests"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make demo        - Run Flask demo application"
	@echo "  make streamlit   - Run Streamlit demo"
	@echo "  make benchmark   - Run performance benchmarks"
	@echo "  make data        - Generate NYC POI data"
	@echo "  make thesis      - Compile LaTeX thesis"
	@echo "  make lint        - Check code style"
	@echo "  make format      - Format code with black"
	@echo "  make typecheck   - Run type checking"
	@echo "  make docs        - Generate documentation"
	@echo "  make user-study  - Run user study interface"
	@echo "  make dist        - Create distribution packages"
	@echo "  make help        - Show this help message"