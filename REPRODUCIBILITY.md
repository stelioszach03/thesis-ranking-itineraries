# Reproducibility Guide

This guide ensures reproducible results for the NYC Itinerary Planning benchmarks, addressing the research goal of improving upon TravelPlanner's 0.6% baseline.

## Overview

Our system implements dynamic itinerary planning algorithms based on:
- Basu Roy et al. (2011) - Interactive Itinerary Planning
- Research framework from "Ranking Itineraries: Dynamic algorithms meet user preferences"
- TravelPlanner benchmark methodology

## Environment Setup

### Option 1: Conda Environment (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate nyc-itinerary-planning

# Verify installation
python -c "import hybrid_planner; print('Setup successful')"
```

### Option 2: Python Virtual Environment

```bash
# Create virtual environment
python3.8 -m venv venv

# Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Docker

```bash
# Build Docker image
docker build -t nyc-itinerary-planning .

# Run benchmarks in container
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/benchmarks/results:/app/benchmarks/results \
           nyc-itinerary-planning
```

## Data Preparation

### 1. Generate NYC POI Data

```bash
# Generate full dataset (requires ~2GB disk space)
python prepare_nyc_data.py

# Generate small dataset for testing
python prepare_nyc_data.py --small --n_pois 100
```

Expected output:
- `data/nyc_pois.json`: POI data with categories and attributes
- `data/distance_matrix.npy`: Pre-computed distances
- `data/user_profiles.json`: Tourist profiles
- `data/spatial_index.idx/.dat`: R-tree index files

### 2. Generate Benchmark Scenarios

```bash
# Generate scenarios (already included in repo)
python benchmarks/scenarios/nyc_tourist_profiles.py
```

Output: `benchmarks/scenarios/nyc_benchmark_scenarios.json`

## Running Benchmarks

### Quick Test (10 scenarios)

```bash
python benchmarks/benchmark_runner.py \
    --scenarios 10 \
    --algorithms greedy,two_phase \
    --output benchmarks/results/quick_test.csv
```

### Full Benchmark Suite

```bash
# Set random seeds
export PYTHONHASHSEED=42

# Run full benchmarks (may take 2-3 hours)
python benchmarks/benchmark_runner.py \
    --scenarios all \
    --algorithms greedy,heap_greedy,astar,two_phase,auto \
    --parallel \
    --output benchmarks/results/full_benchmark.csv
```

### Reproduce Specific Results

```python
# In Python script
import random
import numpy as np
import yaml

# Load seeds
with open('benchmarks/seeds.yaml', 'r') as f:
    seeds = yaml.safe_load(f)

# Set all seeds
random.seed(seeds['global']['python_random'])
np.random.seed(seeds['global']['numpy'])

# Run specific scenario
from benchmarks.benchmark_runner import BenchmarkRunner
runner = BenchmarkRunner(
    pois_file="data/nyc_pois.json",
    distance_matrix_file="data/distance_matrix.npy",
    scenarios_file="benchmarks/scenarios/nyc_benchmark_scenarios.json"
)

result = runner.run_single_scenario(
    scenario_id="NYC_0042",
    algorithm="two_phase"
)
```

## Verifying Results

### 1. Check Success Rates

```bash
python scripts/verify_results.py benchmarks/results/full_benchmark.csv
```

Expected output:
- Overall success rate > 90% (vs TravelPlanner's 0.6%)
- Greedy algorithm: O(n²) complexity verified
- Two-phase approach: Best quality/performance trade-off

### 2. Statistical Tests

```python
# Compare algorithm performance
from benchmarks.benchmark_runner import BenchmarkRunner
import pandas as pd

df = pd.read_csv('benchmarks/results/full_benchmark.csv')
runner = BenchmarkRunner(...)
analysis = runner.analyze_results(df)

# Should show statistical significance (p < 0.05) for:
# - Two-phase vs Greedy on CSS score
# - A* vs Greedy on solution quality
```

### 3. Performance Baselines

Key metrics to verify:
- **Success Rate**: >90% (150x better than TravelPlanner)
- **Runtime**: <1s for typical scenarios
- **CSS Score**: >0.7 average
- **Memory**: <100MB for 1000 POIs

## Generating Paper Results

### LaTeX Tables

```bash
python scripts/generate_tables.py \
    --results benchmarks/results/full_benchmark.csv \
    --output thesis/tables/
```

Generates:
- `algorithm_comparison.tex`: Performance comparison table
- `significance_tests.tex`: Statistical test results
- `complexity_analysis.tex`: Complexity verification

### Figures

```bash
python scripts/generate_figures.py \
    --results benchmarks/results/full_benchmark.csv \
    --output thesis/figures/
```

Generates:
- Performance vs problem size plots
- CSS score distributions
- Runtime complexity graphs

### Interactive Dashboard

```bash
# Generate dashboard
python benchmarks/benchmark_runner.py --generate-dashboard

# View dashboard
open benchmarks/results/interactive_dashboard.html
```

## Common Issues

### 1. Memory Errors with Large Datasets

Solution: Use memory-bounded algorithms
```python
config = PlannerConfig(
    memory_limit_mb=500,  # Limit memory usage
    enable_caching=False  # Disable caching for benchmarks
)
```

### 2. Inconsistent Results

Check:
- All random seeds are set
- No system-dependent operations
- Disable parallel execution for debugging

### 3. Missing Dependencies

```bash
# Verify all dependencies
pip check

# Install missing spatial libraries
sudo apt-get install libspatialindex-dev libgeos-dev
```

## Continuous Integration

GitHub Actions automatically:
1. Runs unit tests on every push
2. Executes mini-benchmarks on PRs
3. Runs full benchmarks weekly
4. Checks for performance regressions

## Contact

For questions about reproducibility:
- Create an issue on GitHub
- Email: [research contact]

## Citation

If you use this code, please cite:
```bibtex
@thesis{zacharioudakis2025ranking,
  title={Ranking Itineraries: Dynamic algorithms meet user preferences},
  author={Zacharioudakis, Stelios},
  year={2025},
  school={National and Kapodistrian University of Athens}
}
```