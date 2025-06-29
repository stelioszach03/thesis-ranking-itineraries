# Project Structure

This document describes the organization of the NYC Itinerary Ranking thesis project.

## Directory Overview

### `/src/` - Core Algorithm Implementations
The heart of the thesis implementation containing all algorithmic contributions:
- `metrics_definitions.py`: CSS formula and quality metrics
- `greedy_algorithms.py`: O(n²) greedy heuristics (GreedyPOISelection, HeapPrunGreedyPOI)
- `astar_itinerary.py`: A* search with MST-based admissible heuristics
- `lpa_star.py`: Lifelong Planning A* for dynamic replanning (70-90% computation reuse)
- `hybrid_planner.py`: Two-phase approach combining greedy selection with optimal routing
- `prepare_nyc_data.py`: Data pipeline for processing 10,847 NYC POIs

### `/demo/` - Interactive Demonstrations
Web applications showcasing the research contributions:
- `demo_nyc.py`: Flask application with real-time NYC map
- `streamlit_demo.py`: Streamlit interface for algorithm comparison
- `templates/index.html`: Frontend interface with Leaflet.js maps
- `demo_scenarios.md`: 8 demonstration scenarios validating research claims

### `/benchmarks/` - Evaluation Framework
Comprehensive testing infrastructure:
- `scenarios/nyc_tourist_profiles.py`: 384 test scenarios (8 profiles × 3 durations × 4 seasons × 4 events)
- `benchmark_runner.py`: Automated performance testing
- `generate_thesis_results.py`: Creates publication-ready figures and tables
- `results/`: Benchmark outputs and comparisons

### `/user_study/` - Human Evaluation Materials
Complete user study protocol with ethics compliance:
- `ethics/`: GDPR compliance, consent forms, IRB templates
- `scenarios/`: 6 NYC-specific JSON scenarios
- `evaluation/`: Questionnaires and interview guides
- `analysis/`: Statistical analysis scripts (t-tests, ANOVA, effect sizes)

### `/thesis/` - LaTeX Document
Complete thesis document following NKUA requirements:
- `thesis_final.tex`: Main LaTeX file with bilingual abstracts
- `chapters/`: All 6 thesis chapters in Markdown
- `figures/`: Figure generation scripts and descriptions
- Core algorithm snippets for appendices

### `/data/` - Datasets and Resources
- `bibliography.csv`: 26 research papers with full metadata
- `nyc_data/`: Directory for NYC POI datasets (not included in git)

### `/docs/` - Additional Documentation
- `architecture.md`: System design and algorithm details
- `bibliography.bib`: BibTeX format references

### `/presentation/` - Conference Materials
- `presentation_outline.md`: 20-slide presentation structure
- `speaker_notes.md`: Detailed speaking points and Q&A preparation

### `/tests/` - Test Suite
Unit and integration tests for all algorithms (to be implemented)

## Key Files

### Configuration Files
- `setup.py`: Python package configuration with entry points
- `requirements.txt`: Python dependencies
- `environment.yml`: Conda environment specification
- `Dockerfile`: Container setup for reproducibility
- `Makefile`: Build automation

### Documentation
- `README.md`: Comprehensive project overview with badges
- `research_context.md`: Complete research framework and motivation
- `REPRODUCIBILITY.md`: Instructions for reproducing all results
- `.gitignore`: Excludes large data files and build artifacts

## Data Flow

1. **Input**: Raw POI data from OpenStreetMap/Foursquare
2. **Processing**: `prepare_nyc_data.py` → cleaned dataset with R-tree index
3. **Algorithms**: Greedy/A*/LPA*/Hybrid process user preferences
4. **Output**: Ranked itineraries with CSS scores
5. **Evaluation**: Benchmarks measure performance, user study validates quality

## Quick Navigation

- **Start here**: `README.md` for overview
- **Research background**: `research_context.md`
- **Try the demo**: `demo/demo_nyc.py`
- **Run benchmarks**: `benchmarks/benchmark_runner.py`
- **Read thesis**: `thesis/thesis_final.tex`

## Dependencies

Core algorithms require:
- Python 3.8+
- NumPy, NetworkX, Numba
- See `requirements.txt` for complete list

Demo applications additionally need:
- Flask/Streamlit
- Folium for maps
- See `demo/demo_requirements.txt`

## Citation

When using this code, please cite:
```
Zacharioudakis, S. (2025). Ranking Itineraries: Dynamic Algorithms Meet User Preferences. 
Master's Thesis, National and Kapodistrian University of Athens.
```