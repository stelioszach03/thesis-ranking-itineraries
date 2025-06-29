"""
Benchmark Runner for NYC Itinerary Planning

Executes comprehensive benchmarks following research_context.md evaluation methodology
Measures performance, quality metrics, and statistical significance

References:
- TravelPlanner benchmark achieving 0.6% baseline
- Evaluation metrics from research_context.md
"""

import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from datetime import datetime
import logging
import pickle
import os
import sys
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import random

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics_definitions import (
    POI, Itinerary, QuantitativeMetrics, QualitativeMetrics,
    CompositeUtilityFunctions
)
from greedy_algorithms import Constraints, InteractiveFeedback
from hybrid_planner import HybridPlanner, AlgorithmType, PlannerConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkMetrics:
    """
    Comprehensive metrics based on research_context.md
    
    Includes:
    - Quantitative: runtime, memory, solution quality
    - Qualitative: diversity, novelty, personalization
    - Composite CSS scores with research weights
    """
    
    @staticmethod
    def measure_runtime(func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure function runtime"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        runtime = time.perf_counter() - start_time
        return result, runtime
    
    @staticmethod
    def measure_memory(func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure peak memory usage (simplified)"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        return result, max(0, memory_used)
    
    @staticmethod
    def calculate_solution_quality(itinerary: List[POI], 
                                 preferences: Dict[str, float],
                                 constraints: Dict) -> Dict[str, float]:
        """
        Calculate comprehensive quality metrics
        
        Based on evaluation framework from research_context.md
        """
        if not itinerary:
            return {
                'css_score': 0.0,
                'satisfaction': 0.0,
                'diversity': 0.0,
                'novelty': 0.0,
                'feasibility': 0.0,
                'efficiency': 0.0
            }
        
        # Create Itinerary object
        itin_obj = Itinerary(
            pois=itinerary,
            start_time=constraints.get('start_time', 9.0),
            transportation_mode=constraints.get('transportation_mode', 'public_transit'),
            user_preferences=preferences
        )
        
        # Calculate metrics
        metrics = {}
        
        # CSS Score (main metric from research)
        metrics['css_score'] = CompositeUtilityFunctions.composite_satisfaction_score(
            itin_obj,
            preferences,
            budget=constraints.get('budget', 100),
            max_time=constraints.get('max_time_hours', 10)
        )
        
        # Individual components
        metrics['satisfaction'] = QualitativeMetrics.user_satisfaction(
            itin_obj, preferences
        )
        metrics['diversity'] = QualitativeMetrics.diversity_score(itin_obj)
        metrics['novelty'] = QualitativeMetrics.novelty_score(itin_obj)
        
        # Efficiency metrics
        metrics['efficiency'] = QuantitativeMetrics.utility_per_time(itin_obj)
        
        # Feasibility (from composite function)
        metrics['feasibility'] = CompositeUtilityFunctions.feasibility_score(
            itin_obj,
            budget=constraints.get('budget', 100),
            max_time=constraints.get('max_time_hours', 10)
        )
        
        # Additional metrics
        metrics['total_distance'] = QuantitativeMetrics.total_distance(itin_obj)
        metrics['total_time'] = QuantitativeMetrics.total_time(itin_obj)
        metrics['total_cost'] = QuantitativeMetrics.total_cost(itin_obj)
        metrics['n_pois'] = len(itinerary)
        
        return metrics


class BenchmarkRunner:
    """
    Main benchmark execution framework
    """
    
    def __init__(self, pois_file: str, distance_matrix_file: str,
                 scenarios_file: str, output_dir: str = "benchmarks/results"):
        """Initialize benchmark runner"""
        # Load data
        with open(pois_file, 'r') as f:
            self.pois_data = json.load(f)
        
        self.distance_matrix = np.load(distance_matrix_file)
        
        with open(scenarios_file, 'r') as f:
            self.scenarios = json.load(f)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize planners with different algorithms
        config = PlannerConfig(
            enable_caching=False,  # Disable for fair comparison
            enable_parallel=True,
            enable_rtree=True
        )
        
        self.planners = {
            'greedy': HybridPlanner(self.pois_data, self.distance_matrix, config),
            'heap_greedy': HybridPlanner(self.pois_data, self.distance_matrix, config),
            'astar': HybridPlanner(self.pois_data, self.distance_matrix, config),
            'two_phase': HybridPlanner(self.pois_data, self.distance_matrix, config),
            'auto': HybridPlanner(self.pois_data, self.distance_matrix, config)
        }
        
        logger.info(f"Initialized benchmark runner with {len(self.scenarios)} scenarios")
    
    def run_single_scenario(self, scenario: Dict, algorithm: str) -> Dict:
        """Run a single benchmark scenario"""
        try:
            # Extract parameters
            preferences = scenario['profile']['preferences']
            constraints = Constraints(**scenario['constraints'])
            
            # Get planner
            planner = self.planners[algorithm]
            
            # Map algorithm name to enum
            algo_map = {
                'greedy': AlgorithmType.GREEDY,
                'heap_greedy': AlgorithmType.HEAP_GREEDY,
                'astar': AlgorithmType.ASTAR,
                'two_phase': AlgorithmType.TWO_PHASE,
                'auto': AlgorithmType.AUTO
            }
            
            # Measure performance
            (result, runtime), memory = BenchmarkMetrics.measure_memory(
                BenchmarkMetrics.measure_runtime,
                planner.plan,
                preferences,
                constraints,
                algorithm=algo_map[algorithm],
                generate_alternatives=False
            )
            
            # Calculate quality metrics
            quality_metrics = BenchmarkMetrics.calculate_solution_quality(
                result.primary_itinerary,
                preferences,
                vars(constraints)
            )
            
            # Compile results
            return {
                'scenario_id': scenario['scenario_id'],
                'algorithm': algorithm,
                'success': len(result.primary_itinerary) > 0,
                'runtime': runtime,
                'memory_mb': memory,
                'metrics': quality_metrics,
                'n_pois': len(result.primary_itinerary),
                'cache_hit': result.cache_hit,
                'algorithm_used': result.algorithm_used,
                'profile': scenario['profile']['name'],
                'duration': scenario['trip']['duration'],
                'season': scenario['trip']['season']
            }
            
        except Exception as e:
            logger.error(f"Error in scenario {scenario.get('scenario_id', 'unknown')}: {e}")
            logger.error(traceback.format_exc())
            return {
                'scenario_id': scenario.get('scenario_id', 'unknown'),
                'algorithm': algorithm,
                'success': False,
                'error': str(e),
                'runtime': -1,
                'memory_mb': -1,
                'metrics': {}
            }
    
    def run_benchmarks(self, algorithms: List[str] = None, 
                      n_scenarios: int = None,
                      parallel: bool = True) -> pd.DataFrame:
        """
        Run comprehensive benchmarks
        
        Args:
            algorithms: List of algorithms to test
            n_scenarios: Number of scenarios to run (None for all)
            parallel: Use parallel execution
            
        Returns:
            DataFrame with benchmark results
        """
        if algorithms is None:
            algorithms = ['greedy', 'heap_greedy', 'two_phase', 'auto']
        
        scenarios_to_run = self.scenarios[:n_scenarios] if n_scenarios else self.scenarios
        
        logger.info(f"Running benchmarks: {len(scenarios_to_run)} scenarios × {len(algorithms)} algorithms")
        
        results = []
        total_runs = len(scenarios_to_run) * len(algorithms)
        completed = 0
        
        if parallel:
            with ProcessPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                futures = {}
                for scenario in scenarios_to_run:
                    for algorithm in algorithms:
                        future = executor.submit(self.run_single_scenario, scenario, algorithm)
                        futures[future] = (scenario['scenario_id'], algorithm)
                
                # Collect results
                for future in as_completed(futures):
                    scenario_id, algorithm = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        if completed % 10 == 0:
                            logger.info(f"Progress: {completed}/{total_runs} ({100*completed/total_runs:.1f}%)")
                    except Exception as e:
                        logger.error(f"Failed {scenario_id} with {algorithm}: {e}")
        else:
            # Sequential execution
            for scenario in scenarios_to_run:
                for algorithm in algorithms:
                    result = self.run_single_scenario(scenario, algorithm)
                    results.append(result)
                    completed += 1
                    if completed % 10 == 0:
                        logger.info(f"Progress: {completed}/{total_runs} ({100*completed/total_runs:.1f}%)")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save raw results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        
        return df
    
    def analyze_results(self, df: pd.DataFrame) -> Dict:
        """
        Analyze benchmark results with statistical tests
        
        Based on evaluation methodology from research_context.md
        """
        analysis = {}
        
        # Overall statistics
        analysis['overall'] = {
            'total_runs': len(df),
            'successful_runs': len(df[df['success'] == True]),
            'success_rate': len(df[df['success'] == True]) / len(df) if len(df) > 0 else 0,
            'avg_runtime': df[df['runtime'] > 0]['runtime'].mean(),
            'avg_memory': df[df['memory_mb'] > 0]['memory_mb'].mean()
        }
        
        # Per-algorithm analysis
        algorithms = df['algorithm'].unique()
        analysis['algorithms'] = {}
        
        for algo in algorithms:
            algo_df = df[df['algorithm'] == algo]
            
            # Extract metrics
            metrics_data = {}
            if 'metrics' in algo_df.columns and len(algo_df) > 0:
                # Safely extract metrics
                valid_metrics = algo_df[algo_df['metrics'].apply(lambda x: isinstance(x, dict) and 'css_score' in x)]
                if len(valid_metrics) > 0:
                    for metric in ['css_score', 'satisfaction', 'diversity', 'novelty', 'efficiency']:
                        values = valid_metrics['metrics'].apply(lambda x: x.get(metric, 0)).values
                        metrics_data[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
            
            analysis['algorithms'][algo] = {
                'success_rate': len(algo_df[algo_df['success'] == True]) / len(algo_df) if len(algo_df) > 0 else 0,
                'avg_runtime': algo_df[algo_df['runtime'] > 0]['runtime'].mean() if len(algo_df[algo_df['runtime'] > 0]) > 0 else 0,
                'avg_memory': algo_df[algo_df['memory_mb'] > 0]['memory_mb'].mean() if len(algo_df[algo_df['memory_mb'] > 0]) > 0 else 0,
                'avg_pois': algo_df[algo_df['n_pois'] > 0]['n_pois'].mean() if len(algo_df[algo_df['n_pois'] > 0]) > 0 else 0,
                'metrics': metrics_data
            }
        
        # Statistical significance tests (if we have multiple algorithms)
        if len(algorithms) > 1:
            analysis['statistical_tests'] = self._run_statistical_tests(df, algorithms)
        
        # Profile-based analysis
        if 'profile' in df.columns:
            analysis['profiles'] = {}
            for profile in df['profile'].unique():
                profile_df = df[df['profile'] == profile]
                analysis['profiles'][profile] = {
                    'success_rate': len(profile_df[profile_df['success'] == True]) / len(profile_df) if len(profile_df) > 0 else 0,
                    'avg_pois': profile_df[profile_df['n_pois'] > 0]['n_pois'].mean() if len(profile_df[profile_df['n_pois'] > 0]) > 0 else 0
                }
        
        return analysis
    
    def _run_statistical_tests(self, df: pd.DataFrame, algorithms: List[str]) -> Dict:
        """Run statistical significance tests between algorithms"""
        tests = {}
        
        # Pairwise comparisons for CSS scores
        for i in range(len(algorithms)):
            for j in range(i+1, len(algorithms)):
                algo1, algo2 = algorithms[i], algorithms[j]
                
                # Extract CSS scores
                scores1 = []
                scores2 = []
                
                df1 = df[(df['algorithm'] == algo1) & (df['success'] == True)]
                df2 = df[(df['algorithm'] == algo2) & (df['success'] == True)]
                
                if 'metrics' in df1.columns:
                    scores1 = df1['metrics'].apply(lambda x: x.get('css_score', 0) if isinstance(x, dict) else 0).values
                if 'metrics' in df2.columns:
                    scores2 = df2['metrics'].apply(lambda x: x.get('css_score', 0) if isinstance(x, dict) else 0).values
                
                if len(scores1) > 0 and len(scores2) > 0:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(scores1, scores2)
                    
                    tests[f"{algo1}_vs_{algo2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'mean_diff': np.mean(scores1) - np.mean(scores2),
                        'effect_size': (np.mean(scores1) - np.mean(scores2)) / np.sqrt((np.std(scores1)**2 + np.std(scores2)**2) / 2)
                    }
        
        return tests
    
    def generate_latex_tables(self, analysis: Dict, output_file: str):
        """
        Generate LaTeX tables for thesis
        
        Following academic paper format
        """
        latex_content = """% Benchmark Results Tables
% Generated for: Ranking Itineraries - Dynamic algorithms meet user preferences

\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{multirow}
\\usepackage{siunitx}

\\begin{document}

% Table 1: Algorithm Performance Comparison
\\begin{table}[htbp]
\\centering
\\caption{Algorithm Performance Comparison on NYC Tourist Scenarios}
\\label{tab:algorithm_performance}
\\begin{tabular}{lcccccc}
\\toprule
Algorithm & Success Rate & Runtime (s) & Memory (MB) & CSS Score & Diversity & Novelty \\\\
\\midrule
"""
        
        # Add algorithm results
        for algo, data in analysis.get('algorithms', {}).items():
            success_rate = data.get('success_rate', 0) * 100
            runtime = data.get('avg_runtime', 0)
            memory = data.get('avg_memory', 0)
            
            metrics = data.get('metrics', {})
            css_score = metrics.get('css_score', {}).get('mean', 0)
            diversity = metrics.get('diversity', {}).get('mean', 0)
            novelty = metrics.get('novelty', {}).get('mean', 0)
            
            latex_content += f"{algo.replace('_', ' ').title()} & {success_rate:.1f}\\% & {runtime:.3f} & {memory:.1f} & {css_score:.3f} & {diversity:.3f} & {novelty:.3f} \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}

% Table 2: Statistical Significance Tests
\\begin{table}[htbp]
\\centering
\\caption{Statistical Significance of CSS Score Differences}
\\label{tab:significance_tests}
\\begin{tabular}{lcccc}
\\toprule
Comparison & Mean Difference & t-statistic & p-value & Significant \\\\
\\midrule
"""
        
        # Add statistical tests
        for test_name, test_data in analysis.get('statistical_tests', {}).items():
            mean_diff = test_data.get('mean_diff', 0)
            t_stat = test_data.get('t_statistic', 0)
            p_value = test_data.get('p_value', 1)
            significant = "Yes" if test_data.get('significant', False) else "No"
            
            comparison = test_name.replace('_', ' vs ')
            latex_content += f"{comparison} & {mean_diff:.4f} & {t_stat:.3f} & {p_value:.4f} & {significant} \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\end{table}

\\end{document}
"""
        
        with open(output_file, 'w') as f:
            f.write(latex_content)
        
        logger.info(f"Generated LaTeX tables in {output_file}")
    
    def compare_with_baseline(self, df: pd.DataFrame) -> Dict:
        """
        Compare results with research baselines
        
        References:
        - TravelPlanner: 0.6% success rate baseline
        - Basu Roy et al.: Greedy baseline performance
        """
        comparison = {}
        
        # Overall success rate vs TravelPlanner
        our_success_rate = len(df[df['success'] == True]) / len(df) if len(df) > 0 else 0
        travelplanner_baseline = 0.006  # 0.6% from research
        
        comparison['vs_travelplanner'] = {
            'our_success_rate': our_success_rate,
            'travelplanner_baseline': travelplanner_baseline,
            'improvement_factor': our_success_rate / travelplanner_baseline if travelplanner_baseline > 0 else float('inf'),
            'improvement_percentage': (our_success_rate - travelplanner_baseline) * 100
        }
        
        # Greedy performance vs Basu Roy baseline
        greedy_df = df[df['algorithm'] == 'greedy']
        if len(greedy_df) > 0:
            greedy_runtime = greedy_df[greedy_df['runtime'] > 0]['runtime'].mean()
            n_pois = len(self.pois_data)
            
            # O(n²) complexity check
            expected_complexity = n_pois ** 2
            actual_complexity = greedy_runtime * 1000  # Convert to ms
            
            comparison['complexity_analysis'] = {
                'algorithm': 'greedy',
                'n_pois': n_pois,
                'expected_complexity': 'O(n²)',
                'actual_runtime_ms': actual_complexity,
                'complexity_ratio': actual_complexity / expected_complexity if expected_complexity > 0 else 0
            }
        
        return comparison


def generate_interactive_dashboard(results_file: str, output_file: str):
    """
    Generate interactive dashboard for results visualization
    
    Uses plotly for interactive charts
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # Load results
    df = pd.read_csv(results_file)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Success Rate by Algorithm', 'Runtime Distribution',
                       'CSS Score by Profile', 'Memory Usage'),
        specs=[[{'type': 'bar'}, {'type': 'box'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    # 1. Success rate by algorithm
    success_rates = df.groupby('algorithm')['success'].mean() * 100
    fig.add_trace(
        go.Bar(x=success_rates.index, y=success_rates.values, name='Success Rate'),
        row=1, col=1
    )
    
    # 2. Runtime distribution
    for algo in df['algorithm'].unique():
        algo_df = df[(df['algorithm'] == algo) & (df['runtime'] > 0)]
        fig.add_trace(
            go.Box(y=algo_df['runtime'], name=algo, showlegend=False),
            row=1, col=2
        )
    
    # 3. CSS Score by profile (if available)
    if 'profile' in df.columns and 'metrics' in df.columns:
        # Extract CSS scores
        css_data = []
        for _, row in df.iterrows():
            if isinstance(row['metrics'], dict) and 'css_score' in row['metrics']:
                css_data.append({
                    'profile': row.get('profile', 'Unknown'),
                    'css_score': row['metrics']['css_score'],
                    'algorithm': row['algorithm']
                })
        
        if css_data:
            css_df = pd.DataFrame(css_data)
            avg_css = css_df.groupby('profile')['css_score'].mean()
            fig.add_trace(
                go.Bar(x=avg_css.index, y=avg_css.values, name='Avg CSS Score'),
                row=2, col=1
            )
    
    # 4. Memory usage
    memory_data = df[df['memory_mb'] > 0]
    if len(memory_data) > 0:
        fig.add_trace(
            go.Scatter(
                x=memory_data['n_pois'],
                y=memory_data['memory_mb'],
                mode='markers',
                marker=dict(
                    color=memory_data['algorithm'].astype('category').cat.codes,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=memory_data['algorithm'],
                name='Memory vs POIs'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='NYC Itinerary Planning Benchmark Results',
        showlegend=True,
        height=800
    )
    
    # Save
    fig.write_html(output_file)
    logger.info(f"Generated interactive dashboard: {output_file}")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize runner
    runner = BenchmarkRunner(
        pois_file="data/nyc_pois.json",
        distance_matrix_file="data/distance_matrix.npy",
        scenarios_file="benchmarks/scenarios/nyc_benchmark_scenarios.json"
    )
    
    # Run benchmarks
    print("Starting benchmark execution...")
    results_df = runner.run_benchmarks(
        algorithms=['greedy', 'heap_greedy', 'two_phase', 'auto'],
        n_scenarios=50,  # Start with subset
        parallel=True
    )
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = runner.analyze_results(results_df)
    
    # Print summary
    print("\n=== Benchmark Results Summary ===")
    print(f"Total runs: {analysis['overall']['total_runs']}")
    print(f"Success rate: {analysis['overall']['success_rate']:.1%}")
    print(f"Average runtime: {analysis['overall']['avg_runtime']:.3f}s")
    
    print("\nPer-algorithm performance:")
    for algo, data in analysis['algorithms'].items():
        print(f"\n{algo}:")
        print(f"  Success rate: {data['success_rate']:.1%}")
        print(f"  Avg runtime: {data['avg_runtime']:.3f}s")
        if 'css_score' in data.get('metrics', {}):
            print(f"  CSS score: {data['metrics']['css_score']['mean']:.3f}")
    
    # Compare with baselines
    comparison = runner.compare_with_baseline(results_df)
    print(f"\n=== Baseline Comparison ===")
    print(f"vs TravelPlanner (0.6%): {comparison['vs_travelplanner']['improvement_factor']:.1f}x better")
    
    # Generate outputs
    runner.generate_latex_tables(analysis, "benchmarks/results/benchmark_tables.tex")
    generate_interactive_dashboard(
        f"benchmarks/results/benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "benchmarks/results/interactive_dashboard.html"
    )