"""
Generate Thesis Results and Comparisons

Creates LaTeX tables and figures comparing:
- Our algorithms vs Basu Roy et al. baselines
- Performance vs TravelPlanner benchmark
- Statistical significance tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
from datetime import datetime

# Set style for thesis figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def generate_algorithm_comparison_table(df: pd.DataFrame, output_dir: str):
    """
    Generate LaTeX table comparing algorithm performance
    
    Table 1 in thesis: Algorithm Performance on NYC Tourist Scenarios
    """
    # Calculate metrics for each algorithm
    algorithms = ['greedy', 'heap_greedy', 'astar', 'two_phase', 'auto']
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Algorithm Performance Comparison on NYC Tourist Benchmark (n=384 scenarios)}
\label{tab:algorithm_performance}
\begin{tabular}{lrrrrrr}
\toprule
\multirow{2}{*}{Algorithm} & \multirow{2}{*}{Success Rate} & \multicolumn{2}{c}{Runtime} & \multicolumn{3}{c}{Solution Quality} \\
\cmidrule(lr){3-4} \cmidrule(lr){5-7}
 & & Mean (s) & Std (s) & CSS Score & Diversity & POIs \\
\midrule
"""
    
    for algo in algorithms:
        algo_df = df[df['algorithm'] == algo]
        
        # Success rate
        success_rate = (algo_df['success'].sum() / len(algo_df)) * 100 if len(algo_df) > 0 else 0
        
        # Runtime stats
        runtime_mean = algo_df[algo_df['runtime'] > 0]['runtime'].mean()
        runtime_std = algo_df[algo_df['runtime'] > 0]['runtime'].std()
        
        # Quality metrics
        css_scores = []
        diversity_scores = []
        for _, row in algo_df.iterrows():
            if isinstance(row.get('metrics'), dict):
                css_scores.append(row['metrics'].get('css_score', 0))
                diversity_scores.append(row['metrics'].get('diversity', 0))
        
        css_mean = np.mean(css_scores) if css_scores else 0
        div_mean = np.mean(diversity_scores) if diversity_scores else 0
        poi_mean = algo_df['n_pois'].mean() if len(algo_df) > 0 else 0
        
        # Algorithm name formatting
        algo_name = {
            'greedy': 'Greedy',
            'heap_greedy': 'HeapPrunGreedy',
            'astar': 'A*',
            'two_phase': 'Two-Phase',
            'auto': 'Hybrid (Auto)'
        }.get(algo, algo)
        
        # Add row
        latex += f"{algo_name} & {success_rate:.1f}\\% & {runtime_mean:.3f} & {runtime_std:.3f} & "
        latex += f"{css_mean:.3f} & {div_mean:.3f} & {poi_mean:.1f} \\\\\n"
    
    latex += r"""
\midrule
TravelPlanner\textsuperscript{a} & 0.6\% & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textsuperscript{a} Baseline from Xie et al. (2024) on similar complexity queries
\end{tablenotes}
\end{table}
"""
    
    # Save to file
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'algorithm_comparison.tex'), 'w') as f:
        f.write(latex)
    
    print(f"Generated algorithm comparison table: {output_dir}/algorithm_comparison.tex")


def generate_statistical_significance_table(df: pd.DataFrame, output_dir: str):
    """
    Generate table of statistical significance tests
    
    Table 2 in thesis: Pairwise Statistical Tests
    """
    algorithms = ['greedy', 'heap_greedy', 'astar', 'two_phase']
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Statistical Significance of CSS Score Differences (Welch's t-test)}
\label{tab:significance}
\begin{tabular}{llrrr}
\toprule
Algorithm 1 & Algorithm 2 & Mean Diff. & t-statistic & p-value \\
\midrule
"""
    
    # Pairwise comparisons
    for i in range(len(algorithms)):
        for j in range(i+1, len(algorithms)):
            algo1, algo2 = algorithms[i], algorithms[j]
            
            # Extract CSS scores
            scores1 = []
            scores2 = []
            
            for _, row in df[df['algorithm'] == algo1].iterrows():
                if isinstance(row.get('metrics'), dict):
                    scores1.append(row['metrics'].get('css_score', 0))
            
            for _, row in df[df['algorithm'] == algo2].iterrows():
                if isinstance(row.get('metrics'), dict):
                    scores2.append(row['metrics'].get('css_score', 0))
            
            if len(scores1) > 1 and len(scores2) > 1:
                # Perform Welch's t-test
                t_stat, p_value = stats.ttest_ind(scores1, scores2, equal_var=False)
                mean_diff = np.mean(scores1) - np.mean(scores2)
                
                # Format algorithm names
                name1 = {'greedy': 'Greedy', 'heap_greedy': 'HeapPrunGreedy',
                        'astar': 'A*', 'two_phase': 'Two-Phase'}.get(algo1, algo1)
                name2 = {'greedy': 'Greedy', 'heap_greedy': 'HeapPrunGreedy',
                        'astar': 'A*', 'two_phase': 'Two-Phase'}.get(algo2, algo2)
                
                # Significance marker
                sig = "$^{***}$" if p_value < 0.001 else "$^{**}$" if p_value < 0.01 else "$^{*}$" if p_value < 0.05 else ""
                
                latex += f"{name1} & {name2} & {mean_diff:+.4f} & {t_stat:.3f} & {p_value:.4f}{sig} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Significance levels: $^{*}p < 0.05$, $^{**}p < 0.01$, $^{***}p < 0.001$
\end{tablenotes}
\end{table}
"""
    
    with open(os.path.join(output_dir, 'significance_tests.tex'), 'w') as f:
        f.write(latex)
    
    print(f"Generated significance tests table: {output_dir}/significance_tests.tex")


def generate_complexity_analysis_figure(df: pd.DataFrame, output_dir: str):
    """
    Generate figure showing algorithm complexity scaling
    
    Figure 1 in thesis: Runtime Complexity Analysis
    """
    # Group by number of POIs and algorithm
    df['n_pois_bin'] = pd.cut(df['n_pois'], bins=[0, 3, 5, 7, 10], labels=['1-3', '4-5', '6-7', '8+'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Runtime vs problem size
    runtime_data = []
    for algo in ['greedy', 'heap_greedy', 'astar', 'two_phase']:
        algo_df = df[df['algorithm'] == algo]
        for bin_label in ['1-3', '4-5', '6-7', '8+']:
            bin_df = algo_df[algo_df['n_pois_bin'] == bin_label]
            if len(bin_df) > 0:
                runtime_data.append({
                    'Algorithm': algo.replace('_', ' ').title(),
                    'POIs': bin_label,
                    'Runtime': bin_df['runtime'].mean()
                })
    
    runtime_df = pd.DataFrame(runtime_data)
    runtime_pivot = runtime_df.pivot(index='POIs', columns='Algorithm', values='Runtime')
    runtime_pivot.plot(kind='bar', ax=ax1)
    
    ax1.set_xlabel('Number of POIs')
    ax1.set_ylabel('Average Runtime (seconds)')
    ax1.set_title('Runtime Scaling with Problem Size')
    ax1.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
    
    # Quality vs runtime trade-off
    for algo in ['greedy', 'heap_greedy', 'astar', 'two_phase']:
        algo_df = df[df['algorithm'] == algo]
        css_scores = []
        runtimes = []
        
        for _, row in algo_df.iterrows():
            if isinstance(row.get('metrics'), dict) and row['runtime'] > 0:
                css_scores.append(row['metrics'].get('css_score', 0))
                runtimes.append(row['runtime'])
        
        if css_scores and runtimes:
            ax2.scatter(runtimes, css_scores, label=algo.replace('_', ' ').title(), alpha=0.6, s=50)
    
    ax2.set_xlabel('Runtime (seconds)')
    ax2.set_ylabel('CSS Score')
    ax2.set_title('Quality vs Runtime Trade-off')
    ax2.legend()
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'complexity_analysis.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'complexity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated complexity analysis figure: {output_dir}/complexity_analysis.pdf")


def generate_profile_performance_figure(df: pd.DataFrame, output_dir: str):
    """
    Generate figure showing performance across tourist profiles
    
    Figure 2 in thesis: Algorithm Performance by Tourist Profile
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Metrics to plot
    metrics = ['success_rate', 'css_score', 'n_pois', 'runtime']
    titles = ['Success Rate', 'CSS Score', 'POIs Selected', 'Runtime (s)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        if metric == 'success_rate':
            # Calculate success rate per profile/algorithm
            data = df.groupby(['profile', 'algorithm'])['success'].mean() * 100
            data = data.reset_index()
            data.columns = ['profile', 'algorithm', 'value']
        elif metric == 'css_score':
            # Extract CSS scores
            data_list = []
            for _, row in df.iterrows():
                if isinstance(row.get('metrics'), dict):
                    data_list.append({
                        'profile': row['profile'],
                        'algorithm': row['algorithm'],
                        'value': row['metrics'].get('css_score', 0)
                    })
            data = pd.DataFrame(data_list)
            data = data.groupby(['profile', 'algorithm'])['value'].mean().reset_index()
        else:
            data = df.groupby(['profile', 'algorithm'])[metric].mean().reset_index()
            data.columns = ['profile', 'algorithm', 'value']
        
        # Create grouped bar plot
        profiles = data['profile'].unique()
        x = np.arange(len(profiles))
        width = 0.15
        
        algorithms = ['greedy', 'heap_greedy', 'astar', 'two_phase']
        colors = sns.color_palette("husl", len(algorithms))
        
        for i, algo in enumerate(algorithms):
            algo_data = data[data['algorithm'] == algo]
            values = [algo_data[algo_data['profile'] == p]['value'].values[0] 
                     if len(algo_data[algo_data['profile'] == p]) > 0 else 0 
                     for p in profiles]
            ax.bar(x + i*width, values, width, label=algo.replace('_', ' ').title(), color=colors[i])
        
        ax.set_xlabel('Tourist Profile')
        ax.set_ylabel(title)
        ax.set_title(f'{title} by Tourist Profile')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([p.replace(' ', '\n') for p in profiles], rotation=0, ha='center')
        
        if idx == 0:  # Only show legend on first plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'profile_performance.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'profile_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated profile performance figure: {output_dir}/profile_performance.pdf")


def generate_summary_statistics(df: pd.DataFrame, output_dir: str):
    """Generate summary statistics for thesis text"""
    
    summary = {
        'total_scenarios': len(df['scenario_id'].unique()),
        'total_runs': len(df),
        'overall_success_rate': (df['success'].sum() / len(df)) * 100,
        'travelplanner_improvement': (df['success'].sum() / len(df)) / 0.006,  # vs 0.6% baseline
        'algorithms_tested': len(df['algorithm'].unique()),
        'profiles_tested': len(df['profile'].unique()) if 'profile' in df.columns else 0
    }
    
    # Best performing algorithm
    algo_success = df.groupby('algorithm')['success'].mean()
    summary['best_algorithm'] = algo_success.idxmax()
    summary['best_success_rate'] = algo_success.max() * 100
    
    # Average metrics
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        css_scores = []
        for _, row in algo_df.iterrows():
            if isinstance(row.get('metrics'), dict):
                css_scores.append(row['metrics'].get('css_score', 0))
        
        summary[f'{algo}_avg_css'] = np.mean(css_scores) if css_scores else 0
        summary[f'{algo}_avg_runtime'] = algo_df[algo_df['runtime'] > 0]['runtime'].mean()
    
    # Save as JSON
    with open(os.path.join(output_dir, 'summary_statistics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate LaTeX macros
    latex_macros = "% Benchmark Statistics Macros\n"
    latex_macros += f"\\newcommand{{\\totalscenarios}}{{{summary['total_scenarios']}}}\n"
    latex_macros += f"\\newcommand{{\\overallsuccessrate}}{{{summary['overall_success_rate']:.1f}\\%}}\n"
    latex_macros += f"\\newcommand{{\\travelplannerimprovement}}{{{summary['travelplanner_improvement']:.0f}x}}\n"
    latex_macros += f"\\newcommand{{\\bestalgorithm}}{{{summary['best_algorithm'].replace('_', ' ').title()}}}\n"
    latex_macros += f"\\newcommand{{\\bestsuccessrate}}{{{summary['best_success_rate']:.1f}\\%}}\n"
    
    with open(os.path.join(output_dir, 'thesis_macros.tex'), 'w') as f:
        f.write(latex_macros)
    
    print(f"Generated summary statistics: {output_dir}/summary_statistics.json")
    print(f"Generated LaTeX macros: {output_dir}/thesis_macros.tex")
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate thesis results from benchmark data')
    parser.add_argument('--results', type=str, required=True, help='Path to benchmark results CSV')
    parser.add_argument('--output', type=str, default='benchmarks/thesis_results', 
                       help='Output directory for thesis materials')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}")
    df = pd.read_csv(args.results)
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Generate all outputs
    print("\nGenerating thesis materials...")
    
    # Tables
    generate_algorithm_comparison_table(df, args.output)
    generate_statistical_significance_table(df, args.output)
    
    # Figures
    generate_complexity_analysis_figure(df, args.output)
    generate_profile_performance_figure(df, args.output)
    
    # Summary statistics
    summary = generate_summary_statistics(df, args.output)
    
    print("\n=== Summary ===")
    print(f"Total scenarios: {summary['total_scenarios']}")
    print(f"Overall success rate: {summary['overall_success_rate']:.1f}%")
    print(f"Improvement over TravelPlanner: {summary['travelplanner_improvement']:.0f}x")
    print(f"Best algorithm: {summary['best_algorithm']}")
    print(f"Best success rate: {summary['best_success_rate']:.1f}%")
    
    print(f"\nAll thesis materials generated in: {args.output}/")
    print("\nTo include in thesis:")
    print("1. Copy .tex files to your thesis/tables/ directory")
    print("2. Copy .pdf figures to your thesis/figures/ directory")
    print("3. Include thesis_macros.tex in your preamble")
    print("4. Use \\totalscenarios, \\overallsuccessrate etc. in your text")