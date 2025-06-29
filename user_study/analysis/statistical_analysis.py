"""
Statistical Analysis for User Study

Performs statistical tests to validate:
- Significance vs baselines
- Algorithm performance differences
- Demographic effects
- Scenario difficulty variations
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestPower
import json
from pathlib import Path
from typing import Dict, Tuple, List


class StatisticalAnalyzer:
    """Perform statistical analysis on user study data"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.baseline_satisfaction = 0.82
        self.travelplanner_success = 0.006
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load processed user study data"""
        return pd.read_csv(csv_path)
    
    def test_vs_baseline(self, data: List[float], baseline: float, 
                        test_name: str) -> Dict:
        """One-sample t-test against baseline"""
        t_stat, p_value = stats.ttest_1samp(data, baseline)
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        ci = stats.t.interval(1 - self.alpha, len(data) - 1, mean, std / np.sqrt(len(data)))
        
        return {
            'test': test_name,
            'n': len(data),
            'mean': mean,
            'std': std,
            'baseline': baseline,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'confidence_interval': ci,
            'effect_size': (mean - baseline) / std if std > 0 else 0
        }
    
    def test_success_rate(self, df: pd.DataFrame) -> Dict:
        """Test success rate vs TravelPlanner baseline"""
        success_rate = df['success'].mean()
        n_trials = len(df)
        n_success = df['success'].sum()
        
        # Binomial test
        p_value = stats.binom_test(n_success, n_trials, self.travelplanner_success, 
                                  alternative='greater')
        
        # Wilson confidence interval for proportion
        z = stats.norm.ppf(1 - self.alpha/2)
        center = (n_success + z**2/2) / (n_trials + z**2)
        margin = z * np.sqrt((n_success * (n_trials - n_success) / n_trials + z**2/4) / 
                           (n_trials + z**2))
        ci = (center - margin, center + margin)
        
        return {
            'test': 'Success Rate vs TravelPlanner',
            'n_trials': n_trials,
            'n_success': n_success,
            'success_rate': success_rate,
            'baseline': self.travelplanner_success,
            'improvement_factor': success_rate / self.travelplanner_success,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'confidence_interval': ci
        }
    
    def test_satisfaction(self, df: pd.DataFrame) -> Dict:
        """Test satisfaction vs literature baseline"""
        # Convert 1-10 scale to 0-1
        satisfaction_normalized = df['satisfaction_rating'] / 10
        
        return self.test_vs_baseline(
            satisfaction_normalized.tolist(),
            self.baseline_satisfaction,
            'Satisfaction vs Literature Baseline'
        )
    
    def test_algorithm_differences(self, df: pd.DataFrame) -> Dict:
        """ANOVA and post-hoc tests for algorithm performance"""
        algorithms = df['algorithm_used'].unique()
        
        # Prepare data for ANOVA
        groups = [df[df['algorithm_used'] == algo]['css_score'].tolist() 
                 for algo in algorithms]
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)
        
        results = {
            'test': 'Algorithm Performance Comparison',
            'algorithms': list(algorithms),
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }
        
        # If significant, perform post-hoc tests
        if p_value < self.alpha:
            # Tukey HSD
            tukey_results = pairwise_tukeyhsd(
                df['css_score'], 
                df['algorithm_used'],
                alpha=self.alpha
            )
            
            results['post_hoc'] = {
                'method': 'Tukey HSD',
                'results': str(tukey_results)
            }
            
            # Pairwise t-tests with Bonferroni correction
            pairwise_results = []
            n_comparisons = len(algorithms) * (len(algorithms) - 1) / 2
            bonferroni_alpha = self.alpha / n_comparisons
            
            for i in range(len(algorithms)):
                for j in range(i + 1, len(algorithms)):
                    algo1, algo2 = algorithms[i], algorithms[j]
                    data1 = df[df['algorithm_used'] == algo1]['css_score']
                    data2 = df[df['algorithm_used'] == algo2]['css_score']
                    
                    t_stat, p_val = stats.ttest_ind(data1, data2)
                    
                    pairwise_results.append({
                        'comparison': f"{algo1} vs {algo2}",
                        'mean_diff': data1.mean() - data2.mean(),
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'significant_bonferroni': p_val < bonferroni_alpha
                    })
            
            results['pairwise_comparisons'] = pairwise_results
        
        return results
    
    def test_scenario_effects(self, df: pd.DataFrame) -> Dict:
        """Test for differences between scenarios"""
        scenarios = df['scenario_id'].unique()
        
        # CSS scores by scenario
        groups = [df[df['scenario_id'] == s]['css_score'].tolist() for s in scenarios]
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Difficulty analysis (based on success rate and completion time)
        scenario_stats = []
        for scenario in scenarios:
            scenario_df = df[df['scenario_id'] == scenario]
            scenario_stats.append({
                'scenario': scenario,
                'n': len(scenario_df),
                'success_rate': scenario_df['success'].mean(),
                'mean_css': scenario_df['css_score'].mean(),
                'mean_time': scenario_df['completion_time'].mean(),
                'mean_satisfaction': scenario_df['satisfaction_rating'].mean()
            })
        
        return {
            'test': 'Scenario Difficulty Analysis',
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'scenario_statistics': scenario_stats
        }
    
    def test_demographic_effects(self, df: pd.DataFrame) -> Dict:
        """Test effects of demographic variables"""
        results = {}
        
        # Age effect on satisfaction
        if 'age_range' in df.columns:
            age_groups = df.groupby('age_range')['satisfaction_rating'].apply(list).to_dict()
            if len(age_groups) > 1:
                f_stat, p_value = stats.f_oneway(*age_groups.values())
                results['age_effect'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
        
        # NYC familiarity effect
        if 'nyc_familiarity' in df.columns:
            familiarity_groups = df.groupby('nyc_familiarity')['css_score'].apply(list).to_dict()
            if len(familiarity_groups) > 1:
                f_stat, p_value = stats.f_oneway(*familiarity_groups.values())
                results['familiarity_effect'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
        
        return results
    
    def test_poi_preference(self, df: pd.DataFrame) -> Dict:
        """Test if users prefer 3-7 POIs as research suggests"""
        poi_counts = []
        satisfaction_by_poi_count = {}
        
        for _, row in df.iterrows():
            if isinstance(row.get('final_itinerary'), str):
                try:
                    itinerary = json.loads(row['final_itinerary'])
                except:
                    continue
            else:
                itinerary = row.get('final_itinerary', {})
                
            if isinstance(itinerary, dict) and 'stops' in itinerary:
                n_pois = len(itinerary['stops'])
                poi_counts.append(n_pois)
                
                if n_pois not in satisfaction_by_poi_count:
                    satisfaction_by_poi_count[n_pois] = []
                satisfaction_by_poi_count[n_pois].append(row['satisfaction_rating'])
        
        # Test if mean is in preferred range
        mean_pois = np.mean(poi_counts)
        in_preferred_range = 3 <= mean_pois <= 7
        
        # Compare satisfaction for preferred range vs outside
        in_range_satisfaction = []
        out_range_satisfaction = []
        
        for n_pois, satisfactions in satisfaction_by_poi_count.items():
            if 3 <= n_pois <= 7:
                in_range_satisfaction.extend(satisfactions)
            else:
                out_range_satisfaction.extend(satisfactions)
        
        if in_range_satisfaction and out_range_satisfaction:
            t_stat, p_value = stats.ttest_ind(in_range_satisfaction, out_range_satisfaction)
            preference_test = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'mean_satisfaction_in_range': np.mean(in_range_satisfaction),
                'mean_satisfaction_out_range': np.mean(out_range_satisfaction)
            }
        else:
            preference_test = None
        
        return {
            'test': 'POI Count Preference Validation',
            'mean_pois': mean_pois,
            'std_pois': np.std(poi_counts),
            'in_preferred_range': in_preferred_range,
            'poi_distribution': dict(zip(*np.unique(poi_counts, return_counts=True))),
            'preference_test': preference_test
        }
    
    def calculate_power_analysis(self, df: pd.DataFrame) -> Dict:
        """Calculate statistical power of the study"""
        power_analyzer = TTestPower()
        
        # Effect size from satisfaction improvement
        satisfaction_normalized = df['satisfaction_rating'] / 10
        effect_size = (satisfaction_normalized.mean() - self.baseline_satisfaction) / satisfaction_normalized.std()
        
        # Calculate achieved power
        n = df['participant_id'].nunique()
        power = power_analyzer.solve_power(
            effect_size=effect_size,
            nobs=n,
            alpha=self.alpha,
            power=None,
            alternative='two-sided'
        )
        
        # Required sample size for 0.8 power
        required_n = power_analyzer.solve_power(
            effect_size=effect_size,
            nobs=None,
            alpha=self.alpha,
            power=0.8,
            alternative='two-sided'
        )
        
        return {
            'achieved_power': power,
            'effect_size': effect_size,
            'sample_size': n,
            'required_n_for_80_power': int(np.ceil(required_n)),
            'adequate_sample': power >= 0.8
        }
    
    def generate_report(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive statistical report"""
        report = {
            'sample_size': df['participant_id'].nunique(),
            'total_tasks': len(df),
            'tests_performed': []
        }
        
        # Success rate test
        report['tests_performed'].append(self.test_success_rate(df))
        
        # Satisfaction test
        report['tests_performed'].append(self.test_satisfaction(df))
        
        # Algorithm comparison
        report['tests_performed'].append(self.test_algorithm_differences(df))
        
        # Scenario effects
        report['tests_performed'].append(self.test_scenario_effects(df))
        
        # Demographic effects
        demographic_results = self.test_demographic_effects(df)
        if demographic_results:
            report['demographic_effects'] = demographic_results
        
        # POI preference validation
        report['poi_preference'] = self.test_poi_preference(df)
        
        # Power analysis
        report['power_analysis'] = self.calculate_power_analysis(df)
        
        return report


def main():
    """Run statistical analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Statistical analysis of user study')
    parser.add_argument('--data', type=str, default='user_study/analysis/report.csv',
                       help='Path to processed data CSV')
    parser.add_argument('--output', type=str, default='user_study/analysis/statistical_report.json',
                       help='Output path for statistical report')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Significance level')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.data)
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer(alpha=args.alpha)
    
    # Generate report
    print("Running statistical tests...")
    report = analyzer.generate_report(df)
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n=== Statistical Analysis Summary ===")
    print(f"Sample size: {report['sample_size']} participants")
    print(f"Total tasks: {report['total_tasks']}")
    print(f"Power analysis: {report['power_analysis']['achieved_power']:.3f}")
    
    print("\n=== Key Findings ===")
    for test in report['tests_performed']:
        if test.get('significant'):
            print(f"✓ {test['test']}: p={test['p_value']:.4f} (significant)")
        else:
            print(f"✗ {test['test']}: p={test['p_value']:.4f} (not significant)")
    
    print(f"\nDetailed report saved to: {args.output}")


if __name__ == "__main__":
    main()