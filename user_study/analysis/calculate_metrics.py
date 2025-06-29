"""
Calculate User Study Metrics

Processes user study data to compute:
- Task completion rates
- CSS component scores
- System usability scores
- Algorithm preference statistics
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class UserStudyAnalyzer:
    """Analyze user study results based on research metrics"""
    
    def __init__(self, data_dir: str = "user_study/results"):
        self.data_dir = Path(data_dir)
        self.baseline_satisfaction = 0.82  # From literature
        self.travelplanner_baseline = 0.006  # 0.6% success rate
        
    def load_participant_data(self, participant_id: str) -> Dict:
        """Load all data for a participant"""
        participant_dir = self.data_dir / participant_id
        
        data = {
            'id': participant_id,
            'pre_study': self._load_json(participant_dir / 'pre_study.json'),
            'tasks': self._load_tasks(participant_dir / 'tasks'),
            'post_study': self._load_json(participant_dir / 'post_study.json'),
            'interview': self._load_json(participant_dir / 'interview.json'),
            'interactions': self._load_json(participant_dir / 'interactions.json')
        }
        
        return data
    
    def _load_json(self, filepath: Path) -> Dict:
        """Load JSON file safely"""
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_tasks(self, tasks_dir: Path) -> List[Dict]:
        """Load all task results"""
        tasks = []
        if tasks_dir.exists():
            for task_file in sorted(tasks_dir.glob('*.json')):
                tasks.append(self._load_json(task_file))
        return tasks
    
    def calculate_sus_score(self, responses: List[int]) -> float:
        """
        Calculate System Usability Scale score
        
        SUS scoring:
        - Odd questions (1,3,5,7,9): score - 1
        - Even questions (2,4,6,8,10): 5 - score
        - Sum all, multiply by 2.5
        """
        if len(responses) != 10:
            return None
            
        score = 0
        for i, response in enumerate(responses):
            if i % 2 == 0:  # Odd question (0-indexed)
                score += (response - 1)
            else:  # Even question
                score += (5 - response)
                
        return score * 2.5
    
    def calculate_task_metrics(self, task_data: Dict) -> Dict:
        """Calculate metrics for a single task"""
        metrics = {
            'scenario_id': task_data.get('scenario_id'),
            'completion_time': task_data.get('completion_time', 0),
            'success': task_data.get('success', False),
            'algorithm_used': task_data.get('algorithm_used', 'unknown'),
            'modifications': task_data.get('num_modifications', 0),
            'satisfaction_rating': task_data.get('satisfaction_rating', 0),
        }
        
        # Calculate CSS components if available
        if 'final_itinerary' in task_data:
            itinerary = task_data['final_itinerary']
            metrics.update(self._calculate_css_components(itinerary))
            
        # Check if requirements met
        if 'requirements_met' in task_data:
            metrics['requirements_met'] = task_data['requirements_met']
            
        return metrics
    
    def _calculate_css_components(self, itinerary: Dict) -> Dict:
        """Calculate CSS score components from itinerary"""
        # Time Utilization Ratio
        total_duration = sum(stop.get('duration', 0) for stop in itinerary.get('stops', []))
        available_time = itinerary.get('available_time', 480)  # 8 hours default
        tur = min(total_duration / available_time, 1.0) if available_time > 0 else 0
        
        # Satisfaction (based on POI ratings/preferences match)
        satisfaction_scores = [stop.get('preference_score', 0.5) for stop in itinerary.get('stops', [])]
        sat = np.mean(satisfaction_scores) if satisfaction_scores else 0
        
        # Feasibility (based on constraint violations)
        feasibility = itinerary.get('feasibility_score', 1.0)
        
        # Diversity (category variety)
        categories = [stop.get('category') for stop in itinerary.get('stops', [])]
        unique_categories = len(set(categories)) if categories else 0
        total_stops = len(categories) if categories else 1
        diversity = unique_categories / total_stops if total_stops > 0 else 0
        
        # Calculate CSS using research weights
        css = 0.25 * tur + 0.35 * sat + 0.25 * feasibility + 0.15 * diversity
        
        return {
            'tur': tur,
            'satisfaction': sat,
            'feasibility': feasibility,
            'diversity': diversity,
            'css_score': css
        }
    
    def analyze_all_participants(self) -> pd.DataFrame:
        """Analyze all participant data"""
        all_results = []
        
        participant_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('P')]
        
        for participant_dir in sorted(participant_dirs):
            participant_id = participant_dir.name
            try:
                data = self.load_participant_data(participant_id)
                
                # Calculate SUS score
                sus_responses = data['post_study'].get('sus_responses', [])
                sus_score = self.calculate_sus_score(sus_responses) if sus_responses else None
                
                # Process each task
                for task in data['tasks']:
                    task_metrics = self.calculate_task_metrics(task)
                    task_metrics['participant_id'] = participant_id
                    task_metrics['sus_score'] = sus_score
                    
                    # Add demographic info
                    task_metrics['age_range'] = data['pre_study'].get('age_range')
                    task_metrics['travel_frequency'] = data['pre_study'].get('travel_frequency')
                    task_metrics['nyc_familiarity'] = data['pre_study'].get('nyc_visits')
                    
                    all_results.append(task_metrics)
                    
            except Exception as e:
                print(f"Error processing {participant_id}: {e}")
                
        return pd.DataFrame(all_results)
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics comparing to baselines"""
        
        summary = {
            'n_participants': df['participant_id'].nunique(),
            'n_tasks': len(df),
            'overall_success_rate': df['success'].mean(),
            'success_vs_travelplanner': df['success'].mean() / self.travelplanner_baseline,
            'mean_satisfaction': df['satisfaction_rating'].mean(),
            'satisfaction_vs_baseline': df['satisfaction_rating'].mean() / 10 / self.baseline_satisfaction,
            'mean_sus_score': df['sus_score'].mean(),
            'mean_css_score': df['css_score'].mean(),
            'mean_completion_time': df['completion_time'].mean(),
            'algorithm_preferences': df['algorithm_used'].value_counts().to_dict()
        }
        
        # CSS component means
        for component in ['tur', 'satisfaction', 'feasibility', 'diversity']:
            if component in df.columns:
                summary[f'mean_{component}'] = df[component].mean()
        
        # Success rate by scenario
        summary['success_by_scenario'] = df.groupby('scenario_id')['success'].mean().to_dict()
        
        # Satisfaction by scenario
        summary['satisfaction_by_scenario'] = df.groupby('scenario_id')['satisfaction_rating'].mean().to_dict()
        
        # POI count analysis (validating 3-7 preference)
        poi_counts = df['final_itinerary'].apply(
            lambda x: len(x.get('stops', [])) if isinstance(x, dict) else 0
        )
        summary['mean_pois_per_itinerary'] = poi_counts.mean()
        summary['poi_count_distribution'] = poi_counts.value_counts().to_dict()
        
        return summary
    
    def plot_results(self, df: pd.DataFrame, output_dir: str = "user_study/analysis/figures"):
        """Generate analysis plots"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        # 1. Success Rate Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scenarios = df['scenario_id'].unique()
        success_rates = [df[df['scenario_id'] == s]['success'].mean() for s in scenarios]
        
        bars = ax.bar(range(len(scenarios)), success_rates)
        ax.axhline(y=self.baseline_satisfaction, color='r', linestyle='--', label='Literature Baseline (82%)')
        ax.axhline(y=self.travelplanner_baseline, color='g', linestyle='--', label='TravelPlanner (0.6%)')
        
        ax.set_xlabel('Scenario')
        ax.set_ylabel('Success Rate')
        ax.set_title('Task Success Rate by Scenario')
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([s.split('_')[1] for s in scenarios], rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'success_rates.png', dpi=300)
        plt.close()
        
        # 2. CSS Components
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        components = ['tur', 'satisfaction', 'feasibility', 'diversity']
        
        for idx, (ax, component) in enumerate(zip(axes.flat, components)):
            if component in df.columns:
                df.boxplot(column=component, by='scenario_id', ax=ax)
                ax.set_title(f'{component.upper()} by Scenario')
                ax.set_xlabel('Scenario')
                ax.set_ylabel(component.capitalize())
                ax.set_xticklabels([s.split('_')[1] for s in scenarios], rotation=45)
        
        plt.suptitle('CSS Components Distribution', y=0.98)
        plt.tight_layout()
        plt.savefig(output_path / 'css_components.png', dpi=300)
        plt.close()
        
        # 3. Algorithm Usage
        fig, ax = plt.subplots(figsize=(10, 6))
        algorithm_counts = df['algorithm_used'].value_counts()
        algorithm_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Usage Count')
        ax.set_title('Algorithm Selection Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'algorithm_usage.png', dpi=300)
        plt.close()
        
        # 4. SUS Score Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sus_scores = df.groupby('participant_id')['sus_score'].first()
        sus_scores.hist(bins=20, ax=ax, edgecolor='black')
        ax.axvline(x=68, color='r', linestyle='--', label='SUS Benchmark (68)')
        ax.set_xlabel('SUS Score')
        ax.set_ylabel('Number of Participants')
        ax.set_title('System Usability Scale Score Distribution')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'sus_distribution.png', dpi=300)
        plt.close()
        
        # 5. POI Count Validation
        fig, ax = plt.subplots(figsize=(10, 6))
        poi_counts = []
        for _, row in df.iterrows():
            if isinstance(row.get('final_itinerary'), dict):
                poi_counts.append(len(row['final_itinerary'].get('stops', [])))
        
        if poi_counts:
            pd.Series(poi_counts).value_counts().sort_index().plot(kind='bar', ax=ax)
            ax.axvspan(2.5, 7.5, alpha=0.2, color='green', label='Preferred Range (3-7)')
            ax.set_xlabel('Number of POIs')
            ax.set_ylabel('Frequency')
            ax.set_title('POIs per Itinerary Distribution')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'poi_distribution.png', dpi=300)
        plt.close()
        
        print(f"Plots saved to {output_path}")


def main():
    """Run analysis on user study data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze user study results')
    parser.add_argument('--data-dir', type=str, default='user_study/results',
                       help='Directory containing participant data')
    parser.add_argument('--output', type=str, default='user_study/analysis/report.json',
                       help='Output file for analysis report')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = UserStudyAnalyzer(args.data_dir)
    
    # Analyze all participants
    print("Loading participant data...")
    df = analyzer.analyze_all_participants()
    
    if len(df) == 0:
        print("No data found. Please ensure participant data is in the correct format.")
        return
    
    # Generate summary statistics
    print("Calculating summary statistics...")
    summary = analyzer.generate_summary_statistics(df)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    # Generate plots
    print("Generating analysis plots...")
    analyzer.plot_results(df)
    
    # Print summary
    print("\n=== User Study Summary ===")
    print(f"Participants: {summary['n_participants']}")
    print(f"Total tasks: {summary['n_tasks']}")
    print(f"Overall success rate: {summary['overall_success_rate']:.1%}")
    print(f"vs TravelPlanner: {summary['success_vs_travelplanner']:.0f}x better")
    print(f"Mean satisfaction: {summary['mean_satisfaction']:.1f}/10")
    print(f"vs Literature baseline: {summary['satisfaction_vs_baseline']:.1%}")
    print(f"Mean SUS score: {summary['mean_sus_score']:.1f}")
    print(f"Mean CSS score: {summary['mean_css_score']:.3f}")
    print(f"Mean POIs per itinerary: {summary['mean_pois_per_itinerary']:.1f}")
    
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()