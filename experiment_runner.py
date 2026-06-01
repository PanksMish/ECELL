"""
EOSNM Main Experiment Runner

Orchestrates the complete experimental pipeline:
1. Data loading and preprocessing
2. EOSNM framework execution
3. Baseline comparisons
4. Evaluation metrics computation
5. Visualization generation
6. Results export

Usage:
    python experiment_runner.py --config configs/default_config.yaml
"""

import argparse
import logging
import os
import yaml
import json
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Import EOSNM modules
from eosnm_core import (
    SkillGrowthModel, NetworkGrowthModel, EmployabilityIndex,
    EOSNMFramework, ResourceAllocation, run_eosnm_pipeline
)
from data_loader import ECellDataLoader, load_ecell_data
from evaluation_metrics import (
    EvaluationMetrics, BaselineAllocations,
    run_comparative_evaluation, compute_confidence_intervals
)
from visualization import EOSNMVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EOSNMExperiment:
    """
    Main experimental orchestrator for EOSNM framework.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize experiment with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self.load_config(config_path)
        self.results_dir = Path(self.config['output']['results_dir'])
        self.figures_dir = Path(self.config['output']['figures_dir'])
        
        # Create output directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Store timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info(f"EOSNM Experiment initialized: {self.timestamp}")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Figures directory: {self.figures_dir}")
    
    def load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file or use defaults.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'data': {
                'path': None,  # None = synthetic data
                'n_students': 391,
                'random_state': 42
            },
            'model': {
                'skill_growth': {
                    'alpha': 0.5,
                    'S_max': 10.0,
                    'beta': 0.3
                },
                'network_growth': {
                    'eta': 0.3,
                    'kappa': 1.0
                },
                'employability_index': {
                    'w_skill': 0.65,
                    'w_network': 0.35,
                    'w_context': 0.0
                }
            },
            'experiment': {
                'time_horizon': 6,
                'budget_fractions': [0.05, 0.10, 0.15, 0.20],
                'n_bootstrap': 100,
                'monte_carlo_runs': 100
            },
            'output': {
                'results_dir': 'results',
                'figures_dir': 'figures',
                'save_predictions': True,
                'save_allocations': True
            }
        }
        
        if config_path and os.path.exists(config_path):
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Merge with defaults
            self._merge_configs(default_config, user_config)
        else:
            logger.info("Using default configuration")
        
        return default_config
    
    def _merge_configs(self, base: dict, update: dict):
        """Recursively merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def run_experiment(self):
        """
        Execute complete experimental pipeline.
        """
        logger.info("="*80)
        logger.info("STARTING EOSNM EXPERIMENT")
        logger.info("="*80)
        
        # Step 1: Load and preprocess data
        logger.info("\n[STEP 1] Loading and preprocessing data...")
        students, df = self.load_and_prepare_data()
        outcomes = df['placement_outcome'].values
        
        # Step 2: Initialize EOSNM framework
        logger.info("\n[STEP 2] Initializing EOSNM framework...")
        eosnm = self.initialize_eosnm()
        
        # Step 3: Compute employability gains
        logger.info("\n[STEP 3] Computing employability gains...")
        time_horizon = self.config['experiment']['time_horizon']
        gains = eosnm.compute_all_gains(students, time_horizon)
        
        # Step 4: Run comparative evaluation
        logger.info("\n[STEP 4] Running comparative evaluation...")
        budget_fractions = self.config['experiment']['budget_fractions']
        results_df = run_comparative_evaluation(
            students, outcomes, gains, budget_fractions
        )
        
        # Step 5: Compute confidence intervals
        logger.info("\n[STEP 5] Computing confidence intervals...")
        ci_results = compute_confidence_intervals(
            students, outcomes, gains,
            budget_frac=0.15,
            n_bootstrap=self.config['experiment']['n_bootstrap']
        )
        
        # Step 6: Generate visualizations
        logger.info("\n[STEP 6] Generating visualizations...")
        self.generate_all_visualizations(students, df, gains, outcomes, results_df)
        
        # Step 7: Save results
        logger.info("\n[STEP 7] Saving results...")
        self.save_results(results_df, ci_results, gains, outcomes)
        
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("="*80)
        
        return results_df, ci_results
    
    def load_and_prepare_data(self):
        """
        Load and preprocess E-Cell data.
        
        Returns:
            Tuple of (students_list, dataframe)
        """
        data_path = self.config['data']['path']
        
        if data_path is None:
            # Generate synthetic data
            n_students = self.config['data']['n_students']
            loader = ECellDataLoader(random_state=self.config['data']['random_state'])
            df = loader.generate_synthetic_data(n_students)
            df = loader.preprocess_data(df)
            students = loader.create_student_states(df)
        else:
            # Load real data
            students, df = load_ecell_data(data_path, preprocess=True)
        
        logger.info(f"Loaded {len(students)} students")
        logger.info(f"Placement rate: {df['placement_outcome'].mean()*100:.1f}%")
        
        return students, df
    
    def initialize_eosnm(self):
        """
        Initialize EOSNM framework with configured parameters.
        
        Returns:
            EOSNMFramework instance
        """
        # Initialize component models
        skill_model = SkillGrowthModel(
            **self.config['model']['skill_growth']
        )
        
        network_model = NetworkGrowthModel(
            **self.config['model']['network_growth']
        )
        
        employability_index = EmployabilityIndex(
            **self.config['model']['employability_index']
        )
        
        # Create framework
        eosnm = EOSNMFramework(skill_model, network_model, employability_index)
        
        return eosnm
    
    def generate_all_visualizations(self, students, df, gains, outcomes, results_df):
        """
        Generate all publication-quality figures.
        
        Args:
            students: List of StudentState objects
            df: DataFrame with student data
            gains: EOSNM employability gains
            outcomes: Binary placement outcomes
            results_df: Comparative results DataFrame
        """
        visualizer = EOSNMVisualizer()
        
        # Compute initial employability scores for all students
        employability_index = EmployabilityIndex(
            **self.config['model']['employability_index']
        )
        E0_scores = np.array([
            employability_index.compute(s.skill_level, s.network_degree)
            for s in students
        ])
        
        # Figure 1: Employability distribution
        visualizer.plot_employability_distribution(
            E0_scores,
            save_path=self.figures_dir / f'fig1_employability_dist_{self.timestamp}.png'
        )
        
        # Figure 2: Skill trajectories (example students)
        eosnm = self.initialize_eosnm()
        trajectories = self.generate_example_trajectories(students, eosnm)
        visualizer.plot_skill_trajectories(
            trajectories,
            time_horizon=self.config['experiment']['time_horizon'],
            save_path=self.figures_dir / f'fig2_skill_trajectories_{self.timestamp}.png'
        )
        
        # Figure 3: Network distribution
        network_degrees = np.array([s.network_degree for s in students])
        visualizer.plot_network_distribution(
            network_degrees,
            save_path=self.figures_dir / f'fig3_network_dist_{self.timestamp}.png'
        )
        
        # Figure 4: Placement yield comparison
        visualizer.plot_placement_yield_comparison(
            results_df,
            save_path=self.figures_dir / f'fig4_placement_yield_{self.timestamp}.png'
        )
        
        # Figure 5: Performance matrix
        visualizer.plot_performance_matrix(
            results_df,
            save_path=self.figures_dir / f'fig5_performance_matrix_{self.timestamp}.png'
        )
        
        # Figure 6: ROC and PR curves
        visualizer.plot_roc_pr_curves(
            E0_scores, outcomes,
            save_path=self.figures_dir / f'fig6_roc_pr_curves_{self.timestamp}.png'
        )
        
        # Figure 7: Cumulative gain
        # Get EOSNM allocation at 15% budget
        budget = int(len(students) * 0.15)
        sorted_indices = np.argsort(-gains)
        allocation = np.zeros(len(students), dtype=int)
        allocation[sorted_indices[:budget]] = 1
        
        visualizer.plot_cumulative_gain(
            allocation, outcomes,
            save_path=self.figures_dir / f'fig7_cumulative_gain_{self.timestamp}.png'
        )
        
        # Figure 8: Group coverage ratio
        dept_labels = df['department'].values if 'department' in df.columns else None
        if dept_labels is not None:
            gcr_dict = EvaluationMetrics.group_coverage_ratio(allocation, dept_labels)
            visualizer.plot_group_coverage(
                gcr_dict,
                save_path=self.figures_dir / f'fig8_group_coverage_{self.timestamp}.png'
            )
        
        logger.info(f"All figures saved to {self.figures_dir}")
    
    def generate_example_trajectories(self, students, eosnm):
        """
        Generate example skill trajectories for low/medium/high E0 students.
        
        Returns:
            Dictionary of trajectories
        """
        employability_index = EmployabilityIndex(
            **self.config['model']['employability_index']
        )
        
        # Compute E0 for all students
        E0_scores = np.array([
            employability_index.compute(s.skill_level, s.network_degree)
            for s in students
        ])
        
        # Select representative students
        low_idx = np.argmin(np.abs(E0_scores - np.percentile(E0_scores, 25)))
        med_idx = np.argmin(np.abs(E0_scores - np.percentile(E0_scores, 50)))
        high_idx = np.argmin(np.abs(E0_scores - np.percentile(E0_scores, 75)))
        
        trajectories = {}
        time_horizon = self.config['experiment']['time_horizon']
        
        for label, idx in [('Low', low_idx), ('Medium', med_idx), ('High', high_idx)]:
            student = students[idx]
            
            # Baseline trajectory
            s_base, _ = eosnm.simulate_baseline_trajectory(student, time_horizon)
            trajectories[f'{label} E0 (baseline)'] = s_base
            
            # Intervention trajectory
            s_int, _ = eosnm.simulate_intervention_trajectory(student, time_horizon)
            trajectories[f'{label} E0 (allocated)'] = s_int
        
        return trajectories
    
    def save_results(self, results_df, ci_results, gains, outcomes):
        """
        Save all experimental results.
        
        Args:
            results_df: Comparative evaluation results
            ci_results: Confidence interval results
            gains: Employability gains array
            outcomes: Placement outcomes array
        """
        # Save comparative results
        results_path = self.results_dir / f'comparative_results_{self.timestamp}.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved comparative results to {results_path}")
        
        # Save confidence intervals
        ci_path = self.results_dir / f'confidence_intervals_{self.timestamp}.json'
        with open(ci_path, 'w') as f:
            json.dump(ci_results, f, indent=2)
        logger.info(f"Saved confidence intervals to {ci_path}")
        
        # Save gains and outcomes
        if self.config['output']['save_predictions']:
            predictions_df = pd.DataFrame({
                'student_id': range(len(gains)),
                'employability_gain': gains,
                'placement_outcome': outcomes
            })
            pred_path = self.results_dir / f'predictions_{self.timestamp}.csv'
            predictions_df.to_csv(pred_path, index=False)
            logger.info(f"Saved predictions to {pred_path}")
        
        # Save summary statistics
        summary = {
            'timestamp': self.timestamp,
            'n_students': len(gains),
            'placement_rate': float(outcomes.mean()),
            'mean_gain': float(gains.mean()),
            'std_gain': float(gains.std()),
            'best_method': results_df.loc[results_df['Placement Yield (%)'].idxmax(), 'Method'],
            'best_yield': float(results_df['Placement Yield (%)'].max()),
            'config': self.config
        }
        
        summary_path = self.results_dir / f'summary_{self.timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")


def main():
    """
    Main entry point for experiment runner.
    """
    parser = argparse.ArgumentParser(
        description='Run EOSNM experiments for employability optimization'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to input data CSV (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = EOSNMExperiment(config_path=args.config)
    
    # Override data path if provided
    if args.data:
        experiment.config['data']['path'] = args.data
    
    # Override output directory if provided
    if args.output_dir:
        experiment.config['output']['results_dir'] = args.output_dir
        experiment.config['output']['figures_dir'] = os.path.join(args.output_dir, 'figures')
        experiment.results_dir = Path(args.output_dir)
        experiment.figures_dir = Path(args.output_dir) / 'figures'
        experiment.results_dir.mkdir(parents=True, exist_ok=True)
        experiment.figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    results_df, ci_results = experiment.run_experiment()
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"\nTop 3 Methods by Placement Yield (15% budget):")
    budget_15 = results_df[results_df['Budget (%)'] == 15.0].sort_values(
        'Placement Yield (%)', ascending=False
    )
    print(budget_15[['Method', 'Placement Yield (%)']].head(3).to_string(index=False))
    
    print(f"\n95% Confidence Intervals (15% budget):")
    print(f"  Placement Yield: {ci_results['placement_yield']['mean']:.2f}% "
          f"[{ci_results['placement_yield']['ci_lower']:.2f}, "
          f"{ci_results['placement_yield']['ci_upper']:.2f}]")
    print(f"  Aggregate Gain: {ci_results['aggregate_gain']['mean']:.4f} "
          f"[{ci_results['aggregate_gain']['ci_lower']:.4f}, "
          f"{ci_results['aggregate_gain']['ci_upper']:.4f}]")
    
    print(f"\nResults saved to: {experiment.results_dir}")
    print(f"Figures saved to: {experiment.figures_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
