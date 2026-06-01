"""
EOSNM Evaluation Metrics and Baseline Comparisons

This module implements:
1. Performance metrics (placement yield, ROC-AUC, Precision@k, etc.)
2. Baseline allocation strategies (random, skill-only, network-only, SEM)
3. Comparative evaluation framework
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
import logging

from eosnm_core import StudentState

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """
    Compute evaluation metrics for employability allocation strategies.
    """
    
    @staticmethod
    def placement_yield(allocation: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Compute placement yield: proportion of allocated students who got placed.
        
        Args:
            allocation: Binary allocation vector (1 = selected)
            outcomes: Binary placement outcomes (1 = placed)
            
        Returns:
            Placement yield as percentage
        """
        selected_indices = np.where(allocation == 1)[0]
        
        if len(selected_indices) == 0:
            return 0.0
        
        placements_among_selected = outcomes[selected_indices].sum()
        yield_pct = (placements_among_selected / len(selected_indices)) * 100
        
        return yield_pct
    
    @staticmethod
    def aggregate_gain(allocation: np.ndarray, gains: np.ndarray) -> float:
        """
        Compute total employability gain from allocation.
        
        Args:
            allocation: Binary allocation vector
            gains: Employability gain estimates
            
        Returns:
            Sum of gains for allocated students
        """
        return np.sum(allocation * gains)
    
    @staticmethod
    def precision_at_k(allocation: np.ndarray, outcomes: np.ndarray, k: int) -> float:
        """
        Compute Precision@k: precision among top-k allocated students.
        
        Args:
            allocation: Binary allocation vector
            outcomes: Binary placement outcomes
            k: Top-k threshold
            
        Returns:
            Precision@k
        """
        # Get indices of top-k allocated students
        allocated_indices = np.where(allocation == 1)[0]
        
        if len(allocated_indices) == 0 or k == 0:
            return 0.0
        
        # Take first k (assumes allocation is already ranked)
        top_k = allocated_indices[:min(k, len(allocated_indices))]
        
        precision = outcomes[top_k].sum() / len(top_k)
        
        return precision
    
    @staticmethod
    def recall_at_k(allocation: np.ndarray, outcomes: np.ndarray, k: int) -> float:
        """
        Compute Recall@k: proportion of all placements captured in top-k.
        
        Args:
            allocation: Binary allocation vector
            outcomes: Binary placement outcomes
            k: Top-k threshold
            
        Returns:
            Recall@k
        """
        total_placements = outcomes.sum()
        
        if total_placements == 0:
            return 0.0
        
        allocated_indices = np.where(allocation == 1)[0]
        top_k = allocated_indices[:min(k, len(allocated_indices))]
        
        recall = outcomes[top_k].sum() / total_placements
        
        return recall
    
    @staticmethod
    def compute_roc_auc(scores: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Compute ROC-AUC for employability scores.
        
        Args:
            scores: Continuous employability scores
            outcomes: Binary placement outcomes
            
        Returns:
            ROC-AUC score
        """
        if len(np.unique(outcomes)) < 2:
            logger.warning("Only one class present, ROC-AUC undefined")
            return 0.5
        
        return roc_auc_score(outcomes, scores)
    
    @staticmethod
    def compute_pr_auc(scores: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Compute Precision-Recall AUC.
        
        Args:
            scores: Continuous employability scores
            outcomes: Binary placement outcomes
            
        Returns:
            PR-AUC score
        """
        if len(np.unique(outcomes)) < 2:
            logger.warning("Only one class present, PR-AUC undefined")
            return 0.0
        
        precision, recall, _ = precision_recall_curve(outcomes, scores)
        return auc(recall, precision)
    
    @staticmethod
    def normalized_cumulative_gain(allocation: np.ndarray, 
                                   outcomes: np.ndarray,
                                   top_fractions: np.ndarray = None) -> np.ndarray:
        """
        Compute normalized cumulative gain curve.
        
        Args:
            allocation: Ranked allocation vector
            outcomes: Binary placement outcomes
            top_fractions: Array of top-% thresholds (default: [5, 10, ..., 50])
            
        Returns:
            Array of cumulative gains at each threshold
        """
        if top_fractions is None:
            top_fractions = np.array([5, 10, 15, 20, 25, 30, 40, 50])
        
        allocated_indices = np.where(allocation == 1)[0]
        total_placements = outcomes.sum()
        
        if total_placements == 0:
            return np.zeros_like(top_fractions, dtype=float)
        
        gains = []
        N = len(outcomes)
        
        for frac in top_fractions:
            k = int(N * frac / 100)
            top_k = allocated_indices[:min(k, len(allocated_indices))]
            captured = outcomes[top_k].sum() / total_placements
            gains.append(captured)
        
        return np.array(gains)
    
    @staticmethod
    def group_coverage_ratio(allocation: np.ndarray, 
                            group_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute Group Coverage Ratio (GCR) for fairness analysis.
        
        GCR = (selected from group / total selected) / (group size / total population)
        GCR = 1 indicates perfect representation
        
        Args:
            allocation: Binary allocation vector
            group_labels: Group membership labels (e.g., departments)
            
        Returns:
            Dictionary mapping group names to GCR values
        """
        unique_groups = np.unique(group_labels)
        gcr_dict = {}
        
        total_population = len(allocation)
        total_selected = allocation.sum()
        
        if total_selected == 0:
            return {str(g): 0.0 for g in unique_groups}
        
        for group in unique_groups:
            group_mask = (group_labels == group)
            group_size = group_mask.sum()
            selected_from_group = (allocation[group_mask]).sum()
            
            expected_ratio = group_size / total_population
            actual_ratio = selected_from_group / total_selected
            
            gcr = actual_ratio / expected_ratio if expected_ratio > 0 else 0.0
            gcr_dict[str(group)] = gcr
        
        return gcr_dict


class BaselineAllocations:
    """
    Implements baseline allocation strategies for comparison.
    """
    
    @staticmethod
    def random_allocation(n_students: int, budget: float) -> np.ndarray:
        """
        Random allocation baseline.
        
        Args:
            n_students: Total number of students
            budget: Number of students to select
            
        Returns:
            Binary allocation vector
        """
        allocation = np.zeros(n_students, dtype=int)
        selected = np.random.choice(n_students, size=int(budget), replace=False)
        allocation[selected] = 1
        
        logger.debug(f"Random allocation: {budget} students")
        
        return allocation
    
    @staticmethod
    def skill_only_allocation(students: List[StudentState], budget: float) -> np.ndarray:
        """
        Allocate based only on skill level (descending order).
        
        Args:
            students: List of StudentState objects
            budget: Number of students to select
            
        Returns:
            Binary allocation vector
        """
        n_students = len(students)
        skills = np.array([s.skill_level for s in students])
        
        # Sort by skill descending
        sorted_indices = np.argsort(-skills)
        
        allocation = np.zeros(n_students, dtype=int)
        allocation[sorted_indices[:int(budget)]] = 1
        
        logger.debug(f"Skill-only allocation: {budget} students")
        
        return allocation
    
    @staticmethod
    def network_only_allocation(students: List[StudentState], budget: float) -> np.ndarray:
        """
        Allocate based only on network degree (descending order).
        
        Args:
            students: List of StudentState objects
            budget: Number of students to select
            
        Returns:
            Binary allocation vector
        """
        n_students = len(students)
        networks = np.array([s.network_degree for s in students])
        
        # Sort by network degree descending
        sorted_indices = np.argsort(-networks)
        
        allocation = np.zeros(n_students, dtype=int)
        allocation[sorted_indices[:int(budget)]] = 1
        
        logger.debug(f"Network-only allocation: {budget} students")
        
        return allocation
    
    @staticmethod
    def sem_based_allocation(students: List[StudentState], 
                            outcomes: np.ndarray,
                            budget: float) -> np.ndarray:
        """
        SEM-based allocation: use logistic regression on skill + network.
        
        This simulates a Structural Equation Model (SEM) approach
        that predicts placement using skill and network features.
        
        Args:
            students: List of StudentState objects
            outcomes: Binary placement outcomes (for training)
            budget: Number of students to select
            
        Returns:
            Binary allocation vector
        """
        n_students = len(students)
        
        # Create feature matrix
        X = np.column_stack([
            [s.skill_level for s in students],
            [s.network_degree for s in students],
            [s.participation_intensity for s in students],
            [1 - s.barrier_index for s in students]  # Invert barriers
        ])
        
        # Train logistic regression
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, outcomes)
        
        # Predict probabilities
        probabilities = model.predict_proba(X)[:, 1]
        
        # Allocate to top-k by predicted probability
        sorted_indices = np.argsort(-probabilities)
        
        allocation = np.zeros(n_students, dtype=int)
        allocation[sorted_indices[:int(budget)]] = 1
        
        logger.debug(f"SEM-based allocation: {budget} students")
        
        return allocation


def run_comparative_evaluation(students: List[StudentState],
                               outcomes: np.ndarray,
                               eosnm_gains: np.ndarray,
                               budget_fractions: List[float] = None) -> pd.DataFrame:
    """
    Run comprehensive comparative evaluation across all methods and budgets.
    
    Args:
        students: List of StudentState objects
        outcomes: Binary placement outcomes
        eosnm_gains: EOSNM-computed employability gains
        budget_fractions: List of budget fractions to test (e.g., [0.05, 0.10, 0.15, 0.20])
        
    Returns:
        DataFrame with comparative results
    """
    if budget_fractions is None:
        budget_fractions = [0.05, 0.10, 0.15, 0.20]
    
    n_students = len(students)
    results = []
    
    logger.info("Running comparative evaluation...")
    
    for budget_frac in budget_fractions:
        budget = int(n_students * budget_frac)
        
        logger.info(f"  Budget: {budget_frac*100:.0f}% ({budget} students)")
        
        # EOSNM Greedy allocation
        costs = np.ones(n_students)
        sorted_indices = np.argsort(-eosnm_gains / costs)
        eosnm_alloc = np.zeros(n_students, dtype=int)
        eosnm_alloc[sorted_indices[:budget]] = 1
        
        # EOSNM-MIP (simplified as greedy for this implementation)
        eosnm_mip_alloc = eosnm_alloc.copy()
        
        # Baseline allocations
        random_alloc = BaselineAllocations.random_allocation(n_students, budget)
        skill_alloc = BaselineAllocations.skill_only_allocation(students, budget)
        network_alloc = BaselineAllocations.network_only_allocation(students, budget)
        sem_alloc = BaselineAllocations.sem_based_allocation(students, outcomes, budget)
        
        # Compute metrics for each method
        methods = {
            'EOSNM-Greedy': eosnm_alloc,
            'EOSNM-MIP': eosnm_mip_alloc,
            'SEM': sem_alloc,
            'Skill-only': skill_alloc,
            'Network-only': network_alloc,
            'Random': random_alloc
        }
        
        for method_name, allocation in methods.items():
            yield_pct = EvaluationMetrics.placement_yield(allocation, outcomes)
            total_gain = EvaluationMetrics.aggregate_gain(allocation, eosnm_gains)
            prec_20 = EvaluationMetrics.precision_at_k(allocation, outcomes, k=20)
            
            results.append({
                'Method': method_name,
                'Budget (%)': budget_frac * 100,
                'Budget (N)': budget,
                'Placement Yield (%)': yield_pct,
                'Aggregate Gain': total_gain,
                'Precision@20': prec_20
            })
    
    results_df = pd.DataFrame(results)
    
    logger.info("Comparative evaluation complete")
    
    return results_df


def compute_confidence_intervals(students: List[StudentState],
                                outcomes: np.ndarray,
                                eosnm_gains: np.ndarray,
                                budget_frac: float = 0.15,
                                n_bootstrap: int = 100) -> Dict:
    """
    Compute confidence intervals using bootstrap resampling.
    
    Args:
        students: List of StudentState objects
        outcomes: Binary placement outcomes
        eosnm_gains: EOSNM employability gains
        budget_frac: Budget fraction
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Dictionary with mean and 95% CI for each metric
    """
    logger.info(f"Computing confidence intervals with {n_bootstrap} bootstrap samples...")
    
    n_students = len(students)
    budget = int(n_students * budget_frac)
    
    yields = []
    gains = []
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_students, n_students, replace=True)
        
        sample_gains = eosnm_gains[indices]
        sample_outcomes = outcomes[indices]
        
        # EOSNM allocation
        sorted_indices = np.argsort(-sample_gains)
        allocation = np.zeros(n_students, dtype=int)
        allocation[sorted_indices[:budget]] = 1
        
        # Compute metrics
        yield_pct = EvaluationMetrics.placement_yield(allocation, sample_outcomes)
        total_gain = EvaluationMetrics.aggregate_gain(allocation, sample_gains)
        
        yields.append(yield_pct)
        gains.append(total_gain)
    
    yields = np.array(yields)
    gains = np.array(gains)
    
    results = {
        'placement_yield': {
            'mean': yields.mean(),
            'ci_lower': np.percentile(yields, 2.5),
            'ci_upper': np.percentile(yields, 97.5)
        },
        'aggregate_gain': {
            'mean': gains.mean(),
            'ci_lower': np.percentile(gains, 2.5),
            'ci_upper': np.percentile(gains, 97.5)
        }
    }
    
    logger.info("Confidence intervals computed")
    
    return results


if __name__ == "__main__":
    # Test evaluation metrics
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    n = 100
    allocation = np.zeros(n, dtype=int)
    allocation[:20] = 1  # Select top 20
    outcomes = np.random.binomial(1, 0.5, n)
    gains = np.random.random(n)
    
    # Test metrics
    print("Testing Evaluation Metrics:")
    print(f"Placement Yield: {EvaluationMetrics.placement_yield(allocation, outcomes):.2f}%")
    print(f"Aggregate Gain: {EvaluationMetrics.aggregate_gain(allocation, gains):.4f}")
    print(f"Precision@10: {EvaluationMetrics.precision_at_k(allocation, outcomes, k=10):.4f}")
    print(f"ROC-AUC: {EvaluationMetrics.compute_roc_auc(gains, outcomes):.4f}")
