"""
EOSNM: Employability Optimization through Skill-Network Modelling
Core Framework Implementation

This module implements the dynamic employability modeling framework
including skill growth, network expansion, and resource-constrained optimization.

Authors: Pankaj Mishra, V Venkataramanan, Anand Nayyar
Institution: K J Somaiya School of Engineering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.integrate import odeint
from scipy.optimize import minimize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StudentState:
    """
    Represents the dynamic state of a student in the EOSNM framework.
    
    Attributes:
        student_id: Unique identifier for the student
        skill_level: Current skill competency score (0 to S_max)
        network_degree: Number of professional connections
        barrier_index: Composite barrier score (0 to 1)
        participation_intensity: Engagement level in E-Cell activities
        demographics: Static demographic and academic features
    """
    student_id: int
    skill_level: float
    network_degree: int
    barrier_index: float
    participation_intensity: float
    demographics: Dict[str, any]


class SkillGrowthModel:
    """
    Implements the logistic skill growth model with barrier effects.
    
    The model follows:
    ds/dt = α * s(t) * (1 - s(t)/S) - β * B(t)
    
    where:
    - α is the intrinsic learning rate
    - S is the skill saturation level (carrying capacity)
    - β is the barrier impact coefficient
    - B(t) is the time-varying barrier index
    """
    
    def __init__(self, alpha: float = 0.5, S_max: float = 10.0, beta: float = 0.3):
        """
        Initialize the skill growth model parameters.
        
        Args:
            alpha: Intrinsic learning rate (default: 0.5)
            S_max: Maximum skill saturation level (default: 10.0)
            beta: Barrier impact coefficient (default: 0.3)
        """
        self.alpha = alpha
        self.S_max = S_max
        self.beta = beta
        logger.info(f"SkillGrowthModel initialized: α={alpha}, S_max={S_max}, β={beta}")
    
    def growth_rate(self, s: float, t: float, barrier: float) -> float:
        """
        Calculate instantaneous skill growth rate.
        
        Args:
            s: Current skill level
            t: Time (not used in autonomous system, kept for ODE interface)
            barrier: Current barrier index
            
        Returns:
            Rate of change of skill level
        """
        logistic_growth = self.alpha * s * (1 - s / self.S_max)
        barrier_effect = self.beta * barrier
        return logistic_growth - barrier_effect
    
    def simulate_trajectory(self, s0: float, barrier: float, 
                          time_horizon: int = 6) -> np.ndarray:
        """
        Simulate skill development trajectory over time horizon.
        
        Args:
            s0: Initial skill level
            barrier: Constant barrier index (simplified assumption)
            time_horizon: Number of time periods to simulate (default: 6 months)
            
        Returns:
            Array of skill levels over time
        """
        t = np.linspace(0, time_horizon, time_horizon + 1)
        trajectory = odeint(self.growth_rate, s0, t, args=(barrier,))
        return trajectory.flatten()
    
    def discrete_update(self, s: float, barrier: float, dt: float = 1.0) -> float:
        """
        Discrete-time skill update (used in simulation loop).
        
        Args:
            s: Current skill level
            barrier: Current barrier index
            dt: Time step size (default: 1.0)
            
        Returns:
            Updated skill level
        """
        ds = self.alpha * s * (1 - s / self.S_max) - self.beta * barrier
        s_next = s + ds * dt
        return max(0, min(s_next, self.S_max))  # Ensure bounds


class NetworkGrowthModel:
    """
    Implements preferential attachment network expansion model.
    
    The model follows:
    dk/dt = η * k(t) / Σ(k_j(t)) * Λ(t)
    
    where:
    - η controls network growth rate
    - κ prevents zero-degree stagnation
    - Λ(t) represents cohort-level interaction activity
    """
    
    def __init__(self, eta: float = 0.3, kappa: float = 1.0):
        """
        Initialize network growth model parameters.
        
        Args:
            eta: Network growth rate coefficient (default: 0.3)
            kappa: Stagnation prevention constant (default: 1.0)
        """
        self.eta = eta
        self.kappa = kappa
        logger.info(f"NetworkGrowthModel initialized: η={eta}, κ={kappa}")
    
    def growth_rate(self, k: int, total_connections: int, 
                    activity_level: float) -> float:
        """
        Calculate network degree growth rate using preferential attachment.
        
        Args:
            k: Current network degree
            total_connections: Sum of all student network degrees
            activity_level: Cohort-level interaction intensity
            
        Returns:
            Expected increase in network degree
        """
        if total_connections == 0:
            return self.eta * activity_level
        
        preferential_term = (k + self.kappa) / (total_connections + len([k]) * self.kappa)
        return self.eta * preferential_term * activity_level
    
    def simulate_trajectory(self, k0: int, activity_level: float,
                          time_horizon: int = 6) -> np.ndarray:
        """
        Simulate network expansion trajectory.
        
        Args:
            k0: Initial network degree
            activity_level: Constant activity level (simplified)
            time_horizon: Simulation duration
            
        Returns:
            Array of network degrees over time
        """
        trajectory = np.zeros(time_horizon + 1)
        trajectory[0] = k0
        
        for t in range(time_horizon):
            # Simplified: assume total connections grow proportionally
            growth = self.eta * (trajectory[t] + self.kappa) * activity_level
            trajectory[t + 1] = trajectory[t] + growth
        
        return trajectory
    
    def discrete_update(self, k: int, activity_level: float) -> int:
        """
        Discrete network degree update.
        
        Args:
            k: Current network degree
            activity_level: Current activity level
            
        Returns:
            Updated network degree
        """
        growth = self.eta * (k + self.kappa) * activity_level
        return int(k + growth)


class EmployabilityIndex:
    """
    Composite employability index integrating skill, network, and context.
    
    E(t) = w_s * φ_s(s(t)) + w_k * φ_k(k(t)) + w_x * φ_x(x)
    
    where φ are normalization functions and w are weights.
    """
    
    def __init__(self, w_skill: float = 0.65, w_network: float = 0.35,
                 w_context: float = 0.0, s_max: float = 10.0, k_max: int = 50):
        """
        Initialize employability index parameters.
        
        Args:
            w_skill: Weight for skill component (default: 0.65)
            w_network: Weight for network component (default: 0.35)
            w_context: Weight for contextual features (default: 0.0)
            s_max: Maximum skill level for normalization
            k_max: Maximum network degree for normalization
        """
        # Ensure weights sum to 1
        total = w_skill + w_network + w_context
        self.w_skill = w_skill / total
        self.w_network = w_network / total
        self.w_context = w_context / total
        
        self.s_max = s_max
        self.k_max = k_max
        
        logger.info(f"EmployabilityIndex initialized: weights=[{self.w_skill:.2f}, "
                   f"{self.w_network:.2f}, {self.w_context:.2f}]")
    
    def normalize_skill(self, s: float) -> float:
        """Normalize skill level to [0, 1]."""
        return min(s / self.s_max, 1.0)
    
    def normalize_network(self, k: int) -> float:
        """Normalize network degree to [0, 1]."""
        return min(k / self.k_max, 1.0)
    
    def compute(self, skill: float, network: int, context: float = 0.0) -> float:
        """
        Compute composite employability index.
        
        Args:
            skill: Current skill level
            network: Current network degree
            context: Contextual score (optional)
            
        Returns:
            Employability index in [0, 1]
        """
        skill_norm = self.normalize_skill(skill)
        network_norm = self.normalize_network(network)
        
        E = (self.w_skill * skill_norm + 
             self.w_network * network_norm + 
             self.w_context * context)
        
        return E
    
    def compute_trajectory(self, skill_trajectory: np.ndarray,
                          network_trajectory: np.ndarray) -> np.ndarray:
        """
        Compute employability index over entire trajectory.
        
        Args:
            skill_trajectory: Array of skill levels over time
            network_trajectory: Array of network degrees over time
            
        Returns:
            Array of employability indices over time
        """
        return np.array([
            self.compute(s, int(k)) 
            for s, k in zip(skill_trajectory, network_trajectory)
        ])


class EOSNMFramework:
    """
    Main EOSNM framework integrating all components.
    
    This class orchestrates:
    1. Dynamic state simulation
    2. Counterfactual trajectory generation
    3. Employability gain estimation
    4. Resource-constrained optimization
    """
    
    def __init__(self, skill_model: SkillGrowthModel,
                 network_model: NetworkGrowthModel,
                 employability_index: EmployabilityIndex):
        """
        Initialize EOSNM framework with component models.
        
        Args:
            skill_model: Configured SkillGrowthModel instance
            network_model: Configured NetworkGrowthModel instance
            employability_index: Configured EmployabilityIndex instance
        """
        self.skill_model = skill_model
        self.network_model = network_model
        self.employability_index = employability_index
        
        logger.info("EOSNM Framework initialized successfully")
    
    def simulate_baseline_trajectory(self, student: StudentState,
                                    time_horizon: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate baseline trajectory WITHOUT intervention.
        
        Args:
            student: StudentState object containing initial conditions
            time_horizon: Simulation duration in months
            
        Returns:
            Tuple of (skill_trajectory, network_trajectory)
        """
        # Simulate skill growth with existing barriers
        skill_traj = self.skill_model.simulate_trajectory(
            student.skill_level, 
            student.barrier_index, 
            time_horizon
        )
        
        # Simulate network growth with baseline activity
        network_traj = self.network_model.simulate_trajectory(
            student.network_degree,
            student.participation_intensity,
            time_horizon
        )
        
        return skill_traj, network_traj
    
    def simulate_intervention_trajectory(self, student: StudentState,
                                        time_horizon: int = 6,
                                        barrier_reduction: float = 0.3,
                                        activity_boost: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate trajectory WITH intervention (reduced barriers, increased activity).
        
        Args:
            student: StudentState object
            time_horizon: Simulation duration
            barrier_reduction: Proportion of barrier reduction (default: 0.3)
            activity_boost: Proportion of activity increase (default: 0.4)
            
        Returns:
            Tuple of (skill_trajectory, network_trajectory)
        """
        # Reduce barriers due to intervention support
        reduced_barrier = student.barrier_index * (1 - barrier_reduction)
        
        # Increase participation due to mentoring/training
        boosted_activity = student.participation_intensity * (1 + activity_boost)
        
        # Simulate with improved conditions
        skill_traj = self.skill_model.simulate_trajectory(
            student.skill_level, 
            reduced_barrier, 
            time_horizon
        )
        
        network_traj = self.network_model.simulate_trajectory(
            student.network_degree,
            boosted_activity,
            time_horizon
        )
        
        return skill_traj, network_traj
    
    def estimate_employability_gain(self, student: StudentState,
                                   time_horizon: int = 6) -> float:
        """
        Estimate cumulative employability gain from intervention.
        
        This implements Equation (4) from the paper:
        ΔE_i = Σ_t [E_i^(1)(t) - E_i^(0)(t)]
        
        Args:
            student: StudentState object
            time_horizon: Planning horizon
            
        Returns:
            Cumulative employability gain
        """
        # Generate baseline trajectory
        s_baseline, k_baseline = self.simulate_baseline_trajectory(student, time_horizon)
        E_baseline = self.employability_index.compute_trajectory(s_baseline, k_baseline)
        
        # Generate intervention trajectory
        s_intervention, k_intervention = self.simulate_intervention_trajectory(
            student, time_horizon
        )
        E_intervention = self.employability_index.compute_trajectory(
            s_intervention, k_intervention
        )
        
        # Compute cumulative gain
        delta_E = np.sum(E_intervention - E_baseline)
        
        logger.debug(f"Student {student.student_id}: ΔE = {delta_E:.4f}")
        
        return delta_E
    
    def compute_all_gains(self, students: List[StudentState],
                         time_horizon: int = 6) -> np.ndarray:
        """
        Compute employability gains for all students.
        
        Args:
            students: List of StudentState objects
            time_horizon: Planning horizon
            
        Returns:
            Array of employability gains
        """
        logger.info(f"Computing employability gains for {len(students)} students...")
        
        gains = np.array([
            self.estimate_employability_gain(student, time_horizon)
            for student in students
        ])
        
        logger.info(f"Gain computation complete. Mean gain: {gains.mean():.4f}")
        
        return gains


class ResourceAllocation:
    """
    Implements resource-constrained allocation strategies.
    
    Supports:
    1. Greedy ranking based on benefit-cost ratio
    2. Mixed-integer programming with fairness constraints
    """
    
    @staticmethod
    def greedy_allocation(gains: np.ndarray, costs: np.ndarray,
                         budget: float) -> np.ndarray:
        """
        Greedy allocation based on benefit-cost ratio.
        
        Solves: max Σ r_i * ΔE_i  subject to  Σ c_i * r_i ≤ B
        
        Args:
            gains: Array of employability gains ΔE_i
            costs: Array of intervention costs c_i
            budget: Total available budget B
            
        Returns:
            Binary allocation vector r ∈ {0,1}^N
        """
        N = len(gains)
        
        # Compute benefit-cost ratios
        ratios = gains / costs
        
        # Sort by ratio in descending order
        sorted_indices = np.argsort(-ratios)
        
        # Greedy selection
        allocation = np.zeros(N, dtype=int)
        cumulative_cost = 0.0
        
        for idx in sorted_indices:
            if cumulative_cost + costs[idx] <= budget:
                allocation[idx] = 1
                cumulative_cost += costs[idx]
        
        logger.info(f"Greedy allocation: selected {allocation.sum()} students "
                   f"with total cost {cumulative_cost:.2f} / {budget:.2f}")
        
        return allocation
    
    @staticmethod
    def allocation_with_constraints(gains: np.ndarray, costs: np.ndarray,
                                   budget: float, groups: Optional[Dict[str, np.ndarray]] = None,
                                   min_coverage: Optional[Dict[str, int]] = None) -> np.ndarray:
        """
        Allocation with fairness/diversity constraints using optimization.
        
        Solves:
            max Σ r_i * ΔE_i
            s.t. Σ c_i * r_i ≤ B
                 Σ_{i ∈ G_k} r_i ≥ L_k  for all groups k
                 r_i ∈ {0, 1}
        
        Args:
            gains: Employability gains
            costs: Intervention costs
            budget: Total budget
            groups: Dictionary mapping group names to student indices
            min_coverage: Minimum coverage per group
            
        Returns:
            Binary allocation vector
        """
        # For simplicity, fall back to greedy if no constraints
        if groups is None or min_coverage is None:
            return ResourceAllocation.greedy_allocation(gains, costs, budget)
        
        # For full MIP implementation, would use scipy.optimize.milp or pulp
        # Here we provide a simplified constrained greedy approach
        
        N = len(gains)
        allocation = np.zeros(N, dtype=int)
        cumulative_cost = 0.0
        
        # First, satisfy minimum coverage constraints
        for group_name, indices in groups.items():
            min_req = min_coverage.get(group_name, 0)
            
            # Sort group members by gain
            group_gains = [(idx, gains[idx]) for idx in indices if allocation[idx] == 0]
            group_gains.sort(key=lambda x: x[1], reverse=True)
            
            # Allocate minimum required
            allocated_in_group = 0
            for idx, _ in group_gains:
                if allocated_in_group >= min_req:
                    break
                if cumulative_cost + costs[idx] <= budget:
                    allocation[idx] = 1
                    cumulative_cost += costs[idx]
                    allocated_in_group += 1
        
        # Then fill remaining budget greedily
        ratios = gains / costs
        sorted_indices = np.argsort(-ratios)
        
        for idx in sorted_indices:
            if allocation[idx] == 0 and cumulative_cost + costs[idx] <= budget:
                allocation[idx] = 1
                cumulative_cost += costs[idx]
        
        logger.info(f"Constrained allocation: selected {allocation.sum()} students")
        
        return allocation


def run_eosnm_pipeline(students: List[StudentState], budget: float,
                       time_horizon: int = 6) -> Dict:
    """
    Complete EOSNM pipeline: simulate, estimate gains, and optimize allocation.
    
    Args:
        students: List of StudentState objects
        budget: Available resource budget
        time_horizon: Planning horizon in months
        
    Returns:
        Dictionary containing allocation results and metrics
    """
    logger.info("=" * 60)
    logger.info("Starting EOSNM Pipeline")
    logger.info("=" * 60)
    
    # Initialize models
    skill_model = SkillGrowthModel(alpha=0.5, S_max=10.0, beta=0.3)
    network_model = NetworkGrowthModel(eta=0.3, kappa=1.0)
    employability_index = EmployabilityIndex(w_skill=0.65, w_network=0.35)
    
    # Create EOSNM framework
    eosnm = EOSNMFramework(skill_model, network_model, employability_index)
    
    # Compute employability gains for all students
    gains = eosnm.compute_all_gains(students, time_horizon)
    
    # Assume unit costs for simplicity
    costs = np.ones(len(students))
    
    # Perform greedy allocation
    allocation = ResourceAllocation.greedy_allocation(gains, costs, budget)
    
    # Compute metrics
    total_gain = np.sum(gains * allocation)
    selected_students = np.where(allocation == 1)[0]
    
    results = {
        'allocation': allocation,
        'gains': gains,
        'total_gain': total_gain,
        'selected_students': selected_students,
        'num_selected': len(selected_students),
        'budget_used': np.sum(costs * allocation),
        'mean_gain_selected': gains[selected_students].mean() if len(selected_students) > 0 else 0
    }
    
    logger.info(f"Pipeline complete: {results['num_selected']} students selected")
    logger.info(f"Total employability gain: {results['total_gain']:.4f}")
    logger.info("=" * 60)
    
    return results
