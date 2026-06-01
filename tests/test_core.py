"""
Unit tests for EOSNM core framework components
"""

import pytest
import numpy as np
from eosnm_core import (
    SkillGrowthModel,
    NetworkGrowthModel,
    EmployabilityIndex,
    EOSNMFramework,
    ResourceAllocation,
    StudentState
)


class TestSkillGrowthModel:
    """Test cases for SkillGrowthModel"""
    
    def test_initialization(self):
        """Test model initialization with default parameters"""
        model = SkillGrowthModel()
        assert model.alpha > 0
        assert model.S_max > 0
        assert model.beta >= 0
    
    def test_growth_rate(self):
        """Test skill growth rate computation"""
        model = SkillGrowthModel(alpha=0.5, S_max=10.0, beta=0.3)
        
        # Test at s=0 (should grow)
        rate_zero = model.growth_rate(s=0.1, t=0, barrier=0)
        assert rate_zero > 0, "Skill should grow from low level"
        
        # Test at s=S_max (should not grow much)
        rate_max = model.growth_rate(s=9.9, t=0, barrier=0)
        assert rate_max < rate_zero, "Growth should slow near saturation"
        
        # Test barrier effect
        rate_no_barrier = model.growth_rate(s=5.0, t=0, barrier=0)
        rate_with_barrier = model.growth_rate(s=5.0, t=0, barrier=0.5)
        assert rate_with_barrier < rate_no_barrier, "Barriers should reduce growth"
    
    def test_trajectory_simulation(self):
        """Test trajectory simulation over time"""
        model = SkillGrowthModel(alpha=0.5, S_max=10.0, beta=0.3)
        trajectory = model.simulate_trajectory(s0=2.0, barrier=0.2, time_horizon=6)
        
        assert len(trajectory) == 7  # T+1 points
        assert trajectory[0] == 2.0  # Initial condition
        assert trajectory[-1] > trajectory[0]  # Should grow
        assert np.all(trajectory <= 10.0)  # Should not exceed S_max
    
    def test_discrete_update(self):
        """Test discrete time update"""
        model = SkillGrowthModel(alpha=0.5, S_max=10.0, beta=0.3)
        s_next = model.discrete_update(s=5.0, barrier=0.3)
        
        assert isinstance(s_next, float)
        assert 0 <= s_next <= 10.0  # Bounded


class TestNetworkGrowthModel:
    """Test cases for NetworkGrowthModel"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = NetworkGrowthModel()
        assert model.eta > 0
        assert model.kappa >= 0
    
    def test_growth_rate(self):
        """Test network growth rate"""
        model = NetworkGrowthModel(eta=0.3, kappa=1.0)
        
        # Higher degree should grow faster (preferential attachment)
        rate_low = model.growth_rate(k=5, total_connections=100, activity_level=1.0)
        rate_high = model.growth_rate(k=15, total_connections=100, activity_level=1.0)
        assert rate_high > rate_low, "Higher degree should attract more connections"
    
    def test_trajectory_simulation(self):
        """Test network trajectory simulation"""
        model = NetworkGrowthModel(eta=0.3, kappa=1.0)
        trajectory = model.simulate_trajectory(k0=5, activity_level=0.8, time_horizon=6)
        
        assert len(trajectory) == 7
        assert trajectory[0] == 5
        assert trajectory[-1] > trajectory[0]  # Should grow
    
    def test_discrete_update(self):
        """Test discrete network update"""
        model = NetworkGrowthModel(eta=0.3, kappa=1.0)
        k_next = model.discrete_update(k=10, activity_level=0.8)
        
        assert isinstance(k_next, int)
        assert k_next >= 10  # Should not decrease


class TestEmployabilityIndex:
    """Test cases for EmployabilityIndex"""
    
    def test_initialization(self):
        """Test index initialization with weights"""
        index = EmployabilityIndex(w_skill=0.7, w_network=0.3)
        
        # Weights should be normalized to sum to 1
        assert np.isclose(index.w_skill + index.w_network + index.w_context, 1.0)
    
    def test_normalization(self):
        """Test skill and network normalization"""
        index = EmployabilityIndex(s_max=10.0, k_max=50)
        
        assert index.normalize_skill(5.0) == 0.5
        assert index.normalize_skill(10.0) == 1.0
        assert index.normalize_network(25) == 0.5
        assert index.normalize_network(50) == 1.0
    
    def test_compute(self):
        """Test employability computation"""
        index = EmployabilityIndex(w_skill=0.6, w_network=0.4, s_max=10.0, k_max=50)
        
        E = index.compute(skill=5.0, network=20)
        
        assert 0 <= E <= 1, "Employability should be in [0, 1]"
        assert isinstance(E, float)
    
    def test_monotonicity(self):
        """Test that higher skill/network gives higher employability"""
        index = EmployabilityIndex(w_skill=0.6, w_network=0.4)
        
        E_low = index.compute(skill=3.0, network=10)
        E_high = index.compute(skill=7.0, network=30)
        
        assert E_high > E_low, "Higher skill and network should increase employability"


class TestEOSNMFramework:
    """Test cases for complete EOSNM framework"""
    
    @pytest.fixture
    def framework(self):
        """Create EOSNM framework for testing"""
        skill_model = SkillGrowthModel(alpha=0.5, S_max=10.0, beta=0.3)
        network_model = NetworkGrowthModel(eta=0.3, kappa=1.0)
        emp_index = EmployabilityIndex(w_skill=0.65, w_network=0.35)
        return EOSNMFramework(skill_model, network_model, emp_index)
    
    @pytest.fixture
    def sample_student(self):
        """Create sample student for testing"""
        return StudentState(
            student_id=1,
            skill_level=5.0,
            network_degree=15,
            barrier_index=0.4,
            participation_intensity=0.7,
            demographics={'year': 3, 'department': 'CS'}
        )
    
    def test_baseline_trajectory(self, framework, sample_student):
        """Test baseline trajectory simulation"""
        skill_traj, network_traj = framework.simulate_baseline_trajectory(
            sample_student, time_horizon=6
        )
        
        assert len(skill_traj) == 7
        assert len(network_traj) == 7
        assert skill_traj[0] == sample_student.skill_level
        assert network_traj[0] == sample_student.network_degree
    
    def test_intervention_trajectory(self, framework, sample_student):
        """Test intervention trajectory simulation"""
        skill_int, network_int = framework.simulate_intervention_trajectory(
            sample_student, time_horizon=6
        )
        
        skill_base, network_base = framework.simulate_baseline_trajectory(
            sample_student, time_horizon=6
        )
        
        # Intervention should lead to better outcomes
        assert skill_int[-1] > skill_base[-1], "Intervention should improve skill"
        assert network_int[-1] > network_base[-1], "Intervention should expand network"
    
    def test_employability_gain(self, framework, sample_student):
        """Test employability gain estimation"""
        gain = framework.estimate_employability_gain(sample_student, time_horizon=6)
        
        assert isinstance(gain, float)
        assert gain > 0, "Intervention should provide positive gain"
    
    def test_compute_all_gains(self, framework):
        """Test gain computation for multiple students"""
        students = [
            StudentState(i, 5.0 + i*0.5, 10+i, 0.3, 0.6, {'year': 3})
            for i in range(10)
        ]
        
        gains = framework.compute_all_gains(students, time_horizon=6)
        
        assert len(gains) == 10
        assert np.all(gains > 0), "All gains should be positive"


class TestResourceAllocation:
    """Test cases for resource allocation strategies"""
    
    def test_greedy_allocation(self):
        """Test greedy allocation algorithm"""
        gains = np.array([5.0, 3.0, 8.0, 2.0, 6.0])
        costs = np.ones(5)
        budget = 3.0
        
        allocation = ResourceAllocation.greedy_allocation(gains, costs, budget)
        
        assert allocation.sum() == 3, "Should allocate exactly budget"
        # Should select indices 2, 4, 0 (highest gains)
        assert allocation[2] == 1  # Highest gain (8.0)
        assert allocation[4] == 1  # Second highest (6.0)
        assert allocation[0] == 1  # Third highest (5.0)
    
    def test_allocation_respects_budget(self):
        """Test that allocation respects budget constraint"""
        np.random.seed(42)
        gains = np.random.random(100)
        costs = np.ones(100)
        budget = 20.0
        
        allocation = ResourceAllocation.greedy_allocation(gains, costs, budget)
        
        total_cost = np.sum(allocation * costs)
        assert total_cost <= budget, "Should not exceed budget"
    
    def test_allocation_with_varying_costs(self):
        """Test allocation with non-uniform costs"""
        gains = np.array([10.0, 8.0, 6.0, 4.0])
        costs = np.array([1.0, 2.0, 1.0, 3.0])
        budget = 3.0
        
        allocation = ResourceAllocation.greedy_allocation(gains, costs, budget)
        
        total_cost = np.sum(allocation * costs)
        assert total_cost <= budget


class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test complete EOSNM pipeline"""
        # Create sample students
        students = [
            StudentState(
                student_id=i,
                skill_level=np.random.uniform(3, 8),
                network_degree=np.random.randint(5, 30),
                barrier_index=np.random.uniform(0.2, 0.6),
                participation_intensity=np.random.uniform(0.4, 0.9),
                demographics={'year': 3, 'department': 'CS'}
            )
            for i in range(50)
        ]
        
        # Run pipeline
        results = run_eosnm_pipeline(students, budget=10, time_horizon=6)
        
        # Verify results structure
        assert 'allocation' in results
        assert 'gains' in results
        assert 'total_gain' in results
        assert 'selected_students' in results
        
        # Verify allocation
        assert results['allocation'].sum() == 10, "Should select 10 students"
        assert len(results['selected_students']) == 10
        
        # Verify gains
        assert len(results['gains']) == 50
        assert results['total_gain'] > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
