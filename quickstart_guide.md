# EOSNM Quick Start Guide

Get up and running with EOSNM in 5 minutes!

## 📦 Installation (2 minutes)

```bash
# Clone repository
git clone https://github.com/PanksMish/ECELL.git
cd ECELL

# Create virtual environment
python -m venv eosnm_env
source eosnm_env/bin/activate  # Windows: eosnm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Run Your First Experiment (3 minutes)

### Option 1: Use Synthetic Data (Fastest)

```bash
python experiment_runner.py
```

This will:
- ✅ Generate synthetic E-Cell data (391 students)
- ✅ Run EOSNM optimization
- ✅ Compare with 5 baseline methods
- ✅ Generate 8 publication-quality figures
- ✅ Save results to `results/` and `figures/`

**Expected output:**
```
==========================================
EOSNM Experiment initialized: 20250113_143022
Results directory: results
Figures directory: figures
==========================================

[STEP 1] Loading and preprocessing data...
Loaded 391 students
Placement rate: 49.4%

[STEP 2] Initializing EOSNM framework...
SkillGrowthModel initialized: α=0.5, S_max=10.0, β=0.3
NetworkGrowthModel initialized: η=0.3, κ=1.0
EmployabilityIndex initialized: weights=[0.65, 0.35, 0.00]

[STEP 3] Computing employability gains...
Computing employability gains for 391 students...
Gain computation complete. Mean gain: 2.1847

[STEP 4] Running comparative evaluation...
  Budget: 5% (19 students)
  Budget: 10% (39 students)
  Budget: 15% (58 students)
  Budget: 20% (78 students)

[STEP 5] Computing confidence intervals...
[STEP 6] Generating visualizations...
[STEP 7] Saving results...

==========================================
EXPERIMENT COMPLETE
==========================================

Top 3 Methods by Placement Yield (15% budget):
         Method  Placement Yield (%)
     EOSNM-MIP                 57.0
  EOSNM-Greedy                 53.0
     SEM-based                 43.0
```

### Option 2: Use Your Own Data

1. **Prepare your CSV file** with these columns:

| Required Columns | Type | Description |
|-----------------|------|-------------|
| `student_id` | int | Unique identifier |
| `baseline_skill_score` | float | Initial skill (0-10) |
| `network_degree` | int | Number of connections |
| `barrier_index` | float | Barrier score (0-1) |
| `participation_intensity` | float | Activity level (0-1) |
| `placement_outcome` | int | Binary outcome (0/1) |
| `department` | string | Academic department |
| `year_of_study` | int | Year level |

2. **Run with your data:**

```bash
python experiment_runner.py --data path/to/your_data.csv --output-dir my_results
```

## 📊 View Results

### Generated Files

**Results** (in `results/` directory):
```
results/
├── comparative_results_20250113_143022.csv     # Performance comparison
├── confidence_intervals_20250113_143022.json   # 95% CIs
├── predictions_20250113_143022.csv             # Per-student predictions
└── summary_20250113_143022.json                # Experiment summary
```

**Figures** (in `figures/` directory):
```
figures/
├── fig1_employability_dist_*.png               # Initial distribution
├── fig2_skill_trajectories_*.png               # Skill growth curves
├── fig3_network_dist_*.png                     # Network structure
├── fig4_placement_yield_*.png                  # Method comparison
├── fig5_performance_matrix_*.png               # Heatmap
├── fig6_roc_pr_curves_*.png                    # ROC/PR curves
├── fig7_cumulative_gain_*.png                  # Gain analysis
└── fig8_group_coverage_*.png                   # Fairness metrics
```

### Quick Analysis

```bash
# View comparative results
cat results/comparative_results_*.csv

# View summary
cat results/summary_*.json | python -m json.tool
```

## 🔧 Basic Customization

### Change Budget Levels

Edit `configs/default_config.yaml`:

```yaml
experiment:
  budget_fractions: [0.05, 0.10, 0.15, 0.20, 0.25]  # Add 25%
```

### Adjust Model Parameters

```yaml
model:
  skill_growth:
    alpha: 0.6     # Faster learning (default: 0.5)
    beta: 0.2      # Lower barrier impact (default: 0.3)
  
  employability_index:
    w_skill: 0.70     # More weight on skills (default: 0.65)
    w_network: 0.30   # Less weight on network (default: 0.35)
```

### Change Planning Horizon

```yaml
experiment:
  time_horizon: 8  # 8 months instead of 6
```

Then run:
```bash
python experiment_runner.py --config configs/default_config.yaml
```

## 🐍 Python API Usage

```python
from eosnm_core import (
    SkillGrowthModel, NetworkGrowthModel,
    EmployabilityIndex, EOSNMFramework,
    ResourceAllocation, StudentState
)
from data_loader import load_ecell_data
import numpy as np

# Load data
students, df = load_ecell_data('data/ecell_data.csv')

# Initialize framework
skill_model = SkillGrowthModel(alpha=0.5, S_max=10.0, beta=0.3)
network_model = NetworkGrowthModel(eta=0.3, kappa=1.0)
emp_index = EmployabilityIndex(w_skill=0.65, w_network=0.35)
eosnm = EOSNMFramework(skill_model, network_model, emp_index)

# Compute gains
gains = eosnm.compute_all_gains(students, time_horizon=6)

# Allocate resources
budget = int(len(students) * 0.15)  # 15% budget
allocation = ResourceAllocation.greedy_allocation(
    gains, 
    costs=np.ones(len(students)), 
    budget=budget
)

# Get selected students
selected = np.where(allocation == 1)[0]
print(f"Selected {len(selected)} students:")
for idx in selected[:5]:  # Show first 5
    s = students[idx]
    print(f"  Student {s.student_id}: skill={s.skill_level:.2f}, "
          f"network={s.network_degree}, gain={gains[idx]:.4f}")
```

## 📝 Common Tasks

### Task 1: Generate Dataset Only

```python
from data_loader import save_synthetic_dataset

# Generate and save
save_synthetic_dataset('data/my_data.csv', n_students=500)
```

### Task 2: Evaluate Single Student

```python
from eosnm_core import StudentState

# Create student
student = StudentState(
    student_id=1,
    skill_level=6.0,
    network_degree=20,
    barrier_index=0.3,
    participation_intensity=0.8,
    demographics={'year': 3, 'dept': 'CS'}
)

# Estimate gain
gain = eosnm.estimate_employability_gain(student, time_horizon=6)
print(f"Expected employability gain: {gain:.4f}")
```

### Task 3: Custom Visualization

```python
from visualization import EOSNMVisualizer
import numpy as np

visualizer = EOSNMVisualizer()

# Plot custom data
E0_scores = np.array([s.skill_level for s in students])
visualizer.plot_employability_distribution(
    E0_scores,
    save_path='my_custom_plot.png'
)
```

## ⚡ Performance Tips

1. **For large cohorts (N > 1000):**
   - Use `EOSNM-Greedy` instead of `EOSNM-MIP`
   - Reduce `n_bootstrap` to 50

2. **For faster prototyping:**
   - Reduce `time_horizon` to 3 months
   - Test with fewer budget fractions

3. **For production deployment:**
   - Set `reproducibility.set_seeds: true`
   - Increase `n_bootstrap` to 200

## 🆘 Troubleshooting

### ImportError: No module named 'eosnm_core'

**Solution:** Make sure you're in the ECELL directory and have activated your virtual environment.

```bash
cd ECELL
source eosnm_env/bin/activate
```

### ValueError: Invalid data format

**Solution:** Check that your CSV has all required columns. Run:

```python
import pandas as pd
df = pd.read_csv('your_data.csv')
print(df.columns)
```

### Low placement yields (<30%)

**Solution:** This may indicate:
- Data quality issues (check for outliers)
- Unrealistic parameter settings
- Very low baseline placement rate

Try regenerating synthetic data or adjusting model parameters.

## 🎓 Next Steps

1. **Read the full paper** to understand the methodology
2. **Explore `notebooks/EOSNM_Demo.ipynb`** for interactive examples
3. **Check `tests/test_core.py`** to understand the API
4. **Join GitHub Discussions** for community support

## 📚 Additional Resources

- **Full Documentation:** [README.md](README.md)
- **API Reference:** Run `pydoc eosnm_core`
- **Paper:** [Link to paper]
- **Issues:** https://github.com/PanksMish/ECELL/issues

---

**Congratulations! You're now ready to use EOSNM.** 🎉

For questions or support, contact: pankaj.mishra@somaiya.edu
