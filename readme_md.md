# EOSNM: Employability Optimization through Skill-Network Modelling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Official Implementation** of the paper:  
*"EOSNM: A Behaviour-Informed Framework for Optimising Student Employability under Budget Constraints"*

**Authors:** Pankaj Mishra, V Venkataramanan, Anand Nayyar  
**Institution:** K J Somaiya School of Engineering, Somaiya Vidyavihar University

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Experiments](#experiments)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

EOSNM is a **dynamic, behaviorally-grounded framework** for optimizing student employability in entrepreneurship education contexts. Unlike traditional descriptive approaches, EOSNM provides **prescriptive, resource-aware decision support** for allocating limited institutional resources (mentoring, training, incubation) to maximize placement outcomes.

### The Problem

Entrepreneurship Cells (E-Cells) face a critical challenge:
- **Limited resources** (mentoring hours, training slots, incubation capacity)
- **Heterogeneous student readiness** (varying skills, networks, barriers)
- **Uncertain outcomes** (which students will benefit most from intervention?)

Traditional approaches use:
- ❌ Subjective committee reviews
- ❌ Simple participation metrics (event attendance)
- ❌ Static skill assessments

### The Solution

EOSNM provides:
- ✅ **Dynamic modeling** of skill growth and network expansion
- ✅ **Counterfactual simulation** (with/without intervention)
- ✅ **Optimization-driven allocation** under budget constraints
- ✅ **Fairness-aware selection** with group coverage guarantees

### Key Results

Compared to baseline methods, EOSNM achieves:
- **12-18% improvement** in placement yield
- **25-32% increase** in aggregate employability gains
- **Consistent performance** across budget levels (5-20% of cohort)

---

## ✨ Key Features

### 1. **Dynamic Employability Modeling**
- Logistic skill growth with barrier effects
- Preferential attachment network expansion
- Composite employability index (skill + network + context)

### 2. **Counterfactual Simulation**
- Baseline trajectory (no intervention)
- Intervention trajectory (with support)
- Cumulative employability gain estimation

### 3. **Resource-Constrained Optimization**
- Greedy allocation (O(N log N) efficiency)
- Mixed-integer programming with fairness constraints
- Multi-period allocation support

### 4. **Comprehensive Evaluation**
- Placement yield, Precision@k, ROC-AUC, PR-AUC
- Cumulative gain analysis
- Group coverage ratio (fairness metric)
- Bootstrap confidence intervals

### 5. **Publication-Quality Visualizations**
- Employability distributions
- Skill/network trajectories
- Comparative performance plots
- ROC/PR curves

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool (venv, conda)

### Step 1: Clone the Repository

```bash
git clone https://github.com/PanksMish/ECELL.git
cd ECELL
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv eosnm_env
source eosnm_env/bin/activate  # On Windows: eosnm_env\Scripts\activate

# OR using conda
conda create -n eosnm python=3.8
conda activate eosnm
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import numpy, pandas, scipy, sklearn; print('Installation successful!')"
```

---

## 🏃 Quick Start

### Option 1: Run with Synthetic Data (No Data Required)

```bash
python experiment_runner.py
```

This will:
1. Generate synthetic E-Cell dataset (391 students)
2. Run EOSNM framework
3. Compare with baselines (Random, Skill-only, Network-only, SEM)
4. Generate all figures and results

### Option 2: Run with Your Own Data

```bash
python experiment_runner.py --data path/to/your/data.csv --output-dir my_results
```

**Data Format:**

Your CSV should contain:

| Column | Description | Type |
|--------|-------------|------|
| `student_id` | Unique identifier | int |
| `baseline_skill_score` | Initial skill level (0-10) | float |
| `network_degree` | Number of professional connections | int |
| `barrier_index` | Composite barrier score (0-1) | float |
| `participation_intensity` | Engagement level (0-1) | float |
| `placement_outcome` | Binary placement outcome | int (0/1) |
| `department` | Academic department | string |
| `year_of_study` | Year level (2-4) | int |

### Option 3: Interactive Notebook

```bash
jupyter notebook notebooks/EOSNM_Demo.ipynb
```

---

## 📁 Project Structure

```
EOSNM/
├── eosnm_core.py              # Core framework implementation
├── data_loader.py             # Data loading and preprocessing
├── evaluation_metrics.py      # Metrics and baseline comparisons
├── visualization.py           # Publication-quality plotting
├── experiment_runner.py       # Main orchestrator
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
│
├── configs/                   # Configuration files
│   ├── default_config.yaml    # Default parameters
│   └── custom_config.yaml     # Custom experiment settings
│
├── data/                      # Data directory
│   ├── ecell_data.csv         # Synthetic dataset (generated)
│   └── README.md              # Data format documentation
│
├── notebooks/                 # Jupyter notebooks
│   ├── EOSNM_Demo.ipynb       # Interactive demo
│   └── Analysis.ipynb         # Results analysis
│
├── results/                   # Experiment outputs (generated)
│   ├── comparative_results_*.csv
│   ├── confidence_intervals_*.json
│   └── summary_*.json
│
├── figures/                   # Generated visualizations
│   ├── fig1_employability_dist_*.png
│   ├── fig2_skill_trajectories_*.png
│   └── ...
│
└── tests/                     # Unit tests
    ├── test_core.py
    ├── test_data_loader.py
    └── test_evaluation.py
```

---

## 📚 Usage

### Basic Usage: Python API

```python
from eosnm_core import (
    SkillGrowthModel, NetworkGrowthModel, 
    EmployabilityIndex, EOSNMFramework
)
from data_loader import load_ecell_data

# Load data
students, df = load_ecell_data('data/ecell_data.csv')

# Initialize models
skill_model = SkillGrowthModel(alpha=0.5, S_max=10.0, beta=0.3)
network_model = NetworkGrowthModel(eta=0.3, kappa=1.0)
emp_index = EmployabilityIndex(w_skill=0.65, w_network=0.35)

# Create framework
eosnm = EOSNMFramework(skill_model, network_model, emp_index)

# Compute employability gains
gains = eosnm.compute_all_gains(students, time_horizon=6)

# Perform allocation
from eosnm_core import ResourceAllocation

budget = len(students) * 0.15  # 15% budget
allocation = ResourceAllocation.greedy_allocation(
    gains, costs=np.ones(len(students)), budget=budget
)

print(f"Selected {allocation.sum()} students for intervention")
```

### Advanced: Custom Configuration

Create `configs/my_experiment.yaml`:

```yaml
data:
  path: "data/my_ecell_data.csv"
  random_state: 42

model:
  skill_growth:
    alpha: 0.6        # Faster learning rate
    S_max: 10.0
    beta: 0.25        # Lower barrier impact
  
  network_growth:
    eta: 0.35         # Faster network growth
    kappa: 1.0
  
  employability_index:
    w_skill: 0.70     # Higher skill weight
    w_network: 0.30
    w_context: 0.0

experiment:
  time_horizon: 8     # Longer planning horizon
  budget_fractions: [0.10, 0.15, 0.20, 0.25]
  n_bootstrap: 200
  
output:
  results_dir: "my_results"
  figures_dir: "my_results/figures"
```

Run with custom config:

```bash
python experiment_runner.py --config configs/my_experiment.yaml
```

---

## 🧪 Experiments

### Reproduce Paper Results

```bash
# Run full experimental pipeline
python experiment_runner.py --config configs/default_config.yaml

# Results will be saved to:
# - results/comparative_results_*.csv
# - results/confidence_intervals_*.json
# - figures/fig*_*.png
```

### Comparative Evaluation

The framework automatically compares EOSNM against:

1. **Random**: Random selection baseline
2. **Skill-only**: Allocate by skill level
3. **Network-only**: Allocate by network degree
4. **SEM-based**: Logistic regression on features
5. **EOSNM-Greedy**: Proposed method (greedy)
6. **EOSNM-MIP**: Proposed method with fairness constraints

### Metrics Computed

- **Placement Yield (%)**: Proportion of selected students placed
- **Aggregate Gain (ΔE)**: Total employability improvement
- **Precision@k**: Precision among top-k selections
- **Recall@k**: Coverage of successful placements
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve
- **Group Coverage Ratio**: Fairness metric across groups

---

## 📊 Results

### Table 1: Placement Yield by Method and Budget

| Method | 5% | 10% | 15% | 20% | Aggregate Gain |
|--------|-----|-----|-----|-----|----------------|
| **EOSNM-MIP** | **49.6±1.8** | **53.4±1.9** | **57.0±2.0** | **58.2±2.1** | **27.1±1.2** |
| **EOSNM-Greedy** | 47.2±1.9 | 50.8±1.8 | 53.0±2.0 | 54.5±2.1 | 25.4±1.3 |
| SEM-based | 38.6±2.1 | 41.0±2.0 | 43.0±2.0 | 44.2±2.2 | 19.9±1.1 |
| Skill-only | 35.1±2.2 | 37.5±2.1 | 39.0±2.0 | 40.1±2.1 | 17.5±1.0 |
| Network-only | 33.4±2.4 | 34.8±2.3 | 36.0±3.0 | 37.2±2.8 | 15.8±1.1 |
| Random | 25.8±2.8 | 27.1±3.0 | 28.0±3.0 | 28.9±3.2 | 12.4±1.2 |

**Key Findings:**
- EOSNM-MIP achieves **12-18% higher placement yield** than skill-only baselines
- **25-32% improvement** in aggregate employability gain
- Consistent performance across all budget levels

### Figure Gallery

All figures from the paper are automatically generated:

- **Figure 1:** Initial employability distribution
- **Figure 2:** Skill development trajectories (with/without intervention)
- **Figure 3:** Network degree distribution (power-law)
- **Figure 4:** Placement yield vs. budget comparison
- **Figure 5:** Performance heatmap across methods
- **Figure 6:** ROC and Precision-Recall curves
- **Figure 7:** Cumulative gain analysis
- **Figure 8:** Group coverage ratio (fairness)

---

## 📖 Citation

If you use EOSNM in your research, please cite:

```bibtex
@article{mishra2025eosnm,
  title={EOSNM: A Behaviour-Informed Framework for Optimising Student Employability under Budget Constraints},
  author={Mishra, Pankaj and Venkataramanan, V and Nayyar, Anand},
  journal={[Journal Name]},
  year={2025},
  publisher={[Publisher]}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Pankaj Mishra, V Venkataramanan, Anand Nayyar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👥 Contact

### Authors

**Pankaj Mishra**  
Department of Information Technology  
K J Somaiya School of Engineering  
Email: pankaj.mishra@somaiya.edu

**V Venkataramanan**  
Department of Information Technology  
K J Somaiya School of Engineering  
Email: venkataramanan@somaiya.edu

**Anand Nayyar** (Corresponding Author)  
School of Computer Science  
Duy Tan University, Da Nang, Vietnam  
Email: anandnayyar@duytan.edu.vn

### Issues and Contributions

- **Issues:** Please report bugs or request features via [GitHub Issues](https://github.com/PanksMish/ECELL/issues)
- **Pull Requests:** Contributions are welcome! Please fork and submit a PR
- **Discussions:** Join our [GitHub Discussions](https://github.com/PanksMish/ECELL/discussions) for Q&A

---

## 🙏 Acknowledgments

This research was supported by:
- K J Somaiya School of Engineering, Somaiya Vidyavihar University
- Duy Tan University, Vietnam

We thank all E-Cell coordinators and student participants who made this research possible.

---

## 🔄 Updates

- **v1.0.0** (2025-01-13): Initial release with full implementation
- See [CHANGELOG.md](CHANGELOG.md) for detailed version history

---

**Last Updated:** January 13, 2025
