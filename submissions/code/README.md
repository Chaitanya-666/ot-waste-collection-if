# Municipal Waste Collection Route Optimization

**Vehicle Routing Problem with Intermediate Facilities using Adaptive Large Neighborhood Search (ALNS)**

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)
[![Test Status](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)](#testing)

---

## 📜 Table of Contents

1.  [**Project Overview**](#-project-overview)
2.  [**Theoretical Foundation**](#-theoretical-foundation)
3.  [**Implementation Architecture**](#️-implementation-architecture)
4.  [**Step-by-Step Implementation Guide**](#️-step-by-step-implementation-guide)
5.  [**Author Contributions**](#-author-contributions)
6.  [**Usage Guide**](#-usage-guide)
7.  [**Performance and Benchmarks**](#-performance-and-benchmarks)
8.  [**Testing and Validation**](#-testing-and-validation)
9.  [**Final Deliverables**](#-final-deliverables)
10. [**Project Structure**](#-project-structure)
11. [**Advanced Configuration**](#️-advanced-configuration)
12. [**Academic and Research Applications**](#-academic-and-research-applications)
13. [**Support and Contributing**](#-support-and-contributing)
14. [**License and Citation**](#-license-and-citation)

---

## 🎯 **Project Overview**

This project implements an advanced **Vehicle Routing Problem (VRP) with Intermediate Facilities** specifically designed for municipal waste collection optimization. The system uses **Adaptive Large Neighborhood Search (ALNS)** to find optimal waste collection routes that minimize total transportation costs while respecting vehicle capacity constraints and intermediate facility visits.

### **Problem Context**

**Municipal waste collection** faces unique challenges:
- **Limited vehicle capacity** requires intermediate facility visits for waste disposal
- **Multiple depots and facilities** with different capacities and locations
- **Time-dependent constraints** for collection windows
- **Environmental optimization** to reduce fuel consumption and emissions
- **Cost minimization** for municipal budgeting

Our solution addresses these challenges through intelligent route optimization that automatically determines:
- **Optimal vehicle assignments** to collection zones
- **Efficient visiting sequences** to minimize travel distances
- **Strategic intermediate facility utilization** for capacity management
- **Real-time optimization tracking** through animated visualizations

---

## 🔬 **Theoretical Foundation**

### **Vehicle Routing Problem (VRP) with Intermediate Facilities**

The **VRP with Intermediate Facilities** extends the classical VRP by introducing:

**Core Problem Formulation:**
```
Given:
- Set of customers C with demand d_i
- Set of intermediate facilities F with disposal capacity
- Depot D with vehicle fleet
- Vehicle capacity Q
- Distance matrix dist(i,j)

Objective:
minimize Σ(dist(i,j) × flow(i,j)) for all vehicle routes

Subject to:
- Each customer visited exactly once
- Vehicle capacity constraints: Σ(d_i) ≤ Q
- Facility capacity constraints
- Depot start/end constraints
```

**Key Characteristics:**
- **Multi-depot capability** for distributed collection systems
- **Intermediate facility visits** required for capacity management
- **Dynamic route planning** with real-time constraint satisfaction
- **Scalable solution** for municipal-scale problems (20-100+ customers)

### **Adaptive Large Neighborhood Search (ALNS)**

**ALNS** is a metaheuristic optimization algorithm that combines:

**1. Destruction Operators** - Remove customers from current solution
- **Random removal**: Remove random customers
- **Related removal**: Remove geographically close customers
- **Worst removal**: Remove customers causing high costs
- **Shaw removal**: Remove customers with similar characteristics

**2. Repair Operators** - Reinsert customers into solution
- **Greedy insertion**: Insert at best position by cost
- **Regret insertion**: Use regret-based decision making
- **Random insertion**: Random repositioning for exploration

**3. Adaptive Selection** - Choose operators based on performance
- **Score-based adaptation**: Operators earn scores for improvements
- **Temperature-based acceptance**: Simulated annealing component
- **Response surface adaptation**: Learn from solution landscape

**ALNS Algorithm Steps:**
```
Initialize with constructive heuristic
repeat for max_iterations:
  select destroy operator based on weights
  select repair operator based on weights
  apply operators to create new solution
  evaluate new solution using acceptance criterion
  update operator weights based on performance
  update best solution if improved
```

**Mathematical Foundation:**
- **Cost Function**: Total_distance + α×Time + β×Unassigned_penalty
- **Adaptive Weights**: w_i(t+1) = w_i(t) × (1-ρ) + ρ×score_i/total_score
- **Acceptance Criterion**: exp(-Δ/T) where T decreases over time

---

## 🏗️ **Implementation Architecture**

### **System Design Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Application Layer                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   CLI Interface │  │  Video Creation │  │ Visualization│ │
│  │   (main.py)     │  │ Integration     │  │ (ASCII/MPL)  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                  Core ALNS Optimization Layer               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   ALNS Engine   │  │  Destruction    │  │  Repair      │ │
│  │   (alns.py)     │  │  Operators      │  │  Operators   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                    Problem Definition Layer                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Problem       │  │   Solution      │  │   Route      │ │
│  │   (problem.py)  │  │   (solution.py) │  │  (route.py)  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                   Utility & Analysis Layer                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Performance     │  │   Route         │  │   Data       │ │
│  │   Analyzer      │  │   Visualizer    │  │   Generator  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### **Core Components**

#### **1. Problem Definition (`src/problem.py`)**
```python
class ProblemInstance:
    """Defines VRP-IF problem parameters and constraints"""

    def __init__(self, name: str):
        self.name = name
        self.depot: Location = None
        self.customers: List[Location] = []
        self.intermediate_facilities: List[Location] = []
        self.vehicle_capacity: int = 0
        self.number_of_vehicles: int = float('inf')
        self.disposal_time: float = 0
        self.distance_matrix: List[List[float]] = []

    def calculate_distance_matrix(self):
        """Compute pairwise distances using Euclidean metric"""
        # Implementation details...

    def is_feasible(self) -> Tuple[bool, str]:
        """Check problem feasibility given constraints"""
        # Implementation details...
```

#### **2. ALNS Optimization Engine (`src/alns.py`)**
```python
class ALNS:
    """Adaptive Large Neighborhood Search implementation"""

    def __init__(self, problem: ProblemInstance):
        self.problem = problem
        self.destroy_operators = [
            RandomRemoval(),
            RelatedRemoval(),
            WorstRemoval()
        ]
        self.repair_operators = [
            GreedyInsertion(),
            RegretInsertion()
        ]
        self.operator_weights = [1.0] * len(self.destroy_operators)

    def run(self, max_iterations: int) -> Solution:
        """Execute ALNS optimization"""
        # Implementation details...

    def _select_operator(self, operator_type: str) -> Callable:
        """Adaptive operator selection based on performance"""
        # Implementation details...
```

#### **3. Solution Representation (`src/solution.py`)**
```python
class Solution:
    """Represents a complete VRP-IF solution"""

    def __init__(self):
        self.routes: List[Route] = []
        self.unassigned_customers: List[Location] = []
        self.total_cost: float = 0
        self.total_distance: float = 0
        self.total_time: float = 0

    def evaluate(self, problem: ProblemInstance):
        """Calculate total solution cost"""
        # Implementation details...
```

#### **4. Video Creation Integration (`simple_video_creator.py`)**
```python
class SimpleVideoCreator:
    """Creates animated optimization process videos"""

    def create_optimization_animation(self,
                                     optimization_history: List[Dict],
                                     customer_data: Dict,
                                     depot_location: Tuple[float, float],
                                     intermediate_facilities: List[Tuple[float, float]]) -> str:
        """Generate GIF showing route evolution"""
        # Dynamic scale calculation and animation generation...
```

---

## 🛠️ **Step-by-Step Implementation Guide**

### **Phase 1: Problem Setup and Data Generation**

**Step 1.1: Environment Setup**
```bash
# Clone and setup project
git clone <repository-url>
cd municipal-waste-collection-vrp

# Install dependencies
pip install -r requirements.txt

# Test installation
python main.py --help
```

**Step 1.2: Problem Instance Creation**
```python
from src.data_generator import DataGenerator

# Generate sample problem
problem = DataGenerator.generate_instance(
    name="Municipal Demo",
    n_customers=15,          # Number of collection points
    n_ifs=2,                # Intermediate facilities
    vehicle_capacity=25,     # Vehicle waste capacity
    area_size=100,          # Municipal area coverage
    demand_range=(2, 12),   # Waste generation per customer
    seed=42                 # Reproducible results
)
```

**Step 1.3: Distance Matrix Calculation**
```python
# Compute all pairwise distances
problem.calculate_distance_matrix()

# Validate symmetry and triangle inequality
# Results stored in problem.distance_matrix[i][j]
```

### **Phase 2: ALNS Optimization**

**Step 2.1: Algorithm Initialization**
```python
from src.alns import ALNS

# Initialize solver
solver = ALNS(problem)

# Configure parameters
solver.max_iterations = 1000
solver.temperature_initial = 10.0
solver.cooling_rate = 0.95

# Setup progress tracking
solver.iteration_callback = track_optimization_progress
```

**Step 2.2: Destruction Operators Implementation**

**Random Removal:**
```python
class RandomRemoval:
    def apply(self, solution: Solution, removal_count: int):
        """Remove random customers from solution"""
        customers_to_remove = random.sample(
            solution.served_customers,
            min(removal_count, len(solution.served_customers))
        )
        for customer in customers_to_remove:
            solution.remove_customer(customer)
        return customers_to_remove
```

**Related Removal:**
```python
class RelatedRemoval:
    def apply(self, solution: Solution, removal_count: int):
        """Remove geographically related customers"""
        # Find closest customer pairs
        # Remove clusters for neighborhood exploration
        pass
```

**Step 2.3: Repair Operators Implementation**

**Greedy Insertion:**
```python
class GreedyInsertion:
    def apply(self, solution: Solution, customers_to_insert: List[Location]):
        """Insert customers at best cost positions"""
        for customer in customers_to_insert:
            best_position = self._find_best_insertion_position(solution, customer)
            solution.insert_customer_at_position(customer, best_position)
```

**Step 2.4: Adaptive Weight Management**
```python
def update_operator_weights(self, operator_scores: List[float]):
    """Update operator selection probabilities"""
    for i, score in enumerate(operator_scores):
        self.operator_weights[i] = (
            (1 - self.adaptation_rate) * self.operator_weights[i] +
            self.adaptation_rate * score / sum(operator_scores)
        )
```

### **Phase 3: Video Creation and Visualization**

**Step 3.1: Optimization History Tracking**
```python
class OptimizationVideoTracker:
    def track_state(self, iteration: int, solution: Solution, cost: float):
        """Record optimization state for video creation"""
        route_coordinates = self._extract_route_coordinates(solution)

        state = {
            'iteration': iteration,
            'cost': cost,
            'best_cost': self.current_best_cost,
            'routes': route_coordinates
        }

        self.optimization_history.append(state)
```

**Step 3.2: Dynamic Scale Calculation**
```python
def _calculate_dynamic_bounds(self, points: List[Tuple[float, float]]):
    """Calculate plot bounds based on data extent"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Add 10% padding for better visualization
    padding_x = (max_x - min_x) * 0.1
    padding_y = (max_y - min_y) * 0.1

    return (min_x - padding_x, max_x + padding_x,
            min_y - padding_y, max_y + padding_y)
```

**Step 3.3: Animation Generation**
```python
def create_optimization_animation(self, history: List[Dict], ...):
    """Generate MP4/GIF of optimization process"""

    def animate_frame(frame_num):
        ax.clear()
        current_state = history[frame_num]

        # Plot all components with dynamic scaling
        self._plot_depot(ax, depot_location)
        self._plot_customers(ax, customer_data)
        self._plot_facilities(ax, intermediate_facilities)
        self._plot_routes(ax, current_state['routes'])
        self._add_metrics_text(ax, current_state)

    # Create matplotlib animation
    anim = FuncAnimation(fig, animate_frame, frames=len(history))
    return self._save_animation(anim, output_filename)
```

### **Phase 4: Performance Analysis and Validation**

**Step 4.1: Solution Quality Metrics**
```python
class PerformanceAnalyzer:
    def analyze_solution(self, solution: Solution, problem: ProblemInstance):
        """Comprehensive solution analysis"""

        analysis = {
            'total_cost': solution.total_cost,
            'total_distance': solution.total_distance,
            'total_time': solution.total_time,
            'num_vehicles': len(solution.routes),
            'capacity_utilization': self._calculate_capacity_utilization(solution),
            'distance_efficiency': self._calculate_distance_efficiency(solution),
            'unassigned_customers': len(solution.unassigned_customers)
        }

        return analysis
```

**Step 4.2: Route-Level Analysis**
```python
def analyze_route(self, route: Route, problem: ProblemInstance):
    """Detailed analysis of individual routes"""

    route_analysis = {
        'distance': route.total_distance,
        'time': route.total_time,
        'customers_served': len([n for n in route.nodes if n.type == 'customer']),
        'if_visits': len([n for n in route.nodes if n.type == 'if']),
        'max_load': max(route.load_profile) if route.load_profile else 0,
        'capacity_utilization': sum(route.demands) / problem.vehicle_capacity
    }

    return route_analysis
```

---

## 👥 **Author Contributions**

This project was developed with a 50-50 contribution split between the two authors. The workload was distributed to ensure that both authors contributed equally to the core algorithmic components, user-facing features, and overall project quality.

| Author | ID | Role | Key Responsibilities & Modules |
| :--- | :--- | :--- | :--- |
| **Chaitanya Shinde** | 231070066 | Core Algorithm and Application | `src/alns.py`, `src/problem.py`, `src/solution.py`, `src/destroy_operators.py`, `src/repair_operators.py`, `main.py`, `comprehensive_test_suite.py`, `test_runner.py` |
| **Harsh Sharma** | 231070064 | Data, Visualization, and Utilities | `src/data_generator.py`, `src/utils.py`, `simple_video_creator.py`, `optimization_video_creator.py`, `alns_video_integration.py`, `setup_and_demo.py`, `verify_project.py`, `src/benchmarking.py`, `src/enhanced_construction.py`, `src/enhanced_validator.py`, `tests/test_all.py`, `test_alns_functionality.py`, `test_video_integration.py` |

---

## 🚀 **Usage Guide**

### **Quick Start**

**Basic Usage:**
```bash
# Basic demonstration
python main.py --demo basic

# Comprehensive demonstration with analysis
python main.py --demo comprehensive

# Custom problem with specific parameters
python main.py --demo comprehensive --iterations 500 --vehicle-capacity 25
```

**Video Creation:**
```bash
# Create optimization videos
python main.py --demo comprehensive --video --iterations 300

# Live visualization with video recording
python main.py --live --video

# Benchmark with video creation
python main.py --demo benchmark --video
```

### **Command Line Interface**

| Option | Description | Example |
|--------|-------------|---------|
| `--demo {basic,comprehensive,benchmark}` | Run demonstration mode | `--demo comprehensive` |
| `--video` | Enable optimization video creation | `--video` |
| `--live` | Enable live plotting during optimization | `--live` |
| `--iterations N` | Number of ALNS iterations | `--iterations 500` |
| `--save-plots` | Save visualization plots to files | `--save-plots` |
| `--save-results` | Save results to JSON file | `--save-results` |
| `--instance FILE` | Load problem from JSON file | `--instance problem.json` |
| `--verbose` | Enable detailed output | `--verbose` |
| `--help` | Show help message | `--help` |

---

## 📊 **Performance and Benchmarks**

### **Test Results Summary**

| Test Category | Test Count | Pass Rate | Coverage |
|---------------|------------|-----------|----------|
| **Basic Functionality** | 4 | 100% | Import, Problem Creation, Data Generation |
| **Medium Problems** | 2 | 100% | 12-15 Customer Optimization |
| **Video Creation** | 3 | 100% | Video Generation, Tracking, Display |
| **Large Problems** | 2 | 100% | 25-50 Customer Scalability |
| **Integration Tests** | 1 | 100% | Full Workflow Validation |
| **Total** | **12** | **100%** | **Complete System Testing** |

---

##  deliverables-icon **Final Deliverables**

All final project deliverables are located in the `submissions/` directory. This includes:

- **`project_report.md`**: The main project report.
- **`dataset_documentation.md`**: Documentation for the data generation module.
- **`result_sheet.md`**: The detailed results of the test suite, including hyperparameter tuning and links to video outputs.
- **`hyperparameter_*_output.gif`**: Animated GIF outputs from the hyperparameter tuning tests.
- **Source Code**: The complete and documented source code in the `src/` directory and other Python files.

---

## 📁 **Project Structure**

```
.
└── ot-waste-collection-if
    ├── alns_video_integration.py
    ├── CHANGELOG.md
    ├── comprehensive_test_suite.py
    ├── ENHANCEMENT_COMPLETION.md
    ├── FINAL_IMPLEMENTATION_SUMMARY.md
    ├── main.py
    ├── optimization_video_creator.py
    ├── optimization_videos
    │   ├── alns_optimization_20251114_004115_cost.gif
    │   ├── alns_optimization_20251114_004115.gif
    │   ├── alns_optimization_20251114_005645_cost.gif
    │   ├── alns_optimization_20251114_005645.gif
    │   └── frames
    ├── __pycache__
    │   └── simple_video_creator.cpython-313.pyc
    ├── QUICK_REFERENCE.md
    ├── README_ARCH_LINUX.md
    ├── README.md
    ├── requirements.txt
    ├── setup_and_demo.py
    ├── setup_arch.sh
    ├── simple_video_creator.py
    ├── src
    │   ├── alns.py
    │   ├── benchmarking.py
    │   ├── data_generator.py
    │   ├── destroy_operators.py
    │   ├── enhanced_construction.py
    │   ├── enhanced_validator.py
    │   ├── __init__.py
    │   ├── problem.py
    │   ├── __pycache__
    │   ├── repair_operators.py
    │   ├── solution.py
    │   └── utils.py
    ├── test_alns_functionality.py
    ├── tests
    │   └── test_all.py
    ├── test_video_integration.py
    ├── venv
    │   ├── bin
    │   ├── include
    │   ├── lib
    │   ├── lib64 -> lib
    │   ├── pyvenv.cfg
    │   └── share
    ├── VERIFICATION_REPORT.md
    ├── VERIFICATION_SUMMARY.md
    ├── verify_project.py
    ├── VIDEO_GUIDE.md
    └── video_requirements.txt
```

---

## 📜 **License and Citation**

**Academic License:**
This project is developed for academic and research purposes.

**Citation:**
```
Municipal Waste Collection Route Optimization using ALNS
Chaitanya Shinde - 231070066 | Harsh Sharma - 231070064 | TYBTECH CE (2025)
Vehicle Routing Problem with Intermediate Facilities
```

---

*Last Updated: November 21, 2025*
*Version: 3.0 - Final Submission*
*© 2025 Municipal Waste Collection Route Optimization Project. All rights reserved . Harsh Sharma & Chaitanya Shinde*