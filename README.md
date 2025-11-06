# Optimization of Municipal Solid Waste Collection Routes using Adaptive Large Neighborhood Search

**Course:** Optimization Techniques (OT)
**Team:** Harsh Sharma (231070064) & Chaitanya Shinde (231070066)
**Institution:** T.Y. B.Tech (Computer Engineering)
**Academic Year:** 2024-2025

## 🎯 Project Title
**Municipal Waste Collection Modelling with Intermediate Facilities using Adaptive Large Neighborhood Search (ALNS)**

## 📖 Executive Summary

This project addresses the critical urban planning challenge of optimizing municipal solid waste collection by formulating it as a **Vehicle Routing Problem with Intermediate Facilities (VRP-IF)**. We implement an **Adaptive Large Neighborhood Search (ALNS)** metaheuristic to compute near-optimal waste collection routes that minimize total travel distance while respecting vehicle capacity constraints and the crucial requirement for periodic visits to disposal facilities.

## 🔍 Detailed Problem Statement

### Real-World Waste Collection Challenge
Municipal waste collection vehicles face a unique operational constraint: they must periodically interrupt their collection routes to visit intermediate facilities (landfills, transfer stations, or recycling centers) when they reach capacity. This creates complex routing patterns that cannot be adequately addressed by classical VRP formulations.

### Problem Characteristics

**Core Constraints:**
- **Vehicle Capacity:** Each collection vehicle has a maximum waste carrying capacity
- **Intermediate Facilities (IFs):** Vehicles must visit disposal sites when full to empty their load
- **Demand Satisfaction:** All customer locations must be serviced exactly once
- **Depot Operations:** All routes must start and end at the central depot

**Operational Complexities:**
- **Strategic IF Timing:** Determining optimal points to interrupt collection for disposal
- **Load Management:** Balancing vehicle utilization with disposal frequency
- **Route Segmentation:** Collection routes are divided into segments between IF visits
- **Multiple Decision Layers:** Simultaneous optimization of customer sequencing and IF scheduling

## 🧠 Mathematical Formulation

### VRP-IF as Mixed Integer Linear Program (MILP)

**Sets:**
- \( V = \{0, 1, \dots, n\} \): Vertices (0 = depot, 1-m = customers, m+1-n = IFs)
- \( K \): Set of vehicles

**Parameters:**
- \( d_i \): Demand at customer \( i \)
- \( Q \): Vehicle capacity
- \( c_{ij} \): Travel cost from \( i \) to \( j \)
- \( M \): Large positive number

**Decision Variables:**
- \( x_{ijk} = 1 \) if vehicle \( k \) travels from \( i \) to \( j \), 0 otherwise
- \( u_{ik} \): Cumulative load of vehicle \( k \) after visiting vertex \( i \)

**Objective Function:**
\[
\min \sum_{k \in K} \sum_{i \in V} \sum_{j \in V} c_{ij} \cdot x_{ijk}
\]

**Key Constraints:**
1. **Customer Service:** Each customer served exactly once
2. **Flow Conservation:** Vehicle continuity through routes
3. **Capacity Management:** Load tracking and IF reset mechanisms
4. **IF Visits:** Mandatory disposal when capacity reached

## ⚙️ Algorithmic Approach: Adaptive Large Neighborhood Search

### Why ALNS for Waste Collection?

ALNS is particularly suited for VRP-IF due to its:
- **Adaptive Nature:** Dynamically adjusts to problem structure
- **Flexible Operators:** Handles complex constraints through specialized destroy/repair mechanisms
- **Proven Effectiveness:** State-of-the-art performance on complex VRPs
- **Scalability:** Efficiently handles realistic problem sizes

### ALNS Framework Implementation

```python
class ALNS_WasteCollection:
    def __init__(self, problem_instance):
        self.problem = problem_instance
        self.current_solution = None
        self.best_solution = None
        self.operator_weights = initialize_weights()

    def solve(self, iterations=1000):
        self.initialize_solution()

        for iteration in range(iterations):
            # Adaptive operator selection
            destroy_op, repair_op = self.select_operators()

            # Destroy phase: Remove parts of current solution
            partial_solution = destroy_op.apply(self.current_solution)

            # Repair phase: Reconstruct complete solution
            new_solution = repair_op.apply(partial_solution)

            # Acceptance decision
            if self.accept_solution(new_solution):
                self.current_solution = new_solution

            # Update best solution
            if new_solution.cost < self.best_solution.cost:
                self.best_solution = new_solution.copy()

            # Adaptive weight adjustment
            self.update_operator_weights(destroy_op, repair_op, new_solution)
```

### Specialized Destroy Operators

**1. Random Customer Removal**
```python
def random_removal(solution, removal_count):
    # Randomly select customers to remove
    removed_customers = random.sample(solution.customers, removal_count)
    return solution.remove_customers(removed_customers)
```

**2. Worst-Cost Removal**
```python
def worst_removal(solution, removal_count):
    # Remove customers with highest marginal cost
    customers_by_cost = sorted(solution.customers,
                              key=lambda c: solution.cost_contribution(c),
                              reverse=True)
    return solution.remove_customers(customers_by_cost[:removal_count])
```

**3. Shaw Removal (Similarity-Based)**
```python
def shaw_removal(solution, removal_count):
    # Remove similar customers (proximity + demand similarity)
    seed_customer = random.choice(solution.customers)
    similar_customers = find_similar_customers(seed_customer, solution)
    return solution.remove_customers(similar_customers[:removal_count])
```

**4. Route-Based Removal**
```python
def route_removal(solution, removal_count):
    # Remove entire segments between IF visits
    route_segments = solution.identify_if_segments()
    removed_segment = random.choice(route_segments)
    return solution.remove_segment(removed_segment)
```

### Intelligent Repair Operators

**1. Greedy Insertion**
```python
def greedy_insertion(partial_solution):
    while partial_solution.has_unserved_customers():
        best_customer, best_position, best_cost = None, None, float('inf')

        for customer in partial_solution.unserved_customers:
            for position in partial_solution.feasible_positions(customer):
                insertion_cost = partial_solution.cost_if_inserted(customer, position)
                if insertion_cost < best_cost:
                    best_customer, best_position, best_cost = customer, position, insertion_cost

        partial_solution.insert_customer(best_customer, best_position)
```

**2. Regret Insertion**
```python
def regret_insertion(partial_solution, k_regret=2):
    while partial_solution.has_unserved_customers():
        best_customer = None
        max_regret = -float('inf')

        for customer in partial_solution.unserved_customers:
            insertion_costs = partial_solution.k_best_insertion_costs(customer, k_regret)
            regret = sum(insertion_costs[1:]) - insertion_costs[0] * (k_regret - 1)

            if regret > max_regret:
                max_regret = regret
                best_customer = customer

        best_position = partial_solution.best_insertion_position(best_customer)
        partial_solution.insert_customer(best_customer, best_position)
```

**3. IF-Aware Repair**
```python
def if_aware_repair(partial_solution):
    # Specialized repair considering IF constraints
    for customer in partial_solution.unserved_customers:
        feasible_routes = partial_solution.find_feasible_routes(customer)
        if not feasible_routes:
            # Create new route or insert IF visit
            partial_solution = insert_strategic_if_visit(partial_solution, customer)
        else:
            best_route = min(feasible_routes, key=lambda r: r.cost_increase(customer))
            best_route.insert_customer(customer)
```

### Adaptive Mechanism

**Weight Update Strategy:**
```python
def update_operator_weights(self, destroy_op, repair_op, new_solution):
    score = self.calculate_operator_score(new_solution)

    # Update based on performance
    destroy_op.weight = destroy_op.weight * (1 - learning_rate) + score * learning_rate
    repair_op.weight = repair_op.weight * (1 - learning_rate) + score * learning_rate

    # Normalize weights
    self.normalize_operator_weights()
```

### Acceptance Criteria

**Simulated Annealing Acceptance:**
```python
def accept_solution(self, new_solution):
    cost_difference = new_solution.cost - self.current_solution.cost

    if cost_difference < 0:
        return True  # Always accept improving moves
    else:
        # Probabilistic acceptance of worse solutions
        acceptance_probability = math.exp(-cost_difference / self.temperature)
        return random.random() < acceptance_probability
```

## 💻 Implementation Architecture

### Core Data Structures

**Problem Instance:**
```python
class WasteCollectionProblem:
    def __init__(self):
        self.depot = None
        self.customers = []  # List of Customer objects
        self.intermediate_facilities = []  # List of IF objects
        self.vehicle_capacity = 0
        self.distance_matrix = None
```

**Solution Representation:**
```python
class Solution:
    def __init__(self):
        self.routes = []  # List of Route objects
        self.total_cost = 0
        self.unserved_customers = set()

    class Route:
        def __init__(self):
            self.sequence = []  # [depot, cust1, cust2, IF, cust3, ..., depot]
            self.load_profile = []  # Cumulative load at each node
            self.cost = 0
```

### Project Structure
```
ot-waste-collection-if/
├── src/
│   ├── core/
│   │   ├── problem.py          # Problem definition
│   │   ├── solution.py         # Solution representation
│   │   └── validator.py        # Feasibility checking
│   ├── algorithms/
│   │   ├── alns/               # ALNS implementation
│   │   │   ├── alns_engine.py
│   │   │   ├── destroy_operators.py
│   │   │   ├── repair_operators.py
│   │   │   └── adaptive_mechanism.py
│   │   └── construction/       # Initial solution heuristics
│   ├── utils/
│   │   ├── visualization.py    # Route plotting
│   │   ├── metrics.py          # Performance tracking
│   │   └── data_generator.py   # Synthetic instances
├── data/
├── tests/
├── docs/
└── main.py
```

## 📊 Experimental Framework

### Evaluation Metrics

**Primary Objectives:**
- **Total Travel Distance** (main optimization criterion)
- **Computational Efficiency** (time to reach satisfactory solutions)
- **Solution Quality** (gap from optimal/best-known solutions)

**Secondary Metrics:**
- Vehicle utilization rates
- Number of IF visits per route
- Load factor optimization
- Algorithm convergence behavior

### Benchmarking Strategy

**Comparison Against:**
1. **Genetic Algorithm** (baseline metaheuristic)
2. **Construction Heuristics** (Nearest Neighbor, Savings Algorithm)
3. **Manual Planning** (current practice simulation)

## 🚀 Installation & Execution

### Quick Start
```bash
# Create environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run ALNS on sample instance
python main.py --algorithm alns --instance data/sample_instance.json --iterations 1000
```

### Configuration Options
```python
# ALNS Parameters
config = {
    'iterations': 1000,
    'destroy_degree': 0.15,  # Percentage of customers to remove
    'temperature': 1000,     # Initial SA temperature
    'cooling_rate': 0.9995,  # Temperature reduction factor
    'adaptive_period': 100,  # Weight update frequency
    'seed': 42              # Random seed for reproducibility
}
```

## 📈 Expected Results & Analysis

### Performance Projections

**Solution Quality:**
- **ALNS:** Expected 15-30% improvement over construction heuristics
- **Consistency:** Robust performance across diverse problem instances
- **Scalability:** Effective handling of 50-200 customer problems

**Computational Performance:**
- **Convergence:** Rapid initial improvement with refined search
- **Stability:** Consistent results across multiple runs
- **Efficiency:** Reasonable computation times for practical application

### Comparative Advantages

**ALNS Strengths:**
- ✅ Superior solution quality through adaptive search
- ✅ Effective handling of complex VRP-IF constraints
- ✅ Excellent scalability to realistic problem sizes
- ✅ Proven performance in academic literature

**Implementation Challenges:**
- ⚠️ Complex parameter tuning requirements
- ⚠️ Higher computational overhead than simple heuristics
- ⚠️ Implementation complexity of specialized operators

## 🔮 Future Enhancements

### Immediate Extensions
1. **Multiple Intermediate Facilities** with different characteristics
2. **Time Window Constraints** for customer service
3. **Heterogeneous Fleet** with varying capacities
4. **Dynamic Demand** considerations

### Advanced Research Directions
1. **Real-time Adaptive Routing** based on traffic conditions
2. **Multi-objective Optimization** balancing cost and environmental impact
3. **Machine Learning Integration** for demand prediction
4. **GIS Integration** with real-world spatial data

## 📋 Project Deliverables

### Core Components
1. **Complete ALNS Implementation** with specialized VRP-IF operators
2. **Comprehensive Testing Suite** validating algorithm performance
3. **Visualization Tools** for route analysis and convergence tracking
4. **Documentation** including user guide and technical specification

### Evaluation Artifacts
- Performance comparison against alternative algorithms
- Sensitivity analysis of key parameters
- Scalability assessment on larger instances
- Practical applicability assessment

## 👥 Team Contributions

### Harsh Sharma (231070064)
- ALNS algorithm design and implementation
- Mathematical modeling and optimization core
- Performance analysis and benchmarking
- Technical documentation

### Chaitanya Shinde (231070066)
- System architecture and data structures
- Visualization and user interface components
- Testing framework and validation
- Experimental setup and execution

---

*This project represents a comprehensive implementation of Adaptive Large Neighborhood Search for solving the complex Vehicle Routing Problem with Intermediate Facilities in municipal waste collection contexts. The approach combines state-of-the-art metaheuristic techniques with practical considerations for real-world waste management operations.*
