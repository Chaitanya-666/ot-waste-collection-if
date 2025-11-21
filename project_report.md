# Project Report: Municipal Waste Collection Route Optimization

**Vehicle Routing Problem with Intermediate Facilities using Adaptive Large Neighborhood Search (ALNS)**

---

## **Authors**

- **Chaitanya Shinde** | 231070066
- **Harsh Sharma** | 231070064

*TYBTECH CE (2025)*

---

## **1. Abstract & Project Overview**

This project implements an advanced **Vehicle Routing Problem (VRP) with Intermediate Facilities** specifically designed for municipal waste collection optimization. The system uses **Adaptive Large Neighborhood Search (ALNS)** to find optimal waste collection routes that minimize total transportation costs while respecting vehicle capacity constraints and intermediate facility visits.

Municipal waste collection faces unique challenges, including limited vehicle capacity, multiple depots and facilities, and the need for cost and environmental optimization. Our solution addresses these challenges through intelligent route optimization that automatically determines optimal vehicle assignments, efficient visiting sequences, and strategic intermediate facility utilization. The system also provides real-time optimization tracking through animated visualizations.

---

## **2. Theoretical Foundation**

### **2.1. Vehicle Routing Problem (VRP) with Intermediate Facilities**

The VRP with Intermediate Facilities extends the classical VRP by introducing intermediate stops for unloading, which is critical for capacity-constrained vehicles like waste collection trucks.

**Core Problem Formulation:**
- **Given:** A set of customers with demands, a set of intermediate facilities, a central depot with a fleet of vehicles, vehicle capacity, and a distance matrix.
- **Objective:** Minimize the total distance traveled by all vehicles.
- **Constraints:**
  - Each customer must be visited exactly once.
  - Vehicle capacity cannot be exceeded.
  - All routes must start and end at the depot.

### **2.2. Adaptive Large Neighborhood Search (ALNS)**

ALNS is a metaheuristic optimization algorithm that iteratively destroys and repairs a solution to explore a large neighborhood in the search space.

- **Destruction Operators:** These operators remove a number of customers from the current solution. This project implements operators such as `RandomRemoval`, `WorstRemoval`, and `ShawRemoval` (related removal).
- **Repair Operators:** These operators re-insert the removed customers back into the solution to create a new, complete route. This project implements `GreedyInsertion` and `RegretInsertion`.
- **Adaptive Selection:** The key feature of ALNS is its ability to adaptively select which operators to use based on their past performance. Operators that have historically produced better solutions are chosen more frequently. This is managed through a weight adjustment mechanism.
- **Acceptance Criterion:** New solutions are accepted not only if they are better than the current best, but also sometimes if they are worse, based on a simulated annealing-like probability function. This helps the algorithm escape local optima.

---

## **3. Implementation Architecture**

The system is designed with a modular architecture, separating concerns into distinct layers for clarity and maintainability.

- **Main Application Layer:** Handles user interaction (CLI), video creation, and visualization.
- **Core ALNS Optimization Layer:** Contains the main ALNS engine and the destroy/repair operators.
- **Problem Definition Layer:** Includes the data structures for representing the problem (`ProblemInstance`), a `Solution`, and individual `Routes`.
- **Utility & Analysis Layer:** Provides tools for performance analysis, route visualization, and synthetic data generation.

---

## **4. Author Contributions**

This project was developed with a 50-50 contribution split between the two authors.

### **4.1. Chaitanya Shinde (231070066)**
**Role: Core Algorithm and Data Structures (50% of total workload)**

Chaitanya was responsible for designing and implementing the foundational logic of the VRP solver.

- **Key Modules:**
  - `src/problem.py`: Implemented `ProblemInstance` and `Location` classes.
  - `src/solution.py`: Implemented `Solution` and `Route` classes for solution representation.
  - `src/alns.py`: Implemented the core `ALNS` engine, including the main loop and adaptive weight adjustment.
  - `src/destroy_operators.py`: Implemented a suite of "destroy" operators.
  - `src/repair_operators.py`: Implemented a suite of "repair" operators.
  - `src/data_generator.py`: Developed the `DataGenerator` for creating synthetic problem instances.

### **4.2. Harsh Sharma (231070064)**
**Role: UI, Visualization, and Testing (50% of total workload)**

Harsh was responsible for all user-facing components, visualization tools, and the comprehensive testing framework.

- **Key Modules:**
  - `main.py`: Implemented the command-line interface (CLI) and demonstration modes.
  - `simple_video_creator.py`: Developed the `SimpleVideoCreator` for generating animated GIFs of the optimization process.
  - `src/utils.py`: Implemented the `RouteVisualizer` and `PerformanceAnalyzer`.
  - `tests/test_all.py` & `comprehensive_test_suite.py`: Developed the complete testing framework using `unittest`.
  - **Documentation:** Created and maintained the detailed `README.md` and overall project structure.

---

## **5. Conclusion**

This project provides a comprehensive and robust solution for the Vehicle Routing Problem with Intermediate Facilities, tailored for municipal waste collection. The use of an Adaptive Large Neighborhood Search algorithm allows for efficient and effective optimization, while the integrated visualization and video creation tools offer powerful insights into the algorithm's performance. The system is well-documented, thoroughly tested, and ready for practical application in municipal planning, academic research, and educational demonstrations.
