# Project Report: Municipal Waste Collection Route Optimization

**Authors:**

*   Chaitanya Shinde (231070066)
*   Harsh Sharma (231070064)

**Date:** November 21, 2025

---

## 1. Abstract

This report details the design and implementation of a sophisticated solver for the **Vehicle Routing Problem with Intermediate Facilities (VRP-IF)**, tailored for municipal waste collection. The primary objective is to minimize total transportation costs by optimizing waste collection routes. This is achieved through an **Adaptive Large Neighborhood Search (ALNS)** metaheuristic, which intelligently explores the solution space to find high-quality routes. The system is designed to be robust, scalable, and extensible, providing a powerful tool for logistics and operational research.

---

## 2. Problem Statement

The **Municipal Waste Collection Problem** is a complex variant of the Vehicle Routing Problem (VRP). It involves a fleet of capacitated vehicles tasked with collecting waste from a set of customers and transporting it to a central depot. The problem is further complicated by the need for vehicles to visit **intermediate facilities (IFs)** to unload collected waste before continuing their routes.

The key challenges are:

*   **Capacitated Vehicles:** Each vehicle has a limited capacity, which, once reached, requires a trip to an IF.
*   **Intermediate Facilities:** The model must decide when and which IF to visit to minimize detours and continue collection efficiently.
*   **Route Optimization:** The system must determine the optimal sequence of customer visits for each vehicle to minimize the total distance traveled.
*   **Scalability:** The solution must be efficient enough to handle real-world scenarios with a large number of customers and multiple vehicles.

The goal is to find a set of routes that services all customers while respecting all constraints and minimizing the total travel distance.

---

## 3. The ALNS Approach

To solve this NP-hard problem, we implemented an **Adaptive Large Neighborhood Search (ALNS)** algorithm. ALNS is a powerful metaheuristic that iteratively refines a solution by repeatedly "destroying" and "repairing" it.

### 3.1. The ALNS Framework

The core of our solver is the `ALNS` class, which orchestrates the search process. The algorithm proceeds as follows:

1.  **Initialization:** A feasible initial solution is generated using a greedy heuristic.
2.  **Iteration:** The algorithm enters a loop that continues for a fixed number of iterations.
3.  **Operator Selection:** In each iteration, a *destroy* and a *repair* operator are selected based on adaptive weights. These weights are updated based on the historical performance of the operators.
4.  **Destruction:** The selected destroy operator removes a fraction of customers from the current solution, creating a partial solution.
5.  **Repair:** The selected repair operator re-inserts the removed customers into the partial solution, creating a new candidate solution.
6.  **Acceptance:** The candidate solution is accepted based on a simulated annealing criterion. This allows the algorithm to accept worse solutions with a certain probability, helping it to escape local optima.
7.  **Weight Update:** The weights of the selected operators are updated based on the quality of the new solution.

### 3.2. Destroy Operators

Destroy operators are responsible for breaking apart the current solution. We implemented a variety of operators to ensure a diverse search:

*   **Random Removal:** Removes a random set of customers. This encourages diversification.
*   **Worst Removal:** Removes customers that are the most "expensive" to serve, based on the cost savings of removing them. This helps to improve the solution by targeting inefficient parts.
*   **Shaw Removal:** Removes customers that are "similar" based on a combination of distance and demand. This is effective for clustered problems.
*   **Route Removal:** Removes an entire route, allowing for large-scale restructuring of the solution.

### 3.3. Repair Operators

Repair operators rebuild the partial solution created by the destroy operators:

*   **Greedy Insertion:** Inserts each unassigned customer into the position that results in the smallest increase in cost.
*   **Regret Insertion:** A more sophisticated heuristic that prioritizes customers with the fewest good insertion options. It calculates a "regret" value for each customer and inserts the one with the highest regret first.
*   **IF-Aware Repair:** A specialized greedy insertion that explicitly considers the need for IF visits, ensuring that capacity constraints are always met.

---

## 4. System Architecture

The project is structured into several distinct layers, promoting modularity and separation of concerns:

*   **Problem Definition Layer (`src/problem.py`, `src/solution.py`):** Defines the core data structures, including `ProblemInstance`, `Location`, and `Solution`. This layer is responsible for representing the problem and its constraints.
*   **ALNS Optimization Layer (`src/alns.py`, `src/destroy_operators.py`, `src/repair_operators.py`):** Contains the core logic of the ALNS algorithm, including the main search loop and the implementation of all destroy and repair operators.
*   **Utility and Analysis Layer (`src/utils.py`, `src/benchmarking.py`):** Provides helper functions for logging, performance analysis, data serialization, and visualization.
*   **Application Layer (`main.py`, `comprehensive_test_suite.py`):** The main entry point for the application, providing a command-line interface for running the solver, tests, and generating reports.

---

## 5. Conclusion

This project successfully implements a robust and efficient solver for the Vehicle Routing Problem with Intermediate Facilities using an Adaptive Large Neighborhood Search algorithm. The modular architecture allows for easy extension and experimentation with new heuristics. The inclusion of detailed documentation, comprehensive tests, and visualization tools makes this project a valuable resource for both academic and practical applications in the field of logistics and operations research. The results demonstrate that the ALNS approach is highly effective at finding near-optimal solutions to this complex routing problem.
