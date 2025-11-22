# Project Report: Optimized Waste Collection Route Planning using Adaptive Large Neighborhood Search (ALNS)

## 1. Problem Statement

The increasing urbanization and population growth present significant challenges in urban logistics, particularly in waste management. Efficient waste collection is crucial for public health, environmental sustainability, and operational cost reduction. This project addresses the Vehicle Routing Problem (VRP) specifically tailored for waste collection, which involves designing optimal routes for a fleet of waste collection vehicles to pick up waste from various customer locations and transport it to designated depots. The primary objectives are to minimize total travel distance, fuel consumption, and operational time, while respecting vehicle capacities, time windows (if applicable), and other operational constraints.

The complexity of this problem arises from the large number of potential customer locations, diverse waste volumes, and the need to efficiently manage a fleet of vehicles with varying capacities. Traditional exact methods often become computationally intractable for real-world instances, necessitating the use of heuristic and metaheuristic approaches.

## 2. Approach Used: Adaptive Large Neighborhood Search (ALNS)

This project employs the Adaptive Large Neighborhood Search (ALNS) metaheuristic to solve the waste collection VRP. ALNS is a powerful optimization framework that iteratively explores the solution space by intelligently destroying parts of the current solution and then repairing them. Its "adaptive" nature comes from dynamically adjusting the probabilities of selecting different destroy and repair operators based on their past performance, allowing the algorithm to learn which operators are most effective for a given problem instance.

### 2.1 Solution Representation

A solution to the waste collection VRP is represented as a collection of routes, where each route is an ordered sequence of customer visits assigned to a specific vehicle. Each customer has a demand (waste volume) that needs to be collected. Vehicles have a maximum capacity. The `Solution` class in `src/solution.py` encapsulates this structure, including methods for calculating the total cost (e.g., total distance) of a solution and validating its feasibility (e.g., capacity constraints).

### 2.2 Core ALNS Algorithm

The ALNS algorithm proceeds as follows:

1.  **Initial Solution Generation:** A feasible initial solution is constructed. The `src/enhanced_construction.py` module likely contains methods for creating such a starting point, possibly using greedy heuristics or other constructive algorithms.
2.  **Iterative Improvement:** The algorithm then enters an iterative loop:
    *   **Operator Selection:** A destroy operator and a repair operator are selected based on their adaptive weights, which are updated during the search.
    *   **Destroy Phase:** The selected destroy operator removes a subset of customers from the current solution, creating an incomplete (and often infeasible) solution. The `src/destroy_operators.py` module implements various strategies for this, such as:
        *   **Random Removal:** Randomly removes customers from routes.
        *   **Worst Removal:** Removes customers that contribute most to the solution cost or are "hardest" to serve.
        *   **Related Removal:** Removes customers that are geographically or temporally close to each other, aiming to create larger "holes" in the solution.
    *   **Repair Phase:** The selected repair operator reinserts the removed customers into the incomplete solution, aiming to create a new, feasible solution. The `src/repair_operators.py` module contains operators like:
        *   **Greedy Insertion:** Inserts customers into the cheapest available position.
        *   **Regret-k Insertion:** Considers the k-best insertion positions and chooses the one that minimizes the maximum regret (difference between the best and second-best insertion cost).
    *   **Acceptance Criterion:** The newly generated solution is evaluated. If it's better than the current best solution, it's accepted. Even if it's worse, it might be accepted with a certain probability (e.g., using a simulated annealing-like mechanism) to escape local optima.
    *   **Weight Adaptation:** The weights of the destroy and repair operators are updated based on their performance in generating good solutions. Operators that frequently lead to improvements receive higher weights, increasing their selection probability in future iterations.
3.  **Termination:** The algorithm terminates after a predefined number of iterations or when a certain quality threshold is met.

The `src/alns.py` module orchestrates this entire process, managing the main ALNS loop, operator selection, and weight updates.

### 2.3 Data Generation

The project includes a robust data generation module (`src/data_generator.py`) capable of creating diverse problem instances. This allows for thorough testing and benchmarking of the ALNS algorithm under various conditions. The generator supports:

*   **Customer Locations:** Generating customer locations either uniformly across a defined area or clustered around specific points, mimicking real-world population distributions.
*   **Customer Demands:** Assigning random waste demands to each customer.
*   **Depot Locations:** Defining one or more central depots from which vehicles start and end their routes.
*   **Vehicle Parameters:** Specifying the number of vehicles and their capacities.

This flexible data generation capability is crucial for evaluating the ALNS algorithm's performance and robustness.

## 3. Reference Papers

The implementation of the Adaptive Large Neighborhood Search algorithm in this project draws inspiration from foundational works in the field of metaheuristics and vehicle routing. Specifically, the following references provide theoretical background and practical insights:

*   **OT PROJECT DETAILS.pdf**
*   **OTminiProject.pdf**

These documents served as primary resources for understanding the problem context and guiding the architectural design of the ALNS solution framework.

## 4. Implementation Details (Codebase Overview)

The project's codebase is structured logically to separate concerns and facilitate maintainability. Key modules include:

*   **`main.py`**: The main entry point for running the optimization, handling command-line arguments, loading problem data, executing the ALNS solver, and potentially visualizing results.
*   **`src/problem.py`**: Defines the `Problem` class, which encapsulates all static information about a specific waste collection instance (e.g., customer locations, demands, depot locations, vehicle capacities).
*   **`src/solution.py`**: Defines the `Solution` class, representing a candidate solution to the VRP, including a list of routes and methods for calculating its cost and checking feasibility.
*   **`src/alns.py`**: Contains the core `ALNS` class that implements the adaptive large neighborhood search metaheuristic, managing the destroy and repair operators, acceptance criterion, and weight adaptation.
*   **`src/destroy_operators.py`**: Houses various destroy operators used by the ALNS algorithm, such as random removal, worst removal, and related removal.
*   **`src/repair_operators.py`**: Contains various repair operators, including greedy insertion and regret-k insertion strategies.
*   **`src/data_generator.py`**: Provides functionality to generate synthetic problem instances with configurable parameters for customer distribution (uniform, clustered), demands, and vehicle characteristics.
*   **`src/enhanced_construction.py`**: Likely contains methods for constructing an initial feasible solution, which is a crucial first step for the ALNS algorithm.
*   **`src/enhanced_validator.py`**: Provides robust validation logic to ensure that generated or modified solutions adhere to all problem constraints.
*   **`scripts/optimization_video_creator.py`**: A utility script for generating visual representations (e.g., GIFs) of the ALNS optimization process, showing how routes evolve over iterations.
*   **`comprehensive_test_suite.py`**: A script for running various tests and benchmarks to evaluate the performance and correctness of the ALNS implementation.

This modular design promotes reusability and allows for easy experimentation with different operators, acceptance criteria, and problem configurations.