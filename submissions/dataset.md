# Dataset Documentation

**Authors:**

*   Chaitanya Shinde (231070066)
*   Harsh Sharma (231070064)

**Date:** November 21, 2025

---

## 1. Overview

This document describes the data generation process for the **Municipal Waste Collection Route Optimization** project. Since this project focuses on a metaheuristic solver, we do not use a fixed, pre-existing dataset. Instead, we have developed a flexible and powerful `DataGenerator` module (`src/data_generator.py`) that allows for the creation of a wide variety of synthetic problem instances.

This approach provides several advantages:

*   **Flexibility:** We can generate instances of any size and complexity to thoroughly test the algorithm.
*   **Reproducibility:** By using a fixed random seed, we can generate the exact same instance every time, which is crucial for benchmarking and debugging.
*   **Control:** We can control various parameters to test specific aspects of the algorithm (e.g., high-demand customers, clustered locations).

---

## 2. Data Generation Module

The `DataGenerator` class is the core of our data generation process. It provides a static method, `generate_instance`, that creates a `ProblemInstance` object with the following configurable parameters:

*   `name`: A string identifier for the instance.
*   `n_customers`: The number of customers to generate.
*   `n_ifs`: The number of intermediate facilities.
*   `vehicle_capacity`: The capacity of each vehicle.
*   `area_size`: The size of the square area in which locations are generated.
*   `demand_range`: A tuple specifying the minimum and maximum demand for each customer.
*   `seed`: An optional random seed for reproducibility.

### 2.1. Location Generation

Locations (customers, depots, and IFs) are generated within a square area. The `DataGenerator` supports two types of spatial distributions:

*   **Uniform Distribution:** Locations are spread randomly and uniformly across the entire area.
*   **Clustered Distribution:** Customers are grouped into a specified number of clusters. This is more representative of real-world scenarios where customers are often geographically clustered.

### 2.2. Example Instance

Here is an example of how to generate a problem instance:

```python
from src.data_generator import DataGenerator

# Generate a small, clustered instance
problem = DataGenerator.generate_instance(
    name="Small_Clustered_Example",
    n_customers=20,
    n_ifs=3,
    vehicle_capacity=50,
    area_size=100,
    demand_range=(5, 15),
    cluster_factor=0.5,  # Creates a moderate level of clustering
    seed=42
)
```

---

## 3. Benchmark Instances

For the purpose of evaluating the performance of our ALNS solver, we have defined a set of standard benchmark instances of varying sizes and complexities. These instances are used in our `comprehensive_test_suite.py` to ensure the solver is performing as expected.

The benchmark suite includes:

*   **Small Instances:** 10-20 customers, 1-2 IFs.
*   **Medium Instances:** 25-50 customers, 3-5 IFs.
*   **Large Instances:** 50-100 customers, 5-10 IFs.

These instances are generated with fixed seeds to ensure that the same problem is solved in every test run, allowing for fair comparisons of different algorithm configurations.

---

## 4. Data Format

The `ProblemInstance` object encapsulates all the data for a given problem. The key attributes are:

*   `depot`: A `Location` object representing the central depot.
*   `customers`: A list of `Location` objects representing the customers.
*   `intermediate_facilities`: A list of `Location` objects for the IFs.
*   `vehicle_capacity`: A float representing the capacity of the vehicles.
*   `distance_matrix`: A pre-computed matrix of distances between all locations.

Each `Location` object has the following attributes:

*   `id`: A unique integer identifier.
*   `x`, `y`: The coordinates of the location.
*   `demand`: The demand at the location (0 for depots and IFs).
*   `type`: A string indicating the type of location ('depot', 'customer', or 'if').
