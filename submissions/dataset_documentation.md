# Dataset Generation Documentation

---

## **1. Overview**

The `src/data_generator.py` module is a crucial component of this project, responsible for creating synthetic problem instances for the Vehicle Routing Problem. Since real-world municipal waste collection datasets are often proprietary or difficult to obtain, this generator allows for the creation of diverse and reproducible scenarios for testing, benchmarking, and demonstration purposes.

The primary goal of the data generator is to produce `ProblemInstance` objects that are realistic and varied, enabling robust evaluation of the ALNS algorithm's performance under different conditions.

## **2. Integration in the Project**

The `DataGenerator` is used in several key areas of the project:

- **Demonstrations (`main.py`):** Both the `basic` and `comprehensive` demonstration modes use the `DataGenerator` to create problem instances on the fly.
- **Testing (`comprehensive_test_suite.py`):** The test suite relies heavily on the `DataGenerator` to create instances of varying sizes and characteristics (small, medium, large, edge cases) to ensure the solver is robust.
- **Benchmarking:** The benchmark mode programmatically generates a suite of problems to measure the solver's performance and scalability.

## **3. Core Functionality: `generate_instance`**

The main function in this module is `DataGenerator.generate_instance()`. This static method creates a complete `ProblemInstance` object based on a set of specified parameters.

### **Method Signature**
```python
def generate_instance(
    name: str,
    n_customers: int,
    n_ifs: int,
    vehicle_capacity: int = 20,
    area_size: int = 100,
    demand_range: Tuple[int, int] = (1, 10),
    service_time_range: Tuple[int, int] = (1, 5),
    seed: Optional[int] = None,
    cluster_factor: float = 0.3,
    depot_position: str = "center",
) -> ProblemInstance
```

### **Parameters**

- **`name` (str):** The name for the problem instance (e.g., "Municipal Demo").
- **`n_customers` (int):** The number of customer locations (collection points) to generate.
- **`n_ifs` (int):** The number of intermediate facilities (disposal sites) to generate.
- **`vehicle_capacity` (int):** The maximum capacity of each vehicle. *Default: 20*.
- **`area_size` (int):** The size of the square area in which the locations will be generated (e.g., a value of 100 creates a 100x100 grid). *Default: 100*.
- **`demand_range` (Tuple[int, int]):** The minimum and maximum range for the demand of each customer. A demand value will be randomly chosen from this range for each customer. *Default: (1, 10)*.
- **`service_time_range` (Tuple[int, int]):** The minimum and maximum range for the time required to service a customer. *Default: (1, 5)*.
- **`seed` (Optional[int]):** A random seed for reproducibility. If a seed is provided, the exact same problem instance will be generated every time the function is called with the same parameters. *Default: None*.
- **`cluster_factor` (float):** A value between 0.0 and 1.0 that controls the geographical distribution of customers.
  - `0.0`: Customers are distributed uniformly across the entire area.
  - `1.0`: Customers are tightly grouped into a small number of dense clusters.
  - Values in between create varying degrees of clustering. *Default: 0.3*.
- **`depot_position` (str):** Determines the placement of the main depot. Can be one of `'center'`, `'corner'`, or `'random'`. *Default: "center"*.

## **4. How It Works**

1.  **Initialization:** A new `ProblemInstance` is created. A random seed is set if provided.
2.  **Depot Generation:** The depot `Location` is created based on the `depot_position` parameter.
3.  **Intermediate Facility Generation:** The specified number of intermediate facilities are generated. Their placement is influenced by the `cluster_factor`—in clustered scenarios, they are placed nearer to the depot.
4.  **Customer Generation:**
    - The generator first determines a number of cluster centers based on the `cluster_factor`.
    - It then distributes the customer locations around these centers using a Gaussian (normal) distribution. If `cluster_factor` is 0, customers are spread uniformly.
    - For each customer, a random demand and service time are assigned based on the specified ranges.
5.  **Distance Matrix Calculation:** Finally, the `calculate_distance_matrix()` method of the `ProblemInstance` is called to pre-compute the Euclidean distances between all pairs of locations (depot, customers, and facilities). This is done to speed up the solving process, as these distances are used repeatedly by the ALNS algorithm.

## **5. Example Usage**

Here is a programmatic example of how to use the `DataGenerator` to create a problem instance:

```python
from src.data_generator import DataGenerator

# Generate a medium-sized problem instance with clustered customers
problem = DataGenerator.generate_instance(
    name="Medium Clustered Demo",
    n_customers=25,
    n_ifs=3,
    vehicle_capacity=30,
    area_size=150,
    demand_range=(5, 15),
    seed=2025,
    cluster_factor=0.7,
    depot_position="corner"
)

# The 'problem' object is now a fully initialized ProblemInstance
# and can be passed directly to the ALNS solver.
print(f"Successfully generated problem: '{problem.name}'")
print(f" - Customers: {len(problem.customers)}")
print(f" - Intermediate Facilities: {len(problem.intermediate_facilities)}")
print(f" - Depot at: ({problem.depot.x}, {problem.depot.y})")
```
