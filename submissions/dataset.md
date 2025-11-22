# Dataset Description

The performance and robustness of the Adaptive Large Neighborhood Search (ALNS) algorithm for the waste collection Vehicle Routing Problem (VRP) are heavily dependent on the characteristics of the problem instances it is applied to. This project utilizes a flexible and configurable data generation module (`src/data_generator.py`) to create diverse datasets, allowing for comprehensive testing and evaluation.

## 1. Data Generation Process

The `src/data_generator.py` module is responsible for creating synthetic VRP instances. It allows for the specification of several key parameters that influence the complexity and nature of the generated problem:

*   **Number of Customers:** Determines the scale of the problem.
*   **Customer Distribution:**
    *   **Uniform:** Customer locations are randomly scattered across a defined geographical area (e.g., a square grid). This represents a scenario where waste generation points are evenly distributed.
    *   **Clustered:** Customer locations are grouped around several predefined cluster centers. This mimics real-world scenarios where waste collection points might be concentrated in residential areas, industrial zones, or specific neighborhoods.
*   **Customer Demands:** Each customer is assigned a demand (e.g., waste volume in cubic meters or tons) that the vehicles need to collect. Demands can be generated within a specified range.
*   **Number of Depots:** The number of central locations from which vehicles start and end their routes. For simplicity, most instances might use a single depot, but the generator can accommodate multiple.
*   **Vehicle Capacity:** The maximum load each vehicle can carry. This is a critical constraint that influences route planning.
*   **Vehicle Speed/Travel Costs:** Implied through the distance metric between locations. Travel costs are typically proportional to Euclidean distances between points.

### Example Data Generation Parameters:

The `data_generator.py` typically uses arguments or configurations similar to the following to define a problem instance:

*   `--num_customers`: (e.g., 50, 100, 200)
*   `--customer_distribution`: (e.g., `uniform`, `clustered`)
*   `--num_depots`: (e.g., 1)
*   `--vehicle_capacity`: (e.g., 100, 150)
*   `--demand_range`: (e.g., `(10, 25)`)
*   `--area_size`: (e.g., 100, for a 100x100 grid)
*   `--num_clusters` (if `clustered`): (e.g., 3, 5)

## 2. Dataset Characteristics

The generated datasets are characterized by:

*   **Node Coordinates:** Each customer and depot is defined by its (x, y) coordinates in a 2D plane.
*   **Customer Demands:** A positive integer representing the waste volume for each customer.
*   **Distance Matrix:** Implicitly calculated using the Euclidean distance between any two nodes. The cost function for the VRP typically sums these distances.
*   **Vehicle Fleet:** Defined by the number of available vehicles and their capacities.

These datasets, stored internally as `Problem` objects, provide a realistic and controlled environment for developing, testing, and benchmarking optimization algorithms for waste collection. By varying the generation parameters, it is possible to create instances ranging from small, simple problems to large, complex scenarios, thus allowing for thorough performance analysis of the ALNS algorithm.

## 3. Data Storage

While the `data_generator.py` creates `Problem` objects in memory, these can be serialized or used directly by the ALNS solver. The project does not rely on external, fixed datasets but rather on this internal generation capability to create tailored problem instances for each test run or experiment.