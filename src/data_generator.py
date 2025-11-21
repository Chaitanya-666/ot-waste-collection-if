"""
Data Generation for Vehicle Routing Problem with Intermediate Facilities (VRP-IF)
==============================================================================

Author: Harsh Sharma (231070064) - Data generation and instance management

This module provides comprehensive data generation capabilities for the VRP-IF,
including creation of synthetic problem instances, loading/saving instances,
and generating test cases for benchmarking and validation.

Key Features:
- Generate synthetic problem instances with configurable parameters
- Create clustered or uniformly distributed customer locations
- Save and load problem instances to/from JSON files
- Generate benchmark suites and edge cases for testing
- Support for reproducible random generation with seed values
"""

import json
import random
import math
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path

from .problem import ProblemInstance, Location


class DataGenerator:
    """
    A utility class for generating synthetic VRP-IF problem instances.
    
    This class provides static methods to create problem instances with various
    characteristics, including different customer distributions, facility locations,
    and problem constraints. It supports both programmatic generation and
    configuration-based instance creation.
    
    The generator can create:
    - Random instances with uniform or clustered customer distributions
    - Benchmark instances of varying sizes and complexities
    - Edge cases for testing algorithm robustness
    - Instances from configuration files for reproducible testing
    
    Author: Harsh Sharma (231070064)
    """

    @staticmethod
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
    ) -> ProblemInstance:
        """
        Generate a complete, synthetic VRP-IF problem instance.
        
        This method creates a problem instance with the specified number of customers
        and intermediate facilities, distributed according to the given parameters.
        The instance includes all necessary attributes for solving the VRP-IF.

        Args:
            name: A descriptive name for the problem instance.
            n_customers: Number of customer locations to generate (≥ 0).
            n_ifs: Number of intermediate facilities to generate (≥ 1).
            vehicle_capacity: Maximum capacity of each vehicle (default: 20).
            area_size: Size of the square area for locations (default: 100).
            demand_range: Tuple of (min, max) demand for customers (default: (1, 10)).
            service_time_range: Tuple of (min, max) service times (default: (1, 5)).
            seed: Random seed for reproducibility (default: None).
            cluster_factor: Controls customer clustering (0.0 = uniform, 1.0 = highly clustered).
            depot_position: Position of depot ('center', 'corner', or 'random').

        Returns:
            ProblemInstance: A fully initialized problem instance.
            
        Raises:
            ValueError: If input parameters are invalid.
            
        Example:
            >>> instance = DataGenerator.generate_instance(
            ...     name="Test1",
            ...     n_customers=10,
            ...     n_ifs=2,
            ...     vehicle_capacity=25,
            ...     seed=42
            ... )
            >>> len(instance.customers)
            10
            
        Note:
            - Customers are assigned random demands within the specified range
            - Service times are randomly assigned within the given range
            - The depot is always assigned ID 0
            - Customer IDs start from 1
            - IF IDs start from 1000
            
        Author: Harsh Sharma (231070064)
        """
        if seed is not None:
            random.seed(seed)

        problem = ProblemInstance(name)
        problem.vehicle_capacity = vehicle_capacity
        problem.number_of_vehicles = max(2, n_customers // 5)  # Heuristic for vehicle count
        problem.disposal_time = 2

        # Generate depot location based on the specified position.
        if depot_position == "center":
            depot = Location(0, area_size // 2, area_size // 2, 0, "depot")
        elif depot_position == "corner":
            depot = Location(0, 5, 5, 0, "depot")
        elif depot_position == "random":
            depot = Location(
                0,
                random.randint(10, area_size - 10),
                random.randint(10, area_size - 10),
                0,
                "depot",
            )
        else:
            depot = Location(0, area_size // 2, area_size // 2, 0, "depot")

        problem.depot = depot

        # Generate intermediate facilities (IFs).
        for i in range(n_ifs):
            if cluster_factor > 0:
                # Place IFs closer to the depot in clustered scenarios.
                base_x, base_y = depot.x, depot.y
                x = base_x + random.gauss(0, area_size * cluster_factor * 0.3)
                y = base_y + random.gauss(0, area_size * cluster_factor * 0.3)
            else:
                # Distribute IFs uniformly across the area.
                x = random.randint(10, area_size - 10)
                y = random.randint(10, area_size - 10)

            # Ensure IFs are within the defined area bounds.
            x = max(5, min(area_size - 5, x))
            y = max(5, min(area_size - 5, y))

            if_node = Location(i + 1000, int(x), int(y), 0, "if")
            problem.intermediate_facilities.append(if_node)

        # Generate customer locations, potentially in clusters.
        customer_clusters = DataGenerator._generate_clusters(
            n_customers, area_size, cluster_factor, seed
        )

        for i, cluster_center in enumerate(customer_clusters):
            x, y = cluster_center
            demand = random.randint(*demand_range)
            service_time = random.randint(*service_time_range)

            customer = Location(i + 1, int(x), int(y), demand, "customer")
            customer.service_time = service_time
            problem.customers.append(customer)

        # Pre-calculate the distance matrix to speed up the solver.
        try:
            problem.calculate_distance_matrix()
        except Exception:
            pass

        return problem

    @staticmethod
    def _generate_clusters(
        n_points: int, area_size: int, cluster_factor: float, seed: Optional[int]
    ) -> List[Tuple[float, float]]:
        """
        Generate coordinates for points with optional clustering.
        
        This internal method creates point coordinates that are either:
        - Uniformly distributed (when cluster_factor = 0)
        - Clustered around random centers (when cluster_factor > 0)
        
        Args:
            n_points: Number of points to generate.
            area_size: Size of the square area.
            cluster_factor: Controls clustering (0.0 to 1.0).
            seed: Random seed for reproducibility.
            
        Returns:
            List of (x, y) coordinate tuples.
            
        Note:
            - Higher cluster_factor creates more tightly grouped points
            - Points are constrained to be within the area bounds
            
        Author: Harsh Sharma (231070064)
        """
        if seed is not None:
            random.seed(seed)

        if cluster_factor <= 0:
            # Uniform distribution if no clustering is specified.
            return [
                (random.randint(10, area_size - 10), random.randint(10, area_size - 10))
                for _ in range(n_points)
            ]

        # Generate a number of cluster centers.
        n_clusters = max(1, int(n_points * cluster_factor * 0.1))
        cluster_centers = []

        for _ in range(n_clusters):
            center_x = random.randint(20, area_size - 20)
            center_y = random.randint(20, area_size - 20)
            cluster_centers.append((center_x, center_y))

        # Distribute customer points around the generated cluster centers.
        points = []
        for i in range(n_points):
            if cluster_centers:
                center_x, center_y = random.choice(cluster_centers)
                # Use a Gaussian distribution to place points around the center.
                x = center_x + random.gauss(0, area_size * cluster_factor * 0.2)
                y = center_y + random.gauss(0, area_size * cluster_factor * 0.2)
            else:
                x = random.randint(10, area_size - 10)
                y = random.randint(10, area_size - 10)

            # Ensure points are within bounds.
            x = max(5, min(area_size - 5, x))
            y = max(5, min(area_size - 5, y))

            points.append((x, y))

        return points

    @staticmethod
    def generate_instances_from_file(filename: Union[str, Path]) -> List[ProblemInstance]:
        """
        Generate problem instances from a JSON configuration file.
        
        The configuration file should be a JSON file containing a list of instance
        definitions, each with parameters matching the `generate_instance` method.
        
        Args:
            filename: Path to the JSON configuration file.
            
        Returns:
            List of generated ProblemInstance objects.
            
        Example config.json:
            {
                "instances": [
                    {
                        "name": "Small Instance",
                        "customers": 10,
                        "ifs": 2,
                        "capacity": 20,
                        "area_size": 100,
                        "demand_range": [1, 10],
                        "service_time_range": [1, 5],
                        "cluster_factor": 0.3,
                        "depot_position": "center",
                        "seed": 42
                    },
                    ...
                ]
            }
            
        Note:
            - If the file is not found, returns a list with a single default instance
            - All parameters except 'name' are optional and have default values
            
        Author: Harsh Sharma (231070064)
        """
        instances = []

        try:
            with open(filename, "r") as f:
                config = json.load(f)

            for instance_config in config.get("instances", []):
                instance = DataGenerator.generate_instance(
                    name=instance_config["name"],
                    n_customers=instance_config["customers"],
                    n_ifs=instance_config["ifs"],
                    vehicle_capacity=instance_config.get("capacity", 20),
                    area_size=instance_config.get("area_size", 100),
                    demand_range=tuple(instance_config.get("demand_range", [1, 10])),
                    service_time_range=tuple(
                        instance_config.get("service_time_range", [1, 5])
                    ),
                    seed=instance_config.get("seed"),
                    cluster_factor=instance_config.get("cluster_factor", 0.3),
                    depot_position=instance_config.get("depot_position", "center"),
                )
                instances.append(instance)

        except FileNotFoundError:
            print(
                f"Warning: Configuration file {filename} not found. Using default instance."
            )
            # Generate a default instance if the file is not found.
            instances.append(DataGenerator.generate_instance("Default", 10, 1))

        return instances

    @staticmethod
    def create_config_template(filename: Union[str, Path] = "instances_config.json") -> None:
        """
        Create a template configuration file for generating problem instances.
        
        Args:
            filename: Path where the template file will be saved.
            
        Note:
            - Creates a JSON file with example instance configurations
            - The template includes both small and medium-sized instances
            - Users can modify this file to create custom instance sets
            
        Author: Harsh Sharma (231070064)
        """
        template = {
            "instances": [
                {
                    "name": "Small Instance",
                    "customers": 6,
                    "ifs": 1,
                    "capacity": 20,
                    "area_size": 100,
                    "demand_range": [1, 10],
                    "service_time_range": [1, 5],
                    "cluster_factor": 0.3,
                    "depot_position": "center",
                    "seed": 42,
                },
                {
                    "name": "Medium Instance",
                    "customers": 20,
                    "ifs": 2,
                    "capacity": 30,
                    "area_size": 150,
                    "demand_range": [1, 15],
                    "service_time_range": [1, 8],
                    "cluster_factor": 0.5,
                    "depot_position": "center",
                    "seed": 123,
                },
            ]
        }

        with open(filename, "w") as f:
            json.dump(template, f, indent=2)

        print(f"Configuration template created: {filename}")

    @staticmethod
    def generate_edge_cases() -> List[ProblemInstance]:
        """
        Generate a set of edge-case problem instances for testing.
        
        These instances are designed to test the robustness of the solver
        by including challenging scenarios such as:
        - Single customer instances
        - Customers with very high demand
        - Many intermediate facilities relative to customers
        
        Returns:
            List of ProblemInstance objects representing various edge cases.
            
        Note:
            - Each instance has a descriptive name indicating its purpose
            - Fixed random seeds ensure reproducibility
            
        Author: Harsh Sharma (231070064)
        """
        instances = []

        # Case 1: A single customer.
        instances.append(
            DataGenerator.generate_instance(
                "Single Customer", 1, 1, vehicle_capacity=20, seed=1001
            )
        )

        # Case 2: A customer with demand higher than vehicle capacity (should be infeasible).
        instances.append(
            DataGenerator.generate_instance(
                "High Demand Customer",
                5,
                1,
                vehicle_capacity=15,
                demand_range=(20, 25),
                seed=1002,
            )
        )

        # Case 3: Many intermediate facilities relative to customers.
        instances.append(
            DataGenerator.generate_instance(
                "Many IFs", 10, 5, vehicle_capacity=25, seed=1003
            )
        )

        return instances

    @staticmethod
    def generate_benchmark_suite() -> List[ProblemInstance]:
        """
        Generate a standardized set of instances for benchmarking.
        
        The benchmark suite includes:
        - Small instances: 6-15 customers, 1-2 IFs
        - Medium instances: 20-40 customers, 2-3 IFs
        
        Returns:
            List of ProblemInstance objects for benchmarking.
            
        Note:
            - Instance names follow the pattern: "{Size}_{N}C_{M}IF"
            - Fixed random seeds ensure consistent generation
            - Vehicle capacity scales with instance size
            
        Author: Harsh Sharma (231070064)
        """
        instances = []

        # Small instances
        for customers in [6, 10, 15]:
            for ifs in [1, 2]:
                instances.append(
                    DataGenerator.generate_instance(
                        f"Small_{customers}C_{ifs}IF",
                        customers,
                        ifs,
                        vehicle_capacity=20,
                        seed=2000 + customers + ifs,
                    )
                )

        # Medium instances
        for customers in [20, 30, 40]:
            for ifs in [2, 3]:
                instances.append(
                    DataGenerator.generate_instance(
                        f"Medium_{customers}C_{ifs}IF",
                        customers,
                        ifs,
                        vehicle_capacity=30,
                        seed=3000 + customers + ifs,
                    )
                )

        return instances

    @staticmethod
    def save_instance_to_file(instance: ProblemInstance, filename: Union[str, Path]) -> None:
        """
        Save a problem instance to a JSON file.
        
        The saved file includes all instance data, including:
        - Instance metadata (name, creation time, parameters)
        - Depot, customer, and IF locations
        - Distance matrix (if available)
        
        Args:
            instance: The ProblemInstance to save.
            filename: Path where the instance will be saved.
            
        Note:
            - Uses JSON format for portability
            - Includes a timestamp in the metadata
            - The distance matrix is included if it has been computed
            
        Author: Harsh Sharma (231070064)
        """
        import json
        from datetime import datetime

        instance_data = {
            "metadata": {
                "name": instance.name,
                "created": datetime.now().isoformat(),
                "customers": len(instance.customers),
                "ifs": len(instance.intermediate_facilities),
                "vehicle_capacity": instance.vehicle_capacity,
                "number_of_vehicles": instance.number_of_vehicles,
                "disposal_time": instance.disposal_time,
            },
            "depot": {
                "id": instance.depot.id,
                "x": instance.depot.x,
                "y": instance.depot.y,
                "type": instance.depot.type,
            },
            "intermediate_facilities": [
                {
                    "id": if_node.id,
                    "x": if_node.x,
                    "y": if_node.y,
                    "type": if_node.type,
                }
                for if_node in instance.intermediate_facilities
            ],
            "customers": [
                {
                    "id": customer.id,
                    "x": customer.x,
                    "y": customer.y,
                    "demand": customer.demand,
                    "service_time": customer.service_time,
                    "type": customer.type,
                }
                for customer in instance.customers
            ],
            "distance_matrix": instance.distance_matrix.tolist()
            if instance.distance_matrix is not None
            else None,
        }

        with open(filename, "w") as f:
            json.dump(instance_data, f, indent=2)

        print(f"Instance saved to: {filename}")

    @staticmethod
    def load_instance_from_file(filename: Union[str, Path]) -> ProblemInstance:
        """
        Load a problem instance from a JSON file.
        
        Args:
            filename: Path to the JSON file containing the instance data.
            
        Returns:
            ProblemInstance: The loaded problem instance.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            KeyError: If required fields are missing from the file.
            
        Note:
            - Restores all instance attributes, including the distance matrix
            - Handles both numpy arrays and nested lists for the distance matrix
            
        Author: Harsh Sharma (231070064)
        """
        import json

        with open(filename, "r") as f:
            data = json.load(f)

        instance = ProblemInstance(data["metadata"]["name"])
        instance.vehicle_capacity = data["metadata"]["vehicle_capacity"]
        instance.number_of_vehicles = data["metadata"]["number_of_vehicles"]
        instance.disposal_time = data["metadata"]["disposal_time"]

        # Load depot, IFs, and customers from the file data.
        depot_data = data["depot"]
        instance.depot = Location(
            depot_data["id"], depot_data["x"], depot_data["y"], 0, depot_data["type"]
        )

        for if_data in data["intermediate_facilities"]:
            if_node = Location(
                if_data["id"], if_data["x"], if_data["y"], 0, if_data["type"]
            )
            instance.intermediate_facilities.append(if_node)

        for customer_data in data["customers"]:
            customer = Location(
                customer_data["id"],
                customer_data["x"],
                customer_data["y"],
                customer_data["demand"],
                customer_data["type"],
            )
            customer.service_time = customer_data["service_time"]
            instance.customers.append(customer)

        # Load distance matrix if it exists in the file.
        if data["distance_matrix"] is not None:
            import numpy as np

            instance.distance_matrix = np.array(data["distance_matrix"])

        return instance
