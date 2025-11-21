# Author: Harsh Sharma (231070064)
#
# This file is responsible for generating synthetic data for problem instances.
# It allows for the creation of diverse and reproducible scenarios for testing,
# benchmarking, and demonstration purposes.
"""
Synthetic data generation for waste collection problems

This module provides comprehensive data generation capabilities for:
- Creating synthetic problem instances with various characteristics
- Loading instances from configuration files
- Generating benchmark instances for testing
- Creating edge cases and stress test scenarios
"""

import json
import random
import math
from typing import List, Dict, Tuple, Optional
from .problem import ProblemInstance, Location


class DataGenerator:
    """A collection of static methods to generate synthetic problem instances."""

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
        Generates a complete, synthetic problem instance with specified characteristics.

        Args:
            name: The name for the problem instance.
            n_customers: The number of customer locations to generate.
            n_ifs: The number of intermediate facilities to generate.
            vehicle_capacity: The capacity for each vehicle.
            area_size: The size of the square area (e.g., 100x100).
            demand_range: The (min, max) range for customer demands.
            service_time_range: The (min, max) range for customer service times.
            seed: A random seed for reproducibility.
            cluster_factor: A value from 0.0 (uniform) to 1.0 (highly clustered)
                            that controls customer distribution. A higher value
                            means customers are more likely to be grouped together.
            depot_position: The position of the depot ('center', 'corner', or 'random').
                            This affects the overall structure of the routes.

        Returns:
            A fully initialized ProblemInstance object.
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
        """Generates coordinates for customers, potentially grouped into clusters."""
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
    def generate_instances_from_file(filename: str) -> List[ProblemInstance]:
        """
        Generates a list of problem instances from a JSON configuration file.
        This allows for easy definition and reproduction of multiple test scenarios.
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
    def create_config_template(filename: str = "instances_config.json"):
        """Creates a template JSON file for defining multiple problem instances."""
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
        """Generates a list of edge-case problem instances for stress testing the solver."""
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
        """Generates a comprehensive suite of instances for benchmarking purposes."""
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
    def save_instance_to_file(instance: ProblemInstance, filename: str):
        """Saves a problem instance to a JSON file for persistence and sharing."""
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
    def load_instance_from_file(filename: str) -> ProblemInstance:
        """Loads a problem instance from a JSON file."""
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
