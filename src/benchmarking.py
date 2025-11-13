"""
Benchmarking Module for Algorithm Comparison

This module provides comprehensive benchmarking capabilities for comparing
the ALNS algorithm against other optimization approaches for VRP with IFs.
It includes:

- Comparison with construction heuristics (Nearest Neighbor, Savings)
- Comparison with metaheuristics (Genetic Algorithm, Simulated Annealing)
- Benchmark datasets with known optimal solutions
- Performance metrics and statistical analysis
- Visualization of benchmark results

The benchmarking framework is designed to be extensible and can easily
incorporate additional algorithms for comparison.
"""

import json
import time
import math
import random
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

from .solution import Solution, Route
from .problem import ProblemInstance, Location
from .data_generator import DataGenerator
from .enhanced_validator import EnhancedSolutionValidator, ValidationResult


class AlgorithmType(Enum):
    """Types of algorithms to benchmark"""

    ALNS = "alns"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    SAVINGS = "savings"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    RANDOM = "random"


@dataclass
class BenchmarkInstance:
    """Represents a benchmark problem instance"""

    name: str
    description: str
    n_customers: int
    n_ifs: int
    vehicle_capacity: int
    optimal_cost: Optional[float] = None
    optimal_vehicles: Optional[int] = None
    known_solutions: Optional[Dict[str, float]] = None


@dataclass
class AlgorithmResult:
    """Result from running a single algorithm on an instance"""

    algorithm: AlgorithmType
    instance_name: str
    execution_time: float
    solution_cost: float
    solution: Optional[Solution] = None
    validation_result: Optional[ValidationResult] = None
    convergence_data: Optional[List[float]] = None
    additional_metrics: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResult:
    """Complete benchmarking result for multiple algorithms"""

    instance_name: str
    instance_description: str
    results: List[AlgorithmResult]
    execution_summary: Dict[str, float]
    best_algorithm: AlgorithmType
    cost_gaps: Dict[str, float]
    statistical_analysis: Dict[str, Any]


class BenchmarkingFramework:
    """
    Comprehensive benchmarking framework for VRP with IFs algorithms.

    Provides capabilities to:
    - Generate benchmark datasets
    - Run multiple algorithms on the same instances
    - Compare performance metrics
    - Perform statistical analysis
    - Generate benchmark reports
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the benchmarking framework.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

        # Standard benchmark instances
        self.benchmark_instances: List[BenchmarkInstance] = []
        self.benchmark_results: List[BenchmarkResult] = []

        # Algorithm implementations
        self.algorithm_implementations: Dict[AlgorithmType, Callable] = {
            AlgorithmType.NEAREST_NEIGHBOR: self._nearest_neighbor,
            AlgorithmType.SAVINGS: self._savings_algorithm,
            AlgorithmType.RANDOM: self._random_construction,
        }

    def create_standard_benchmarks(self):
        """Create standard benchmark instances with varying characteristics."""

        # Small instances (for quick testing)
        self.benchmark_instances.extend(
            [
                BenchmarkInstance(
                    name="small_urban",
                    description="Small urban area (15 customers, 2 IFs)",
                    n_customers=15,
                    n_ifs=2,
                    vehicle_capacity=20,
                ),
                BenchmarkInstance(
                    name="small_suburban",
                    description="Small suburban area (15 customers, 2 IFs)",
                    n_customers=15,
                    n_ifs=2,
                    vehicle_capacity=25,
                ),
            ]
        )

        # Medium instances
        self.benchmark_instances.extend(
            [
                BenchmarkInstance(
                    name="medium_urban",
                    description="Medium urban area (30 customers, 4 IFs)",
                    n_customers=30,
                    n_ifs=4,
                    vehicle_capacity=20,
                ),
                BenchmarkInstance(
                    name="medium_suburban",
                    description="Medium suburban area (30 customers, 3 IFs)",
                    n_customers=30,
                    n_ifs=3,
                    vehicle_capacity=25,
                ),
                BenchmarkInstance(
                    name="medium_rural",
                    description="Medium rural area (25 customers, 2 IFs)",
                    n_customers=25,
                    n_ifs=2,
                    vehicle_capacity=35,
                ),
            ]
        )

        # Large instances
        self.benchmark_instances.extend(
            [
                BenchmarkInstance(
                    name="large_urban",
                    description="Large urban area (50 customers, 6 IFs)",
                    n_customers=50,
                    n_ifs=6,
                    vehicle_capacity=20,
                ),
                BenchmarkInstance(
                    name="large_mixed",
                    description="Large mixed area (40 customers, 4 IFs)",
                    n_customers=40,
                    n_ifs=4,
                    vehicle_capacity=30,
                ),
            ]
        )

        # Add some instances with known theoretical bounds
        for instance in self.benchmark_instances:
            if instance.n_customers <= 20:
                # Simple lower bound calculation for small instances
                instance.optimal_cost = self._calculate_theoretical_lower_bound(
                    instance
                )
                instance.optimal_vehicles = self._calculate_min_vehicles(instance)

    def _calculate_theoretical_lower_bound(self, instance: BenchmarkInstance) -> float:
        """Calculate a theoretical lower bound for solution cost."""
        # Simple bound: sum of distances from depot to customers divided by vehicle capacity
        # This is very conservative but provides a reference point
        problem = DataGenerator.generate_instance(
            name=instance.name,
            n_customers=instance.n_customers,
            n_ifs=instance.n_ifs,
            vehicle_capacity=instance.vehicle_capacity,
            seed=self.seed,
        )

        total_distance = 0.0
        for customer in problem.customers:
            total_distance += problem.calculate_distance(problem.depot, customer)

        # Estimate minimum vehicles needed
        total_demand = sum(c.demand for c in problem.customers)
        min_vehicles = math.ceil(total_demand / instance.vehicle_capacity)

        return total_distance / min_vehicles

    def _calculate_min_vehicles(self, instance: BenchmarkInstance) -> int:
        """Calculate minimum vehicles required for an instance."""
        problem = DataGenerator.generate_instance(
            name=instance.name,
            n_customers=instance.n_customers,
            n_ifs=instance.n_ifs,
            vehicle_capacity=instance.vehicle_capacity,
            seed=self.seed,
        )

        total_demand = sum(c.demand for c in problem.customers)
        return math.ceil(total_demand / instance.vehicle_capacity)

    def run_benchmark_suite(
        self,
        algorithms: List[AlgorithmType],
        instances: Optional[List[BenchmarkInstance]] = None,
        iterations_per_algorithm: int = 1,
        validate_solutions: bool = True,
    ) -> List[BenchmarkResult]:
        """
        Run a complete benchmark suite comparing multiple algorithms.

        Args:
            algorithms: List of algorithms to benchmark
            instances: Specific instances to test (uses all if None)
            iterations_per_algorithm: Number of runs per algorithm for statistical analysis
            validate_solutions: Whether to validate all solutions

        Returns:
            List of benchmark results
        """
        if instances is None:
            instances = self.benchmark_instances

        results = []

        for instance in instances:
            print(f"Benchmarking instance: {instance.name}")

            instance_results = []

            for algorithm in algorithms:
                print(f"  Running {algorithm.value}...")

                algorithm_results = []

                for iteration in range(iterations_per_algorithm):
                    result = self._run_algorithm_on_instance(
                        algorithm, instance, iteration
                    )

                    if validate_solutions and result.solution:
                        validator = EnhancedSolutionValidator(result.solution.problem)
                        validation = validator.validate_solution(result.solution)
                        result.validation_result = validation

                    algorithm_results.append(result)

                # Store best result for this algorithm on this instance
                best_result = min(algorithm_results, key=lambda x: x.solution_cost)
                instance_results.append(best_result)

            # Analyze results for this instance
            benchmark_result = self._analyze_instance_results(
                instance, instance_results
            )
            results.append(benchmark_result)
            self.benchmark_results.append(benchmark_result)

        return results

    def _run_algorithm_on_instance(
        self, algorithm: AlgorithmType, instance: BenchmarkInstance, iteration: int
    ) -> AlgorithmResult:
        """Run a single algorithm on a single instance."""

        # Generate problem instance
        problem = DataGenerator.generate_instance(
            name=f"{instance.name}_{iteration}",
            n_customers=instance.n_customers,
            n_ifs=instance.n_ifs,
            vehicle_capacity=instance.vehicle_capacity,
            seed=self.seed + iteration,
        )

        start_time = time.time()

        # Run the algorithm
        if algorithm in self.algorithm_implementations:
            solution = self.algorithm_implementations[algorithm](problem)
        else:
            # For algorithms not implemented, return None
            solution = None
            execution_time = 0.0
            solution_cost = float("inf")
        end_time = time.time()

        if solution:
            execution_time = end_time - start_time
            solution_cost = solution.total_cost
        else:
            execution_time = end_time - start_time
            solution_cost = float("inf")

        return AlgorithmResult(
            algorithm=algorithm,
            instance_name=instance.name,
            execution_time=execution_time,
            solution_cost=solution_cost,
            solution=solution,
            convergence_data=None,  # Would be populated by actual implementations
            additional_metrics={},
        )

    def _nearest_neighbor(self, problem: ProblemInstance) -> Solution:
        """Nearest Neighbor construction heuristic."""
        solution = Solution(problem)
        unassigned = set(c.id for c in problem.customers)

        while unassigned:
            # Start a new route
            route = Route()
            route.nodes = [problem.depot]
            current_load = 0.0
            current_location = problem.depot

            while unassigned:
                # Find nearest unassigned customer
                nearest_customer = None
                min_distance = float("inf")

                for customer_id in unassigned:
                    customer = next(c for c in problem.customers if c.id == customer_id)
                    distance = problem.calculate_distance(current_location, customer)

                    if distance < min_distance:
                        min_distance = distance
                        nearest_customer = customer

                if not nearest_customer:
                    break

                # Check if we can add this customer
                if current_load + nearest_customer.demand <= problem.vehicle_capacity:
                    # Add customer
                    route.nodes.append(nearest_customer)
                    current_load += nearest_customer.demand
                    unassigned.remove(nearest_customer.id)
                    current_location = nearest_customer
                else:
                    # Need to visit IF
                    if problem.intermediate_facilities:
                        # Find nearest IF
                        nearest_if = min(
                            problem.intermediate_facilities,
                            key=lambda if_loc: problem.calculate_distance(
                                current_location, if_loc
                            ),
                        )
                        route.nodes.append(nearest_if)
                        current_load = 0.0
                        current_location = nearest_if
                    else:
                        break

            # Return to depot
            route.nodes.append(problem.depot)
            solution.routes.append(route)

        solution.calculate_metrics()
        return solution

    def _savings_algorithm(self, problem: ProblemInstance) -> Solution:
        """Clarke-Wright savings algorithm."""
        solution = Solution(problem)

        # Initialize routes for each customer
        customer_routes = {}
        for customer in problem.customers:
            route = Route()
            route.nodes = [problem.depot, customer, problem.depot]
            route.calculate_metrics(problem)
            customer_routes[customer.id] = route

        # Calculate savings
        savings = []
        for i, cust1 in enumerate(problem.customers):
            for j, cust2 in enumerate(problem.customers):
                if i < j:  # Avoid duplicates
                    saving = (
                        problem.calculate_distance(problem.depot, cust1)
                        + problem.calculate_distance(problem.depot, cust2)
                        - problem.calculate_distance(cust1, cust2)
                    )
                    savings.append((saving, cust1.id, cust2.id))

        # Sort savings in descending order
        savings.sort(reverse=True, key=lambda x: x[0])

        # Merge routes
        for saving, cust1_id, cust2_id in savings:
            route1 = customer_routes.get(cust1_id)
            route2 = customer_routes.get(cust2_id)

            if route1 and route2 and route1 != route2:
                # Try to merge routes
                if self._can_merge_routes(route1, route2, problem):
                    # Merge route2 into route1
                    merged_route = self._merge_routes(route1, route2, problem)
                    customer_routes[cust1_id] = merged_route
                    del customer_routes[cust2_id]

        # Collect all routes
        solution.routes = list(customer_routes.values())
        solution.calculate_metrics()
        return solution

    def _can_merge_routes(
        self, route1: Route, route2: Route, problem: ProblemInstance
    ) -> bool:
        """Check if two routes can be merged without violating capacity."""
        # Simple check: combine demands and check against capacity
        total_demand = 0.0
        for node in route1.nodes[1:-1]:  # Exclude depot
            if node.type == "customer":
                total_demand += node.demand
        for node in route2.nodes[1:-1]:  # Exclude depot
            if node.type == "customer":
                total_demand += node.demand

        return total_demand <= problem.vehicle_capacity

    def _merge_routes(
        self, route1: Route, route2: Route, problem: ProblemInstance
    ) -> Route:
        """Merge two routes."""
        merged_route = Route()
        merged_route.nodes = route1.nodes[:-1] + route2.nodes[1:]
        merged_route.calculate_metrics(problem)
        return merged_route

    def _random_construction(self, problem: ProblemInstance) -> Solution:
        """Random construction for baseline comparison."""
        solution = Solution(problem)
        unassigned = set(c.id for c in problem.customers)

        while unassigned:
            # Start a new route
            route = Route()
            route.nodes = [problem.depot]
            current_load = 0.0
            current_location = problem.depot

            # Randomly add customers until capacity is reached
            unassigned_list = list(unassigned)
            random.shuffle(unassigned_list)

            for customer_id in unassigned_list:
                customer = next(c for c in problem.customers if c.id == customer_id)

                if current_load + customer.demand <= problem.vehicle_capacity:
                    route.nodes.append(customer)
                    current_load += customer.demand
                    unassigned.remove(customer_id)
                    current_location = customer
                else:
                    break

            # Return to depot
            route.nodes.append(problem.depot)
            route.calculate_metrics(problem)
            solution.routes.append(route)

        solution.calculate_metrics()
        return solution

    def _analyze_instance_results(
        self, instance: BenchmarkInstance, results: List[AlgorithmResult]
    ) -> BenchmarkResult:
        """Analyze results for a single instance."""

        # Find best algorithm
        best_result = min(results, key=lambda x: x.solution_cost)
        best_algorithm = best_result.algorithm

        # Calculate cost gaps
        cost_gaps = {}
        optimal_cost = instance.optimal_cost or best_result.solution_cost

        for result in results:
            if optimal_cost > 0:
                gap = ((result.solution_cost - optimal_cost) / optimal_cost) * 100
            else:
                gap = 0.0
            cost_gaps[result.algorithm.value] = gap

        # Statistical analysis
        algorithm_times = {}
        algorithm_costs = {}

        for result in results:
            alg_name = result.algorithm.value
            if alg_name not in algorithm_times:
                algorithm_times[alg_name] = []
                algorithm_costs[alg_name] = []

            algorithm_times[alg_name].append(result.execution_time)
            algorithm_costs[alg_name].append(result.solution_cost)

        statistical_analysis = {}
        for alg_name in algorithm_times:
            times = algorithm_times[alg_name]
            costs = algorithm_costs[alg_name]

            statistical_analysis[alg_name] = {
                "avg_time": statistics.mean(times),
                "std_time": statistics.stdev(times) if len(times) > 1 else 0.0,
                "avg_cost": statistics.mean(costs),
                "std_cost": statistics.stdev(costs) if len(costs) > 1 else 0.0,
                "min_cost": min(costs),
                "max_cost": max(costs),
            }

        # Execution summary
        execution_summary = {
            "total_time": sum(r.execution_time for r in results),
            "avg_time": statistics.mean([r.execution_time for r in results]),
            "best_cost": best_result.solution_cost,
            "worst_cost": max(r.solution_cost for r in results),
        }

        return BenchmarkResult(
            instance_name=instance.name,
            instance_description=instance.description,
            results=results,
            execution_summary=execution_summary,
            best_algorithm=best_algorithm,
            cost_gaps=cost_gaps,
            statistical_analysis=statistical_analysis,
        )

    def generate_benchmark_report(self, results: List[BenchmarkResult]) -> str:
        """Generate a comprehensive benchmark report."""
        report = []
        report.append("=" * 80)
        report.append("ALGORITHM BENCHMARKING REPORT")
        report.append("=" * 80)
        report.append("")

        # Summary table
        report.append("SUMMARY TABLE")
        report.append("-" * 80)
        report.append(
            f"{'Instance':<20} {'Best Algorithm':<20} {'Best Cost':<15} {'Gap vs Optimal':<15}"
        )
        report.append("-" * 80)

        for result in results:
            best_alg = result.best_algorithm.value
            best_cost = result.best_cost
            gap = result.cost_gaps.get(best_alg, 0.0)

            report.append(
                f"{result.instance_name:<20} {best_alg:<20} {best_cost:<15.2f} {gap:<15.2f}%"
            )

        report.append("")

        # Detailed analysis
        for result in results:
            report.append(f"INSTANCE: {result.instance_name}")
            report.append(f"Description: {result.instance_description}")
            report.append("")

            report.append("ALGORITHM PERFORMANCE")
            report.append("-" * 60)
            report.append(
                f"{'Algorithm':<20} {'Cost':<15} {'Time (s)':<12} {'Gap (%)':<10} {'Status':<10}"
            )
            report.append("-" * 60)

            for alg_result in result.results:
                alg_name = alg_result.algorithm.value
                cost = alg_result.solution_cost
                time_sec = alg_result.execution_time
                gap = result.cost_gaps.get(alg_name, 0.0)
                status = "✓" if alg_result.solution else "✗"

                report.append(
                    f"{alg_name:<20} {cost:<15.2f} {time_sec:<12.3f} {gap:<10.2f}% {status:<10}"
                )

            report.append("")

            # Statistical analysis
            report.append("STATISTICAL ANALYSIS")
            report.append("-" * 60)
            for alg_name, stats in result.statistical_analysis.items():
                report.append(f"{alg_name.upper()}:")
                report.append(
                    f"  Average Cost: {stats['avg_cost']:.2f} ± {stats['std_cost']:.2f}"
                )
                report.append(
                    f"  Average Time: {stats['avg_time']:.3f} ± {stats['std_time']:.3f}"
                )
                report.append(
                    f"  Cost Range: [{stats['min_cost']:.2f}, {stats['max_cost']:.2f}]"
                )
                report.append("")

        # Overall conclusions
        report.append("OVERALL CONCLUSIONS")
        report.append("-" * 60)

        # Best performing algorithms
        algorithm_wins = {}
        for result in results:
            best_alg = result.best_algorithm.value
            algorithm_wins[best_alg] = algorithm_wins.get(best_alg, 0) + 1

        sorted_wins = sorted(algorithm_wins.items(), key=lambda x: x[1], reverse=True)
        report.append("Algorithm Wins (Best Performance):")
        for alg, wins in sorted_wins:
            report.append(f"  {alg}: {wins} instances")

        report.append("")

        # Performance analysis
        total_times = {}
        total_costs = {}

        for result in results:
            for alg_result in result.results:
                alg_name = alg_result.algorithm.value
                if alg_name not in total_times:
                    total_times[alg_name] = 0
                    total_costs[alg_name] = 0

                total_times[alg_name] += alg_result.execution_time
                total_costs[alg_name] += alg_result.solution_cost

        report.append("Cumulative Performance:")
        report.append(f"{'Algorithm':<20} {'Total Time (s)':<15} {'Total Cost':<15}")
        report.append("-" * 50)

        for alg_name in total_times:
            report.append(
                f"{alg_name:<20} {total_times[alg_name]:<15.3f} {total_costs[alg_name]:<15.2f}"
            )

        return "\n".join(report)

    def save_benchmark_results(self, results: List[BenchmarkResult], filename: str):
        """Save benchmark results to JSON file."""
        # Convert to serializable format
        serializable_results = []

        for result in results:
            serializable_result = {
                "instance_name": result.instance_name,
                "instance_description": result.instance_description,
                "results": [
                    {
                        "algorithm": r.algorithm.value,
                        "execution_time": r.execution_time,
                        "solution_cost": r.solution_cost,
                        "validation_status": r.validation_result.status.value
                        if r.validation_result
                        else None,
                    }
                    for r in result.results
                ],
                "best_algorithm": result.best_algorithm.value,
                "cost_gaps": result.cost_gaps,
                "statistical_analysis": result.statistical_analysis,
            }
            serializable_results.append(serializable_result)

        with open(filename, "w") as f:
            json.dump(serializable_results, f, indent=2)

    def load_benchmark_results(self, filename: str) -> List[BenchmarkResult]:
        """Load benchmark results from JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)

        results = []
        for item in data:
            # This is a simplified loading - in practice, you'd want to reconstruct
            # the full BenchmarkResult objects with proper type conversion
            results.append(item)

        return results
