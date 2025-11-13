#!/usr/bin/env python3
"""
Comprehensive test suite for the ALNS VRP-IF project

This test suite covers:
- Problem instance creation and validation
- Solution feasibility and operations
- Destroy operators functionality
- Repair operators functionality
- ALNS algorithm integration
- Performance and convergence testing
- Edge cases and stress testing

All tests use proper import paths relative to the project root.
"""

import unittest
import sys
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
import time

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.problem import ProblemInstance, Location
from src.solution import Solution, Route
from src.alns import ALNS
from src.destroy_operators import (
    RandomRemoval,
    WorstRemoval,
    ShawRemoval,
    RouteRemoval,
    DestroyOperatorManager,
)
from src.repair_operators import (
    GreedyInsertion,
    RegretInsertion,
    IFAwareRepair,
    SavingsInsertion,
    RepairOperatorManager,
)
from src.data_generator import DataGenerator
from src.utils import RouteVisualizer, PerformanceAnalyzer, save_solution_to_file


class TestProblemInstance(unittest.TestCase):
    """Test ProblemInstance class functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.problem = ProblemInstance("Test Instance")
        self.problem.vehicle_capacity = 20
        self.problem.number_of_vehicles = 3
        self.problem.disposal_time = 2

        # Add depot
        self.depot = Location(0, 0, 0, 0, "depot")
        self.problem.depot = self.depot

        # Add customers
        self.customers = [
            Location(1, 5, 2, 4, "customer"),
            Location(2, 3, 8, 6, "customer"),
            Location(3, 9, 1, 5, "customer"),
        ]
        for c in self.customers:
            self.problem.customers.append(c)

        # Add IFs
        self.if1 = Location(100, 20, 20, 0, "if")
        self.problem.intermediate_facilities.append(self.if1)

        self.problem.calculate_distance_matrix()

    def test_problem_creation(self):
        """Test basic problem instance creation"""
        self.assertEqual(self.problem.name, "Test Instance")
        self.assertEqual(self.problem.vehicle_capacity, 20)
        self.assertEqual(len(self.problem.customers), 3)
        self.assertEqual(len(self.problem.intermediate_facilities), 1)

    def test_distance_calculation(self):
        """Test distance calculation between locations"""
        customer1 = self.problem.customers[0]
        customer2 = self.problem.customers[1]

        distance = self.problem.calculate_distance(customer1, customer2)
        self.assertIsInstance(distance, (int, float))
        self.assertGreater(distance, 0)

    def test_distance_matrix(self):
        """Test distance matrix calculation"""
        self.assertIsNotNone(self.problem.distance_matrix)
        if hasattr(self.problem.distance_matrix, "shape"):
            self.assertEqual(
                self.problem.distance_matrix.shape, (5, 5)
            )  # depot + 3 customers + 1 IF

    def test_min_vehicles_calculation(self):
        """Test minimum vehicles calculation"""
        min_vehicles = self.problem.get_min_vehicles_needed()
        total_demand = sum(c.demand for c in self.problem.customers)
        expected_min = max(1, int((total_demand + self.problem.vehicle_capacity - 1) // self.problem.vehicle_capacity))
        self.assertEqual(min_vehicles, expected_min)

    def test_route_feasibility(self):
        """Test feasibility checking for routes"""
        route = Route()
        route.nodes = [self.depot, self.customers[0], self.customers[1], self.depot]
        route.calculate_metrics(self.problem)

        feasible, message = self.problem.is_route_feasible(route)
        self.assertTrue(feasible)
        self.assertEqual(message, "Route is feasible")


class TestSolution(unittest.TestCase):
    """Test Solution class functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.problem = ProblemInstance("Test")
        self.problem.vehicle_capacity = 20
        self.problem.disposal_time = 2

        # Add locations
        self.depot = Location(0, 0, 0, 0, "depot")
        self.problem.depot = self.depot

        self.customer1 = Location(1, 5, 2, 4, "customer")
        self.customer2 = Location(2, 3, 8, 6, "customer")
        self.if1 = Location(100, 20, 20, 0, "if")

        self.problem.customers = [self.customer1, self.customer2]
        self.problem.intermediate_facilities = [self.if1]

        self.problem.calculate_distance_matrix()

    def test_solution_creation(self):
        """Test basic solution creation"""
        solution = Solution(self.problem)
        self.assertEqual(len(solution.routes), 0)
        self.assertEqual(
            len(solution.unassigned_customers), 2
        )  # All customers initially unassigned
        self.assertEqual(solution.total_cost, 0)

    def test_route_creation(self):
        """Test route creation and basic operations"""
        route = Route()
        route.nodes = [self.depot, self.customer1, self.depot]

        self.assertEqual(len(route.nodes), 3)
        self.assertEqual(route.nodes[0], self.depot)
        self.assertEqual(route.nodes[-1], self.depot)

    def test_solution_feasibility(self):
        """Test solution feasibility checking"""
        solution = Solution(self.problem)

        # Create a feasible route
        route = Route()
        route.nodes = [self.depot, self.customer1, self.customer2, self.depot]
        route.calculate_metrics(self.problem)

        solution.routes = [route]
        solution.unassigned_customers = set()
        solution.calculate_metrics()

        feasible, message = solution.is_feasible(self.problem)
        self.assertTrue(feasible)
        self.assertEqual(message, "Solution is feasible")

    def test_solution_copy(self):
        """Test solution copying functionality"""
        solution = Solution(self.problem)
        route = Route()
        route.nodes = [self.depot, self.customer1, self.depot]
        solution.routes = [route]
        solution.unassigned_customers = {1}

        copied_solution = solution.copy()

        self.assertEqual(len(copied_solution.routes), len(solution.routes))
        self.assertEqual(
            copied_solution.unassigned_customers, solution.unassigned_customers
        )

        # Ensure it's a deep copy
        copied_solution.routes[0].nodes.append(self.customer2)
        self.assertEqual(len(solution.routes[0].nodes), 3)


class TestDestroyOperators(unittest.TestCase):
    """Test destroy operators functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.problem = DataGenerator.generate_instance("Test", 10, 2, seed=42)
        self.initial_solution = self._create_feasible_solution()

    def _create_feasible_solution(self):
        """Create a feasible solution for testing"""
        solution = Solution(self.problem)

        # Create simple routes
        for i in range(3):
            route = Route()
            route.nodes = [self.problem.depot]

            # Add some customers
            for j in range(3):
                customer_idx = i * 3 + j
                if customer_idx < len(self.problem.customers):
                    route.nodes.append(self.problem.customers[customer_idx])

            route.nodes.append(self.problem.depot)
            route.calculate_metrics(self.problem)
            solution.routes.append(route)

        solution.unassigned_customers = set()
        solution.calculate_metrics()
        return solution

    def test_random_removal(self):
        """Test random removal operator"""
        operator = RandomRemoval()
        removal_count = 3

        partial_solution = operator.apply(self.initial_solution, removal_count)

        # Check that customers were removed
        self.assertLessEqual(len(partial_solution.unassigned_customers), removal_count)
        self.assertGreater(len(partial_solution.unassigned_customers), 0)

    def test_worst_removal(self):
        """Test worst removal operator"""
        operator = WorstRemoval()
        removal_count = 3

        partial_solution = operator.apply(self.initial_solution, removal_count)

        # Check that customers were removed
        self.assertLessEqual(len(partial_solution.unassigned_customers), removal_count)
        self.assertGreater(len(partial_solution.unassigned_customers), 0)

    def test_shaw_removal(self):
        """Test Shaw removal operator"""
        operator = ShawRemoval()
        removal_count = 3

        partial_solution = operator.apply(self.initial_solution, removal_count)

        # Check that customers were removed
        self.assertLessEqual(len(partial_solution.unassigned_customers), removal_count)
        self.assertGreater(len(partial_solution.unassigned_customers), 0)

    def test_route_removal(self):
        """Test route removal operator"""
        operator = RouteRemoval()
        removal_count = 1

        partial_solution = operator.apply(self.initial_solution, removal_count)

        # Check that entire routes were removed
        self.assertLessEqual(
            len(partial_solution.routes),
            len(self.initial_solution.routes) - removal_count,
        )

    def test_destroy_operator_manager(self):
        """Test destroy operator manager"""
        manager = DestroyOperatorManager(self.problem)

        # Test operator selection
        selected_operator = manager.select_operator()
        self.assertIn(selected_operator, manager.operators.keys())

        # Test operator application
        removal_count = 3
        partial_solution = manager.apply_operator(
            self.initial_solution, selected_operator, removal_count
        )

        self.assertIsInstance(partial_solution, Solution)


class TestRepairOperators(unittest.TestCase):
    """Test repair operators functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.problem = DataGenerator.generate_instance("Test", 10, 2, seed=42)
        self.partial_solution = self._create_partial_solution()

    def _create_partial_solution(self):
        """Create a partial solution for testing"""
        solution = Solution(self.problem)

        # Create one route with some customers
        route = Route()
        route.nodes = [
            self.problem.depot,
            self.problem.customers[0],
            self.problem.depot,
        ]
        route.calculate_metrics(self.problem)
        solution.routes = [route]

        # Mark some customers as unassigned
        solution.unassigned_customers = {c.id for c in self.problem.customers[1:5]}

        return solution

    def test_greedy_insertion(self):
        """Test greedy insertion operator"""
        operator = GreedyInsertion()

        repaired_solution = operator.apply(self.partial_solution)

        # Check that some customers were assigned
        self.assertLessEqual(
            len(repaired_solution.unassigned_customers),
            len(self.partial_solution.unassigned_customers),
        )

    def test_regret_insertion(self):
        """Test regret insertion operator"""
        operator = RegretInsertion(k=2)

        repaired_solution = operator.apply(self.partial_solution)

        # Check that some customers were assigned
        self.assertLessEqual(
            len(repaired_solution.unassigned_customers),
            len(self.partial_solution.unassigned_customers),
        )

    def test_if_aware_repair(self):
        """Test IF-aware repair operator"""
        operator = IFAwareRepair()

        repaired_solution = operator.apply(self.partial_solution)

        # Check that some customers were assigned
        self.assertLessEqual(
            len(repaired_solution.unassigned_customers),
            len(self.partial_solution.unassigned_customers),
        )

    def test_savings_insertion(self):
        """Test savings insertion operator"""
        operator = SavingsInsertion()

        repaired_solution = operator.apply(self.partial_solution)

        # Check that some customers were assigned
        self.assertLessEqual(
            len(repaired_solution.unassigned_customers),
            len(self.partial_solution.unassigned_customers),
        )

    def test_repair_operator_manager(self):
        """Test repair operator manager"""
        manager = RepairOperatorManager()

        # Test operator selection
        selected_operator = manager.select()
        self.assertIn(selected_operator.name, [op.name for op in manager.operators])

        # Test operator application
        repaired_solution = selected_operator.apply(self.partial_solution)

        self.assertIsInstance(repaired_solution, Solution)


class TestALNSAlgorithm(unittest.TestCase):
    """Test ALNS algorithm integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.problem = DataGenerator.generate_instance("Test", 6, 1, seed=42)

    def test_alns_initialization(self):
        """Test ALNS algorithm initialization"""
        solver = ALNS(self.problem)

        self.assertEqual(solver.problem, self.problem)
        self.assertIsNotNone(solver.destroy_operators)
        self.assertIsNotNone(solver.repair_operators)
        self.assertEqual(len(solver.destroy_weights), len(solver.destroy_operators))
        self.assertEqual(len(solver.repair_weights), len(solver.repair_operators))

    def test_alns_initial_solution(self):
        """Test ALNS initial solution generation"""
        solver = ALNS(self.problem)

        initial_solution = solver._generate_initial_solution()

        self.assertIsInstance(initial_solution, Solution)
        self.assertEqual(len(initial_solution.unassigned_customers), 0)

    def test_alns_run(self):
        """Test ALNS algorithm run"""
        solver = ALNS(self.problem)
        solver.max_iterations = 50  # Short run for testing

        solution = solver.run()

        self.assertIsInstance(solution, Solution)
        self.assertEqual(len(solution.unassigned_customers), 0)
        self.assertGreater(solution.total_cost, 0)

    def test_alns_convergence(self):
        """Test ALNS convergence tracking"""
        solver = ALNS(self.problem)
        solver.max_iterations = 100

        solution = solver.run()

        # Check that convergence history was recorded
        self.assertGreater(len(solver.convergence_history), 0)
        self.assertLessEqual(len(solver.convergence_history), solver.max_iterations)

        # Check that solution improved over time
        if len(solver.convergence_history) > 1:
            self.assertLessEqual(
                solver.convergence_history[-1], solver.convergence_history[0]
            )

    def test_vehicle_constraint_handling(self):
        """Test that ALNS respects vehicle constraints"""
        # Create a problem that requires multiple vehicles
        problem = DataGenerator.generate_instance("Multi-Vehicle Test", 15, 2, seed=42)
        
        # Ensure minimum vehicles are set correctly
        min_needed = int(problem.get_min_vehicles_needed())
        if problem.number_of_vehicles == float('inf') or problem.number_of_vehicles < min_needed:
            problem.number_of_vehicles = min_needed
        
        solver = ALNS(problem)
        solver.max_iterations = 50
        
        solution = solver.run()
        
        # Solution should not exceed vehicle limit
        self.assertLessEqual(len(solution.routes), problem.number_of_vehicles)


class TestDataGenerator(unittest.TestCase):
    """Test data generation functionality"""

    def test_generate_instance(self):
        """Test basic instance generation"""
        instance = DataGenerator.generate_instance("Test", 10, 2, seed=42)

        self.assertEqual(instance.name, "Test")
        self.assertEqual(len(instance.customers), 10)
        self.assertEqual(len(instance.intermediate_facilities), 2)
        self.assertEqual(instance.vehicle_capacity, 20)

    def test_instance_reproducibility(self):
        """Test that instances are reproducible with same seed"""
        instance1 = DataGenerator.generate_instance("Test", 10, 2, seed=42)
        instance2 = DataGenerator.generate_instance("Test", 10, 2, seed=42)

        # Check that customers have same coordinates
        for i, (c1, c2) in enumerate(zip(instance1.customers, instance2.customers)):
            self.assertEqual(c1.x, c2.x)
            self.assertEqual(c1.y, c2.y)
            self.assertEqual(c1.demand, c2.demand)

    def test_config_file_generation(self):
        """Test configuration file generation"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_file = f.name

        try:
            DataGenerator.create_config_template(config_file)

            # Check that file was created and is valid JSON
            with open(config_file, "r") as f:
                config = json.load(f)

            self.assertIn("instances", config)
            self.assertGreater(len(config["instances"]), 0)

        finally:
            os.unlink(config_file)

    def test_instance_saving_loading(self):
        """Test instance saving and loading"""
        instance = DataGenerator.generate_instance("Test", 10, 2, seed=42)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            instance_file = f.name

        try:
            # Save instance
            DataGenerator.save_instance_to_file(instance, instance_file)

            # Load instance
            loaded_instance = DataGenerator.load_instance_from_file(instance_file)

            # Check that loaded instance matches original
            self.assertEqual(loaded_instance.name, instance.name)
            self.assertEqual(len(loaded_instance.customers), len(instance.customers))
            self.assertEqual(
                len(loaded_instance.intermediate_facilities),
                len(instance.intermediate_facilities),
            )

        finally:
            os.unlink(instance_file)


class TestUtilities(unittest.TestCase):
    """Test utility functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.problem = DataGenerator.generate_instance("Test", 6, 1, seed=42)
        self.solution = self._create_test_solution()

    def _create_test_solution(self):
        """Create a test solution"""
        solution = Solution(self.problem)

        # Create a simple route
        route = Route()
        route.nodes = [
            self.problem.depot,
            self.problem.customers[0],
            self.problem.customers[1],
            self.problem.depot,
        ]
        route.calculate_metrics(self.problem)
        solution.routes = [route]
        solution.unassigned_customers = set()
        solution.calculate_metrics()

        return solution

    def test_performance_analyzer(self):
        """Test performance analyzer"""
        analyzer = PerformanceAnalyzer(self.problem)

        analysis = analyzer.analyze_solution(self.solution)

        self.assertIn("total_cost", analysis)
        self.assertIn("total_distance", analysis)
        self.assertIn("num_vehicles", analysis)
        self.assertIn("route_details", analysis)
        self.assertIn("efficiency_metrics", analysis)

        # Check that all routes are analyzed
        self.assertEqual(len(analysis["route_details"]), len(self.solution.routes))

    def test_efficiency_metrics_bounds(self):
        """Test that efficiency metrics are within reasonable bounds"""
        analyzer = PerformanceAnalyzer(self.problem)
        analysis = analyzer.analyze_solution(self.solution)
        metrics = analysis["efficiency_metrics"]

        # Vehicle efficiency should be between 0% and 100%
        self.assertGreaterEqual(metrics['vehicle_efficiency'], 0)
        self.assertLessEqual(metrics['vehicle_efficiency'], 1)

        # Capacity utilization should be between 0% and 100%
        self.assertGreaterEqual(metrics['capacity_utilization'], 0)
        self.assertLessEqual(metrics['capacity_utilization'], 1)

    def test_performance_report(self):
        """Test performance report generation"""
        analyzer = PerformanceAnalyzer(self.problem)

        report = analyzer.generate_report(self.solution)

        self.assertIsInstance(report, str)
        self.assertIn("WASTE COLLECTION PERFORMANCE REPORT", report)
        self.assertIn("Total Cost:", report)
        self.assertIn("Vehicles Used:", report)

    @patch("matplotlib.pyplot.show")
    def test_route_visualization(self, mock_show):
        """Test route visualization"""
        visualizer = RouteVisualizer(self.problem)

        # Test solution plotting
        fig = visualizer.plot_solution(self.solution)
        self.assertIsNotNone(fig)

        # Test convergence plotting
        convergence_history = [100, 90, 80, 70, 60]
        fig = visualizer.plot_convergence(convergence_history)
        self.assertIsNotNone(fig)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and stress scenarios"""

    def test_single_customer_instance(self):
        """Test instance with single customer"""
        problem = DataGenerator.generate_instance("Single", 1, 1, seed=1001)
        solver = ALNS(problem)

        solution = solver.run(max_iterations=50)

        self.assertEqual(len(solution.unassigned_customers), 0)
        self.assertGreater(len(solution.routes), 0)

    def test_high_demand_customer(self):
        """Test instance with high demand customer"""
        problem = DataGenerator.generate_instance(
            "High Demand", 5, 1, vehicle_capacity=15, demand_range=(20, 25), seed=1002
        )
        
        # This should be infeasible
        feasible, msg = problem.is_feasible()
        self.assertFalse(feasible)
        self.assertIn("demand", msg.lower())

    def test_many_ifs_instance(self):
        """Test instance with many intermediate facilities"""
        problem = DataGenerator.generate_instance("Many IFs", 10, 5, seed=1003)
        solver = ALNS(problem)

        solution = solver.run(max_iterations=50)

        self.assertEqual(len(solution.unassigned_customers), 0)

    def test_clustered_customers(self):
        """Test instance with clustered customers"""
        problem = DataGenerator.generate_instance(
            "Clustered", 20, 1, cluster_factor=0.9, seed=1005
        )
        solver = ALNS(problem)

        solution = solver.run(max_iterations=50)

        self.assertEqual(len(solution.unassigned_customers), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def test_complete_workflow(self):
        """Test complete workflow from instance generation to solution"""
        # Generate instance
        problem = DataGenerator.generate_instance("Integration Test", 15, 2, seed=2001)

        # Solve with ALNS
        solver = ALNS(problem)
        solver.max_iterations = 100

        start_time = time.time()
        solution = solver.run()
        end_time = time.time()

        # Validate solution
        self.assertEqual(len(solution.unassigned_customers), 0)
        self.assertGreater(solution.total_cost, 0)
        self.assertGreater(end_time - start_time, 0)

        # Analyze performance
        analyzer = PerformanceAnalyzer(problem)
        analysis = analyzer.analyze_solution(solution)

        self.assertIn("total_cost", analysis)
        self.assertIn("efficiency_metrics", analysis)

        # Test efficiency metrics are reasonable
        metrics = analysis["efficiency_metrics"]
        self.assertLessEqual(metrics["vehicle_efficiency"], 1.0)
        self.assertLessEqual(metrics["capacity_utilization"], 1.0)

        # Generate report
        report = analyzer.generate_report(solution)
        self.assertIsInstance(report, str)

    def test_multiple_instances(self):
        """Test solving multiple instances"""
        instances = [
            DataGenerator.generate_instance("Test1", 8, 1, seed=3001),
            DataGenerator.generate_instance("Test2", 12, 2, seed=3002),
            DataGenerator.generate_instance("Test3", 20, 3, seed=3003),
        ]

        results = []

        for problem in instances:
            solver = ALNS(problem)
            solver.max_iterations = 50

            start_time = time.time()
            solution = solver.run()
            end_time = time.time()

            results.append(
                {
                    "name": problem.name,
                    "cost": solution.total_cost,
                    "vehicles": len(solution.routes),
                    "time": end_time - start_time,
                    "feasible": len(solution.unassigned_customers) == 0,
                }
            )

        # Check that all instances were solved successfully
        for result in results:
            self.assertTrue(result["feasible"])
            self.assertGreater(result["cost"], 0)
            self.assertGreater(result["time"], 0)


def run_all_tests():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestProblemInstance,
        TestSolution,
        TestDestroyOperators,
        TestRepairOperators,
        TestALNSAlgorithm,
        TestDataGenerator,
        TestUtilities,
        TestEdgeCases,
        TestIntegration,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n=== TEST SUMMARY ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )

    if result.failures:
        print(f"\n=== FAILURES ===")
        for test, traceback in result.failures:
            print(f"{test}: {traceback}")

    if result.errors:
        print(f"\n=== ERRORS ===")
        for test, traceback in result.errors:
            print(f"{test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)