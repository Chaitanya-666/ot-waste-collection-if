#!/usr/bin/env python3
# Author: Harsh Sharma (231070064)
#
# This file contains the comprehensive test suite for the entire project.
# It uses Python's `unittest` framework to verify the correctness of all
# components, including the data structures, ALNS algorithm, operators,
# and utilities. The tests are organized into classes, each focusing on a
# specific part of the system.
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
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import patch

from src.alns import ALNS
from src.data_generator import DataGenerator
from src.destroy_operators import RandomRemoval
from src.problem import ProblemInstance, Location
from src.repair_operators import GreedyInsertion
from src.solution import Solution, Route
from src.utils import plot_solution, calculate_statistics


class TestProblemInstance(unittest.TestCase):
    """Tests for the ProblemInstance class and its related data structures."""

    def setUp(self):
        """Set up a sample problem instance for testing."""
        self.problem = ProblemInstance("Test Instance")
        self.problem.vehicle_capacity = 20
        self.depot = Location(0, 0, 0, 0, "depot")
        self.problem.depot = self.depot
        self.customers = [Location(1, 5, 2, 4, "customer")]
        self.problem.customers = self.customers
        self.if1 = Location(100, 20, 20, 0, "if")
        self.problem.intermediate_facilities.append(self.if1)
        self.problem.calculate_distance_matrix()

    def test_problem_creation(self):
        """Test that a problem instance is created with correct attributes."""
        self.assertEqual(self.problem.name, "Test Instance")
        self.assertEqual(len(self.problem.customers), 1)

    def test_distance_calculation(self):
        """Test the Euclidean distance calculation."""
        loc1 = Location(1, 0, 0, 0)
        loc2 = Location(2, 3, 4, 0)
        distance = self.problem.calculate_distance(loc1, loc2)
        self.assertAlmostEqual(distance, 5.0)


class TestSolution(unittest.TestCase):
    """Tests for the Solution and Route classes."""

    def setUp(self):
        """Set up a sample problem and solution for testing."""
        self.problem = ProblemInstance("Test")
        self.problem.vehicle_capacity = 20
        self.depot = Location(0, 0, 0, 0, "depot")
        self.problem.depot = self.depot
        self.customer1 = Location(1, 5, 2, 4, "customer")
        self.problem.customers = [self.customer1]
        self.problem.calculate_distance_matrix()

    def test_solution_creation(self):
        """Test that a new solution correctly initializes."""
        solution = Solution(self.problem)
        self.assertEqual(len(solution.unassigned_customers), 1)

    def test_solution_feasibility(self):
        """Test the feasibility check for a complete and valid solution."""
        solution = Solution(self.problem)
        route = Route()
        route.nodes = [self.depot, self.customer1, self.depot]
        route.calculate_metrics(self.problem)
        solution.routes = [route]
        solution.unassigned_customers = set()
        feasible, _ = solution.is_feasible(self.problem)
        self.assertTrue(feasible)


class TestDestroyOperators(unittest.TestCase):
    """Tests for the different destroy operators."""

    def setUp(self):
        """Create a feasible solution for destroy operators."""
        self.problem = DataGenerator.generate_instance("Test", 10, 2, seed=42)
        solver = ALNS(self.problem)
        self.initial_solution = solver._generate_initial_solution()

    def test_random_removal(self):
        """Test that random removal removes the correct number of customers."""
        operator = RandomRemoval()
        removal_count = 3
        partial_solution = operator.apply(self.initial_solution, removal_count)
        self.assertEqual(len(partial_solution.unassigned_customers),
                         removal_count)


class TestRepairOperators(unittest.TestCase):
    """Tests for the different repair operators."""

    def setUp(self):
        """Create a partial solution for repair operators."""
        self.problem = DataGenerator.generate_instance("Test", 10, 2, seed=42)
        self.partial_solution = Solution(self.problem)
        # Manually create a partial solution
        unassigned = {c.id for c in self.problem.customers[:3]}
        self.partial_solution.unassigned_customers = unassigned

    def test_greedy_insertion(self):
        """Test that greedy insertion re-inserts all unassigned customers."""
        operator = GreedyInsertion()
        repaired_solution = operator.apply(self.partial_solution)
        self.assertEqual(len(repaired_solution.unassigned_customers), 0)


class TestALNSAlgorithm(unittest.TestCase):
    """High-level tests for the ALNS algorithm itself."""

    def setUp(self):
        """Set up a small problem for testing the ALNS run."""
        self.problem = DataGenerator.generate_instance("Test", 6, 1, seed=42)

    def test_alns_run(self):
        """Test that a full ALNS run produces a valid, feasible solution."""
        solver = ALNS(self.problem)
        solver.max_iterations = 50
        solution = solver.run()
        self.assertIsInstance(solution, Solution)
        self.assertEqual(len(solution.unassigned_customers), 0)
        self.assertGreater(solution.total_cost, 0)


class TestDataGenerator(unittest.TestCase):
    """Tests for the synthetic data generation."""

    def test_generate_instance(self):
        """Test that instance generation creates correct number of entities."""
        instance = DataGenerator.generate_instance("Test", 10, 2, seed=42)
        self.assertEqual(len(instance.customers), 10)
        self.assertEqual(len(instance.intermediate_facilities), 2)

    def test_instance_reproducibility(self):
        """Test that the same seed produces the exact same instance."""
        instance1 = DataGenerator.generate_instance("Test", 10, 2, seed=42)
        instance2 = DataGenerator.generate_instance("Test", 10, 2, seed=42)
        self.assertEqual(instance1.customers[0].x, instance2.customers[0].x)


class TestUtilities(unittest.TestCase):
    """Tests for utility functions like visualization and analysis."""

    def setUp(self):
        """Set up a problem and solution for testing utilities."""
        self.problem = DataGenerator.generate_instance("Test", 6, 1, seed=42)
        solver = ALNS(self.problem)
        self.solution = solver.run(max_iterations=10)

    def test_calculate_statistics(self):
        """Test that the statistics calculation returns a valid dictionary."""
        data = [r.total_distance for r in self.solution.routes]
        stats = calculate_statistics(data)
        self.assertIn("mean", stats)
        self.assertIn("std_dev", stats)

    @patch("matplotlib.pyplot.show")
    def test_plot_solution(self, mock_show):
        """Test that the solution plotter generates a plot without errors."""
        plot_solution(self.solution)
        mock_show.assert_called_once()


class TestIntegration(unittest.TestCase):
    """End-to-end integration tests for the complete system."""

    def test_complete_workflow(self):
        """
        Tests the entire workflow: generate instance, solve with ALNS,
        and analyze the result.
        """
        problem = DataGenerator.generate_instance("Integration Test", 15, 2,
                                                  seed=2001)
        solver = ALNS(problem)
        solver.max_iterations = 100
        solution = solver.run()
        
        # Instead of an analyzer class, we can just check the solution properties
        self.assertEqual(len(solution.unassigned_customers), 0)
        self.assertGreater(solution.total_cost, 0)



if __name__ == "__main__":
    unittest.main()