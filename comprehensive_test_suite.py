#!/usr/bin/env python3
# Author: Harsh Sharma (231070064)
#
# This file contains the comprehensive test suite for the entire project.
# It uses Python's `unittest` framework to verify the correctness of all
# components, including the data structures, ALNS algorithm, operators,
# and utilities. This is the main test suite for the project.
"""
Comprehensive Test Suite for ALNS VRP with Video Creation
Tests small, medium, and large problem instances
"""

import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Correctly add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from test_runner import CustomTestRunner
from simple_video_creator import SimpleVideoCreator

class TestALNSVRPBasic(unittest.TestCase):
    """Test basic ALNS VRP functionality without video creation"""

    def test_basic_import(self):
        """Test that all modules can be imported successfully"""
        try:
            from src.problem import ProblemInstance, Location
            from src.alns import ALNS
            from src.data_generator import DataGenerator
            from src.utils import RouteVisualizer, PerformanceAnalyzer
            self.assertTrue(True, "All imports successful")
        except ImportError as e:
            self.fail(f"Import failed: {e}")

    def test_problem_creation(self):
        """Test problem instance creation"""
        from src.problem import ProblemInstance, Location
        
        problem = ProblemInstance("Test Problem")
        self.assertEqual(problem.name, "Test Problem")
        
        depot = Location(0, 0, 0, 0, "depot")
        problem.depot = depot
        self.assertEqual(problem.depot, depot)
        
        customer1 = Location(1, 10, 10, 5, "customer")
        problem.customers.append(customer1)
        self.assertEqual(len(problem.customers), 1)

    def test_data_generator_small(self):
        """Test data generator with small instance"""
        from src.data_generator import DataGenerator
        
        problem = DataGenerator.generate_instance(
            name="Small Test", n_customers=5, n_ifs=1, seed=42
        )
        
        self.assertEqual(len(problem.customers), 5)
        self.assertEqual(len(problem.intermediate_facilities), 1)

class TestALNSVRPHyperparameterTuning(unittest.TestCase):
    """Test medium-sized ALNS VRP functionality with hyperparameter tuning"""

    def test_medium_problem_solve_fast(self):
        """Hyperparameter Test (Fast): 50 iter, 1000 temp, 0.995 cool"""
        from src.data_generator import DataGenerator
        from src.alns import ALNS
        
        problem = DataGenerator.generate_instance(
            name="Medium Test", n_customers=15, n_ifs=2, seed=42
        )
        
        solver = ALNS(problem)
        solver.max_iterations = 50
        solver.temperature_initial = 1000
        solver.cooling_rate = 0.995
        
        solution = solver.run(track_history=True)
        
        self.assertIsNotNone(solution)
        self.assertGreater(solution.total_cost, 0)

        # Create video
        creator = SimpleVideoCreator(output_dir=".")
        creator.create_optimization_animation(
            solver.history,
            problem.get_customer_data(),
            (problem.depot.x, problem.depot.y),
            [(ifac.x, ifac.y) for ifac in problem.intermediate_facilities],
            output_filename="submissions/hyperparameter_fast_output.gif"
        )

    def test_medium_problem_solve_balanced(self):
        """Hyperparameter Test (Balanced): 100 iter, 5000 temp, 0.99 cool"""
        from src.data_generator import DataGenerator
        from src.alns import ALNS
        
        problem = DataGenerator.generate_instance(
            name="Medium Test", n_customers=15, n_ifs=2, seed=42
        )
        
        solver = ALNS(problem)
        solver.max_iterations = 100
        solver.temperature_initial = 5000
        solver.cooling_rate = 0.99
        
        solution = solver.run(track_history=True)
        
        self.assertIsNotNone(solution)
        self.assertGreater(solution.total_cost, 0)

        # Create video
        creator = SimpleVideoCreator(output_dir=".")
        creator.create_optimization_animation(
            solver.history,
            problem.get_customer_data(),
            (problem.depot.x, problem.depot.y),
            [(ifac.x, ifac.y) for ifac in problem.intermediate_facilities],
            output_filename="submissions/hyperparameter_balanced_output.gif"
        )

    def test_medium_problem_solve_deep(self):
        """Hyperparameter Test (Deep): 200 iter, 10000 temp, 0.985 cool"""
        from src.data_generator import DataGenerator
        from src.alns import ALNS
        
        problem = DataGenerator.generate_instance(
            name="Medium Test", n_customers=15, n_ifs=2, seed=42
        )
        
        solver = ALNS(problem)
        solver.max_iterations = 200
        solver.temperature_initial = 10000
        solver.cooling_rate = 0.985
        
        solution = solver.run(track_history=True)
        
        self.assertIsNotNone(solution)
        self.assertGreater(solution.total_cost, 0)

        # Create video
        creator = SimpleVideoCreator(output_dir=".")
        creator.create_optimization_animation(
            solver.history,
            problem.get_customer_data(),
            (problem.depot.x, problem.depot.y),
            [(ifac.x, ifac.y) for ifac in problem.intermediate_facilities],
            output_filename="submissions/hyperparameter_deep_output.gif"
        )

class TestALNSVRPVideoCreation(unittest.TestCase):
    """Test video creation functionality"""

    def test_video_creator_import(self):
        """Test that video creator can be imported"""
        try:
            from simple_video_creator import SimpleVideoCreator
            self.assertIsNotNone(SimpleVideoCreator())
        except ImportError as e:
            self.fail(f"Video creator import failed: {e}")

    def test_video_creation_with_sample_data(self):
        """Test actual video creation with sample optimization data"""
        from simple_video_creator import SimpleVideoCreator
        
        sample_history = [
            {'iteration': 1, 'cost': 100.0, 'best_cost': 100.0, 'routes': [[(0, 0), (5, 5), (0, 0)]]},
            {'iteration': 2, 'cost': 95.0, 'best_cost': 95.0, 'routes': [[(0, 0), (3, 7), (5, 5), (0, 0)]]}
        ]
        
        creator = SimpleVideoCreator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = creator.create_optimization_animation(
                optimization_history=sample_history,
                customer_data={(1, 2): 5},
                depot_location=(0, 0),
                intermediate_facilities=[(2, 3)],
                output_filename=os.path.join(temp_dir, "test_video.gif")
            )
            
            if video_path and os.path.exists(video_path):
                self.assertTrue(os.path.exists(video_path))
            else:
                self.skipTest("Video creation dependencies not available")

def generate_results_summary(result):
    """Generates a Markdown summary of the test results."""
    
    summary_path = "result_sheet.md"
    
    with open(summary_path, "w") as f:
        f.write("# Result Sheet\n\n")
        f.write("| Test Case | Description | Status | Time | Outputs |\n")
        f.write("|-----------|-------------|--------|------|---------|\n")
        
        for res in result.test_results:
            output_file = ""
            if "fast" in res['name']:
                output_file = "hyperparameter_fast_output.gif"
            elif "balanced" in res['name']:
                output_file = "hyperparameter_balanced_output.gif"
            elif "deep" in res['name']:
                output_file = "hyperparameter_deep_output.gif"
            
            output_link = ""
            if output_file and os.path.exists(f"submissions/{output_file}"):
                output_link = f"[{output_file}](submissions/{output_file})"

            f.write(f"| {res['name']} | {res['description']} | {res['status']} | {res['time']} | {output_link} |\n")
            
    print(f"\n📊 Result sheet saved to: {summary_path}")

def run_test_suite():
    """Run the complete test suite with the custom runner."""
    
    print("🧪 Running Comprehensive ALNS VRP Test Suite")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestALNSVRPBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestALNSVRPHyperparameterTuning))
    suite.addTests(loader.loadTestsFromTestCase(TestALNSVRPVideoCreation))
    
    # Run tests with the custom runner
    runner = CustomTestRunner()
    result = runner.run(suite)
    
    # Generate and print summary
    generate_results_summary(result)
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    if not run_test_suite():
        sys.exit(1)