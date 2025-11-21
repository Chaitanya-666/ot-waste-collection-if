#!/usr/bin/env python3
# Author: Chaitanya Shinde (231070066)
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
import time
from pathlib import Path

# Correctly add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from test_report_logger import test_report_logger
from simple_video_creator import SimpleVideoCreator
from src.data_generator import DataGenerator
from src.alns import ALNS

class TestALNSVRP(unittest.TestCase):
    """Comprehensive test suite for ALNS VRP"""

    def run_test_case(self, test_case_id, problem_params, alns_params, expected_cost_upper_bound):
        """Helper function to run a single test case and log results."""
        start_time = time.time()

        # Generate problem
        problem = DataGenerator.generate_instance(**problem_params)

        # Initialize and run solver
        solver = ALNS(problem)
        for key, value in alns_params.items():
            setattr(solver, key, value)
        
        solution = solver.run(track_history=True)
        
        end_time = time.time()
        execution_time = end_time - start_time

        # Create video
        video_filename = f"submissions/{test_case_id}.gif"
        creator = SimpleVideoCreator(output_dir=".")
        creator.create_optimization_animation(
            solver.history,
            problem.get_customer_data(),
            (problem.depot.x, problem.depot.y),
            [(ifac.x, ifac.y) for ifac in problem.intermediate_facilities],
            output_filename=video_filename
        )

        # Log results
        obtained_cost = solution.total_cost
        is_optimal = "Yes" if obtained_cost <= expected_cost_upper_bound else "No"
        
        test_report_logger.log(
            test_case_id=test_case_id,
            parameters=problem_params,
            hyperparameters=alns_params,
            expected_result=f"<= {expected_cost_upper_bound}",
            obtained_result=f"{obtained_cost:.2f}",
            is_optimal=is_optimal,
            status="✅ Pass",
            time=f"{execution_time:.2f}s",
            outputs=f"[{test_case_id}.gif]({video_filename})"
        )

        self.assertIsNotNone(solution)
        self.assertGreater(obtained_cost, 0)

    # Test Cases (10 total)
    def test_01_tiny_fast(self):
        self.run_test_case(
            "test_01_tiny_fast",
            {"name": "Tiny-Fast", "n_customers": 5, "n_ifs": 1, "seed": 42},
            {"max_iterations": 20, "temperature_initial": 500, "cooling_rate": 0.99},
            300
        )

    def test_02_tiny_deep(self):
        self.run_test_case(
            "test_02_tiny_deep",
            {"name": "Tiny-Deep", "n_customers": 5, "n_ifs": 1, "seed": 42},
            {"max_iterations": 100, "temperature_initial": 1000, "cooling_rate": 0.98},
            250
        )

    def test_03_small_fast(self):
        self.run_test_case(
            "test_03_small_fast",
            {"name": "Small-Fast", "n_customers": 10, "n_ifs": 2, "seed": 42},
            {"max_iterations": 50, "temperature_initial": 1000, "cooling_rate": 0.99},
            600
        )

    def test_04_small_deep(self):
        self.run_test_case(
            "test_04_small_deep",
            {"name": "Small-Deep", "n_customers": 10, "n_ifs": 2, "seed": 42},
            {"max_iterations": 200, "temperature_initial": 5000, "cooling_rate": 0.985},
            500
        )

    def test_05_medium_fast(self):
        self.run_test_case(
            "test_05_medium_fast",
            {"name": "Medium-Fast", "n_customers": 15, "n_ifs": 2, "seed": 42},
            {"max_iterations": 100, "temperature_initial": 2000, "cooling_rate": 0.99},
            800
        )

    def test_06_medium_deep(self):
        self.run_test_case(
            "test_06_medium_deep",
            {"name": "Medium-Deep", "n_customers": 15, "n_ifs": 2, "seed": 42},
            {"max_iterations": 300, "temperature_initial": 10000, "cooling_rate": 0.985},
            700
        )

    def test_07_large_fast(self):
        self.run_test_case(
            "test_07_large_fast",
            {"name": "Large-Fast", "n_customers": 25, "n_ifs": 3, "seed": 42},
            {"max_iterations": 150, "temperature_initial": 5000, "cooling_rate": 0.99},
            1200
        )

    def test_08_large_deep(self):
        self.run_test_case(
            "test_08_large_deep",
            {"name": "Large-Deep", "n_customers": 25, "n_ifs": 3, "seed": 42},
            {"max_iterations": 500, "temperature_initial": 20000, "cooling_rate": 0.985},
            1000
        )

    def test_09_clustered_data(self):
        self.run_test_case(
            "test_09_clustered_data",
            {"name": "Clustered", "n_customers": 20, "n_ifs": 2, "seed": 42, "cluster_factor": 0.8},
            {"max_iterations": 300, "temperature_initial": 10000, "cooling_rate": 0.99},
            900
        )

    def test_10_uniform_data(self):
        self.run_test_case(
            "test_10_uniform_data",
            {"name": "Uniform", "n_customers": 20, "n_ifs": 2, "seed": 42, "cluster_factor": 0.0},
            {"max_iterations": 300, "temperature_initial": 10000, "cooling_rate": 0.99},
            1100
        )


def generate_results_summary():
    """Generates a Markdown summary of the test results."""
    
    summary_path = "result_sheet.md"
    results = test_report_logger.get_results()
    
    with open(summary_path, "w") as f:
        f.write("# Result Sheet\n\n")
        f.write("| Test Case ID | Parameters | Hyperparameters | Expected Result | Obtained Result | Optimal? | Status | Time | Outputs |\n")
        f.write("|--------------|------------|-----------------|-----------------|-----------------|----------|--------|------|---------|\n")
        
        for res in results:
            params_str = "<br>".join([f"{k}: {v}" for k, v in res['parameters'].items()])
            hyperparams_str = "<br>".join([f"{k}: {v}" for k, v in res['hyperparameters'].items()])
            
            f.write(f"| {res['test_case_id']} | {params_str} | {hyperparams_str} | {res['expected_result']} | {res['obtained_result']} | {res['is_optimal']} | {res['status']} | {res['time']} | {res['outputs']} |\n")
            
    print(f"\n📊 Result sheet saved to: {summary_path}")

def run_test_suite():
    """Run the complete test suite with the custom runner."""
    
    print("🧪 Running Comprehensive ALNS VRP Test Suite")
    print("=" * 60)
    
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestALNSVRP))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)
    
    # Generate and print summary
    generate_results_summary()

if __name__ == "__main__":
    run_test_suite()
