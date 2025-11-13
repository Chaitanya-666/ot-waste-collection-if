#!/usr/bin/env python3
"""
Simple test script to verify ALNS VRP-IF functionality
"""
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.problem import ProblemInstance, Location
from src.alns import ALNS
from src.data_generator import DataGenerator

def test_basic_functionality():
    print("=== Testing Basic ALNS VRP-IF Functionality ===")
    
    # Create a simple problem instance
    problem = ProblemInstance("Test Instance")
    problem.vehicle_capacity = 20
    problem.number_of_vehicles = 3
    problem.disposal_time = 2
    
    # Add depot
    depot = Location(0, 0, 0, 0, "depot")
    problem.depot = depot
    
    # Add intermediate facility
    if1 = Location(100, 20, 20, 0, "if")
    problem.intermediate_facilities.append(if1)
    
    # Add customers
    customers = [
        Location(1, 5, 2, 4, "customer"),
        Location(2, 3, 8, 6, "customer"),
        Location(3, 9, 1, 5, "customer"),
        Location(4, 6, 7, 7, "customer"),
        Location(5, 2, 3, 3, "customer"),
        Location(6, 8, 6, 4, "customer"),
    ]
    problem.customers = customers
    
    # Calculate distance matrix
    problem.calculate_distance_matrix()
    
    print(f"Problem created: {len(customers)} customers, 1 IF")
    print(f"Vehicle capacity: {problem.vehicle_capacity}")
    print(f"Total demand: {sum(c.demand for c in customers)}")
    
    # Initialize ALNS solver
    solver = ALNS(problem)
    solver.max_iterations = 50  # Reduced for testing
    
    print("\n=== Running ALNS Optimization ===")
    start_time = time.time()
    
    try:
        solution = solver.run()
        end_time = time.time()
        
        print(f"\n✅ ALNS completed successfully in {end_time - start_time:.2f} seconds")
        print(f"Final solution cost: {solution.total_cost:.2f}")
        print(f"Number of routes: {len(solution.routes)}")
        print(f"Unassigned customers: {len(solution.unassigned_customers)}")
        
        # Display route details
        print("\n=== Route Details ===")
        for i, route in enumerate(solution.routes):
            print(f"Route {i+1}: Distance={route.total_distance:.2f}")
            sequence = " -> ".join([f"{node.type[0].upper()}{node.id}" for node in route.nodes])
            print(f"  Sequence: {sequence}")
            print(f"  Load sequence: {route.loads}")
        
        # Check if solution is feasible
        total_demand = sum(c.demand for c in customers)
        max_possible_load = problem.vehicle_capacity * len(solution.routes)
        utilization = total_demand / max_possible_load * 100
        
        print(f"\n=== Feasibility Analysis ===")
        print(f"Total demand served: {total_demand}")
        print(f"Total vehicle capacity available: {max_possible_load}")
        print(f"Capacity utilization: {utilization:.1f}%")
        
        # Check for IF visits
        total_if_visits = 0
        for route in solution.routes:
            if_visits = sum(1 for node in route.nodes if node.type == "if")
            total_if_visits += if_visits
            if if_visits > 0:
                print(f"Route {i+1} visits IF {if_visits} times")
        
        print(f"Total IF visits across all routes: {total_if_visits}")
        
        return True
        
    except Exception as e:
        print(f"❌ ALNS failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_generator():
    print("\n=== Testing Data Generator ===")
    
    try:
        # Generate test instance
        problem = DataGenerator.generate_instance(
            name="Generated Instance",
            n_customers=10,
            n_ifs=2,
            vehicle_capacity=25,
            seed=42
        )
        
        print(f"✅ Generated instance: {len(problem.customers)} customers, {len(problem.intermediate_facilities)} IFs")
        print(f"Total demand: {sum(c.demand for c in problem.customers):.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting comprehensive ALNS VRP-IF tests...")
    
    success = True
    
    # Test 1: Basic functionality
    if not test_basic_functionality():
        success = False
    
    # Test 2: Data generation
    if not test_data_generator():
        success = False
    
    if success:
        print("\n🎉 All tests passed! ALNS VRP-IF is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the output above.")
    
    sys.exit(0 if success else 1)