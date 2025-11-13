#!/usr/bin/env python3
"""
Quick verification script for the ALNS VRP project
This script performs basic verification without running heavy tests
"""

import sys
import os
import time
from datetime import datetime

# Ensure the project's `src` directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def test_imports():
    """Test that all imports work correctly"""
    print("=" * 60)
    print("IMPORT VERIFICATION")
    print("=" * 60)
    
    try:
        from src.problem import ProblemInstance, Location
        print("✅ Problem imports successful")
        
        from src.solution import Solution, Route
        print("✅ Solution imports successful")
        
        from src.alns import ALNS
        print("✅ ALNS imports successful")
        
        from src.destroy_operators import RandomRemoval, WorstRemoval
        print("✅ Destroy operators imports successful")
        
        from src.repair_operators import GreedyInsertion, RegretInsertion
        print("✅ Repair operators imports successful")
        
        from src.data_generator import DataGenerator
        print("✅ Data generator imports successful")
        
        from src.utils import RouteVisualizer, PerformanceAnalyzer
        print("✅ Utils imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic project functionality"""
    print("\n" + "=" * 60)
    print("BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    try:
        from src.data_generator import DataGenerator
        from src.alns import ALNS
        from src.utils import PerformanceAnalyzer
        
        # Create a small test instance
        problem = DataGenerator.generate_instance("Quick Test", 5, 1, seed=42)
        
        print(f"✅ Generated problem: {len(problem.customers)} customers, {problem.vehicle_capacity} capacity")
        
        # Ensure vehicle constraints are set properly
        min_needed = int(problem.get_min_vehicles_needed())
        # Set vehicle limit higher than minimum to ensure feasibility
        problem.number_of_vehicles = max(min_needed + 1, problem.number_of_vehicles)
        print(f"✅ Set vehicle limit to: {problem.number_of_vehicles} (minimum: {min_needed})")
        
        # Run ALNS for a few iterations
        solver = ALNS(problem)
        solver.max_iterations = 30  # Very short run for verification
        
        start_time = time.time()
        solution = solver.run()
        end_time = time.time()
        
        print(f"✅ ALNS completed in {end_time - start_time:.2f} seconds")
        print(f"✅ Solution cost: {solution.total_cost:.2f}")
        print(f"✅ Routes used: {len(solution.routes)}")
        print(f"✅ Unassigned customers: {len(solution.unassigned_customers)}")
        
        # Test efficiency calculation
        analyzer = PerformanceAnalyzer(problem)
        analysis = analyzer.analyze_solution(solution)
        metrics = analysis["efficiency_metrics"]
        
        print(f"✅ Vehicle efficiency: {metrics['vehicle_efficiency']:.1%}")
        print(f"✅ Capacity utilization: {metrics['capacity_utilization']:.1%}")
        
        # Check that efficiency metrics are reasonable (not > 100%)
        if metrics['vehicle_efficiency'] > 1.0:
            print("❌ Vehicle efficiency exceeds 100% - potential bug")
            return False
        if metrics['capacity_utilization'] > 1.0:
            print("❌ Capacity utilization exceeds 100% - potential bug")  
            return False
            
        print("✅ All efficiency metrics are reasonable")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def test_vehicle_constraints():
    """Test that vehicle constraints are properly enforced"""
    print("\n" + "=" * 60)
    print("VEHICLE CONSTRAINT TEST")
    print("=" * 60)
    
    try:
        from src.data_generator import DataGenerator
        from src.alns import ALNS
        
        # Create a larger problem that requires multiple vehicles
        problem = DataGenerator.generate_instance("Multi-Vehicle Test", 15, 2, seed=42)
        
        min_needed = int(problem.get_min_vehicles_needed())
        print(f"✅ Problem requires minimum {min_needed} vehicles")
        
        # Set vehicle limit slightly higher than minimum for feasibility
        problem.number_of_vehicles = min_needed + 2
        print(f"✅ Set vehicle limit to {problem.number_of_vehicles} (minimum: {min_needed})")
        
        # Solve
        solver = ALNS(problem)
        solver.max_iterations = 50
        
        solution = solver.run()
        
        print(f"✅ Solution uses {len(solution.routes)} vehicles")
        
        # Check that solution doesn't exceed vehicle limit
        if len(solution.routes) > problem.number_of_vehicles:
            print(f"❌ Solution uses {len(solution.routes)} vehicles, exceeds limit of {problem.number_of_vehicles}")
            return False
        else:
            print("✅ Vehicle constraint respected")
            
        # Check feasibility
        feasible, message = solution.is_feasible(problem)
        if not feasible:
            print(f"❌ Solution is not feasible: {message}")
            return False
        else:
            print("✅ Solution is feasible")
            
        return True
        
    except Exception as e:
        print(f"❌ Vehicle constraint test failed: {e}")
        return False

def main():
    """Run all verification tests"""
    print("ALNS VRP PROJECT - QUICK VERIFICATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Vehicle Constraints", test_vehicle_constraints),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            failed += 1
            print(f"❌ {test_name} FAILED")
    
    # Summary
    total_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/(passed+failed)*100:.1f}%")
    print(f"Completed: {total_time}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Project is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} TEST(S) FAILED. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)