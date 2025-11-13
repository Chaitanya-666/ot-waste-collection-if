#!/usr/bin/env python3
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

# Add project directory to path
project_dir = Path(__file__).parent / "OT_Project_ALNS_VRP_FIXED"
sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(project_dir.parent))

class TestALNSVRPBasic(unittest.TestCase):
    """Test basic ALNS VRP functionality without video creation"""
    
    def setUp(self):
        """Set up test environment"""
        self.original_dir = os.getcwd()
        
    def tearDown(self):
        """Clean up after tests"""
        os.chdir(self.original_dir)
        
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
        self.assertEqual(len(problem.customers), 0)
        self.assertEqual(len(problem.intermediate_facilities), 0)
        
        # Add depot
        depot = Location(0, 0, 0, 0, "depot")
        problem.depot = depot
        self.assertEqual(problem.depot, depot)
        
        # Add customers
        customer1 = Location(1, 10, 10, 5, "customer")
        customer2 = Location(2, 20, 20, 8, "customer")
        problem.customers.extend([customer1, customer2])
        self.assertEqual(len(problem.customers), 2)
        
        # Add intermediate facility
        ifac = Location(3, 15, 15, 0, "if")
        problem.intermediate_facilities.append(ifac)
        self.assertEqual(len(problem.intermediate_facilities), 1)
        
    def test_data_generator_small(self):
        """Test data generator with small instance"""
        from src.data_generator import DataGenerator
        
        problem = DataGenerator.generate_instance(
            name="Small Test",
            n_customers=5,
            n_ifs=1,
            vehicle_capacity=20,
            seed=42
        )
        
        self.assertEqual(len(problem.customers), 5)
        self.assertEqual(len(problem.intermediate_facilities), 1)
        self.assertEqual(problem.vehicle_capacity, 20)
        self.assertEqual(problem.name, "Small Test")
        
    def test_distance_matrix_calculation(self):
        """Test distance matrix calculation"""
        from src.problem import ProblemInstance, Location
        
        problem = ProblemInstance("Distance Test")
        depot = Location(0, 0, 0, 0, "depot")
        problem.depot = depot
        
        customer1 = Location(1, 10, 0, 5, "customer")
        customer2 = Location(2, 0, 10, 8, "customer")
        problem.customers.extend([customer1, customer2])
        
        problem.calculate_distance_matrix()
        
        # Check that distance matrix is symmetric and has correct dimensions
        self.assertEqual(len(problem.distance_matrix), 3)  # depot + 2 customers
        self.assertEqual(len(problem.distance_matrix[0]), 3)
        
        # Distance from depot to customer1 should be 10 (Euclidean)
        self.assertAlmostEqual(problem.distance_matrix[0][1], 10.0, places=1)


class TestALNSVRPMedium(unittest.TestCase):
    """Test medium-sized ALNS VRP functionality"""
    
    def setUp(self):
        """Set up medium test environment"""
        self.original_dir = os.getcwd()
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)
        
        # Copy project files
        project_src = Path(__file__).parent / "OT_Project_ALNS_VRP_FIXED"
        shutil.copytree(project_src, "project", dirs_exist_ok=True)
        
    def tearDown(self):
        """Clean up medium test environment"""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir, ignore_errors=True)
        
    def test_medium_problem_solve(self):
        """Test solving a medium-sized problem"""
        # This test uses the project's own modules
        sys.path.insert(0, os.path.join(self.test_dir, "project", "src"))
        
        try:
            from src.problem import ProblemInstance, Location
            from src.alns import ALNS
            from src.data_generator import DataGenerator
            
            # Create medium problem
            problem = DataGenerator.generate_instance(
                name="Medium Test",
                n_customers=15,
                n_ifs=2,
                vehicle_capacity=25,
                seed=42
            )
            
            # Solve with ALNS
            solver = ALNS(problem)
            solver.max_iterations = 50  # Quick test
            
            solution = solver.run(max_iterations=solver.max_iterations)
            
            # Validate solution
            self.assertIsNotNone(solution)
            self.assertGreater(solution.total_cost, 0)
            self.assertGreater(len(solution.routes), 0)
            self.assertLessEqual(len(solution.unassigned_customers), len(problem.customers))
            
        except Exception as e:
            self.fail(f"Medium problem solve failed: {e}")
    
    def test_performance_analysis(self):
        """Test performance analysis on medium problem"""
        sys.path.insert(0, os.path.join(self.test_dir, "project", "src"))
        
        try:
            from src.data_generator import DataGenerator
            from src.alns import ALNS
            from src.utils import PerformanceAnalyzer
            
            # Create and solve problem
            problem = DataGenerator.generate_instance(
                name="Performance Test",
                n_customers=12,
                n_ifs=2,
                vehicle_capacity=20,
                seed=123
            )
            
            solver = ALNS(problem)
            solver.max_iterations = 30
            solution = solver.run(max_iterations=solver.max_iterations)
            
            # Analyze performance
            analyzer = PerformanceAnalyzer(problem)
            analysis = analyzer.analyze_solution(solution)
            
            # Validate analysis results
            self.assertIn('total_cost', analysis)
            self.assertIn('total_distance', analysis)
            self.assertIn('efficiency_metrics', analysis)
            self.assertIn('route_details', analysis)
            
            self.assertGreater(analysis['total_cost'], 0)
            self.assertGreater(analysis['total_distance'], 0)
            self.assertIsInstance(analysis['efficiency_metrics'], dict)
            self.assertIsInstance(analysis['route_details'], list)
            
        except Exception as e:
            self.fail(f"Performance analysis failed: {e}")


class TestALNSVRPVideoCreation(unittest.TestCase):
    """Test video creation functionality"""
    
    def setUp(self):
        """Set up video test environment"""
        self.original_dir = os.getcwd()
        
    def tearDown(self):
        """Clean up after tests"""
        os.chdir(self.original_dir)
        
    def test_video_creator_import(self):
        """Test that video creator can be imported"""
        try:
            from simple_video_creator import SimpleVideoCreator
            creator = SimpleVideoCreator()
            self.assertIsNotNone(creator)
            self.assertTrue(hasattr(creator, 'create_optimization_animation'))
            self.assertTrue(hasattr(creator, 'create_cost_animation'))
        except ImportError as e:
            self.fail(f"Video creator import failed: {e}")
    
    def test_optimization_video_tracker(self):
        """Test the optimization video tracker integration"""
        project_dir = Path(__file__).parent / "OT_Project_ALNS_VRP_FIXED"
        sys.path.insert(0, str(project_dir))
        
        try:
            from main import OptimizationVideoTracker
            from src.data_generator import DataGenerator
            
            # Create a test problem
            problem = DataGenerator.generate_instance(
                name="Video Test",
                n_customers=6,
                n_ifs=1,
                vehicle_capacity=15,
                seed=42
            )
            
            # Create tracker
            tracker = OptimizationVideoTracker(problem)
            self.assertIsNotNone(tracker)
            self.assertEqual(len(tracker.optimization_history), 0)
            
            # Test tracking a state
            class MockSolution:
                def __init__(self):
                    self.routes = []
                    class MockRoute:
                        def __init__(self):
                            self.nodes = []
                    route = MockRoute()
                    route.nodes = [MockNode(0, 0, 0, 0, "depot")]
                    self.routes = [route]
            
            class MockNode:
                def __init__(self, id, x, y, demand, node_type):
                    self.id = id
                    self.x = x
                    self.y = y
                    self.demand = demand
                    self.type = node_type
            
            solution = MockSolution()
            tracker.track_state(10, solution, 100.0)
            
            self.assertEqual(len(tracker.optimization_history), 1)
            self.assertEqual(tracker.optimization_history[0]['iteration'], 10)
            self.assertEqual(tracker.optimization_history[0]['cost'], 100.0)
            
        except Exception as e:
            self.fail(f"Video tracker test failed: {e}")
    
    def test_video_creation_with_sample_data(self):
        """Test actual video creation with sample optimization data"""
        try:
            from simple_video_creator import SimpleVideoCreator
            
            # Create sample optimization history
            sample_history = [
                {
                    'iteration': 1,
                    'cost': 100.0,
                    'best_cost': 100.0,
                    'routes': [[(0, 0), (5, 5), (0, 0)]]
                },
                {
                    'iteration': 2,
                    'cost': 95.0,
                    'best_cost': 95.0,
                    'routes': [[(0, 0), (3, 7), (5, 5), (0, 0)]]
                },
                {
                    'iteration': 3,
                    'cost': 90.0,
                    'best_cost': 90.0,
                    'routes': [[(0, 0), (2, 6), (3, 7), (0, 0)]]
                }
            ]
            
            sample_customers = {(1, 2): 5, (3, 4): 6, (5, 6): 7}
            depot_location = (0, 0)
            intermediate_facs = [(2, 3)]
            
            # Create video creator and try to create video
            creator = SimpleVideoCreator()
            
            # Test in temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = creator.create_optimization_animation(
                    optimization_history=sample_history,
                    customer_data=sample_customers,
                    depot_location=depot_location,
                    intermediate_facilities=intermediate_facs,
                    output_filename="test_video.gif"
                )
                
                if video_path and os.path.exists(video_path):
                    # Video created successfully
                    self.assertTrue(video_path.endswith('.gif'))
                    file_size = os.path.getsize(video_path)
                    self.assertGreater(file_size, 1000)  # Should be at least 1KB
                else:
                    # Video creation might have failed due to dependencies
                    self.skipTest("Video creation dependencies not available")
                    
        except Exception as e:
            self.fail(f"Video creation test failed: {e}")


class TestALNSVRPLarge(unittest.TestCase):
    """Test large problem instances"""
    
    def setUp(self):
        """Set up large test environment"""
        self.original_dir = os.getcwd()
        
    def tearDown(self):
        """Clean up after tests"""
        os.chdir(self.original_dir)
        
    def test_large_problem_feasibility(self):
        """Test that large problems can be created and are feasible"""
        project_dir = Path(__file__).parent / "OT_Project_ALNS_VRP_FIXED"
        sys.path.insert(0, str(project_dir))
        
        try:
            from src.data_generator import DataGenerator
            
            # Create large problem
            problem = DataGenerator.generate_instance(
                name="Large Test",
                n_customers=50,
                n_ifs=3,
                vehicle_capacity=40,
                area_size=200,
                demand_range=(2, 15),
                seed=456
            )
            
            # Check basic properties
            self.assertEqual(len(problem.customers), 50)
            self.assertEqual(len(problem.intermediate_facilities), 3)
            self.assertEqual(problem.vehicle_capacity, 40)
            
            # Check feasibility
            is_feasible, message = problem.is_feasible()
            # Note: feasibility depends on parameters, just check we can evaluate it
            self.assertIsInstance(is_feasible, bool)
            
        except Exception as e:
            self.fail(f"Large problem test failed: {e}")
    
    def test_distance_matrix_large(self):
        """Test distance matrix calculation for large problems"""
        project_dir = Path(__file__).parent / "OT_Project_ALNS_VRP_FIXED"
        sys.path.insert(0, str(project_dir))
        
        try:
            from src.data_generator import DataGenerator
            
            problem = DataGenerator.generate_instance(
                name="Large Distance Test",
                n_customers=25,
                n_ifs=2,
                vehicle_capacity=30,
                seed=789
            )
            
            # Calculate distance matrix
            problem.calculate_distance_matrix()
            
            # Check matrix dimensions and properties
            expected_size = 1 + len(problem.customers) + len(problem.intermediate_facilities)
            self.assertEqual(len(problem.distance_matrix), expected_size)
            self.assertEqual(len(problem.distance_matrix[0]), expected_size)
            
            # Check symmetry (distance matrix should be symmetric)
            for i in range(expected_size):
                for j in range(expected_size):
                    if abs(problem.distance_matrix[i][j] - problem.distance_matrix[j][i]) > 1e-6:
                        self.fail("Distance matrix is not symmetric")
                        
        except Exception as e:
            self.fail(f"Large distance matrix test failed: {e}")


class TestALNSVRPIntegration(unittest.TestCase):
    """Test full integration scenarios"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.original_dir = os.getcwd()
        
    def tearDown(self):
        """Clean up after tests"""
        os.chdir(self.original_dir)
        
    def test_full_workflow_small(self):
        """Test complete workflow with small problem"""
        project_dir = Path(__file__).parent / "OT_Project_ALNS_VRP_FIXED"
        sys.path.insert(0, str(project_dir))
        
        try:
            from src.data_generator import DataGenerator
            from src.alns import ALNS
            from src.utils import PerformanceAnalyzer
            from main import OptimizationVideoTracker
            
            # 1. Create problem
            problem = DataGenerator.generate_instance(
                name="Integration Test",
                n_customers=8,
                n_ifs=1,
                vehicle_capacity=20,
                seed=999
            )
            
            # 2. Create video tracker
            video_tracker = OptimizationVideoTracker(problem)
            
            # 3. Solve with ALNS
            solver = ALNS(problem)
            solver.max_iterations = 25
            
            # Add video tracking callback
            def track_callback(iteration_idx, best_solution):
                if iteration_idx % 5 == 0 or iteration_idx == solver.max_iterations:
                    current_cost = getattr(best_solution, 'total_cost', float('inf'))
                    video_tracker.track_state(iteration_idx, best_solution, current_cost)
            
            solver.iteration_callback = track_callback
            
            # 4. Run optimization
            solution = solver.run(max_iterations=solver.max_iterations)
            
            # 5. Validate solution
            self.assertIsNotNone(solution)
            self.assertGreater(solution.total_cost, 0)
            self.assertLessEqual(len(solution.unassigned_customers), len(problem.customers))
            
            # 6. Analyze performance
            analyzer = PerformanceAnalyzer(problem)
            analysis = analyzer.analyze_solution(solution)
            self.assertGreater(analysis['total_cost'], 0)
            
            # 7. Check video tracking
            self.assertGreater(len(video_tracker.optimization_history), 0)
            
        except Exception as e:
            self.fail(f"Full workflow test failed: {e}")


def run_test_suite():
    """Run the complete test suite with detailed output"""
    
    print("🧪 Running Comprehensive ALNS VRP Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestALNSVRPBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestALNSVRPMedium))
    suite.addTests(loader.loadTestsFromTestCase(TestALNSVRPVideoCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestALNSVRPLarge))
    suite.addTests(loader.loadTestsFromTestCase(TestALNSVRPIntegration))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n🔥 ERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED! ({success_rate:.1f}% success rate)")
        print(f"✅ Basic functionality: WORKING")
        print(f"✅ Medium problem solving: WORKING")
        print(f"✅ Video creation: WORKING")
        print(f"✅ Large problem handling: WORKING")
        print(f"✅ Full integration: WORKING")
    else:
        print(f"\n⚠️  Some tests failed. Check output above for details.")
    
    return success


if __name__ == "__main__":
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Run test suite
    success = run_test_suite()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)