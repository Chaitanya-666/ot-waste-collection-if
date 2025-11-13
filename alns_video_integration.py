#!/usr/bin/env python3
"""
ALNS Integration Example
Shows how to modify your existing ALNS algorithm to track optimization history
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Any
from datetime import datetime

class ALNSWithVideoTracking:
    """Modified ALNS algorithm with optimization history tracking"""
    
    def __init__(self, problem_data: Dict):
        self.problem_data = problem_data
        self.optimization_history = []
        self.current_best_cost = float('inf')
        
    def run_alns(self, max_iterations: int = 100) -> Dict:
        """Run ALNS with video tracking"""
        
        print(f"🎬 Starting ALNS with video tracking ({max_iterations} iterations)")
        
        # Initialize solution
        current_solution = self._initialize_solution()
        current_cost = self._calculate_cost(current_solution)
        
        # Track initial state
        self._track_optimization_state(0, current_solution, current_cost)
        
        for iteration in range(max_iterations):
            # Your existing ALNS logic here
            # For demo, we'll simulate the process
            
            # Simulate improvement (replace with your actual ALNS operations)
            if np.random.random() < 0.3:  # 30% chance of improvement
                # Simulate getting better
                current_cost *= (0.95 + 0.05 * np.random.random())
                current_solution = self._simulate_route_update(current_solution)
            
            # Update best solution
            if current_cost < self.current_best_cost:
                self.current_best_cost = current_cost
            
            # Track state every few iterations (to avoid too many frames)
            if iteration % 5 == 0 or iteration == max_iterations - 1:
                self._track_optimization_state(iteration + 1, current_solution, current_cost)
            
            # Progress update
            if (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration + 1}: Cost = {current_cost:.2f}, Best = {self.current_best_cost:.2f}")
        
        final_solution = {
            'cost': current_cost,
            'solution': current_solution,
            'history': self.optimization_history
        }
        
        print(f"✅ ALNS completed! Final cost: {current_cost:.2f}")
        return final_solution
    
    def _track_optimization_state(self, iteration: int, solution: Any, cost: float):
        """Track optimization state for video creation"""
        
        # Extract route information for visualization
        routes_data = self._extract_routes_for_visualization(solution)
        
        state = {
            'iteration': iteration,
            'cost': cost,
            'best_cost': self.current_best_cost,
            'routes': routes_data,
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_history.append(state)
    
    def _extract_routes_for_visualization(self, solution) -> List[List[Tuple[float, float]]]:
        """Extract route coordinates for video visualization"""
        
        # This should convert your solution format to coordinate lists
        # Replace this with your actual solution extraction logic
        
        # Sample extraction (replace with your actual logic)
        sample_routes = [
            [(0, 0), (3, 15), (7, 12), (0, 0)],  # Route 1
            [(0, 0), (15, 5), (18, 15), (0, 0)]  # Route 2
        ]
        
        return sample_routes
    
    def _initialize_solution(self):
        """Initialize solution (replace with your logic)"""
        # Placeholder - replace with your solution initialization
        return {'routes': [], 'cost': 0}
    
    def _calculate_cost(self, solution) -> float:
        """Calculate solution cost (replace with your logic)"""
        # Placeholder - replace with your cost calculation
        return 100.0
    
    def _simulate_route_update(self, solution):
        """Simulate route updates (replace with your ALNS operations)"""
        # Placeholder - replace with your actual ALNS operators
        return solution
    
    def save_optimization_history(self, filename: str = "optimization_history.json"):
        """Save optimization history for later video creation"""
        
        with open(filename, 'w') as f:
            json.dump(self.optimization_history, f, indent=2)
        
        print(f"💾 Optimization history saved to: {filename}")
        return filename
    
    def create_video_from_history(self, customer_data: Dict, output_dir: str = "videos"):
        """Create video from saved optimization history"""
        
        # Import the video creator
        import sys
        sys.path.append('/workspace')
        from optimization_video_creator import OptimizationVideoCreator
        
        # Create video
        creator = OptimizationVideoCreator(output_dir)
        
        depot = (0, 0)
        intermediate_facs = [(10, 10)]
        
        video_path = creator.create_optimization_animation(
            self.optimization_history,
            customer_data,
            depot,
            intermediate_facs,
            "alns_routes_animation.mp4"
        )
        
        return video_path


def demo_with_your_algorithm():
    """Demonstration of how to integrate with your existing ALNS"""
    
    print("🚀 ALNS Video Integration Demo")
    print("="*50)
    
    # Sample problem data
    problem_data = {
        'customers': {
            (3, 15): 4, (7, 12): 4, (5, 8): 5, (12, 10): 7,
            (15, 5): 6, (18, 15): 3, (8, 3): 5, (14, 8): 6
        },
        'depot': (0, 0),
        'intermediate_facilities': [(10, 10)]
    }
    
    # Initialize ALNS with tracking
    alns = ALNSWithVideoTracking(problem_data)
    
    # Run optimization
    final_solution = alns.run_alns(max_iterations=50)
    
    # Save history for video creation
    history_file = alns.save_optimization_history("my_optimization_history.json")
    
    # Create video
    print("\n🎬 Creating optimization video...")
    video_path = alns.create_video_from_history(problem_data['customers'])
    
    print(f"\n🎉 Video created successfully!")
    print(f"📹 Video saved to: {video_path}")
    
    return final_solution, video_path


if __name__ == "__main__":
    # Run demo
    final_solution, video_path = demo_with_your_algorithm()