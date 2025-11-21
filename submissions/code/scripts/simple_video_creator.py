# Author: Harsh Sharma (231070064)
#
# This file is responsible for creating animated visualizations (GIFs) of the
# optimization process. It takes the history of solutions from the ALNS
# algorithm and uses matplotlib to generate frame-by-frame animations of
# both the route evolution and the cost convergence.
#!/usr/bin/env python3
"""
Simple ALNS Video Creator - Works with basic matplotlib
Creates optimization animations using matplotlib only
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple

# Configure matplotlib for non-interactive mode to prevent plots from displaying
plt.ioff()

class SimpleVideoCreator:
    """Creates animated GIFs of the ALNS optimization process using matplotlib."""
    
    def __init__(self, output_dir: str = "optimization_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_optimization_animation(self, 
                                    optimization_history: List[Dict],
                                    customer_data: Dict,
                                    depot_location: Tuple[float, float],
                                    intermediate_facilities: List[Tuple[float, float]],
                                    output_filename: str = "optimization_process.gif") -> str:
        """
        Creates a GIF animation showing the evolution of routes over iterations.
        
        Args:
            optimization_history: A list of snapshots, where each snapshot is a
                                  dictionary containing the state of the solution
                                  at a particular iteration.
            customer_data: A dictionary mapping customer locations to their demands.
            depot_location: The coordinates of the depot.
            intermediate_facilities: A list of coordinates for the IFs.
            output_filename: The name of the output GIF file.

        Returns:
            The path to the created GIF file, or None on failure.
        """
        
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')

        def animate_frame(frame_num):
            """Draws a single frame of the animation."""
            ax.clear()
            current_state = optimization_history[frame_num]
            
            # Dynamically calculate plot bounds to fit all points
            all_x = [depot_location[0]] + [loc[0] for loc in customer_data.keys()] + [loc[0] for loc in intermediate_facilities]
            all_y = [depot_location[1]] + [loc[1] for loc in customer_data.keys()] + [loc[1] for loc in intermediate_facilities]
            if 'routes' in current_state:
                for route in current_state['routes']:
                    for point in route:
                        all_x.append(point[0])
                        all_y.append(point[1])
            
            # Add padding to the plot
            padding_x = max(2.0, (max(all_x) - min(all_x)) * 0.1)
            padding_y = max(2.0, (max(all_y) - min(all_y)) * 0.1)
            ax.set_xlim(min(all_x) - padding_x, max(all_x) + padding_x)
            ax.set_ylim(min(all_y) - padding_y, max(all_y) + padding_y)
            
            # Plot depot, IFs, and customers
            ax.scatter(depot_location[0], depot_location[1], c='red', s=200, marker='s', label='Depot', zorder=10, edgecolor='black')
            for i, if_loc in enumerate(intermediate_facilities):
                ax.scatter(if_loc[0], if_loc[1], c='orange', s=150, marker='^', label='IF' if i == 0 else "", zorder=8, edgecolor='black')
            for i, (loc, demand) in enumerate(customer_data.items()):
                ax.scatter(loc[0], loc[1], c='blue', s=100, marker='o', label='Customer' if i == 0 else "", zorder=7, alpha=0.7)
                ax.annotate(f'C{i+1}\n({demand})', (loc[0], loc[1]), xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Plot the routes for the current iteration
            if 'routes' in current_state:
                colors = ['green', 'purple', 'brown', 'pink', 'gray', 'olive']
                for route_idx, route in enumerate(current_state['routes']):
                    if len(route) > 1:
                        x_coords, y_coords = zip(*route)
                        ax.plot(x_coords, y_coords, color=colors[route_idx % len(colors)], linewidth=3, alpha=0.7, label=f'Route {route_idx + 1}')
            
            # Display metrics for the current iteration
            metrics_text = f"Iteration: {frame_num + 1}/{len(optimization_history)}\n" \
                           f"Current Cost: {current_state.get('cost', 'N/A'):.2f}\n" \
                           f"Best Cost: {current_state.get('best_cost', 'N/A'):.2f}"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_title(f'ALNS Optimization Process (Frame {frame_num + 1})')
        
        # Create and save the animation
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(optimization_history), interval=1000, repeat=True)
        output_path = self.output_dir / output_filename
        print(f"🎬 Creating animation: {output_path}")
        
        try:
            anim.save(str(output_path), writer='pillow', fps=1)
            print(f"✅ Animation saved: {output_path}")
        except Exception as e:
            print(f"❌ Error creating animation: {e}")
            return None
        
        plt.close(fig)
        return str(output_path)
    
    def create_cost_animation(self, costs: List[float], output_filename: str = "cost_convergence.gif") -> str:
        """Creates a GIF animation showing the convergence of the cost function."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')

        def animate_cost(frame):
            """Draws a single frame of the cost convergence animation."""
            ax.clear()
            
            # Plot the cost curve up to the current frame
            ax.plot(range(frame + 1), costs[:frame + 1], 'b-', linewidth=2, label='Total Cost')
            ax.scatter(frame, costs[frame], c='green', s=100, marker='*', zorder=10, label=f'Current: {costs[frame]:.2f}')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Total Cost')
            ax.set_title(f'Cost Convergence (Iteration {frame + 1}/{len(costs)})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Dynamically adjust y-axis limits
            if costs:
                cost_min, cost_max = min(costs), max(costs)
                padding = max((cost_max - cost_min) * 0.1, 1)
                ax.set_ylim(cost_min - padding, cost_max + padding)
        
        # Create and save the animation
        anim = animation.FuncAnimation(fig, animate_cost, frames=len(costs), interval=800, repeat=True)
        output_path = self.output_dir / output_filename
        print(f"📊 Creating cost convergence animation: {output_path}")
        
        try:
            anim.save(str(output_path), writer='pillow', fps=1.5)
            print(f"✅ Cost animation saved: {output_path}")
        except Exception as e:
            print(f"❌ Error creating cost animation: {e}")
        
        plt.close(fig)
        return str(output_path)


if __name__ == "__main__":
    # This block runs a demonstration of the video creator when the script is executed directly.
    print("🎬 Simple ALNS Video Creator Demo")
    print("="*50)
    
    # Create sample data for the demo
    depot = (0, 0)
    customers = {(3, 15): 4, (7, 12): 4, (5, 8): 5, (12, 10): 7}
    intermediate_facs = [(10, 10)]
    
    # Generate a dummy optimization history
    history = [{'iteration': i, 'cost': 80 - i*2, 'best_cost': 80 - i*2, 'routes': [[(0,0), (3,15), (0,0)]]} for i in range(15)]
    costs = [h['cost'] for h in history]
    
    # Initialize and run the creator
    creator = SimpleVideoCreator()
    creator.create_optimization_animation(history, customers, depot, intermediate_facs, "demo_alns_routes.gif")
    creator.create_cost_animation(costs, "demo_cost_convergence.gif")
    
    print(f"\n🎉 Demo completed! Check the '{creator.output_dir}' directory.")