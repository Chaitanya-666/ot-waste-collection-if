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

# Configure matplotlib for non-interactive mode
plt.ioff()

class SimpleVideoCreator:
    """Creates animated GIFs of optimization processes using matplotlib"""
    
    def __init__(self, output_dir: str = "optimization_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_optimization_animation(self, 
                                    optimization_history: List[Dict],
                                    customer_data: Dict,
                                    depot_location: Tuple[float, float],
                                    intermediate_facilities: List[Tuple[float, float]],
                                    output_filename: str = "optimization_process.gif") -> str:
        """Create GIF animation of optimization process"""
        
        def animate_frame(frame_num):
            ax.clear()
            current_state = optimization_history[frame_num]
            
            # Calculate dynamic scale based on all points
            all_x = [depot_location[0]]
            all_y = [depot_location[1]]
            
            # Add customer locations
            for loc, demand in customer_data.items():
                all_x.append(loc[0])
                all_y.append(loc[1])
            
            # Add intermediate facilities
            for if_loc in intermediate_facilities:
                all_x.append(if_loc[0])
                all_y.append(if_loc[1])
            
            # Add route points if available
            if 'routes' in current_state:
                for route in current_state['routes']:
                    for point in route:
                        all_x.append(point[0])
                        all_y.append(point[1])
            
            # Calculate dynamic bounds with padding
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            
            # Add padding (10% of range, minimum 2 units)
            x_range = max_x - min_x
            y_range = max_y - min_y
            padding_x = max(2.0, x_range * 0.1)
            padding_y = max(2.0, y_range * 0.1)
            
            plot_min_x = min_x - padding_x
            plot_max_x = max_x + padding_x
            plot_min_y = min_y - padding_y
            plot_max_y = max_y + padding_y
            
            # Plot depot
            ax.scatter(depot_location[0], depot_location[1], 
                      c='red', s=200, marker='s', label='Depot', 
                      zorder=10, edgecolor='black', linewidth=2)
            
            # Plot intermediate facilities
            for i, if_loc in enumerate(intermediate_facilities):
                ax.scatter(if_loc[0], if_loc[1], c='orange', s=150, 
                          marker='^', label='IF' if i == 0 else "", 
                          zorder=8, edgecolor='black')
            
            # Plot customers
            for i, (loc, demand) in enumerate(customer_data.items()):
                ax.scatter(loc[0], loc[1], c='blue', s=100, marker='o', 
                          label='Customer' if i == 0 else "", 
                          zorder=7, alpha=0.7)
                ax.annotate(f'C{i+1}\n({demand})', (loc[0], loc[1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, ha='left')
            
            # Plot routes
            if 'routes' in current_state:
                colors = ['green', 'purple', 'brown', 'pink', 'gray', 'olive']
                for route_idx, route in enumerate(current_state['routes']):
                    if len(route) > 1:
                        color = colors[route_idx % len(colors)]
                        x_coords = [point[0] for point in route]
                        y_coords = [point[1] for point in route]
                        
                        ax.plot(x_coords, y_coords, color=color, 
                               linewidth=3, alpha=0.7, 
                               label=f'Route {route_idx + 1}')
            
            # Add metrics text
            metrics_text = f"""
            Iteration: {frame_num + 1}/{len(optimization_history)}
            Current Cost: {current_state.get('cost', 'N/A'):.2f}
            Routes: {len(current_state.get('routes', []))}
            Best Cost: {current_state.get('best_cost', 'N/A'):.2f}
            """
            
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Set limits and formatting using dynamic bounds
            ax.set_xlim(plot_min_x, plot_max_x)
            ax.set_ylim(plot_min_y, plot_max_y)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_title(f'ALNS Optimization Process (Frame {frame_num + 1})')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        # Create animation
        frames = len(optimization_history)
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=frames, 
            interval=1000, repeat=True
        )
        
        # Save as GIF
        output_path = self.output_dir / output_filename
        print(f"🎬 Creating animation: {output_path}")
        
        try:
            # Save as GIF
            anim.save(str(output_path), writer='pillow', fps=1)
            print(f"✅ Animation saved: {output_path}")
            
            # Also create individual frames for inspection
            frame_dir = self.output_dir / "frames"
            frame_dir.mkdir(exist_ok=True)
            
            for i, state in enumerate(optimization_history):
                fig_frame, ax_frame = plt.subplots(figsize=(12, 8))
                fig_frame.patch.set_facecolor('white')
                
                animate_frame(i)
                
                frame_path = frame_dir / f"frame_{i:03d}.png"
                fig_frame.savefig(frame_path, dpi=100, bbox_inches='tight')
                plt.close(fig_frame)
            
            print(f"📸 Individual frames saved in: {frame_dir}")
            
        except Exception as e:
            print(f"❌ Error creating animation: {e}")
            return None
        
        plt.close(fig)
        return str(output_path)
    
    def create_cost_animation(self, costs: List[float], output_filename: str = "cost_convergence.gif") -> str:
        """Create cost convergence animation"""
        
        def animate_cost(frame):
            ax.clear()
            
            # Plot cost curve up to current frame
            iteration_range = range(frame + 1)
            cost_subset = costs[:frame + 1]
            
            ax.plot(iteration_range, cost_subset, 'b-', linewidth=2, label='Total Cost')
            ax.scatter(iteration_range, cost_subset, c='red', s=30, zorder=5)
            
            # Highlight current point
            if frame > 0:
                ax.scatter([frame], [costs[frame]], c='green', s=100, 
                          marker='*', zorder=10, label=f'Current: {costs[frame]:.2f}')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Total Cost')
            ax.set_title(f'Cost Convergence (Iteration {frame + 1}/{len(costs)})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set y-axis limits with dynamic padding
            if cost_subset:
                cost_min, cost_max = min(cost_subset), max(cost_subset)
                if cost_max > cost_min:
                    # Use 10% padding, minimum 5% of range
                    padding = max((cost_max - cost_min) * 0.1, (cost_max - cost_min) * 0.05)
                else:
                    # If all costs are the same, use a small range
                    padding = cost_max * 0.1 if cost_max > 0 else 1
                ax.set_ylim(cost_min - padding, cost_max + padding)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        anim = animation.FuncAnimation(
            fig, animate_cost, frames=len(costs),
            interval=800, repeat=True
        )
        
        output_path = self.output_dir / output_filename
        print(f"📊 Creating cost convergence animation: {output_path}")
        
        try:
            anim.save(str(output_path), writer='pillow', fps=1.5)
            print(f"✅ Cost animation saved: {output_path}")
        except Exception as e:
            print(f"❌ Error creating cost animation: {e}")
        
        plt.close(fig)
        return str(output_path)


def create_demo_history() -> List[Dict]:
    """Create demonstration optimization history"""
    
    # Sample problem data
    depot = (0, 0)
    customers = {
        (3, 15): 4, (7, 12): 4, (5, 8): 5, (12, 10): 7,
        (15, 5): 6, (18, 15): 3, (8, 3): 5, (14, 8): 6
    }
    
    history = []
    base_cost = 80
    
    for i in range(15):  # 15 frames for demo
        # Simulate optimization progress
        improvement = i * 2 + np.random.normal(0, 1)
        cost = max(42.67, base_cost - improvement)
        
        # Create evolving routes
        if i < 5:
            # Early iterations - poor routes
            routes = [
                [(0, 0), (3, 15), (7, 12), (0, 0)],
                [(0, 0), (5, 8), (12, 10), (0, 0)]
            ]
        elif i < 10:
            # Middle iterations - improving routes  
            routes = [
                [(0, 0), (3, 15), (5, 8), (12, 10), (0, 0)],
                [(0, 0), (7, 12), (15, 5), (0, 0)]
            ]
        else:
            # Final iterations - optimized routes
            routes = [
                [(0, 0), (3, 15), (5, 8), (7, 12), (12, 10), (0, 0)],
                [(0, 0), (15, 5), (18, 15), (14, 8), (0, 0)]
            ]
        
        history.append({
            'iteration': i + 1,
            'cost': cost,
            'best_cost': min([h['cost'] for h in history] + [cost]),
            'routes': routes
        })
    
    return history


if __name__ == "__main__":
    print("🎬 Simple ALNS Video Creator Demo")
    print("="*50)
    
    # Create sample data
    depot = (0, 0)
    customers = {
        (3, 15): 4, (7, 12): 4, (5, 8): 5, (12, 10): 7,
        (15, 5): 6, (18, 15): 3, (8, 3): 5, (14, 8): 6
    }
    intermediate_facs = [(10, 10)]
    
    # Generate demo history
    optimization_history = create_demo_history()
    costs = [state['cost'] for state in optimization_history]
    
    # Initialize creator
    creator = SimpleVideoCreator()
    
    # Create route animation
    print("\n1️⃣ Creating route optimization animation...")
    route_video = creator.create_optimization_animation(
        optimization_history, customers, depot, intermediate_facs,
        "demo_alns_routes.gif"
    )
    
    # Create cost animation
    print("\n2️⃣ Creating cost convergence animation...")
    cost_video = creator.create_cost_animation(costs, "demo_cost_convergence.gif")
    
    print(f"\n🎉 Demo completed!")
    print(f"📁 Output directory: {creator.output_dir}")
    
    if route_video:
        print(f"🎬 Route animation: {Path(route_video).name}")
    if cost_video:
        print(f"📊 Cost animation: {Path(cost_video).name}")
    
    print(f"\n💡 These GIFs show how your ALNS algorithm would evolve:")
    print(f"   • Route structure changes over iterations")
    print(f"   • Cost improvement visualization")
    print(f"   • Perfect for presentations and analysis!")
    
    # List created files
    print(f"\n📋 Files created:")
    for file in creator.output_dir.glob("*"):
        size = file.stat().st_size / 1024  # Size in KB
        print(f"   📄 {file.name} ({size:.1f} KB)")