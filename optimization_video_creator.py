#!/usr/bin/env python3
"""
Optimization Process Video Creator
Creates animated videos showing ALNS optimization progress
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import json
import os
from pathlib import Path
import imageio
from typing import List, Dict, Tuple, Optional

class OptimizationVideoCreator:
    """Creates animated videos of optimization processes"""
    
    def __init__(self, output_dir: str = "optimization_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Video settings
        self.fps = 2  # 2 frames per second for optimization (slower to see changes)
        self.dpi = 150
        
        # Set up matplotlib for non-interactive mode
        plt.ioff()  # Turn off interactive mode
        
    def create_optimization_animation(self, 
                                    optimization_history: List[Dict],
                                    customer_data: Dict,
                                    depot_location: Tuple[float, float],
                                    intermediate_facilities: List[Tuple[float, float]],
                                    output_filename: str = "optimization_process.mp4") -> str:
        """
        Create MP4 animation of optimization process
        
        Args:
            optimization_history: List of dicts with cost, routes, iteration info
            customer_data: Customer locations and demands
            depot_location: Depot coordinates
            intermediate_facilities: IF locations
            output_filename: Output video filename
        """
        
        def animate_frame(frame_num):
            """Animation function called for each frame"""
            ax.clear()
            current_state = optimization_history[frame_num]
            
            # Plot all points
            self._plot_locations(ax, depot_location, intermediate_facilities, customer_data)
            
            # Plot current routes
            if 'routes' in current_state:
                self._plot_routes(ax, current_state['routes'], frame_num)
            
            # Plot title and metrics
            self._plot_metrics(ax, current_state, frame_num, len(optimization_history))
            
            # Set equal aspect ratio and limits
            self._set_plot_limits(ax)
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        
        # Create animation
        frames = len(optimization_history)
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=frames, 
            interval=1000/self.fps, blit=False, repeat=True
        )
        
        # Save as MP4
        output_path = self.output_dir / output_filename
        print(f"Creating video: {output_path}")
        
        try:
            # Try saving with ffmpeg first
            anim.save(str(output_path), writer='ffmpeg', fps=self.fps, bitrate=1800)
            print(f"✅ Video saved successfully: {output_path}")
        except Exception as e:
            print(f"⚠️ FFmpeg not available: {e}")
            print("Creating GIF instead...")
            
            # Fallback to GIF
            gif_path = output_path.with_suffix('.gif')
            anim.save(str(gif_path), writer='pillow', fps=self.fps)
            print(f"✅ GIF saved: {gif_path}")
            return str(gif_path)
        
        plt.close(fig)
        return str(output_path)
    
    def create_cost_convergence_video(self, 
                                    costs: List[float],
                                    output_filename: str = "cost_convergence.mp4") -> str:
        """Create video showing cost improvement over iterations"""
        
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
            
            # Set y-axis limits with some padding
            cost_min, cost_max = min(costs), max(costs)
            padding = (cost_max - cost_min) * 0.1
            ax.set_ylim(cost_min - padding, cost_max + padding)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        
        anim = animation.FuncAnimation(
            fig, animate_cost, frames=len(costs),
            interval=500, repeat=True
        )
        
        output_path = self.output_dir / output_filename
        print(f"Creating cost convergence video: {output_path}")
        
        try:
            anim.save(str(output_path), writer='ffmpeg', fps=4)
            print(f"✅ Cost convergence video saved: {output_path}")
        except Exception as e:
            print(f"⚠️ FFmpeg not available, creating GIF: {e}")
            gif_path = output_path.with_suffix('.gif')
            anim.save(str(gif_path), writer='pillow', fps=4)
            print(f"✅ Cost convergence GIF saved: {gif_path}")
            return str(gif_path)
        
        plt.close(fig)
        return str(output_path)
    
    def create_side_by_side_video(self, 
                                optimization_history: List[Dict],
                                customer_data: Dict,
                                depot_location: Tuple[float, float],
                                intermediate_facilities: List[Tuple[float, float]],
                                output_filename: str = "optimization_comparison.mp4") -> str:
        """Create video with routes and cost on side-by-side plots"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor('white')
        
        def animate_dual(frame):
            current_state = optimization_history[frame]
            
            # Left plot: Route visualization
            ax1.clear()
            self._plot_locations(ax1, depot_location, intermediate_facilities, customer_data)
            if 'routes' in current_state:
                self._plot_routes(ax1, current_state['routes'], frame)
            self._plot_metrics(ax1, current_state, frame, len(optimization_history))
            self._set_plot_limits(ax1)
            ax1.set_title(f'Route Visualization (Iter {frame + 1})')
            
            # Right plot: Cost convergence up to current frame
            ax2.clear()
            costs = [state.get('cost', 0) for state in optimization_history[:frame + 1]]
            iterations = list(range(len(costs)))
            
            ax2.plot(iterations, costs, 'b-', linewidth=2)
            ax2.scatter(iterations, costs, c='red', s=20, zorder=5)
            if frame > 0:
                ax2.scatter([frame], [costs[-1]], c='green', s=100, 
                           marker='*', zorder=10)
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Total Cost')
            ax2.set_title(f'Cost Convergence')
            ax2.grid(True, alpha=0.3)
            
            # Add current cost annotation
            if costs:
                ax2.annotate(f'Current: {costs[-1]:.2f}', 
                           xy=(frame, costs[-1]), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        anim = animation.FuncAnimation(
            fig, animate_dual, frames=len(optimization_history),
            interval=1000, repeat=True
        )
        
        output_path = self.output_dir / output_filename
        print(f"Creating side-by-side video: {output_path}")
        
        try:
            anim.save(str(output_path), writer='ffmpeg', fps=1.5)
            print(f"✅ Side-by-side video saved: {output_path}")
        except Exception as ffmpeg_error:
            print(f"⚠️ FFmpeg not available: {ffmpeg_error}")
            gif_path = output_path.with_suffix('.gif')
            anim.save(str(gif_path), writer='pillow', fps=1.5)
            print(f"✅ Side-by-side GIF saved: {gif_path}")
            return str(gif_path)
        
        plt.close(fig)
        return str(output_path)
    
    def _plot_locations(self, ax, depot, intermediate_facs, customers):
        """Plot depot, IFs, and customers"""
        
        # Plot depot
        ax.scatter(depot[0], depot[1], c='red', s=200, marker='s', 
                  label='Depot', zorder=10, edgecolor='black', linewidth=2)
        
        # Plot intermediate facilities
        for i, if_loc in enumerate(intermediate_facs):
            ax.scatter(if_loc[0], if_loc[1], c='orange', s=150, marker='^', 
                      label='IF' if i == 0 else "", zorder=8, edgecolor='black')
        
        # Plot customers
        for i, (loc, demand) in enumerate(customers.items()):
            ax.scatter(loc[0], loc[1], c='blue', s=100, marker='o', 
                      label='Customer' if i == 0 else "", zorder=7, alpha=0.7)
            
            # Add customer ID and demand
            ax.annotate(f'C{i+1}\n({demand})', (loc[0], loc[1]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, ha='left')
    
    def _plot_routes(self, ax, routes, iteration):
        """Plot current routes with different colors"""
        colors = ['green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for route_idx, route in enumerate(routes):
            color = colors[route_idx % len(colors)]
            
            # Extract coordinates for plotting
            x_coords = [route[i][0] for i in range(len(route))]
            y_coords = [route[i][1] for i in range(len(route))]
            
            # Plot route path
            ax.plot(x_coords, y_coords, color=color, linewidth=3, 
                   alpha=0.7, label=f'Route {route_idx + 1}')
            
            # Add arrows to show direction
            for i in range(len(x_coords) - 1):
                dx, dy = x_coords[i+1] - x_coords[i], y_coords[i+1] - y_coords[i]
                ax.arrow(x_coords[i], y_coords[i], dx*0.3, dy*0.3,
                        head_width=0.1, head_length=0.1, 
                        fc=color, ec=color, alpha=0.8)
    
    def _plot_metrics(self, ax, state, frame, total_frames):
        """Plot optimization metrics"""
        # Add text box with metrics
        metrics_text = f"""
        Iteration: {frame + 1}/{total_frames}
        Current Cost: {state.get('cost', 'N/A'):.2f}
        Routes: {len(state.get('routes', []))}
        """
        
        if 'best_cost' in state:
            metrics_text += f"Best Cost: {state['best_cost']:.2f}"
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _set_plot_limits(self, ax):
        """Set consistent plot limits"""
        ax.set_aspect('equal')
        ax.set_xlim(-2, 22)
        ax.set_ylim(-2, 22)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


def create_sample_optimization_history() -> List[Dict]:
    """Create sample optimization history for demonstration"""
    
    # Sample coordinates
    depot = (0, 0)
    customers = {
        (3, 15): 4, (7, 12): 4, (5, 8): 5, (12, 10): 7,
        (15, 5): 6, (18, 15): 3, (8, 3): 5, (14, 8): 6
    }
    intermediate_facs = [(10, 10)]
    
    history = []
    base_cost = 100
    
    for i in range(20):
        # Simulate gradual improvement
        improvement = i * 1.5 + np.random.normal(0, 2)
        cost = max(42.67, base_cost - improvement)
        
        # Create sample routes that get better
        if i < 5:
            # Poor initial routes
            routes = [[depot, (3, 15), (5, 8), depot], [depot, (7, 12), depot]]
        elif i < 10:
            # Medium routes
            routes = [[depot, (3, 15), (7, 12), (5, 8), depot], 
                     [depot, (15, 5), (18, 15), depot]]
        else:
            # Good final routes
            routes = [[depot, (3, 15), (5, 8), (12, 10), (15, 5), depot], 
                     [depot, (7, 12), (14, 8), depot]]
        
        history.append({
            'iteration': i,
            'cost': cost,
            'best_cost': min([h.get('cost', float('inf')) for h in history] + [cost]),
            'routes': routes
        })
    
    return history


if __name__ == "__main__":
    # Initialize video creator
    creator = OptimizationVideoCreator()
    
    # Create sample data
    print("🎬 Creating sample optimization videos...")
    
    # Sample data
    depot = (0, 0)
    customers = {
        (3, 15): 4, (7, 12): 4, (5, 8): 5, (12, 10): 7,
        (15, 5): 6, (18, 15): 3, (8, 3): 5, (14, 8): 6
    }
    intermediate_facs = [(10, 10)]
    
    # Generate sample optimization history
    optimization_history = create_sample_optimization_history()
    costs = [state['cost'] for state in optimization_history]
    
    # Create different types of videos
    print("\n1️⃣ Creating main optimization animation...")
    main_video = creator.create_optimization_animation(
        optimization_history, customers, depot, intermediate_facs,
        "alns_optimization_process.mp4"
    )
    
    print("\n2️⃣ Creating cost convergence video...")
    cost_video = creator.create_cost_convergence_video(costs, "cost_convergence.mp4")
    
    print("\n3️⃣ Creating side-by-side comparison video...")
    dual_video = creator.create_side_by_side_video(
        optimization_history, customers, depot, intermediate_facs,
        "optimization_comparison.mp4"
    )
    
    print(f"\n🎉 All videos created successfully!")
    print(f"📁 Output directory: {creator.output_dir}")
    print(f"📹 Videos created:")
    print(f"   • {Path(main_video).name}")
    print(f"   • {Path(cost_video).name}")  
    print(f"   • {Path(dual_video).name}")
    
    print(f"\n💡 To integrate with your ALNS algorithm:")
    print(f"   1. Track optimization state in each iteration")
    print(f"   2. Store cost, routes, and metrics for each step")
    print(f"   3. Pass the history to create_optimization_animation()")
    print(f"   4. Get beautiful MP4/GIF videos of your optimization!")