#!/usr/bin/env python3
# Author: Gemini
#
# This script provides a definitive, end-to-end verification of the project's
# functionality, with a specific focus on generating and saving video outputs.
# It runs a complete optimization process and ensures that the resulting
# animation GIFs are saved to the 'optimization_videos' directory.

import os
import sys
from datetime import datetime

# Ensure the project's `src` directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from main import (
    create_sample_instance,
    OptimizationVideoTracker,
    VIDEO_CREATOR_AVAILABLE,
)
from src.alns import ALNS


def verify_video_generation():
    """
    Runs a full, end-to-end test to verify video generation and saving.
    """
    print("=" * 60)
    print("🚀 STARTING END-TO-END VIDEO GENERATION VERIFICATION")
    print("=" * 60)

    if not VIDEO_CREATOR_AVAILABLE:
        print("❌ ERROR: Video creator dependencies are not available.")
        print("Please install the required packages from video_requirements.txt")
        sys.exit(1)

    # 1. Create a problem instance
    problem = create_sample_instance()
    print(f"\n✅ Problem instance '{problem.name}' created.")

    # 2. Initialize the ALNS solver
    solver = ALNS(problem)
    solver.max_iterations = 150  # A short run for verification purposes
    print(f"✅ ALNS solver initialized for {solver.max_iterations} iterations.")

    # 3. Set up the video tracker
    video_tracker = OptimizationVideoTracker(problem)
    print("✅ Video tracker initialized.")

    # Define a callback to track the solution state at each iteration
    def _video_callback(iteration_idx, best_solution):
        if iteration_idx % 5 == 0 or iteration_idx == solver.max_iterations:
            current_cost = getattr(best_solution, 'total_cost', float('inf'))
            video_tracker.track_state(iteration_idx, best_solution, current_cost)

    solver.iteration_callback = _video_callback
    print("✅ Solver callback for video tracking is set.")

    # 4. Run the optimization
    print("\n⏳ Running ALNS optimization...")
    solution = solver.run()
    print("✅ Optimization complete.")
    print(f"   - Final cost: {solution.total_cost:.2f}")

    # 5. Generate and save the video
    print("\n🎬 Generating optimization videos...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_verification')
    output_filename = f"alns_optimization_{timestamp}.gif"
    
    # Ensure the output directory exists
    output_dir = "optimization_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_path = video_tracker.create_video(output_filename=output_filename)

    # 6. Verify the output
    if video_path and os.path.exists(video_path):
        print(f"\n🎉 SUCCESS: Video successfully generated and saved to:")
        print(f"   -> {os.path.abspath(video_path)}")
        
        cost_video_path = video_path.replace('.gif', '_cost.gif')
        if os.path.exists(cost_video_path):
            print(f"   -> {os.path.abspath(cost_video_path)}")
            
        print("\nVerification complete. The system is fully functional end-to-end.")
        print("=" * 60)
        return True
    else:
        print("\n❌ FAILURE: Video generation failed.")
        print("Please check for errors in the console output above.")
        print("=" * 60)
        return False


if __name__ == "__main__":
    if not verify_video_generation():
        sys.exit(1)
