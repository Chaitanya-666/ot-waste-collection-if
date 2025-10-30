#!/usr/bin/env python3
"""
OT Project: Municipal Waste Collection Modelling with Intermediate Facilities
Main entry point
"""

from src.problem import ProblemInstance, Location
from src.alns import ALNS

def main():
    print("=== Waste Collection Route Optimization ===")
    print("Project: VRP with Intermediate Facilities using ALNS")

    # Create a simple test problem
    problem = ProblemInstance("Test Instance")
    problem.vehicle_capacity = 20

    # Add depot
    depot = Location(0, 0, 0, 0, "depot")
    problem.depot = depot

    print(f"Created problem: {problem}")

    # Initialize solver
    solver = ALNS(problem)

    # Run optimization
    solution = solver.run()
    print(f"Optimization completed!")

if __name__ == "__main__":
    main()
