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

    # Create a sample test problem (depot, customers, IFs)
    problem = ProblemInstance("Sample Instance")
    problem.vehicle_capacity = 20
    problem.number_of_vehicles = 3
    problem.disposal_time = 2

    # Add depot
    depot = Location(0, 0, 0, 0, "depot")
    problem.depot = depot

    # Add intermediate facility(ies)
    # Use an ID outside the customer range to avoid accidental equality
    if1 = Location(100, 20, 20, 0, "if")
    problem.intermediate_facilities.append(if1)

    # Add customers (id, x, y, demand, type)
    customers = [
        Location(1, 5, 2, 4, "customer"),
        Location(2, 3, 8, 6, "customer"),
        Location(3, 9, 1, 5, "customer"),
        Location(4, 6, 7, 7, "customer"),
        Location(5, 2, 3, 3, "customer"),
        Location(6, 8, 6, 4, "customer"),
    ]
    for c in customers:
        problem.customers.append(c)

    print(f"Created problem: {problem}")
    print(f\"Customers: {len(problem.customers)}, IFs: {len(problem.intermediate_facilities)}\")

    # Initialize solver
    solver = ALNS(problem)

    # Run optimization (short smoke-run)
    solution = solver.run(max_iterations=200)
    print(\"Optimization completed!\")
    print(\"Best solution summary:\")
    print(solution)
    for idx, r in enumerate(solution.routes):
        print(f\"Route {idx+1}: {r}\")

if __name__ == "__main__":
    main()
