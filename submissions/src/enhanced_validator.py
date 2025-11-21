# Author: Harsh Sharma (231070064)
#
# This file contains the enhanced solution validator for the VRP with
# Intermediate Facilities. It provides comprehensive validation and
# verification capabilities for ALNS solutions.
"""
Enhanced Solution Validator for VRP with Intermediate Facilities

This module provides comprehensive validation and verification capabilities for ALNS solutions.
It includes:
- Detailed feasibility checking with specific constraint violations
- Solution quality assessment against known benchmarks
- Comprehensive validation reports
- Performance metrics calculation
- Cross-validation with multiple validation strategies

The validator is designed to be thorough and provide detailed feedback about
solution quality and constraint compliance.
"""

import math
import time
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .solution import Solution, Route
from .problem import ProblemInstance, Location


class ValidationStatus(Enum):
    """Validation result status codes"""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    SUBOPTIMAL = "suboptimal"
    INVALID = "invalid"


class ConstraintType(Enum):
    """Types of constraints that can be validated"""

    CAPACITY = "capacity"
    TIME = "time"
    DISTANCE = "distance"
    IF_VISITS = "if_visits"
    DEPOT = "depot"
    CUSTOMER_SERVICE = "customer_service"
    ROUTE_CONTINUITY = "route_continuity"
    LOAD_BALANCE = "load_balance"


@dataclass
class ConstraintViolation:
    """Detailed information about a constraint violation"""

    constraint_type: ConstraintType
    severity: str  # "critical", "warning", "info"
    description: str
    route_id: Optional[int] = None
    customer_id: Optional[int] = None
    violation_value: float = 0.0
    allowed_value: float = 0.0


@dataclass
class ValidationResult:
    """Complete validation result"""

    status: ValidationStatus
    is_feasible: bool
    total_violations: int
    critical_violations: int
    warnings: int
    constraint_violations: List[ConstraintViolation]
    quality_metrics: Dict[str, float]
    execution_time: float


class EnhancedSolutionValidator:
    """
    Enhanced validator for VRP with Intermediate Facilities solutions.

    Provides comprehensive validation including:
    - Basic feasibility checking
    - Detailed constraint violation analysis
    - Solution quality assessment
    - Performance benchmarking
    """

    def __init__(self, problem: ProblemInstance, tolerance: float = 1e-6):
        """
        Initialize the validator.

        Args:
            problem: The problem instance to validate against
            tolerance: Numerical tolerance for comparisons
        """
        self.problem = problem
        self.tolerance = tolerance
        self.validation_history: List[ValidationResult] = []

    def validate_solution(
        self, solution: Solution, check_quality: bool = True, benchmark: bool = False
    ) -> ValidationResult:
        """
        Perform comprehensive validation of a solution.

        Args:
            solution: The solution to validate
            check_quality: Whether to perform quality assessment
            benchmark: Whether to benchmark against known solutions

        Returns:
            ValidationResult: Complete validation report
        """
        start_time = time.time()

        # Initialize validation result
        violations = []
        metrics = {}

        # Basic feasibility checks
        feasibility_result = self._check_basic_feasibility(solution)
        violations.extend(feasibility_result.violations)

        # Detailed constraint validation
        constraint_violations = self._validate_constraints(solution)
        violations.extend(constraint_violations)

        # Calculate quality metrics
        if check_quality:
            quality_metrics = self._calculate_quality_metrics(solution)
            metrics.update(quality_metrics)

            # Benchmark if requested
            if benchmark:
                benchmark_metrics = self._benchmark_solution(solution, quality_metrics)
                metrics.update(benchmark_metrics)

        # Determine overall status
        is_feasible = len([v for v in violations if v.severity == "critical"]) == 0
        status = (
            ValidationStatus.FEASIBLE if is_feasible else ValidationStatus.INFEASIBLE
        )

        # Count violations by severity
        critical_count = len([v for v in violations if v.severity == "critical"])
        warning_count = len([v for v in violations if v.severity == "warning"])

        execution_time = time.time() - start_time

        result = ValidationResult(
            status=status,
            is_feasible=is_feasible,
            total_violations=len(violations),
            critical_violations=critical_count,
            warnings=warning_count,
            constraint_violations=violations,
            quality_metrics=metrics,
            execution_time=execution_time,
        )

        # Store in history
        self.validation_history.append(result)

        return result

    def _check_basic_feasibility(self, solution: Solution) -> "ValidationResult":
        """Check basic feasibility requirements."""
        violations = []

        # Check if all customers are assigned
        if solution.unassigned_customers:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.CUSTOMER_SERVICE,
                    severity="critical",
                    description=f"{len(solution.unassigned_customers)} customers not assigned",
                    violation_value=len(solution.unassigned_customers),
                )
            )

        # Check if routes exist
        if not solution.routes:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.ROUTE_CONTINUITY,
                    severity="critical",
                    description="No routes created",
                )
            )

        return ValidationResult(
            status=ValidationStatus.INFEASIBLE
            if violations
            else ValidationStatus.FEASIBLE,
            is_feasible=len(violations) == 0,
            total_violations=len(violations),
            critical_violations=len(violations),
            warnings=0,
            constraint_violations=violations,
            quality_metrics={},
            execution_time=0.0,
        )

    def _validate_constraints(self, solution: Solution) -> List[ConstraintViolation]:
        """Validate all solution constraints in detail."""
        violations = []

        for route_id, route in enumerate(solution.routes):
            # Validate route constraints
            route_violations = self._validate_route_constraints(route, route_id)
            violations.extend(route_violations)

            # Validate load constraints
            load_violations = self._validate_load_constraints(route, route_id)
            violations.extend(load_violations)

            # Validate IF visit constraints
            if_violations = self._validate_if_constraints(route, route_id)
            violations.extend(if_violations)

        return violations

    def _validate_route_constraints(
        self, route: Route, route_id: int
    ) -> List[ConstraintViolation]:
        """Validate route-level constraints."""
        violations = []

        # Check route starts and ends at depot
        if len(route.nodes) >= 2:
            if route.nodes[0] != self.problem.depot:
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.DEPOT,
                        severity="critical",
                        description=f"Route {route_id} does not start at depot",
                        route_id=route_id,
                    )
                )

            if route.nodes[-1] != self.problem.depot:
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.DEPOT,
                        severity="critical",
                        description=f"Route {route_id} does not end at depot",
                        route_id=route_id,
                    )
                )

        # Check route time constraint
        if route.total_time > self.problem.max_route_time:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.TIME,
                    severity="critical",
                    description=f"Route {route_id} time exceeds maximum: {route.total_time:.1f} > {self.problem.max_route_time}",
                    route_id=route_id,
                    violation_value=route.total_time,
                    allowed_value=self.problem.max_route_time,
                )
            )

        # Check route distance constraint
        if route.total_distance > self.problem.max_route_length:
            violations.append(
                ConstraintViolation(
                    constraint_type=ConstraintType.DISTANCE,
                    severity="critical",
                    description=f"Route {route_id} distance exceeds maximum: {route.total_distance:.1f} > {self.problem.max_route_length}",
                    route_id=route_id,
                    violation_value=route.total_distance,
                    allowed_value=self.problem.max_route_length,
                )
            )

        return violations

    def _validate_load_constraints(
        self, route: Route, route_id: int
    ) -> List[ConstraintViolation]:
        """Validate load constraints for a route."""
        violations = []

        for i, (node, load) in enumerate(zip(route.nodes, route.loads)):
            # Check capacity constraint
            if load > self.problem.vehicle_capacity + self.tolerance:
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.CAPACITY,
                        severity="critical",
                        description=f"Route {route_id} load exceeds capacity at node {node.id}: {load:.1f} > {self.problem.vehicle_capacity}",
                        route_id=route_id,
                        violation_value=load,
                        allowed_value=self.problem.vehicle_capacity,
                    )
                )

            # Check for negative loads
            if load < -self.tolerance:
                violations.append(
                    ConstraintViolation(
                        constraint_type=ConstraintType.CAPACITY,
                        severity="warning",
                        description=f"Route {route_id} has negative load at node {node.id}: {load:.1f}",
                        route_id=route_id,
                        violation_value=load,
                    )
                )

        return violations

    def _validate_if_constraints(
        self, route: Route, route_id: int
    ) -> List[ConstraintViolation]:
        """Validate intermediate facility visit constraints."""
        violations = []

        # Track cumulative load
        cumulative_load = 0.0
        last_if_index = -1

        for i, node in enumerate(route.nodes):
            if node.type == "customer":
                cumulative_load += node.demand
            elif node.type == "if":
                # Check if we needed to visit IF
                if last_if_index >= 0:
                    load_since_last_if = cumulative_load
                    if load_since_last_if > self.problem.vehicle_capacity:
                        violations.append(
                            ConstraintViolation(
                                constraint_type=ConstraintType.IF_VISITS,
                                severity="critical",
                                description=f"Route {route_id} should have visited IF earlier (load: {load_since_last_if:.1f} > {self.problem.vehicle_capacity})",
                                route_id=route_id,
                                violation_value=load_since_last_if,
                                allowed_value=self.problem.vehicle_capacity,
                            )
                        )

                cumulative_load = 0.0
                last_if_index = i

        return violations

    def _calculate_quality_metrics(self, solution: Solution) -> Dict[str, float]:
        """Calculate solution quality metrics."""
        metrics = {}

        # Basic metrics
        metrics["total_cost"] = solution.total_cost
        metrics["total_distance"] = solution.total_distance
        metrics["total_time"] = solution.total_time
        metrics["num_vehicles"] = len(solution.routes)
        metrics["num_if_visits"] = sum(
            len([n for n in route.nodes if n.type == "if"]) for route in solution.routes
        )

        # Efficiency metrics
        total_demand = sum(c.demand for c in self.problem.customers)
        metrics["distance_efficiency"] = total_demand / max(
            solution.total_distance, 1.0
        )
        metrics["capacity_utilization"] = (
            total_demand / (len(solution.routes) * self.problem.vehicle_capacity)
        ) * 100

        # Route balance metrics
        if solution.routes:
            route_distances = [route.total_distance for route in solution.routes]
            route_times = [route.total_time for route in solution.routes]

            metrics["route_distance_std"] = math.sqrt(
                sum(
                    (d - sum(route_distances) / len(route_distances)) ** 2
                    for d in route_distances
                )
                / len(route_distances)
            )
            metrics["route_time_std"] = math.sqrt(
                sum((t - sum(route_times) / len(route_times)) ** 2 for t in route_times)
                / len(route_times)
            )
        else:
            metrics["route_distance_std"] = 0.0
            metrics["route_time_std"] = 0.0

        return metrics

    def _benchmark_solution(
        self, solution: Solution, quality_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Benchmark solution against known criteria."""
        benchmark_metrics = {}

        # Simple benchmarks (these could be enhanced with actual known optimal solutions)
        total_demand = sum(c.demand for c in self.problem.customers)

        # Distance benchmark (theoretical minimum)
        if self.problem.customers:
            # Simple lower bound: sum of distances from depot to customers divided by vehicle capacity
            min_distance_per_vehicle = (
                sum(
                    self.problem.calculate_distance(self.problem.depot, c)
                    for c in self.problem.customers
                )
                / self.problem.vehicle_capacity
            )
            benchmark_metrics["distance_gap"] = (
                solution.total_distance - min_distance_per_vehicle
            ) / max(min_distance_per_vehicle, 1.0)
        else:
            benchmark_metrics["distance_gap"] = 0.0

        # Vehicle utilization benchmark
        benchmark_metrics["vehicle_utilization"] = min(
            100.0, quality_metrics["capacity_utilization"]
        )

        # IF visit efficiency
        if_visits = quality_metrics["num_if_visits"]
        vehicles_used = quality_metrics["num_vehicles"]
        benchmark_metrics["if_efficiency"] = if_visits / max(vehicles_used, 1.0)

        return benchmark_metrics

    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate a human-readable validation report."""
        report = []
        report.append("=" * 60)
        report.append("SOLUTION VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Status: {result.status.value.upper()}")
        report.append(f"Execution Time: {result.execution_time:.3f} seconds")
        report.append("")

        if result.constraint_violations:
            report.append("CONSTRAINT VIOLATIONS:")
            report.append("-" * 30)

            for violation in result.constraint_violations:
                report.append(
                    f"[{violation.severity.upper()}] {violation.constraint_type.value}:"
                )
                report.append(f"  {violation.description}")
                if violation.route_id is not None:
                    report.append(f"  Route ID: {violation.route_id}")
                if violation.customer_id is not None:
                    report.append(f"  Customer ID: {violation.customer_id}")
                report.append("")
        else:
            report.append("✅ No constraint violations found")
            report.append("")

        if result.quality_metrics:
            report.append("QUALITY METRICS:")
            report.append("-" * 30)

            for metric, value in result.quality_metrics.items():
                if isinstance(value, float):
                    report.append(f"{metric}: {value:.3f}")
                else:
                    report.append(f"{metric}: {value}")
            report.append("")

        report.append(
            f"Summary: {result.total_violations} total violations "
            f"({result.critical_violations} critical, {result.warnings} warnings)"
        )

        return "\n".join(report)

    def get_validation_history(self) -> List[ValidationResult]:
        """Get validation history for analysis."""
        return self.validation_history

    def clear_history(self):
        """Clear validation history."""
        self.validation_history.clear()
