# ALNS Algorithm Parameters Guide

## Overview

This document provides detailed guidance on configuring the Adaptive Large Neighborhood Search (ALNS) algorithm for waste collection route optimization. Proper parameter tuning is crucial for achieving optimal performance on different problem instances.

## Core Algorithm Parameters

### Basic Configuration

| Parameter | Default Value | Recommended Range | Description |
|-----------|---------------|-------------------|-------------|
| `max_iterations` | 200 | 100-2000 | Maximum number of ALNS iterations |
| `seed` | 42 | Any integer | Random seed for reproducibility |
| `temperature` | 1000.0 | 500-5000 | Initial simulated annealing temperature |
| `cooling_rate` | 0.995 | 0.990-0.999 | Temperature reduction factor per iteration |

### Learning and Adaptation

| Parameter | Default Value | Recommended Range | Description |
|-----------|---------------|-------------------|-------------|
| `learning_rate` | 0.1 | 0.05-0.3 | Rate of operator weight adaptation |
| `adaptive_period` | 50 | 20-200 | Frequency of operator weight updates |

### Initial Solution Construction

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `initial_method` | "greedy" | Method for generating initial solution |
| Options: | "greedy" | Greedy nearest-neighbor construction |
| | "savings" | Clarke-Wright savings algorithm |
| | "random" | Random route construction |

## Destroy Operators

### Random Removal
- **Parameter**: `removal_ratio` (default: 0.15)
- **Description**: Fraction of customers to remove randomly
- **Range**: 0.05-0.30
- **Use Case**: Good for general exploration

### Worst Removal
- **Parameter**: `removal_ratio` (default: 0.15)
- **Description**: Remove customers with highest marginal cost
- **Range**: 0.05-0.30
- **Use Case**: Effective for local improvement

### Shaw Removal
- **Parameter**: `removal_ratio` (default: 0.15)
- **Description**: Remove similar customers (proximity + demand)
- **Range**: 0.05-0.30
- **Use Case**: Good for clustered customer distributions

### Route Removal
- **Parameter**: `removal_ratio` (default: 0.15)
- **Description**: Remove entire route segments between IF visits
- **Range**: 0.05-0.30
- **Use Case**: Effective for route-level exploration

## Repair Operators

### Greedy Insertion
- **Parameter**: `insertion_attempts` (default: 3)
- **Description**: Number of insertion attempts per customer
- **Range**: 1-10
- **Use Case**: Fast, reliable insertion

### Regret Insertion (k-regret)
- **Parameter**: `k_regret` (default: 2)
- **Description**: Number of best insertion costs to consider
- **Range**: 2-5
- **Use Case**: Better solution quality, slower execution

### IF-Aware Repair
- **Parameter**: `if_penalty_factor` (default: 1.5)
- **Description**: Additional cost for IF visits during insertion
- **Range**: 1.0-3.0
- **Use Case**: Ensures proper IF visit scheduling

### Savings Insertion
- **Parameter**: `merge_threshold` (default: 0.8)
- **Description**: Threshold for route merging
- **Range**: 0.5-1.0
- **Use Case**: Good for route consolidation

## Problem-Specific Parameters

### Vehicle Constraints
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `vehicle_capacity` | 25 | Maximum waste per vehicle |
| `max_route_time` | ∞ | Maximum route duration |
| `max_route_length` | ∞ | Maximum route distance |
| `disposal_time` | 0 | Time spent at intermediate facilities |

### Customer Properties
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `demand_range` | (1, 10) | Range for customer waste generation |
| `service_time_range` | (1, 5) | Range for customer service times |
| `cluster_factor` | 0.3 | Customer clustering tendency |

## Performance Tuning Guidelines

### Small Instances (≤ 20 customers)
- **Iterations**: 100-300
- **Temperature**: 500-1000
- **Learning Rate**: 0.1-0.2
- **Focus**: Fast convergence

### Medium Instances (21-50 customers)
- **Iterations**: 300-800
- **Temperature**: 1000-2000
- **Learning Rate**: 0.05-0.15
- **Focus**: Balanced exploration/exploitation

### Large Instances (> 50 customers)
- **Iterations**: 800-2000
- **Temperature**: 2000-5000
- **Learning Rate**: 0.03-0.1
- **Focus**: Exploration, diversity

### Problem Types

#### Dense Urban Areas
- **High clustering factor**: 0.6-0.9
- **Lower vehicle capacity**: 15-20
- **More IF visits**: Higher disposal frequency

#### Suburban Areas
- **Medium clustering factor**: 0.3-0.6
- **Medium vehicle capacity**: 20-30
- **Balanced IF visits**: Regular disposal

#### Rural Areas
- **Low clustering factor**: 0.1-0.3
- **Higher vehicle capacity**: 25-40
- **Fewer IF visits**: Longer collection routes

## Configuration Templates

### Basic Configuration
```json
{
  "algorithm": {
    "max_iterations": 200,
    "seed": 42,
    "temperature": 1000.0,
    "cooling_rate": 0.995