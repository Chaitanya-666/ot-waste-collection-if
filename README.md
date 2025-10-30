# OT Project: Municipal Waste Collection Modelling with Intermediate Facilities

**Course:** Optimization Techniques
**Team:** Harsh G. Sharma (231070064) & Chaitanya Shinde (231070066)
**Institution:** T.Y. B.Tech (Computer)

## Abstract
This project addresses the optimization of municipal solid waste collection routes by formulating it as a **Vehicle Routing Problem with Intermediate Facilities (VRP-IF)**. Traditional waste collection vehicles must periodically visit disposal facilities (landfills/recycling centers) when full, creating complex routing challenges. We implement an **Adaptive Large Neighborhood Search (ALNS)** algorithm to compute near-optimal collection routes that minimize total travel distance while respecting vehicle capacity constraints and intermediate facility visits.

## Key Features
- Mathematical modelling of waste collection as VRP-IF
- ALNS metaheuristic with destroy/repair operators
- Synthetic dataset generation for testing
- Route visualization and performance analysis
- Support for multiple intermediate facilities

## Project Structure
```text
ot-waste-collection-if/
├── src/ # Source code
├── data/ # Problem instances
├── tests/ # Unit tests
├── docs/ # Documentation
├── main.py # Entry point
├── generate_data.py # Dataset generator
└── requirements.txt # Dependencies
```

## Installation & Setup

### For Windows
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### For Linux
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
