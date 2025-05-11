# HAV (Hierarchical ABM Validation)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-%3E%3D%203.10-blue)](https://www.python.org/downloads/)

## Summary

**HAV** is a Python package for Hierarchical Agent-Based Model (ABM) Validation, designed to help users comprehensively validate agent-based models at three levels: Agent, Model, and Output. It integrates various statistical methods and machine learning techniques to ensure the accuracy and reliability of ABM simulations.

### HAV Package
| File          | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `HAV.py`      | Core validation logic, defines the `HAV` class and validation workflow      |
| `methods.py`  | Implements validation methods (MSE test, KS test, causal analysis, etc.)    |
| `utils.py`    | Provides utility functions (data flattening, Bayesian tools, etc.)    |

### Test Scripts
| File               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `traffic_model.py`    | Defines a Traffic Flow Model as a case ABM model |
| `wealth_model.py`      | Defines a Wealth Transfer Model as a case ABM model          |
| `examples.py`      | Demonstrates how to perform multi-level validation on an ABM model          |


## Installation

### 1. Install the base package
`pip install hav`

### 2. Install full dependencies (including optional analysis tools)
`pip install agentpy==0.1.5 numpy==1.25.2 pandas==1.5.3 pgmpy==0.1.25 scipy==1.14.0 statsmodels==0.14.0 catboost>=1.2.5 emcee>=3.1.6 SALib>=1.5.0`


## Quick Start

### 1. Determine a ABM model to be validated (Taking `traffic_model.py` as an example)
`from traffic_model import TrafficModel`
### 2. Set paramenters required in Traffic Model
`    p = {    `
`        'road_length': 50000,  `
`        'speed_limit': 120 * 1000 / 60, `
`        'num_vehicles': 50,  `
`        'slow_down_prob': 0.15,  `
`        'acceleration': 2.5 * 1000 / 60,  `
`        'deceleration': 3.5 * 1000 / 60, `
`        'vehicle_lengths': [4, 5, 6, 12],  `
`        'vehicle_length_probs': [0.7, 0.15, 0.1, 0.05], `
`        'steps': 100,`
`        'observer_id': 10`
`    }`
### 3. Set Benchmark Data in validation
`    Benchmark = {`
`        'Speed_range': (580, 300),`
`        'SIR_beta_range': (0.2, 1.0),`
`        'Pearson_corr_range': (0.8, 1.0)`
`    }`
### 4. Declare the parameters required in each validation level

Declare the parameters required in Agent(A), Model(M) and Output(O) level, including Simulated data(S) and Benchmark data(B)

`    A = {'S': ['Speed'], 'B': ['Speed_range']}`
`    M = {'S': ['Congestion'], 'B': ['SIR_beta_range']}`
`    O = {'S': ['num_vehicles', 'Congestion'], 'B': ['Pearson_corr_range']}`

### 5. Create HAV and do validation level by level`
`# 1 Create a HAV object`
`my_hav = HAV(TrafficModel, p, A, M, O)`

`# 2 Transmit validation methods that required in each level`
`my_hav.validate('Agent', test_SPEED_dist)`
`my_hav.validate('Model', test_CONGESTION_trans)`
`my_hav.validate('Output', test_CORR_FLOW_CONGESTION)`

Among them, `test_SPEED_dist`, `test_CONGESTION_trans` and `test_CORR_FLOW_CONGESTION` are validation methods that require user customization



## License
This project is licensed under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).