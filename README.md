# HAV (Hierarchical ABM Validation)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-%3E%3D%203.10-blue)](https://www.python.org/downloads/)

## Summary

**HAV** is a Python package for Hierarchical Agent-Based Model (ABM) Validation, designed to help users comprehensively validate agent-based models at three levels: Agent, Model, and Output. It integrates various statistical methods and machine learning techniques to ensure the accuracy and reliability of ABM simulations.

| Validation Level | Supported Methods                          | Code Module          |
|------------------|--------------------------------------------|----------------------|
| **Agent Level**  | MSE Test, KS Test                          | `methods.MSE`        |
| **Model Level**  | Causal Analysis, Bayesian Calibration      | `methods.bayesian`   |
| **Output Level** | Variance/Skewness/Kurtosis, Sensitivity    | `methods.sobol`      |

### HAV Package
| File          | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `HAV.py`      | Core validation logic, defines the `HAV` class and validation workflow      |
| `methods.py`  | Implements validation methods (MSE test, KS test, causal analysis, etc.)    |
| `utils.py`    | Provides utility functions (data flattening, ABM runner, Bayesian tools)    |

### Test Scripts
| File               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `test_model.py`    | Defines an example ABM model (wealth transfer model) and generates benchmark data |
| `test_hav.py`      | Demonstrates how to perform multi-level validation on an ABM model          |


## Installation
```bash
# Install the base package
pip install hav

# Install full dependencies (including optional analysis tools)
pip install agentpy==0.1.5 numpy==1.25.2 pandas==1.5.3 pgmpy==0.1.25 scipy==1.14.0 statsmodels==0.14.0 catboost>=1.2.5 emcee>=3.1.6 SALib>=1.5.0
```

## Quick Start

### 1. Define an ABM Model (`test_model.py`)
```python
from test_model import Environment

# Define benchmark parameters
benchmark_par = {
    'ag_num': 100,
    'init_wealth': [100000, 30000, 3000],
    'tran_mat': [[0.2,0.7,0.1], [0.3,0.5,0.2], [0.1,0.8,0.1]],
    'steps': 100
}

# Initialize the model
model = Environment(parameters=benchmark_par)
```

### 2. Run Validation (`test_hav.py`)
```python
from HAV import HAV

# Initialize the validator
hav = HAV(model, benchmark_data)

# Perform agent-level validation (MSE test)
hav.validate_agent_level(
    method='MSE test',
    I=['tran_mat', 'init_wealth'],
    O=['ev_wealth']
)

# View results
for level, result in hav.Results.items():
    if result:
        result.print_result()
```

## License
This project is licensed under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).