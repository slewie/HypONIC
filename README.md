# HypONIC -- Hyperparameter Optimization with Nature Inspired Computing

*HypONIC* is a hyperparameter optimization library that uses various nature inspired computing algorithms to optimize
the hyperparameters of machine learning models. The library provides a simple interface for sklearn and keras models.

## Features
*HypONIC* supports different nature inspired computing algorithms.
The following algorithms are currently implemented or are planned to be implemented:

**Swarm-based:**
- - [x] Particle Swarm Optimization (PSO)
- - [x] Inertia Weight PSO (IWPSO)
- - [x] Ant Colony Optimization (ACO)
- - [x] Artificial Bee Colony (ABC)
- - [ ] Grey Wolf Optimizer (GWO)
- - [ ] Cuckoo Search (CS)
- - [ ] Firefly Algorithm (FA)

**Physics-based:**
- - [x] Simulated Annealing (SA)

**Genetic-based:**
- - [x] Genetic Algorithm

## Minimal Example

This is a minimal 

```python
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from hyponic.metrics import mse
from hyponic.optimizer import HypONIC

X, y = load_diabetes(return_X_y=True)
model = RandomForestRegressor()

hyperparams = {
    "min_samples_split": (0.01, 0.9),  # Continious space
    "min_samples_leaf": (0.01, 0.9),  # Continious space
    "min_weight_fraction_leaf": (0.0, 0.5),  # Continious space
    "min_impurity_decrease": (0.0, 0.9),  # Continious space
    "criterion": ["absolute_error", "squared_error"]  # Discrete space
}

optimizer_kwargs = {
    "epoch": 50,
    "pop_size": 50,
}

hyponic = HypONIC(model, X, y, mse, **optimizer_kwargs)
hyponic.optimize(hyperparams, verbose=True)

print(hyponic.get_optimized_parameters())
print(hyponic.get_optimized_metric())
print(hyponic.get_optimized_model())
```
