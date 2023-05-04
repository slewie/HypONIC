# HypONIC -- Hyperparameter Optimization with Nature Inspired Computing

*HypONIC* is a hyperparameter optimization library that uses various nature inspired computing algorithms to optimize
the hyperparameters of machine learning models. The library provides a simple interface for sklearn and keras models.

## Implementation details
*HypONIC* library follows a common high-level approach by providing a unified interface for applying an optimization algorithm
of choice to the parameters of a machine learning model (based on the `BaseEstimator` from the `sklearn`). Evalution of the
optimization process is done by a metric of choice.

### Metrics
*HypONIC* provides a list of classical evalution metrics:

- Mean Squared Error
- Mean Absolute Error
- Root Mean Square Error
- Huber Loss
- Log Loss
- Binary Crossentropy
- Precision Score
- Accuracy Score
- Recall Score
- F1 Score
- R2 Score
- ...and many others.

*HypONIC* library provides decorators for annotating custom metrics: `@minimize_metric` and `@maximize_metric`. These decorators
allows using names of the metric functions as string literals instead of passing the function itself, i.e.

```python
from hyponic.optimizer import HypONIC
...
hyponic = HypONIC(model, X, y, "mse", **optimizer_kwargs)
...
```

instead of

```python
from hyponic.optimizer import HypONIC
from hyponic.metrics import mse
...
hyponic = HypONIC(model, X, y, mse, **optimizer_kwargs)
...
```

### Hyperparameter Space
*HypONIC* library supports mixed space of hyperparameters, meaning that one optimization space can have both
dimensions of discrete and continuous parameters. The notation for each space is the following:

```python
hyperparams = {
    "min_impurity_decrease": (0.0, 0.9),              # Continious space -- (lower bound, upper bound)
    "min_samples_split": [2, 3, 4, 5, 6],             # Discrete space   -- a list of values of any type
    "criterion": ["absolute_error", "squared_error"]  # Discrete space   -- a list of values of any type
}
```

For the discrete space an automatic uniform mapping* to the continious space is provided if needed. Discrete space of
non numerical values is encoded first and then uniformly mapped to the continious space.

*is subject to change

## Features
*HypONIC* supports different nature inspired computing algorithms.
Please note that the library is currently under development and may contain a significant number of bugs and errors.
All algorithms are subject to further improvement.

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

This is a minimal example for tuning hyperparameters of the `RandomForestRegressor` from `sklearn`. The default optimizator is *Inertia Weight Particle Swarm Optimization* (IWPSO), as it has high applicability and has shown fast convergence.

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

# The default optimizator is the Inertia Weight Particle Swarm Optimization (IWPSO)
hyponic = HypONIC(model, X, y, mse, **optimizer_kwargs)
hyponic.optimize(hyperparams, verbose=True)

print(hyponic.get_optimized_parameters())
print(hyponic.get_optimized_metric())
print(hyponic.get_optimized_model())
```
