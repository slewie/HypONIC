# HypONIC -- Hyperparameter Optimization with Nature Inspired Computing

*HypONIC* is a hyperparameter optimization library that uses various nature inspired computing algorithms to optimize
the hyperparameters of machine learning models. The library provides a simple interface for sklearn and keras models.

## Library Structure
```
|-- hyponic/
|   |-- hyponic.py
|   |-- space.py
|   |
|   |-- metrics/
|   |   |-- classification.py
|   |   |-- regression.py
|   |   |-- decorators.py
|   |
|   |-- optimizers/
|   |   |-- base_optimizer.py
|   |   |
|   |   |-- swarm_based/
|   |   |   |-- ABC.py
|   |   |   |-- ACO.py
|   |   |   |-- PSO.py
|   |   |
|   |   |-- physics_based/
|   |   |   |-- SA.py
|   |   |
|   |   |-- genetic_based/
|   |       |-- GA.py
|   |
|   |-- space/
|   |   |-- encoders.py (planned)
|   |   |-- mappers.py  (planned)
|   |
|   |-- utils/
|
|-- examples/
    |-- datasets/
```

## Implementation details
*HypONIC* library follows a common high-level approach by providing a unified interface for applying an optimization algorithm
of choice to the parameters of a machine learning model (based on the `BaseEstimator` from the `sklearn`). Evaluation of the
optimization process is done by a metric of choice.

### Metrics
*HypONIC* provides a list of classical evaluation metrics:

- Mean Squared Error
- Mean Absolute Error
- Root Mean Square Error
- Huber Loss
- Log Loss
- Binary Cross entropy
- Precision Score
- Accuracy Score
- Recall Score
- F1 Score
- R2 Score
- ...and many others.

*HypONIC* library provides decorators for annotating custom metrics:
`@minimize_metric`, `@maximize_metric`, and `@add_metric_aliases`.

Use case for the first two decorators is to annotate the metric function with the optimization goal, i.e. let's say
we want to add a custom metric function `custom_metric` to the library, and we want to minimize it. To do so, we
annotate the function with the `@minimize_metric` decorator:

```python
from hyponic.hyponic import HypONIC
from hyponic.optimizers.swarm_based.PSO import IWPSO
from hyponic.metrics.decorators import minimize_metric

model = ...
X, y = ..., ...
optimizer_kwargs = ...

# -- HERE IS THE DECORATOR --
@minimize_metric
def custom_metric(y_true, y_pred):
    ...

hyponic = HypONIC(model, X, y, custom_metric, optimizer=IWPSO, **optimizer_kwargs)
```

Under the hood, the decorator adds a `minimize` attribute to the function, which is later used by the optimizer to
determine the optimization goal.

Not only that, but the decorator also adds the function to the list of available metrics, so that it can be used
as a string literal in the `HypONIC` constructor, i.e.

```python
from hyponic.hyponic import HypONIC
from hyponic.optimizers.swarm_based.PSO import IWPSO
from hyponic.metrics.decorators import minimize_metric

model = ...
X, y = ..., ...
optimizer_kwargs = ...

@minimize_metric
def custom_metric(y_true, y_pred):
    ...

hyponic = HypONIC(model, X, y, "custom_metric", optimizer=IWPSO, **optimizer_kwargs)
```

For the standard metrics, the decorators are already applied, so that they can be used as string literals in the
`HypONIC` constructor without the need to import them from the `hyponic.metrics` module.

By default, function aliases are generated from the function name, however, it is possible to add custom aliases
using the `@add_metric_aliases` decorator:

```python
from hyponic.hyponic import HypONIC
from hyponic.optimizers.swarm_based.PSO import IWPSO
from hyponic.metrics.decorators import minimize_metric, add_metric_aliases

model = ...
X, y = ..., ...
optimizer_kwargs = ...

hyperparams = ...

@minimize_metric
@add_metric_aliases("custom_metric", "short_alias")
def custom_metric(y_true, y_pred):
    ...

hyponic = HypONIC(model, X, y, "short_alias", optimizer=IWPSO, **optimizer_kwargs)
hyponic.optimize(hyperparams, verbose=True)
```

Output that this code produces will show that the `short_alias`, used as a metric alias, resolves to the `custom_metric`.


### Hyperparameter Space
*HypONIC* library supports mixed space of hyperparameters, meaning that one optimization space can have both
dimensions of discrete and continuous parameters. The notation for each space is the following:

```python
hyperparams = {
    "min_impurity_decrease": (0.0, 0.9),              # Continuous space -- (lower bound, upper bound)
    "min_samples_split": [2, 3, 4, 5, 6],             # Discrete space   -- a list of values of any type
    "criterion": ["absolute_error", "squared_error"]  # Discrete space   -- a list of values of any type
}
```

For the discrete space an automatic uniform mapping* to the continuous space is provided if needed. Discrete space of
non-numerical values is encoded first and then uniformly mapped to the continuous space.

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
- - [x] Genetic Algorithm (GA)

## Minimal Example
This is a minimal example for tuning hyperparameters of the `RandomForestRegressor` from `sklearn`.
The default optimization is *Inertia Weight Particle Swarm Optimization* (IWPSO),
as it has high applicability and has shown fast convergence.

```python
# 1. Import of the model, dataset, metric and optimizer
#    with the algorithm of choice
from sklearn.svm import SVR
from sklearn.datasets import load_wine

from hyponic.hyponic import HypONIC

#    ...in this case, log_loss and Inertia Weight PSO
from hyponic.metrics.classification import log_loss
from hyponic.optimizers.swarm_based.PSO import IWPSO

# 2. Creating the model and loading the dataset
model = SVR()
X, y = load_wine(return_X_y=True)

# 3. Defining our problem
hyperparams = {
    "C":       (0.01, 1),
    "epsilon": (0.01, 0.9),
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
}

# 4. Parameterizing the optimizer
optimizer_kwargs = {
    "epoch": 10,
    "pop_size": 10,
}

# 5. Passing all to the optimizer
hyponic = HypONIC(
    model,
    X, y,
    log_loss,
    optimizer=IWPSO,
    **optimizer_kwargs
)

# 6. Solving for the given problem
hyponic.optimize(hyperparams, verbose=True)
print(hyponic.get_optimized_parameters())
print(hyponic.get_optimized_metric())
print(hyponic.get_optimized_model())
```
