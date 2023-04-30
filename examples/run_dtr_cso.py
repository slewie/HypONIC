from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from hyponic.optimizers.PSO import IWPSO, PSO

from hyponic.metrics import mse
from hyponic.optimizer import HypONIC

X, y = load_diabetes(return_X_y=True)
model = DecisionTreeRegressor()

hyperparams = {
    "min_samples_split": (0.01, 0.9),
    "min_samples_leaf": (0.01, 0.9),
    "min_weight_fraction_leaf": (0.0, 0.5),
    "min_impurity_decrease": (0.0, 0.9),
}

optimizer_kwargs = {
    "epoch": 20,
    "pop_size": 100,
}

# dp: PSO works better than CSO for this example
hyponic = HypONIC(model, X, y, mse, optimizer=PSO, **optimizer_kwargs)
hyponic.optimize(hyperparams, verbose=True)

print(hyponic.get_optimized_parameters())
print(hyponic.get_optimized_metric())
print(hyponic.get_optimized_model())
