from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from hyponic.metrics.regression import mse
from hyponic.hyponic import HypONIC

X, y = load_diabetes(return_X_y=True)
model = RandomForestRegressor()

hyperparams = {
    "min_samples_split": (0.01, 0.9),
    "min_samples_leaf": (0.01, 0.9),
    "min_weight_fraction_leaf": (0.0, 0.5),
    "min_impurity_decrease": (0.0, 0.9),
    "criterion": ["absolute_error", "squared_error"],
}

optimizer_kwargs = {
    "epoch": 50,
    "population_size": 50
}

hyponic = HypONIC(model, X, y, "rmse", **optimizer_kwargs)
hyponic.optimize(hyperparams, verbose=True)

print(hyponic.get_optimized_parameters())
print(hyponic.get_optimized_metric())
print(hyponic.get_optimized_model())
