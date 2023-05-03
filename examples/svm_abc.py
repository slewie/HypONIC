from sklearn.svm import SVR
from sklearn.datasets import load_wine

from hyponic.optimizers.ABC import ABC
from hyponic.optimizer import HypONIC

X, y = load_wine(return_X_y=True)
model = SVR()

hyperparams = {
    "C": (0.01, 1),
    "epsilon": (0.01, 0.9),
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
}

optimizer_kwargs = {
    "epoch": 10,
    "pop_size": 10,
}

hyponic = HypONIC(model, X, y, "log_loss", optimizer=ABC, **optimizer_kwargs)
hyponic.optimize(hyperparams, verbose=True)
print(hyponic.get_optimized_parameters())
print(hyponic.get_optimized_metric())
print(hyponic.get_optimized_model())
