from sklearn.svm import SVC
from sklearn.datasets import load_wine

from hyponic.optimizers.swarm_based.ABC import ABC
from hyponic.hyponic import HypONIC

X, y = load_wine(return_X_y=True)
model = SVC()

hyperparams = {
    "C": (0.01, 1),
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": [2, 3, 4, 5],
}

optimizer_kwargs = {
    "epoch": 10,
    "pop_size": 10,
}

hyponic = HypONIC(model, X, y, optimizer=ABC, **optimizer_kwargs)
hyponic.optimize(hyperparams, verbose=True)
print(hyponic.get_optimized_parameters())
print(hyponic.get_optimized_metric())
print(hyponic.get_optimized_model())
