"""
in this file, we will test the performance of the model with multithreading and without it
"""

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from hyponic.metrics.regression import mse
from hyponic.hyponic import HypONIC
import time
import numpy as np


def run(optimizer_kwargs, num_iter=15):
    X, y = load_diabetes(return_X_y=True)
    times = np.zeros(num_iter)
    hyperparams = {
        "n_neighbors": [i for i in range(1, 10)],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [i for i in range(1, 50)],
        "p": (1, 2)
    }
    for i in range(num_iter):
        model = KNeighborsRegressor()
        hyponic = HypONIC(model, X, y, mse, **optimizer_kwargs)
        start = time.time()
        hyponic.optimize(hyperparams)
        end = time.time()
        times[i] = end - start

    return np.mean(times)


def test_single():
    optimizer_kwargs = {
        "epoch": 50,
        "pop_size": 100,
        "mode": "single"
    }
    print("single thread: ", run(optimizer_kwargs))


def test_threads():
    optimizer_kwargs = {
        "epoch": 10,
        "pop_size": 10,
        "mode": "multithread",
        "n_workers": 4
    }
    print("multithread: ", run(optimizer_kwargs))


def test_process():
    optimizer_kwargs = {
        "epoch": 50,
        "pop_size": 50,
        "mode": "multiprocess"
    }
    print("multiprocess: ", run(optimizer_kwargs))


if __name__ == "__main__":
    test_single()

# with numexpr - 21.09
# without numexpr -
