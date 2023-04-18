from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from mealpy.swarm_based.CSO import OriginalCSO
from hyponic.metrics import mse
from hyponic.optimizers.optimizer import Optimizer

X, y = load_diabetes(return_X_y=True)
model = DecisionTreeRegressor()
hyperparameters = {
    'min_samples_split': [0.01, 0.9],
    'min_samples_leaf': [0.01, 0.9],
    'min_weight_fraction_leaf': [0.0, 0.5],
    'min_impurity_decrease': [0.0, 0.9],
}
optimizer_kwargs = {
    'epoch': 100,
    'pop_size': 100,
}

optimizer = Optimizer(model, X, y, mse, hyperparameters, optimizer=OriginalCSO, **optimizer_kwargs)
optimizer.optimize()
print(optimizer.get_optimized_parameters())
print(optimizer.get_optimized_metric())
print(optimizer.get_optimized_model())