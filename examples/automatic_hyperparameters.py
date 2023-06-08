from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_wine

from hyponic.hyponic import HypONIC
from hyponic.optimizers.swarm_based.CS import CS

X, y = load_wine(return_X_y=True)
model = GradientBoostingClassifier()

optimizer_kwargs = {
    "epoch": 20,
    "pop_size": 50,
}

hyponic = HypONIC(model, X, y, optimizer=CS, **optimizer_kwargs)
hyponic.optimize(verbose=True)
print(hyponic.get_optimized_parameters())
print(hyponic.get_optimized_metric())
print(hyponic.get_optimized_model())