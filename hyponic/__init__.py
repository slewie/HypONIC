__version__ = '0.1.1'

from .hyponic import HypONIC

from .metrics.decorators import add_metric_to_dict, add_metric_info,\
    add_metric_aliases, minimize_metric, maximize_metric

from .metrics.classification import accuracy, precision, recall, f1_score,\
    fbeta, confusion_matrix, binary_crossentropy, categorical_crossentropy, log_loss

from .metrics.regression import mae, mse, rmse, rmsle, r2, adjusted_r2, huber_loss

from .optimizers.genetic_based.GA import GA
from .optimizers.physics_based.SA import SA
from .optimizers.swarm_based.ABC import ABC
from .optimizers.swarm_based.ACO import ACO
from .optimizers.swarm_based.PSO import PSO, IWPSO
from .optimizers.swarm_based.GWO import GWO
from .optimizers.swarm_based.CS import CS
from .optimizers.base_optimizer import BaseOptimizer

from .space import Space, Dimension, Continuous, Discrete

from .utils.history import History
from .utils.problem_identifier import ProblemIdentifier, ProblemType
