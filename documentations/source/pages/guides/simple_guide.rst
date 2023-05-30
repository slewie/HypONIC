Simple Guide
============

Installation
------------

Install the package via pip(https://pypi.org/project/hyponic/):

::

    $ pip install hyponic

Example
-------

.. code-block:: python

   from sklearn.datasets import load_diabetes
   from sklearn.ensemble import RandomForestRegressor

   from hyponic.metrics.regression import mse
   from hyponic.optimizers.swarm_based.PSO import PSO
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

   hyponic = HypONIC(model, X, y, mse, PSO, **optimizer_kwargs)
   hyponic.optimize(hyperparams, verbose=True)

   print(hyponic.get_optimized_parameters())
   print(hyponic.get_optimized_metric())
   print(hyponic.get_optimized_model())

Models
------

The models must be a scikit-learn compatible model. It must have a fit method and a predict method. In the future, we will add support for other frameworks.

Metrics
-------

In the library we have implemented the most common metrics for classification and regression. They can be passed to optimizer as a function or as a string.

Optimizers
----------

We have implemented swarm-based, physics-based and genetic algorithms. These algorithms have their parameters that can be passed to the optimizer as a dictionary.
All optimizers have a common parameters, such as:

   + epoch: number of iterations
   + population_size: number of individuals in the population
   + early_stopping: number of iterations without improvement to stop the optimization
   + minmax: 'min' or 'max', depending on whether the problem is a minimization or maximization problem
   + verbose: True or False, depending on whether you want to print the progress of the optimization
   + mode: 'single' or 'multithread', depending on whether to use multithreading or not
   + n_workers: number of workers to use in multithreading mode
   + early_stopping: number of epochs to wait before stopping the optimization process. If None, then early stopping is not used

Optimization
------------

To optimize the hyperparameters of a model, you must first create an instance of the optimizer. Then you have to call the optimize method passing the hyperparameters to optimize and the metric to use.
After optimization you can get the optimized parameters, the optimized metric and the optimized model.

