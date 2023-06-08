Advanced guide
==============

Installation
------------

Install the package via pip(https://pypi.org/project/hyponic/):

::

    $ pip install hyponic

Example
-------

.. code-block:: python

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
   hyponic.visualize_history_time()
   hyponic.visualize_history_fitness()

Models
------

The models must be a scikit-learn, xgboost or lightgbm compatible model. It must have a fit method and a predict method. In the future, we will add support for other frameworks.
We have implemented default hyperparameters for several sklearn models. You can check them in the config/sklearn_models.py. For example, if you want to optimize the hyperparameters of the SVC model:

.. code-block:: python

   hyponic = HypONIC(SVC(), X, y)
   hyponic.optimize()
   print(hyponic.get_optimized_parameters())
   print(hyponic.get_optimized_metric())
   print(hyponic.get_optimized_model())

Our library tune the hyperparameters independently, so you can use the optimized model directly.

Metrics
-------

In the example above, we don't use any metric, because library identifies problem type and automatically selects the metric based on target variable. Now it supports classification and regression problems. For regression problems, it uses mean squared error as metric. For binary classification problems, it uses binary crossentropy as metric. For multiclass classification problems, it uses log loss.  But if you want to use a specific metric, you can pass it as a parameter to the HypONIC class as function or string:

::

   HypONIC(model, X, y, metric="log_loss")
   HypONIC(model, X, y, metric=log_loss)

Also, you can create your custom metric function and pass it to the HypONIC class, you can mark it with @minimize or @maximize decorator. If you don't mark it, it will be considered as a metric to minimize.

.. code-block:: python

   @minimize
   def custom_metric(y_true, y_pred):
       return 1 - accuracy_score(y_true, y_pred)

   HypONIC(model, X, y, metric=custom_metric)

Optimizers
----------

HypONIC supports several swarm-based, physic-based and genetic-based algorithms. You can use them by passing them as a parameter to the HypONIC class, by default it uses Inertia Weight PSO. You can also pass the parameters of the optimizer as a dictionary to the HypONIC class.

Library supports early stopping, which stops the optimization process if the fitness value does not change for a certain number of epochs. You can pass the early stopping parameters as a dictionary to the HypONIC class. Also, library has parallelization support, you can turn it on by passing the mode parameter as "multithreading" and set the number of threads with the n_jobs parameter. If you want to use multiprocessing, you can pass the mode parameter as "multiprocessing" and set the number of processes with the n_jobs parameter. If you don't pass the mode parameter, it will use the default mode of the optimizer.

You can create your own optimizers by inheriting the BaseOptimizer class and overriding the evolve method. This method used for one iteration on the meta heuristic algorithm.


