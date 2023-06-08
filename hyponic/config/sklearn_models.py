models_dict = {
    "<class 'sklearn.svm._classes.SVC'>": {
        "C": (0.01, 1),
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": range(2, 6),
    },
    "<class 'sklearn.svm._classes.SVR'>": {
        "C": (0.01, 1),
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": range(2, 6),
    },
    "<class 'sklearn.ensemble._forest.RandomForestClassifier'>": {
        "n_estimators": range(10, 100, 10),
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 10),
        "min_samples_split": range(2, 10),
        "min_samples_leaf": range(1, 10),
        "max_features": ["sqrt", "log2"],
    },
    "<class 'sklearn.ensemble._forest.RandomForestRegressor'>": {
        "n_estimators": range(10, 100, 10),
        "criterion": ["squared_error", "absolute_error"],
        "max_depth": range(1, 10),
        "min_samples_split": range(2, 10),
        "min_samples_leaf": range(1, 10),
        "max_features": ["sqrt", "log2"],
    },
    "<class 'sklearn.neighbors._classification.KNeighborsClassifier'>": {
        'n_neighbors': range(1, 20),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    },
    "<class 'sklearn.neighbors._regression.KNeighborsRegressor'>": {
        'n_neighbors': range(1, 20),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    },
    "<class 'sklearn.tree._classes.DecisionTreeClassifier'>": {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': range(1, 20),
        'min_samples_split': range(2, 20),
        'min_samples_leaf': range(1, 20),
    },
    "<class 'sklearn.tree._classes.DecisionTreeRegressor'>": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
        'splitter': ['best', 'random'],
        'max_depth': [i for i in range(1, 20)],
        'min_samples_split': [i for i in range(2, 20)],
        'min_samples_leaf': [i for i in range(1, 20)]
    },
    "<class 'sklearn.linear_model._ridge.Ridge'>": {
        'alpha': (0, 20),
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    },
    "<class 'sklearn.linear_model._logistic.LogisticRegression'>": {
        'C': (0.1, 20),
        'solver': ['liblinear', 'sag', 'saga'],
    },
    "<class 'sklearn.linear_model._coordinate_descent.Lasso'>": {
        'alpha': (0.01, 20),
        'selection': ['cyclic', 'random'],
    },
    "<class 'sklearn.naive_bayes.GaussianNB'>": {
        'var_smoothing': (1e-9, 1e-1),
    },
    "<class 'sklearn.naive_bayes.MultinomialNB'>": {
        'alpha': (0.1, 20),
        'fit_prior': [True, False],
    },
    "<class 'sklearn.naive_bayes.BernoulliNB'>": {
        'alpha': (0.1, 20),
        'fit_prior': [True, False],
    },
    "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>": {
        'loss': ['absolute_error', 'squared_error', 'huber', 'quantile'],
        'learning_rate': (0.01, 1),
        'n_estimators': range(10, 100, 10),
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': range(1, 10),
        'min_samples_split': range(2, 10),
    },
    "<class 'sklearn.ensemble._gb.GradientBoostingClassifier'>": {
        'learning_rate': (0.01, 1),
        'n_estimators': range(10, 100, 10),
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': range(1, 10),
        'min_samples_split': range(2, 10),
    },
    "<class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>": {
        'n_estimators': range(10, 100, 10),
        'learning_rate': (0.01, 1),
        'algorithm': ['SAMME', 'SAMME.R'],
    },
    "<class 'sklearn.ensemble._weight_boosting.AdaBoostRegressor'>": {
        'n_estimators': range(10, 100, 10),
        'learning_rate': (0.01, 1),
        'loss': ['linear', 'square', 'exponential'],
    },
    "<class 'sklearn.ensemble._bagging.BaggingClassifier'>": {
        'n_estimators': range(10, 100, 10),
        'max_samples': (0.01, 1),
        'max_features': (0.01, 1),
        'bootstrap': [True, False],
        'bootstrap_features': [True, False],
    },
    "<class 'sklearn.ensemble._bagging.BaggingRegressor'>": {
        'n_estimators': range(10, 100, 10),
        'max_samples': (0.01, 1),
        'max_features': (0.01, 1),
        'bootstrap': [True, False],
        'bootstrap_features': [True, False],
    },

}
