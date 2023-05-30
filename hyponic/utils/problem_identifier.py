import numpy as np
from enum import Enum


class ProblemType(Enum):
    """
    This class is used to identify the problem type using dataset. It can be used for regression and classification only
    """
    REGRESSION = 1
    BINARY_CLASSIFICATION = 2
    MULTICLASS_CLASSIFICATION = 3


class ProblemIdentifier:
    """
    This class is used to identify the problem type using dataset. It can be used for regression and classification only
    """

    def __init__(self, y):
        self.y = y

    def get_problem_type(self):
        """
        This method identifies the problem type using number of classes and data type of y.
        If number of classes is 2 and data type of y is int64, then it is a binary classification problem
        If number of classes is more than 2 and data type of y is int64, then it is a multiclass classification problem
        If data type of y is not int64, then it is a regression problem
        """
        target_type = type(self.y[0])
        number_of_classes = len(np.unique(self.y))
        if target_type in [np.int64, np.int32, np.int16, np.int8]:
            if number_of_classes == 2:
                return ProblemType.BINARY_CLASSIFICATION
            else:
                return ProblemType.MULTICLASS_CLASSIFICATION
        else:
            return ProblemType.REGRESSION
