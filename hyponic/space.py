"""
This module contains the Space class, which is used to define the search space.
"""
import numpy as np

from typing import Callable, Any


class Space:
    """
    A class that represents the search space of a hyperparameter optimization problem.
    """
    def __init__(self, in_dict: dict[str, Any]):
        self.__dict = dict
        self.dimensions = {}
        self.dimensions_names = []

        for k, v in in_dict.items():
            # Converting range to list
            if isinstance(v, range):
                v = list(v)

            if isinstance(v, Discrete):
                self.dimensions[k] = v
            elif isinstance(v, Continuous):
                self.dimensions[k] = v
            elif isinstance(v, tuple):
                if len(v) != 2:
                    raise ValueError(f"Value for key {k} is not valid")

                self.dimensions[k] = Continuous(*v, name=k)
            elif isinstance(v, list):
                self.dimensions[k] = Discrete(v, name=k)
            else:
                raise ValueError(f"Value for key {k} is not valid")

            self.dimensions_names.append(k)

    def get_continuous_mappings(
            self, scales: dict | int | float = None, origins: dict | int | float = None
    ) -> dict[str, (Callable, (float, float))]:
        """
        Returns a function that maps a discrete value to a continuous value.
        """

        if scales is None:
            scales = {}
        elif isinstance(scales, (int, float)):
            scales = {key: scales for key in self.dimensions}

        if origins is None:
            origins = {}
        elif isinstance(origins, (int, float)):
            origins = {key: origins for key in self.dimensions}

        mappings = {}
        for key in self.dimensions:
            mappings[key] = self.dimensions[key].get_continuous_mapping(
                scales.get(key, 1),
                origins.get(key, None)
            )

        return mappings

    def map_to_original_space(self, values: list[float]) -> dict[str, Any]:
        """
        Maps a list of values from the continuous space to the original space.
        """
        mappings = self.get_continuous_mappings()
        return {
            name: mappings[name][0](value) for name, value in zip(self.dimensions_names, values)
        }

    def __str__(self):
        if len(self.dimensions) == 0:
            return "Hyperparameter Search Space is empty."

        # Pretty table of dimensions
        dim_names = list(self.dimensions.keys())
        offset = max([len(k) for k in dim_names])

        out_strs = [f"Hyperparameter Search Space with {len(dim_names)} dimensions:"]
        out_strs += [
            f"\t{k}: {'-'*(offset - len(k))} {self.dimensions[k]}" for k in dim_names
        ]

        return "\n".join(out_strs)


class Dimension:
    def __init__(self, lbound, ubound, name=None):
        self.name = name

        self._lbound = lbound
        self._ubound = ubound

    def get_continuous_mapping(self, scale=1, origin=None) -> (Callable, (float, float)):
        """
        Returns a function that maps set of values to a continuous value
        """
        assert scale != 0, "Scale cannot be 0."

        if origin is None:
            origin = self._lbound

        low = origin
        high = origin + (self._ubound - self._lbound) * scale

        def mapping_func(x):
            x = np.clip(x, low, high)
            x = self._lbound + (x - origin) / scale
            return self.get_value(x)

        return mapping_func, (low, high)

    def get_value(self, x):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()


class Continuous(Dimension):
    def __init__(self, low, high, name=None):
        super().__init__(low, high, name)

    def get_value(self, x):
        # Note that x is already in the correct range. No need to clip.
        return x

    def __str__(self):
        return super().__str__() + f"({self._lbound}, {self._ubound})"

    def __repr__(self):
        return self.__str__()


class Discrete(Dimension):
    def __init__(self, values, name=None):
        super().__init__(0, len(values), name)
        self.values = np.array(values)

    def get_value(self, x):
        x = np.clip(x, 0, len(self.values) - 1)
        return self.values[int(x)]

    def __str__(self):
        return super().__str__() + f"{self.values}"

    def __repr__(self):
        return self.__str__()
