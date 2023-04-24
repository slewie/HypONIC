from typing import Callable, Any


def clip(x, low, high):
    return max(low, min(x, high))


class Space:
    def __init__(self, in_dict: dict[str, Any]):
        self.__dict = dict
        self.dimensions = {}
        self.dimensions_names = []

        for key, value in in_dict.items():
            # Converting range to list
            if isinstance(value, range):
                value = list(value)

            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(f'Value for key {key} is not valid')
                self.dimensions[key] = Continuous(*value, name=key)
            elif isinstance(value, list):
                self.dimensions[key] = Discrete(value, name=key)
            else:
                raise ValueError(f'Value for key {key} is not valid')

            self.dimensions_names.append(key)

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
        out = f"Space with {len(self.dimensions)} dimensions"
        for key in self.dimensions:
            # TODO: pretty print
            out += f"\n\t{key}:\t{self.dimensions[key]}"
        return out


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
            x = clip(x, low, high)
            x = self._lbound + (x - origin) / scale
            return self.get_value(x)

        return mapping_func, (low, high)

    def get_value(self, x):
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}({self._lbound}, {self._ubound})"

    def __repr__(self):
        return self.__str__()


class Continuous(Dimension):
    def __init__(self, low, high, name=None):
        super().__init__(low, high, name)

    def get_value(self, x):
        # Note that x is already in the correct range. No need to clip.
        # > maybe add transformation here? E.g. log, exp, etc.
        # TODO: non-linear mapping
        return x


class Discrete(Dimension):
    def __init__(self, values, name=None):
        super().__init__(0, len(values) - 1, name)
        self.values = values

    def get_value(self, x):
        # Note that x is already in the correct range. No need to clip.
        return self.values[int(x)]
