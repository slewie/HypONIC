from typing import Callable, Any


class Space:
    def __init__(self, in_dict: dict[str, Any]):
        self.__dict = dict
        self.dimensions = {}
        self.dimensions_names = []

        for key, value in in_dict.items():
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
    def __init__(self, name=None):
        self.name = name

    def get_continuous_mapping(self, scale=1, origin=None) -> (Callable, (float, float)):
        raise NotImplementedError


class Continuous(Dimension):
    def __init__(self, low, high, name=None):
        super().__init__(name)
        self.low = low
        self.high = high

    def get_continuous_mapping(self, scale=1, origin=None) -> (Callable, (float, float)):
        """
        Returns a function that maps a discrete value to a continuous value.
        """
        assert scale != 0, "Scale cannot be 0."

        if origin is None:
            origin = self.low

        low = origin
        high = origin + (self.high - self.low) * scale

        def mapping_func(x):
            if x <= low:
                return self.low
            elif x >= high:
                return self.high
            return self.low + (x - origin) / scale

        return mapping_func, (low, high)

    def __str__(self):
        return f"Continuous({self.low}, {self.high})"

    def __repr__(self):
        return f"Continuous({self.low}, {self.high})"


class Discrete(Dimension):
    def __init__(self, values, name=None):
        super().__init__(name)
        self.values = values

    def get_continuous_mapping(self, scale: float = 1, origin: None | float = None) -> (Callable, (float, float)):
        """
        Returns a function that maps a discrete value to a continuous value.
        """
        assert scale != 0, "Scale cannot be 0."

        if origin is None:
            origin = 0

        low = origin
        high = origin + len(self.values) * scale

        def mapping_func(x: int | float) -> Any:
            if x <= low:
                return self.values[0]
            elif x >= high:
                return self.values[-1]
            return self.values[int((x - origin) / scale)]

        return mapping_func, (low, high)

    def __str__(self):
        return f"Discrete({self.values})"

    def __repr__(self):
        return f"Discrete({self.values})"
