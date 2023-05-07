class Symbol:
    def __init__(self, symtable, name, *args, **kwargs):
        self._symtable = symtable
        self._name = name
        self._args = args
        self._kwargs = kwargs  # TODO: Implement kwargs

    def __call__(self, *args):
        return Symbol(self._symtable, self._name, *args)

    def __repr__(self):
        return f"{self._name}({', '.join(repr(arg) for arg in self._args)})"

    def eval(self):
        args = [arg.eval() if isinstance(arg, Symbol) else arg for arg in self._args]

        # If all arguments are literals, evaluate the function
        # Otherwise, evaluate partially
        if all(not isinstance(arg, Symbol) for arg in args):
            if self._name in self._symtable.get_symbol_list():
                return self._symtable.get_symbol(self._name)(*args)
            else:
                return self
        else:
            # Partially evaluate
            self._args = args
            return self


class SymbolTable:
    def __init__(self):
        self._symbols = {
            "int": int,
            "float": float,
            "add": lambda x, y: x + y,
        }

    def get_symbol(self, name):
        return self._symbols[name]

    def get_symbol_list(self):
        return self._symbols.keys()

    def add_symbol(self, name, func):
        self._symbols[name] = func

    def __getattr__(self, name):
        return Symbol(self, name)


# scope = SymbolTable()
# expr = scope.add(scope.int(1), scope.int(2.5))
# expr.eval()
