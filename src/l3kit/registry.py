import importlib
import logging

_log = logging.getLogger("cleo Registry")


def load(name: str):
    """Resolve a lazy 'module.path:ClassName' entry point without importing eagerly."""
    mod_name, _sep, attr_name = name.partition(":")
    if not _sep:
        raise ValueError(f"Entry point must contain a colon separator. Got: {name}")
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr_name)


class ModuleSpec:
    def __init__(self, id, entry_point=None, kwargs=None):
        self.id = id
        self.entry_point = entry_point
        self._kwargs = {} if kwargs is None else kwargs

    def make(self, **kwargs):
        if self.entry_point is None:
            raise ValueError(
                f"Attempting to make deprecated module '{self.id}' "
                "(HINT: is there a newer registered version of this module?)"
            )
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            return self.entry_point(**_kwargs)
        cls = load(self.entry_point)
        return cls(**_kwargs)


class ModuleRegistry:
    def __init__(self):
        self.module_specs = {}

    def register(self, id, **kwargs):
        if id in self.module_specs:
            raise ValueError(f"Cannot re-register id: '{id}'")
        self.module_specs[id] = ModuleSpec(id, **kwargs)

    def spec(self, id):
        try:
            return self.module_specs[id]
        except KeyError:
            known = ", ".join(sorted(self.module_specs)) or "<none registered>"
            raise KeyError(f"No registered module with id '{id}'. Known ids: {known}")

    def make(self, id, **kwargs):
        if kwargs:
            _log.info("Making module: %s (%s)", id, kwargs)
        else:
            _log.info("Making module: %s", id)
        return self.spec(id).make(**kwargs)

    def all(self):
        return list(self.module_specs.values())


module_registry = ModuleRegistry()


def register(id, **kwargs):
    return module_registry.register(id, **kwargs)


def make(id, **kwargs):
    return module_registry.make(id, **kwargs)


def spec(id):
    return module_registry.spec(id)


def list_registered_modules():
    return list(module_registry.module_specs.keys())
