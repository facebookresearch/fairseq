from .multiobj_sgd import MultiObjSGD

multiobj_optims = {}


def register_multiobj_optim(name):
    """Decorator to register a new multiobjective optimizer."""

    def register_multiobj_optim_cls(cls):
        if name in multiobj_optims:
            raise ValueError(f"Cannot register duplicate optimizer ({name})")
        if not issubclass(cls, MultiObjSGD):
            raise ValueError(f"Optimizer ({name}: {cls.__name__}) "
                             "must extend FairseqOptimizer")
        if cls.__name__ in multiobj_optims.values():
            # We use the optimizer class name as a unique identifier in
            # checkpoints, so all optimizer must have unique class names.
            raise ValueError("Cannot register optimizer with duplicate class "
                             f"name ({cls.__name__})")
        multiobj_optims[name] = cls
        return cls

    return register_multiobj_optim_cls
