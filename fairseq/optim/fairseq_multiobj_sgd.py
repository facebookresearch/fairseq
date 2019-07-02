from . import FairseqOptimizer, register_optimizer

from .multiobj_optim import multiobj_optims


@register_optimizer('multiobj_sgd')
class FairseqMultiObjSGD(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        name = getattr(args, "multiobj_optim_name", "avg")
        self.optimizer_config["always_project"] = args.always_project
        self.optimizer_config["reverse"] = args.reverse_constraint
        if name in multiobj_optims:
            self._optimizer = multiobj_optims[name](
                params, **self.optimizer_config)
        else:
            raise ValueError(f"Unknown optimizer {name}")

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument('--multiobj-optim-name',
                            default='avg', metavar='NAME')
        parser.add_argument('--always-project', action="store_true")
        parser.add_argument('--reverse-constraint', action="store_true")

    def save_auxiliary(self):
        """This saves the gradients wrt. the auxiliary objective"""
        self.optimizer.save_auxiliary()

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'momentum': self.args.momentum,
            'weight_decay': self.args.weight_decay,
        }
