"""Abstract task module."""
import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing
import torch.nn
import torch.optim
import yaml
from packaging.version import parse as V
from torch.utils.data import DataLoader
from typeguard import check_argument_types, check_return_type

from examples.speech_synthesis.preprocessing.tfgridnet.train.espnet_model import AbsESPnetModel
from examples.speech_synthesis.preprocessing.tfgridnet.train.class_choices import ClassChoices

try:
    import wandb
except Exception:
    wandb = None

if V(torch.__version__) >= V("1.5.0"):
    from torch.multiprocessing.spawn import ProcessContext
else:
    from torch.multiprocessing.spawn import SpawnContext as ProcessContext


optim_classes = dict(
    adam=torch.optim.Adam,
    adamw=torch.optim.AdamW,
    #sgd=SGD,
    adadelta=torch.optim.Adadelta,
    adagrad=torch.optim.Adagrad,
    adamax=torch.optim.Adamax,
    asgd=torch.optim.ASGD,
    lbfgs=torch.optim.LBFGS,
    rmsprop=torch.optim.RMSprop,
    rprop=torch.optim.Rprop,
)
if V(torch.__version__) >= V("1.10.0"):
    # From 1.10.0, RAdam is officially supported
    optim_classes.update(
        radam=torch.optim.RAdam,
    )
try:
    import torch_optimizer

    optim_classes.update(
        accagd=torch_optimizer.AccSGD,
        adabound=torch_optimizer.AdaBound,
        adamod=torch_optimizer.AdaMod,
        diffgrad=torch_optimizer.DiffGrad,
        lamb=torch_optimizer.Lamb,
        novograd=torch_optimizer.NovoGrad,
        pid=torch_optimizer.PID,
        # torch_optimizer<=0.0.1a10 doesn't support
        # qhadam=torch_optimizer.QHAdam,
        qhm=torch_optimizer.QHM,
        sgdw=torch_optimizer.SGDW,
        yogi=torch_optimizer.Yogi,
    )
    if V(torch_optimizer.__version__) < V("0.2.0"):
        # From 0.2.0, RAdam is dropped
        optim_classes.update(
            radam=torch_optimizer.RAdam,
        )
    del torch_optimizer
except ImportError:
    pass
try:
    import apex

    optim_classes.update(
        fusedadam=apex.optimizers.FusedAdam,
        fusedlamb=apex.optimizers.FusedLAMB,
        fusednovograd=apex.optimizers.FusedNovoGrad,
        fusedsgd=apex.optimizers.FusedSGD,
    )
    del apex
except ImportError:
    pass
try:
    import fairscale
except ImportError:
    fairscale = None


scheduler_classes = dict(
    ReduceLROnPlateau=torch.optim.lr_scheduler.ReduceLROnPlateau,
    lambdalr=torch.optim.lr_scheduler.LambdaLR,
    steplr=torch.optim.lr_scheduler.StepLR,
    multisteplr=torch.optim.lr_scheduler.MultiStepLR,
    exponentiallr=torch.optim.lr_scheduler.ExponentialLR,
    CosineAnnealingLR=torch.optim.lr_scheduler.CosineAnnealingLR,
    #noamlr=NoamLR,
    #warmuplr=WarmupLR,
    #warmupsteplr=WarmupStepLR,
    #warmupReducelronplateau=WarmupReduceLROnPlateau,
    cycliclr=torch.optim.lr_scheduler.CyclicLR,
    onecyclelr=torch.optim.lr_scheduler.OneCycleLR,
    CosineAnnealingWarmRestarts=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    #CosineAnnealingWarmupRestarts=CosineAnnealingWarmupRestarts,
)
# To lower keys
optim_classes = {k.lower(): v for k, v in optim_classes.items()}
scheduler_classes = {k.lower(): v for k, v in scheduler_classes.items()}


@dataclass
class IteratorOptions:
    preprocess_fn: callable
    collate_fn: callable
    data_path_and_name_and_type: list
    shape_files: list
    batch_size: int
    batch_bins: int
    batch_type: str
    max_cache_size: float
    max_cache_fd: int
    allow_multi_rates: bool
    distributed: bool
    num_batches: Optional[int]
    num_iters_per_epoch: Optional[int]
    train: bool


class AbsTask(ABC):
    # Use @staticmethod, or @classmethod,
    # instead of instance method to avoid God classes

    # If you need more than one optimizers, change this value in inheritance
    num_optimizers: int = 1
    #trainer = Trainer
    class_choices_list: List[ClassChoices] = []

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @abstractmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        pass

    @classmethod
    @abstractmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[[Sequence[Dict[str, np.ndarray]]], Dict[str, torch.Tensor]]:
        """Return "collate_fn", which is a callable object and given to DataLoader.

        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(collate_fn=cls.build_collate_fn(args, train=True), ...)

        In many cases, you can use our common collate_fn.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Define the required names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as following,

        >>> from espnet2.train.abs_espnet_model import AbsESPnetModel
        >>> class Model(AbsESPnetModel):
        ...     def forward(self, input, output, opt=None):  pass

        then "required_data_names" should be as

        >>> required_data_names = ('input', 'output')
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Define the optional names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as follows,

        >>> from espnet2.train.abs_espnet_model import AbsESPnetModel
        >>> class Model(AbsESPnetModel):
        ...     def forward(self, input, output, opt=None):  pass

        then "optional_data_names" should be as

        >>> optional_data_names = ('opt',)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_model(cls, args: argparse.Namespace) -> AbsESPnetModel:
        raise NotImplementedError


    # ~~~~~~~~~ The methods below are mainly used for inference ~~~~~~~~~
    @classmethod
    def build_model_from_file(
        cls,
        config_file: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        device: str = "cpu",
    ) -> Tuple[AbsESPnetModel, argparse.Namespace]:
        """Build model from the files.

        This method is used for inference or fine-tuning.

        Args:
            config_file: The yaml file saved when training.
            model_file: The model file saved when training.
            device: Device type, "cpu", "cuda", or "cuda:N".

        """
        assert check_argument_types()
        if config_file is None:
            assert model_file is not None, (
                "The argument 'model_file' must be provided "
                "if the argument 'config_file' is not specified."
            )
            config_file = Path(model_file).parent / "config.yaml"
        else:
            config_file = Path(config_file)

        logging.info("config file: {}".format(config_file))
        with config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args = argparse.Namespace(**args)
        model = cls.build_model(args)
        if not isinstance(model, AbsESPnetModel):
            raise RuntimeError(
                f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"
            )
        model.to(device)

        # For LoRA finetuned model, create LoRA adapter
        use_lora = getattr(args, "use_lora", False)
        if use_lora:
            create_lora_adapter(model, **args.lora_conf)

        if model_file is not None:
            if device == "cuda":
                # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
                #   in PyTorch<=1.4
                device = f"cuda:{torch.cuda.current_device()}"
            try:
                model.load_state_dict(
                    torch.load(model_file, map_location=device),
                    strict=not use_lora,
                )
            except RuntimeError:
                # Note(simpleoier): the following part is to be compatible with
                #   pretrained model using earlier versions before `0a625088`
                state_dict = torch.load(model_file, map_location=device)
                if any(["frontend.upstream.model" in k for k in state_dict.keys()]):
                    if any(
                        [
                            "frontend.upstream.upstream.model" in k
                            for k in dict(model.named_parameters())
                        ]
                    ):
                        state_dict = {
                            k.replace(
                                "frontend.upstream.model",
                                "frontend.upstream.upstream.model",
                            ): v
                            for k, v in state_dict.items()
                        }
                        model.load_state_dict(state_dict, strict=not use_lora)
                    else:
                        raise
                else:
                    raise

        return model, args
