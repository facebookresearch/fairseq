import argparse
import copy
import os
from typing import Callable, Collection, Dict, List, Optional, Tuple

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from examples.speech_synthesis.preprocessing.tfgridnet.mask import AbsMask, MultiMask
from examples.speech_synthesis.preprocessing.tfgridnet.enh.decoder import (
    AbsDecoder,
    ConvDecoder,
    STFTDecoder,
    NullDecoder,
)
from examples.speech_synthesis.preprocessing.tfgridnet.enh.encoder import (
    AbsEncoder,
    ConvEncoder,
    STFTEncoder,
    NullEncoder,
)
from examples.speech_synthesis.preprocessing.tfgridnet.train.espnet_model import ESPnetEnhancementModel
from examples.speech_synthesis.preprocessing.tfgridnet.enh.loss_criterion import AbsEnhLoss, SISNRLoss
from examples.speech_synthesis.preprocessing.tfgridnet.enh.wrappers import AbsLossWrapper, PITSolver, FixedOrderSolver
from examples.speech_synthesis.preprocessing.tfgridnet.enh.separator import (
    AbsSeparator,
    RNNSeparator,
    TCNSeparator,
    TFGridNetMasking,
)

from examples.speech_synthesis.preprocessing.tfgridnet.tasks.abs_task import AbsTask
from examples.speech_synthesis.preprocessing.tfgridnet.torch_utils.initialize import initialize
from examples.speech_synthesis.preprocessing.tfgridnet.train.class_choices import ClassChoices
from examples.speech_synthesis.preprocessing.tfgridnet.train.preprocessor import (
    AbsPreprocessor,
    DynamicMixingPreprocessor,
    EnhPreprocessor,
)

encoder_choices = ClassChoices(
    name="encoder",
    classes=dict(stft=STFTEncoder, conv=ConvEncoder, same=NullEncoder),
    type_check=AbsEncoder,
    default="stft",
)

separator_choices = ClassChoices(
    name="separator",
    classes=dict(
        rnn=RNNSeparator, tcn=TCNSeparator, tfgridnet_masking=TFGridNetMasking
    ),
    type_check=AbsSeparator,
    default="rnn",
)

mask_module_choices = ClassChoices(
    name="mask_module",
    classes=dict(multi_mask=MultiMask),
    type_check=AbsMask,
    default="multi_mask",
)

decoder_choices = ClassChoices(
    name="decoder",
    classes=dict(stft=STFTDecoder, conv=ConvDecoder, same=NullDecoder),
    type_check=AbsDecoder,
    default="stft",
)

loss_wrapper_choices = ClassChoices(
    name="loss_wrappers",
    classes=dict(
        pit=PITSolver,
        fixed_order=FixedOrderSolver,
    ),
    type_check=AbsLossWrapper,
    default="pit",
)

criterion_choices = ClassChoices(
    name="criterions",
    classes=dict(
        si_snr=SISNRLoss,
    ),
    type_check=AbsEnhLoss,
    default="SISNRLoss",
)

preprocessor_choices = ClassChoices(
    name="preprocessor",
    classes=dict(
        dynamic_mixing=DynamicMixingPreprocessor,
        enh=EnhPreprocessor,
    ),
    type_check=AbsPreprocessor,
    default=None,
)

MAX_REFERENCE_NUM = 100


class EnhancementTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    class_choices_list = [
        # --encoder and --encoder_conf
        encoder_choices,
        # --separator and --separator_conf
        separator_choices,
        # --decoder and --decoder_conf
        decoder_choices,
        # --mask_module and --mask_module_conf
        mask_module_choices,
        # --preprocessor and --preprocessor_conf
        preprocessor_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    # trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel),
            help="The keyword arguments for model class.",
        )

        group.add_argument(
            "--criterions",
            action=NestedDictAction,
            default=[
                {
                    "name": "si_snr",
                    "conf": {},
                    "wrapper": "fixed_order",
                    "wrapper_conf": {},
                },
            ],
            help="The criterions binded with the loss wrappers.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--speech_volume_normalize",
            type=str_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value or range. "
            "e.g. --speech_volume_normalize 1.0 scales it to 1.0.\n"
            "--speech_volume_normalize 0.5_1.0 scales it to a random number in "
            "the range [0.5, 1.0)",
        )
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of signal-to-noise ratio (SNR) level in decibel.",
        )
        group.add_argument(
            "--short_noise_thres",
            type=float,
            default=0.5,
            help="If len(noise) / len(speech) is smaller than this threshold during "
            "dynamic mixing, a warning will be displayed.",
        )
        group.add_argument(
            "--use_reverberant_ref",
            type=str2bool,
            default=False,
            help="Whether to use reverberant speech references "
            "instead of anechoic ones",
        )
        group.add_argument(
            "--num_spk",
            type=int,
            default=1,
            help="Number of speakers in the input signal.",
        )
        group.add_argument(
            "--num_noise_type",
            type=int,
            default=1,
            help="Number of noise types.",
        )
        group.add_argument(
            "--sample_rate",
            type=int,
            default=8000,
            help="Sampling rate of the data (in Hz).",
        )
        group.add_argument(
            "--force_single_channel",
            type=str2bool,
            default=False,
            help="Whether to force all data to be single-channel.",
        )
        group.add_argument(
            "--channel_reordering",
            type=str2bool,
            default=False,
            help="Whether to randomly reorder the channels of the "
            "multi-channel signals.",
        )
        group.add_argument(
            "--categories",
            nargs="+",
            default=[],
            type=str,
            help="The set of all possible categories in the dataset. Used to add the "
            "category information to each sample",
        )
        group.add_argument(
            "--speech_segment",
            type=int_or_none,
            default=None,
            help="Truncate the audios to the specified length (in samples) if not None",
        )
        group.add_argument(
            "--avoid_allzero_segment",
            type=str2bool,
            default=True,
            help="Only used when --speech_segment is specified. If True, make sure "
            "all truncated segments are not all-zero",
        )
        group.add_argument(
            "--flexible_numspk",
            type=str2bool,
            default=False,
            help="Whether to load variable numbers of speakers in each sample. "
            "In this case, only the first-speaker files such as 'spk1.scp' and "
            "'dereverb1.scp' are used, which are expected to have multiple columns. "
            "Other numbered files such as 'spk2.scp' and 'dereverb2.scp' are ignored.",
        )

        group.add_argument(
            "--dynamic_mixing",
            type=str2bool,
            default=False,
            help="Apply dynamic mixing",
        )
        group.add_argument(
            "--utt2spk",
            type=str_or_none,
            default=None,
            help="The file path of utt2spk file. Only used in dynamic_mixing mode.",
        )
        group.add_argument(
            "--dynamic_mixing_gain_db",
            type=float,
            default=0.0,
            help="Random gain (in dB) for dynamic mixing sources",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()

        return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()

        use_preprocessor = getattr(args, "preprocessor", None) is not None

        if use_preprocessor:
            # TODO(simpleoier): To make this as simple as model parts, e.g. encoder
            if args.preprocessor == "dynamic_mixing":
                retval = preprocessor_choices.get_class(args.preprocessor)(
                    train=train,
                    source_scp=os.path.join(
                        os.path.dirname(args.train_data_path_and_name_and_type[0][0]),
                        args.preprocessor_conf.get("source_scp_name", "spk1.scp"),
                    ),
                    ref_num=args.preprocessor_conf.get(
                        "ref_num", args.separator_conf["num_spk"]
                    ),
                    dynamic_mixing_gain_db=args.preprocessor_conf.get(
                        "dynamic_mixing_gain_db", 0.0
                    ),
                    speech_name=args.preprocessor_conf.get("speech_name", "speech_mix"),
                    speech_ref_name_prefix=args.preprocessor_conf.get(
                        "speech_ref_name_prefix", "speech_ref"
                    ),
                    mixture_source_name=args.preprocessor_conf.get(
                        "mixture_source_name", None
                    ),
                    utt2spk=getattr(args, "utt2spk", None),
                    categories=args.preprocessor_conf.get("categories", None),
                )
            elif args.preprocessor == "enh":
                kwargs = dict(
                    # NOTE(kamo): Check attribute existence for backward compatibility
                    rir_scp=getattr(args, "rir_scp", None),
                    rir_apply_prob=getattr(args, "rir_apply_prob", 1.0),
                    noise_scp=getattr(args, "noise_scp", None),
                    noise_apply_prob=getattr(args, "noise_apply_prob", 1.0),
                    noise_db_range=getattr(args, "noise_db_range", "13_15"),
                    short_noise_thres=getattr(args, "short_noise_thres", 0.5),
                    speech_volume_normalize=getattr(
                        args, "speech_volume_normalize", None
                    ),
                    use_reverberant_ref=getattr(args, "use_reverberant_ref", None),
                    num_spk=getattr(args, "num_spk", 1),
                    num_noise_type=getattr(args, "num_noise_type", 1),
                    sample_rate=getattr(args, "sample_rate", 8000),
                    force_single_channel=getattr(args, "force_single_channel", False),
                    channel_reordering=getattr(args, "channel_reordering", False),
                    categories=getattr(args, "categories", None),
                    speech_segment=getattr(args, "speech_segment", None),
                    avoid_allzero_segment=getattr(args, "avoid_allzero_segment", True),
                    flexible_numspk=getattr(args, "flexible_numspk", False),
                )
                kwargs.update(args.preprocessor_conf)
                retval = preprocessor_choices.get_class(args.preprocessor)(
                    train=train, **kwargs
                )
            else:
                raise ValueError(
                    f"Preprocessor type {args.preprocessor} is not supported."
                )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech_ref1",)
        else:
            # Inference mode
            retval = ("speech_mix",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["speech_mix"]
        retval += ["dereverb_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["speech_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval += ["category"]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhancementModel:
        assert check_argument_types()

        encoder = encoder_choices.get_class(args.encoder)(**args.encoder_conf)
        separator = separator_choices.get_class(args.separator)(
            encoder.output_dim, **args.separator_conf
        )
        decoder = decoder_choices.get_class(args.decoder)(**args.decoder_conf)
        if args.separator.endswith("nomask"):
            mask_module = mask_module_choices.get_class(args.mask_module)(
                input_dim=encoder.output_dim,
                **args.mask_module_conf,
            )
        else:
            mask_module = None

        loss_wrappers = []

        if getattr(args, "criterions", None) is not None:
            # This check is for the compatibility when load models
            # that packed by older version
            for ctr in args.criterions:
                criterion_conf = ctr.get("conf", {})
                criterion = criterion_choices.get_class(ctr["name"])(**criterion_conf)
                loss_wrapper = loss_wrapper_choices.get_class(ctr["wrapper"])(
                    criterion=criterion, **ctr["wrapper_conf"]
                )
                loss_wrappers.append(loss_wrapper)

        # 1. Build model
        model = ESPnetEnhancementModel(
            encoder=encoder,
            separator=separator,
            decoder=decoder,
            loss_wrappers=loss_wrappers,
            mask_module=mask_module,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 2. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
