import copy
import torch
import logging
from argparse import Namespace
import yaml
from fairseq import options
from examples.speech_to_speech.benchmarking.core import (
    Processing,
    SpeechGeneration,
    Cascaded2StageS2ST,
    Cascaded3StageS2ST,
    S2UT,
)
from examples.speech_to_speech.benchmarking.data_utils import (
    load_dataset_npy,
    load_dataset_raw_to_waveforms,
)


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(1)
torch.set_deterministic(True)


def make_parser():
    """Note: As the names indicate use s2x_args(ex:ST, ASR etc) for models with speech input,
    x2s_args for models with speech output(ex:TTS) and mt_args for translation models (ex: mt, T2U etc).
    For direct S2ST models, use x2s_args to provide model details.
    """
    parser = options.get_speech_generation_parser()
    parser.add_argument("--target-is-code", action="store_true", default=False)
    parser.add_argument("--config", type=str)
    parser.add_argument(
        "--model-type",
        default="S2U",
        choices=["S2S", "TTS", "S2UT", "MT", "S2T", "2StageS2ST", "3StageS2ST"],
        help="Choose one of the models. For model inference implementation, refer to core.py",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="""File to load dataset from. Assumes dataset is a list of samples.
        Each sample is a dict of format {'net_input':{'src_tokens':torch.tenor(),'src_lengths':torch.tensor()}}""",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="npy",
        choices=["npy", "raw"],
        help="""Type of input dataset file""",
    )
    parser.add_argument(
        "--read-using-sf",
        type=str,
        default=False,
        help="""If sound file should be used to read the raw dataset""",
    )
    parser.add_argument(
        "--dataset-size",
        default=None,
        type=int,
        help="Dataset size to use for benchmarking",
    )
    parser.add_argument(
        "--dump-speech-waveforms-dir",
        default=None,
        type=str,
        help="Directory to dump the speech waveforms computed on the dataset.",
    )
    parser.add_argument(
        "--dump-waveform-file-prefix",
        default="",
        type=str,
        help="File name prefix for the saved speech waveforms",
    )
    parser.add_argument(
        "--feat-dim", default=80, type=int, help="Input feature dimension"
    )
    parser.add_argument(
        "--target-sr",
        default=16000,
        type=int,
        help="Target sample rate for dumping waveforms",
    )

    options.add_generation_args(parser)
    options.get_interactive_generation_parser(parser)
    return parser


def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)

    with open(
        args.config,
        "r",
    ) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dict_args = vars(args)
    dict_args.update(config["general"])
    args = Namespace(**dict_args)

    i = 1
    stage_args = []
    while i <= 3:
        var = f"stage{i}"
        tmp_args = copy.deepcopy(dict_args)
        if var in config:
            tmp_args.update(config[var])
            stage_args.append(Namespace(**tmp_args))
            i += 1
        else:
            break

    if args.model_type == "S2S" or args.model_type == "TTS":
        model = SpeechGeneration(stage_args[0])
    elif args.model_type == "S2UT":
        model = S2UT(stage_args[0], stage_args[1] if len(stage_args) > 1 else None)
    elif args.model_type == "MT" or args.model_type == "S2T":
        model = Processing(stage_args[0])
    elif args.model_type == "2StageS2ST":
        model = Cascaded2StageS2ST(stage_args[0], stage_args[1])
    elif args.model_type == "3StageS2ST":
        model = Cascaded3StageS2ST(stage_args[0], stage_args[2], stage_args[1])
    else:
        raise Exception(f"Currently unsupported model type {args.model_type}")

    print(f"Evaluating on dataset - {args.dataset_path}\n")

    if args.dataset_type == "npy":
        dataset = load_dataset_npy(args.dataset_path, dataset_size=args.dataset_size)
    elif args.dataset_type == "raw":
        dataset = load_dataset_raw_to_waveforms(
            args.dataset_path,
            dataset_size=args.dataset_size,
            read_using_soundfile=args.read_using_sf,
        )
    else:
        raise Exception(f"Invalid dataset type {args.dataset_type}")

    model.warm_up(sample=dataset[0], repeat=2)

    run_time, memory, flops = model.gather_all_metrics(dataset, repeat=1)
    print(f"run_time = {run_time}sec \tmemory = {memory}MiB \tflops = {flops}")

    if args.dump_speech_waveforms_dir:
        model.dump_final_speech_output(
            dataset,
            args.dump_speech_waveforms_dir,
            lambda x: x,
            args.target_sr,
            prefix=args.dump_waveform_file_prefix,
        )


if __name__ == "__main__":
    cli_main()
