from fairseq import tasks
import numpy as np
import logging
import random
from fairseq import options
import torch
import os
import soundfile as sf

from fairseq.data.audio.audio_utils import (
    get_waveform,
    parse_path,
)

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(1)
np.random.seed(1)
random_number_generator = np.random.RandomState(30)


def generate_random_data_sample(T, B=1, D=80):
    """Generate random data sample given the T, B, D values"""
    net_input = {
        "src_tokens": torch.tensor(random_number_generator.randn(B, T, D)).float(),
        "src_lengths": torch.tensor([T]),
    }
    return {"net_input": net_input}


def generate_random_dataset(T_range_min, T_range_max, B=1, D=80, dataset_size=100):
    """Generate random dataset with T values within a given range, B, D"""
    T_values = [random.randint(T_range_min, T_range_max) for i in range(dataset_size)]
    dataset = []
    for t in T_values:
        dataset.append(generate_random_data_sample(t, B, D))
    return dataset, sum(T_values) / dataset_size


def load_dataset_npy(file_name, dataset_size=None):
    """Load dataset from a .npy file."""
    data = np.load(file_name, allow_pickle=True)
    if dataset_size:
        data = data[:dataset_size]
    return data


def load_dataset_raw_to_waveforms(
    file_name,
    dataset_size=None,
    need_waveform=True,
    sample_rate=16000,
    read_using_soundfile=False,
):
    """Load raw dataset from w2v tsv file. Optionally get waveforms"""
    data = []
    with open(file_name, "r") as fp:
        lines = fp.readlines()
        data = [
            os.path.join(lines[0].strip(), line.strip().split("\t")[0])
            for line in lines[1:]
        ]

    if dataset_size:
        data = data[:dataset_size]

    if not need_waveform:
        return data

    features = []
    if read_using_soundfile:
        for _i, d in enumerate(data):
            wav = sf.read(d)[0]
            if wav.ndim == 2:
                wav = wav.mean(-1)
            features.append(torch.from_numpy(wav).float().view(1, -1))
    else:
        for i, d in enumerate(data):
            _path, slice_ptr = parse_path(d)
            if len(slice_ptr) == 0:
                feat = get_waveform(
                    _path, always_2d=True, output_sample_rate=sample_rate
                )[0]
                features.append(
                    {
                        "id": i,
                        "net_input": {
                            "src_tokens": torch.tensor(feat),
                            "src_lengths": torch.tensor([feat.shape[1]]),
                        },
                    }
                )
            else:
                raise Exception("Currently unsupported data format")
    return features


def load_dataset_task(
    args,
    batch_size=1,
    limit_size=None,
    ref_dataset=None,
):
    """Loads dataset based on args by creating a task"""
    if not args.data or not args.subset or not args.task:
        raise Exception(
            "Please provide necessary arguments to load the dataset - data, subset and task"
        )
    task = tasks.setup_task(args)

    task.load_dataset(args.subset)
    if not limit_size:
        limit_size = len(task.dataset(args.subset))

    iter = task.get_batch_iterator(
        dataset=task.dataset(args.subset), max_sentences=batch_size
    ).next_epoch_itr(shuffle=False)
    dataset = []
    for i, sample in enumerate(iter):
        sample = {
            "id": task.datasets[args.subset].ids[sample["id"].item()],
            "net_input": {
                "src_tokens": sample["net_input"]["src_tokens"],
                "src_lengths": sample["net_input"]["src_lengths"],
            },
        }
        dataset.append(sample)
        if i == limit_size - 1:
            break

    if ref_dataset:
        try:
            ids = get_ids_from_dataset(ref_dataset)
        except Exception as e:
            raise Exception(f"{e} - Cannot extract ids from reference dataset")

        filtered_dataset = []
        for sample in dataset:
            if (
                sample["id"] in ids
                or sample["id"][5:] in ids
                or f"dev_{sample['id']}" in ids
            ):
                filtered_dataset.append(sample)
        dataset = filtered_dataset

    max_len, min_len, avg_len = get_dataset_stats(dataset)
    print(
        f"{args.subset} dataset stats : num_samples={len(dataset)} max_len = {max_len} min_len = {min_len} avg_len = {avg_len}"
    )

    return dataset


def randomly_sample_subset(dataset, size=500):
    """Randomly sample subset from a dataset"""
    random_indices = [random.randint(0, len(dataset) - 1) for i in range(size)]
    return [dataset[i] for i in random_indices]


def get_short_data_subset(dataset, size=500):
    """Get a subset of desired size by sorting based on src_lengths"""
    return sort_dataset(dataset)[:size]


def get_long_data_subset(dataset, size=500):
    """Get a subset of desired size by sorting based on src_lengths descending"""
    return sort_dataset(dataset, reverse=True)[:size]


def sort_dataset(dataset, reverse=False):
    return sorted(
        dataset, key=lambda x: x["net_input"]["src_lengths"].item(), reverse=reverse
    )


def save_dataset_npy(dataset, file_name):
    """Save a dataset as .npy file"""
    np.save(file_name, dataset)


def get_dataset_stats(dataset):
    """Get stats about dataset based on src_lengths of samples"""
    max_len = 0
    min_len = 100000
    avg_len = 0
    for d in dataset:
        max_len = max(max_len, d["net_input"]["src_lengths"].item())
        min_len = min(min_len, d["net_input"]["src_lengths"].item())
        avg_len += d["net_input"]["src_lengths"].item()

    return max_len, min_len, avg_len / len(dataset)


def make_parser():
    """
    Additional args:
        1. Provide the dataset dir path using --data.
        2. Loading the dataset doesn't require config, provide --config-yaml to apply additional feature transforms
    """
    parser = options.get_speech_generation_parser()
    parser.add_argument(
        "--subset",
        default=None,
        type=str,
        required=True,
        help="Subset to use for dataset generation",
    )
    parser.add_argument(
        "--dataset-save-dir",
        default=None,
        type=str,
        required=False,
        help="Dir path in which the datasets are to be saved",
    )
    parser.add_argument(
        "--ref-dataset",
        default=None,
        type=str,
        required=False,
        help="If provided, the ids in the reference dataset will be used to filter the new dataset generated.",
    )
    parser.add_argument("--dataset-save-token", default="", type=str, required=False)

    options.add_generation_args(parser)
    return parser


def get_ids_from_dataset(dataset):
    return {sample["id"]: 1 for sample in dataset}


def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    dataset = load_dataset_task(args)

    random_dataset = randomly_sample_subset(dataset)
    short_dataset = get_short_data_subset(dataset)
    long_dataset = get_long_data_subset(dataset)

    if args.dataset_save_token:
        args.dataset_save_token = f"_{args.dataset_save_token}_"

    if args.dataset_save_dir:
        save_dataset_npy(
            random_dataset,
            f"{args.dataset_save_dir}/random_dataset{args.dataset_save_token}w_ids.npy",
        )
        save_dataset_npy(
            short_dataset,
            f"{args.dataset_save_dir}/short_dataset{args.dataset_save_token}w_ids.npy",
        )
        save_dataset_npy(
            long_dataset,
            f"{args.dataset_save_dir}/long_dataset{args.dataset_save_token}w_ids.npy",
        )


if __name__ == "__main__":
    cli_main()
