import torch
from fairseq.data.text_compressor import TextCompressionLevel, TextCompressor
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq import checkpoint_utils, data, options, tasks
from fairseq.data import FileAudioDataset, AddTargetDataset, Dictionary
from fairseq.tasks.audio_classification import LabelEncoder
import copy
from tqdm import tqdm
import tempfile
import numpy as np
import json

    
def subset_manifest(infer_manifest, veri_pair):
    with open(infer_manifest) as ff, open(veri_pair) as gg, tempfile.NamedTemporaryFile(
        "w", delete=False
    ) as ww:
        fnames = ff.read().strip().split("\n")
        basedir = fnames[0]
        needed_fname = []
        for gi in gg.read().strip().split("\n"):
            _, x1, x2 = gi.split()
            needed_fname.append(x1)
            needed_fname.append(x2)
        needed_fname = set(needed_fname)

        ww.write(basedir + "\n")
        for ii in range(1, len(fnames)):
            x1, x2 = fnames[ii].split()
            if x1 in needed_fname:
                ww.write(fnames[ii] + "\n")
    print(f"| subset manifest for verification: {ww.name}")
    return ww.name


def wrap_target_dataset(infer_manifest, dataset, task):
    label_path = infer_manifest.replace(".tsv", ".lang")
    text_compressor = TextCompressor(level=TextCompressionLevel.none)
    with open(label_path, "r") as f:
        labels = [text_compressor.compress(l) for i,l in enumerate(f)]
        assert len(labels) == len(dataset)
        
    process_label = LabelEncoder(task.target_dictionary)
    dataset = AddTargetDataset(
        dataset,
        labels,
        pad=task.target_dictionary.pad(),
        eos=task.target_dictionary.eos(),
        batch_targets=True,
        process_label=process_label,
        add_to_input=False,
    )
    return dataset


def resample_data(source, padding_mask, n_sample, max_sample_len):
    # source: BxT
    # padding_mask: BxT
    B = source.shape[0]
    T = source.shape[1]
    sources = []
    padding_masks = []
    if B == 1:
        return [source], [None]
    seq_len = (~padding_mask).sum(1)
    for jj in range(n_sample):
        new_source = source.new_zeros(B, max_sample_len)
        new_padding_mask = padding_mask.new_zeros(B, max_sample_len)
        for ii in range(B):
            if seq_len[ii] > max_sample_len:
                start = np.random.randint(0, seq_len[ii] - max_sample_len + 1)
                end = start + max_sample_len
            else:
                start = 0
                end = seq_len[ii]
            new_source[ii, 0 : end - start] = source[ii, start:end]
            new_padding_mask[ii, end - start + 1 :] = True
        sources.append(new_source)
        padding_masks.append(new_padding_mask)
    return sources, padding_masks


def resample_sample(sample, n_sample, max_sample_len):
    new_sources, new_padding_masks = resample_data(
        sample["net_input"]["source"],
        sample["net_input"]["padding_mask"],
        n_sample,
        max_sample_len,
    )
    new_samples = []
    for ii in range(n_sample):
        new_sample = copy.deepcopy(sample)
        new_sample["net_input"]["source"] = new_sources[ii]
        new_sample["net_input"]["padding_mask"] = new_padding_masks[ii]
        new_samples.append(new_sample)
    return new_samples


def dict_to_nparr(dd):
    dict_class = []
    dict_idx = []
    for ii, jj in enumerate(dd.symbols):
        dict_idx.append(ii)
        dict_class.append(jj)
    dict_idx = np.array(dict_idx)
    dict_class = np.array(dict_class)
    return dict_class, dict_idx


if __name__ == "__main__":
    np.random.seed(123)
    # Parse command-line arguments for generation
    parser = options.get_generation_parser(default_task="audio_classification")
    # parser.add_argument('--infer-merge', type=str, default='mean')
    parser.add_argument("--infer-xtimes", type=int, default=1)
    parser.add_argument("--infer-num-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--infer-max-sample-size", type=int, default=5 * 16000
    )  # 5 secs
    parser.add_argument("--infer-manifest", required=True, type=str)
    parser.add_argument("--output-path", default="/tmp/", type=str)

    args = options.parse_args_and_arch(parser)
    # Setup task
    # task = tasks.setup_task(args)
    use_cuda = not args.cpu

    # Load model & task
    print("| loading model from {}".format(args.path))
    arg_overrides = {
        "task": {
            "data": args.data
        },
        # 'mask_prob': 0
        #'max_sample_size': sys.maxsize,
        #'min_sample_size': 0,
    }
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path, arg_overrides)

    models, _model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path], arg_overrides=arg_overrides, task=None, state=state
    )
    model = models[0]
    model.eval()
    if use_cuda:
        model.cuda()
    # Load dataset

    dict_class, dict_idx = dict_to_nparr(task.target_dictionary)

    infer_manifest = args.infer_manifest
    infer_dataset = FileAudioDataset(
        infer_manifest,
        sample_rate=task.cfg.sample_rate,
        max_sample_size=10**10,  # task.cfg.max_sample_size,
        min_sample_size=1,  # task.cfg.min_sample_size,
        pad=True,
        normalize=task.cfg.normalize,
    )
    # add target (if needed)
    infer_dataset = wrap_target_dataset(infer_manifest, infer_dataset, task)

    itr = task.get_batch_iterator(
        dataset=infer_dataset,
        max_sentences=1,
        # max_tokens=args.max_tokens,
        num_workers=4,
    ).next_epoch_itr(shuffle=False)
    predictions = {}
    with torch.no_grad():
        for _, sample in tqdm(enumerate(itr)):
            # resample if needed
            samples = resample_sample(
                sample, args.infer_xtimes, args.infer_max_sample_size
            )
            for sample in samples:
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                try:
                    latent = model.forward_latent(**sample["net_input"])
                except:
                    latent = None
                logit = model.forward(**sample["net_input"])
                logit_lsm = torch.log_softmax(logit.squeeze(), dim=-1)
                scores, indices  = torch.topk(logit_lsm, args.top_k, dim=-1)
                scores = torch.exp(scores).to("cpu").tolist()
                indices = indices.to("cpu").tolist()
                assert sample["id"].numel() == 1
                sample_idx = sample["id"].to("cpu").tolist()[0]
                assert sample_idx not in predictions
                predictions[sample_idx] = [(task.target_dictionary[int(i)], s) for s, i in zip(scores, indices)]

    with open(f"{args.output_path}/predictions.txt", "w") as fo:
        for idx in range(len(infer_dataset)):
            fo.write(json.dumps(predictions[idx]) + "\n")

    print(f"Outputs will be located at - {args.output_path}/predictions.txt")
