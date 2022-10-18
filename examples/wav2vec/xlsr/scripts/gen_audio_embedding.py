"""
Usage:
    This script is used to extract the embedding / logit for speech classification task.
    1. Set fdir into your model checkpoint directory 
    2. Run the following command (preferrably on GPU machine to speed up the inference process)

   CUDA_VISIBLE_DEVICES=0 python3 examples/wav2vec/gen_audio_embedding.py /fsx/data/VoxLingua107/manifest --path ${fdir} \
    --task audio_classification --batch-size 90 --gen-subset test \
    --infer-manifest /fsx/data/VoxLingua107/manifest/test.tsv \
    --infer-xtimes 10 --infer-max-sample-size 160000 --output-path $odir 

    Example:
    Case: LID logit extraction
    fdir='/fsx/androstj/exps/voxlingua_lid_train_all/ckpt_100pct_300m_voxling-act_linear-pool_mean_fast-lr_1e-4-phase_0.1_0.4_0.5-maxupd_100000-ufreq_1-mprob_0.5-fz_0-cr_softmax/0/checkpoints/checkpoint_best.pt'
    python3  examples/wav2vec/gen_audio_embedding.py /fsx/data/VoxLingua107/manifest --path ${fdir} \
        --task audio_classification --batch-size 90 --gen-subset test \
        --infer-manifest /fsx/data/VoxLingua107/manifest/test.tsv \
        --infer-xtimes 10 --infer-max-sample-size 160000 --output-path $odir

"""
import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from fairseq import checkpoint_utils, data, options, tasks
from fairseq.data import FileAudioDataset, AddTargetDataset, Dictionary
from fairseq.tasks.audio_classification import LabelEncoder
import ipdb
import copy
import sys
from tqdm import tqdm
import tempfile
import numpy as np
import sklearn

def subset_manifest(infer_manifest, veri_pair):
    with open(infer_manifest) as ff, open(veri_pair) as gg, \
            tempfile.NamedTemporaryFile('w', delete=False) as ww:
        fnames = ff.read().strip().split("\n")
        basedir = fnames[0]
        needed_fname = []
        for gi in gg.read().strip().split('\n'):
            _, x1, x2 = gi.split()
            needed_fname.append(x1)
            needed_fname.append(x2)
        needed_fname = set(needed_fname)

        ww.write(basedir+'\n') 
        for ii in range(1, len(fnames)):
            x1,x2 = fnames[ii].split()
            if x1 in needed_fname:
                ww.write(fnames[ii]+'\n')
    print(f'| subset manifest for verification: {ww.name}')
    return ww.name

def wrap_target_dataset(infer_manifest, dataset, task):
    label_path = infer_manifest.replace(".tsv", ".label")
    with open(label_path, "r") as f:
        labels = f.read().strip().split("\n")
        assert len(labels) == len(dataset)
    process_label = LabelEncoder(task.target_dictionary)
    dataset = AddTargetDataset(dataset, labels, 
            pad=task.target_dictionary.pad(),
            eos=task.target_dictionary.eos(),
            batch_targets=True,
            process_label=process_label,
            add_to_input=False)
    return dataset

def resample_data(source, padding_mask, n_sample, max_sample_len):
    # source: BxT
    # padding_mask: BxT
    B = source.shape[0]
    T = source.shape[1]
    sources = []
    padding_masks = []
    seq_len = (~padding_mask).sum(1)
    for jj in range(n_sample):
        new_source = source.new_zeros(B, max_sample_len)
        new_padding_mask = padding_mask.new_zeros(B, max_sample_len)
        for ii in range(B):
            if seq_len[ii] > max_sample_len:
                start = np.random.randint(0, seq_len[ii]-max_sample_len+1)
                end = start + max_sample_len
            else :
                start = 0
                end = seq_len[ii]
            new_source[ii, 0:end-start] = source[ii, start:end]
            new_padding_mask[ii, end-start+1:] = True
        sources.append(new_source)
        padding_masks.append(new_padding_mask)
    return sources, padding_masks

def resample_sample(sample, n_sample, max_sample_len):
    new_sources, new_padding_masks = resample_data(sample['net_input']['source'], sample['net_input']['padding_mask'], n_sample, max_sample_len)
    new_samples = []
    for ii in range(n_sample):
        new_sample = copy.deepcopy(sample)
        new_sample['net_input']['source'] = new_sources[ii]
        new_sample['net_input']['padding_mask'] = new_padding_masks[ii]
        new_samples.append(new_sample)
    return new_samples

if __name__ == '__main__':
    np.random.seed(123)
    # Parse command-line arguments for generation
    parser = options.get_generation_parser(default_task='audio_classification')
    # parser.add_argument('--infer-merge', type=str, default='mean')
    parser.add_argument('--infer-xtimes', type=int, default=1)
    parser.add_argument('--infer-max-sample-size', type=int, default=5*16000)  # 5 secs
    parser.add_argument('--infer-manifest', type=str)
    parser.add_argument('--verification-pair', type=str, required=False, 
            help='''
            a file that contains pairs of utts to evaluated if they are from same speaker or not
            format: (following voxceleb)
            1/0 <wav_pair_a> <wav_pair_b>
            ''')
    parser.add_argument('--output-path', type=str)
    # parser.add_argument('--infer-xtimes', type=int, default=1)

    args = options.parse_args_and_arch(parser)
    # Setup task
    # task = tasks.setup_task(args)
    use_cuda = not args.cpu

    # Load model & task
    print('| loading model from {}'.format(args.path))
    arg_overrides = {
        'data': args.data,
        # 'mask_prob': 0
        #'max_sample_size': sys.maxsize,
        #'min_sample_size': 0,
    }
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    # move to AWS
    state['cfg']['model']['w2v_path'] = state['cfg']['model']['w2v_path'].replace('/checkpoint/arbabu/XLSR2/model_versions/', '/fsx/data/model_versions/').replace('/checkpoint/kushall/final_model_checkpoints/wav2vec2/', '/fsx/data/wav2vec_ckpt/')
    state['cfg']['task']['data'] = state['cfg']['task']['data'].replace('/checkpoint/kushall/data/', '/fsx/data/')
    
    models, _model_args, task = checkpoint_utils.load_model_ensemble_and_task([args.path], 
            arg_overrides=arg_overrides, 
            task=None,
            state=state)
    model = models[0]
    model.eval()
    if use_cuda:
        model.cuda()


    # Load dataset
    task.load_dataset(args.gen_subset)
    dataset = task.dataset(args.gen_subset)
    infer_manifest = args.infer_manifest
    # only decode needed utts
    # infer_manifest = subset_manifest(infer_manifest,
            # args.verification_pair)
    infer_dataset = FileAudioDataset(infer_manifest, 
            sample_rate=task.cfg.sample_rate,
            max_sample_size=10**10, #task.cfg.max_sample_size,
            min_sample_size=1, #task.cfg.min_sample_size,
            pad=True,
            normalize=task.cfg.normalize)
    # add target (if needed)
    infer_dataset = wrap_target_dataset(infer_manifest, infer_dataset, task) 
    itr = task.get_batch_iterator(
            dataset=infer_dataset,
            max_sentences=args.batch_size,
            ).next_epoch_itr(shuffle=False)


    # correct = 0
    # total = 0
    list_uttname = []
    list_latent = []
    list_logit = []
    list_target = []
    list_src_len = []
    with torch.no_grad():
        for _, sample in tqdm(enumerate(itr)):
            # resample if needed
            samples = resample_sample(sample, args.infer_xtimes, args.infer_max_sample_size)
            list_uttname.extend(sample['name'])
            list_target.extend(sample['target'][:, 0].cpu().numpy())
            list_src_len.extend((~sample['net_input']['padding_mask']).sum(1).cpu().numpy())
            latents = []
            logits = []
            for sample in samples:
                sample = utils.move_to_cuda(sample) if use_cuda else sample
                try:
                    latent = model.forward_latent(**sample['net_input'])
                    latents.append(latent.detach().cpu().numpy())
                except:
                    latent = None
                logit = model.forward(**sample['net_input'])
                logits.append(logit.detach().cpu().numpy())

            if len(latents) > 0:
                latents = np.stack(latents, 1) # B,X,D
            logits = np.stack(logits,  1) # B,X,Cls
            list_latent.extend(latents)
            list_logit.extend(logits)
            
    # create big npz
    list_uttname = np.array(list_uttname)
    list_latent = np.array(list_latent)
    list_target = np.array(list_target)
    list_logit = np.array(list_logit)
    list_src_len = np.array(list_src_len)
    # save to npz
    output_path = args.output_path
    if (output_path is None):
        output_path = tempfile.NamedTemporaryFile('wb', delete=False).name

    with open(output_path, 'wb') as ww:
        np.savez(ww, name=list_uttname, 
                latent=list_latent, 
                target=list_target, 
                logit=list_logit,
                src_len=list_src_len)

    print("="*10 + " REPORT " + "="*10)
    print(f'| latent saved in {output_path}')
    print(f'| {list_uttname.shape=}, {list_latent.shape=}, {list_target.shape=}, {list_logit.shape=}, {list_src_len.shape=}')
