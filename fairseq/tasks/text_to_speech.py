# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import os.path as op

import torch
import torch.nn.functional as F
import numpy as np

from fairseq.data.audio.text_to_speech_dataset import TextToSpeechDatasetCreator
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask
from fairseq.speech_generator import (
    AutoRegressiveSpeechGenerator, NonAutoregressiveSpeechGenerator,
    TeacherForcingAutoRegressiveSpeechGenerator
)

logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO
)
logger = logging.getLogger(__name__)


try:
    from tensorboardX import SummaryWriter
except ImportError:
    logger.info("Please install tensorboardX: pip install tensorboardX")
    SummaryWriter = None


@register_task('text_to_speech')
class TextToSpeechTask(SpeechToTextTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('data', help='manifest root path')
        parser.add_argument(
            '--config-yaml', type=str, default='config.yaml',
            help='Configuration YAML filename (under manifest root)'
        )
        parser.add_argument('--max-source-positions', default=1024, type=int,
                            metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1200, type=int,
                            metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument("--n-frames-per-step", type=int, default=1)
        parser.add_argument("--eos-prob-threshold", type=float, default=0.5)
        parser.add_argument("--eval-inference", action="store_true")
        parser.add_argument("--eval-tb-nsample", type=int, default=8)
        parser.add_argument("--vocoder", type=str, default="griffin_lim")
        parser.add_argument("--spec-bwd-max-iter", type=int, default=8)

    def __init__(self, args, src_dict):
        super().__init__(args, src_dict)
        self.src_dict = src_dict
        self.sr = self.data_cfg.config.get("features").get("sample_rate")

        self.tensorboard_writer = None
        self.tensorboard_dir = ""
        if args.tensorboard_logdir and SummaryWriter is not None:
            self.tensorboard_dir = os.path.join(args.tensorboard_logdir,
                                                "valid_extra")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith('train')
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = TextToSpeechDatasetCreator.from_tsv(
            self.args.data, self.data_cfg, split, self.src_dict,
            pre_tokenizer, bpe_tokenizer, is_train_split=is_train_split,
            epoch=epoch, seed=self.args.seed,
            n_frames_per_step=self.args.n_frames_per_step,
            speaker_to_id=self.speaker_to_id
        )

    @property
    def target_dictionary(self):
        return None

    @property
    def source_dictionary(self):
        return self.src_dict

    def get_speaker_embeddings_path(self):
        speaker_emb_path = None
        if self.data_cfg.config.get("speaker_emb_filename") is not None:
            speaker_emb_path = op.join(
                self.args.data, self.data_cfg.config.get("speaker_emb_filename")
            )
        return speaker_emb_path

    @classmethod
    def get_speaker_embeddings(cls, args):
        embed_speaker = None
        if args.speaker_to_id is not None:
            if args.speaker_emb_path is None:
                embed_speaker = torch.nn.Embedding(
                    len(args.speaker_to_id), args.speaker_embed_dim
                )
            else:
                speaker_emb_mat = np.load(args.speaker_emb_path)
                assert speaker_emb_mat.shape[1] == args.speaker_embed_dim
                embed_speaker = torch.nn.Embedding.from_pretrained(
                    torch.from_numpy(speaker_emb_mat), freeze=True,
                )
                logger.info(
                    f"load speaker embeddings from {args.speaker_emb_path}. "
                    f"train embedding? {embed_speaker.weight.requires_grad}\n"
                    f"embeddings:\n{speaker_emb_mat}"
                )
        return embed_speaker

    def build_model(self, cfg):
        cfg.pitch_min = self.data_cfg.config["features"].get("pitch_min", None)
        cfg.pitch_max = self.data_cfg.config["features"].get("pitch_max", None)
        cfg.energy_min = self.data_cfg.config["features"].get("energy_min", None)
        cfg.energy_max = self.data_cfg.config["features"].get("energy_max", None)
        cfg.speaker_emb_path = self.get_speaker_embeddings_path()
        model = super().build_model(cfg)
        self.generator = None
        if getattr(cfg, "eval_inference", False):
            self.generator = self.build_generator([model], cfg)
        return model

    def build_generator(self, models, cfg, vocoder=None, **unused):
        if vocoder is None:
            vocoder = self.build_default_vocoder()
        model = models[0]
        if getattr(model, "NON_AUTOREGRESSIVE", False):
            return NonAutoregressiveSpeechGenerator(
                model, vocoder, self.data_cfg
            )
        else:
            generator = AutoRegressiveSpeechGenerator
            if getattr(cfg, "teacher_forcing", False):
                generator = TeacherForcingAutoRegressiveSpeechGenerator
                logger.info("Teacher forcing mode for generation")
            return generator(
                model, vocoder, self.data_cfg,
                max_iter=self.args.max_target_positions,
                eos_prob_threshold=self.args.eos_prob_threshold
            )

    def build_default_vocoder(self):
        from fairseq.models.text_to_speech.vocoder import get_vocoder
        vocoder = get_vocoder(self.args, self.data_cfg)
        if torch.cuda.is_available() and not self.args.cpu:
            vocoder = vocoder.cuda()
        else:
            vocoder = vocoder.cpu()
        return vocoder

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(
            sample, model, criterion
        )

        if getattr(self.args, "eval_inference", False):
            hypos, inference_losses = self.valid_step_with_inference(
                sample, model, self.generator
            )
            for k, v in inference_losses.items():
                assert(k not in logging_output)
                logging_output[k] = v

            picked_id = 0
            if self.tensorboard_dir and (sample["id"] == picked_id).any():
                self.log_tensorboard(
                    sample,
                    hypos[:self.args.eval_tb_nsample],
                    model._num_updates,
                    is_na_model=getattr(model, "NON_AUTOREGRESSIVE", False)
                )
        return loss, sample_size, logging_output

    def valid_step_with_inference(self, sample, model, generator):
        hypos = generator.generate(model, sample, has_targ=True)

        losses = {
            "mcd_loss": 0.,
            "targ_frames": 0.,
            "pred_frames": 0.,
            "nins": 0.,
            "ndel": 0.,
        }
        rets = batch_mel_cepstral_distortion(
            [hypo["targ_waveform"] for hypo in hypos],
            [hypo["waveform"] for hypo in hypos],
            self.sr,
            normalize_type=None
        )
        for d, extra in rets:
            pathmap = extra[-1]
            losses["mcd_loss"] += d.item()
            losses["targ_frames"] += pathmap.size(0)
            losses["pred_frames"] += pathmap.size(1)
            losses["nins"] += (pathmap.sum(dim=1) - 1).sum().item()
            losses["ndel"] += (pathmap.sum(dim=0) - 1).sum().item()

        return hypos, losses

    def log_tensorboard(self, sample, hypos, num_updates, is_na_model=False):
        if self.tensorboard_writer is None:
            self.tensorboard_writer = SummaryWriter(self.tensorboard_dir)
        tb_writer = self.tensorboard_writer
        for b in range(len(hypos)):
            idx = sample["id"][b]
            text = sample["src_texts"][b]
            targ = hypos[b]["targ_feature"]
            pred = hypos[b]["feature"]
            attn = hypos[b]["attn"]

            if is_na_model:
                data = plot_tts_output(
                    [targ.transpose(0, 1), pred.transpose(0, 1)],
                    [f"target (idx={idx})", "output"], attn,
                    "alignment", ret_np=True, suptitle=text,
                )
            else:
                eos_prob = hypos[b]["eos_prob"]
                data = plot_tts_output(
                    [targ.transpose(0, 1), pred.transpose(0, 1), attn],
                    [f"target (idx={idx})", "output", "alignment"], eos_prob,
                    "eos prob", ret_np=True, suptitle=text,
                )

            tb_writer.add_image(
                f"inference_sample_{b}", data, num_updates,
                dataformats="HWC"
            )

            if hypos[b]["waveform"] is not None:
                targ_wave = hypos[b]["targ_waveform"].detach().cpu().float()
                pred_wave = hypos[b]["waveform"].detach().cpu().float()
                tb_writer.add_audio(
                    f"inference_targ_{b}",
                    targ_wave,
                    num_updates,
                    sample_rate=self.sr
                )
                tb_writer.add_audio(
                    f"inference_pred_{b}",
                    pred_wave,
                    num_updates,
                    sample_rate=self.sr
                )


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


DEFAULT_V_MIN = np.log(1e-5)


def plot_tts_output(
        data_2d, title_2d, data_1d, title_1d, figsize=(24, 4),
        v_min=DEFAULT_V_MIN, v_max=3, ret_np=False, suptitle=""
):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    except ImportError:
        raise ImportError("Please install Matplotlib: pip install matplotlib")

    data_2d = [
        x.detach().cpu().float().numpy()
        if isinstance(x, torch.Tensor) else x for x in data_2d
    ]
    fig, axes = plt.subplots(1, len(data_2d) + 1, figsize=figsize)
    if suptitle:
        fig.suptitle(suptitle[:400])  # capped at 400 chars
    axes = [axes] if len(data_2d) == 0 else axes
    for ax, x, name in zip(axes, data_2d, title_2d):
        ax.set_title(name)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax.imshow(
            x, origin="lower", aspect="auto", vmin=max(x.min(), v_min),
            vmax=min(x.max(), v_max)
        )
        fig.colorbar(im, cax=cax, orientation='vertical')

    if isinstance(data_1d, torch.Tensor):
        data_1d = data_1d.detach().cpu().numpy()
    axes[-1].plot(data_1d)
    axes[-1].set_title(title_1d)
    plt.tight_layout()

    if ret_np:
        fig.canvas.draw()
        data = save_figure_to_numpy(fig)
        plt.close(fig)
        return data


def antidiag_indices(offset, min_i=0, max_i=None, min_j=0, max_j=None):
    """
    for a (3, 4) matrix with min_i=1, max_i=3, min_j=1, max_j=4, outputs

    offset=2 (1, 1),
    offset=3 (2, 1), (1, 2)
    offset=4 (2, 2), (1, 3)
    offset=5 (2, 3)

    constraints:
        i + j = offset
        min_j <= j < max_j
        min_i <= offset - j < max_i
    """
    if max_i is None:
        max_i = offset + 1
    if max_j is None:
        max_j = offset + 1
    min_j = max(min_j, offset - max_i + 1, 0)
    max_j = min(max_j, offset - min_i + 1, offset + 1)
    j = torch.arange(min_j, max_j)
    i = offset - j
    return torch.stack([i, j])


def batch_dynamic_time_warping(distance, shapes=None):
    """full batched DTW without any constraints

    distance:  (batchsize, max_M, max_N) matrix
    shapes: (batchsize,) vector specifying (M, N) for each entry
    """
    # ptr: 0=left, 1=up-left, 2=up
    ptr2dij = {0: (0, -1), 1: (-1, -1), 2: (-1, 0)}

    bsz, m, n = distance.size()
    cumdist = torch.zeros_like(distance)
    backptr = torch.zeros_like(distance).type(torch.int32) - 1

    # initialize
    cumdist[:, 0, :] = distance[:, 0, :].cumsum(dim=-1)
    cumdist[:, :, 0] = distance[:, :, 0].cumsum(dim=-1)
    backptr[:, 0, :] = 0
    backptr[:, :, 0] = 2

    # DP with optimized anti-diagonal parallelization, O(M+N) steps
    for offset in range(2, m + n - 1):
        ind = antidiag_indices(offset, 1, m, 1, n)
        c = torch.stack(
            [cumdist[:, ind[0], ind[1] - 1], cumdist[:, ind[0] - 1, ind[1] - 1],
             cumdist[:, ind[0] - 1, ind[1]], ],
            dim=2
        )
        v, b = c.min(axis=-1)
        backptr[:, ind[0], ind[1]] = b.int()
        cumdist[:, ind[0], ind[1]] = v + distance[:, ind[0], ind[1]]

    # backtrace
    pathmap = torch.zeros_like(backptr)
    for b in range(bsz):
        i = m - 1 if shapes is None else (shapes[b][0] - 1).item()
        j = n - 1 if shapes is None else (shapes[b][1] - 1).item()
        dtwpath = [(i, j)]
        while (i != 0 or j != 0) and len(dtwpath) < 10000:
            assert (i >= 0 and j >= 0)
            di, dj = ptr2dij[backptr[b, i, j].item()]
            i, j = i + di, j + dj
            dtwpath.append((i, j))
        dtwpath = dtwpath[::-1]
        indices = torch.from_numpy(np.array(dtwpath))
        pathmap[b, indices[:, 0], indices[:, 1]] = 1

    return cumdist, backptr, pathmap


def compute_l2_dist(x1, x2):
    """compute an (m, n) L2 distance matrix from (m, d) and (n, d) matrices"""
    return torch.cdist(x1.unsqueeze(0), x2.unsqueeze(0), p=2).squeeze(0).pow(2)


def compute_rms_dist(x1, x2):
    l2_dist = compute_l2_dist(x1, x2)
    return (l2_dist / x1.size(1)).pow(0.5)


def get_divisor(pathmap, normalize_type):
    if normalize_type is None:
        return 1
    elif normalize_type == "len1":
        return pathmap.size(0)
    elif normalize_type == "len2":
        return pathmap.size(1)
    elif normalize_type == "path":
        return pathmap.sum().item()
    else:
        raise ValueError(f"normalize_type {normalize_type} not supported")


def batch_compute_distortion(y1, y2, sr, feat_fn, dist_fn, normalize_type):
    d, s, x1, x2 = [], [], [], []
    for cur_y1, cur_y2 in zip(y1, y2):
        assert (cur_y1.ndim == 1 and cur_y2.ndim == 1)
        cur_x1 = feat_fn(cur_y1)
        cur_x2 = feat_fn(cur_y2)
        x1.append(cur_x1)
        x2.append(cur_x2)

        cur_d = dist_fn(cur_x1, cur_x2)
        d.append(cur_d)
        s.append(d[-1].size())
    max_m = max(ss[0] for ss in s)
    max_n = max(ss[1] for ss in s)
    d = torch.stack(
        [F.pad(dd, (0, max_n - dd.size(1), 0, max_m - dd.size(0))) for dd in d]
    )
    s = torch.LongTensor(s).to(d.device)
    cumdists, backptrs, pathmaps = batch_dynamic_time_warping(d, s)

    rets = []
    itr = zip(s, x1, x2, d, cumdists, backptrs, pathmaps)
    for (m, n), cur_x1, cur_x2, dist, cumdist, backptr, pathmap in itr:
        cumdist = cumdist[:m, :n]
        backptr = backptr[:m, :n]
        pathmap = pathmap[:m, :n]
        divisor = get_divisor(pathmap, normalize_type)

        distortion = cumdist[-1, -1] / divisor
        ret = distortion, (cur_x1, cur_x2, dist, cumdist, backptr, pathmap)
        rets.append(ret)
    return rets


def batch_mel_cepstral_distortion(
        y1, y2, sr, normalize_type="path", mfcc_fn=None
):
    """
    https://arxiv.org/pdf/2011.03568.pdf

    The root mean squared error computed on 13-dimensional MFCC using DTW for
    alignment. MFCC features are computed from an 80-channel log-mel
    spectrogram using a 50ms Hann window and hop of 12.5ms.

    y1: list of waveforms
    y2: list of waveforms
    sr: sampling rate
    """

    try:
        import torchaudio
    except ImportError:
        raise ImportError("Please install torchaudio: pip install torchaudio")

    if mfcc_fn is None or mfcc_fn.sample_rate != sr:
        melkwargs = {
            "n_fft": int(0.05 * sr), "win_length": int(0.05 * sr),
            "hop_length": int(0.0125 * sr), "f_min": 20,
            "n_mels": 80, "window_fn": torch.hann_window
        }
        mfcc_fn = torchaudio.transforms.MFCC(
            sr, n_mfcc=13, log_mels=True, melkwargs=melkwargs
        ).to(y1[0].device)
    return batch_compute_distortion(
        y1, y2, sr, lambda y: mfcc_fn(y).transpose(-1, -2), compute_rms_dist,
        normalize_type
    )
