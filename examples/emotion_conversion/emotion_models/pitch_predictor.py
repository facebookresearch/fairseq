import logging
import os
import random
import sys
from collections import defaultdict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from scipy.io.wavfile import read
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

dir_path = os.path.dirname(__file__)
resynth_path = os.path.dirname(dir_path) + "/speech-resynthesis"
sys.path.append(resynth_path)
from dataset import parse_speaker, parse_style
from .utils import F0Stat

MAX_WAV_VALUE = 32768.0
logger = logging.getLogger(__name__)


def quantize_f0(speaker_to_f0, nbins, normalize, log):
    f0_all = []
    for speaker, f0 in speaker_to_f0.items():
        f0 = f0.raw_data
        if log:
            f0 = f0.log()
        mean = speaker_to_f0[speaker].mean_log if log else speaker_to_f0[speaker].mean
        std = speaker_to_f0[speaker].std_log if log else speaker_to_f0[speaker].std
        if normalize == "mean":
            f0 = f0 - mean
        elif normalize == "meanstd":
            f0 = (f0 - mean) / std
        f0_all.extend(f0.tolist())

    hist, bin_x = np.histogram(f0_all, 100000)
    cum_hist = np.cumsum(hist) / len(f0_all) * 100

    bin_offset = []
    bin_size = 100 / nbins
    threshold = bin_size
    for i in range(nbins - 1):
        index = (np.abs(cum_hist - threshold)).argmin()
        bin_offset.append(bin_x[index])
        threshold += bin_size
    bins = np.array(bin_offset)
    bins = torch.FloatTensor(bins)

    return bins


def save_ckpt(model, path, model_class, f0_min, f0_max, f0_bins, speaker_stats):
    ckpt = {
        "state_dict": model.state_dict(),
        "padding_token": model.padding_token,
        "model_class": model_class,
        "speaker_stats": speaker_stats,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "f0_bins": f0_bins,
    }
    torch.save(ckpt, path)


def load_ckpt(path):
    ckpt = torch.load(path)
    ckpt["model_class"]["_target_"] = "emotion_models.pitch_predictor.CnnPredictor"
    model = hydra.utils.instantiate(ckpt["model_class"])
    model.load_state_dict(ckpt["state_dict"])
    model.setup_f0_stats(
        ckpt["f0_min"],
        ckpt["f0_max"],
        ckpt["f0_bins"],
        ckpt["speaker_stats"],
    )
    return model


def freq2bin(f0, f0_min, f0_max, bins):
    f0 = f0.clone()
    f0[f0 < f0_min] = f0_min
    f0[f0 > f0_max] = f0_max
    f0 = torch.bucketize(f0, bins)
    return f0


def bin2freq(x, f0_min, f0_max, bins, mode):
    n_bins = len(bins) + 1
    assert x.shape[-1] == n_bins
    bins = torch.cat([torch.tensor([f0_min]), bins]).to(x.device)
    if mode == "mean":
        f0 = (x * bins).sum(-1, keepdims=True) / x.sum(-1, keepdims=True)
    elif mode == "argmax":
        idx = F.one_hot(x.argmax(-1), num_classes=n_bins)
        f0 = (idx * bins).sum(-1, keepdims=True)
    else:
        raise NotImplementedError()
    return f0[..., 0]


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def l1_loss(input, target):
    return F.l1_loss(input=input.float(), target=target.float(), reduce=False)


def l2_loss(input, target):
    return F.mse_loss(input=input.float(), target=target.float(), reduce=False)


class Collator:
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        tokens = [item[0] for item in batch]
        lengths = [len(item) for item in tokens]
        tokens = torch.nn.utils.rnn.pad_sequence(
            tokens, batch_first=True, padding_value=self.padding_idx
        )
        f0 = [item[1] for item in batch]
        f0 = torch.nn.utils.rnn.pad_sequence(
            f0, batch_first=True, padding_value=self.padding_idx
        )
        f0_raw = [item[2] for item in batch]
        f0_raw = torch.nn.utils.rnn.pad_sequence(
            f0_raw, batch_first=True, padding_value=self.padding_idx
        )
        spk = [item[3] for item in batch]
        spk = torch.LongTensor(spk)
        gst = [item[4] for item in batch]
        gst = torch.LongTensor(gst)
        mask = tokens != self.padding_idx
        return tokens, f0, f0_raw, spk, gst, mask, lengths


class CnnPredictor(nn.Module):
    def __init__(
        self,
        n_tokens,
        emb_dim,
        channels,
        kernel,
        dropout,
        n_layers,
        spk_emb,
        gst_emb,
        n_bins,
        f0_pred,
        f0_log,
        f0_norm,
    ):
        super(CnnPredictor, self).__init__()
        self.n_tokens = n_tokens
        self.emb_dim = emb_dim
        self.f0_log = f0_log
        self.f0_pred = f0_pred
        self.padding_token = n_tokens
        self.f0_norm = f0_norm
        # add 1 extra embedding for padding token, set the padding index to be the last token
        # (tokens from the clustering start at index 0)
        self.token_emb = nn.Embedding(
            n_tokens + 1, emb_dim, padding_idx=self.padding_token
        )

        self.spk_emb = spk_emb
        self.gst_emb = nn.Embedding(20, gst_emb)
        self.setup = False

        feats = emb_dim + gst_emb
        # feats = emb_dim + gst_emb + (256 if spk_emb else 0)
        layers = [
            nn.Sequential(
                Rearrange("b t c -> b c t"),
                nn.Conv1d(
                    feats, channels, kernel_size=kernel, padding=(kernel - 1) // 2
                ),
                Rearrange("b c t -> b t c"),
                nn.ReLU(),
                nn.LayerNorm(channels),
                nn.Dropout(dropout),
            )
        ]
        for _ in range(n_layers - 1):
            layers += [
                nn.Sequential(
                    Rearrange("b t c -> b c t"),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel,
                        padding=(kernel - 1) // 2,
                    ),
                    Rearrange("b c t -> b t c"),
                    nn.ReLU(),
                    nn.LayerNorm(channels),
                    nn.Dropout(dropout),
                )
            ]
        self.conv_layer = nn.ModuleList(layers)
        self.proj = nn.Linear(channels, n_bins)

    def forward(self, x, gst=None):
        x = self.token_emb(x)
        feats = [x]

        if gst is not None:
            gst = self.gst_emb(gst)
            gst = rearrange(gst, "b c -> b c 1")
            gst = F.interpolate(gst, x.shape[1])
            gst = rearrange(gst, "b c t -> b t c")
            feats.append(gst)

        x = torch.cat(feats, dim=-1)

        for i, conv in enumerate(self.conv_layer):
            if i != 0:
                x = conv(x) + x
            else:
                x = conv(x)

        x = self.proj(x)
        x = x.squeeze(-1)

        if self.f0_pred == "mean":
            x = torch.sigmoid(x)
        elif self.f0_pred == "argmax":
            x = torch.softmax(x, dim=-1)
        else:
            raise NotImplementedError
        return x

    def setup_f0_stats(self, f0_min, f0_max, f0_bins, speaker_stats):
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_bins = f0_bins
        self.speaker_stats = speaker_stats
        self.setup = True

    def inference(self, x, spk_id=None, gst=None):
        assert (
            self.setup == True
        ), "make sure that `setup_f0_stats` was called before inference!"
        probs = self(x, gst)
        f0 = bin2freq(probs, self.f0_min, self.f0_max, self.f0_bins, self.f0_pred)
        for i in range(f0.shape[0]):
            mean = (
                self.speaker_stats[spk_id[i].item()].mean_log
                if self.f0_log
                else self.speaker_stats[spk_id[i].item()].mean
            )
            std = (
                self.speaker_stats[spk_id[i].item()].std_log
                if self.f0_log
                else self.speaker_stats[spk_id[i].item()].std
            )
            if self.f0_norm == "mean":
                f0[i] = f0[i] + mean
            if self.f0_norm == "meanstd":
                f0[i] = (f0[i] * std) + mean
        if self.f0_log:
            f0 = f0.exp()
        return f0


class PitchDataset(Dataset):
    def __init__(
        self,
        tsv_path,
        km_path,
        substring,
        spk,
        spk2id,
        gst,
        gst2id,
        f0_bins,
        f0_bin_type,
        f0_smoothing,
        f0_norm,
        f0_log,
    ):
        lines = open(tsv_path, "r").readlines()
        self.root, self.tsv = lines[0], lines[1:]
        self.root = self.root.strip()
        self.km = open(km_path, "r").readlines()
        print(f"loaded {len(self.km)} files")

        self.spk = spk
        self.spk2id = spk2id
        self.gst = gst
        self.gst2id = gst2id

        self.f0_bins = f0_bins
        self.f0_smoothing = f0_smoothing
        self.f0_norm = f0_norm
        self.f0_log = f0_log

        if substring != "":
            tsv, km = [], []
            for tsv_line, km_line in zip(self.tsv, self.km):
                if substring.lower() in tsv_line.lower():
                    tsv.append(tsv_line)
                    km.append(km_line)
            self.tsv, self.km = tsv, km
            print(f"after filtering: {len(self.km)} files")

        self.speaker_stats = self._compute_f0_stats()
        self.f0_min, self.f0_max = self._compute_f0_minmax()
        if f0_bin_type == "adaptive":
            self.f0_bins = quantize_f0(
                self.speaker_stats, self.f0_bins, self.f0_norm, self.f0_log
            )
        elif f0_bin_type == "uniform":
            self.f0_bins = torch.linspace(self.f0_min, self.f0_max, self.f0_bins + 1)[
                1:-1
            ]
        else:
            raise NotImplementedError
        print(f"f0 min: {self.f0_min}, f0 max: {self.f0_max}")
        print(f"bins: {self.f0_bins} (shape: {self.f0_bins.shape})")

    def __len__(self):
        return len(self.km)

    def _load_f0(self, tsv_line):
        tsv_line = tsv_line.split("\t")[0]
        f0 = self.root + "/" + tsv_line.replace(".wav", ".yaapt.f0.npy")
        f0 = np.load(f0)
        f0 = torch.FloatTensor(f0)
        return f0

    def _preprocess_f0(self, f0, spk):
        mask = f0 != -999999  # process all frames
        # mask = (f0 != 0)  # only process voiced frames
        mean = (
            self.speaker_stats[spk].mean_log
            if self.f0_log
            else self.speaker_stats[spk].mean
        )
        std = (
            self.speaker_stats[spk].std_log
            if self.f0_log
            else self.speaker_stats[spk].std
        )
        if self.f0_log:
            f0[f0 == 0] = 1e-5
            f0[mask] = f0[mask].log()
        if self.f0_norm == "mean":
            f0[mask] = f0[mask] - mean
        if self.f0_norm == "meanstd":
            f0[mask] = (f0[mask] - mean) / std
        return f0

    def _compute_f0_minmax(self):
        f0_min, f0_max = float("inf"), -float("inf")
        for tsv_line in tqdm(self.tsv, desc="computing f0 minmax"):
            spk = self.spk2id[parse_speaker(tsv_line, self.spk)]
            f0 = self._load_f0(tsv_line)
            f0 = self._preprocess_f0(f0, spk)
            f0_min = min(f0_min, f0.min().item())
            f0_max = max(f0_max, f0.max().item())
        return f0_min, f0_max

    def _compute_f0_stats(self):
        from functools import partial

        speaker_stats = defaultdict(partial(F0Stat, True))
        for tsv_line in tqdm(self.tsv, desc="computing speaker stats"):
            spk = self.spk2id[parse_speaker(tsv_line, self.spk)]
            f0 = self._load_f0(tsv_line)
            mask = f0 != 0
            f0 = f0[mask]  # compute stats only on voiced parts
            speaker_stats[spk].update(f0)
        return speaker_stats

    def __getitem__(self, i):
        x = self.km[i]
        x = x.split(" ")
        x = list(map(int, x))
        x = torch.LongTensor(x)

        gst = parse_style(self.tsv[i], self.gst)
        gst = self.gst2id[gst]
        spk = parse_speaker(self.tsv[i], self.spk)
        spk = self.spk2id[spk]

        f0_raw = self._load_f0(self.tsv[i])
        f0 = self._preprocess_f0(f0_raw.clone(), spk)

        f0 = F.interpolate(f0.unsqueeze(0).unsqueeze(0), x.shape[0])[0, 0]
        f0_raw = F.interpolate(f0_raw.unsqueeze(0).unsqueeze(0), x.shape[0])[0, 0]

        f0 = freq2bin(f0, f0_min=self.f0_min, f0_max=self.f0_max, bins=self.f0_bins)
        f0 = F.one_hot(f0.long(), num_classes=len(self.f0_bins) + 1).float()
        if self.f0_smoothing > 0:
            f0 = torch.tensor(
                gaussian_filter1d(f0.float().numpy(), sigma=self.f0_smoothing)
            )
        return x, f0, f0_raw, spk, gst


def train(cfg):
    device = "cuda:0"
    # add 1 extra embedding for padding token, set the padding index to be the last token
    # (tokens from the clustering start at index 0)
    padding_token = cfg.n_tokens
    collate_fn = Collator(padding_idx=padding_token)
    train_ds = PitchDataset(
        cfg.train_tsv,
        cfg.train_km,
        substring=cfg.substring,
        spk=cfg.spk,
        spk2id=cfg.spk2id,
        gst=cfg.gst,
        gst2id=cfg.gst2id,
        f0_bins=cfg.f0_bins,
        f0_bin_type=cfg.f0_bin_type,
        f0_smoothing=cfg.f0_smoothing,
        f0_norm=cfg.f0_norm,
        f0_log=cfg.f0_log,
    )
    valid_ds = PitchDataset(
        cfg.valid_tsv,
        cfg.valid_km,
        substring=cfg.substring,
        spk=cfg.spk,
        spk2id=cfg.spk2id,
        gst=cfg.gst,
        gst2id=cfg.gst2id,
        f0_bins=cfg.f0_bins,
        f0_bin_type=cfg.f0_bin_type,
        f0_smoothing=cfg.f0_smoothing,
        f0_norm=cfg.f0_norm,
        f0_log=cfg.f0_log,
    )
    train_dl = DataLoader(
        train_ds,
        num_workers=0,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_dl = DataLoader(
        valid_ds, num_workers=0, batch_size=16, shuffle=False, collate_fn=collate_fn
    )

    f0_min = train_ds.f0_min
    f0_max = train_ds.f0_max
    f0_bins = train_ds.f0_bins
    speaker_stats = train_ds.speaker_stats

    model = hydra.utils.instantiate(cfg["model"]).to(device)
    model.setup_f0_stats(f0_min, f0_max, f0_bins, speaker_stats)

    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    best_loss = float("inf")
    for epoch in range(cfg.epochs):
        train_loss, train_l2_loss, train_l2_voiced_loss = run_epoch(
            model, train_dl, optimizer, device, cfg, mode="train"
        )
        valid_loss, valid_l2_loss, valid_l2_voiced_loss = run_epoch(
            model, valid_dl, None, device, cfg, mode="valid"
        )
        print(
            f"[epoch {epoch}] train loss: {train_loss:.3f}, l2 loss: {train_l2_loss:.3f}, l2 voiced loss: {train_l2_voiced_loss:.3f}"
        )
        print(
            f"[epoch {epoch}] valid loss: {valid_loss:.3f}, l2 loss: {valid_l2_loss:.3f}, l2 voiced loss: {valid_l2_voiced_loss:.3f}"
        )
        if valid_l2_voiced_loss < best_loss:
            path = f"{os.getcwd()}/pitch_predictor.ckpt"
            save_ckpt(model, path, cfg["model"], f0_min, f0_max, f0_bins, speaker_stats)
            best_loss = valid_l2_voiced_loss
            print(f"saved checkpoint: {path}")
        print(f"[epoch {epoch}] best loss: {best_loss:.3f}")


def run_epoch(model, loader, optimizer, device, cfg, mode):
    if mode == "train":
        model.train()
    else:
        model.eval()

    epoch_loss = 0
    l1 = 0
    l1_voiced = 0
    for x, f0_bin, f0_raw, spk_id, gst, mask, _ in tqdm(loader):
        x, f0_bin, f0_raw, spk_id, gst, mask = (
            x.to(device),
            f0_bin.to(device),
            f0_raw.to(device),
            spk_id.to(device),
            gst.to(device),
            mask.to(device),
        )
        b, t, n_bins = f0_bin.shape
        yhat = model(x, gst)
        nonzero_mask = (f0_raw != 0).logical_and(mask)
        yhat_raw = model.inference(x, spk_id, gst)
        expanded_mask = mask.unsqueeze(-1).expand(-1, -1, n_bins)
        if cfg.f0_pred == "mean":
            loss = F.binary_cross_entropy(
                yhat[expanded_mask], f0_bin[expanded_mask]
            ).mean()
        elif cfg.f0_pred == "argmax":
            loss = F.cross_entropy(
                rearrange(yhat, "b t d -> (b t) d"),
                rearrange(f0_bin.argmax(-1), "b t -> (b t)"),
                reduce=False,
            )
            loss = rearrange(loss, "(b t) -> b t", b=b, t=t)
            loss = (loss * mask).sum() / mask.float().sum()
        else:
            raise NotImplementedError
        l1 += F.l1_loss(yhat_raw[mask], f0_raw[mask]).item()
        l1_voiced += F.l1_loss(yhat_raw[nonzero_mask], f0_raw[nonzero_mask]).item()
        epoch_loss += loss.item()

        if mode == "train":
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    print(f"{mode} example    y: {f0_bin.argmax(-1)[0, 50:60].tolist()}")
    print(f"{mode} example yhat: {yhat.argmax(-1)[0, 50:60].tolist()}")
    print(f"{mode} example    y: {f0_raw[0, 50:60].round().tolist()}")
    print(f"{mode} example yhat: {yhat_raw[0, 50:60].round().tolist()}")
    return epoch_loss / len(loader), l1 / len(loader), l1_voiced / len(loader)


@hydra.main(config_path=dir_path, config_name="pitch_predictor.yaml")
def main(cfg):
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    from hydra.core.hydra_config import HydraConfig

    overrides = {
        x.split("=")[0]: x.split("=")[1]
        for x in HydraConfig.get().overrides.task
        if "/" not in x
    }
    print(f"{cfg}")
    train(cfg)


if __name__ == "__main__":
    main()
