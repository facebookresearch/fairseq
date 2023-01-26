import logging
import os

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader, Dataset

from .utils import Accuracy

logger = logging.getLogger(__name__)


def save_ckpt(model, path, model_class):
    ckpt = {
        "state_dict": model.state_dict(),
        "padding_token": model.padding_token,
        "model_class": model_class,
    }
    torch.save(ckpt, path)


def load_ckpt(path):
    ckpt = torch.load(path)
    ckpt["model_class"]["_target_"] = "emotion_models.duration_predictor.CnnPredictor"
    model = hydra.utils.instantiate(ckpt["model_class"])
    model.load_state_dict(ckpt["state_dict"])
    model.padding_token = ckpt["padding_token"]
    model = model.cpu()
    model.eval()
    return model


class Collator:
    def __init__(self, padding_idx):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        x = [item[0] for item in batch]
        lengths = [len(item) for item in x]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.padding_idx)
        y = [item[1] for item in batch]
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.padding_idx)
        mask = (x != self.padding_idx)
        return x, y, mask, lengths


class Predictor(nn.Module):
    def __init__(self, n_tokens, emb_dim):
        super(Predictor, self).__init__()
        self.n_tokens = n_tokens
        self.emb_dim = emb_dim
        self.padding_token = n_tokens
        # add 1 extra embedding for padding token, set the padding index to be the last token
        # (tokens from the clustering start at index 0)
        self.emb = nn.Embedding(n_tokens + 1, emb_dim, padding_idx=self.padding_token)

    def inflate_input(self, batch):
        """ get a sequence of tokens, predict their durations
        and inflate them accordingly """
        batch_durs = self.forward(batch)
        batch_durs = torch.exp(batch_durs) - 1
        batch_durs = batch_durs.round()
        output = []
        for seq, durs in zip(batch, batch_durs):
            inflated_seq = []
            for token, n in zip(seq, durs):
                if token == self.padding_token:
                    break
                n = int(n.item())
                token = int(token.item())
                inflated_seq.extend([token for _ in range(n)])
            output.append(inflated_seq)
        output = torch.LongTensor(output)
        return output


class CnnPredictor(Predictor):
    def __init__(self, n_tokens, emb_dim, channels, kernel, output_dim, dropout, n_layers):
        super(CnnPredictor, self).__init__(n_tokens=n_tokens, emb_dim=emb_dim)
        layers = [
            Rearrange("b t c -> b c t"),
            nn.Conv1d(emb_dim, channels, kernel_size=kernel, padding=(kernel - 1) // 2),
            Rearrange("b c t -> b t c"),
            nn.ReLU(),
            nn.LayerNorm(channels),
            nn.Dropout(dropout),
        ]
        for _ in range(n_layers-1):
            layers += [
                Rearrange("b t c -> b c t"),
                nn.Conv1d(channels, channels, kernel_size=kernel, padding=(kernel - 1) // 2),
                Rearrange("b c t -> b t c"),
                nn.ReLU(),
                nn.LayerNorm(channels),
                nn.Dropout(dropout),
            ]
        self.conv_layer = nn.Sequential(*layers)
        self.proj = nn.Linear(channels, output_dim)

    def forward(self, x):
        x = self.emb(x)
        x = self.conv_layer(x)
        x = self.proj(x)
        x = x.squeeze(-1)
        return x


def l2_log_loss(input, target):
    return F.mse_loss(
        input=input.float(),
        target=torch.log(target.float() + 1),
        reduce=False
    )


class DurationDataset(Dataset):
    def __init__(self, tsv_path, km_path, substring=""):
        lines = open(tsv_path, "r").readlines()
        self.root, self.tsv = lines[0], lines[1:]
        self.km = open(km_path, "r").readlines()
        logger.info(f"loaded {len(self.km)} files")

        if substring != "":
            tsv, km = [], []
            for tsv_line, km_line in zip(self.tsv, self.km):
                if substring.lower() in tsv_line.lower():
                    tsv.append(tsv_line)
                    km.append(km_line)
            self.tsv, self.km = tsv, km
            logger.info(f"after filtering: {len(self.km)} files")

    def __len__(self):
        return len(self.km)

    def __getitem__(self, i):
        x = self.km[i]
        x = x.split(" ")
        x = list(map(int, x))

        y = []
        xd = []
        count = 1
        for x1, x2 in zip(x[:-1], x[1:]):
            if x1 == x2:
                count += 1
                continue
            else:
                y.append(count)
                xd.append(x1)
                count = 1

        xd = torch.LongTensor(xd)
        y = torch.LongTensor(y)
        return xd, y


def train(cfg):
    device = "cuda:0"
    model = hydra.utils.instantiate(cfg[cfg.model]).to(device)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    # add 1 extra embedding for padding token, set the padding index to be the last token
    # (tokens from the clustering start at index 0)
    collate_fn = Collator(padding_idx=model.padding_token)
    logger.info(f"data: {cfg.train_tsv}")
    train_ds = DurationDataset(cfg.train_tsv, cfg.train_km, substring=cfg.substring)
    valid_ds = DurationDataset(cfg.valid_tsv, cfg.valid_km, substring=cfg.substring)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    best_loss = float("inf")
    for epoch in range(cfg.epochs):
        train_loss, train_loss_scaled = train_epoch(model, train_dl, l2_log_loss, optimizer, device)
        valid_loss, valid_loss_scaled, *acc = valid_epoch(model, valid_dl, l2_log_loss, device)
        acc0, acc1, acc2, acc3 = acc
        if valid_loss_scaled < best_loss:
            path = f"{os.getcwd()}/{cfg.substring}.ckpt"
            save_ckpt(model, path, cfg[cfg.model])
            best_loss = valid_loss_scaled
            logger.info(f"saved checkpoint: {path}")
            logger.info(f"[epoch {epoch}] train loss: {train_loss:.3f}, train scaled: {train_loss_scaled:.3f}")
            logger.info(f"[epoch {epoch}] valid loss: {valid_loss:.3f}, valid scaled: {valid_loss_scaled:.3f}")
            logger.info(f"acc: {acc0,acc1,acc2,acc3}")


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_loss_scaled = 0
    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        yhat = model(x)
        loss = criterion(yhat, y) * mask
        loss = torch.mean(loss)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        # get normal scale loss
        yhat_scaled = torch.exp(yhat) - 1
        yhat_scaled = torch.round(yhat_scaled)
        scaled_loss = torch.mean(torch.abs(yhat_scaled - y) * mask)
        epoch_loss_scaled += scaled_loss.item()
    return epoch_loss / len(loader), epoch_loss_scaled / len(loader)


def valid_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_loss_scaled = 0
    acc = Accuracy()
    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        yhat = model(x)
        loss = criterion(yhat, y) * mask
        loss = torch.mean(loss)
        epoch_loss += loss.item()
        # get normal scale loss
        yhat_scaled = torch.exp(yhat) - 1
        yhat_scaled = torch.round(yhat_scaled)
        scaled_loss = torch.sum(torch.abs(yhat_scaled - y) * mask) / mask.sum()
        acc.update(yhat_scaled[mask].view(-1).float(), y[mask].view(-1).float())
        epoch_loss_scaled += scaled_loss.item()
    logger.info(f"example y: {y[0, :10].tolist()}")
    logger.info(f"example yhat: {yhat_scaled[0, :10].tolist()}")
    acc0 = acc.acc(tol=0)
    acc1 = acc.acc(tol=1)
    acc2 = acc.acc(tol=2)
    acc3 = acc.acc(tol=3)
    logger.info(f"accs: {acc0,acc1,acc2,acc3}")
    return epoch_loss / len(loader), epoch_loss_scaled / len(loader), acc0, acc1, acc2, acc3


@hydra.main(config_path=".", config_name="duration_predictor.yaml")
def main(cfg):
    logger.info(f"{cfg}")
    train(cfg)


if __name__ == "__main__":
    main()
