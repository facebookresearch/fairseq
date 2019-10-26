# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import torch
from torch import Tensor


@torch.jit.script
def script_skip_tensor_list(x: List[Tensor], mask):
    res = [xi[mask] if xi.size(0) == mask.size(0) else xi[:, mask] for xi in x]
    outputs = []
    for i, t in enumerate(res):
        if t.numel() != 0:
            outputs.append(t)
        else:
            outputs.append(x[i])
    return outputs


@torch.jit.script
def script_skip_tensor(x: Tensor, mask):
    # None case
    if x.size(0) == 0:
        return x
    res = x[mask] if x.size(0) == mask.size(0) else x[:, mask]
    if res.numel() == 0:
        return x
    else:
        return res


@torch.jit.script
def script_skip_tensor_dict(x: Dict[str, Tensor], mask):
    outputs = {}
    for s, t in x.items():
        outputs[s] = t[mask] if t.size(0) == mask.size(0) else t[:, mask]
    return outputs


def skip_tensors(x, mask):
    """
    Getting sliced (dim=0) tensor by mask. Supporting tensor and list/dict of tensors.
    """
    if isinstance(x, int):
        return x

    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if x.size(0) == mask.size(0):
            return x[mask]
        elif x.size(1) == mask.size(0):
            return x[:, mask]

    if isinstance(x, list):
        return [skip_tensors(x_i, mask) for x_i in x]

    if isinstance(x, dict):
        return {k: skip_tensors(v, mask) for k, v in x.items()}

    raise NotImplementedError


@torch.jit.script
def expand_2d_or_3d_tensor(x, trg_dim: int, padding_idx: int):
    """
    Expand 2D/3D tensor on dim=1
    """
    if x is None:
        return None

    assert x.dim() == 2 or x.dim() == 3
    assert trg_dim >= x.size(1), (trg_dim, x.size())
    if trg_dim == x.size(1):
        return x

    dims = [x.size(0), trg_dim - x.size(1)]
    if x.dim() == 3:
        dims.append(x.size(2))
    x = torch.cat([x, torch.zeros(dims).to(x).fill_(padding_idx)], 1)

    return x


@torch.jit.script
def fill_tensors(x, mask, y, padding_idx: int):
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None or x.size()[0] == 0:
        return torch.empty([0])
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or (x.dim() == 3 and x.size(2) == y.size(2))

    n_selected = mask.sum()
    if n_selected == 0:
        return x
    assert n_selected == y.size(0)
    if n_selected == x.size(0):
        return y

    if x.size(1) < y.size(1):
        x = expand_2d_or_3d_tensor(x, y.size(1), padding_idx)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = torch.tensor(padding_idx).type_as(x)
        if x.dim() == 2:
            x[mask, :y.size(1)] = y
        else:
            x[mask, :y.size(1), :] = y
    else:
        x[mask] = y
    return x


def _apply_ins_masks(
    in_tokens, in_scores, mask_ins_pred, padding_idx, unk_idx, eos_idx
):

    in_masks = in_tokens.ne(padding_idx)
    in_lengths = in_masks.sum(1)

    # HACK: hacky way to shift all the paddings to eos first.
    in_tokens.masked_fill_(~in_masks, eos_idx)
    mask_ins_pred.masked_fill_(~in_masks[:, 1:], 0)

    out_lengths = in_lengths + mask_ins_pred.sum(1)
    out_max_len = out_lengths.max()
    out_masks = (
        torch.arange(out_max_len, device=out_lengths.device)[None, :]
        < out_lengths[:, None]
    )

    reordering = (mask_ins_pred + in_masks[:, 1:].long()).cumsum(1)
    out_tokens = (
        in_tokens.new_zeros(in_tokens.size(0), out_max_len)
        .fill_(padding_idx)
        .masked_fill_(out_masks, unk_idx)
    )
    out_tokens[:, 0] = in_tokens[:, 0]
    out_tokens.scatter_(1, reordering, in_tokens[:, 1:])

    out_scores = None
    if in_scores is not None:
        in_scores.masked_fill_(~in_masks, 0)
        out_scores = in_scores.new_zeros(*out_tokens.size())
        out_scores[:, 0] = in_scores[:, 0]
        out_scores.scatter_(1, reordering, in_scores[:, 1:])

    return out_tokens, out_scores


def _apply_ins_words(in_tokens, in_scores, word_ins_pred, word_ins_scores, unk_idx):
    word_ins_masks = in_tokens.eq(unk_idx)
    out_tokens = in_tokens.masked_scatter(word_ins_masks, word_ins_pred[word_ins_masks])

    if in_scores is not None:
        out_scores = in_scores.masked_scatter(
            word_ins_masks, word_ins_scores[word_ins_masks]
        )
    else:
        out_scores = None

    return out_tokens, out_scores


def _apply_del_words(
    in_tokens, in_scores, in_attn, word_del_pred, padding_idx, bos_idx, eos_idx
):
    # apply deletion to a tensor
    in_masks = in_tokens.ne(padding_idx)
    bos_eos_masks = in_tokens.eq(bos_idx) | in_tokens.eq(eos_idx)

    max_len = in_tokens.size(1)
    word_del_pred.masked_fill_(~in_masks, 1)
    word_del_pred.masked_fill_(bos_eos_masks, 0)

    reordering = (
        torch.arange(max_len, device=in_tokens.device)[None, :]
        .expand_as(in_tokens)
        .contiguous()
        .masked_fill_(word_del_pred, max_len)
        .sort(1)[1]
    )

    out_tokens = in_tokens.masked_fill(word_del_pred, padding_idx).gather(1, reordering)

    out_scores = None
    if in_scores is not None:
        out_scores = in_scores.masked_fill(word_del_pred, 0).gather(1, reordering)

    out_attn = None
    if in_attn is not None:
        _mask = word_del_pred[:, :, None].expand_as(in_attn)
        _reordering = reordering[:, :, None].expand_as(in_attn)
        out_attn = in_attn.masked_fill(_mask, 0.).gather(1, _reordering)

    return out_tokens, out_scores, out_attn
