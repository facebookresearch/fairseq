import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from inspect import isfunction
from operator import mul
from functools import reduce, wraps

from aml.multimodal_video.utils.einops.lib import rearrange, repeat
from aml.multimodal_video.utils.einops.lib.layers.torch import Rearrange

from fairseq.modules.local_attention import LocalAttention

# constants

TOKEN_SELF_ATTN_VALUE = -5e4
KMEAN_INIT_ITERS = 10

# helper functions


def exists(val):
    return val is not None


def identity(x, *args, **kwargs):
    return x


def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x


def cast_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if exists(cache):
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def to(t):
    return {"device": t.device, "dtype": t.dtype}


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


def is_empty(t):
    return t.nelement() == 0


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(2, expand_dim(indices, -1, last_dim))


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def scatter_mean(src, t, index, dim, eps=1e-5):
    numer = src.scatter_add(dim, index, t)
    denom = src.scatter_add(dim, index, torch.ones_like(t))
    return numer / (denom + eps)


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim : dim + 1] = split_dims
    return t.reshape(shape)


def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)


def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


# helper classes


def map_first_tuple_or_el(x, fn):
    if isinstance(x, tuple):
        return (fn(x[0]),) + x[1:]
    return fn(x)


class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim=-1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks <= 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim=self.dim)


class PreNorm(nn.ModuleList):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.residual_weight = nn.Parameter(torch.zeros(1))
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs)
        return map_first_tuple_or_el(x, lambda t: t * self.residual_weight)


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x):
        def norm(t):
            n = torch.norm(t, dim=-1, keepdim=True).clamp(min=self.eps)
            return t / n * self.g

        return map_first_tuple_or_el(x, norm)


class ProjectInOut(nn.Module):
    def __init__(self, fn, dim_in, dim_out, project_out=True):
        super().__init__()
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else identity

    def forward(self, x, **kwargs):
        x = self.project_in(x)
        x, loss = self.fn(x, **kwargs)
        x = self.project_out(x)
        return x, loss


class MatrixMultiply(nn.Module):
    def __init__(self, tensor, transpose=False):
        super().__init__()
        self.tensor = tensor
        self.transpose = transpose

    def forward(self, x):
        tensor = self.tensor
        if self.transpose:
            tensor = tensor.t()
        return x @ tensor


# positional embeddings


class DepthWiseConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, bias=True, causal=False):
        super().__init__()
        self.padding = (
            ((kernel_size - 1), 0) if causal else (kernel_size // 2, kernel_size // 2)
        )

        self.net = nn.Sequential(
            nn.Conv1d(
                dim_in,
                dim_in,
                kernel_size=kernel_size,
                groups=dim_in,
                stride=stride,
                bias=bias,
            ),
            nn.Conv1d(dim_in, dim_out, 1, bias=bias),
        )

    def forward(self, x):
        x = F.pad(x, self.padding, value=0.0)
        return self.net(x)


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer("emb", emb)

    def forward(self, x):
        return self.emb[None, : x.shape[1], :].to(x)


def rotate_every_two(x):
    x = rearrange(x, "... (d j) -> ... d j", j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d j -> ... (d j)")


def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, "() n (j d) -> n j d", j=2)
    sin, cos = sinu_pos.unbind(dim=-2)
    sin, cos = map(lambda t: repeat(t, "b n -> b (n j)", j=2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k


# kmeans related function and class


def update_kmeans_on_backwards(module):
    module.kmean_modules = find_modules(module, Kmeans)

    def hook(_, grad_in, grad_out):
        for m in module.kmean_modules:
            m.update()

    return module.register_backward_hook(hook)


def similarity(x, means):
    return torch.einsum("bhld,hcd->bhlc", x, means)


def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets


def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out


def kmeans_iter(x, means, buckets=None):
    b, h, _, d, dtype, num_clusters = *x.shape, x.dtype, means.shape[1]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_clusters).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, h, num_clusters, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)

    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means


def distribution(dists, window_size):
    _, topk_indices = dists.topk(k=window_size, dim=-2)
    indices = topk_indices.transpose(-2, -1)
    return indices.reshape(*indices.size()[:2], -1)


class Kmeans(nn.Module):
    def __init__(
        self, num_heads, head_dim, num_clusters, ema_decay=0.999, commitment=1e-4
    ):
        super().__init__()
        self.commitment = commitment
        self.ema_decay = ema_decay

        self.register_buffer("means", torch.randn(num_heads, num_clusters, head_dim))
        self.register_buffer("initted", torch.tensor(False))
        self.num_new_means = 0
        self.new_means = None

    @torch.no_grad()
    def init(self, x):
        if self.initted:
            return
        _, h, _, d, device, _ = *x.shape, x.device, x.dtype

        num_clusters = self.means.shape[1]

        means = x.transpose(0, 1).contiguous().view(h, -1, d)
        num_samples = means.shape[1]

        if num_samples >= num_clusters:
            indices = torch.randperm(num_samples, device=device)[:num_clusters]
        else:
            indices = torch.randint(0, num_samples, (num_clusters,), device=device)

        means = means[:, indices]

        for _ in range(KMEAN_INIT_ITERS):
            means = kmeans_iter(x, means)

        self.num_new_means = 0
        self.means.data.copy_(means)
        self.initted.data.copy_(torch.tensor(True))

    @torch.no_grad()
    def update(self, new_means=None):
        new_means = default(new_means, self.new_means)
        assert exists(new_means), "new kmeans has not been supplied"
        ema_inplace(self.means, new_means, self.ema_decay)

        del self.new_means
        self.new_means = None
        self.num_new_means = 0

    def forward(self, x, update_means=False):
        self.init(x)

        b, dtype = x.shape[0], x.dtype
        means = self.means.type(dtype)
        x = F.normalize(x, 2, dim=-1).type(dtype)

        with torch.no_grad():
            dists, buckets = dists_and_buckets(x, means)

        routed_means = batched_index_select(expand_dim(means, 0, b), buckets)
        loss = F.mse_loss(x, routed_means) * self.commitment

        if update_means:
            with torch.no_grad():
                means = kmeans_iter(x, means, buckets)
            self.new_means = ema(
                self.new_means, means, self.num_new_means / (self.num_new_means + 1)
            )
            self.num_new_means += 1

        return dists, loss


# kmeans attention class


class KmeansAttention(nn.Module):
    def __init__(
        self,
        num_clusters,
        window_size,
        num_heads,
        head_dim,
        causal=False,
        dropout=0.0,
        ema_decay=0.999,
        commitment=1e-4,
        context_window_size=None,
        receives_context=False,
        num_mem_kv=0,
        shared_qk=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.head_dim = head_dim

        self.window_size = window_size
        self.context_window_size = default(context_window_size, window_size)
        self.causal = causal

        self.shared_qk = shared_qk
        self.receives_context = receives_context
        self.kmeans = Kmeans(num_heads, head_dim, num_clusters, ema_decay, commitment)
        self.dropout = nn.Dropout(dropout)

        self.num_mem_kv = max(num_mem_kv, 1 if causal and not shared_qk else 0)
        self.mem_key = nn.Parameter(
            torch.randn(num_heads, num_clusters, self.num_mem_kv, head_dim)
        )
        self.mem_value = nn.Parameter(
            torch.randn(num_heads, num_clusters, self.num_mem_kv, head_dim)
        )

    def forward(self, q, k, v, query_mask=None, key_mask=None, **kwargs):
        b, h, t, d, kv_t, wsz, c_wsz, nc, device, dtype = (
            *q.shape,
            k.shape[2],
            self.window_size,
            self.context_window_size,
            self.num_clusters,
            q.device,
            q.dtype,
        )
        is_reverse = kwargs.pop("_reverse", False)

        out = torch.zeros_like(q, dtype=dtype)

        update_kmeans = self.training and not is_reverse

        key_mask = (
            default(key_mask, query_mask) if not self.receives_context else key_mask
        )
        kv_wsz = wsz if not self.receives_context else c_wsz

        wsz = min(wsz, t)
        kv_wsz = min(kv_wsz, kv_t)

        if not self.shared_qk or self.receives_context:
            dists, aux_loss = self.kmeans(torch.cat((q, k), dim=2), update_kmeans)
            q_dists, k_dists = split_at_index(2, t, dists)
            indices = distribution(q_dists, wsz)
            kv_indices = distribution(k_dists, kv_wsz)
        else:
            dists, aux_loss = self.kmeans(q, update_kmeans)
            k = F.normalize(k, dim=-1).to(q)
            indices = distribution(dists, wsz)
            kv_indices = indices

        q = batched_index_select(q, indices)
        k = batched_index_select(k, kv_indices)
        v = batched_index_select(v, kv_indices)

        reshape_with_window = lambda x: x.reshape(b, h, nc, -1, d)
        q, k, v = map(reshape_with_window, (q, k, v))

        m_k, m_v = map(
            lambda x: expand_dim(x, 0, b).to(q), (self.mem_key, self.mem_value)
        )
        k, v = map(lambda x: torch.cat(x, dim=3), ((m_k, k), (m_v, v)))

        dots = torch.einsum("bhnid,bhnjd->bhnij", q, k) * (d ** -0.5)

        mask_value = max_neg_value(dots)

        if exists(query_mask) or exists(key_mask):
            query_mask = default(
                query_mask, lambda: torch.ones((b, t), device=device).bool()
            )
            key_mask = default(
                key_mask, lambda: torch.ones((b, kv_t), device=device).bool()
            )

            q_mask = expand_dim(query_mask, 1, h).gather(2, indices)
            kv_mask = expand_dim(key_mask, 1, h).gather(2, kv_indices)
            q_mask, kv_mask = map(lambda t: t.reshape(b, h, nc, -1), (q_mask, kv_mask))
            mask = q_mask[:, :, :, :, None] * kv_mask[:, :, :, None, :]
            mask = F.pad(mask, (self.num_mem_kv, 0), value=1)
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal:
            q_mask, kv_mask = map(
                lambda t: t.reshape(b, h, nc, -1), (indices, kv_indices)
            )
            mask = q_mask[:, :, :, :, None] >= kv_mask[:, :, :, None, :]
            mask = F.pad(mask, (self.num_mem_kv, 0), value=1)
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.shared_qk:
            q_mask, kv_mask = map(
                lambda t: t.reshape(b, h, nc, -1), (indices, kv_indices)
            )
            mask = q_mask[:, :, :, :, None] == kv_mask[:, :, :, None, :]
            mask = F.pad(mask, (self.num_mem_kv, 0), value=0)
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask

        dots = dots.softmax(dim=-1)
        dots = self.dropout(dots)

        bo = torch.einsum("bhcij,bhcjd->bhcid", dots, v)
        so = torch.reshape(bo, (b, h, -1, bo.shape[-1])).type(dtype)
        out = scatter_mean(out, so, indices.unsqueeze(-1).expand_as(so), -2)
        return out, aux_loss


# feedforward


class GELU_(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
            )
        )


GELU = nn.GELU if hasattr(nn, "GELU") else GELU_


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0, activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


# self attention


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        max_seq_len,
        heads,
        local_attn_heads,
        window_size,
        dim_head=None,
        local_attn_window_size=None,
        local_attn_radius_blocks=1,
        causal=False,
        attn_dropout=0.0,
        dropout=0.0,
        kmeans_ema_decay=0.999,
        commitment_factor=1e-4,
        receives_context=False,
        context_window_size=None,
        rel_pos_emb=True,
        num_mem_kv=0,
        shared_qk=False,
        conv_query_kernel=9,
    ):
        super().__init__()
        assert (
            dim_head or (dim % heads) == 0
        ), "hidden dimension must be divisible by number of heads"
        assert (
            max_seq_len % window_size
        ) == 0, "maximum sequence length must be divisible by the target window size"
        assert (
            local_attn_heads <= heads
        ), "number of local attention heads must be less than total heads"
        assert not (
            receives_context and local_attn_heads > 0
        ), "local attention cannot be used for self attention with context"
        assert not (
            receives_context and causal
        ), "contextual attention layer cannot be causal"

        local_attn_window_size = default(local_attn_window_size, window_size)
        context_window_size = default(context_window_size, window_size)

        self.shared_qk = shared_qk
        self.receives_context = receives_context
        self.heads = heads
        self.local_attn_heads = local_attn_heads
        self.global_attn_heads = heads - local_attn_heads

        self.causal = causal
        self.window_size = window_size

        dim_head = default(dim_head, dim // heads)
        dim_heads = dim_head * heads
        self.dim_head = dim_head

        num_clusters = max_seq_len // window_size

        # local

        local_dim_heads = dim_head * self.local_attn_heads

        if self.local_attn_heads > 0:
            rel_pos_emb_config = (dim_head, local_attn_heads) if rel_pos_emb else None
            self.local_attn = LocalAttention(
                dim=dim_head,
                window_size=local_attn_window_size,
                causal=causal,
                dropout=attn_dropout,
                rel_pos_emb_config=rel_pos_emb_config,
                look_backward=local_attn_radius_blocks,
                look_forward=0 if causal else local_attn_radius_blocks,
            )
            self.local_to_qkv = nn.Linear(dim, 3 * local_dim_heads)

        # global

        global_dim_heads = dim_head * self.global_attn_heads

        if self.global_attn_heads > 0:
            self.global_attn = KmeansAttention(
                num_clusters,
                window_size,
                self.global_attn_heads,
                dim_head,
                causal=causal,
                dropout=attn_dropout,
                ema_decay=kmeans_ema_decay,
                commitment=commitment_factor,
                receives_context=receives_context,
                num_mem_kv=num_mem_kv,
                shared_qk=shared_qk,
            )

        self.to_q = nn.Sequential(
            Rearrange("b n c -> b c n"),
            DepthWiseConv1d(dim, global_dim_heads, conv_query_kernel, causal=causal),
            Rearrange("b c n -> b n c"),
        )

        self.to_v = nn.Linear(dim, global_dim_heads, bias=False)

        if not self.shared_qk:
            self.to_k = nn.Linear(dim, global_dim_heads, bias=False)

        # out

        self.to_out = nn.Linear(dim_heads, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        context=None,
        key_padding_mask=None,
        context_mask=None,
        pos_emb=None,
        **kwargs
    ):
        assert not (
            self.receives_context and not exists(context)
        ), "context must be passed if self attention is set to receive context"
        input_mask = key_padding_mask
        x = query.transpose(0, 1)
        b, t, _, h, dh = *x.shape, self.heads, self.dim_head
        has_local, has_global = map(
            lambda x: x > 0, (self.local_attn_heads, self.global_attn_heads)
        )

        split_heads = (
            lambda v: reshape_dim(v, -1, (-1, dh)).transpose(1, 2).contiguous()
        )

        if has_local:
            local_qkv = self.local_to_qkv(x).chunk(3, dim=-1)
            lq, lk, lv = map(split_heads, local_qkv)

        if has_global:
            kv_input = x if not self.receives_context else context

            q, v = self.to_q(x), self.to_v(kv_input)

            if not self.shared_qk:
                k = self.to_k(kv_input)
            else:
                k = self.to_q(kv_input) if self.receives_context else q

            q, k, v = map(split_heads, (q, k, v))

        out = []
        total_loss = torch.tensor(0.0, requires_grad=True, **to(x))

        if has_local:
            local_out = self.local_attn(lq, lk, lv, input_mask=input_mask)
            out.append(local_out)

        if has_global:
            if not self.receives_context and exists(pos_emb):
                q, k = apply_rotary_pos_emb(q, k, pos_emb)

            global_out, loss = self.global_attn(
                q, k, v, query_mask=input_mask, key_mask=context_mask
            )
            total_loss = total_loss + loss

            out.append(global_out)

        out = torch.cat(out, dim=1)
        out = out.reshape(b, h, t, -1).transpose(1, 2).reshape(b, t, -1)
        out = self.dropout(out.transpose(0, 1))
        # out = self.to_out(out)
        return out, total_loss
