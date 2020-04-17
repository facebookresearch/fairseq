"""
Pytorch modules for T5 model.

The code based on:
 * https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_t5.py

The original code is licensed under Apache 2.0.

"""

import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """Construct a layernorm module in the T5 style (no bias and no substraction of mean)

        Args:
        hidden_size: size of a hidden state
        eps: epsilon to use to variance normalization

        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.variance_epsilon)
        return self.weight * x


class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        h = self.wi(hidden_states)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.wo(h)
        return h


class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.DenseReluDense = T5DenseReluDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x)
        layer_output = hidden_states + self.dropout(y)
        return layer_output


class T5Attention(nn.Module):
    NEW_ID = itertools.count()

    def __init__(self, config, has_relative_attention_bias=False, is_cross_attention=False):
        super().__init__()
        self.layer_id = next(T5Attention.NEW_ID)
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.output_attentions = getattr(config, 'output_attentions', False)
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv
        self.is_cross_attention = is_cross_attention

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            if not is_cross_attention:
                self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    # def prune_heads(self, heads):
    #     if len(heads) == 0:
    #         return
    #     mask = torch.ones(self.n_heads, self.d_kv)
    #     heads = set(heads) - self.pruned_heads
    #     for head in heads:
    #         head -= sum(1 if h < head else 0 for h in self.pruned_heads)
    #         mask[head] = 0
    #     mask = mask.view(-1).contiguous().eq(1)
    #     index = torch.arange(len(mask))[mask].long()
    #     # Prune linear layers
    #     self.q = prune_linear_layer(self.q, index)
    #     self.k = prune_linear_layer(self.k, index)
    #     self.v = prune_linear_layer(self.v, index)
    #     self.o = prune_linear_layer(self.o, index, dim=1)
    #     # Update hyper params
    #     self.n_heads = self.n_heads - len(heads)
    #     self.inner_dim = self.d_kv * self.n_heads
    #     self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=not self.is_decoder,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(self.k.weight.device)
        if self.is_cross_attention:
            values = nn.functional.embedding(
                rp_bucket,
                torch.zeros(
                    self.relative_attention_num_buckets,
                    self.n_heads,
                    device=self.k.weight.device
                )
            )
        else:
            values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, input, mask=None, kv=None, position_bias=None, cache=None, head_mask=None):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        bs, qlen, dim = input.size()
        if kv is None:
            klen = qlen if cache is None else cache["slen"] + qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)
        if kv is None:
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif cache is None or self.layer_id not in cache:
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if cache is not None:
            if self.layer_id in cache:
                if kv is None:
                    k_, v_ = cache[self.layer_id]
                    k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                    v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
                else:
                    k, v = cache[self.layer_id]
            cache[self.layer_id] = (k, v)

        # q = q / math.sqrt(dim_per_head)                                     # No scaling in T5
        scores = torch.einsum("bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(qlen, klen)
                if mask is not None:
                    position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)
                scores += position_bias
        else:
            scores += position_bias

        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context,)
        if self.output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, head_mask=None):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x, mask=attention_mask, position_bias=position_bias, head_mask=head_mask
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            is_cross_attention=True)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, kv, attention_mask=None, position_bias=None, head_mask=None):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            norm_x, mask=attention_mask, kv=kv, position_bias=position_bias, head_mask=head_mask
        )
        y = attention_output[0]
        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, has_relative_attention_bias=has_relative_attention_bias))
            self.layer.append(T5LayerFF(config))
        else:
            self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        head_mask=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states, attention_mask=attention_mask, position_bias=position_bias, head_mask=head_mask
        )
        hidden_states = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # Keep self-attention outputs and relative position weights

        if not self.is_decoder:
            hidden_states = self.layer[1](hidden_states)
        else:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
            )
            hidden_states = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:]
            )  # Keep cross-attention outputs and relative position weights
            hidden_states = self.layer[2](hidden_states)

        outputs = (hidden_states,) + outputs  # add attentions if we output them
        return outputs  # hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), \
        # (cross-attention position bias)


class T5Stack(nn.Module):
    def __init__(self, config, embed_tokens=None):
        super().__init__()
        self.config = config
        self.output_attentions = getattr(config, 'output_attentions', False)
        self.output_hidden_states = getattr(config, 'output_hidden_states', False)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to intialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(batch_size, encoder_seq_length).to(inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                seq_ids = torch.arange(seq_length, device=inputs_embeds.device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(attention_mask)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -1e9 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
        # extended_attention_mask = (extended_attention_mask == extended_attention_mask.transpose(-1, -2))

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9

        if self.is_decoder:
            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

            # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
            # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow/transformer/transformer_layers.py#L270
            # encoder_extended_attention_mask = (encoder_extended_attention_mask == encoder_extended_attention_mask.transpose(-1, -2))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_layers

        all_hidden_states = ()
        all_attentions = ()
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        for i, layer_module in enumerate(self.block):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
            )
            # layer_outputs is a tuple with:
            # hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states = layer_outputs[0]
            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[2 if self.output_attentions else 1]
                if self.is_decoder:
                    encoder_decoder_position_bias = layer_outputs[4 if self.output_attentions else 2]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)  # We keep only self-attention weights for now

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)
