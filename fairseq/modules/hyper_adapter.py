import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    FairseqDropout,
)
from fairseq import utils
logger = logging.getLogger(__name__)


# We want to use the same init as in transformer.py, but to avoid circular imports
# we temporarily copy the Linear() function here
def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class ResidualBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            dropout: float,
            activation_fn: str,
            normalize_before : bool
    ):
        """
        The residual block used in each layer of the encoder of the hyper-network
        Args:
            hidden_dim: the dimensionality of the input representations
            dropout: dropout probability for the activations of the block
            activation_fn: the activation function used in the block
            normalize_before: apply pre-norm
        """
        super().__init__()
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout_module = FairseqDropout(dropout)
        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.normalize_before = normalize_before

    def forward(self, x: torch.Tensor):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x += residual
        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class HyperNetworkEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
            activation_fn: str,
            normalize_before: bool
    ):
        """
        This the encoder of the hyper-network. It takes as input the hyper-network's embedding
        and produces an output representation, which is then feed to each projection that produces the meta-weights.
        Args:
            input_dim: the dimensionality of the input representation
            hidden_dim: the dimensionality of the hidden representations
            num_layers: the number of (residual) blocks of the encoder
            dropout: dropout probability for the (residual) blocks
            activation_fn: the activation function used in the (residual) blocks
            normalize_before: apply pre-norm
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.normalize_before = normalize_before
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout_module = FairseqDropout(dropout)
        self.fc_input = Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            ResidualBlock(
                hidden_dim, 
                dropout, 
                activation_fn, 
                normalize_before
            ) for _ in range(self.num_layers)
        ])

        self.layer_norm = nn.LayerNorm(input_dim) if normalize_before else nn.LayerNorm(output_dim)

        
    def forward(self, x: torch.Tensor):
        if self.normalize_before:
            x = self.layer_norm(x)
            
        x = self.fc_input(x)
        x = self.activation_fn(x)
        x = self.dropout_module(x)

        for layers in self.layers:
            x = layers(x)

        if not self.normalize_before:
            x = self.layer_norm(x)

        return x


class HyperNetwork(nn.Module):
    def __init__(self,
                 num_languages: int,
                 num_layers: int,
                 lang_embedding_dim: int,
                 layer_embedding_dim: int,
                 mainnet_input_dim: int,
                 bottleneck_dim: int,
                 hidden_dim: int,
                 num_hidden_layers: int,
                 dropout: float,
                 activation_fn: str,
                 generate_layernorm: bool,
                 normalize_before: bool,
                 language_embedding_tied: bool,
                 init_method: str):
        """
        A hyper-network model class that produces the parameters of an adapter layer
        input -> embedding -> encoder -> projections -> hyper-adapters
        Args:
            num_languages: The total number of languages in the data
            num_layers: The total number of Transformer layers (enc+dec)
            lang_embedding_dim: The size of the language embeddings
            layer_embedding_dim: The size of the layer embeddings
            mainnet_input_dim: The size of the main-network representations
            bottleneck_dim: The bottleneck size of the (generated) hyper-adapters
            hidden_dim: The size of the hidden layers of the hyper-network encoder
            num_hidden_layers: The number of the hyper-network encoder layers
            dropout: The dropout probability for regularizing the hyper-network encoder
            activation_fn: The activation function used in the hyper-network and hyper-adapters
            nn.layernorm_input: Whether to apply LN to the hyper-network input
                (i.e., before encoder)
            nn.layernorm_output: Whether to apply LN to the hyper-network output
                (i.e., before generating the hyper-adapter weights)
            generate_layernorm: Whether to generate the input-specific
                LN parameters of each hyper-adapters. If false, then we use non-trainable LN
            language_embedding_tied: Whether to tie the source and target language embedding
            init_method: How to initialize the hyper-network projection layers
        """
        super().__init__()
        self.num_languages = num_languages
        self.num_layers = num_layers
        self.lang_embedding_dim = lang_embedding_dim
        self.layer_embedding_dim = layer_embedding_dim
        self.mainnet_input_dim = mainnet_input_dim
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.generate_layernorm = generate_layernorm
        self.language_embedding_tied = language_embedding_tied
        self.normalize_before = normalize_before
        self.dropout_module = FairseqDropout
        self.init_method = init_method

        # compute the size of the hyper-network output weights
        self.down_weights_dim = mainnet_input_dim * bottleneck_dim
        self.down_bias_dim = bottleneck_dim
        self.up_weights_dim = bottleneck_dim * mainnet_input_dim
        self.up_bias_dim = mainnet_input_dim
        self.layer_norm_gamma_dim = mainnet_input_dim
        self.layer_norm_beta_dim = mainnet_input_dim

        # hyper-network input encoding
        self.task_embedding_dim = 2 * layer_embedding_dim + layer_embedding_dim

        # Embedding Layers
        self.src_lang_emb = nn.Embedding(num_languages, lang_embedding_dim)
        self.tgt_lang_emb = nn.Embedding(num_languages, lang_embedding_dim)
        self.layer_emb = nn.Embedding(num_layers, layer_embedding_dim)

        if language_embedding_tied:
            self.src_lang_emb.weight = self.tgt_lang_emb.weight

        # Encoder(s)
        self.encoder = self.build_encoder()

        # Projection Heads, that generate hyper-adapter weights
        self.head_down_weights = Linear(self.encoder.output_dim,
                                        self.down_weights_dim)
        self.head_down_bias = Linear(self.encoder.output_dim,
                                     self.down_bias_dim)
        self.head_up_weights = Linear(self.encoder.output_dim,
                                      self.up_weights_dim)
        self.head_up_bias = Linear(self.encoder.output_dim,
                                   self.up_bias_dim)

        if self.generate_layernorm:
            self.head_layer_norm_gamma = Linear(self.encoder.output_dim,
                                                self.layer_norm_gamma_dim)
            self.head_layer_norm_beta = Linear(self.encoder.output_dim,
                                               self.layer_norm_beta_dim)

        # regularization, activations etc.
        self.dropout_module = FairseqDropout(dropout)
        self.activation = utils.get_activation_fn(activation_fn)

        # apply custom weight initialization
        self.init_weights()

        # for logging purposes
        for n, p in self.named_parameters():
            p.hyper, p.label = True, n


    def build_encoder(self):
        return HyperNetworkEncoder(
            input_dim=self.task_embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_hidden_layers,
            dropout=self.dropout,
            activation_fn=self.activation_fn,
            normalize_before=self.normalize_before)

    def hyper_init(self, layer, target_in, target_out):
        with torch.no_grad():
            # feed random embedding into the hyper-network
            # and generate the weights for the given layer
            input_i = torch.randint(1, (1,)).squeeze()
            x = self.embed(input_i, input_i, input_i, [1, 1, 1])

            # feed x to the hyper-network's layer and obtain the hyper-weight
            h = self.encoder(x)
            dh_sqrt = torch.sqrt(h.new([h.shape[0]])).squeeze()
            hyper_weights = layer(h)

            hyper_weights = hyper_weights / dh_sqrt
            hyper_std = hyper_weights.std()

            # create regular (adapter) Linear layer(s)
            # to estimate the target STD for the hyper-layers
            target_std = Linear(target_in, target_out).weight.std()

            # scale down the weights of the layer,
            # to produce hyper-layer with same std as the regular layer
            factor = target_std / hyper_std
            layer.weight.data *= factor

        return target_std

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                if m.weight.requires_grad:
                    nn.init.trunc_normal_(m.weight)

        if "hyper" in self.init_method:
            self.target_std_down = self.hyper_init(self.head_down_weights,
                                                   self.mainnet_input_dim,
                                                   self.bottleneck_dim)
            self.target_std_up = self.hyper_init(self.head_up_weights,
                                                 self.bottleneck_dim,
                                                 self.mainnet_input_dim)
            bound = 1e-2  # ensure that the generated weights will be close to zero
            self.head_down_bias.weight.data.uniform_(-bound, bound)
            self.head_up_bias.weight.data.uniform_(-bound, bound)

            if self.generate_layernorm:
                self.head_layer_norm_gamma.weight.data.uniform_(-bound, bound)
                self.head_layer_norm_beta.weight.data.uniform_(-bound, bound)

    def embed(self, 
              src_lang: torch.Tensor, 
              trg_lang: torch.Tensor, 
              layer: torch.Tensor, 
              input_mask: list):

        s = self.src_lang_emb(src_lang)
        t = self.tgt_lang_emb(trg_lang)
        l = self.layer_emb(layer)

        # (zero) masking of input embeddings
        x = torch.cat([s * input_mask[0], t * input_mask[1], l * input_mask[2]])
        return x

    def generate(self, x: torch.Tensor):
        """
        Given the hyper-network embedding, encode it and produce the adapter layers
        """
        h = self.encoder(x)
        dh_sqrt = torch.sqrt(h.new([h.shape[0]])).squeeze()

        down_w = self.head_down_weights(h) / dh_sqrt
        down_b = self.head_down_bias(h) / dh_sqrt
        up_w = self.head_up_weights(h) / dh_sqrt
        up_b = self.head_up_bias(h) / dh_sqrt

        if self.generate_layernorm:
            ln_g = self.head_layer_norm_gamma(h) / dh_sqrt
            ln_b = self.head_layer_norm_beta(h) / dh_sqrt
        else:
            ln_g, ln_b = None, None

        return down_w, down_b, up_w, up_b, ln_g, ln_b

    def forward(self, 
                x: torch.Tensor, 
                src_lang: torch.Tensor,
                trg_lang: torch.Tensor, 
                layer: torch.Tensor,
                input_mask: list):
        """
        Args:
            x: the input sequence
            src_lang: the source language id
            trg_lang: the target language id
            layer: the layer id (e.g., enc-1, dec-3 etc.)
            input_mask:
        Returns:
            x (Tensor): the transformed input sequence `(src_len, batch, embed_dim)`
        """
        residual = x

        # 1. obtain the input_id embeddings
        emb = self.embed(src_lang, trg_lang, layer, input_mask)

        # 2. feed embedding to hyper-network and produce hyper-adapter weights
        down_w, down_b, up_w, up_b, ln_g, ln_b = self.generate(emb)

        # 3. apply hyper-adapter to input
        if self.generate_layernorm:
            # ensure ln_g will have mean of 1, instead of 0
            ln_g += torch.ones_like(ln_g)

        if self.normalize_before:
            x = F.layer_norm(x, (self.mainnet_input_dim,), ln_g, ln_b)

        x = F.linear(x, down_w.view(-1, self.mainnet_input_dim), down_b)
        x = self.activation(x)
        x = F.linear(x, up_w.view(self.mainnet_input_dim, -1), up_b)
        x = self.dropout_module(x)
        x += residual

        if not self.normalize_before:
            x = F.layer_norm(x, (self.mainnet_input_dim,), ln_g, ln_b)

        return x