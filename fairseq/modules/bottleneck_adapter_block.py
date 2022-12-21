import torch.nn as nn
from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


class BottleneckAdapter(nn.Module):
    """
    A simple adapter piece specific to a language-pair
    """
    def __init__(self, cfg, encoder=True):
        super().__init__()
        self.in_dim = cfg.encoder.embed_dim if encoder else cfg.decoder.embed_dim
        self.red_factor = cfg.encoder.adapter_reduction_factor if encoder else cfg.decoder.adapter_reduction_factor
        self.bottleneck_dim = self.in_dim//self.red_factor
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)

        self.layer_norm = nn.LayerNorm(self.in_dim)
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        
        self.normalize_before = cfg.encoder.normalize_before if encoder else cfg.decoder.normalize_before
        
        self.fc1 = self.build_fc(
            self.in_dim, 
            self.bottleneck_dim, 
            cfg.quant_noise.pq, 
            cfg.quant_noise.pq_block_size
        )
        self.fc2 = self.build_fc(
            self.bottleneck_dim, 
            self.in_dim, 
            cfg.quant_noise.pq, 
            cfg.quant_noise.pq_block_size
        )

    def build_fc(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), 
            p=q_noise,
            block_size=qn_block_size
        )

    def residual_connection(self, x, residual):
        return x + residual

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)
        x = self.fc2(self.activation_fn(self.fc1(x)))
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.layer_norm(x)
        return x
        

class BottleneckAdapterBlock(nn.Module):
    """
    A simple adapter block which houses mulitple mulitple adapter pieces, i.e. one per langauge-pair
    """
    def __init__(self, cfg, encoder=True):
        super().__init__()
        self.lang_ids = cfg.encoder.adapter_lang_ids if encoder else cfg.decoder.adapter_lang_ids
        self.adapters = nn.ModuleDict({id: BottleneckAdapter(cfg, encoder=encoder) for id in self.lang_ids})

    def forward(self, x, lang_id):
        # result = 0
        # for key, adapter in self.adapters.items():
        #     if lang_id in key.split(':'):
        #         result += adapter(x)
        # return result
        return self.adapters[lang_id](x)