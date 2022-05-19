#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ThreadPoolExecutor
import logging
from omegaconf import MISSING
import os
import torch
from typing import Optional
import warnings


from dataclasses import dataclass
from fairseq.dataclass import FairseqDataclass
from .kaldi_initializer import KaldiInitializerConfig, initalize_kaldi


logger = logging.getLogger(__name__)


@dataclass
class KaldiDecoderConfig(FairseqDataclass):
    hlg_graph_path: Optional[str] = None
    output_dict: str = MISSING

    kaldi_initializer_config: Optional[KaldiInitializerConfig] = None

    acoustic_scale: float = 0.5
    max_active: int = 10000
    beam_delta: float = 0.5
    hash_ratio: float = 2.0

    is_lattice: bool = False
    lattice_beam: float = 10.0
    prune_interval: int = 25
    determinize_lattice: bool = True
    prune_scale: float = 0.1
    max_mem: int = 0
    phone_determinize: bool = True
    word_determinize: bool = True
    minimize: bool = True

    num_threads: int = 1


class KaldiDecoder(object):
    def __init__(
        self,
        cfg: KaldiDecoderConfig,
        beam: int,
        nbest: int = 1,
    ):
        try:
            from kaldi.asr import FasterRecognizer, LatticeFasterRecognizer
            from kaldi.base import set_verbose_level
            from kaldi.decoder import (
                FasterDecoder,
                FasterDecoderOptions,
                LatticeFasterDecoder,
                LatticeFasterDecoderOptions,
            )
            from kaldi.lat.functions import DeterminizeLatticePhonePrunedOptions
            from kaldi.fstext import read_fst_kaldi, SymbolTable
        except:
            warnings.warn(
                "pykaldi is required for this functionality. Please install from https://github.com/pykaldi/pykaldi"
            )

        # set_verbose_level(2)

        self.acoustic_scale = cfg.acoustic_scale
        self.nbest = nbest

        if cfg.hlg_graph_path is None:
            assert (
                cfg.kaldi_initializer_config is not None
            ), "Must provide hlg graph path or kaldi initializer config"
            cfg.hlg_graph_path = initalize_kaldi(cfg.kaldi_initializer_config)

        assert os.path.exists(cfg.hlg_graph_path), cfg.hlg_graph_path

        if cfg.is_lattice:
            self.dec_cls = LatticeFasterDecoder
            opt_cls = LatticeFasterDecoderOptions
            self.rec_cls = LatticeFasterRecognizer
        else:
            assert self.nbest == 1, "nbest > 1 requires lattice decoder"
            self.dec_cls = FasterDecoder
            opt_cls = FasterDecoderOptions
            self.rec_cls = FasterRecognizer

        self.decoder_options = opt_cls()
        self.decoder_options.beam = beam
        self.decoder_options.max_active = cfg.max_active
        self.decoder_options.beam_delta = cfg.beam_delta
        self.decoder_options.hash_ratio = cfg.hash_ratio

        if cfg.is_lattice:
            self.decoder_options.lattice_beam = cfg.lattice_beam
            self.decoder_options.prune_interval = cfg.prune_interval
            self.decoder_options.determinize_lattice = cfg.determinize_lattice
            self.decoder_options.prune_scale = cfg.prune_scale
            det_opts = DeterminizeLatticePhonePrunedOptions()
            det_opts.max_mem = cfg.max_mem
            det_opts.phone_determinize = cfg.phone_determinize
            det_opts.word_determinize = cfg.word_determinize
            det_opts.minimize = cfg.minimize
            self.decoder_options.det_opts = det_opts

        self.output_symbols = {}
        with open(cfg.output_dict, "r") as f:
            for line in f:
                items = line.rstrip().split()
                assert len(items) == 2
                self.output_symbols[int(items[1])] = items[0]

        logger.info(f"Loading FST from {cfg.hlg_graph_path}")
        self.fst = read_fst_kaldi(cfg.hlg_graph_path)
        self.symbol_table = SymbolTable.read_text(cfg.output_dict)

        self.executor = ThreadPoolExecutor(max_workers=cfg.num_threads)

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions, padding = self.get_emissions(models, encoder_input)
        return self.decode(emissions, padding)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        model = models[0]

        all_encoder_out = [m(**encoder_input) for m in models]

        if len(all_encoder_out) > 1:

            if "encoder_out" in all_encoder_out[0]:
                encoder_out = {
                    "encoder_out": sum(e["encoder_out"] for e in all_encoder_out)
                    / len(all_encoder_out),
                    "encoder_padding_mask": all_encoder_out[0]["encoder_padding_mask"],
                }
                padding = encoder_out["encoder_padding_mask"]
            else:
                encoder_out = {
                    "logits": sum(e["logits"] for e in all_encoder_out)
                    / len(all_encoder_out),
                    "padding_mask": all_encoder_out[0]["padding_mask"],
                }
                padding = encoder_out["padding_mask"]
        else:
            encoder_out = all_encoder_out[0]
            padding = (
                encoder_out["padding_mask"]
                if "padding_mask" in encoder_out
                else encoder_out["encoder_padding_mask"]
            )

        if hasattr(model, "get_logits"):
            emissions = model.get_logits(encoder_out, normalize=True)
        else:
            emissions = model.get_normalized_probs(encoder_out, log_probs=True)

        return (
            emissions.cpu().float().transpose(0, 1),
            padding.cpu() if padding is not None and padding.any() else None,
        )

    def decode_one(self, logits, padding):
        from kaldi.matrix import Matrix

        decoder = self.dec_cls(self.fst, self.decoder_options)
        asr = self.rec_cls(
            decoder, self.symbol_table, acoustic_scale=self.acoustic_scale
        )

        if padding is not None:
            logits = logits[~padding]

        mat = Matrix(logits.numpy())

        out = asr.decode(mat)

        if self.nbest > 1:
            from kaldi.fstext import shortestpath
            from kaldi.fstext.utils import (
                convert_compact_lattice_to_lattice,
                convert_lattice_to_std,
                convert_nbest_to_list,
                get_linear_symbol_sequence,
            )

            lat = out["lattice"]

            sp = shortestpath(lat, nshortest=self.nbest)

            sp = convert_compact_lattice_to_lattice(sp)
            sp = convert_lattice_to_std(sp)
            seq = convert_nbest_to_list(sp)

            results = []
            for s in seq:
                _, o, w = get_linear_symbol_sequence(s)
                words = list(self.output_symbols[z] for z in o)
                results.append(
                    {
                        "tokens": words,
                        "words": words,
                        "score": w.value,
                        "emissions": logits,
                    }
                )
            return results
        else:
            words = out["text"].split()
            return [
                {
                    "tokens": words,
                    "words": words,
                    "score": out["likelihood"],
                    "emissions": logits,
                }
            ]

    def decode(self, emissions, padding):
        if padding is None:
            padding = [None] * len(emissions)

        ret = list(
            map(
                lambda e, p: self.executor.submit(self.decode_one, e, p),
                emissions,
                padding,
            )
        )
        return ret
