import json 
import logging 
from pathlib import Path
from argparse import Namespace
import numpy as np
import math 

from typing import Dict, Sequence, Tuple
from collections import OrderedDict, defaultdict

import torch 
import torch.nn.functional as F

from fairseq import metrics, options, utils
from fairseq.data import (
    FairseqDataset,
    LanguagePairDataset,
    NoisingDataset,
    PrependTokenDataset,
    RoundRobinZipDatasets,
    TransformEosLangPairDataset,
    data_utils,
    encoders,
)
from fairseq.tasks import register_task

from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
)

from fairseq.tasks.online_backtranslation import PiecewiseLinearFn
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)

def _lang_token(lang: str) -> str:
    return f"__{lang}__"


def _lang_token_index(dictionary, lang: str) -> int:
    return dictionary.index(_lang_token(lang))

@register_task("online_multilingual_backtranslation")
class MultilingualOnlineBackTranslationTask(TranslationMultiSimpleEpochTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        # Generic translation args
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument('--mono-langs', metavar='MONO_LANGS',
                            help='monolingual languages for training')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # Denoising args
        parser.add_argument('--max-word-shuffle-distance', default=3.0, type=float, metavar='N',
                            help='maximum word shuffle distance for denoising autoencoding data generation')
        parser.add_argument('--word-dropout-prob', default=0.1, type=float, metavar='N',
                            help='word dropout probability for denoising autoencoding data generation')
        parser.add_argument('--word-blanking-prob', default=0.2, type=float, metavar='N',
                            help='word blanking probability for denoising autoencoding data generation')

        # Backtranslation args
        parser.add_argument('--lambda-bt', default="1.0", type=str, metavar='N',
                            help='back-translation weight')
        parser.add_argument('--lambda-dae', default="1.0", type=str, metavar='N',
                            help='denoising auto-encoder weight')

        # Evaluation args
        parser.add_argument('--generate-one-by-one', action='store_true',
                            help='generate one sentence at a time for backtranslation')

        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        parser.add_argument('--pad-to-fixed-length', default=False, type=bool,
                            help='pad batch to fixed sequence length')
        

    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        self.mono_langs = args.mono_langs 


        self.SHOW_SAMPLES_INTERVAL = 1000
        # Start by showing samples
        self._show_samples_ctr = self.SHOW_SAMPLES_INTERVAL
        self.SHOW_SAMPLES_NUMBER = 5
        self.lambda_bt = PiecewiseLinearFn.from_string(args.lambda_bt)
        self.lambda_dae = PiecewiseLinearFn.from_string(args.lambda_dae)



        self.args = args
        self.data = utils.split_paths(self.args.data)
        if len(self.data) == 1:
            shards = list(Path(self.data[0]).glob("shard*"))
            if len(shards) > 0:
                # keep this as strings, since it can also be a manifold path
                old_data = self.data
                self.data = [str(shard) for shard in shards]
                logger.warning(f"Expanded data directory {old_data} to {self.data}")

        self.dictionary = self.dicts[0]

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """


        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        assert args.mono_langs is not None

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')

        langs, dicts, training = MultilingualDatasetManager.prepare(
            cls.load_dictionary, args, **kwargs
        )

        return cls(args, langs, dicts, training)
    

    def load_train_dataset(self, data_path: str) -> FairseqDataset:
        """The training dataset is made of backtranslation dataset and denoising dataset."""
        data = []
        for lang in self.mono_langs:
            train_path = os.path.join(data_path, lang, "train")
            # TODO: could we do the BT using denoise sample ?
            # this would half the data loading work
            data.append((f"{lang}-BT", self.load_bt_dataset(train_path, lang)))
            data.append(
                (f"{lang}-DENOISE", self.load_denoise_dataset(train_path, lang))
            )

        return RoundRobinZipDatasets(OrderedDict(data))
    


    def load_bt_dataset(self, data_path: str, lang: str) -> FairseqDataset:
        """The BT dataset is generated with (tgt, tgt) pairs.
        The actual translation to a (generated_src, tgt) pair
        is done on the fly during training.
        """
        mono_dataset = data_utils.load_indexed_dataset(
            data_path, self.dicts, self.args.dataset_impl
        )
        assert mono_dataset is not None, f"No dataset found for {lang}"

        mono_dataset_src = PrependTokenDataset(
            mono_dataset, _lang_token_index(self.dictionary, lang)
        )

        mono_dataset_bt = self._langpair_dataset(mono_dataset_src, mono_dataset)
        logger.info(
            f"mono_lang = {lang} "
            f"lang token index = {_lang_token_index(self.dictionary, lang)} "
            f"lang token = {_lang_token(lang)}"
        )

        mono_dataset_bt = self._prepend_lang_bos_to_target(mono_dataset_bt, lang)
        return mono_dataset_bt


    def _langpair_dataset(
        self, src: FairseqDataset, tgt: FairseqDataset
    ) -> LanguagePairDataset:
        return LanguagePairDataset(
            src,
            src.sizes,
            self.dictionary,
            tgt=tgt,
            tgt_sizes=tgt.sizes,
            tgt_dict=self.dictionary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target
        )
    
    def _prepend_lang_bos_to_target(
        self, dataset: LanguagePairDataset, lang: str
    ) -> LanguagePairDataset:
        bos = _lang_token_index(self.dicts[0], lang)
        return TransformEosLangPairDataset(
            dataset,
            src_eos=self.dictionary.eos(),
            new_src_eos=self.dictionary.eos(),
            tgt_bos=self.dictionary.eos(),
            new_tgt_bos=bos,
        )
    
    def load_denoise_dataset(self, data_path: str, lang: str) -> FairseqDataset:
        """Classic denoising dataset"""
        dataset = data_utils.load_indexed_dataset(
            data_path, self.dicts[0], self.args.dataset_impl
        )
        noisy_dataset = NoisingDataset(
            dataset,
            self.dicts[0],
            seed=1,
            max_word_shuffle_distance=self.args.max_word_shuffle_distance,
            word_dropout_prob=self.args.word_dropout_prob,
            word_blanking_prob=self.args.word_blanking_prob,
        )
        noisy_dataset = PrependTokenDataset(
            noisy_dataset, _lang_token_index(self.dictionary, lang)
        )

        clean_dataset = data_utils.load_indexed_dataset(
            data_path, self.dictionary, self.args.dataset_impl
        )
        denoising_dataset = self._langpair_dataset(noisy_dataset, clean_dataset)
        denoising_dataset = self._prepend_lang_bos_to_target(denoising_dataset, lang)
        return denoising_dataset
    
    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        detok_args =  {}
        self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.args.eval_bleu_detok, **detok_args)
        )
        gen_args = json.loads(self.args.eval_bleu_args)
        self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
        )
        return model
    
    def display_samples_once_in_a_while(self, smp, mono_lang, other_lang):
        self._show_samples_ctr += 1
        if self._show_samples_ctr < self.SHOW_SAMPLES_INTERVAL:
            return
        self._show_samples_ctr = 0

        ln = smp["net_input"]["src_tokens"].shape[0]

        logger.info(
            f"(r:{self.args.distributed_rank}) : "
            f"{other_lang} ---> {mono_lang} "
            f"({other_lang} was generated by back-translation.) {ln} samples"
        )

        for i in range(min(ln, self.SHOW_SAMPLES_NUMBER)):
            src_tokens = smp["net_input"]["src_tokens"][i]
            tgt_tokens = smp["target"][i]

            src_str = self.dictionary.string(src_tokens, "sentencepiece")
            tgt_str = self.dictionary.string(tgt_tokens, "sentencepiece")
            logger.info(
                f"\n{i}\t\t[{other_lang} generated]  {src_str}\n"
                f"\t\t[{mono_lang} original ]  {tgt_str}\n"
                f"\t\t[ src tokens]  {src_tokens}\n"
            )

    def backtranslate_sample(self, smp, orig_lang, other_lang) -> None:
        """
        * WARNING: smp is modified in place.
        * At the start of this function, `smp` has the same input and target:
          |--------------------------------------------------------|
          | smp['net_input']['src_tokens'] |  smp['target']        |
          | (from data) __en__ hello world |  __en__ hello world   |
          |--------------------------------------------------------|

        * We call generator.generate(smp, bos_token = token("ro")),
        and copy the result as input
        * At the end, `smp` has the translation to other language.
          |--------------------------------------------------------|
          | smp['net_input']['src_tokens'] |  smp['target']        |
          | (generated) __ro__ salut lume  |  __en__ hello world   |
          |--------------------------------------------------------|

        """
        bos_token = _lang_token_index(self.dictionary, other_lang)
        generated = self.sequence_generator.generate(
            models=[], sample=smp, bos_token=bos_token
        )

        max_lngth = max([gn[0]["tokens"].size(0) for gn in generated])
        net_input = smp["net_input"]
        n_src_tokens = torch.empty(
            size=(len(generated), max_lngth + 1), dtype=net_input["src_tokens"].dtype
        )
        n_src_lengths = torch.empty(
            len(generated), dtype=net_input["src_lengths"].dtype
        )

        for i, gn in enumerate(generated):
            tokens = gn[0]["tokens"]
            tokens_size = tokens.size(0)
            padding_needed = max_lngth - tokens_size
            tokens = torch.cat([tokens.new([bos_token]), tokens])
            tokens = F.pad(tokens, (0, padding_needed), value=self.dictionary.pad())
            n_src_tokens[i] = tokens
            n_src_lengths[i] = tokens_size + 1

        device = net_input["src_tokens"].device
        # This seems to be important
        del net_input["src_tokens"]
        del net_input["src_lengths"]
        net_input["src_tokens"] = n_src_tokens.to(device)
        net_input["src_lengths"] = n_src_lengths.to(device)

    def generate(self, smp, model):
        model.eval()
        orig_lang = (
            self.dictionary[smp["net_input"]["src_tokens"][0][0]]
            .replace(" ", "")
            .replace("_", "")
        )
        bos_token = smp["net_input"]["prev_output_tokens"][0][0]
        with torch.no_grad():
            generated = self.sequence_generator.generate(
                models=[model], sample=smp, bos_token=bos_token
            )
        return generated
    
    def get_other_lang(self, lang):
        # TODO: allow more complex mapping
        if lang != self.mono_langs[0]:
            return self.mono_langs[0]
        if len(self.mono_langs) == 2:
            return self.mono_langs[1]
        return self.mono_langs[np.random.randint(1, len(self.mono_langs))]
    

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):

        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size = 0.0, 0.0
        agg_logging_output: Dict[str, float] = defaultdict(float)

        dataset_keys = self.datasets["train"].datasets.keys()

        weights = {
            "BT": self.lambda_bt(update_num),
            "DENOISE": self.lambda_dae(update_num),
        }
        log_keys = {"BT": "bt_", "DENOISE": "dae_"}

        for dataset_key in dataset_keys:
            smp = sample[dataset_key]
            mono_lang, task_subtype = dataset_key.split("-")
            if weights[task_subtype] == 0:
                continue

            if task_subtype == "BT":
                with torch.autograd.profiler.record_function("backtranslation"):
                    model.eval()
                    # TODO: Could we translate to several language at once ?
                    # this would allow to share encoder_out and maximize GPU usage.
                    other_lang = self.get_other_lang(mono_lang)
                    self.backtranslate_sample(smp, mono_lang, other_lang)
                    self.display_samples_once_in_a_while(smp, mono_lang, other_lang)
                    model.train()

            # Like in FairseqTask.train_step
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion(model, smp)
            loss *= weights[task_subtype]
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)

            agg_loss += loss.item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[log_keys[task_subtype] + k] += logging_output[k]
                agg_logging_output[k] += logging_output[k]

        return agg_loss, agg_sample_size, agg_logging_output
    
    def get_bos_token_from_sample(self, sample):
        net_input = sample["net_input"]
        source_lang_token_id = torch.unique(net_input["src_tokens"][:, 0]).item()
        source_lang_token = self.dictionary[source_lang_token_id].replace("_", "")
        target_lang_token_id = _lang_token_index(
            self.dictionary, self.get_other_lang(source_lang_token)
        )

        return target_lang_token_id

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        bt_sample_size = sum(x.get("bt_sample_size", 0) for x in logging_outputs)
        if bt_sample_size:
            bt_loss_sum = sum(x.get("bt_loss", 0) for x in logging_outputs)
            bt_loss_sum *= 1 / bt_sample_size / math.log(2)
            metrics.log_scalar("bt_loss", bt_loss_sum, bt_sample_size, round=3)

            bt_nll_loss_sum = sum(x.get("bt_nll_loss", 0) for x in logging_outputs)
            bt_ntokens = sum(x.get("bt_ntokens", 0) for x in logging_outputs)
            bt_nll_loss_sum *= 1 / bt_ntokens / math.log(2)
            metrics.log_scalar("bt_nll_loss", bt_nll_loss_sum, bt_ntokens, round=3)
            metrics.log_derived(
                "bt_ppl", lambda meters: utils.get_perplexity(meters["bt_nll_loss"].avg)
            )

        dae_sample_size = sum(x.get("dae_sample_size", 0) for x in logging_outputs)
        if dae_sample_size:
            dae_loss_sum = sum(x.get("dae_loss", 0) for x in logging_outputs)
            dae_loss_sum *= 1 / dae_sample_size / math.log(2)
            metrics.log_scalar("dae_loss", dae_loss_sum, dae_sample_size, round=3)

            dae_nll_loss_sum = sum(x.get("dae_nll_loss", 0) for x in logging_outputs)
            dae_ntokens = sum(x.get("dae_ntokens", 0) for x in logging_outputs)
            dae_nll_loss_sum *= 1 / dae_ntokens / math.log(2)
            metrics.log_scalar("dae_nll_loss", dae_nll_loss_sum, dae_ntokens, round=3)
            metrics.log_derived(
                "dae_ppl",
                lambda meters: utils.get_perplexity(meters["dae_nll_loss"].avg),
            )