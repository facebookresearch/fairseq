# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as op
import logging

from fairseq.tasks import FairseqTask, register_task
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset
from fairseq.data.audio.speech_to_text_dataset_creator import SpeechToTextDatasetCreator
from fairseq.data import encoders, Dictionary

logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
logger = logging.getLogger(__name__)


@register_task('speech_to_text')
class SpeechToTextTask(FairseqTask):
    """Task for training speech-to-text models."""

    @staticmethod
    def add_args(parser):
        parser.add_argument('data', help='path to data directory')
        parser.add_argument(
            '--vocab-filename', type=str, default='dict.txt',
            help='The filename for the vocab file'
        )
        parser.add_argument(
            '--config-yaml', type=str, default='config.yaml',
            help='The filename for the data config YAML file'
        )
        parser.add_argument('--max-source-positions', default=6000, type=int,
                            metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int,
                            metavar='N',
                            help='max number of tokens in the target sequence')

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        if (
            getattr(args, "bpe", None) == "sentencepiece"
            and getattr(args, "sentencepiece_vocab", None) is None):
            # Hack here to automatic find spm model
            from os import listdir
            import re
            model_file = [
                f for f in listdir(args.data)
                if re.search(r"^spm_.+\.model$", f)
            ]
            assert len(model_file) == 1
            args.sentencepiece_vocab = op.join(args.data, model_file[0])

        self.tgt_dict = tgt_dict
        self.data_config = self.get_data_config()

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = op.join(args.data, args.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f'Dict not found: {dict_path}')
        tgt_dict = Dictionary.load(dict_path)
        logger.info(f'dictionary size: {len(tgt_dict):,}')
        return cls(args, tgt_dict)

    def get_data_config(self):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                'Please pip install pyyaml to load config YAML file'
            )

        config_path = op.join(self.args.data, self.args.config_yaml)
        if op.isfile(config_path):
            try:
                with open(config_path) as f:
                    config = f.read()
                return yaml.load(config, Loader=yaml.FullLoader)
            except Exception as e:
                logger.info(f'Failed to load config: {e}')
        return {}

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith('train')
        pre_tokenizer = encoders.build_tokenizer(self.args)
        bpe_tokenizer = encoders.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.args.data, self.data_config, split, self.tgt_dict,
            pre_tokenizer, bpe_tokenizer, is_train_split=is_train_split,
            epoch=epoch, seed=self.args.seed
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_generator(
            self, models, args,
            seq_gen_cls=None, extra_gen_cls_kwargs=None,
    ):
        lang_token_ids = {
            i for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {'symbols_to_strip_from_output': lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_dataset_for_inference(self, audio_paths, n_frames):
        return SpeechToTextDataset('interactive', False, {}, audio_paths,
                                   n_frames)

    def get_batch_iterator(
            self, dataset, max_tokens=None, max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False, required_batch_size_multiple=1,
            seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=1,
    ):
        # Recreate epoch iterator every epoch cause the underlying
        # datasets are dynamic due to sampling.
        self.dataset_to_epoch_iter = {}
        epoch_iter = super().get_batch_iterator(
            dataset, max_tokens, max_sentences, max_positions,
            ignore_invalid_inputs, required_batch_size_multiple,
            seed, num_shards, shard_id, num_workers, epoch,
        )
        self.dataset_to_epoch_iter = {}
        return epoch_iter
