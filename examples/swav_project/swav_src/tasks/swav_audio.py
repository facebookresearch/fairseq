from argparse import Namespace
from dataclasses import dataclass, field
from fairseq.data.audio.raw_audio_dataset import BinarizedAudioDataset
from fairseq.data.audio.swav_audio_dataset import FileSwavExtrapolateNoNoiseLangAudioDataset
from fairseq.dataclass.configs import FairseqDataclass
import logging
import os
from fairseq.tasks import register_task
from fairseq.tasks.audio_pretraining import AudioPretrainingConfig, AudioPretrainingTask

logger = logging.getLogger(__name__)


@dataclass
class SwavAudioPretrainingConfig(AudioPretrainingConfig):
    rand_factor: int = field(
        default=2,
        metadata={
            "help": "rand factor"
        },
    )
    swav_langs: str = field(
        default='english',
        metadata={
            "help": "rand factor"
        },
    )


@register_task("swav_audio_pretraining", dataclass=SwavAudioPretrainingConfig)
class SwavAudioPretrainingTask(AudioPretrainingTask):
    cfg: SwavAudioPretrainingConfig

    def __init__(self, cfg: SwavAudioPretrainingConfig):
        super().__init__(cfg)
    
    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg
        langs = self.cfg.swav_langs.split(',')

        lang_dict = {l: i for i, l in enumerate(langs)}
        langs_str = ','.join(langs)

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"

        def path_to_langid_fn(path):
            # path: /datasets01/mls/mls_english/train/audio/9955/9413/9955_9413_000006.flac
            return lang_dict[path.split("/")[3].split("_")[1]]

        if getattr(task_cfg, "binarized_dataset", False):
            self.datasets[split] = BinarizedAudioDataset(
                data_path,
                split=split,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
                **self._get_mask_precompute_kwargs(task_cfg),
            )
            raise NotImplementedError('binarized_dataset not ready')
        else:
            manifest_path = os.path.join(data_path, "{}.tsv".format(split))

            self.datasets[split] = FileSwavExtrapolateNoNoiseLangAudioDataset(
                langs_str=langs_str,
                rand_factor=self.cfg.rand_factor,
                manifest_path=manifest_path,
                sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
                max_sample_size=self.cfg.max_sample_size,
                min_sample_size=self.cfg.min_sample_size,
                pad=task_cfg.labels is not None or task_cfg.enable_padding,
                normalize=task_cfg.normalize,
                num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
                compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
                **self._get_mask_precompute_kwargs(task_cfg),
            )

        if self.cfg.tpu and task_cfg["mask_channel_prob"] == 0.0:
            logger.info(
                "Pretraining on TPUs may suffer convergence "
                "issues when training with `mask_channel_prob` value of "
                "0. You may want to set this to a low value close to 0."
            )




