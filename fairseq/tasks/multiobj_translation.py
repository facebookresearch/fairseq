from . import register_task
from .translation import TranslationTask, load_langpair_dataset


@register_task('multiobj-translation')
class MultiObjTranslationTask(TranslationTask):

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        data_paths = self.args.data.split(':')
        assert len(data_paths) > 0
        # Load datasets sequentially
        self.datasets[split] = [
            load_langpair_dataset(
                data_path, split, src, self.src_dict, tgt, self.tgt_dict,
                combine=combine, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
            for data_path in data_paths
        ]

    def dataset(self, split, idx=0):
        """
        Return a loaded dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
            idx (int): Dataset idx (defaults to 0)

        Returns:
            a :class:`~fairseq.data.FairseqDataset` corresponding to *split*
        """
        from fairseq.data import FairseqDataset
        if split not in self.datasets:
            raise KeyError('Dataset not loaded: ' + split)
        if not isinstance(self.datasets[split], FairseqDataset):
            raise TypeError(
                'Datasets are expected to be of type FairseqDataset')
        if idx < 0 or idx >= len(self.args.data):
            raise ValueError(f'Invalid dataset idx: {idx}')
        return self.datasets[split][idx]
