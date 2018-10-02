# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


from fairseq import sequence_generator

from . import FairseqDataset, language_pair_dataset


class BacktranslationDataset(FairseqDataset):
    def __init__(
        self,
        tgt_dataset,
        tgt_dict,
        backtranslation_model,
        unkpen,
        sampling,
        beam,
        max_len_a,
        max_len_b,
    ):
        """
        Sets up a backtranslation dataset which takes a tgt batch, generates
        a src using a tgt-src backtranslation_model, and returns the
        corresponding {generated src, input tgt} batch
        Args:
            tgt_dataset: dataset which will be used to build self.tgt_dataset --
                a LanguagePairDataset with tgt dataset as the source dataset and
                None as the target dataset.
                We use language_pair_dataset here to encapsulate the tgt_dataset
                so we can re-use the LanguagePairDataset collater to format the
                batches in the structure that SequenceGenerator expects.
            tgt_dict: tgt dictionary (typically a joint src/tgt BPE dictionary)
            backtranslation_model: tgt-src model to use in the SequenceGenerator
                to generate backtranslations from tgt batches
            unkpen, sampling, beam, max_len_a, max_len_b: generation args for
                the backtranslation SequenceGenerator
        """
        self.tgt_dataset = language_pair_dataset.LanguagePairDataset(
            src=tgt_dataset,
            src_sizes=None,
            src_dict=tgt_dict,
            tgt=None,
            tgt_sizes=None,
            tgt_dict=None,
        )
        self.backtranslation_generator = sequence_generator.SequenceGenerator(
            [backtranslation_model],
            tgt_dict,
            unk_penalty=unkpen,
            sampling=sampling,
            beam_size=beam,
        )
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.beam = beam

    def __getitem__(self, index):
        """
        Returns a single sample. Multiple samples are fed to the collater to
        create a backtranslation batch. Note you should always use collate_fn
        BacktranslationDataset.collater() below if given the option to
        specify which collate_fn to use (e.g. in a dataloader which uses this
        BacktranslationDataset -- see corresponding unittest for an example).
        """
        return self.tgt_dataset[index]

    def __len__(self):
        """
        The length of the backtranslation dataset is the length of tgt.
        """
        return len(self.tgt_dataset)

    def collater(self, samples):
        """
        Using the samples from the tgt dataset, load a collated tgt sample to
        feed to the backtranslation model. Then take the generated translation
        with best score as the source and the orignal net input as the target.
        """
        collated_tgt_only_sample = self.tgt_dataset.collater(samples=samples)
        backtranslation_hypos = self._generate_hypotheses(
            sample=collated_tgt_only_sample
        )

        # Go through each tgt sentence in batch and its corresponding best
        # generated hypothesis and create a backtranslation data pair
        # {id: id, source: generated backtranslation, target: original tgt}
        generated_samples = []
        for input_sample, hypos in zip(samples, backtranslation_hypos):
            generated_samples.append(
                {
                    "id": input_sample["id"],
                    "source": hypos[0]["tokens"],  # first hypo is best hypo
                    "target": input_sample["source"],
                }
            )

        return language_pair_dataset.collate(
            samples=generated_samples,
            pad_idx=self.tgt_dataset.src_dict.pad(),
            eos_idx=self.tgt_dataset.src_dict.eos(),
        )

    def get_dummy_batch(self, num_tokens, max_positions):
        """ Just use the tgt dataset get_dummy_batch """
        self.tgt_dataset.get_dummy_batch(num_tokens, max_positions)

    def num_tokens(self, index):
        """ Just use the tgt dataset num_tokens """
        self.tgt_dataset.num_tokens(index)

    def ordered_indices(self):
        """ Just use the tgt dataset ordered_indices """
        self.tgt_dataset.ordered_indices

    def valid_size(self, index, max_positions):
        """ Just use the tgt dataset size """
        self.tgt_dataset.valid_size(index, max_positions)

    def _generate_hypotheses(self, sample):
        """
        Generates hypotheses from a LanguagePairDataset collated / batched
        sample. Note in this case, sample["target"] is None, and
        sample["net_input"]["src_tokens"] is really in tgt language.
        """
        self.backtranslation_generator.cuda()
        input = sample["net_input"]
        srclen = input["src_tokens"].size(1)
        hypos = self.backtranslation_generator.generate(
            input,
            maxlen=int(
                self.max_len_a * srclen + self.max_len_b
            ),
        )
        return hypos
