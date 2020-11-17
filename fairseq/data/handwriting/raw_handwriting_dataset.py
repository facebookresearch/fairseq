# Copyright (c) UWr and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import logging
import numpy as np
import sys

import torch
import torch.nn.functional as F

from .utils import num_between
from . import scribblelens
from .. import FairseqDataset


logger = logging.getLogger(__name__)


class RawHandwritingDataset(FairseqDataset):
    def __init__(
        self,
        max_sample_size=None,
        min_sample_size=None,
        pad_to_multiples_of=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
        labels=False,  # [!] if True, need to set pad, blank and eos indices (set_special_indices)
    ):
        super().__init__()

        # We don't really have a sampling rate - out of audio (JCh)
        # self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad_to_multiples_of = pad_to_multiples_of
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize
        self.labels = labels
        self.label_pad_idx = None
        self.label_blank_idx = None
        self.label_eos_idx = None

    def set_special_indices(
        self,
        label_pad_idx,
        label_blank_idx,  # to ignore in alignment when getting cropped label etc
        decoder_fun,
        label_eos_idx=None,  # leave None for not appending EOS
    ):
        self.label_pad_idx = label_pad_idx
        self.label_blank_idx = label_blank_idx
        self.label_eos_idx = label_eos_idx

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        # TODO(jch): verify if this makes sense, prob not!
        # if feats.dim() == 2:
        #     feats = feats.mean(-1)

        # # Doesn't make sense - JCh
        # # if curr_sample_rate != self.sample_rate:
        # #     raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        # assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size_dim1, alignment=None):  # TODO perhaps change to just return indices, a bit cleaner?

        # if alignment set, cut it too - TODO maybe also mask half a letter etc., also in data!
        # but maybe do this on crop labels stuff (mask if less than half of a letter visible or so)

        size = wav.shape[1] #len(wav)
        diff = size - target_size_dim1
        if diff <= 0:
            if alignment is not None:
                return wav, alignment, (0, size-1)
            else:
                return wav, (0, size-1)

        if self.shuffle:
            start = np.random.randint(0, diff + 1)
        else:
            # Deterministically pick the middle part
            start = (diff + 1) //2
        end = size - diff + start
        if alignment is not None:
            return wav[:, start:end], alignment[start:end], (start, end-1)
        else:
            return wav[:, start:end], (start, end-1)  # end inclusive
        
    def collater(self, samples):

        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [s.shape[1] for s in sources]
        heigths = [s.shape[0] for s in sources]
        assert all([h==heigths[0] for h in heigths])

        pad_to_multiples_of = self.pad_to_multiples_of
        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
            if pad_to_multiples_of:
                # round up to pad_to_multiples_of
                target_size = ((target_size + pad_to_multiples_of - 1) // pad_to_multiples_of) * pad_to_multiples_of
        else:
            target_size = min(min(sizes), self.max_sample_size)
            if pad_to_multiples_of:
                # round down to pad_to_multiples_of
                target_size = (target_size // pad_to_multiples_of) * pad_to_multiples_of

        collated_sources = sources[0].new_zeros((len(sources), heigths[0], target_size))
        pad_shape = list(collated_sources.shape)
        pad_shape[1] = 1  # we mask all pixels in exactly the same way
        padding_mask = (
            torch.BoolTensor(size=pad_shape).fill_(False) if self.pad else None
        )
        if self.labels:
            collated_labels_nontensor = []
            #collated_texts_nontensor = []  # TODO
            collated_alignments = samples[0]["alignment"].new_full((len(sources), target_size), self.label_pad_idx)

        available_labels = torch.BoolTensor(size=(len(samples),)).fill_(False)
        available_alignments = torch.BoolTensor(size=(len(samples),)).fill_(False)
        # labels_with_ranges_arr is not a tensor, it just has None where unavailable

        labels_with_ranges_arr = []
        for i, (sample, size) in enumerate(zip(samples, sizes)):
            source = sample["source"]
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
                if self.labels:
                    if sample["label_available"]:
                        available_labels[i] = True
                        collated_labels_nontensor.append(sample["label"])
                    else:
                        collated_labels_nontensor.append(None)
                    #collated_texts_nontensor.append(sample["text"])
                    if sample["alignment_available"]:
                        available_alignments[i] = True
                        collated_alignments[i] = sample["alignment"]
                    if sample["alignment_available"] and sample["label_available"] and len(sample["label"]) == len(sample["alignment_text"]):
                        labels_with_ranges = self.get_letter_ranges(sample["alignment"], sample["label"])
                    else:
                        labels_with_ranges = None
                    labels_with_ranges_arr.append(labels_with_ranges)
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((heigths[0], -diff), 0.0)],
                    dim=1
                )
                padding_mask[i, :, diff:] = True
                if self.labels:
                    if sample["label_available"]:
                        available_labels[i] = True
                        collated_labels_nontensor.append(sample["label"])
                    else:
                        collated_labels_nontensor.append(None)
                    if sample["alignment_available"]:
                        available_alignments[i] = True
                        collated_alignments[i] = torch.cat([sample["alignment"], sample["alignment"].new_full((-diff,), self.label_pad_idx)])    
                    #collated_texts_nontensor.append(coll_text)
                    if sample["alignment_available"] and sample["label_available"] and len(sample["label"]) == len(sample["alignment_text"]):
                        labels_with_ranges = self.get_letter_ranges(sample["alignment"], sample["label"])
                    else:
                        labels_with_ranges = None
                    labels_with_ranges_arr.append(labels_with_ranges)
            else:
                # only case with cropping
                if self.labels:
                    collated_sources[i], collated_alignments[i], (start, end) = self.crop_to_max_size(source, target_size, alignment=sample["alignment"])
                    
                    if sample["label_available"]:
                        available_labels[i] = True
                    if sample["alignment_available"]:
                        available_alignments[i] = True
                    if sample["alignment_available"] and sample["label_available"] and len(sample["label"]) == len(sample["alignment_text"]):
                        # also update alignments - possible padding of halves of letters on the borders etc.
                        coll_labels, coll_labels_w_spaces, labels_with_ranges, collated_alignments[i], pad_begin = self.collate_labels(sample["alignment"], collated_alignments[i], sample["label"], start, end)  #, sample["text"])
                    else:
                        labels_with_ranges = None
                        coll_labels = None
                        pad_begin = None
                    labels_with_ranges_arr.append(labels_with_ranges)
                    collated_labels_nontensor.append(coll_labels)
                    if pad_begin is not None:
                        padding_mask[i, :, pad_begin:] = True
                    #collated_texts_nontensor.append(coll_text)
                else:
                    collated_sources[i], _ = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        if self.labels:
            assert self.pad
            collated_labels = torch.IntTensor(size=(len(collated_labels_nontensor), max([len(i) for i in collated_labels_nontensor]))).fill_(self.label_pad_idx)
            for i, label in enumerate(collated_labels_nontensor):
                collated_labels[i][:len(label)] = torch.tensor(label)
            # TODO check collate labels to common length in a tensor
            # TODO EOS stuff (?)

            # zeros where None
            target_lengths = torch.LongTensor([len(t) if t is not None else 0 for t in collated_labels_nontensor])

            abcd = 1
            # [!] stuff with "_available" tells if data "\is actually present in the tensors or are there some defaults or sth
            return {
                "id": torch.LongTensor([s["id"] for s in samples]), 
                "net_input": input,
                "target": collated_labels,  # data_utils.collate_tokens(collated_labels_nontensor, pad_idx=self.pad, left_pad=False),
                "target_available": available_labels,
                "target_lengths": target_lengths,  # 0 where there are no labels
                "ntokens": target_lengths.sum().item(), # only sums tokens where there are labels
                "alignments": collated_alignments,
                "alignments_available": available_alignments,
                "labels_with_ranges": labels_with_ranges_arr  # None where unavailable
                #"label_texts": collated_texts_nontensor,  # TODO?  non-collated texts of collated stuff
                }
        else:
            return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}

    @staticmethod
    def get_chars_ranges_uniform(num_chars, chars_begin, chars_end):
        full_len_diff = chars_end - chars_begin
        for_1_letter = float(full_len_diff) / float(num_chars)
        # assuming will always have at least 1 frame for a letter on average
        begin = chars_begin
        end = int(round(chars_begin + for_1_letter))
        collated_label = []
        for j in range(num_chars):
            collated_label.append((begin, end))
            begin = end + 1
            end = int(round(chars_begin + (j+2)*for_1_letter))
        return collated_label

    def get_letter_ranges(self, full_alignments_original, full_label_original, cut_start=None, cut_end=None):

        if cut_start is None:
            cut_start = 0
        if cut_end is None:
            cut_end = len(full_alignments_original) - 1

        last_idx = self.label_blank_idx
        full_alignments = torch.cat([full_alignments_original, full_alignments_original.new_full((1,), self.label_pad_idx)])  # for no special case 
        #decode_dict = {x.item(): y for x,y in zip(full_label, full_text)}  # can zip like that as full_label is already a list
        naive_collated_label = []
        naive_letter_ranges = []
        letter_begin = 0
        
        for i, numTensor in enumerate(full_alignments):
            num = numTensor.item()
            if num != last_idx:
                if last_idx != self.label_pad_idx and last_idx != self.label_eos_idx:
                    naive_collated_label.append(last_idx)
                    naive_letter_ranges.append((letter_begin, i-1))
                letter_begin = i
            last_idx = num

        full_label = torch.cat([full_label_original, full_label_original.new_full((1,), self.label_pad_idx)])  
        # ^ so will always append thing before pad without additional case after the loop
        label_qties = []
        last_idx = full_label[0]  #self.label_blank_idx
        qty = 0
        for numTensor in full_label:
            num = numTensor.item()
            if num != last_idx and last_idx != self.label_pad_idx and last_idx != self.label_eos_idx:
                label_qties.append((last_idx, qty))
                qty = 1
            else:
                qty += 1
            last_idx = num

        # TODO if stuff empty, return empty tensor or array or so

        next_ground_id = 0
        next_ground_seen = False
        last_idx = self.label_blank_idx
        collated_labels_with_ranges = []
        naive_letter_ranges.append((-1, -1))
        naive_collated_label.append(self.label_pad_idx)
        ground_ranges = []
        max_end_before = -1
        min_begin_after = 2*len(full_alignments_original)

        for char, (begin, end) in list(zip(naive_collated_label, naive_letter_ranges)):
            # first append any blanks from ground truth, blanks from naive stuff are omitted later in the loop
            # can also do it this way - would ignore some random 1-length blanks in alignments, although that should rather NOT happen
            if label_qties[next_ground_id][0] == self.label_blank_idx:  # rather can't happen, blanks not in label; TODO? maybe could also treat pad/sth else similarly, but rather not needed
                # don't append blanks there, those are not spaces!
                next_ground_id += 1
            if next_ground_id >= len(label_qties):
                break
            if char == self.label_blank_idx: # omit blanks in naive stuff not existent in ground, and calculate repetitions otherwise; blanks calculated from ground, above
                continue                     # TODO? maybe could also treat pad/sth else similarly, but rather not needed
            # from here no blanks in both places
            if next_ground_seen and char != label_qties[next_ground_id][0]:
                if len(ground_ranges) == label_qties[next_ground_id][1]:
                    #_ = RawHandwritingDataset.get_chars_ranges_uniform(3, begin, end)
                    ranges = ground_ranges
                else:
                    ranges = RawHandwritingDataset.get_chars_ranges_uniform(label_qties[next_ground_id][1], begin, end)
                for a, b in ranges:
                    mid = (a + b) // 2
                    if mid < cut_start:
                        max_end_before = max(max_end_before, b - cut_start)
                        continue
                    elif mid > cut_end:
                        min_begin_after = min(min_begin_after, a - cut_start)
                        continue
                    collated_labels_with_ranges.append([label_qties[next_ground_id][0], [a, b]])  # need to update with label_qties[next_ground_id][0], NOT char - char is next
                next_ground_id += 1
                next_ground_seen = False
                ground_ranges = []  # to be all seen in char == label_qties[next_ground_id][0] case - also in this loop spin
            if next_ground_id >= len(label_qties):
                break
            if char == label_qties[next_ground_id][0]:
                next_ground_seen = True
                ground_ranges.append((begin, end))
            # TODO else some error/warning or just ignore? could still work with messy alignments then

        # no blanks, no need to span stuff; spaces according to alignment

        if cut_start is not None or cut_end is not None:
            return collated_labels_with_ranges, ((max_end_before if max_end_before >= 0 else None), (min_begin_after if min_begin_after < cut_end - cut_start + 1 else None))
        else:
            return collated_labels_with_ranges

    # TODO separate function for getting the ranges of the letters and do this also when not cropping
    # modifies initial collated_alignments if needed
    def collate_labels(self, full_alignments_original, collated_alignments, full_label_original, cut_start, cut_end):  #, full_text):  # label is a list, text is a string
        
        collated_labels_with_ranges, (mask_to, mask_from) = self.get_letter_ranges(full_alignments_original, full_label_original, cut_start=cut_start, cut_end=cut_end)

        # mask_to & mask_from are indices

        pad_begin = None
        # [!] masks before first in collated_labels_with_ranges and after last - if cut_start < first range, similarly with end
        # (case when we have there e.g. <half of some letter)
        if mask_to is not None or mask_from is not None:
            changed_begin = mask_to + 1 if mask_to is not None else 0
            changed_end = mask_from - 1 if mask_from is not None else cut_end
            torch.roll(collated_alignments, -changed_begin)
            collated_alignments[changed_end - changed_begin + 1:] = self.label_pad_idx  
            pad_begin = changed_end - changed_begin + 1
            # blanks can sometimes get padded if spaces are not in the label

        # [!] can be negative if e.g. we have half of the letter after the cut
        collated_labels_with_ranges = [(a, (b - cut_start,c - cut_start)) for a, (b,c) in collated_labels_with_ranges]  # change to tuple and shift to new indices

        collated_label_with_spaces_on_ends = [char for char, _ in collated_labels_with_ranges] # here returning non-tensor [!], later fill in tensor
        if collated_label_with_spaces_on_ends[0] == self.label_blank_idx and collated_label_with_spaces_on_ends[-1] == self.label_blank_idx:
            collated_label = collated_label_with_spaces_on_ends[1:-1]
        elif collated_label_with_spaces_on_ends[0] == self.label_blank_idx:
            collated_label = collated_label_with_spaces_on_ends[1:]
        elif collated_label_with_spaces_on_ends[-1] == self.label_blank_idx:
            collated_label = collated_label_with_spaces_on_ends[:-1]
        else:
            collated_label = collated_label_with_spaces_on_ends[:]

        # [!] changes collated alignments tensor before
        return collated_label, collated_label_with_spaces_on_ends, collated_labels_with_ranges, collated_alignments, pad_begin #, collated_text


    def num_tokens(self, index):
        return self.size(index)  # TODO this doesn't really seem correct if tokens are letters as I think

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)
        
        # TODO stuff with labels? in addTargetDataset there is a 2nd dim then

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        # TODO [!] separate ordering of labeled and unlabeled data (if labels not everywhere)

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)  # TODO (?) should return also label size with labels? (as in AddTargetDataset), but this screws up much other stuff
        return np.lexsort(order)[::-1]


class FileHandwritingDataset(RawHandwritingDataset):
    def __init__(
        self,
        dist_root,
        split,
        max_sample_size=None,
        min_sample_size=None,
        pad_to_multiples_of=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
        labels=False,
    ):
        super().__init__(
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            pad_to_multiples_of=pad_to_multiples_of,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
            labels=labels,
        )
        self.dataset = scribblelens.ScribbleLensDataset(
            root=dist_root + '/scribblelens.corpus.v1.2.zip',           # Path to zip with images
            alignment_root=dist_root + '/scribblelens.paths.1.4b.zip',  # Path to the alignments, i.e. info aou char boundaries
            slice='tasman',                                     # Part of the data, here a single scribe https://en.wikipedia.org/wiki/Abel_Tasman
            split=split,                                      # Train, test, valid or unsupervised. Train/Test/Valid have character transcripts, unspuervised has only images
            # Not used in the simple ScribbleLens loader
            transcript_mode=5,                                  # Legacy space handling, has to be like that
            vocabulary=FileHandwritingDataset.vocabularyPath(dist_root),  # Path
        )
        if labels:
            self.set_special_indices(
                self.dataset.alphabet.pad(),
                self.dataset.alphabet.blank(),
                self.dataset.alphabet.eos()
            )
        # self.labels in superclass

        for data in self.dataset:
            sizeHere = data['image'].shape
            #print(sizeHere)
            #if self.labels:
            #    self.sizes.append((sizeHere[0], data['text'].shape[0]))  # ?
            #else:
            # not sure why AddTargetDataset has label size in sizes and makes it a tuple, of course not a single comment and incompatible because why anything would be 
            self.sizes.append(sizeHere[0])  # 1/2 dim TODO? rather this 1 dim is correct

        # self.fnames = []

        # skipped = 0
        # with open(manifest_path, "r") as f:
        #     self.root_dir = f.readline().strip()
        #     for line in f:
        #         items = line.strip().split("\t")
        #         assert len(items) == 2, line
        #         sz = int(items[1])
        #         if min_length is not None and sz < min_length:
        #             skipped += 1
        #             continue
        #         self.fnames.append(items[0])
        #         self.sizes.append(sz)
        # logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    @staticmethod
    def vocabularyPathSuffix():
        return '/tasman.alphabet.plus.space.mode5.json'

    @staticmethod
    def vocabularyPath(prefix):
        return prefix + FileHandwritingDataset.vocabularyPathSuffix()

    def __getitem__(self, index):
        # import soundfile as sf

        # fname = os.path.join(self.root_dir, self.fnames[index])
        # wav, curr_sample_rate = sf.read(fname)
        # feats = torch.from_numpy(wav).float()
        # feats = self.postprocess(feats, curr_sample_rate)

        feats = (self.dataset[index]['image'][:,:,0]).T  # image 32 x W

        if self.labels:
            label_available = self.dataset[index]['text_available']
            label = self.dataset[index]['text'] if label_available else None
            alignment_available = 'alignment' in self.dataset[index].keys() and len(self.dataset[index]['alignment']) == feats.shape[1]
            alignment = self.dataset[index]['alignment'] if alignment_available else None
            alignment_text = self.dataset[index]['alignment_text'] if alignment_available else None
            return {
                "id": index, 
                "source": feats,
                "alignment_available": alignment_available,
                "alignment": alignment,
                "label_available": label_available,
                "label": label,
                "alignment_text": alignment_text   # [!] not using this for letter ranges, using just label as letters shouldn't be missing [TODO check] 
                                                   # and this stuff is a bit processed from original letter indices, BUT if letters are missing can output weird stuff
            }
        else:
            return {"id": index, "source": feats}