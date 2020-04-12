"""
Classes and methods used for training and extraction of parallel pairs
from a comparable dataset.
Author: Alabi Jesujoba
"""
import collections
import math
import random

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.models import ARCH_MODEL_REGISTRY
from fairseq.data import (
    AppendTokenDataset,
    MonolingualDataset,
    ConcatDataset,
    data_utils,
    LanguagePairDataset
)

def get_src_len(src, use_gpu):
    if use_gpu:
        return torch.tensor([src.size(0)])#.cuda()
    else:
        return torch.tensor([src.size(0)])

class PairBank():
    """
    Class that saves and prepares parallel pairs and their resulting
    batches.
    Args:
        batch_size(int): number of examples in a batch
        opt(argparse.Namespace): option object
    """
    def __init__(self, batcher, args):
        self.pairs = []
        self.index_memory = set()
        self.batch_size = args.max_sentences
        self.batcher = batcher
        if args.cpu==False:
            self.use_gpu=True
        else:
            self.use_gpu = False


    def add_example(self, src, tgt):
        """ Add an example from a batch to the PairBank (self.pairs).
        Args:
            src(torch.Tensor): src sequence (size(seq))
            tgt(torch.Tensor): tgt sequence(size(tgt))
            fields(list(str)): list of keys of fields
        """
        # Get example from src/tgt and remove original padding
        #src = PairBank.removePadding(src)
        #tgt = PairBank.removePadding(tgt)
        src_length = get_src_len(src, self.use_gpu)
        tgt_length = get_src_len(tgt, self.use_gpu)
        index = None
        # Create CompExample object holding all information needed for later
        # batch creation.
        example = CompExample(index, src, tgt, src_length, tgt_length, index)
        #dataset, src, tgt, src_length, tgt_length, index
        # Add to pairs
        self.pairs.append(example)
        # Remember unique src-tgt combination
        self.index_memory.add(hash((str(src), str(tgt))))
        return None

    def contains_batch(self):
        """Check if enough parallel pairs found to create a batch.
        """
        return (len(self.pairs) >= self.batch_size)

    def no_limit_reached(self, src, tgt):
        """ Check if no assigned limit of unique src-tgt pairs is reached.
        Args:
            src(torch.Tensor): src sequence (size(seq))
            tgt(torch.Tensor): tgt sequence(size(tgt))
        """
        #src = PairBank.removePadding(src)
        #tgt = PairBank.removePadding(tgt)
        return (hash((str(src), str(tgt))) in self.index_memory or len(self.index_memory) < self.limit)

    def get_num_examples(self):
        """Returns batch size if no maximum number of extracted parallel data
        used for training is met. Otherwise returns number of examples that can be yielded
        without exceeding that maximum.
        """
        if len(self.pairs) < self.batch_size:
            return len(self.pairs)
        return self.batch_size

    def yield_batch(self):
        """ Prepare and yield a new batch from self.pairs.
        Returns:
            batch(fairseq.data.LanguagePairDataset): batch of extracted parallel data
        """
        src_examples = []
        tgt_examples = []
        src_lengths = []
        tgt_lengths = []
        indices = []
        num_examples = self.get_num_examples()

        # Get as many examples as needed to fill a batch or a given limit
        random.shuffle(self.pairs)
        for ex in range(num_examples):
            example = self.pairs.pop()
            src_examples.append(example.src)
            tgt_examples.append(example.tgt)
            src_lengths.append(example.src_length)
            tgt_lengths.append(example.tgt_length)
            indices.append(example.index)

        dataset = None
        #fields = CompExample.get_fields()
        batch = self.batcher.create_batch(src_examples, tgt_examples, src_lengths, tgt_lengths)
        #enumerate to yield batch here
        return batch

class CompExample():
    """
    Class that stores the information of one parallel data example.
    Args:
        dataset(fairseq.data): dataset object
        src(torch.Tensor): src sequence (size(seq))
        tgt(torch.Tensor): tgt sequence (size(seq))
        src_length(torch.Tensor): the length of the src sequence (size([]))
        index(torch.Tensor): the index of the example in dataset
    """
    # These should be the same for all examples (else: consistency problem)
    _dataset = None

    def __init__(self, dataset, src, tgt, src_length, tgt_length, index):
        self.src = src
        self.tgt = tgt
        self.src_length = src_length
        self.tgt_length = tgt_length
        self.index = index


        if CompExample._dataset == None:
            CompExample._dataset = dataset


class BatchCreator():
    def __init__(self, task, args):
        self.task = task
        self.args = args

    def create_batch(self, src_examples, tgt_examples, src_lengths, tgt_lengths, no_target=False):
        """ Creates a batch object from previously extracted parallel data.
                Args:
                    src_examples(list): list of src sequence tensors
                    tgt_examples(list): list of tgt sequence tensors
                    src_lenths(list): list of the lengths of each src sequence
                    tgt_lenths(list): list of the lengths of each tgt sequence
                    indices(list): list of indices of example instances in dataset
                    dataset(fairseq.data): dataset object
                Returns:
                    batch(fairseq.data.LanguagePairDataset): batch object
                """
        pairData = LanguagePairDataset(
            src_examples, src_lengths, self.task.src_dict,
            tgt_examples, tgt_lengths, self.task.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )
        batch = pairData
        return batch

class Comparable():
    """
    Class that controls the extraction of parallel sentences and manages their
    storage and training.
    Args:
        model(:py:class:'fairseq.models'):
            translation model used for extraction and training
        trainer(:obj:'fairseq.trainer'):
            trainer that controlls the training process
        fields(dict): fields and vocabulary
        logger(logging.RootLogger):
            logger that reports information about extraction and training
        opt(argparse.Namespace): option object
    """

    def __init__(self, model, trainer, task, args):
        self.sim_measure = args.sim_measure
        self.threshold = args.threshold
        self.model = trainer.get_model().encoder
        self.encoder = trainer.get_model().encoder
        self.trainer = trainer
        self.batcher = BatchCreator(task, args)
        self.similar_pairs = PairBank(self.batcher, args)
        self.accepted = 0
        self.accepted_limit = 0
        self.declined = 0
        self.total = 0
        self.args = args
        self.comp_log = args.comp_log
        self.cove_type = args.cove_type
        self.k = 4
        self.trainstep = 0
        self.second = args.second
        self.representations = args.representations
        self.task = task
        self.write_dual = args.write_dual
        self.no_swaps = args.no_swaps
        self.symmetric = args.symmetric
        self.add_noise = args.add_noise
        self.use_bt = args.use_bt

    def write_sentence(self, src, tgt, status, score=None):
        """
        Writes an accepted parallel sentence candidate pair to a file.
        Args:
            src(torch.tensor): src sequence (size(seq))
            tgt(torch.tensor): tgt sequence (size(seq))
            status(str): ['accepted', 'accepted-limit', 'rejected']
            score(float): score of the sentence pair
        """
        src_words = self.task.src_dict.string(src)
        tgt_words = self.task.tgt_dict.string(tgt)
        out = 'src: {}\ttgt: {}\tsimilarity: {}\tstatus: {}\n'.format(' '.join(src_words),
                                            ' '.join(tgt_words), score, status)
        if 'accepted' in status:
            self.accepted_file.write(out)
            #print(out)
        elif status == 'embed_only':
            with open(self.embed_file, 'a', encoding='utf8') as f:
                f.write(out)
        elif status == 'hidden_only':
            with open(self.hidden_file, 'a', encoding='utf8') as f:
                f.write(out)
        return None

    def extract_parallel_sents(self, candidates, candidate_pool):
        """
        Extracts parallel sentences from candidates and adds them to the
        PairBank (secondary filter).
        Args:
            candidates(list(tuple(torch.Tensor...)): list of src-tgt candidates
            candidate_pool(list(hash)): list of hashed C_e candidates
        """
        print("extract parallel")
        for candidate in candidates:
            candidate_pair = hash((str(candidate[0]), str(candidate[1])))
            # For dual representation systems...
            if candidate_pool:
                # ...skip C_h pairs not in C_e (secondary filter)
                if self.in_candidate_pool(candidate, candidate_pool) == False:
                    self.declined += 1
                    self.total += 1
                    if self.write_dual:
                        self.write_sentence(candidate[0], candidate[1],
                                            'hidden_only', candidate[2])
                    continue
            if self.no_swaps:
                swap = False
            # Swap src-tgt direction randomly
            else:
                swap = np.random.randint(2)
            if swap:
                src = candidate[1]
                tgt = candidate[0]
            else:
                src = candidate[0]
                tgt = candidate[1]

            score = candidate[2]
            # Apply threshold (single-representation systems only)
            if score >= self.threshold:
                print("Score is greater than threshold")
                # Check if no maximum of allowed unique accepted pairs reached
                #if self.similar_pairs.no_limit_reached(src, tgt):
                # Add to PairBank
                self.similar_pairs.add_example(src, tgt)
                self.write_sentence(src, tgt, 'accepted', score)
                if self.add_noise:
                    noisy_src = self.apply_noise(src, tgt)
                    self.similar_pairs.add_example(noisy_src, tgt)
                    self.write_sentence(noisy_src, tgt, 'accepted-noise', score)

                if self.symmetric:
                    self.similar_pairs.add_example(tgt, src)
                    self.write_sentence(tgt, src, 'accepted', score)
                    if self.add_noise:
                        noisy_tgt = self.apply_noise(tgt, src)
                        self.similar_pairs.add_example(noisy_tgt, src)
                        self.write_sentence(noisy_tgt, src, 'accepted-noise', score)
                self.accepted += 1
                if self.use_bt:
                    self.remove_from_bt_candidates(src)
                    self.remove_from_bt_candidates(tgt)
                '''else:
                    self.accepted_limit += 1
                    self.write_sentence(src, tgt, 'accepted-limit', score)'''
            else:
                self.declined +=1
            self.total += 1

        return None

    def write_embed_only(self, candidates, cand_embed):
        """ Writes C_e scores to file (if --write-dual is set).
        Args:
            candidates(list): list of src, tgt pairs (C_h)
            cand_embed(list): list of src, tgt pairs (C_e)
        """
        candidate_pool = set([hash((str(c[0]), str(c[1]))) for c in candidates])

        for candidate in cand_embed:
            candidate_pair = hash((str(candidate[0]), str(candidate[1])))
            # Write statistics only if C_e pair not in C_h
            if candidate_pair not in candidate_pool:
                src = candidate[0]
                tgt = candidate[1]
                score = candidate[2]
                self.write_sentence(src, tgt, 'embed_only', score)


    def score_sents(self, src_sents, tgt_sents):
        """ Score source and target combinations.
        Args:
            src_sents(list(tuple(torch.Tensor...))):
                list of src sentences in their sequential and semantic representation
            tgt_sents(list(tuple(torch.Tensor...))): list of tgt sentences
        Returns:
            src2tgt(dict(dict(float))): dictionary mapping a src to a tgt and their score
            tgt2src(dict(dict(float))): dictionary mapping a tgt to a src and their score
            similarities(list(float)): list of cosine similarities
            scores(list(float)): list of scores
        """
        src2tgt = defaultdict(dict)
        tgt2src = defaultdict(dict)
        similarities = []
        scores= []
        for src, src_cove in src_sents:
            for tgt, tgt_cove in tgt_sents:
                # Ignore combination if both sentences from the same language
                if src[0] == tgt[0]:
                    continue
                # Calculate cosine similarity of combination
                sim = self.calculate_similarity(src_cove, tgt_cove)
                src2tgt[src][tgt] = sim
                tgt2src[tgt][src] = sim
                similarities.append(sim)
        # Return cosine similarity if that is the scoring function
        if self.sim_measure == 'cosine':
            return src2tgt, tgt2src, similarities, similarities

        # Else, continue to calculate margin-based score
        # Calculate denominator (average cosine similarity to k-nearest neighbors)
        for src, _ in src_sents:
            src2tgt[src]['sum'] = self._sum_k_nearest(src2tgt, src)

        for tgt, _ in tgt_sents:
            tgt2src[tgt]['sum'] = self._sum_k_nearest(tgt2src, tgt)

        for src, _ in src_sents:
            for tgt, _ in tgt_sents:
                if src[0] == tgt[0]:
                    continue
                # Apply denominator to each combination...
                src2tgt[src][tgt] /= (src2tgt[src]['sum'] + tgt2src[tgt]['sum'])

        for tgt, tgt_cove in tgt_sents:
            for src, src_cove in src_sents:
                if src[0] == tgt[0]:
                    continue
                # ... in both language directions
                tgt2src[tgt][src] /= (src2tgt[src]['sum'] + tgt2src[tgt]['sum'])
            del tgt2src[tgt]['sum']

        # Get list of scores for statistics
        for src in list(src2tgt.keys()):
            del src2tgt[src]['sum']
            scores += list(src2tgt[src].values())

        return src2tgt, tgt2src, similarities, scores

    def _sum_k_nearest(self, mapping, cove):
        """ Calculates average score of a sequence to its k-nearest neighbors.
        Args:
            mapping(dict(dict(float))): L1-L2 mapping with their respective cosine sim
            cove(torch.Tensor): sentence representation of L1 sequence
        Returns:
            float: denominator of margin-based scoring function
        """
        # Get k-nearest neighbors
        k_nearest = sorted(mapping[cove].items(), key=lambda x: x[1], reverse=True)[:self.k]
        # Sum scores and return denominator
        sum_k_nearest = sum([ex[1] for ex in k_nearest])
        return sum_k_nearest / (2 * len(k_nearest))

    def get_article_coves(self, article, representation='memory', mean=False):
        """ Get representations (C_e or C_h) for sentences in a document.
        Args:
            article(inputters.OrderedIterator): iterator over sentences in document
            representation(str): if 'memory', create C_h; if 'embed', create C_e
            fast(boolean): if true, only look at first batch in article
            mean(boolean): if true, use mean over time-step representations; else, sum
        Returns:
            sents(list(tuple(torch.Tensor...))):
                list of sentences in their sequential (seq) and semantic representation (cove)
        """
        sents = []
        for i, k in enumerate(article):
            sent_repr = None
            #print(i, k['id'], k['net_input']['src_tokens'], k['net_input']['src_lengths'])
            #print(type(k['net_input']['src_tokens']))

            if self.args.modeltype == "lstm":  # if the model architecture is LSTM
                output = self.encoder.forward(k['net_input']['src_tokens'],
                                                                       k['net_input']['src_lengths'])
                print("In the lstm representation")
                if representation == 'memory':
                    sent_repr = output['encoder_out'][1].squeeze()
                    #print("In the lstm representation",sent_repr)
                elif representation == 'embed':
                    hidden_embed = output['encoder_out'][0]
                    if mean:
                        sent_repr = torch.mean(hidden_embed, dim=0)
                    else:
                        sent_repr = torch.sum(hidden_embed, dim=0)
            elif self.args.modeltype == "transformer":
                encoderOut = self.encoder.forward(k['net_input']['src_tokens'],
                                                                           k['net_input']['src_lengths'])
                if representation == 'memory':
                    hidden_embed = getattr(encoderOut, 'encoder_out')  # T x B x C
                    if mean:
                        sent_repr = torch.mean(hidden_embed, dim=0)
                    else:
                        sent_repr = torch.sum(hidden_embed, dim=0)
                elif representation == 'embed':
                    input_emb = getattr(encoderOut, 'encoder_embedding')  # B x T x C
                    input_emb = input_emb.transpose(0, 1)
                    if mean:
                        sent_repr = torch.mean(input_emb, dim=0)
                    else:
                        sent_repr = torch.sum(input_emb, dim=0)

            for i in range(k['net_input']['src_tokens'].shape[0]):
                sents.append((k['net_input']['src_tokens'][i], sent_repr[i]))
            return sents

        return sents

    def get_comparison_pool(self, src_embeds, tgt_embeds):
        """ Perform scoring and filtering for C_e (in dual representation system)
        Args:
            src_embeds(list): list of source embeddings (C_e)
            tgt_embeds(list): list of target embeddings (C_e)
        Returns:
            candidate_pool(set): set of hashed src-tgt C_e pairs
            candidate_embed(list): list of src-tgt C_e pairs
        """
        # Scoring
        src2tgt_embed, tgt2src_embed, _, _ = self.score_sents(src_embeds, tgt_embeds)
        # Filtering (primary filter)
        candidates_embed = self.filter_candidates(src2tgt_embed, tgt2src_embed)
        # Create set of hashed pairs (for easy comparison in secondary filter)
        set_embed = set([hash((str(c[0]), str(c[1]))) for c in candidates_embed])
        candidate_pool = set_embed
        return candidate_pool, candidates_embed

    def in_candidate_pool(self, candidate, candidate_pool):
        candidate_pair = hash((str(candidate[0]), str(candidate[1])))
        # For dual representation systems...
        # ...skip C_h pairs not in C_e (secondary filter)
        if candidate_pair in candidate_pool:
            return True
        return False

    def filter_candidates(self, src2tgt, tgt2src, second=False):
        """ Filter candidates (primary filter), such that only those which are top candidates in
        both src2tgt and tgt2src direction pass.
        Args:
            src2tgt(dict(dict(float))): mapping src sequence to tgt sequence and score
            tgt2src(dict(dict(float))): mapping tgt sequence to src sequence and score
            second(boolean): if true, also include second-best candidate for src2tgt direction
                (medium permissibility mode only)
        Returns:
            candidates(list(tuple(torch.Tensor...)): list of src-tgt candidates
        """
        src_tgt_max = set()
        tgt_src_max = set()
        src_tgt_second = set()
        tgt_src_second = set()

        # For each src...
        for src in list(src2tgt.keys()):
            toplist = sorted(src2tgt[src].items(), key=lambda x: x[1], reverse=True)
            # ... get the top scoring tgt
            max_tgt = toplist[0]
            # Get src, tgt and score
            src_tgt_max.add((src, max_tgt[0], max_tgt[1]))
            if second:
                # If high permissibility mode, also get second-best tgt
                second_tgt = toplist[1]
                src_tgt_second.add((src, second_tgt[0], second_tgt[1]))

        # For each tgt...
        for tgt in list(tgt2src.keys()):
            toplist = sorted(tgt2src[tgt].items(), key=lambda x: x[1], reverse=True)
            # ... get the top scoring src
            max_src = toplist[0]
            tgt_src_max.add((max_src[0], tgt, max_src[1]))

        if second:
            # Intersection as defined in medium permissibility mode
            src_tgt = (src_tgt_max | src_tgt_second) & tgt_src_max
            candidates = list(src_tgt)
            return candidates

        # Intersection as defined in low permissibility
        candidates = list(src_tgt_max & tgt_src_max)
        return candidates

    def _get_iterator(self, sent, dictn, max_position, fix_batches_to_gpus=False):
        """
        Creates an iterator object from a text file.
        Args:
            path(str): path to text file to process
        Returns:
            data_iter(.EpochIterator): iterator object
        """
        # get indices ordered by example size
        with data_utils.numpy_seed(1):
            indices = sent.ordered_indices()
        # filter out examples that are too large
        max_positions = (max_position)
        if max_positions is not None:
            indices = data_utils.filter_by_size(indices, sent, max_positions, raise_exception=(not True), )
        # create mini-batches with given size constraints
        max_sentences = 5
        batch_sampler = data_utils.batch_by_size(indices, sent.num_tokens, max_sentences=max_sentences,
                                                 required_batch_size_multiple=5, )
        itrs = iterators.EpochBatchIterator(dataset=sent, collate_fn=sent.collater, batch_sampler=batch_sampler,
                                            seed=1, epoch=0)
        data_iter = itrs.next_epoch_itr(shuffle=True, fix_batches_to_gpus=fix_batches_to_gpus)

        return data_iter

    def forward(self, side, representation='memory'):
        """ F-prop a src or tgt batch through the encoder.
        Args:
            side(torch.Tensor): batch to be f-propagated
                (size(seq, batch, 1))
            representation(str): if 'memory', access hidden states; else embeddings
        Returns:
            memory_bank(torch.Tensor): encoder outputs
                (size(seq, batch, fets))
        """
        # Do not accumulate gradients
        with torch.no_grad():
            if representation == 'embed':
                # word embeddings
                embeddings = self.encoder.embeddings(side)
                return embeddings
            else:
                # hidden states/encoder output
                embeddings, memory_bank, src_lengths = self.encoder(side, None)
                return memory_bank

    def calculate_similarity(self, src, tgt):
        """ Calculates the cosine similarity between two sentence representations.
        Args:
            src(torch.Tensor): src sentence representation (size(fets))
            tgt(torch.Tensor): tgt sentence representation (size(fets))
        Returns:
            float: cosine similarity
        """
        return nn.functional.cosine_similarity(src, tgt, dim=0).tolist()

    def get_cove(self, memory, ex, mean=False):
        """ Get sentence representation.
        Args:
            memory(torch.Tensor): hidden states or word embeddings of batch
            ex(int): index of example in batch
            mean(boolean): if true, take mean over time-steps; else, sum
        Returns:
            cove(torch.Tensor): sentence representation C_e or C_h
        """
        # Get current example
        seq_ex = memory[:, ex, :]
        if self.cove_type == 'mean':
            cove = torch.mean(seq_ex, dim=0)
        else:
            cove = torch.sum(seq_ex, dim=0)
        return cove

    def extract_and_train(self, comparable_data_list):
        """ Manages the alternating extraction of parallel sentences and training.
        Args:
            comparable_data_list(str): path to list of mapped documents
        Returns:
            train_stats(:obj:'onmt.Trainer.Statistics'): epoch loss statistics
        """
        # Start first epoch
        # self.trainer.next_epoch()
        self.accepted_file = open('accepted_sentences.txt', 'w+', encoding='utf8')
        """self.accepted_file = \
            open('{}_accepted-e{}.txt'.format(self.comp_log, self.trainer.cur_epoch), 'w+', encoding='utf8')
        
        self.status_file = '{}_status-e{}.txt'.format(self.comp_log, self.trainer.cur_epoch)"""
        if self.write_dual:
            self.embed_file = 'embedded.txt'
            self.hidden_file = 'hiddenembed.txt'

        epoch_similarities = []
        epoch_scores = []
        """
        counter = 0
        self.idx_update = 0
        """
        src_sents = []
        tgt_sents = []
        src_embeds = []
        tgt_embeds = []
        """
        if self.mono:
            # TODO: Debug
            with open(self.mono, encoding='utf8') as m:
                mono_pairs = m.read().split('\n')
            for mono_pair in mono_pairs:
                mono_pair = mono_pair.split('\t')
                if len(mono_pair) < 2:
                    continue
                mono_src = self._get_iterator(mono_pair[0])
                mono_tgt = self._get_iterator(mono_pair[1])
                self.logger.info('...scoring monolingual dataset {}'.format(mono_pair[0]))
                self.extract_sentences_mono(mono_src, mono_tgt)
                train_stats = self.train()
            self.logger.info('...continuing with comparable data.')

        """
        # Go through comparable data
        """with open(comparable_data_list, encoding='utf8') as c:
            comp_list = c.read().split('\n')
            num_articles = len(comp_list)
            cur_article = 0

            for article_pair in comp_list:
                cur_article += 1
                articles = article_pair.split('\t')
                # Discard malaligned documents
                if len(articles) != 2:
                    continue
        """
        # Prepare iterator objects for current src/tgt document
        trainingSetSrc = data_utils.load_indexed_dataset(self.args.data + "train.de-en.en", self.task.src_dict,
                                                         dataset_impl=self.args.dataset_impl, combine=False,
                                                         default='cached')
        trainingSetTgt = data_utils.load_indexed_dataset(self.args.data + "train.de-en.de", self.task.tgt_dict,
                                                         dataset_impl=self.args.dataset_impl, combine=False,
                                                         default='cached')

        print(type(trainingSetSrc))
        print("sentenxez length = ", trainingSetSrc.sizes)
        src_mono = MonolingualDataset(dataset=trainingSetSrc, sizes=trainingSetSrc.sizes, src_vocab=self.task.src_dict,
                                      tgt_vocab=None, shuffle=False, add_eos_for_other_targets=False)
        tgt_mono = MonolingualDataset(dataset=trainingSetTgt, sizes=trainingSetTgt.sizes,
                                      src_vocab=self.task.tgt_dict,
                                      tgt_vocab=None, shuffle=False, add_eos_for_other_targets=False)
        print("Created Monolingual Dataset")
        src_article = self._get_iterator(src_mono, dictn=self.task.src_dict,
                                         max_position=self.args.max_source_positions, fix_batches_to_gpus=False)
        tgt_article = self._get_iterator(tgt_mono, dictn=self.task.tgt_dict,
                                         max_position=self.args.max_target_positions, fix_batches_to_gpus=False)
        print("Got Iterators for source and target......")
        # Get sentence representations
        try:
            if self.representations == 'embed-only':
                print("Using Embeddings only for representation")
                # C_e
                src_sents += self.get_article_coves(src_article, representation='embed', mean=True)
                tgt_sents += self.get_article_coves(tgt_article, representation='embed', mean=True)
            else:
                print("Using Dual for representation")
                # C_h
                src_sents += self.get_article_coves(src_article, representation='memory', mean=True)
                tgt_sents += self.get_article_coves(tgt_article, representation='memory', mean=True)
                # C_e
                src_embeds += self.get_article_coves(src_article, representation='embed', mean=True)
                tgt_embeds += self.get_article_coves(tgt_article, representation='embed', mean=True)
        except:
            # Skip document pair in case of errors
            src_sents = []
            tgt_sents = []
            src_embeds = []
            tgt_embeds = []
            # continue
        print("LEngth of Embeddings=",len(src_embeds))
        print(src_embeds[0][1].shape)
        print(src_sents[0][1].shape)


        # Ensure enough sentences are accumulated (otherwise scoring becomes unstable)
        if len(src_sents) < 15 or len(tgt_sents) < 15:
            print("")
            #continue

        # Score src and tgt sentences
        # TODO: Debug this
        print("In all we have got ",len(src_sents),"source sentences and ",len(tgt_sents),"target" )
        try:
            print("calculating similarities using ",self.sim_measure)
            src2tgt, tgt2src, similarities, scores = self.score_sents(src_sents, tgt_sents)
            print("calculated similarities using ", self.sim_measure)
        except:
            #print('Error occured in: {}\n'.format(article_pair), flush=True)
            print(src_sents, flush=True)
            print(tgt_sents, flush=True)
            src_sents = []
            tgt_sents = []
            #continue
        print("We have similartities length = ", len(similarities))
        print("similartities = ",similarities)
        print("scores = ",scores)

        # Keep statistics
        epoch_similarities += similarities
        epoch_scores += scores
        src_sents = []
        tgt_sents = []

        # Filter candidates (primary filter)
        try:
            if self.representations == 'dual':
                # For dual representation systems, filter C_h...
                candidates = self.filter_candidates(src2tgt, tgt2src, second=self.second)
                # ...and C_e
                comparison_pool, cand_embed = self.get_comparison_pool(src_embeds,
                                                                               tgt_embeds)
                src_embeds = []
                tgt_embeds = []
                if self.write_dual:
                    print("writing the sentences to file....")
                    self.write_embed_only(candidates, cand_embed)
            else:
                # Filter C_e or C_h for single representation system
                candidates = self.filter_candidates(src2tgt, tgt2src)
                comparison_pool = None
        except:
            # Skip document pair in case of errors
            print("Error Occured!!!!")
            #print('Error occured in: {}\n'.format(article_pair), flush=True)
            src_embeds = []
            tgt_embeds = []
            #continue
        print("candidates**********************")
        print(len(candidates))

        # Extract parallel samples (secondary filter)
        self.extract_parallel_sents(candidates, comparison_pool)
        print("The length of pairbank = ",len(self.similar_pairs.pairs))

        # Train on remaining partial batch
        if len((self.similar_pairs.pairs)) > 0:
            print(self.similar_pairs.yield_batch())

        """
        # Train on extracted sentences
        train_stats = self.train()
        """
        self.accepted_file.close()
        return None
