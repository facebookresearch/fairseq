"""
Classes and methods used for training and extraction of parallel pairs
from a comparable dataset.
Author: Alabi Jesujoba
"""
# import tracemalloc
# import gc
#import os
import re
#import itertools
import random
from collections import defaultdict
import torch
'''import cProfile
import pstats
'''
#from torch import nn

from fairseq.data import (
    MonolingualDataset,
    LanguagePairDataset
)
from fairseq.data.data_utils import load_indexed_dataset, numpy_seed, batch_by_size, filter_by_size
from fairseq.data.iterators import EpochBatchIterator, GroupedIterator
from fairseq import (
    checkpoint_utils, metrics, progress_bar, utils
)

def get_src_len(src, use_gpu):
    if use_gpu:
        return torch.tensor([src.size(0)]).cuda()
    else:
        return torch.tensor([src.size(0)])

# this method is to remove spaces added within strings when dict.string is used.
# it removed remove spaces between characters and consecutive spaces
def removeSpaces(s):
    k = re.sub(' (?! )', "", s)
    k = re.sub(' +', ' ', k)
    return k

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
        self.use_gpu = False
        self.update_freq = args.update_freq
        if args.cpu == False:
            self.use_gpu = True
        else:
            self.use_gpu = False
        self.explen = self.batch_size * self.update_freq[-1]

    def removePadding(side):
        """ Removes original padding from a sequence.
        Args:
            side(torch.Tensor): src/tgt sequence (size(seq))
        Returns:
            side(torch.Tensor): src/tgt sequence without padding
        NOTE: This only works as long as PAD_ID==1!
        """
        # Get indexes of paddings in sequence
        padding_idx = (side == 1).nonzero()
        # If there is any padding, cut sequence from first occurence of a pad
        if padding_idx.size(0) != 0:
            first_pad = padding_idx.data.tolist()[0][0]
            side = side[:first_pad]
        return side

    def add_example(self, src, tgt):
        """ Add an example from a batch to the PairBank (self.pairs).
        Args:
            src(torch.Tensor): src sequence (size(seq))
            tgt(torch.Tensor): tgt sequence(size(tgt))
            fields(list(str)): list of keys of fields
        """
        # Get example from src/tgt and remove original padding
        src = PairBank.removePadding(src)
        tgt = PairBank.removePadding(tgt)
        src_length = get_src_len(src, self.use_gpu)
        tgt_length = get_src_len(tgt, self.use_gpu)
        index = None
        # Create CompExample object holding all information needed for later
        # batch creation.
        # print((src,tgt))
        example = CompExample(index, src, tgt, src_length, tgt_length, index)
        # dataset, src, tgt, src_length, tgt_length, index
        # Add to pairs
        self.pairs.append(example)
        # Remember unique src-tgt combination
        self.index_memory.add(hash((str(src), str(tgt))))
        return None

    def contains_batch(self):
        """Check if enough parallel pairs found to create a batch.
        """
        return (len(self.pairs) >= self.explen)

    def no_limit_reached(self, src, tgt):
        """ Check if no assigned limit of unique src-tgt pairs is reached.
        Args:
            src(torch.Tensor): src sequence (size(seq))
            tgt(torch.Tensor): tgt sequence(size(tgt))
        """
        # src = PairBank.removePadding(src)
        # tgt = PairBank.removePadding(tgt)
        return (hash((str(src), str(tgt))) in self.index_memory or len(self.index_memory) < self.limit)

    def get_num_examples(self):
        """Returns batch size if no maximum number of extracted parallel data
        used for training is met. Otherwise returns number of examples that can be yielded
        without exceeding that maximum.
        """
        if len(self.pairs) < self.explen:
            return len(self.pairs)
        return self.explen

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
        # fields = CompExample.get_fields()
        batch = self.batcher.create_batch(src_examples, tgt_examples, src_lengths, tgt_lengths)
        # enumerate to yield batch here
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

        with numpy_seed(self.args.seed):
            indices = pairData.ordered_indices()

        batch_sampler = batch_by_size(indices, pairData.num_tokens, max_sentences=self.args.max_sentences,
                                      required_batch_size_multiple=self.args.required_batch_size_multiple, )
        itrs = EpochBatchIterator(dataset=pairData, collate_fn=pairData.collater, batch_sampler=batch_sampler,
                                  seed=self.args.seed, epoch=0)
        indices = None
        return itrs


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
        self.model_name = args.model_name
        self.save_dir = args.save_dir
        # self.model = trainer.get_model().encoder
        self.update_freq = args.update_freq
        print("Update Frequency = ", self.update_freq)
        self.trainer = trainer
        self.task = self.trainer.task
        self.encoder = self.trainer.get_model().encoder
        self.batcher = BatchCreator(self.task, args)
        self.similar_pairs = PairBank(self.batcher, args)
        self.accepted = 0
        self.accepted_limit = 0
        self.declined = 0
        self.total = 0
        self.args = args
        self.comp_log = args.comp_log
        self.cove_type = args.cove_type
        self.k = 4
        self.div = 2 * torch.tensor(self.k).cuda()
        self.trainstep = 0
        self.second = False #args.second
        self.representations = args.representations
        self.write_dual = args.write_dual
        self.no_swaps = False  # args.no_swaps
        self.symmetric = args.symmetric
        self.add_noise = args.add_noise
        self.use_bt = args.use_bt
        self.stats = None
        self.progress = None
        self.src, self.tgt = args.source_lang, args.target_lang
        self.use_gpu = False
        if self.args.cpu == False:
            self.use_gpu = True
        else:
            self.use_gpu = False

        self.use_phrase = self.args.use_phrase


    def getstring(self, vec, dict):
        words = dict.string(vec)
        return removeSpaces(' '.join(words))

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
        out = 'src: {}\ttgt: {}\tsimilarity: {}\tstatus: {}\n'.format(removeSpaces(' '.join(src_words)),
                                                                      removeSpaces(' '.join(tgt_words)), score, status)
        # out_src = ' '.join(src_words) + "\n"
        # out_tgt = ' '.join(tgt_words) + "\n"
        if 'accepted' in status:
            self.accepted_file.write(out)
            # self.accepted_file_src.write(removeSpaces(out_src))
            # self.accepted_file_tgt.write(removeSpaces(out_tgt))
            # print(out)
        elif 'phrase' in status:
            self.accepted_phrase.write(out)
        elif status == 'embed_only':
            with open(self.embed_file, 'a', encoding='utf8') as f:
                f.write(out)
        elif status == 'hidden_only':
            with open(self.hidden_file, 'a', encoding='utf8') as f:
                f.write(out)
        return None

    def extract_parallel_sents(self, candidates, candidate_pool, phrasese=False):
        """
        Extracts parallel sentences from candidates and adds them to the
        PairBank (secondary filter).
        Args:
            candidates(list(tuple(torch.Tensor...)): list of src-tgt candidates
            candidate_pool(list(hash)): list of hashed C_e candidates
        """
        # print("extract parallel")
        for candidate in candidates:
            ##candidate_pair = hash((str(candidate[0]), str(candidate[1])))
            # For dual representation systems...
            # print("Dual representation checking")

            if candidate_pool:
                # ...skip C_h pairs not in C_e (secondary filter)
                if self.in_candidate_pool(candidate, candidate_pool) == False:
                    self.declined += 1
                    self.total += 1
                    if self.write_dual:
                        self.write_sentence(candidate[0], candidate[1],
                                            'hidden_only', candidate[2])
                        # instead of just continuing, add the content to the phrase level bank for translation
                    continue
            '''if self.no_swaps:
                swap = False
            # Swap src-tgt direction randomly
            else:
                swap = np.random.randint(2)
            if swap:
                src = candidate[1]
                tgt = candidate[0]
            else:
                src = candidate[0]
                tgt = candidate[1]'''

            src = candidate[0]
            tgt = candidate[1]
            score = candidate[2]

            # Apply threshold (single-representation systems only)
            if score >= self.threshold:
                # print("Score is greater than threshold")
                # Check if no maximum of allowed unique accepted pairs reached
                # if self.similar_pairs.no_limit_reached(src, tgt):
                # Add to PairBank
                self.similar_pairs.add_example(src, tgt)
                self.write_sentence(removePadding(src), removePadding(tgt), 'accepted', score)
                self.accepted += 1

                if self.symmetric:
                    self.similar_pairs.add_example(tgt, src)
                    #self.write_sentence(removePadding(tgt), removePadding(src), 'accepted', score)

                '''
                if self.use_phrase and phrasese is False:
                    print("checking phrases to remove.......")
                    src_rm = removePadding(src)
                    self.phrases.remove_from_phrase_candidates(src_rm, 'src')
                    tgt_rm = removePadding(tgt)
                    self.phrases.remove_from_phrase_candidates(tgt_rm, 'tgt')
                    # write the accepted phrases to file
                if self.use_phrase and phrasese is True and self.args.write_phrase:
                    self.write_sentence(removePadding(src), removePadding(tgt), 'phrase', score)
                '''



                '''if self.add_noise:
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
                else:
                    self.accepted_limit += 1
                    self.write_sentence(src, tgt, 'accepted-limit', score)'''
            else:
                # print("threshold not met!!!")
                # if thresholid is not met add to phrase bank
                # self.phrases.add_example(self.getstring(removePadding(src),self.task.src_dict), self.getstring(removePadding(tgt),self.task.tgt_dict))
                self.declined += 1
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
        scores = []

        # print("At the point of unzipping the list of tuple....................")
        # unzip the list ot tiples to have two lists of equal length each sent, repre
        srcSent, srcRep = zip(*src_sents)
        tgtSent, tgtRep = zip(*tgt_sents)

        # print("Stacking the representations to cuda....................")
        # stack the representation list into a tensor and use that to compute the similarity
        if self.use_gpu:
            srcRp = torch.stack(srcRep).cuda()
            tgtRp = torch.stack(tgtRep).cuda()
        else:
            srcRp = torch.stack(srcRep)
            tgtRp = torch.stack(tgtRep)

        # Return cosine similarity if that is the scoring function
        if self.sim_measure == 'cosine':
            matx = self.sim_matrix(srcRp, tgtRp)
            for i in range(len(srcSent)):
                for j in range(len(tgtSent)):
                    if srcSent[i][0] == tgtSent[j][0]:
                        continue
                    src2tgt[srcSent[i]][tgtSent[j]] = matx[i][j].tolist()
                    tgt2src[tgtSent[j]][srcSent[i]] = matx[i][j].tolist()
                    similarities.append(matx[i][j].tolist())
            return src2tgt, tgt2src, similarities, similarities
        else:
            sim_mt, sumDistSource, sumDistTarget = self.sim_matrix(srcRp, tgtRp)
            for i in range(len(srcSent)):
                for j in range(len(tgtSent)):
                    if srcSent[i][0] == tgtSent[j][0]:
                        continue
                    #x =  sim_mt[i][j].tolist() / (sumDistSource[i].tolist() + sumDistTarget[j].tolist())
                    tgt2src[tgtSent[j]][srcSent[i]] = src2tgt[srcSent[i]][tgtSent[j]] = sim_mt[i][j].tolist() / (sumDistSource[i].tolist() + sumDistTarget[j].tolist())

        return src2tgt, tgt2src, similarities, scores

    def sim_matrix(self, a, b, eps=1e-20):
        """
        added eps for numerical stability
        """
        # compute the euc norm for division
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        del a_n, b_n, a_norm, b_norm
        if self.sim_measure == 'cosine':
            return sim_mt
        nearestSrc = torch.topk(sim_mt, self.k, dim=1, largest=True, sorted=False, out=None)
        nearestTgt = torch.topk(sim_mt, self.k, dim=0, largest=True, sorted=False, out=None)
        '''if b.shape[0]<self.k:
            nearestSrc = torch.topk(sim_mt, b.shape[0], dim=1, largest=True, sorted=False, out=None)
            divs = 2 * b.shape[0]
        else:
            nearestSrc = torch.topk(sim_mt, self.k, dim=1, largest=True, sorted=False, out=None)
            divs = 2 * self.k
        # nearestSrc = nearestSrc / self.div
        # sumDistSource = torch.sum(nearestSrc[0], 1)
        if a.shape[0] < self.k:
            nearestTgt = torch.topk(sim_mt, a.shape[0], dim=0, largest=True, sorted=False, out=None)
            divt = 2 * a.shape[0]
        else:
            nearestTgt = torch.topk(sim_mt, self.k, dim=0, largest=True, sorted=False, out=None)
            divt = 2 * self.k
        '''

        # nearestTgt = nearestTgt / self.div
        # sumDistTarget = torch.sum(nearestTgt[0], 0)


        #return sim_mt, torch.sum(nearestSrc[0], 1)/divs, torch.sum(nearestTgt[0], 0)/divt
        return sim_mt, torch.sum(nearestSrc[0], 1) / self.div, torch.sum(nearestTgt[0], 0) / self.div


    def get_article_coves(self, article, representation='memory', mean=False, side='phr', use_phrase=False):
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
        # for k in article:#tqdm(article):
        for k in article:
            sent_repr = None
            if self.args.modeltype == "lstm":  # if the model architecture is LSTM
                lengths = k['net_input']['src_lengths']
                texts = k['net_input']['src_tokens']
                ordered_len, ordered_idx = lengths.sort(0, descending=True)
                texts = texts[ordered_idx]
                with torch.no_grad():
                    output = self.encoder.forward(texts.cuda(), ordered_len)

                    if representation == 'memory':
                        sent_repr = output['encoder_out'][1].squeeze()
                        # print("In the lstm representation",sent_repr)
                    elif representation == 'embed':
                        # print("Collecting Embedding")
                        hidden_embed = output['encoder_out'][0]
                        # print(hidden_embed)tr
                        if mean:
                            sent_repr = torch.mean(hidden_embed, dim=0)
                        else:
                            sent_repr = torch.sum(hidden_embed, dim=0)
            elif self.args.modeltype == "transformer":
                with torch.no_grad():
                    encoderOut = self.encoder.forward(k['net_input']['src_tokens'].cuda(),
                                                      k['net_input']['src_lengths'].cuda())

                    # print("In the transformer representation")
                    if representation == 'memory':
                        hidden_embed = getattr(encoderOut, 'encoder_out')  # T x B x C
                        if mean:
                            sent_repr = torch.mean(hidden_embed, dim=0)
                        else:
                            sent_repr = torch.sum(hidden_embed, dim=0)
                    elif representation == 'embed':
                        input_emb = getattr(encoderOut, 'encoder_embedding')  # B x T x C
                        # print(type(input_emb))
                        input_emb = input_emb.transpose(0, 1)
                        if mean:
                            sent_repr = torch.mean(input_emb, dim=0)
                        else:
                            sent_repr = torch.sum(input_emb, dim=0)
            if self.args.modeltype == "transformer":
                for i in range(k['net_input']['src_tokens'].shape[0]):
                    sents.append((k['net_input']['src_tokens'][i], sent_repr[i]))
                    '''if side == 'src' and use_phrase is True:
                        st = removePadding(k['net_input']['src_tokens'][i])
                        self.phrases.sourcesent.add((hash(str(st)), st))
                    elif side == 'tgt' and use_phrase is True:
                        st = removePadding(k['net_input']['src_tokens'][i])
                        self.phrases.targetsent.add((hash(str(st)), st))'''
            elif self.args.modeltype == "lstm":
                for i in range(texts.shape[0]):
                    sents.append((texts[i], sent_repr[i]))

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
        # print("Length of s2t max",len(src_tgt_max))
        # print("Length of t2s max", len(tgt_src_max))
        # print("Intersection = ",list(src_tgt_max & tgt_src_max))
        candidates = list(src_tgt_max & tgt_src_max)
        return candidates

    def _get_iterator(self, sent, dictn, max_position, epoch, fix_batches_to_gpus=False):
        """
        Creates an iterator object from a text file.
        Args:
            path(str): path to text file to process
        Returns:
            data_iter(.EpochIterator): iterator object
        """
        # get indices ordered by example size
        with numpy_seed(self.args.seed):
            indices = sent.ordered_indices()
        # filter out examples that are too large
        max_positions = (max_position)
        if max_positions is not None:
            indices = filter_by_size(indices, sent, max_positions, raise_exception=(not True), )
        # create mini-batches with given size constraints
        max_sentences = self.args.max_sentences  # 30
        # print("batch size = ", max_sentences)
        batch_sampler = batch_by_size(indices, sent.num_tokens, max_sentences=max_sentences,
                                      required_batch_size_multiple=self.args.required_batch_size_multiple, )
        itrs = EpochBatchIterator(dataset=sent, collate_fn=sent.collater, batch_sampler=batch_sampler, seed=self.args.seed,
                                  num_workers=self.args.num_workers, epoch=epoch)
        # data_iter = itrs.next_epoch_itr(shuffle=False, fix_batches_to_gpus=fix_batches_to_gpus)

        return itrs
        # return data_iter
        # return data_loader

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

    def getdata(self, articles):
        trainingSetSrc = load_indexed_dataset(self.args.data + articles[0], self.task.src_dict,
                                              dataset_impl=self.args.dataset_impl, combine=False,
                                              default='cached')
        trainingSetTgt = load_indexed_dataset(self.args.data + articles[1], self.task.tgt_dict,
                                              dataset_impl=self.args.dataset_impl, combine=False,
                                              default='cached')
        # print("read the text file ")
        # convert the read files to Monolingual dataset to make padding easy
        src_mono = MonolingualDataset(dataset=trainingSetSrc, sizes=trainingSetSrc.sizes,
                                      src_vocab=self.task.src_dict,
                                      tgt_vocab=None, shuffle=False, add_eos_for_other_targets=False)
        tgt_mono = MonolingualDataset(dataset=trainingSetTgt, sizes=trainingSetTgt.sizes,
                                      src_vocab=self.task.tgt_dict,
                                      tgt_vocab=None, shuffle=False, add_eos_for_other_targets=False)

        del trainingSetSrc, trainingSetTgt
        return src_mono, tgt_mono



    def extract_and_train(self, comparable_data_list, epoch):
        # tracemalloc.start()
        # task specific setup per epoch
        self.task.begin_epoch(epoch, self.trainer.get_model())
        """ Manages the alternating extraction of parallel sentences and training.
        Args: =
            comparable_data_list(str): path to list of mapped documents
        Returns:
            train_stats(:obj:'onmt.Trainer.Statistics'): epoch loss statistics
        """
        # self.progress = None

        self.accepted_file = open('{}_accepted-e{}.txt'.format(self.comp_log, epoch), 'w+', encoding='utf8')
        '''if self.use_phrase == True:
            self.accepted_phrase = open('{}_accepted_phrase-e{}.txt'.format(self.comp_log, epoch), 'w+',
                                        encoding='utf8')
        '''
        # self.accepted_file_src = open('{}_accepted_src-e{}.txt'.format(self.comp_log, epoch), 'w+', encoding='utf8')
        # self.accepted_file_tgt = open('{}_accepted_tgt-e{}.txt'.format(self.comp_log, epoch), 'w+', encoding='utf8')
        self.status_file = '{}_status-e{}.txt'.format(self.comp_log, epoch)
        if self.write_dual:
            self.embed_file = '{}_accepted_embed-e{}.txt'.format(self.comp_log,
                                                                 epoch)
            self.hidden_file = '{}_accepted_hidden-e{}.txt'.format(self.comp_log,
                                                                   epoch)

        epoch_similarities = []
        epoch_scores = []
        src_sents = []
        tgt_sents = []
        src_embeds = []
        tgt_embeds = []

        '''profile = cProfile.Profile()
        profile.enable()'''
        # Go through comparable data
        with open(comparable_data_list, encoding='utf8') as c:
            comp_list = c.read().split('\n')
            # num_articles = len(comp_list)
            cur_article = 0
            for article_pair in comp_list:
                cur_article += 1
                articles = article_pair.split('\t')
                # Discard malaligned documents
                if len(articles) != 2:
                    continue
                # load the dataset from the files for both source and target
                src_mono, tgt_mono = self.getdata(articles)
                # Prepare iterator objects for current src/tgt document
                src_article = self._get_iterator(src_mono, dictn=self.task.src_dict,
                                                 max_position=self.args.max_source_positions, epoch=epoch,
                                                 fix_batches_to_gpus=False)
                tgt_article = self._get_iterator(tgt_mono, dictn=self.task.tgt_dict,
                                                 max_position=self.args.max_target_positions, epoch=epoch,
                                                 fix_batches_to_gpus=False)

                # Get sentence representations
                #try:
                if self.representations == 'embed-only':
                    # print("Using Embeddings only for representation")
                    # C_e
                    src_sents += self.get_article_coves(src_article, representation='embed', mean=False, side='src')
                    tgt_sents += self.get_article_coves(tgt_article, representation='embed', mean=False, side='tgt')
                else:
                    # C_e and C_h

                    it1 = src_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
                    src_embeds += self.get_article_coves(it1, representation='embed', mean=False, side='src',
                                                         use_phrase=self.use_phrase)
                    it1 = src_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
                    src_sents += self.get_article_coves(it1, representation='memory', mean=False, side='src')

                    it3 = tgt_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
                    tgt_embeds += self.get_article_coves(it3, representation='embed', mean=False, side='tgt',
                                                         use_phrase=self.use_phrase)
                    it3 = tgt_article.next_epoch_itr(shuffle=False, fix_batches_to_gpus=False)
                    tgt_sents += self.get_article_coves(it3, representation='memory', mean=False, side='tgt')

                    # return
                '''except:
                    # Skip document pair in case of errors
                    print("error")
                    src_sents = []
                    tgt_sents = []
                    src_embeds = []
                    tgt_embeds = []
                    continue
                '''
                # free resources for Gabbage, not necessary tho
                src_mono.dataset.tokens_list = None
                src_mono.dataset.sizes = None
                src_mono.sizes = None
                tgt_mono.sizes = None
                del tgt_mono
                del src_mono

                if len(src_sents) < 15 or len(tgt_sents) < 15:
                    # print("Length LEss tahn 15")
                    continue

                # Score src and tgt sentences

                #try:
                src2tgt, tgt2src, similarities, scores = self.score_sents(src_sents, tgt_sents)
                '''except:
                    # print('Error occurred in: {}\n'.format(article_pair), flush=True)
                    print(src_sents, flush=True)
                    print(tgt_sents, flush=True)
                    src_sents = []
                    tgt_sents = []
                    continue
                '''
                # print("source 2 target ", src2tgt)
                # Keep statistics
                # epoch_similarities += similarities
                # epoch_scores += scores
                src_sents = []
                tgt_sents = []

                #try:
                if self.representations == 'dual':
                    # For dual representation systems, filter C_h...
                    candidates = self.filter_candidates(src2tgt, tgt2src, second=self.second)
                    # ...and C_e
                    comparison_pool, cand_embed = self.get_comparison_pool(src_embeds,
                                                                           tgt_embeds)
                    src_embeds = []
                    tgt_embeds = []
                    if self.write_dual:
                        # print("writing the sentences to file....")
                        self.write_embed_only(candidates, cand_embed)
                else:
                    print("Using Embedings only for Filtering ......")
                    # Filter C_e or C_h for single representation system
                    candidates = self.filter_candidates(src2tgt, tgt2src)
                    comparison_pool = None
                '''except:
                    # Skip document pair in case of errors
                    print("Error Occured!!!!")
                    # print('Error occured in: {}\n'.format(article_pair), flush=True)
                    src_embeds = []
                    tgt_embeds = []
                    continue
                '''

                # Extract parallel samples (secondary filter)
                self.extract_parallel_sents(candidates, comparison_pool)
                #print('Pair bank content = ',self.similar_pairs.get_num_examples())


                #print("pair bank  = ",len((self.similar_pairs.pairs)))
                # Train on extracted sentences
                self.train(epoch)

                '''profile.disable()
                ps = pstats.Stats(profile)
                ps.sort_stats("call","cumtime")
                ps.print_stats(30)'''
                #del src2tgt, tgt2src
                # gc.collect()
                # Add to leaky code within python_script_being_profiled.py



                # Train on remaining partial batch
            if len((self.similar_pairs.pairs)) > 0:
                print("batching and training")
                self.train(epoch, last=True)

        self.accepted_file.close()

        # self.accepted_file_src.close()
        # self.accepted_file_tgt.close()

        # log end-of-epoch stats
        num_updates = self.trainer.get_num_updates()
        stats = get_training_stats(metrics.get_smoothed_values('train'))
        self.progress.print(stats, tag='train', step=num_updates)
        # reset epoch-level meters
        metrics.reset_meters('train')
        return None

    '''@metrics.aggregate('train')
    def trainRest(self, epoch):
        itrs = self.similar_pairs.yield_batch()
        itr = itrs.next_epoch_itr(shuffle=True, fix_batches_to_gpus=False)
        itr = GroupedIterator(itr, 1)
        self.progress = progress_bar.build_progress_bar(
            self.args, itr, epoch, no_progress_bar='simple',
        )
        for samples in self.progress:
            log_output = self.trainer.train_step(samples)
            num_updates = self.trainer.get_num_updates()
            if log_output is None:
                continue
            # log mid-epoch stats
        stats = get_training_stats(metrics.get_smoothed_values('train'))
        self.progress.print(stats, tag='train', step=num_updates)
        self.progress.log(stats, tag='train', step=num_updates)
            #print("logging here .....................****************************")


    def train(self, epoch , last = False):
        # Check if enough parallel sentences were collected
        train_stats = None
        while self.similar_pairs.contains_batch():
            # print("IT has batch.....")
            # try:
            itrs = self.similar_pairs.yield_batch()
            itr = itrs.next_epoch_itr(shuffle=True, fix_batches_to_gpus=False)
            itr = GroupedIterator(itr, 1)
            self.progress = progress_bar.build_progress_bar(
                self.args, itr, epoch, no_progress_bar='simple',
            )

            for samples in self.progress:
                log_output = self.trainer.train_step(samples)
                num_updates = self.trainer.get_num_updates()
                if log_output is None:
                    continue
                # log mid-epoch stats
                stats = get_training_stats(metrics.get_smoothed_values('train'))
                self.progress.log(stats, tag='train', step=num_updates)
        '''


    @metrics.aggregate('train')
    def train(self, epoch, last = False):
        # Check if enough parallel sentences were collected
        #if last is False:
        while self.similar_pairs.contains_batch():
            # print("IT has batch.....")
            # try:
            itrs = self.similar_pairs.yield_batch()
            itr = itrs.next_epoch_itr(shuffle=True, fix_batches_to_gpus=False)
            itr = GroupedIterator(itr, self.update_freq[-1])

            self.progress = progress_bar.build_progress_bar(
            self.args, itr, epoch, no_progress_bar='simple',
            )

            for samples in self.progress:
                with metrics.aggregate('train_inner'):
                    #print("Size of the sampel = ",len(samples))
                    log_output = self.trainer.train_step(samples)
                    num_updates = self.trainer.get_num_updates()
                    if log_output is None:
                        continue
                # log mid-epoch stats
                stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
                self.progress.print(stats, tag='train_inner', step=num_updates)
                self.progress.log(stats, tag='train_inner', step=num_updates)
                metrics.reset_meters('train_inner')
        '''else:
            #numberofex = self.similar_pairs.get_num_examples()
            itrs = self.similar_pairs.yield_batch()
            itr = itrs.next_epoch_itr(shuffle=True, fix_batches_to_gpus=False)

            itr = GroupedIterator(itr, self.update_freq[-1])
            self.progress = progress_bar.build_progress_bar(
                self.args, itr, epoch, no_progress_bar='simple',
            )
            for samples in self.progress:
                with metrics.aggregate('train_inner'):
                    log_output = self.trainer.train_step(samples)
                    num_updates = self.trainer.get_num_updates()
                    if log_output is None:
                        continue
                # log mid-epoch stats
                stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
                self.progress.print(stats, tag='train_inner', step=num_updates)
                self.progress.log(stats, tag='train_inner', step=num_updates)
                metrics.reset_meters('train_inner')'''


    def validate(self, epoch, subsets):
        """Evaluate the model on the validation set(s) and return the losses."""

        if self.args.fixed_validation_seed is not None:
            # set fixed seed for every validation
            utils.set_torch_seed(self.args.fixed_validation_seed)

        valid_losses = []
        for subset in subsets:
            # Initialize data iterator
            itr = self.task.get_batch_iterator(
                dataset=self.task.dataset(subset),
                max_tokens=self.args.max_tokens_valid,
                max_sentences=self.args.max_sentences_valid,
                max_positions=utils.resolve_max_positions(
                    self.task.max_positions(),
                    self.trainer.get_model().max_positions(),
                ),
                ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=self.args.required_batch_size_multiple,
                seed=self.args.seed,
                num_shards=self.args.distributed_world_size,
                shard_id=self.args.distributed_rank,
                num_workers=self.args.num_workers,
            ).next_epoch_itr(shuffle=False)
            progress = progress_bar.build_progress_bar(
                self.args, itr, epoch,
                prefix='valid on \'{}\' subset'.format(subset),
                no_progress_bar='simple'
            )

            # create a new root metrics aggregator so validation metrics
            # don't pollute other aggregators (e.g., train meters)
            with metrics.aggregate(new_root=True) as agg:
                for sample in progress:
                    self.trainer.valid_step(sample)

            # log validation stats
            stats = get_valid_stats(self.args, self.trainer, agg.get_smoothed_values())
            progress.print(stats, tag=subset, step=self.trainer.get_num_updates())

            valid_losses.append(stats[self.args.best_checkpoint_metric])
        return valid_losses

    def save_comp_chkp(self, epoch):
        dirs = self.save_dir + '/' + self.model_name + '_' + str(epoch) + self.src + "-" + self.tgt + ".pt"
        self.trainer.save_checkpoint(dirs, {"train_iterator": {"epoch": epoch}})


def get_valid_stats(args, trainer, stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args.best_checkpoint_metric],
        )
    return stats


def get_training_stats(stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def removePadding(side):
    """ Removes original padding from a sequence.
    Args:
        side(torch.Tensor): src/tgt sequence (size(seq))
    Returns:
        side(torch.Tensor): src/tgt sequence without padding
    NOTE: This only works as long as PAD_ID==1!
    """
    # Get indexes of paddings in sequence
    padding_idx = (side == 1).nonzero()
    # If there is any padding, cut sequence from first occurence of a pad
    if padding_idx.size(0) != 0:
        first_pad = padding_idx.data.tolist()[0][0]
        side = side[:first_pad]
    return side

