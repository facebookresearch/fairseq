import logging
import os
import torch
from fairseq import utils
from fairseq.data.noising import UnsupervisedMTNoising

from ..data.swav_dataset import SwavExtrapolateNoisingDataset
from ..data.swav_dataset import SwavExtrapolatePrependTokenDataset
from fairseq.distributed import utils as distributed_utils

logger = logging.getLogger(__name__)


class UnsupervisedMTNoisingNoBpe(UnsupervisedMTNoising):
    """
    UnsupervisedMTNoising but not consider bpe token at all
    """
    def __init__(
        self, dictionary, max_word_shuffle_distance, word_dropout_prob, 
        word_blanking_prob, bpe_cont_marker=None, bpe_end_marker=None
    ):
        bpe_cont_marker = None
        bpe_end_marker = None
        super().__init__(
            dictionary, max_word_shuffle_distance, word_dropout_prob, word_blanking_prob, 
            bpe_cont_marker=bpe_cont_marker, bpe_end_marker=bpe_end_marker
        )


def decode_fn(x, bpe, tokenizer):
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x


def tokens_to_texts(tokens, src_dict, post_process, decode_fn):
    texts = []
    for i in range(tokens.size(0)):
        tok = utils.strip_pad(tokens[i], src_dict.pad())
        src_str = src_dict.string(tok, post_process)
        src_str = decode_fn(src_str)
        texts.append(src_str)
    return texts


def tokens_list_to_texts(tokens_list, src_dict, post_process, decode_fn):
    texts = []
    for tokens in tokens_list:
        for i in range(tokens.size(0)):
            tok = utils.strip_pad(tokens[i], src_dict.pad())
            src_str = src_dict.string(tok, post_process)
            src_str = decode_fn(src_str)
            texts.append(src_str)
    return texts


class VanillaSwavBaseTaskWrapper(object):
    # NOTE this class is to provide functionality of Swav Task to Downstream tasks
    @classmethod
    def add_swav_args(cls, parser):
        # taken from UnsupervisedMTNoising
        parser.add_argument('--max-word-shuffle-distance', default=3.0, type=float, metavar='N',
                            help='maximum word shuffle distance for denoising autoencoding data generation')
        parser.add_argument('--word-dropout-prob', default=0.1, type=float, metavar='N',
                            help='word dropout probability for denoising autoencoding data generation')
        parser.add_argument('--word-blanking-prob', default=0.2, type=float, metavar='N',
                            help='word blanking probability for denoising autoencoding data generation')
        parser.add_argument('--noising-module', default=UnsupervisedMTNoising.__name__, type=str,
                            help='name of the noising module use')
        parser.add_argument("--subsequent-loss", default=False, action="store_true",
                            help="Compute mlm->swav loss, instead of mlm+swav lss")
        parser.add_argument("--no-token-block", default=False, action="store_true",
                            help="Turn off TokenBlockDataset in Dataset")
        parser.add_argument("--prepend-eos", default=False, action="store_true",
                            help="Turn off TokenBlockDataset in Dataset")
        parser.add_argument("--swav-eval-mode", default='train_prot', type=str,
                            help="swav-eval-mode")
        parser.add_argument("--aly-exclude", default='', type=str,
                            help="elements to exclude from analayize")

        parser.add_argument("--freeze-prototypes-niters", default=313, type=int,
                            help="freeze the prototypes during this many iterations from the start")
    
    @staticmethod
    def get_noising_module(args):
        module_name = getattr(args, 'noising_module', None)
        if module_name is None:
            module_name = UnsupervisedMTNoising.__name__
        if module_name == "UnsupervisedMTNoising":
            return UnsupervisedMTNoising
        elif module_name == "UnsupervisedMTNoisingNoBpe":
            return UnsupervisedMTNoisingNoBpe
        else:
            raise ValueError(f'noising module {module_name} not found')
    
    def analyze_noswav_step(self, sample, model, criterion, cfg, **kwargs):
        """
        Analyze anything related to swav mdoel, BUT NOT FOR SWAV model
        use with swav_project/scripts/fi_analyze.py
        fi_analyze.py temporarily allow breaking fairseq cfg structure to acquire to cfg variable
        """
        is_dummy = kwargs.get('is_dummy', False)
        aly_exclude = self.args.aly_exclude.split(",")
        model.eval()
        for k in aly_exclude:
            self._aly_agg_data.pop(k, None)
            
        with torch.no_grad():
            embed = model(**sample['net_input'], features_only=True)[0]
            prot_embed = embed.detach()[:, 0]

            return_output = {
                'id': sample['id'].cpu(),
            }
            if "lang_id" not in aly_exclude:
                return_output['lang_id'] = sample['lang_id'].cpu()
            
            if "embed" not in aly_exclude:
                return_output['embed'] = prot_embed.detach().float().cpu()
            
            if "tokens" not in aly_exclude:
                return_output['tokens'] = sample['net_input']['src_tokens'].cpu()

            if not is_dummy:
                for k in return_output.keys():
                    self._aly_agg_data[k].append(return_output[k])
        return return_output
        
    def analyze_step(self, sample, model, criterion, cfg, **kwargs):
        """
        Analyze anything related to swav mdoel, use with swav_project/scripts/fi_analyze.py
        fi_analyze.py temporarily allow breaking fairseq cfg structure to acquire to cfg variable
        """

        criterion.distributed_world_size = cfg.distributed_training.distributed_world_size
        aly_exclude = self.args.aly_exclude.split(",")
        is_dummy = kwargs.get('is_dummy', False)

        self._aly_agg_data = getattr(self, "_aly_agg_data", {
            "id": [],
            "lang_id": [],
            "tokens": [],
            "embed": [],
        })

        self._aly_queue = None  # FIXME nxphi: queue is not impl now

        if not hasattr(model, 'prototypes'):
            return self.analyze_noswav_step(sample, model, criterion, cfg, **kwargs)

        for k in ['prototypes', 'sinkhorn_prototypes']:
            if k not in self._aly_agg_data:
                self._aly_agg_data[k] = []
        
        for k in aly_exclude:
            self._aly_agg_data.pop(k, None)
        
        model.eval()
        with torch.no_grad():
            try:
                sinkhorn_out, prot_out, prot_embed, rest_extra = criterion.compute_sinkhorn_prototypes(
                    model, sample, queue=self._aly_queue
                )
            except Exception:
                sinkhorn_out, prot_out, prot_embed = criterion.compute_sinkhorn_prototypes(
                    model, sample, queue=self._aly_queue)
                rest_extra = None
            return_output = {
                'id': sample['id'].cpu(),
            }
            if "lang_id" not in aly_exclude:
                return_output['lang_id'] = sample['lang_id'].cpu()
            
            if "embed" not in aly_exclude:
                return_output['embed'] = prot_embed.detach().float().cpu()
            
            if "tokens" not in aly_exclude:
                return_output['tokens'] = sample['net_input']['src_tokens'].cpu()

            if "sinkhorn_prototypes" not in aly_exclude:
                return_output['sinkhorn_prototypes'] = sinkhorn_out.detach().float().cpu()
            
            if "prototypes" not in aly_exclude:
                return_output['prototypes'] = prot_out.detach().float().cpu()
            
            if not is_dummy:
                for k in return_output.keys():
                    self._aly_agg_data[k].append(return_output[k])

                if rest_extra is not None and bool(rest_extra):
                    rest_extra = {k: v.detach().float().cpu() for k, v in rest_extra.items()}
                    if "extra" not in self._aly_agg_data:
                        self._aly_agg_data['extra'] = {}
                        for k, v in rest_extra.items():
                            self._aly_agg_data['extra'][k] = [v]
                    else:
                        for k, v in rest_extra.items():
                            self._aly_agg_data['extra'][k].append(v)
        return return_output
        
    def analyze_done(self, cfg, infer_name, save_path, only_master=False, save=True, convert_text=True, **kwargs):
        """
        Finish and saving the outputs from models into save_obj
        """
        aly_exclude = self.args.aly_exclude.split(",")
        if only_master and not distributed_utils.is_master(cfg.distributed_training):
            _ = [delattr(self, at) for at in dir(self) if at.startswith('_aly_')]
            return
        world_size, rank = kwargs.get('world_size', 1), kwargs.get('rank', 0)
        rank_reprs = f'[{rank}/{world_size}]'

        self._aly_decode_fn = getattr(self, '_aly_decode_fn', lambda x: decode_fn(
            x, self.build_bpe(cfg.bpe), self.build_tokenizer(cfg.tokenizer)))
        
        save_obj = {}
        texts = None
        if "tokens" in self._aly_agg_data and convert_text and "text" not in aly_exclude:
            logger.warning(f'{rank_reprs} Building bpe text from tokens')
            texts = tokens_list_to_texts(
                self._aly_agg_data['tokens'], self.source_dictionary, 
                cfg.common_eval.post_process, self._aly_decode_fn)
            save_obj['text'] = texts
            logger.warning(f'{rank_reprs} Agg text: {len(texts)}')
            for i, t in enumerate(texts[:5]):
                logger.warning(f'{i}-{t}')
        
        for k, v in self._aly_agg_data.items():
            if isinstance(v, dict):
                # extra
                save_obj[k] = {}
                for dk, dv in v.items():
                    save_obj[k][dk] = torch.cat(dv, 0)
                    logger.warning(f'{rank_reprs} Agg {k}[{dk}]: {save_obj[k][dk].size()}')
            else:
                assert isinstance(v, list)
                try:
                    save_obj[k] = torch.cat(v, 0)
                    logger.warning(f'{rank_reprs} Agg {k}: {save_obj[k].size()}')
                except Exception:
                    save_obj[k] = v
                    logger.warning(f'{rank_reprs} Agg: Not concat {k}, {len(v)}')
                
        if save:
            proto_fname = '{}.pth'.format(infer_name)
            os.makedirs(save_path, exist_ok=True)
            proto_path = os.path.join(save_path, proto_fname)
            torch.save(save_obj, proto_path)
            logger.warning(f'{rank_reprs} Saved prots at {proto_path}')
        else:
            logger.warning(f'{rank_reprs} not saving prototypes data...')
        _ = [delattr(self, at) for at in dir(self) if at.startswith('_aly_')]
        return save_obj
    
    @property
    def swav_prepend_token(self):
        return None
    
    def create_swav_noising_dataset(self, dataset, dictionary, **kwargs):
        # NOTE: expected raw FairseqDataset (sentence-based) (not token-blocked)
        #   but input TokenBlockDatset may still work, right?
        dataset = SwavExtrapolateNoisingDataset(
            dataset, dictionary, 
            seed=kwargs.get('seed', self.args.seed), 
            rand_factor=kwargs.get('rand_factor', self.args.rand_factor),
            noising_class=self.get_noising_module(self.args),
            max_word_shuffle_distance=kwargs.get('max_word_shuffle_distance', self.args.max_word_shuffle_distance),
            word_dropout_prob=kwargs.get('word_dropout_prob', self.args.word_dropout_prob),
            word_blanking_prob=kwargs.get('word_blanking_prob', self.args.word_blanking_prob),
        )
        if self.swav_prepend_token is not None:
            dataset = SwavExtrapolatePrependTokenDataset(dataset, self.swav_prepend_token)
        return dataset



