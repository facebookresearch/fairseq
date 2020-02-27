from vizseq.scorers.bleu import BLEUScorer
from vizseq.scorers.ter import TERScorer
from vizseq.scorers.meteor import METEORScorer
from examples.simultaneous_translation.utils.eval_latency import LatencyScorer
from collections import defaultdict
import json

DEFAULT_EOS = '</s>'
class SimulScorer(object):
    def __init__(self, args):
        self.tokenizer = args.tokenizer
        if args.output is not None:
            self.output_files = {
                "text": args.output + ".text",
                "delay": args.output + ".delay",
                "scores": args.output + ".scores"
            }
        else:
            self.output_files = None
        self.eos = DEFAULT_EOS
        self.data = {"tgt": []}
        self.reset()
    
    def get_info(self):
        return {"num_sentences": len(self)}
        
    def send_src(self, sent_id):
        raise NotImplementedError

    def recv_hyp(self, hypo):
        for sent_id, trans in hypo.items():
            self.translations[
                int(sent_id)
            ].append(
                (
                    trans, 
                    self.steps[int(sent_id)]
                )
            )

    def reset(self):
        self.steps = defaultdict(int)
        self.translations = defaultdict(list)
    
    def src_lengths(self):
        raise NotImplementedError

    def score(self):
        translations = []
        delays = []
        for i in range(1 + max(self.translations.keys())):
            translations += [" ".join(t[0] for t in self.translations[i][:-1])]
            delays += [[t[1] for t in self.translations[i][:-1]]]

        bleu_score = BLEUScorer(
            sent_level=False, corpus_level=True,
            extra_args={'bleu_tokenizer': self.tokenizer}
        ).score(translations, [self.data["tgt"]])

        ter_score = TERScorer(sent_level=False, corpus_level=True).score(
            translations, [self.data["tgt"]]
        )
        meteor_score = METEORScorer(sent_level=False, corpus_level=True).score(
            translations, [self.data["tgt"]]
        )


        latency_score = LatencyScorer().score(
            [{"src_len" : src_len, "delays" : delay} for src_len, delay in zip(self.src_lengths(), delays)]
        )

        scores = {
            'BLEU': bleu_score[0], 
            'TER': ter_score[0], 
            'METEOR': meteor_score[0],
            'DAL' : latency_score['differentiable_average_lagging'],
            'AL' : latency_score['average_lagging'],
            'AP' : latency_score['average_proportion'],
        }

        if self.output_files is not None:
            self.write_results_to_file(translations, delays, scores)
        
        return scores

    def write_results_to_file(self, translations, delays, scores):
        if self.output_files["text"] is not None: 
            with open(self.output_files["text"], "w") as f:
                for line in translations:
                    f.write(line + "\n")

        if self.output_files["delay"] is not None: 
            with open(self.output_files["delay"], "w") as f:
                for i, delay in enumerate(delays):
                    f.write(
                        json.dumps(
                            {
                                "src_len": self.src_lengths()[i],
                                "delays" : delay
                            }
                        ) + "\n"
                    )

        with open(self.output_files["scores"], "w") as f:
            for key, value in scores.items():
                f.write(f"{key}, {value}\n")
        

    @classmethod
    def _load_text_file(cls, file, split=False):
        with open(file) as f:
            if split:
                return [r.strip().split() for r in f]
            else:
                return [r.strip() for r in f]
    
    @classmethod
    def _load_text_from_json(cls, file):
        list_to_return = []
        with open(file) as f:
            content = json.load(f)
            for item in content["utts"].values():
                list_to_return.append(item["output"]["text"].strip())
        return list_to_return

    @classmethod
    def _load_wav_info_from_json(cls, file):
        list_to_return = []
        with open(file) as f:
            content = json.load(f)
            for item in content["utts"].values():
                list_to_return.append(
                    {
                        "path": item["input"]["path"].strip(),
                        "length": item["input"]["length_ms"]
                    }
                )
        return list_to_return

    def __len__(self):
        return len(self.data["tgt"])