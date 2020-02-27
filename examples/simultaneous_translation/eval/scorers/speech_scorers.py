import sys
import os
import soundfile as sf
sys.path.append("..") 
from . scorer import SimulScorer
from . import register_scorer

@register_scorer("speech")
class SimulSpeechScorer(SimulScorer):
    def __init__(self, args):
        super().__init__(args)
        if args.src_file is not None:
            sys.stderr.write(f"src_file {args.src_file} will be ignored.\n")

        self.tokenizer = args.tokenizer
        self.lengths = self._load_wav_info_from_json(args.tgt_file),
        self.data = {
            "src" : self._load_wav_info_from_json(args.tgt_file),
            "tgt" : self._load_text_from_json(args.tgt_file)
        }
        self.block_size = args.block_size
        self.sample_rate = args.sample_rate

    @staticmethod
    def add_args(parser):
        parser.add_argument('--sample-rate', type=int, default=16000,
                            help='Sample rate for the audio (Hz)')
        parser.add_argument('--block-size', type=int, default=10,
                            help='Block size (ms)')

    def send_src(self, sent_id, block_size=10):

        assert block_size >= self.block_size # in ms

        if (
            self.steps[sent_id] == 0 
            and "wav" not in self.data["src"][sent_id]
        ):
            # Load audio file
            self.data["src"][sent_id]["wav"] = self._load_audio_from_path(
                self.data["src"][sent_id]["path"]
            ) 
        
        num_block = block_size // self.block_size

        if self.steps[sent_id] < self.data["src"][sent_id]["length"]:

            segment = []
            start_block_idx = self.steps[sent_id] // self.block_size 
            for i in range(num_block):
                segment += self.data["src"][sent_id]["wav"][start_block_idx + i]

            dict_to_return = {
                "sent_id" : sent_id,
                "segment_id": self.steps[sent_id],
                "segment": segment
            }

            self.steps[sent_id] = min(
                [
                    self.data["src"][sent_id]["length"],
                    self.steps[sent_id] + self.block_size * num_block
                ]
            )


        else:
            # Finish reading this audio
            dict_to_return = {
                "sent_id" : sent_id,
                "segment_id": self.steps[sent_id],
                "segment": self.eos
            }
            del self.data["src"][sent_id]["wav"]

        return dict_to_return

    def src_lengths(self):
        return [item["length"] for item in self.data["src"]]
    
    def _load_audio_from_path(self, wav_path):
        assert os.path.isfile(wav_path) and wav_path.endswith('.wav')
        frames_10ms = self.sample_rate // 1000 * self.block_size
        wav_blocks = [b.tolist() for b in sf.blocks(wav_path, blocksize=frames_10ms)]
        return wav_blocks
