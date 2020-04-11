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
        self.segment_size = args.segment_size
        self.sample_rate = args.sample_rate
        self.wav_data_type = args.wav_data_type

    @staticmethod
    def add_args(parser):
        parser.add_argument('--sample-rate', type=int, default=16000,
                            help='Sample rate for the audio (Hz)')
        parser.add_argument('--segment-size', type=int, default=10,
                            help='Segment size (ms)')
        parser.add_argument('--wav-data-type', type=str, default="int16",
                            help='The data type of the wav that would be transfer to client')

    def send_src(self, sent_id, client_segment_size):
        if client_segment_size is not None:
            assert client_segment_size >= self.segment_size # in ms
        else:
            client_segment_size = self.segment_size

        if (
            self.steps[sent_id] == 0 
            and "segments" not in self.data["src"][sent_id]
        ):
            # Load audio file
            self.data["src"][sent_id]["segments"] = self._load_audio_from_path(
                self.data["src"][sent_id]["path"]
            ) 
        
        num_segments = client_segment_size // self.segment_size

        if self.steps[sent_id] < self.data["src"][sent_id]["length"]:

            segment = []
            start_idx = self.steps[sent_id] // self.segment_size 
            for i in range(num_segments):
                if start_idx + i < len(self.data["src"][sent_id]["segments"]):
                    segment += self.data["src"][sent_id]["segments"][start_idx + i]

            dict_to_return = {
                "sent_id" : sent_id,
                "segment_id": int(self.steps[sent_id] / self.segment_size),
                "segment": segment
            }

            self.steps[sent_id] = min(
                [
                    self.data["src"][sent_id]["length"],
                    self.steps[sent_id] + self.segment_size * num_segments
                ]
            )


        else:
            # Finish reading this audio
            dict_to_return = {
                "sent_id" : sent_id,
                "segment_id": int(self.steps[sent_id] / self.segment_size),
                "segment": self.eos
            }
            if "segments" in self.data["src"][sent_id]:
                del self.data["src"][sent_id]["segments"]

        return dict_to_return

    def src_lengths(self):
        return [item["length"] for item in self.data["src"]]
    
    def _load_audio_from_path(self, wav_path):
        assert os.path.isfile(wav_path) and wav_path.endswith('.wav')
        frames_10ms = self.sample_rate // 1000 * self.segment_size
        wav_blocks = [
            b.tolist() 
            for b in sf.blocks(
                wav_path,
                blocksize=frames_10ms,
                dtype=self.wav_data_type
                )
        ]
        return wav_blocks
