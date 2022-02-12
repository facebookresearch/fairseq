import timeit
import logging
import torch
from pypapi import events, papi_high as high
from memory_profiler import memory_usage
from torch import nn
from argparse import Namespace
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.data import data_utils as fairseq_data_utils
from fairseq import checkpoint_utils, tasks, utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder
from examples.hubert.simple_kmeans.dump_hubert_feature import HubertFeatureReader
from examples.hubert.simple_kmeans.dump_km_label import ApplyKmeans
from fairseq_cli.generate import get_symbols_to_strip_from_output
import soundfile as sf
import ast
import json

logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


torch.manual_seed(1)
torch.set_deterministic(True)


class BenchmarkingBase(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.s2x_task = None

    def warm_up(self, sample, repeat):
        """Warm up the model"""
        for _i in range(repeat):
            self.forward(sample)
        logger.info(f"Model warmed up by running inference {repeat} times")

    def benchmark_run_time(self, dataset, repeat):
        """Benchmark average runtime for the model by calling benchmark_run_time_single_sample function"""
        logger.info("Starting run time benchmarking")
        time_elapsed = 0
        for i, sample in enumerate(dataset):
            time_elapsed += self.benchmark_run_time_single_sample(sample, repeat=repeat)
            if i % 100 == 0:
                logger.info(f"Benchmarked run time for {i}/{len(dataset)} samples")
        total_time_elapsed = time_elapsed / len(dataset)
        return total_time_elapsed

    def benchmark_run_time_single_sample(self, sample, repeat):
        """Benchmark average runtime for a single sample using timeit library. Units are seconds"""
        timer = timeit.Timer(lambda: self.forward(sample))
        time_elapsed = timer.timeit(repeat)
        return time_elapsed / repeat

    def count_flops(
        self,
        dataset,
        repeat,
    ):
        """Use PYPAPI library to count average flops for model inference.
        Note: It only works if the model is being run on cpu"""
        logger.info("Starting flop counter")
        high.start_counters([events.PAPI_DP_OPS])
        for i, sample in enumerate(dataset):
            for _r in range(repeat):
                self.forward(sample)
            if i % 100 == 0:
                logger.info(f"Counted flops for {i}/{len(dataset)} samples")
        flops = high.stop_counters()
        flops = round(flops[0] / (repeat * len(dataset)))
        return flops

    def max_memory(self, dataset, repeat):
        """Compute average max memory consumed by model inference. Units are MiB"""
        logger.info("Starting memory benchmarking")
        total_memory = 0
        for i, sample in enumerate(dataset):
            for _r in range(repeat):
                total_memory += max(memory_usage((self.forward, (sample,), {})))
            if i % 100 == 0:
                logger.info(f"Benchmarked memory for {i}/{len(dataset)} samples")
        total_memory = total_memory / (repeat * len(dataset))
        return total_memory

    def gather_all_metrics(self, dataset, repeat):
        run_time = self.benchmark_run_time(dataset, repeat)
        max_memory = self.max_memory(dataset, repeat)
        flops = self.count_flops(dataset, repeat)

        return run_time, max_memory, flops

    def dump_final_speech_output(
        self, dataset, output_dir, resample_fn, sample_rate, prefix=None
    ):

        for i, sample in enumerate(dataset):
            hypo = self.forward(sample)[0]

            def to_np(x):
                return x.detach().cpu().numpy()

            try:
                wave_preds = to_np(resample_fn(hypo["waveform"]))
                sf.write(
                    f"{output_dir}/{prefix}_{i}_pred.wav",
                    wave_preds,
                    sample_rate,
                )
            except Exception as e:
                raise Exception(
                    f" Encountered {e} - Invalid waveform. Make sure the model outputs a waveform"
                )


class Processing(BenchmarkingBase):
    """Class similar to fairseq_cli/generate.py. Supports ASR, MT and ST model inference"""

    def __init__(self, args):
        super().__init__()
        self.use_cuda = not getattr(args, "cpu", False)
        self.setUp(args)
        self.training = False
        self.s2x_task = self.task

    def setUp(self, cfg):
        if isinstance(cfg, Namespace):
            cfg = convert_namespace_to_omegaconf(cfg)

        self.task = tasks.setup_task(cfg.task)
        self.tgt_dict = self.task.target_dictionary

        # Load ensemble
        logger.info("loading model(s) from {}".format(cfg.common_eval.path))
        models, _ = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides={},
            task=self.task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=False,
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )
        if len(models) > 1:
            raise Exception("Currently loading multiple models is not supported")
        self.model = models[0]

        # Optimize model for generation
        if cfg.common.fp16:
            self.model.half()
        if self.use_cuda:
            self.model.cuda()
        self.model.prepare_for_inference_(cfg)

        self.generator = self.task.build_generator(
            [self.model],
            cfg.generation,
            extra_gen_cls_kwargs={},
        )
        # Handle tokenization and BPE
        self.tokenizer = self.task.build_tokenizer(cfg.tokenizer)
        self.bpe = self.task.build_bpe(cfg.bpe)
        self.remove_bpe = cfg.common_eval.post_process

    def encode_source(self, src):
        """Method to generate source tokens from a string"""
        if self.tokenizer is not None:
            src = self.tokenizer.encode(src)
        if self.bpe is not None:
            src = self.bpe.encode(src)
        src_tokens = self.task.source_dictionary.encode_line(src).long()
        src_lens = src_tokens.size(0)
        return {
            "net_input": {
                "src_tokens": src_tokens.view(1, src_lens),
                "src_lengths": torch.tensor([src_lens]),
            }
        }

    def decode_target(self, hypos):
        """Method to decode target string from tokens"""
        hypo_str = self.tgt_dict.string(
            hypos[0][0]["tokens"].int().cpu(),
            self.remove_bpe,
            get_symbols_to_strip_from_output(self.generator),
        )
        if self.bpe is not None:
            hypo_str = self.bpe.decode(hypo_str)
        if self.tokenizer is not None:
            hypo_str = self.tokenizer.decode(hypo_str)
        return hypo_str

    def forward(self, sample):
        hypos = self.task.inference_step(
            self.generator,
            [self.model],
            sample,
            prefix_tokens=None,
            constraints=None,
        )
        return hypos


class GenerateWaveformFromCode(BenchmarkingBase):
    """Class to support waveform generation from code. Currently, vocoder only supports single speaker"""

    def __init__(self, args):
        super().__init__()
        with open(args.vocoder_cfg) as f:
            vocoder_cfg = json.load(f)
        self.dur_prediction = args.dur_prediction
        self.vocoder = CodeHiFiGANVocoder(args.vocoder, vocoder_cfg)

    def format_units(self, input):
        code = torch.LongTensor(list(map(int, input.strip().split()))).view(1, -1)
        return {"code": code}

    def generate_vocoder_input(self, dataset):
        return [self.format_units(sample) for sample in dataset]

    def forward(self, sample):
        return [{"waveform": self.vocoder(sample, self.dur_prediction)}]


class HubertUnitExtractor(BenchmarkingBase):
    def __init__(self, args):
        self.feature_reader = HubertFeatureReader(
            args.hubert_ckpt_path, args.hubert_layer
        )
        self.kmeans = ApplyKmeans(args.hubert_km_path)

    def forward(self, sample):
        with torch.no_grad():
            feat = []
            for start in range(0, sample.size(1), self.feature_reader.max_chunk):
                x_chunk = sample[:, start : start + self.max_chunk]
                feat_chunk, _ = self.feature_reader.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
            torch.cat(feat, 1).squeeze(0)
        return self.kmeans(feat).tolist()


class SpeechGeneration(BenchmarkingBase):
    """Class similar to examples/text_to_speech/generate_waveform.py.
    Supports models with speech generation as end goal (TTS, Direct S2ST models etc)"""

    def __init__(self, args):
        super().__init__()
        self.use_cuda = not getattr(args, "cpu", False)
        self.setUp(args)
        self.s2x_task = self.task

    def setUp(self, args):
        if args.task == "speech_to_speech":
            args.normalize_waveform = False
        self.task = tasks.setup_task(args)
        self.pre_tokenizer = self.task.build_tokenizer(args)
        self.bpe_tokenizer = self.task.build_bpe(args)
        try:
            self.src_dict = self.task.src_dict
        except Exception:
            self.src_dict = None
        ensemble, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [args.path],
            arg_overrides=ast.literal_eval(args.model_overrides),
            task=self.task,
            strict=False,
        )
        self.model = ensemble[0]
        if self.use_cuda:
            self.model.cuda()
            # criterion.cuda()
        self.model.eval()
        self.generator = self.task.build_generator(
            [self.model],
            args,
        )

    def processTextInput(self, text):
        """Generate source tokens from text input"""
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.bpe_tokenizer is not None:
            text = self.bpe_tokenizer.encode(text)
        target = self.src_dict.encode_line(
            text, add_if_not_exist=False, append_eos=True
        ).long()
        target = fairseq_data_utils.collate_tokens(
            [target],
            self.src_dict.pad(),
            self.src_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        src_lengths = torch.tensor([target.size(1)], dtype=torch.long)
        prev_output_tokens = None
        sample = {
            "net_input": {
                "src_tokens": target,
                "src_lengths": src_lengths,
                "prev_output_tokens": prev_output_tokens,
            }
        }
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample
        return sample

    def forward(self, sample):
        sample["speaker"] = None
        output = self.generator.generate(self.model, sample)  # , has_targ=False
        return output


class S2UT(BenchmarkingBase):
    """Class to support S2UT models. Also supports generating waveforms from the units predicted"""

    def __init__(self, s2u_args, vocoder_args=None):
        super().__init__()
        self.s2u = Processing(s2u_args)
        self.vocoder = None
        if vocoder_args:
            self.vocoder = GenerateWaveformFromCode(vocoder_args)
        self.vocoder_input = None

    def forward(self, sample):
        s2u_hypos = self.s2u(sample)
        s2u_output = self.s2u.decode_target(s2u_hypos)
        if not self.vocoder:
            return s2u_output
        units = self.vocoder.format_units(s2u_output)
        vocoder_output = self.vocoder(units)
        return vocoder_output

    def generate_s2u_outputs(self, dataset):
        return [self.s2u.decode_target(self.s2u(sample)) for sample in dataset]

    def compute_metrics(self, metric_type, dataset, repeat=None):
        """Generic function to compute metrics ignoring the io processing time"""
        if self.vocoder and not self.vocoder_input:
            self.s2u_output = self.generate_s2u_outputs(dataset)
            self.vocoder_input = self.vocoder.generate_vocoder_input(self.s2u_output)

        s2u_metrics = getattr(self.s2u, metric_type)(
            dataset,
            repeat,
        )
        vocoder_metrics = 0
        if self.vocoder:
            vocoder_metrics = getattr(self.vocoder, metric_type)(
                self.vocoder_input,
                repeat,
            )
        print(
            f"metric_type = {metric_type} s2u_metrics = {s2u_metrics} \t vocoder_metrics = {vocoder_metrics}"
        )
        if metric_type == "max_memory":
            return max(s2u_metrics, vocoder_metrics)
        else:
            return s2u_metrics + vocoder_metrics

    def benchmark_run_time(self, dataset, repeat):
        return self.compute_metrics("benchmark_run_time", dataset, repeat)

    def count_flops(self, dataset, repeat):
        return self.compute_metrics("count_flops", dataset, repeat)

    def max_memory(self, dataset, repeat):
        return self.compute_metrics("max_memory", dataset, repeat)


class Cascaded2StageS2ST(BenchmarkingBase):
    """ST + TTS"""

    def __init__(self, s2t_args, tts_args):
        super().__init__()
        self.s2t = Processing(s2t_args)
        self.s2x_task = self.s2t.task
        self.tts = SpeechGeneration(tts_args) if tts_args else None
        self.training = False
        self.tts_inputs = None

    def forward(self, sample):
        if not self.tts:
            raise Exception(
                "Forward function is not callable without tts. Reinitialize the class with tts_args"
            )
        s2t_hypos = self.s2t(sample)
        s2t_output = self.s2t.decode_target(s2t_hypos)
        tts_input = self.tts.processTextInput(s2t_output)
        tts_output = self.tts(tts_input)
        return tts_output

    def generate_s2t_outputs(self, dataset):
        """Process dataset and generate s2t outputs"""
        return [self.s2t.decode_target(self.s2t(sample)) for sample in dataset]

    def generate_tts_inputs(self, dataset):
        """Process dataset and generate tts inputs"""
        return [self.tts.processTextInput(sample) for sample in dataset]

    def compute_metrics(self, metric_type, dataset, repeat=None):
        """Generic function to compute metrics ignoring the io processing time"""
        if not self.tts_inputs:
            s2t_outputs = self.generate_s2t_outputs(dataset)
            self.tts_inputs = self.generate_tts_inputs(s2t_outputs)

        s2t_metrics = getattr(self.s2t, metric_type)(
            dataset,
            repeat,
        )

        tts_metrics = getattr(self.tts, metric_type)(
            self.tts_inputs,
            repeat,
        )
        print(
            f"metric_type = {metric_type} s2t_metrics = {s2t_metrics} \t tts_metrics = {tts_metrics}"
        )
        if metric_type == "max_memory":
            return max(s2t_metrics, tts_metrics)
        else:
            return s2t_metrics + tts_metrics

    def benchmark_run_time(self, dataset, repeat):
        return self.compute_metrics("benchmark_run_time", dataset, repeat)

    def count_flops(self, dataset, repeat):
        return self.compute_metrics("count_flops", dataset, repeat)

    def max_memory(self, dataset, repeat):
        return self.compute_metrics("max_memory", dataset, repeat)


class Cascaded3StageS2ST(Cascaded2StageS2ST):
    """ASR + MT + TTS"""

    def __init__(self, s2t_args, tts_args, mt_args):
        super().__init__(s2t_args, tts_args)
        self.mt = Processing(mt_args)
        self.mt_inputs = []

    def forward(self, sample):
        s2t_hypos = self.s2t(sample)
        s2t_output = self.s2t.decode_target(s2t_hypos)
        mt_input = self.mt.encode_source(s2t_output)
        mt_hypos = self.mt(mt_input)
        mt_output = self.mt.decode_target(mt_hypos)
        tts_input = self.tts.processTextInput(mt_output)
        tts_output = self.tts(tts_input)
        return tts_output

    def generate_mt_inputs(self, dataset):
        """Process dataset to generate mt model inputs"""
        return [self.mt.encode_source(sample) for sample in dataset]

    def generate_mt_outputs(self, dataset):
        """Process dataset to generate mt model outputs"""
        return [self.mt.decode_target(self.mt(sample)) for sample in dataset]

    def compute_metrics(self, metric_type, dataset, repeat=None):
        """Generic function to compute metrics ignoring the io processing time"""
        if not self.tts_inputs:
            s2t_outputs = self.generate_s2t_outputs(dataset)
            self.mt_inputs = self.generate_mt_inputs(s2t_outputs)
            mt_outputs = self.generate_mt_outputs(self.mt_inputs)
            self.tts_inputs = self.generate_tts_inputs(mt_outputs)

        s2t_metrics = getattr(self.s2t, metric_type)(
            dataset,
            repeat,
        )
        mt_metrics = getattr(self.mt, metric_type)(self.mt_inputs, repeat)
        tts_metrics = getattr(self.tts, metric_type)(
            self.tts_inputs,
            repeat,
        )
        print(
            f"metric_type = {metric_type}  s2t_metrics = {s2t_metrics} \t mt_metrics = {mt_metrics} \t tts_metrics = {tts_metrics}"
        )
        if metric_type == "max_memory":
            return max(s2t_metrics, mt_metrics, tts_metrics)
        else:
            return s2t_metrics + mt_metrics + tts_metrics
