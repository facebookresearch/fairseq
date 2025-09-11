# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import torch
import torchaudio

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar

from examples.speech_synthesis.generate_waveform import (
    make_parser,
    dump_result,
)

from pathlib import Path
from collections import defaultdict
import json
import time
from examples.speech_synthesis.incremental_text_to_speech.eval_latency import calc_avg_stats
from examples.speech_synthesis.incremental_text_to_speech.phonemize import Phonemizer


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_parser_incremental():
    parser = make_parser()
    # Options for incremental TTS
    parser.add_argument("--incremental-tts-input-text", required=True, type=str)
    parser.add_argument("--incremental-tts-input-from-simuleval", action="store_true")
    parser.add_argument("--lookahead-words", default=0, type=int,
                        help="Number of lookahead words. "
                             "When lookahead=0, directly output synthesized chunks w/o waiting.")
    parser.add_argument("--phonemizer-lang", default="en", type=str,
                        help="Language of the phonemizer")
    parser.add_argument("--append-eos-to-partial-input", action="store_true",
                        help="If true, append EOS token to the partial input.")
    parser.add_argument("--use-pseudo-lookahead", action="store_true",
                        help="Use pseudo lookahead from ST system")
    parser.add_argument("--generate-audio-with-discontinuity", action="store_true",
                        help="If true, include silence (due to no output playing) in the synthesized utterances")
    return parser


def main(args):
    assert (args.dump_features or args.dump_waveforms or args.dump_attentions
            or args.dump_eos_probs or args.dump_plots)

    if args.generate_audio_with_discontinuity:
        assert args.incremental_tts_input_from_simuleval, \
            "Output with discontinuity is only possible when input is timestamped (by simuleval log file)."

    if args.use_pseudo_lookahead:
        assert args.lookahead_words == 0, "Pseudo lookahead only works when not using actual lookahead."

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 8000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    task = tasks.setup_task(args)
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        task=task,
    )
    model = models[0].cuda() if use_cuda else models[0]
    task.args.n_frames_per_step = saved_cfg.task.n_frames_per_step
    task.load_dataset(args.gen_subset, task_cfg=saved_cfg.task)

    data_cfg = task.data_cfg
    sample_rate = data_cfg.config.get("features", {}).get("sample_rate", 22050)
    resample_fn = {
        False: lambda x: x,
        True: lambda x: torchaudio.sox_effects.apply_effects_tensor(
            x.detach().cpu().unsqueeze(0), sample_rate,
            [['rate', str(args.output_sample_rate)]]
        )[0].squeeze(0)
    }.get(args.output_sample_rate != sample_rate)
    if args.output_sample_rate != sample_rate:
        logger.info(f"resampling to {args.output_sample_rate}Hz")

    generator = task.build_generator([model], args)
    Path(args.results_path).mkdir(exist_ok=True, parents=True)
    is_na_model = getattr(model, "NON_AUTOREGRESSIVE", False)
    vocoder = task.args.vocoder

    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    main_results_path = Path(args.results_path)

    if args.incremental_tts_input_from_simuleval:
        logger.info(f"Generating waveforms from SimulEval log file {args.incremental_tts_input_from_simuleval}")
        with open(args.incremental_tts_input_text, "r") as f:
            lines = f.readlines()
    else:
        logger.info(f"Generating waveforms from txt file {args.incremental_tts_input_from_simuleval}")
        lines = []
        with open(args.incremental_tts_input_text, "r") as f:
            for l in f.readlines():
                lines.append(l.strip())
    phonemizer = Phonemizer(args.phonemizer_lang)

    with progress_bar.build_progress_bar(args, enumerate(lines)) as t:
        for s in t:
            sample_idx, entry = s

            args.results_path = main_results_path / str(sample_idx)
            args.results_path.mkdir(parents=True, exist_ok=True)
            latency_log = defaultdict(dict)

            if args.incremental_tts_input_from_simuleval:
                entry = json.loads(entry)
                words = entry["prediction"].split()[:-1]  # exclude </s> from simuleval outputs
                timestamps = entry["elapsed"][:-1]

                if args.use_pseudo_lookahead:
                    assert "pseudo_lookahead" in entry
                    assert args.lookahead_words == 0
                    pseudo_lookahead = entry["pseudo_lookahead"]
                    if args.ignore_single_char_pseudo_lookahead:
                        pseudo_lookahead = [p if len(p) > 1 else "_" for p in pseudo_lookahead]

                assert len(words) == len(timestamps)
            else:
                words = entry.split()

            available_phonemes = []
            prev_end_token = 0

            # Track the number of phonemes from the lookahead words
            lookahead_phoneme_cnt = [0] * args.lookahead_words
            # Loop through words in current sentence
            for w_idx, w in enumerate(words):
                word_generation_start_time = time.time()
                is_last_word = w_idx == len(words) - 1
                # Convert word to phonemes
                current_word_phonemes = phonemizer.convert_word_to_phonemes(w)

                pseudo_lookahead_phonemes = []
                if args.use_pseudo_lookahead and not is_last_word and len(pseudo_lookahead[w_idx]) > 0:
                    pseudo_lookahead_phonemes = phonemizer.convert_word_to_phonemes(pseudo_lookahead[w_idx])

                pseudo_lookahead_cnt = len(pseudo_lookahead_phonemes)
                current_word_phonemes += pseudo_lookahead_phonemes
                available_phonemes.extend(current_word_phonemes)
                # Update the number of phonemes from the lookahead words
                if args.lookahead_words > 0:
                    lookahead_phoneme_cnt[w_idx % args.lookahead_words] = len(current_word_phonemes)
                # Do not synthesize initial lookahead words
                if w_idx < args.lookahead_words and not is_last_word:
                    continue

                src_tokens = [task.src_dict.indices[t] for t in available_phonemes if t in task.src_dict.indices]
                if args.append_eos_to_partial_input or is_last_word:
                    src_tokens.append(task.src_dict.eos())

                sample = dict()
                sample["src_tokens"] = torch.LongTensor(src_tokens).unsqueeze(0).to(device)
                sample["src_lengths"] = torch.LongTensor([len(src_tokens)]).to(device)

                if is_last_word:
                    end_token_idx = len(src_tokens) - 1
                else:
                    end_token_idx = len(src_tokens) - 1 - pseudo_lookahead_cnt - sum(lookahead_phoneme_cnt)

                try:
                    hypo = generator.generate_simple(model, sample,
                                                     start_token_idx=prev_end_token,
                                                     end_token_idx=end_token_idx).squeeze()

                    wave_pred = resample_fn(hypo).detach().cpu().numpy()

                    dump_result(is_na_model,
                                args, vocoder=vocoder, sample_id=f"{sample_idx}_{w_idx}",
                                text=None, attn=None, eos_prob=None,
                                feat_pred=None, wave_pred=wave_pred, feat_targ=None, wave_targ=None,
                                )

                    # Remove pseudo-lookaheads from available phonemes for next iteration
                    if args.use_pseudo_lookahead:
                        available_phonemes = available_phonemes[:len(available_phonemes) - pseudo_lookahead_cnt]

                    if args.lookahead_words > 0:
                        prev_end_token = end_token_idx + 1
                    else:
                        prev_end_token = len(available_phonemes)

                    latency_log[w_idx]["computation_duration"] = time.time() - word_generation_start_time
                    # Min accounts for last word
                    emit_after_word = min(w_idx + args.lookahead_words, len(words) - 1)
                    latency_log[w_idx]["output_play_duration"] = wave_pred.shape[0] / sample_rate
                    if args.incremental_tts_input_from_simuleval:
                        # convert ms to s
                        latency_log[w_idx]["emit_after_input_word_timestamp"] = timestamps[emit_after_word] / 1_000
                    else:
                        latency_log[w_idx]["emit_after_input_word_timestamp"] = 0
                    dump_latency_log(args, sample_idx, latency_log)

                except RuntimeError as e:
                    logger.warning(f"Failed to synthesize for word {w_idx} of sample {sample_idx}: {e}.")
                    # Remove pseudo-lookaheads from available phonemes for next iteration
                    if args.use_pseudo_lookahead:
                        available_phonemes = available_phonemes[:len(available_phonemes) - pseudo_lookahead_cnt]

    logger.info("Finished generating incremental utterances!")

    logger.info("Joining partial utterances and summarizing latency...")
    summarize_latency(args,
                      f"{main_results_path}/*/{args.audio_format}_{sample_rate}hz_{vocoder}/*_latency.log",
                      f"{main_results_path}/{args.audio_format}_{sample_rate}hz_{vocoder}_latency.txt",
                      sample_rate
                      )
    logger.info(f"Wrote latency result to: "
                f"{main_results_path}/{args.audio_format}_{sample_rate}hz_{vocoder}_latency.txt")


def dump_latency_log(args, sample_id, latency_log):
    ext = args.audio_format
    sample_rate = args.output_sample_rate
    vocoder = args.vocoder
    with open(f"{args.results_path}/{ext}_{sample_rate}hz_{vocoder}/{sample_id}_latency.log", "w") as f:
        f.write(json.dumps(latency_log))


def summarize_latency(args, log_paths, out_file, sample_rate):
    avg_latency, avg_compute, avg_duration, avg_discontinuity = calc_avg_stats(log_paths,
                                                                               args.generate_audio_with_discontinuity,
                                                                               sample_rate)

    with open(out_file, "w") as f:
        f.write(f"Avg speaking latency (s), i.e. time elapsed between last input word and end of synthesized speech: "
                "{avg_latency:.2f}\n")
        f.write(f"Avg computation time (s): {avg_compute:.2f}\n")
        f.write(f"Avg utterance play time (s): {avg_duration:.2f}\n")
        f.write(f"Avg discontinuity (s): {avg_discontinuity:.2f}\n")


def cli_main():
    parser = make_parser_incremental()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
