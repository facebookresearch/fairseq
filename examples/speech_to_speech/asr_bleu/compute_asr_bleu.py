import os
from typing import Dict, List
import sacrebleu
import pandas as pd
from glob import glob
from pathlib import Path
from utils import retrieve_asr_config, ASRGenerator
from tqdm import tqdm
from argparse import ArgumentParser


def merge_tailo_init_final(text):
    """
    Hokkien ASR hypothesis post-processing.
    """
    sps = text.strip().split()
    results = []
    last_syllable = ""
    for sp in sps:
        if sp == "NULLINIT" or sp == "nullinit":
            continue
        last_syllable += sp
        if sp[-1].isnumeric():
            results.append(last_syllable)
            last_syllable = ""
    if last_syllable != "":
        results.append(last_syllable)
    return " ".join(results)


def remove_tone(text):
    """
    Used for tone-less evaluation of Hokkien
    """
    return " ".join([t[:-1] for t in text.split()])


def extract_audio_for_eval(audio_dirpath: str, audio_format: str):
    if audio_format == "n_pred.wav":
        """
        The assumption here is that 0_pred.wav corresponds to the reference at line position 0 from the reference manifest
        """
        audio_list = []
        audio_fp_list = glob((Path(audio_dirpath) / "*_pred.wav").as_posix())
        audio_fp_list = sorted(
            audio_fp_list, key=lambda x: int(os.path.basename(x).split("_")[0])
        )
        for i in range(len(audio_fp_list)):
            try:
                audio_fp = (Path(audio_dirpath) / f"{i}_pred.wav").as_posix()
                assert (
                    audio_fp in audio_fp_list
                ), f"{Path(audio_fp).name} does not exist in {audio_dirpath}"
            except AssertionError:
                # check the audio with random speaker
                audio_fp = Path(audio_dirpath) / f"{i}_spk*_pred.wav"
                audio_fp = glob(
                    audio_fp.as_posix()
                )  # resolve audio filepath with random speaker
                assert len(audio_fp) == 1
                audio_fp = audio_fp[0]

            audio_list.append(audio_fp)
    else:
        raise NotImplementedError

    return audio_list


def extract_text_for_eval(
    references_filepath: str, reference_format: str, reference_tsv_column: str = None
):
    if reference_format == "txt":
        reference_sentences = open(references_filepath, "r").readlines()
        reference_sentences = [l.strip() for l in reference_sentences]
    elif reference_format == "tsv":
        tsv_df = pd.read_csv(references_filepath, sep="\t", quoting=3)
        reference_sentences = tsv_df[reference_tsv_column].to_list()
        reference_sentences = [l.strip() for l in reference_sentences]
    else:
        raise NotImplementedError

    return reference_sentences


def compose_eval_data(
    audio_dirpath: str,
    audio_format: str,
    references_filepath: str,
    reference_format: str,
    reference_tsv_column: str = None,
    save_manifest_filepath=None,
):
    """
    Speech matrix decoding pipeline produces audio with the following mask "N_pred.wav" where N is the order of the corresponding input sample
    """

    reference_sentences = extract_text_for_eval(
        references_filepath, reference_format, reference_tsv_column
    )
    predicted_audio_fp_list = extract_audio_for_eval(audio_dirpath, audio_format)
    assert len(predicted_audio_fp_list) == len(reference_sentences)

    audio_text_pairs = [
        (audio, reference)
        for audio, reference in zip(predicted_audio_fp_list, reference_sentences)
    ]

    tsv_manifest = pd.DataFrame(audio_text_pairs, columns=["prediction", "reference"])

    if save_manifest_filepath is not None:
        tsv_manifest.to_csv(save_manifest_filepath, sep="\t", quoting=3)

    return tsv_manifest


def load_eval_data_from_tsv(eval_data_filepath: str):
    """
    We may load the result of `compose_eval_data` directly if needed
    """
    eval_df = pd.from_csv(eval_data_filepath, sep="\t")

    return eval_df


def run_asr_bleu(args):

    asr_config = retrieve_asr_config(
        args.lang, args.asr_version, json_path="./asr_model_cfgs.json"
    )
    asr_model = ASRGenerator(asr_config)

    eval_manifest = compose_eval_data(
        audio_dirpath=args.audio_dirpath,
        audio_format=args.audio_format,
        references_filepath=args.reference_path,
        reference_format=args.reference_format,
        reference_tsv_column=args.reference_tsv_column,
        save_manifest_filepath=None,
    )

    prediction_transcripts = []
    for _, eval_pair in tqdm(
        eval_manifest.iterrows(),
        desc="Transcribing predictions",
        total=len(eval_manifest),
    ):
        transcription = asr_model.transcribe_audiofile(eval_pair.prediction)
        prediction_transcripts.append(transcription.lower())

    if args.lang == "hok":
        prediction_transcripts = [
            merge_tailo_init_final(text) for text in prediction_transcripts
        ]

    references = eval_manifest["reference"].tolist()
    bleu_score = sacrebleu.corpus_bleu(prediction_transcripts, [references])

    print(bleu_score)

    return prediction_transcripts, bleu_score


def main():
    parser = ArgumentParser(
        description="This script computes the ASR-BLEU metric between model's generated audio and the text reference sequences."
    )

    parser.add_argument(
        "--lang",
        help="The target language used to initialize ASR model, see asr_model_cfgs.json for available languages",
        type=str,
    )
    parser.add_argument(
        "--asr_version",
        type=str,
        default="oct22",
        help="For future support we add and extra layer of asr versions. The current most recent version is oct22 meaning October 2022",
    )
    parser.add_argument(
        "--audio_dirpath",
        type=str,
        help="Path to the directory containing the audio predictions from the translation model",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        help="Path to the file containing reference translations in the form of normalized text (to be compared to ASR predictions",
    )
    parser.add_argument(
        "--reference_format",
        choices=["txt", "tsv"],
        help="Format of reference file. Txt means plain text format where each line represents single reference sequence",
    )
    parser.add_argument(
        "--reference_tsv_column",
        default=None,
        type=str,
        help="If format is tsv, then specify the column name which contains reference sequence",
    )
    parser.add_argument(
        "--audio_format",
        default="n_pred.wav",
        choices=["n_pred.wav"],
        help="Audio format n_pred.wav corresponds to names like 94_pred.wav or 94_spk7_pred.wav where spk7 is the speaker id",
    )
    parser.add_argument(
        "--results_dirpath",
        default=None,
        type=str,
        help="If specified, the resulting BLEU score will be written to this file path as txt file",
    )
    parser.add_argument(
        "--transcripts_path",
        default=None,
        type=str,
        help="If specified, the predicted transcripts will be written to this path as a txt file.",
    )

    args = parser.parse_args()

    prediction_transcripts, bleu_score = run_asr_bleu(args)
    result_filename = f"{args.reference_format}_{args.lang}_bleu.txt"
    if args.results_dirpath is not None:
        if not Path(args.results_dirpath).exists():
            Path(args.results_dirpath).mkdir(parents=True)
        with open(Path(args.results_dirpath) / result_filename, "w") as f:
            f.write(bleu_score.format(width=2))

    if args.transcripts_path is not None:
        with open(args.transcripts_path, "w") as f:
            for transcript in prediction_transcripts:
                f.write(transcript + "\n")


if __name__ == "__main__":
    main()


"""
Example to load Sl audio and references, compute BLEU:

export lang=fi; split=vp && python compute_asr_bleu.py --lang $lang --audio_dirpath /checkpoint/hygong/S2S/speech_matrix_release_ckpts/generated_waveform_release/en-$lang/test_$split/checkpoint.pt --audio_format n_pred.wav --reference_path /large_experiments/ust/hygong/S2S/SpeechEncoder/manifests/vp-vp/en-$lang/test_$split.$lang --reference_format txt --results_dirpath ./
"""
