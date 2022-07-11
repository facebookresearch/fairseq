#!/usr/bin/env python

# Script for BART-large speech fine-tuning
# on translation task using units as input and
# a different target dictionary

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam

def get_grid(args):
    grid = []

    ckpt_root = None
    restore_file = None

    # restore_file = '/checkpoint/padentomasello/projects/nlu/stop/parse/stop_qa/stop_qa.stop_hubert_asr_pretrained_checkpoint.best_wer_11.9370.spt-hubert_seq2seq.mt1600000.adam.beta9999.eps1e-08.lr0.0001.ngpu8/checkpoint_best.pt'
    # restore_file = '/checkpoint/padentomasello/projects/nlu/stop_tts/parse/stop_qa/stop_qa.spt-hubert_seq2seq.mt1600000.adam.beta9999.eps1e-08.lr0.0001.ngpu8/checkpoint_last.pt'

    # Hubert LS
    # pretrained_dataset = 'ls_hubert'
    # ckpt_root = '/checkpoint/padentomasello/projects/nlu/stop//asr/asr.archhubert_seq2seq.mt1600000.trainset-train.adam.beta9999.eps1e-08.lr0.0001.ngpu8/'
    # ckpt_name = 'checkpoint.best_wer_32.2450'

    # pretrained_dataset = 'ls_wav2vec'
    # ckpt_root = '/checkpoint/padentomasello/projects/nlu/stop//asr/asr.archwav2vec_seq2seq.mt1600000.trainset-train.adam.beta9999.eps1e-08.lr0.0001.ngpu8/'
    # ckpt_name = 'checkpoint.best_wer_32.5900'

    # pretrained_dataset = 'stop_hubert'
    # ckpt_root = '/checkpoint/padentomasello/projects/nlu/stop/asr/stop_qa/stop_qa.archhubert_seq2seq.mt1600000.trainsetstop.adam.beta9999.eps1e-08.lr0.0001.ngpu8/'
    # ckpt_name = 'checkpoint.best_wer_11.9370'

    # pretrained_dataset = 'stop_wav2vec'
    # ckpt_root = '/checkpoint/padentomasello/projects/nlu/stop/asr/stop_qa/stop_qa.archwav2vec_seq2seq.mt1600000.trainsetstop.adam.beta9999.eps1e-08.lr0.0001.ngpu8/'
    # ckpt_name = 'checkpoint.best_wer_11.5730'

    # pretrained_dataset = 'stop_heldin_hubert'
    # ckpt_root = '/checkpoint/padentomasello/projects/nlu/stop/asr/stop_qa/stop_qa.archhubert_seq2seq.mt1600000.trainsetstopheldin.adam.beta9999.eps1e-08.lr0.0001.ngpu8/'
    # ckpt_name = 'checkpoint.best_wer_11.1380'

    # pretrained_dataset = 'stop_heldin_wav2vec'
    # ckpt_root = '/checkpoint/padentomasello/projects/nlu/stop/asr/stop_qa/stop_qa.archwav2vec_seq2seq.mt1600000.trainsetstopheldin.adam.beta9999.eps1e-08.lr0.0001.ngpu8/'
    # ckpt_name = 'checkpoint.best_wer_12.1870'

    # OLD #
    # OLD #
    # OLD #
    # pretrained_dataset = 'stop_hubert'
    # ckpt_root = '/checkpoint/padentomasello/projects/nlu/stop/asr/stop/stop.archhubert_seq2seq.mt1600000.trainsetstop.adam.beta9999.eps1e-08.lr0.0001.ngpu8/'
    # ckpt_name = 'checkpoint.best_wer_19.1900'

    # pretrained_dataset = 'stop_wav2vec'
    # ckpt_root = '/checkpoint/padentomasello/projects/nlu/stop/asr/stop/stop.archwav2vec_seq2seq.mt1600000.trainsetstop.adam.beta9999.eps1e-08.lr0.0001.ngpu8/'
    # ckpt_name = 'checkpoint.best_wer_17.9200'

    # pretrained_dataset = 'libritts'
    # ckpt_root = '/checkpoint/padentomasello/projects/nlu/stop/asr-tts/asr-tts.archhubert_seq2seq.mt1600000.trainsetlibritts_xvector_vits.adam.beta9999.eps1e-08.lr0.0001.ngpu8/'
    # ckpt_name = 'checkpoint.best_wer_3.7370'

    # pretrained_dataset = 'vctk_multi_spk_vits'
    # ckpt_root = '/checkpoint/padentomasello/projects/nlu/stop/asr-tts/asr-tts.archhubert_seq2seq.mt1600000.trainsetvctk_multi_spk_vits.adam.beta9999.eps1e-08.lr0.0001.ngpu8/'
    # ckpt_name = 'checkpoint.best_wer_11.1310'

    if restore_file:
        grid += [
            hyperparam(
                "--restore-file",
                restore_file,
            ),
            ]

    if ckpt_root:
        grid += [
            # hyperparam(
                # "--restore-file",
                # f"{ckpt_root}/{ckpt_name}.pt",
                # save_dir_key=lambda x: f'checkpoint_topv2asr{ckpt_name}'
            # ),
            # hyperparam(
                # "--restore-file",
                # f"{ckpt_root}/{ckpt_name}.pt",
                # save_dir_key=lambda x: f'checkpoint_lsasr{ckpt_name}'
            # ),
            # hyperparam("--reset-optimizer"),
            # hyperparam("--reset-lr-scheduler"),
            # hyperparam("--reset-dict"),
            # hyperparam("--seq2seq-path",
                 # f"{ckpt_root}/{ckpt_name}.pt",
                 # save_dir_key= lambda x: f'{pretrained_dataset}_asr_pretrained_{ckpt_name}')
        ]

    grid += [
            hyperparam("--best-checkpoint-metric", "em_error"),
            hyperparam("--keep-best-checkpoints", 1)
    ]

    #task
    grid += [
        hyperparam("--task", "nlu_finetuning"),
        hyperparam('--autoregressive'),
        # hyperparam('--no-normalize'),
        hyperparam('--labels', 'parse'),
        hyperparam('--eval-wer-parse'),
        # hyperparam('--eval-wer-post-process', 'sentencepiece')
    ]

    grid += [
        hyperparam("--ddp-backend", "legacy_ddp"),
        hyperparam("--distributed-world-size", args.num_gpus)
    ]

    grid += [
        hyperparam("--criterion", "label_smoothed_cross_entropy")
    ]




    # model settings
    grid += [
        hyperparam("--w2v-path", "/private/home/padentomasello/models/hubert/hubert_base_ls960.pt"),
        hyperparam("--arch", "hubert_seq2seq", save_dir_key=lambda val: f'spt-{val}'),

        # hyperparam("--arch", "wav2vec_seq2seq", save_dir_key= lambda val: f'spt-{val}'),
        # hyperparam("--w2v-path", "/private/home/padentomasello/models/wav2vec2/wav2vec_small.pt"),

        # hyperparam("--arch", "wav2vec_seq2seq", save_dir_key= lambda val: f'spt-none'),
        # hyperparam("--w2v-path", "/private/home/padentomasello/models/wav2vec2/wav2vec_small.pt"),
        # hyperparam("--no-pretrained-weights"),

        hyperparam("--apply-mask"),
        hyperparam("--mask-prob", 0.5),
        hyperparam("--mask-channel-length", 64),
        hyperparam("--layerdrop", 0.1),
        hyperparam("--activation-dropout", 0.1),
        hyperparam("--feature-grad-mult", 0.0),
        hyperparam("--freeze-finetune-updates", 0)
    ]

    #dataset

    # genders = ['male', 'female']
    # genders = [ 'gender/test_' + x for x in genders ]
    # domains = ['alarm', 'event', 'messaging', 'music', 'navigation', 'reminder', 'timer', 'weather']
    # domains = ['domain_splits/test_' + x for x in domains ]
    # natives = [ 'yes', 'no']
    # natives = ['native/test_' + x for x in natives]
    # flats = ['flat', 'notflat']
    # flats = ['depth/test_' + x for x in flats ]
    # val_splits = genders + domains + natives + flats


    natural_valid = ["eval", "test"]
    # natural_valid = ['full/' + x for x in natural_valid ]
    val_splits = natural_valid

    # domain_test = ["alarm", "event", "messaging", "music", "navigation", "reminder", "timer", "weather"]
    # domain_test = [ "domain_splits/test_" + x for x in domain_test]

    # gender_test = ["male", "female"]
    # gender_test = ["gender/test_" + x for x in gender_test ]

    # native_test = ["yes", "no"]
    # native_test = ["native/test_" + x for x in native_test]

    # val_splits = domain_test + gender_test + native_test

    natural_train = "train"
    tts_train = "libritts_xvector_vits/train"
    # tts_train = "vctk_multi_spk_vits/train"

    # held_in_root = "low_resource_splits/held_in/"
    # held_in_train = held_in_root + "held_in_train"

    # held_in_valid = [ held_in_root + x for x in ["held_in_eval", "held_in_test"]]

    # prod_tts_valid = ["valid", "test"]
    # root = "prod_tts_top/parse_decoupled_seq2seq/"
    # prod_tts_valid = [ root + x for x in prod_tts_valid ]
    # val_splits.extend(prod_tts_valid)
    # print(val_splits)

    # prod_tts_train = root + "train"

    # weather_root = 'low_resource_splits/weather'
    # # spiss = [ 25 ]
    # spis = 25
    # low_resource_splits = f'{weather_root}/weather_train_{spis}spis'
    # val = [ f'{weather_root}/test_weather',f'{weather_root}/weather_valid_{spis}spis' ]

    grid += [
        hyperparam("--num-workers", 6),
        hyperparam("--max-tokens", 1600000, save_dir_key=lambda val: f"mt{val}"),
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        # hyperparam("--valid-subset", "stop_test,stop_eval"),
        # hyperparam("--train-subset", "stop_train"),
        # hyperparam("--valid-subset", ','.join(val_splits)),
        hyperparam("--valid-subset", ','.join(val_splits)),
        hyperparam("--train-subset", natural_train)
        # Prod TTS Topv2
        # hyperparam("--valid-subset", ','.join(val_splits)),
        # hyperparam("--train-subset", natural_train)
        # hyperparam("--train-subset", held_in_train, save_dir_key= lambda x: 'trainsetstopheldin'),
        # hyperparam("--valid-subset", ','.join(held_in_valid)),
        # hyperparam("--train-subset", ["libritts_xvector_vits/train", "vctk_multi_spk_vits/train"], 
            # save_dir_key=lambda x: f'trainset{x.split("/")[0]}')
        # hyperparam("--train-subset", low_resource_splits, 
            # save_dir_key=lambda x: f'trainset{x.split("/")[2]}')
        # hyperparam("--train-subset", "l),
        # hyperparam("--valid-subset", "eval"),
        # hyperparam("--train-subset", "train"),
    ]


    # # optimization settings
    grid += [
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "beta9999"),
        hyperparam("--adam-eps", 1e-08, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--max-update", 320000),
        hyperparam("--lr", 0.0001, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--sentence-avg")
        # hyperparam("--max-epoch", 300)
        # hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"clip{val}"),
    ]

    # # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "tri_stage"),
        hyperparam("--phase-ratio", "[0.1,0.4,0.5]"),
        hyperparam("--final-lr-scale", 0.05)
        # hyperparam("--total-num-update", total_num_udpates),
        # hyperparam(
            # "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"warm{val}"
        # ),
    ]
    # grid += [
        # hyperparam("--lr-scheduler", "tri_stage"),
        # hyperparam("--lr", lr, save_dir_key=lambda val: f"lr{val}"),
        # hyperparam("--hold-steps", 0),
        # hyperparam("--decay-steps", 72000),
        # hyperparam("--final-lr-scale", 0.05),
        # hyperparam(
            # "--warmup-steps", warmup_updates, save_dir_key=lambda val: f"warm{val}"
        # ),
    # ]
    # grid += [
        # hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
    # ]

    grid += [
        # hyperparam("--eval-bleu"),
        # hyperparam("--eval-bleu-args", '{"beam": 4, "max_len_b": 200, "no_repeat_ngram_size": 3}'),
        # hyperparam("--eval-bleu-detok", 'moses'),
        # hyperparam("--eval-bleu-remove-bpe"),
    ]

    # # data loading settings
    # grid += [
        # hyperparam("--num-workers", num_data_loaders),
    # ]

    # # validation and checkpoint settings
    grid += [
        hyperparam("--validate-interval", 10),
        hyperparam("--save-interval", 10),
        # hyperparam("--validate-interval-updates", 1),
        # hyperparam("--save-interval-updates", 10),
        # debug
        # hyperparam("--validate-interval-updates", 1),
        # # hyperparam("--save-interval-updates", 1),
        # hyperparam("--max-valid-steps", 1),
        # ===== 
        # hyperparam("--validate-after-updates", freeze_finetune_num_updates),
        # hyperparam("--no-epoch-checkpoints"),
        # hyperparam("--no-save"),
        # hyperparam("--reset-meters"),
        # hyperparam("--reset-optimizer"),
        # hyperparam("--reset-lr-scheduler"),
    ]

    # grid += [
        # # hyperparam("--share-all-embeddings"),
        # hyperparam("--layernorm-embedding"),
        # hyperparam("--share-decoder-input-output-embed"),
    # ]


    # # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 10),
        hyperparam("--fp16")
    ]

    if args.local:
        grid += [
            hyperparam("--log-format", "json"),
            hyperparam("--log-interval", 10),
        ]
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
