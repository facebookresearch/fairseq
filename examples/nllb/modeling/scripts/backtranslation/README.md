# Helper scripts for NLLB Modeling

## `generate_backtranslations.sh`

This script takes a binarized, sharded monolingual corpus and backtranslates it. The script will look for the binarized monolingual corpus under `${MONODIR}/data_bin/shard{000,001,...}/*.{bin,idx}`. We split the monolingual corpus into shards as jobs on slurm may fail or be pre-empted, and this allows us to restart individual shards when it they fail rather than having to redo the whole corpus.

The script will take care of re-running any shards that are incomplete – all you have to do is run it again with the same arguments. To detect whether a shard was not fully backtranslated, it will compare the number of lines in the output with the number of lines in the input (using the utility script `count_idx.py`). If the translated lines are >95% of the input lines, the shard is considered complete. Otherwise, the incomplete outputs will be deleted and the shard will be re-run. *NB*: Make sure all backtranslation jobs are complete before re-running the script with the same arguments – jobs that are still running will otherwise be mistaken by the script as having failed.

Before running the script please make sure you populate these variables in the script:
* `$MODEL`: The path to the model checkpoint file.
* `$MODEL_LANGS`: The list of languages supported by the model, in the same order they were passed during training on fairseq.
* `$MONODIR`: The path to the monolingual data directory.
* `$OUTDIR`: The path to the output directory where you want to store the generated backtranslations.


Example command:
```
examples/nllb/modeling/scripts/generate_backtranslations.sh x_en awa_Deva,fra_Latn,rus_Cyrl
```

Parameters:
* The first argument is whether the backtranslation generation is from or out of english, `en_x` or `x_en` respectively.
* The second argument is the list of languages to generate backtranslations for, a comma separated string of lang names. Eg: `awa_Deva,fra_Latn,rus_Cyrl`

## `extract_fairseq_bt.py`

This command takes BT shard outputs and extracts a parallel corpus from them.
It optionally can perform some very basic filtering of the sentences, but generally a
more complete filtering pipeline should be used, e.g. the one included in the
[STOPES](https://github.com/facebookresearch/stopes) library.

Example command to run locally:
```
python extract_fairseq_bt.py --local-run --directions eng_Latn-fuv_Latn \
  --strip-orig-tag --spm-decode $SPM_MODEL_FOLDER/sentencepiece.model \
  --corpus-name test_corpus \
  $BASE_FOLDER
```
Parameters:
* `$BASE_FOLDER` is where the script looks for BT shard outputs, specifically under `$BASE_FOLDER/$direction/*_{0..999}.out`
* `--output-folder` is where the script will place filtered outputs, specifically: `$output_folder/$direction/$corpus_name.$lang.gz`.
* `--spm-decode` tells the model to perform SPM decoding using the provided model on both the original and backtranslated text.
* `--directions` is a space-separated list of language pairs. They are interpreted as `$bt_lang-$original_lang` (`$original_lang` corresponds to `S-*` lines in fairseq's output, and `$bt_lang` to `H-*` lines).
* To run on SLURM, remove the `--local-run` flag. You may additionally want to specify `--slurm-partition` and `--slurm-timeout`.
* See `--help` for further information.


## `extract_moses_bt.py`

This command is analogous to `extract_fairseq_bt.py` above, but for data translated with
MOSES. It can additionally perform some very basic MOSES-specific filtering (based on the ratio of copied tokens).

Example command to run locally:
```
python extract_moses_bt.py  --local-run --directions eng_Latn-fuv_Latn \
  --detruecase $MOSES_FOLDER/scripts/recaser/detruecase.perl \
  --spm-decode $SPM_MODEL_FOLDER/sentencepiece.model \
  --corpus-name test_corpus \
  $BASE_FOLDER
```

Parameters:
* `$BASE_FOLDER` is the location of input and output shards. The script expects them in `$BASE_FOLDER/{src_sharded,tgt_sharded}/$direction/$lang.{000..999}`.
* Filtered outputs will be placed in `$output_folder/$direction/$corpus_name.$lang.gz`.
* `--detrucase` tells the script to perform detruecasing on both the original and backtranslated shards. You will need to pass the location of the MOSES `detruecase.perl` script.
* `--directions` is interpreted as in `extract_fairseq_bt.py`.
* To run on SLURM, remove the `--local-run` flag. You may additionally want to specify `--slurm-partition` and `--slurm-timeout`.
* See `--help` for further information.

## `count_idx.py`

This is a utility script to count the number of sentences in a fairseq binarized dataset (`.idx` file).
It's used by `generate_backtranslations.sh` to figure out which shards of a corpus have been fully backtranslated.
