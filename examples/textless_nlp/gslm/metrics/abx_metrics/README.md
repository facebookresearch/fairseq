# ABX-based evaluation

ABX is used to evaluate the quality of the obtained discrete units.

The life cycle of the ABX-based evaluation for the Speech-to-Unit contains the following steps:
1. Training an acoustic model (or use an existing acoustic model) ([description](./../..))
2. Perform quantization of speech by learning a K-means clustering model ([description](./../..))
3. Compute discrete features for ABX computation using the learned clusters
4. Compute the ABX score over the discrete features taking advantage of [libri-light's ABX evaluation script][ll-abx]

Here we assume that you already went throught the first two steps and focus solely on extracting features and computing ABX scores.

## Libri-light setup

Follow [libri-light's instructions][ll-instructions] for installation and [ABX evaluation setup][ll-abx] (including the download of the data items required for ABX computation).

## Computing ABX

### Dumping quantized features

The first step for the ABX computation is to dump the quantized representations corresponding to the test files.

```shell
TYPE="hubert"
LAYER=6
CKPT_PATH="<PATH_TO_HUBERT_MODEL_CHECKPOINT_FILE>"
KM_MODEL_PATH="<PATH_TO_PRETRAINED_KM_MODEL_FILE>"

SUBSET="dev-clean"
MANIFEST="<PATH_TO_MANIFEST_FOR_LS_DEV-CLEAN>"
DATA_DIR="<PATH_TO_DIR_TO_STORE_FEATURES>/$SUBSET"

PYTHONPATH=. python examples/textless_nlp/gslm/metrics/abx_metrics/dump_abx_feats.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --checkpoint_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_dir_path $DATA_DIR \
    --extension ".flac"
```

Again the manifest file follows the same structure than elsewhere in the codebase.

### Compute ABX with Libri-light

Use libri-light's `eval_ABX.py` script (within the appropriate environment set up) as followed:

```shell
LIBRILIGHT_ROOT="<PATH_TO_LIBRILIGHT>"

SUBSET="dev-clean"
DATA_DIR="<PATH_TO_DIR_TO_STORE_FEATURES>/$SUBSET"
ITEM_FILE_PATH="$LIBRILIGHT_ROOT/eval/ABX_data/$SUBSET.item"
OUT_DIR="<PATH_TO_DIR_TO_STORE_ABX_SCORES>/$SUBSET"

FILE_EXTENSION=".npy"
FEATURE_SIZE=0.02 # depends on the model used

PYTHONPATH=$LIBRILIGHT_ROOT \
  python $LIBRILIGHT_ROOT/eval/eval_ABX.py \
    $DATA_DIR \
    $ITEM_FILE_PATH \
    --file_extension $FILE_EXTENSION \
    --feature_size $FEATURE_SIZE \
    --out $OUT_DIR \
    --mode "all"
```

Note that `FEATURE_SIZE` will depend on the model type you are using to extract the acoustic features:
* For HuBERT and Wav2Vec2.0, use `FEATURE_SIZE=0.02`
* For CPC and Log Mel, use `FEATURE_SIZE=0.01`

If you have a gpu available, make sure you add the `--cuda` flag for faster computation.

[ll-instructions]: https://github.com/facebookresearch/libri-light
[ll-abx]: https://github.com/facebookresearch/libri-light/tree/master/eval#abx
