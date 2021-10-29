# Speech to Unit Model (speech2unit)

## Acoustic Model
For quantizing speech we learn a K-means clustering over acoustic representations for which we either use Log-Mel Filterbank or pretrained acoustic representation models. For using pretrained models, please download from their respective locations linked below.
* [Modified CPC](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/cpc_big_ll6kh_top_ctc.pt)
* [HuBERT-Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)
* [Wav2Vec 2.0-Base](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt)

## Quantization Model
You can download pretrained quantized model from the list below.

K-Means Model | Download Link
|-|-
Log Mel Filterbank + KM50 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/km50/km.bin)
Log Mel Filterbank + KM100 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/km100/km.bin)
Log Mel Filterbank + KM200 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/logmel/km200/km.bin)
Modified CPC + KM50 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/km50/km.bin)
Modified CPC + KM100 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/km100/km.bin)
Modified CPC + KM200 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc/km200/km.bin)
HuBERT Base + KM50 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km50/km.bin)
HuBERT Base + KM100 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin)
HuBERT Base + KM200 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km200/km.bin)
wav2vec 2.0 Large + KM50 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/km50/km.bin)
wav2vec 2.0 Large + KM100 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/km100/km.bin)
wav2vec 2.0 Large + KM200 | [download](https://dl.fbaipublicfiles.com/textless_nlp/gslm/w2v2/km200/km.bin)

### Quantization
For quantizing speech with a given acoustic representation, please follow the steps below.
1. Learn K-means clustering model
```
N_CLUSTERS=<number_of_clusters_used_for_kmeans>
TYPE=<one_of_logmel/cpc/hubert/w2v2>
CKPT_PATH=<path_of_pretrained_acoustic_model>
LAYER=<layer_of_acoustic_model_to_extract_features_from>
MANIFEST=<tab_separated_manifest_of_audio_files_for_training_kmeans>
KM_MODEL_PATH=<output_path_of_the_kmeans_model>

PYTHONPATH=. python examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py \
    --num_clusters $N_CLUSTERS \
    --feature_type $TYPE \
    --checkpoint_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_kmeans_model_path $KM_MODEL_PATH
```
2. Quantize using the learned clusters
```
MANIFEST=<tab_separated_manifest_of_audio_files_to_quantize>
OUT_QUANTIZED_FILE=<output_quantized_audio_file_path>

python examples/textless_nlp/gslm/speech2unit/clustering/del/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --checkpoint_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ".flac"
```

Note about the manifest file is a file with paths and length of input audio files. The format of the file is as follows:
```
<path_of_root_directory_containing_audio_files>
<relative_path_of_audio_file_1>\t<number_of_frames_1>
<relative_path_of_audio_file_2>\t<number_of_frames_1>
...
```

