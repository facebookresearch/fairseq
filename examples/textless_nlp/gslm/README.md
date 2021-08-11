# Generative Spoken Language Modeling

## Speech to Unit Model (S2U)
### Acoustic Model
For quantizing speech we learn a K-means clustering over acoustic representations for which we either use Log-Mel Filterbank or pretrained acoustic representation models. For using pretrained models, please download from their respective locations linked below.
* [HuBERT-Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)
* [Wav2Vec 2.0-Base](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt)
* [CPC](https://dl.fbaipublicfiles.com/textless_nlp/gslm/cpc_big_ll6kh_top_ctc.pt)

### Quantization
For quantizing speech with a given acoustic representation, please follow the steps below.
1. Learn K-means clustering model
```
N_CLUSTERS=<num_cluster>
TYPE=<logmel/hubert/w2v2/cpc>
CKPT_PATH=<path_of_pretrained_acoustic_model>
LAYER=<layer_of_acoustic_model_to_extract_features_from>
MANIFEST=<path_manifest_of_input_audio_files_to_train_with>
KM_MODEL_PATH=<path_of_trained_kmeans_model>

PYTHONPATH=. python examples/textless_nlp/gslm/u2s/clustering/cluster_kmeans.py \
    --num_clusters $N_CLUSTERS \
    --feature_type $TYPE \
    --checkpoint_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_kmeans_model_path $KM_MODEL_PATH
```
2. Quantize using the learned clusters
```
MANIFEST=<path_manifest_of_input_audio_files_to_quantize>
OUT_QUANT_FILE=<path_quzntized_audio_file>

python examples/textless_nlp/gslm/u2s/clustering/del/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --checkpoint_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANT_FILE \
    --extension .flac
```

## Unit Language Model (ULM)
Unit Language Model is a generative LM trained on quantized speech. We use it to generate novel quantized spoken language with and without prompt.

## Unit to Speech Model (U2S)
Unit to speech model is modified Tacotron2 model that learns to syntehsize speech from discrete speech units. We use to synthesize quantized spoken language.

