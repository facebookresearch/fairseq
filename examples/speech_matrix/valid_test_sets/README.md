## SpeechMatrix: Prepare valid and test sets from [VoxPopuli](https://github.com/facebookresearch/voxpopuli), [EuroParl-ST](https://www.mllp.upv.es/europarl-st/) and [FLEURS](https://huggingface.co/datasets/google/fleurs)

Valid sets are prepared for speech-to-speech model training, and test set are for model evaluation.


### VoxPopuli Valid and Test Set

Download the prepared audios for VoxPopuli valid and test data.

```bash
# SAVE_ROOT: the directory to save mined data
python3 download_vp_valid_test.py \
    --save-root {SAVE_ROOT}
```


### EuroParl-ST Testset

1. Download and unzip the [EuroParl v1.1 dataset](https://www.mllp.upv.es/europarl-st/) to EPST_DIR

2. Data processing

```bash
# EPST_DIR: the directory of original EPST data.
# PROC_EPST_DIR: the directory of EPST processed data.
# SAVE_ROOT: the directory to save SpeechMatrix mined data
python3 prep_epst_test_data.py \
    --epst-dir ${EPST_DIR} \
    --proc-epst-dir ${PROC_EPST_DIR} \
    --save-root ${SAVE_ROOT}
```


### FLEURS Testset

1. Download [FloRes](https://github.com/facebookresearch/flores) to `FLORES_ROOT`.

2. Preprocessing

```bash
# PROC_FLEURS_DIR: the directory of FLEURS processed data 
python3 preproc_fleurs_data.py \
    --proc-fleurs-dir ${PROC_FLEURS_DIR}
```


3. Aligning speech in FLEURS with texts in FLORES

```bash
# FLORES_ROOT: the directory where FLoRes data is saved
# PROC_FLEURS_DIR: the directory of FLEURS processed data 
# SAVE_ROOT: the directory to save mined data
python3 align_fleurs_data.py \
    --flores-root ${FLORES_ROOT} \
    --proc-fleurs-dir ${PROC_FLEURS_DIR} \
    --save-root ${MANIFEST_ROOT} \
```


4. Prepare test set

```bash
# PROC_FLEURS_DIR: the directory of FLEURS processed data. 
# SAVE_ROOT: the directory to save mined data
python3 prep_fleurs_test_data.py  \
    --proc-fleurs-dir ${PROC_FLEURS_DIR} \
    --save-root ${SAVE_ROOT}
```


## FLEURS Valid Set

# FLEURS
1. Download HuBERT models and kmeans models to HUBERT_MODEL_DIR for target unit preparation.

## Pre-trained Models

### HuBERT & Kmeans
Model | Pretraining Data | Model | Kmeans/Quantizer
|---|---|---|---
mHuBERT Base | [VoxPopuli](https://github.com/facebookresearch/voxpopuli) Roman data (ca, es, fr, it, pt, ro) | [download](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_roman_it3.pt) | [it](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_it_it3_L11_km800.bin), [pt](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_pt_it3_L11_km1000.bin), [ro](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_ro_it3_L11_km1000.bin)
mHuBERT Base | [VoxPopuli](https://github.com/facebookresearch/voxpopuli) Slavic data (cs, hr, lv, lt, pl, sk, sl,) | [download](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_slavic_it3.pt) | [cs](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_cs_it3_L11_km1000.bin), [hr](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_hr_it3_L11_km1000.bin), [lt](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_lt_it3_L11_km1000.bin), [pl](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_pl_it3_L11_km1000.bin), [sk](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_sk_it3_L11_km1000.bin), [sl](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_sl_it3_L11_km1000.bin)
mHuBERT Base | [VoxPopuli](https://github.com/facebookresearch/voxpopuli) Germanic data (en, de, nl, sv) | [download](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_germanic_it3.pt) | [de](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_de_it3_L11_km1000.bin), [nl](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_nl_it3_L11_km1000.bin)
mHuBERT Base | [VoxPopuli](https://github.com/facebookresearch/voxpopuli) Uralic data (et, fi, hu) | [download](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_uralic_it3.pt) | [et](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_et_it3_L11_km1000.bin), [fi](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_fi_it3_L11_km1000.bin), [hu](https://dl.fbaipublicfiles.com/speech_matrix/hubert/mhubert_base_vp_hu_it3_L11_km1000.bin)


2. Prepare valid set

```bash
# PROC_FLEURS_DIR: the directory of FLEURS processed data
# SAVE_ROOT: the directory to save mined data
# HUBERT_MODEL_DIR: the diretory where HuBERT and kmeans models are downloaded
python3 prep_fleurs_valid_data.py \
    --proc-fleurs-dir ${PROC_FLEURS_DIR} \
    --save-root ${SAVE_ROOT} \
    --hubert-model-dir ${HUBERT_MODEL_DIR}
```


