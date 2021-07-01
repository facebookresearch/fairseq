# wav2vec Unsupervised  (wav2vec-U)
  
Wav2vec Unsupervised (wav2vec-U) is a framework for building speech recognition systems without any labeled training data as described in [Unsupervised Speech Recognition (Baevski et al., 2021)](https://ai.facebook.com/research/publications/unsupervised-speech-recognition).  The model takes as input wav2vec 2.0 or XLSR representations (see [pretrained models](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec)) as well as unlabeled speech and text data.  
  
  The wav2vec-U training procedure consists of three consecutive main steps:
* Preparation of speech representations and text data
* Generative adversarial training (GAN)
* Iterative self-training + Kaldi LM-decoding

## Preparation of speech and text data
Similar to [wav2vec 2.0](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md),  data folders contain {train,valid,test}.{tsv,wrd,phn} files, where audio paths are stored in tsv files, and word, letter or phoneme transcriptions are stored in .{wrd,ltr,phn}.

In **/path/to/data/with_silence** you need a *train.tsv* file as well as (optionally) *{valid,test}.{tsv,wrd,phn}*. It is nice to have *10h.{tsv,phn}* files there too for reproducing the ablation study on  layer selection. In **/path/to/data/without_silence** you have the same files, except *.tsv* files contain audios with silences removed using rVAD.

Pre-requisites:
* set FAIRSEQ_ROOT environmental variable to your fairseq installation
* set RVAD_ROOT environmental variable to a checkout of [rVADfast](https://github.com/zhenghuatan/rVADfast)
* set KENLM_ROOT environmental variable to the location of [KenLM](https://github.com/kpu/kenlm) binaries
* install [PyKaldi](https://github.com/pykaldi/pykaldi) and set KALDI_ROOT environmental variable to the location of your kaldi installation. To use the version bundled with PyKaldi, you can use /path/to/pykaldi/tools/kaldi

Create new audio files without silences:
```shell
# create a manifest file for the set original of audio files
python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /dir/to/save/audio/files --ext wav --dest /path/to/new/train.tsv --valid-percent 0

python scripts/vads.py -r $RVAD_ROOT < /path/to/train.tsv > train.vads

python scripts/remove_silence.py --tsv /path/to/train.tsv --vads train.vads --out /dir/to/save/audio/files

python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /dir/to/save/audio/files --ext wav --dest /path/to/new/train.tsv --valid-percent 0.01
```

Next, we need to preprocess the audio data to better match phonemized text data:

```shell
zsh scripts/prepare_audio.sh /dir/with/{train,test,valid}.tsv /output/dir /path/to/wav2vec2/model.pt 512 14
```
Note that if you have splits different than train/valid/test, you will need to modify this script. The last two arguments are the PCA dimensionality and the 0-based index of the layer from which to extract representations.

Now we need to prepare text data:
```shell
zsh scripts/prepare_text.sh language /path/to/text/file /output/dir 1000 espeak /path/to/fasttext/lid/model
```

The fourth argument is minimum number observations of phones to keep. If your text corpus is small, you might want to reduce this number.

The fifth argument is which phonemizer to use. Supported values are [espeak](http://espeak.sourceforge.net/), [espeak-ng](https://github.com/espeak-ng/espeak-ng), and [G2P](https://github.com/Kyubyong/g2p) (english only).

Pre-trained fasttext LID models can be downloaded [here](https://fasttext.cc/docs/en/language-identification.html).

### Prepare TIMIT data
TIMIT transcripts include silence. Therefore VAD is not used for audio preprocessing, and we do not wrap transcripts with silences or insert random silence in between words.

To prepare TIMIT data for both the matched an unmatched setup:
```shell
bash scripts/prepare_timit.sh /dir/to/timit/raw/data /output/dir /path/to/wav2vec2/model.pt
```

Note that we assume the TIMIT distribution with capitalized directories and filenames are used (e.g., `TRAIN/DR1/FCJF0/SA1.PHN`).

## Generative adversarial training (GAN)

We then use a GAN model to build a first unsupervised ASR model. The data preparation above of both speech features and text data is a necessary procedure that enables the generator to match speech to text in an unsupervised way. 

Launching GAN training on top of preprocessed features, with default hyperparameters can be done with:

```
PREFIX=w2v_unsup_gan_xp
TASK_DATA=/path/to/features/precompute_unfiltered_pca512_cls128_mean_pooled  
TEXT_DATA=/path/to/data/phones  # path to fairseq-preprocessed GAN data (phones dir)
KENLM_PATH=/path/to/data/phones/kenlm.phn.o4.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
    -m --config-dir config/gan \
    --config-name w2vu \
    task.data=${TASK_DATA} \
    task.text_data=${TEXT_DATA} \
    task.kenlm_path=${KENLM_PATH} \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    model.code_penalty=2,4 model.gradient_penalty=1.5,2.0 \
    model.smoothness_weight=0.5,0.75,1.0 'common.seed=range(0,5)'
```


Once we find the best checkpoint (chosen using unsupervised metric that combined language model perplexity and vocabulary usage), we can use it to generate phone labels (or word labels with an appropriate kaldi WFST):

```shell
python w2vu_generate.py --config-dir config/generate --config-name viterbi \
fairseq.common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
fairseq.task.data=/path/to/dir/with/features \
fairseq.common_eval.path=/path/to/gan/checkpoint \ 
fairseq.dataset.gen_subset=valid results_path=/where/to/save/transcriptions
```

The decoding without LM works best on the same adjacent-mean-pooled features that the gan was trained on, while decoding with LM works better on features before the adjacent timestep mean-pooling step (without the "_pooled" suffix).

## Iterative self-training + Kaldi LM-decoding
After the GAN training provides a first unsupervised model, we can then progressively refine the quality of transcriptions using several iterations of semi-supervised learning. We perform two iterations: first, pseudo-label the training data with the unsupervised GAN model and train an HMM on the pseudo-labels. Second, we relabel the training data with the HMM and then fine-tune the original wav2vec 2.0 model using the HMM pseudo-labels with a CTC loss. Note that HMM models use phonemes as output, while wav2vec 2.0 use letter. Both are decoded using WFST decoders into words.


Please see [this README](kaldi_self_train/README.md) for more instructions on how to do iterative self-training + Kaldi LM-decoding.

*** Note: these instructions are a work in progress and will be updated over the next few days
