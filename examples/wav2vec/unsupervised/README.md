# wav2vec Unsupervised  (wav2vec-U)
  
Wav2vec Unsupervised (wav2vec-U) is a framework for building speech recognition systems without any labeled training data as described in [Unsupervised Speech Recognition (Baevski et al., 2021)](https://ai.facebook.com/research/publications/unsupervised-speech-recognition).  The model takes as input wav2vec 2.0 or XLSR representations (see [pretrained models](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec)) as well as unlabeled speech and text data.  
  
  The wav2vec-U training procedure consists of three consecutive main steps:
* Preparation of speech representations and text data
* Generative adversarial training (GAN)
* Iterative self-training + Kaldi LM-decoding

## Preparation of speech and text data
Similar to [wav2vec 2.0](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md),  data folders contain {train,valid,test}.{tsv,wrd,phn} files, where audio paths are stored in tsv files, and word, letter or phoneme transcriptions are stored in .{wrd,ltr,phn}.

In **/path/to/data/with_silence** you need a *train.tsv* file as well as (optionally) *{valid,test}.{tsv,wrd,phn}*. It is nice to have *10h.{tsv,phn}* files there too for reproducing the ablation study on  layer selection. In **/path/to/data/without_silence** you have the same files, except *.tsv* files contain audios with silences removed using rVAD.

Here is how you can create new audio files without silences from a list of input audio files:
```shell
python scripts/vads.py < /path/to/train.tsv > train.vads

python scripts/remove_silence.py --tsv /path/to/train.tsv --vads train.vads --out /dir/to/save/audio/files

python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py /dir/to/save/audio/files --ext wav --dest /path/to/new/train.tsv --valid-percent 0
```

You will need to add the path to rVAD directory to vads.py.

Next, we need to preprocess the audio data to better match phonemized text data:

```shell
zsh scripts/prepare_audio.sh /dir/with/{train,test,valid}.tsv /output/dir /path/to/wav2vec2/model.pt
```
Note that if you have splits different than train/valid/test, you will need to modify this script.

Now we need to prepare text data:
```shell
zsh scripts/prepare_text.sh language /path/to/text/file /output/dir
```

Note that if you want to use a different phonemizer, such as G2P, you will need to modify this script.


## Generative adversarial training (GAN)

We then use a GAN model to build a first unsupervised ASR model. The data preparation above of both speech features and text data is a necessary procedure that enables the generator to match speech to text in an unsupervised way. 

Launching GAN training on top of preprocessed features, with default hyperparameters can be done with:

```
PREFIX=w2v_unsup_gan_xp
TASK_DATA=/path/to/features/unfiltered/precompute_unfiltered_pca512_cls128_mean_pooled  
TEXT_DATA=/path/to/data  # path to fairseq-preprocessed GAN data
KENLM_PATH=/path/to/data/kenlm.phn.o4.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

PREFIX=$PREFIX fairseq-hydra-train \
	-m --config-dir configs/gan \
	--config-name w2vu \
	task.data=${TASK_DATA} \
	task.text_data=${TEXT_DATA} \
	task.kenlm_path=${KENLM_PATH} \
	'common.seed=range(0,5)' &
```

Once we find the best checkpoint (chosen using unsupervised metric that combined language model perplexity and vocabulary usage), we can use it to generate phone labels (or word labels with an appropriate kaldi WFST):

```shell
python w2vu_generate.py --config-dir config/generate --config-name viterbi \
fairseq.task.data=/path/to/dir/with/tsvs fairseq.common_eval.path=/path/to/gan/checkpoint \ 
fairseq.dataset.gen_subset=valid results_path=/where/to/save/transcriptions
```
## Iterative self-training + Kaldi LM-decoding
After the GAN training provides a first unsupervised model, we can then progressively refine the quality of transcriptions using several iterations of semi-supervised learning. We perform two iterations: first, pseudo-label the training data with the unsupervised GAN model and train an HMM on the pseudo-labels. Second, we relabel the training data with the HMM and then fine-tune the original wav2vec 2.0 model using the HMM pseudo-labels with a CTC loss. Note that HMM models use phonemes as output, while wav2vec 2.0 use letter. Both are decoded using WFST decoders into words.


Please see [this README](kaldi_self_train/README.md) for more instructions on how to do iterative self-training + Kaldi LM-decoding.

*** Note: these instructions are a work in progress and will be updated over the next few days
