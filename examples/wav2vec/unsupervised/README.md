
# wav2vec Unsupervised  (wav2vec-U)
  
Wav2vec Unsupervised (wav2vec-U) is a framework for building speech recognition systems without any labeled training data as described in [Unsupervised Speech Recognition (Baevski et al., 2021)](https://ai.facebook.com/research/publications/unsupervised-speech-recognition).  The model takes as input wav2vec 2.0 or XLSR representations (see [pretrained models](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec)) as well as unlabeled speech and text data.  
  
  The wav2vec-U training procedure consists of three consecutive main steps:
* Preparation of speech representations and text data
* Generative adversarial training (GAN)
* Iterative self-training + Kaldi LM-decoding


## Preparation of speech and text data
Similar to [wav2vec 2.0](https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md),  data folders contain {train,valid,test}.{tsv,wrd,phn} files, where audio paths are stored in tsv files, and word, letter or phoneme transcriptions are stored in .{wrd,ltr,phn}.


In **/path/to/data/with_silence** you need a *train.tsv* file as well as *{valid,test}.{tsv,wrd,phn}*. It is nice to have *10h.{tsv,phn}* files there too for reproducing the ablation study on  layer selection. In **/path/to/data/without_silence** you have the same files, except *.tsv* files contain audios with silences removed using rVAD.

Here is how you can create new audio files without silences from a list of input audio files:
```
python scripts/unsupervised/remove_silences.py /path/to/data/with_silence/train.tsv \
	--save-dir /path/to/data/without_silence/audio \
	--output /path/to/data/without_silence/train.tsv &
```


In this first part, we use mostly phonemized text. Here is how you can transform a text file into its phonemized .phn version:
```
# Will phonemize word dictionary and then phonemize text using dict lookup (for language $lg)

python scripts/unsupervised/phonemize.py $path_to_wrd_dict $lg < text.wrd > text.phn &

```
Next, you can reproduce Figure 2/3 of the wav2vec-U paper by training linear models on top of each layer's frozen wav2vec 2.0 representations, using supervised data. You can observe that certain layers provide lower PER, which shows the closeness of their representations to phoneme outputs. Note that this step requires supervision and is thus not necessary.

```
# Learn linear model on top of layer N(=15) using supervised data

fairseq-hydra-train \
    distributed_training.distributed_port=$PORT \
    task.data=/path/to/data/without_silence \
    model.w2v_path=/path/to/model.pt \
    model.layer=15 \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/finetuning \
    --config-name vox_10h_phn
```



We can extract features of layer *N* using the following:
```
# Extract features from layer N
split=train # valid test
python scripts/unsupervised/fb_wav2vec_ctc_filter.py \
	/path/to/data/without_silence \
	--split $split \
	--layer=15 \
	--checkpoint /path/to/model.pt \
	--save-dir /path/to/features &  
```



Next we perform clustering of wav2vec representations (step 2 in the paper):
```
# Identify clusters in the representations with k-means

python scripts/unsupervised/fb_wav2vec_cluster_faiss.py \  
	/path/to/data/train.tsv \
	-f "CLUS128" \  
	--sample-pct 0.5 \ 
	--layer 15 \
	--checkpoint /path/to/model.pt \
	--save-dir /path/to/features/clustering/segmented &
```
  
And use those clusters to segment the audio data (step 3 in the paper):
```
# Transcribe cluster ids of audio data
python scripts/unsupervised/fb_wav2vec_apply_cluster_faiss.py \  
	/path/to/data \
	--split $split \
	--checkpoint /path/to/model.pt \
	--path /path/to/features/clustering/segmented/CLUS128 &
```
  
  Learn and apply PCA to the representations to retain important features
```
# Compute PCA  
python scripts/pca.py \
	/path/to/features/unfiltered/train.npy \  
	--dim 512 \
	--output /path/to/features/unfiltered/unfiltered_pca

# Apply PCA
python scripts/apply_pca.py \
	$outdir \
	--split $split \  
	--pca-path /path/to/features/unfiltered/unfiltered_pca/512_pca \
	--batch-size 1048000 \
	--save-dir /path/to/features/unfiltered/precompute_unfiltered_pca${dim} 
```

Then we build segment representations by mean-pooling representations according to clusters:
  
  

```
# Build segment representations
 
python scripts/unsupervised/merge_clusters.py \  
	/path/to/features/unfiltered/precompute_unfiltered_pca512 \
	--split $split \  
	--cluster-dir /path/to/features/clustering/segmented/CLUS128 \
	--pooling mean \
	--save-dir /path/to/features/unfiltered/precompute_unfiltered_pca512_cls128_mean  &

```
Finally, we found that segment boundaries are noisy due to the lack of supervision and we therefore found it useful to also mean-pool pairs of adjacent segment representations to increase robustness:
```
# Mean-pool adjacent time steps

python scripts/unsupervised/mean_pool.py \  
	/path/to/features/unfiltered/precompute_unfiltered_pca512_cls128_mean \
	--split $split &  
	--save-dir $savedir &  
```
  
For adversarial training, we preprocess the text data by adding silence tokens.
```
# Add <SIL> tokens on text in preparation for GAN training
python scripts/unsupervised/fb_wrd_to_phonemizer.py \
	-s 0.25 --surround < /path/to/data/gan.txt > /path/to/data/gan.txt_s0.25.phns &

# Binarize with fairseq-preprocess
fairseq-preprocess --dataset-impl mmap \  
	--trainpref /path/to/data/gan.txt_s0.25.phns \
	--workers 6 --thresholdsrc 0 --only-source \  
	--destdir /path/to/data --srcdict /path/to/data/dict.phn.txt &   
```


## Generative adversarial training (GAN)

We then use a GAN model to build a first unsupervised ASR model. The data preparation above of both speech features and text data is a necessary procedure that enables the generator to match speech to text in an unsupervised way. 

Launching GAN training on top of preprocessed features, with default hyperparameters can be done with:

```
PREFIX=w2v_unsup_gan_xp
TASK_DATA=/path/to/features/unfiltered/precompute_unfiltered_pca512_cls128_mean_pooled  
TEXT_DATA=/path/to/data  # path to fairseq-preprocessed GAN data
KENLM_PATH=/path/to/data/kenlm.phn.o4.bin  # KenLM 4-gram phoneme language model (LM data = GAN data here)

PREFIX=$PREFIX fairseq-hydra-train \
	distributed_training.distributed_port=$PORT \
	-m --config-dir configs/unsup \
	--config-name gan_feats_by_label \
	dataset.valid_subset=valid \
	task.data=${TASK_DATA} \
	task.text_data=${TEXT_DATA} \
	task.kenlm_path=${KENLM_PATH} \
	'common.seed=range(0,5)' &
```
However, this step requires an hyperparameter search, which can be launched with:
```

```

Note that hyperparameter search and model/epoch selection are done using a fully unsupervised metric (see Section 4.3).

## Iterative self-training + Kaldi LM-decoding
After the GAN training provides a first unsupervised model, we can then progressively refine the quality of transcriptions using several iterations of semi-supervised learning. We perform two iterations: first, pseudo-label the training data with the unsupervised GAN model and train an HMM on the pseudo-labels. Second, we relabel the training data with the HMM and then fine-tune the original wav2vec 2.0 model using the HMM pseudo-labels with a CTC loss. Note that HMM models use phonemes as output, while wav2vec 2.0 use letter. Both are decoded using WFST decoders into words.


Please see [this README](http://github.com/pytorch/fairseq/tree/master/examples/wav2vec/unsupervised/kaldi_st) for more instructions on how to do iterative self-training + Kaldi LM-decoding.
