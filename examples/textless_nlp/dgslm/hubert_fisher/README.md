# Dialogue Speech-to-Unit Encoder for dGSLM: The Fisher HuBERT model
For the speech2unit encoder, we train a [HuBERT model](https://arxiv.org/pdf/2106.07447.pdf) on the [Fisher dataset](http://www.lrec-conf.org/proceedings/lrec2004/pdf/767.pdf) for 3 iterations (see [our paper](https://arxiv.org/pdf/2203.16502.pdf) for more details) and train a k-means model with 500 units on the layer 12 features of the HuBERT model.

The pre-trained HuBERT and k-means model checkpoints can be found here:

| Fisher HuBERT model | k-means model |
|---------------------|---------------|
|[download](https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/hubert/hubert_fisher.pt)|[download](https://dl.fbaipublicfiles.com/textless_nlp/dgslm/checkpoints/hubert/hubert_fisher_km_500.bin)|

Below is an example command to encode a stereo dataset to discrete units using the pre-trained model checkpoints :
```bash
for CHANNEL_ID in 1 2; do
    python examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
        --feature_type hubert \
        --kmeans_model_path path/to/hubert_fisher_km_500.bin \
        --acoustic_model_path path/to/hubert_fisher.pt \
        --layer 12 \
        --manifest_path $MANIFEST_FILE \
        --out_quantized_file_path ${OUTPUT_FILE}-channel${CHANNEL_ID} \
        --extension $EXTENSION \
        --channel_id $CHANNEL_ID
done
```
where MANIFEST_FILE is the output of [wav2vec manifest script](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/wav2vec_manifest.py), which can be obtained through the following command :
```
python examples/wav2vec/wav2vec_manifest.py --valid-percent=0.0 $AUDIO_DIR --dest=$OUTPUT_DIR --ext=$EXTENSION
```