# Self-Training with Kaldi HMM Models
This folder contains recipes for self-training on pseudo phone transcripts and
decoding into phones or words with [kaldi](https://github.com/kaldi-asr/kaldi).

To start, download and install kaldi follow its instruction, and place this
folder in `path/to/kaldi/egs`.

## Training
Assuming the following has been prepared:
- `w2v_dir`: contains features `{train,valid}.{npy,lengths}`, real transcripts `{train,valid}.${label}`, and dict `dict.${label}.txt`
- `lab_dir`: contains pseudo labels `{train,valid}.txt`
- `arpa_lm`: Arpa-format n-gram phone LM for decoding
- `arpa_lm_bin`: Arpa-format n-gram phone LM for unsupervised model selection to be used with KenLM

Set these variables in `train.sh`, as well as `out_dir`, the output directory,
and then run it.

The output will be:
```
==== WER w.r.t. real transcript (select based on unsupervised metric)
INFO:root:./out/exp/mono/decode_valid/scoring/14.0.0.tra.txt: score 0.9178 wer 28.71% lm_ppl 24.4500 gt_wer 25.57%
INFO:root:./out/exp/tri1/decode_valid/scoring/17.1.0.tra.txt: score 0.9257 wer 26.99% lm_ppl 30.8494 gt_wer 21.90%
INFO:root:./out/exp/tri2b/decode_valid/scoring/8.0.0.tra.txt: score 0.7506 wer 23.15% lm_ppl 25.5944 gt_wer 15.78%
```
where `wer` is the word eror rate with respect to the pseudo label, `gt_wer` to
the ground truth label, `lm_ppl` the language model perplexity of HMM prediced
transcripts, and `score` is the unsupervised metric for model selection. We
choose the model and the LM parameter of the one with the lowest score. In the
example above, it is `tri2b`, `8.0.0`.


## Decoding into Phones
In `decode_phone.sh`, set `out_dir` the same as used in `train.sh`, set
`dec_exp` and `dec_lmparam` to the selected model and LM parameter (e.g.
`tri2b` and `8.0.0` in the above example). `dec_script` needs to be set
according to `dec_exp`: for mono/tri1/tri2b, use `decode.sh`; for tri3b, use
`decode_fmllr.sh`.

The output will be saved at `out_dir/dec_data`


## Decoding into Words
`decode_word_step1.sh` prepares WFSTs for word decoding. Besides the variables
mentioned above, set
- `wrd_arpa_lm`: Arpa-format n-gram word LM for decoding
- `wrd_arpa_lm_bin`: Arpa-format n-gram word LM for unsupervised model selection

`decode_word_step1.sh` decodes the `train` and `valid` split into word and runs
unsupervised model selection using the `valid` split. The output is like:
```
INFO:root:./out/exp/tri2b/decodeword_valid/scoring/17.0.0.tra.txt: score 1.8693 wer 24.97% lm_ppl 1785.5333 gt_wer 31.45%
```

After determining the LM parameter (`17.0.0` in the example above), set it in
`decode_word_step2.sh` and run it. The output will be saved at
`out_dir/dec_data_word`.
