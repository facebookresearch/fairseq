# N-best Re-ranking for Multilingual LID+ASR
This project provides N-best re-ranking, a simple inference procedure, for improving multilingual speech recognition (ASR) "in the wild" where models are expected to first predict language identity (LID) before transcribing. Our method considers N-best LID predictions for each utterance, runs the corresponding ASR in N different languages, and then uses external features over the candidate transcriptions to determine re-rank. 

The workflow is as follows: 1) run LID+ASR inference (MMS and Whisper are supported), 2) compute external re-ranking features, 3) tune feature coefficients on dev set, and 4) apply on test set.

For more information about our method, please refer to the paper: "Improving Multilingual ASR in the Wild Using Simple N-best Re-ranking".

## 1) Commands to Run LID+ASR Inference

### Data Prep
Prepare a text file with one path to a wav file in each line:
```
#/path/to/wav/list
/path/to/audio1.wav
/path/to/audio2.wav
/path/to/audio3.wav
```

The following workflow also assumes that LID and ASR references are available (at least for the dev set). We use [3-letter iso codes](https://dl.fbaipublicfiles.com/mms/lid/mms1b_l4017_langs.html) for both Whisper and MMS.

Next run either Whisper or MMS based LID+ASR.

### Whisper
Refer to the [Whisper documentation](https://github.com/openai/whisper) for installation instructions.

First run LID:
```
python whisper/infer_lid.py --wavs "path/to/wav/list" --dst "path/to/lid/results" --model large-v2 --n 10
```
Note that the size of the N-best list is set as 10 here.

Then run ASR, using the top-N LID predictions:
```
python whisper/infer_asr.py --wavs "path/to/wav/list" --lids "path/to/lid/results"/nbest_lid --dst "path/to/asr/results" --model large-v2
```

### MMS
Refer to the [Fairseq documentation](https://github.com/facebookresearch/fairseq/tree/main) for installation instructions.

Prepare data and models following the [instructions from the MMS repository](https://github.com/facebookresearch/fairseq/tree/main/examples/mms). Note that the MMS backend expects a slightly different wav list format, which can be obtained via:
```
python mms/format_wav_list.py --src "/path/to/wav/list" --dst "/path/to/wav/manifest.tsv"
```
Note that MMS also expects LID references in a file named `"/path/to/wav/manifest.lang"`.

Then run LID:
```
cd "path/to/fairseq/dir"
PYTHONPATH='.'  python3  examples/mms/lid/infer.py "path/to/dict/dir" --path "path/to/model" --task audio_classification  --infer-manifest "path/to/wav/manifest.tsv" --output-path "path/to/lid/results" --top-k 10
```
Note that the size of the N-best list is set as 10 here.

Then run ASR, using the top-N LID predictions. Since MMS uses language-specific parameters, we've parallelized inference across languages:
```
#Split data by language
python mms/split_by_lang.py --wavs_tsv "/path/to/wav/manifest.tsv" --lid_preds "path/to/lid/results"predictions.txt --dst "path/to/data/split"

#Write language-specific ASR python commands to an executable file
mms/make_parallel_single_runs.py --dump "path/to/data/split" --model "path/to/model" --dst "path/to/asr/results" --fairseq_dir "path/to/fairseq/dir" > run.sh

#Running each language sequentially (you can also parallelize this)
. ./run.sh

#Merge language-specific results back to original order
python mms/merge_by_run.py --dump "path/to/data/split" --exp "path/to/asr/results"
```

## 2) Commands to Compute External Re-ranking Features

### MaLA - Large Language Model
```
python mala/infer.py --txt "path/to/asr/results"/nbest_asr_hyp --dst "path/to/lm/results"
```

### NLLB - Written LID Model
Download the model from the [official source](https://github.com/facebookresearch/fairseq/tree/nllb#lid-model).

```
python nllb/infer.py --txt "path/to/asr/results"/nbest_asr_hyp --dst "path/to/wlid/results" --model "path/to/nllb/model"
```

### MMS-Zeroshot - U-roman Acoustic Model
Download the model from the [official source](https://huggingface.co/spaces/mms-meta/mms-zeroshot/tree/main).

First run u-romanization on the N-best ASR hypotheses:
```
python mms-zs/uromanize.py --txt "path/to/asr/results"/nbest_asr_hyp --lid "path/to/lid/results"/nbest_lid --dst "path/to/uasr/results" --model "path/to/mms-zeroshot"
```

Then compute the forced alignment score using the MMS-Zeroshot model:
```
python mms-zs/falign.py --uroman_txt "path/to/uasr/results"/nbest_asr_hyp_uroman --wav "path/to/wav/list" --dst "path/to/uasr/results" --model "path/to/mms-zeroshot"
```

## 3) Commands to Tune Feature Coefficients
```
python rerank/tune_coefficients.py --slid "path/to/lid/results"/slid_score --asr "path/to/asr/results"/asr_score --wlid "path/to/wlid/results"/wlid_score --lm "path/to/lm/results"/lm_score --uasr "path/to/uasr/results"/uasr_score --dst "path/to/rerank/results" --ref_lid "ground-truth/lid" --nbest_lid "path/to/lid/results"/nbest_lid --ref_asr "ground-truth/asr" --nbest_asr "path/to/asr/results"/nbest_asr_hyp
```

## 4) Commands to Apply on Test Set
```
python rerank/rerank.py --slid "path/to/lid/results"/slid_score --asr "path/to/asr/results"/asr_score --wlid "path/to/wlid/results"/wlid_score --lm "path/to/lm/results"/lm_score --uasr "path/to/uasr/results"/uasr_score --dst "path/to/rerank/results" --ref_lid "ground-truth/lid" --nbest_lid "path/to/lid/results"/nbest_lid --ref_asr "ground-truth/asr" --nbest_asr "path/to/asr/results"/nbest_asr_hyp --w "path/to/rerank/results"/best_coefficients
```

The re-ranked LID and ASR will be in `"path/to/rerank/results"/reranked_1best_lid` and `"path/to/rerank/results"/reranked_1best_asr_hyp` respectively.

# Citation
```
@article{yan2024wild,
  title={Improving Multilingual ASR in the Wild Using Simple N-best Re-ranking},
  author={Brian Yan, Vineel Pratap, Shinji Watanabe, Michael Auli},
  journal={arXiv},
  year={2024}
}
```