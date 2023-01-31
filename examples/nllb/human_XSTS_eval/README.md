# No Language Left Behind : Human Evaluation

## Cross-Lingual Semantic Textual Similarity (XSTS) Resources

This folder holds supplimentary resources for conducting and understanding the human translation quality evaluations (XSTS) used for NLLB.

## Paper

For more detailed discussion please see our paper:

***"Consistent Human Evaluation of Machine Translation across Language Pairs"***

- [AMTA](https://aclanthology.org/2022.amta-research.24.pdf) 
- [arxiv](https://arxiv.org/abs/2205.08533)

## Resource Files

[`STS_Evaluation_Guidelines.pdf`](STS_Evaluation_Guidelines.pdf)
- XSTS protocol execution guidelines for annotators
- A written document outlining how to conduct the protocol, i.e. instructions to whomever is going to try to do an annotation.

[`Human_Eval_Training_Set.xlsx`](Human_Eval_Training_Set.xlsx)
- Complimentary to the above, a set of XSTS examples with labeled 'answers' and justification. To be used for human evaluator onboarding/training purposes.

`NLLB_200_XSTS_annotation_data.tsv`
- [Archive Download - NLLB_XSTS_Datasets.zip](https://dl.fbaipublicfiles.com/nllb/NLLB_XSTS_Datasets.zip)
- This a copy of XSTS scores recieved from human annotators on an evaluation of the NLLB_200 model and a 'baseline model' (both from the NLLB_200 paper) on FLORES data. 
- It has source and target language, input and output text strings, and evaluator quality scores. Source was always FLORES. The calibration set (see below) is also evaluated.

**XSTS Calibration Set Examples**
- Files included in [NLLB_XSTS_Datasets.zip](https://dl.fbaipublicfiles.com/nllb/NLLB_XSTS_Datasets.zip) with overall annotation data
- When implimenting XSTS it is recommended you generate your own usecase specific dataset, but we provide some examples here as an example to help get started.
- Description:  A dataset of source-target strings used to evaluate and correct for bias (as in too harsh, to lenient graders) in a set of human annotators performing the XSTS protocol. The calibration set items source strings are also FLORES in this case but the target translations are from a variety prototype systems of intentionally variable quality.
- Files:
    - `Cross_Lingual_Calibration_Set_v1_500.tsv` - The Exact Calibration items used for the final NLLB_200 evaluation.  100 items per quality strata.
    - `Cross_Lingual_Calibration_Set_v1.tsv` - A larger set of 1000 items including the above.  200 items per quality strata.
    - `Cross_Lingual_Calibration_Set_v2.tsv` - A second sample of 1000 items drawn from a larger evaluation pool.  200 items per quality strata.

