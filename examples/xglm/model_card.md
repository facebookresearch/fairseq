# XGLM multilingual model
## Version 1.0.0

### Model developer
FAIR (Fundamental Artificial Intelligence Research)

### Model type
A family of multilingual autoregressive language models (ranging from 564 million to 7.5 billion parameters) trained on a balanced corpus of a diverse set of languages. The language model can learn tasks from natural language descriptions and a few examples.

### Model Feedback Channel
https://github.com/pytorch/fairseq

## Intended use
### Primary intended use
For research purposes only, e.g. reproducing model evaluation results. Generation is only used in a limited capacity for explanation/justification or for prompting/probing/priming for class labels.

### Out of scope uses
The primary purpose of the model is not to generate language, although the model is capable of doing that.

## Potential risks
This section lists the potential risks associated with using the model.

### Relevant factors
Based on known problems with NLP technology, potential relevant factors include output correctness, robustness, bias (gender, profession, race and religion), etc.

### Evaluation factors
The model was evaluated on hate speech detection and occupation identification.
* Hate speech detection (Huang et al. (2020)) - A safety task to test language models’ ability to identify hateful and offensive text.
* Occupation identification (De-Arteaga et al., 2019), (Zhao et al., 2020) - A bias task to study language models’ performance divergence between different gender groups on the task of occupation identification.

## Metrics
### Model performance measures
The XGLM model was primarily evaluated on
1. Zero shot and few shot learning by looking at per-language performance on tasks spanning commonsense reasoning (XCOPA, XWinograd), natural language inference (XNLI) and paraphrasing (PAWS-X). The model is also evaluated on XStoryCloze, a new dataset created by FAIR (Fundamental Artificial Intelligence Research).
2. Cross lingual transfer through templates and few-shot examples.
3. Knowledge probing - Evaluate to what extent the XGLM model can effectively store factual knowledge in different languages using the mLAMA benchmark.
4. Translation - We report machine translation results on WMT benchmarks and a subset of FLORES-101 in the main paper.

The model was also evaluated on hate speech datasets introduced by Huang et al. (2020) and an occupation identification dataset by De-Arteaga et al. 2019 to identify bias in the model.

### Approaches to handle uncertainty
Report confidence intervals, variance metrics for the model performance metrics. Few-shot evaluation was conducted with different sampling with 5 seeds. We reported statistical significance.

## Evaluation data
## Zero Shot and Few Shot evaluation

### XNLI (Conneau et al., 2018)
#### Description
The Cross-lingual Natural Language Inference (XNLI) corpus is the extension of the Multi-Genre NLI (MultiNLI) corpus to 15 languages. The dataset was created by manually translating the validation and test sets of MultiNLI into each of those 15 languages.

### XStoryCloze
#### Description
A new dataset created by FAIR along side this work by translating the validation split of the English StoryCloze dataset (Mostafazadeh et al., 2016) (Spring 2016 version) to 10 other typologically diverse languages (ru, zh Simplified, es Latin America, ar, hi, id, te, sw, eu, my).

### XCOPA (Ponti et al., 2020)
#### Description
The Cross-lingual Choice of Plausible Alternatives (XCOPA) dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across languages. The dataset is the translation and reannotation of the English COPA (Roemmele et al. 2011) and covers 11 languages from 11 families and several areas around the globe.

### XWinograd (Tikhonov and Ryabinin, 2021)
#### Description
XWinograd is a multilingual collection of Winograd Schemas in six languages that can be used for evaluation of cross-lingual commonsense reasoning capabilities.

### PAWS-X (Yang et al., 2019)
#### Description
PAWS-X contains 23,659 human translated PAWS evaluation pairs and 296,406 machine translated training pairs in six typologically distinct languages: French, Spanish, German, Chinese, Japanese, and Korean. All translated pairs are sourced from examples in PAWS-Wiki.

## Responsible AI (RAI) evaluation
### Hate speech (Huang et al. 2020)
This is a multilingual Twitter corpus for the task of hate speech detection with inferred four author demographic factors: age, country, gender and race/ethnicity. The corpus covers five languages: English, Italian, Polish, Portuguese and Spanish.

### Bias dataset (De-Arteaga et al. 2019)
The aim of this dataset is to study the gender bias of models that identify a person’s occupation from their bios.

----

## Training data
### CC100-XL
#### Description
Following the recent success of multilingual self-supervised pre-training (Devlin et al., 2019; Lample and Conneau, 2019; Con; Xue et al., 2020; Goyal et al., 2021a; Liu et al., 2020), we train our language models on a mixture of monolingual text of different languages. We extended the pipeline used for mining the CC100 corpus to generate CC100-XL, a significantly larger multilingual dataset covering 68 Common Crawl snapshots (from Summer 2013 to March/April 2020) and 134 languages.

More details on the CC100-XL dataset can be found in the Appendix section of the paper.

## RAI Dimensions
### Fairness (Bias and inclusion)
The XGLM model was evaluated on Hate speech and bias identification datasets. For hate speech, we observe that across the 5 languages in the dataset, in context learning results are only slightly better than random (50%). Another interesting observation is that most few shot results are worse than zero-shot, which indicates that the model is not able to utilize examples using the templates described in the paper. For bias identification, the XGLM (6.7B) English only model achieves the best performance on English and Spanish, while the GPT-3 model of comparable size (6.7B) model achieves the best in French. On certain occupations (e.g. model and teacher), XGLM 6.7B En only model and GPT-3 (6.7B) have very significant bias while XGLM 7.5B is much less biased.

### Privacy and security
The XGLM model did not have any special Privacy and Security considerations. The training data and evaluation data were both public and went through standard Meta privacy and licensing procedures.

### Transparency and control
In the spirit of transparency and accountability we have created this model card and a data card for the CC100-XL which can be found in the Appendix section of the paper.

### Efficiency (Green AI)
From an engineering perspective, XGLM pertains to a family of models that represent single unified models catering to many languages which have wide application across many applications. Such a unified single model saves on carbon footprint as well as energy consumption (comparing to the alternative: separate models for different languages) leading to more energy efficiency. A single model, despite having the risk of being a single point of failure, has the powerful incentive of being easier to maintain, access, distribute, and track.
 
## References
Edoardo Maria Ponti, Goran Glavas, Olga Majewska, Qianchu Liu, Ivan Vulic, and Anna Korhonen. 2020. XCOPA: A multilingual dataset for causal commonsense reasoning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 2362–2376. Association for Computational Linguistics.
XCOPA Dataset | Papers With Code

Alexey Tikhonov and Max Ryabinin. 2021. It’s all in the heads: Using attention heads as a baseline for cross-lingual transfer in commonsense reasoning. In Findings of the Association for Computational Linguistics: ACL/IJCNLP 2021, Online Event, August 1-6, 2021, volume ACL/IJCNLP 2021 of Findings of ACL, pages 3534–3546. Association for Computational Linguistics. 
XWINO Dataset | Papers With Code (XWinograd)

Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. 2019. PAWS-X: A cross-lingual adversarial dataset for paraphrase identification. CoRR, abs/1908.11828.
PAWS-X Dataset | Papers With Code

Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R. Bowman, Holger Schwenk, and Veselin Stoyanov. 2018. XNLI: evaluating cross-lingual sentence representations. CoRR, abs/1809.05053.
XNLI Dataset | Papers With Code 

Xiaolei Huang, Linzi Xing, Franck Dernoncourt, and Michael Paul. 2020. Multilingual twitter corpus and baselines for evaluating demographic bias in hate speech recognition. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 1440–1448. 

Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Geyik, Krishnaram Kenthapadi, and Adam Tauman Kalai. 2019. Bias in bios: A case study of semantic representation bias in a high-stakes setting. In proceedings of the Conference on Fairness, Accountability, and Transparency, pages 120–128.

Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, James F. Allen. A Corpus and Evaluation Framework for Deeper Understanding of Commonsense Stories. CoRR abs/1604.01696.

Jieyu Zhao, Subhabrata Mukherjee, Saghar Hosseini, Kai-Wei Chang, and Ahmed Hassan Awadallah. 2020. Gender bias in multilingual embeddings and crosslingual transfer. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2896–2907.

## Citation details
```
@article{DBLP:journals/corr/abs-2112-10668,
  author    = {Xi Victoria Lin and
               Todor Mihaylov and
               Mikel Artetxe and
               Tianlu Wang and
               Shuohui Chen and
               Daniel Simig and
               Myle Ott and
               Naman Goyal and
               Shruti Bhosale and
               Jingfei Du and
               Ramakanth Pasunuru and
               Sam Shleifer and
               Punit Singh Koura and
               Vishrav Chaudhary and
               Brian O'Horo and
               Jeff Wang and
               Luke Zettlemoyer and
               Zornitsa Kozareva and
               Mona T. Diab and
               Veselin Stoyanov and
               Xian Li},
  title     = {Few-shot Learning with Multilingual Language Models},
  journal   = {CoRR},
  volume    = {abs/2112.10668},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.10668},
  eprinttype = {arXiv},
  eprint    = {2112.10668},
  timestamp = {Tue, 04 Jan 2022 15:59:27 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-10668.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
