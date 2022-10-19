![Universal Speech Translator](ust.png?raw=true "UST")


# Universal Speech Translator

Universal Speech Translator (UST) is the project to enable speech-to-speech translation with real-time, natural-sounding translations at near-human level quality. This code repository open sources our latest works for the mission, including code, model architecture design, data mining, and benchmark efforts.

* [Meta AI Blog Post](https://ai.facebook.com/blog/ai-translation-hokkien)
* [Tech at Meta Blog Post](https://tech.fb.com/ideas/2022/10/ai-translation-unwritten-language/)
* [Hokkien Demo](https://huggingface.co/spaces/facebook/Hokkien_Translation )


## October 2022 Release:
### [Speech-to-speech translation for a real-world unwritten language (English-Hokkien)](https://research.facebook.com/publications/hokkien-direct-speech-to-speech-translation)  [[Project Page](https://github.com/facebookresearch/fairseq/tree/ust/examples/hokkien)]

We study speech-to-speech translation (S2ST) that translates speech from one language into
another language and focuses on building systems to support languages without standard text
writing systems. We use English-Taiwanese Hokkien as a case study, and present an end-to-end solution from training data collection, modeling choices to benchmark dataset release. We open source our best S2ST models and the S2ST benchmark dataset to facilitate future research in this field.


### [Simple and Effective Unsupervised Speech Translation](https://research.facebook.com/publications/unsupervised-direct-speech-to-speech-translation) [Project Page coming soon]

The amount of labeled data to train models for speech tasks is limited for most languages, however, the data scarcity is exacerbated for speech translation which requires labeled data covering two different languages. To address this issue, we study a simple and effective approach to build speech translation systems without any labeled data by leveraging recent advances in unsupervised speech recognition, machine translation and speech synthesis. Furthermore, we present an unsupervised domain adaptation technique for pre-trained speech models which improves the performance of downstream unsupervised speech recognition, especially for low-resource settings. Experiments show that unsupervised speech-to-text translation outperforms the previous unsupervised state of the art by 3.2 BLEU on the Libri-Trans benchmark, on CoVoST 2, our best systems outperform the best supervised end-to-end models (without pre-training) from only two years ago by an average of 5.0 BLEU over five X-En directions. We also report competitive results on MuST-C and CVSS benchmarks.


### [UnitY: Two-pass Direct Speech-to-speech Translation with Discrete Units](https://research.facebook.com/publications/unity-direct-speech-to-speech-translation)  [[Project Page](https://github.com/facebookresearch/fairseq/tree/main/examples/speech_to_speech/unity)]

We present a novel two-pass direct S2ST architecture, UnitY, which first generates textual representations and predicts discrete acoustic units subsequently. We enhance the model performance by subword prediction in the first-pass decoder, advanced two-pass decoder architecture design and search strategy, and better training regularization. To leverage large amounts of unlabeled text data, we pre-train the first-pass text decoder based on the self-supervised denoising auto-encoding task. Experimental evaluations on benchmark datasets at various data scales demonstrate that UnitY outperforms a single-pass speech-to-unit translation model by up to 3.5 ASR-BLEU with 2.83x decoding speed-up.


### [SpeechMatrix: A Large-Scale Mined Corpus of Multilingual Speech-to-Speech Translations](https://research.facebook.com/publications/speechmatrix)  [[Project Page](https://github.com/facebookresearch/fairseq/tree/ust/examples/speech_matrix)]

We present SpeechMatrix, a large-scale multilingual corpus of speech-to-speech translations mined from real speech of European Parliament recordings. It contains speech alignments in 136 language pairs with a total of 418 thousand hours of speech. To evaluate the quality of this parallel speech, we train bilingual speech-to-speech translation models on mined data only and establish extensive baseline results on EuroParl-ST, VoxPopuli and FLEURS test sets. Enabled by the multilinguality of SpeechMatrix, we also explore multilingual speech-to-speech translation, demonstrating that model pre-training and sparse scaling using Mixture-of-Experts bring large gains to translation performance. The mined data and models are freely available.

### [w2v-BERT implementation](https://github.com/facebookresearch/fairseq/tree/ust/examples/w2vbert)

We implement and release w2v-BERT, a SoTA self-supervised speech representation learning algorithm combining contrastive learning and masked language modeling for pre-training speech encoders. We also provide checkpoints pre-trained on the Libri-light corpus as well as those fine-tuned on the LibriSpeech 100h/960h subsets to facilitate future research.

### [ASR-BLEU evaluation toolkit](https://github.com/facebookresearch/fairseq/tree/ust/examples/speech_to_speech/asr_bleu)

We enable a reproducible way to compute the ASR-BLEU metric. We provide a set of publicly available ASR models which covers all target languages we use. A helper script transcribes model’s audio predictions and computes a BLEU score between model’s predicted transcriptions and reference translations.


## License
UST code and fairseq(-py) is MIT-licensed available in [LICENSE](LICENSE) file.
