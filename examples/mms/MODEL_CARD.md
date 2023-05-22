# MMS Model Card

## Model details

**Organization developing the model**  The FAIR team of Meta AI.

**Model version**  This is version 1 of the model.

**Model type**  MMS is speech model, based on the transformer architecture. The pre-trained model comes in two sizes: 300M and 1B parameters. We fine-tune the model for speech recognition and make it available in the 1B variant. We also fine-tune the 1B variant for language identification.

**License**  CC BY-NC

**Where to send questions or comments about the model**  Questions and comments about MMS can be sent via the  [GitHub repository](https://github.com/pytorch/fairseq/tree/master/examples/mms)  of the project , by opening an issue and tagging it as MMS.

## Uses

**Primary intended uses**  The primary use of MMS is to perform speech processing research for many more languages and to perform tasks such as automatic speech recognition, language identification, and speech synthesis.

**Primary intended users**  The primary intended users of the model are researchers in speech processing, machine learning and artificial intelligence.

**Out-of-scope use cases**  Fine-tuning the pre-pretrained models on other labeled datasets or downstream tasks requires further risk evaluation and mitigation.

## Bias and Risks

The MMS models were pre-trained on a blend of data from different domains, including readings of the New Testament. In the paper, we describe two studies analyzing gender bias and the use of religious language which conclude that models perform equally well for both genders and that on average, there is little bias for religious language (section 8 of the paper).

# Training Details

## Training Data

MMS is pre-trained on VoxPopuli (parliamentary speech), MLS (read audiobooks), VoxLingua-107 (YouTube speech), CommonVoice (read Wikipedia text), BABEL (telephone conversations), and MMS-lab-U (New Testament readings), MMS-unlab (various read Christian texts).
Models are fine-tuned on FLEURS, VoxLingua-107, MLS, CommonVoice, and MMS-lab. We obtained the language information for MMS-lab, MMS-lab-U and MMS-unlab from our data soucrce and did not  manually verify it for every language.

## Training Procedure

Please refer to the research paper for details on this.

# Evaluation

## Testing Data, Factors & Metrics

We evaluate the model on a different benchmarks for the downstream tasks. The evaluation details are presented in the paper. The models performance is measured using standard metrics such as character error rate, word error rate, and classification accuracy.


# Citation

**BibTeX:**

```
@article{pratap2023mms,
  title={Scaling Speech Technology to 1,000+ Languages},
  author={Vineel Pratap and Andros Tjandra and Bowen Shi and Paden Tomasello and Arun Babu and Sayani Kundu and Ali Elkahky and Zhaoheng Ni and Apoorv Vyas and Maryam Fazel-Zarandi and Alexei Baevski and Yossi Adi and Xiaohui Zhang and Wei-Ning Hsu and Alexis Conneau and Michael Auli},
  journal={arXiv},
  year={2023}
}

```

# Model Card Contact

Please reach out to the authors at: [vineelkpratap@meta.com](mailto:vineelkpratap@meta.com) [androstj@meta.com](mailto:androstj@meta.com) [bshi@meta.com](mailto:bshi@meta.com) [michaelauli@meta.com](mailto:michaelauli@gmail.com)


