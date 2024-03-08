# Data card for the paper "Efficient Large Scale Language Modeling with Mixtures of Experts"
## Version 1.0.0

We follow the recommendations of Gebru et al. (2018) and provide a datacard for the dataset used to train the 1.1T parameter model.

## Motivation
* **For what purpose was the dataset created? Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.**
The pre-training data for training the 1.1 T model was created by a union of six English language datasets, including five datasets used by RoBERTa (Liu et al 2019) and the English subset of CC 100. These purpose of creating this dataset was to pre-train the language model.

* **Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**
FAIR (Fundamental Artificial Intelligence Research)

* **Who funded the creation of the dataset? If there is an associated grant, please provide the name of the grantor and the grant name and number.**
FAIR (Fundamental Artificial Intelligence Research)

* **Any other comments?**
No.

## Composition

* **What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.**
The instances are textual documents. The overall dataset is composed from a union of the following datasets - 
    * BookCorpus (Zhu et al., 2019) consists of more than 10K unpublished books (4GB);
    * English Wikipedia, excluding lists, tables and headers (12GB);
    * CC-News (Nagel,2016) contains 63 million English news articles crawled between September 2016 and February 2019 (76GB);
    * OpenWebText (Gokaslan and Cohen, 2019), an open source recreation of the WebText dataset used to train GPT-2 (38GB);
    * CC-Stories (Trinh and Le, 2018) contains a subset of CommonCrawl data filtered to match the story-like style of Winograd schemas (31GB);
    * English CC100 (Wenzek et al., 2020), a dataset extracted from CommonCrawl snapshots between January 2018 and December 2018, filtered to match the style of Wikipedia (292GB).

* **How many instances are there in total (of each type, if appropriate)?**
The training data contains 112B tokens corresponding to 453 GB of data.

* **Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).**
The English CC100 section of the dataset is a subset of CommonCrawl snapshots extracted between January 2018 to December 2018, filtered to match the style of Wikipedia. The CC-stories dataset contains a subset of CommonCrawl data filtered to match the story-like style of Winograd schemas.

* **What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features? In either case, please provide a description.**
Each instance consists of raw text data.

* **Is there a label or target associated with each instance? If so, please provide a description.**
No.

* **Is any information missing from individual instances? If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.**
No.

* **Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)? If so, please describe how these relationships are made explicit.**
There are no explicit relationships between individual instances.

* **Are there recommended data splits (e.g., training, development/validation, testing)? If so, please provide a description of these splits, explaining the rationale behind them.** 
We hold out a random validation set of approximately 150MB from the pretraining data, sampled proportionally to each dataset's size in the pretraining corpus.

* **Are there any errors, sources of noise, or redundancies in the dataset? If so, please provide a description.**
N/A

* **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?**
It's self-contained.

* **Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals' non-public communications)? If so, please provide a description.**
The datasets used are publicly available, and the information in them is not considered confidential.

* **Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? If so, please describe why.**
Parts of the dataset are a subset of public Common Crawl data, which could contain sentences that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety.

* **Does the dataset relate to people? If not, you may skip the remaining questions in this section.**
Some documents of this data relate to people, such as news articles, Wikipedia descriptions, etc.

* **Does the dataset identify any subpopulations (e.g., by age, gender)? If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.**
No.

* **Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? If so, please describe how**
In addition to individuals who have Wikipedia pages (celebrities, politicians, etc.), it may be possible to identify other individuals by their names, Twitter account names, etc. if that information is present in Common Crawl.

* **Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? If so, please provide a description.**
The training dataset is partially derived from Common Crawl, which may contain some sensitive information.

* **Any other comments?**
No


## Collection Process

* **How was the data associated with each instance acquired? Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/ derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.**
N/A. The dataset is a union of six publicly available datasets.

* **What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)? How were these mechanisms or procedures validated?**
N/A

* **If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?**
Please refer to the main document for details.

* **Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?**
This data is mined, filtered and sampled by machines.

* **Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)? If not, please describe the timeframe in which the data associated with the instances was created.**
Different parts of the dataset were mined over different time periods.
1. The CC-News dataset contains English news articles crawled between September 2016 and February 2019.
2. The English CC-100 dataset was extracted from CommonCrawl snapshots between January 2018 and December 2018.

* **Were any ethical review processes conducted (e.g., by an institutional review board)? If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.**
No. 

* **Does the dataset relate to people? If not, you may skip the remainder of the questions in this section.**
No.

* **Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?**
N/A

* **Were the individuals in question notified about the data collection? If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.**
N/A

* **Did the individuals in question consent to the collection and use of their data? If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.**
N/A

* **If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).**
N/A

* **Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.**
Some responsible AI related evaluations were performed. Please refer to the main document and the model card for the paper.

* **Any other comments?**
No


## Preprocessing/cleaning/labeling


* **Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? If so, please provide a description. If not, you may skip the remainder of the questions in this section.**
The component datasets went through standard cleaning and re-formatting practices, including removing repetitive/non informative text like "Chapter One", or "This ebook by Project Gutenberg".
    
* **Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)? If so, please provide a link or other access point to the “raw” data.**
The "raw" component datasets is publicly available in their respective locations (more details can be seen in the respective papers linked in references).

* **Is the software used to preprocess/clean/label the instances available? If so, please provide a link or other access point.**
The software is proprietary to Meta Platforms and currently unavailable publicly.

* **Any other comments?**
No


## Uses

* **Has the dataset been used for any tasks already? If so, please provide a description.**
Yes, this dataset was used to pre-train the models described in the paper.

* **Is there a repository that links to any or all papers or systems that use the dataset? If so, please provide a link or other access point.**
No.

* **What (other) tasks could the dataset be used for?**
This data can be used to pretrain English language models, which are foundation to many current and future language tasks.

* **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks) If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?**
The pipeline for creating this dataset paves a way for building a scalable infrastructure for mining datasets to be be used for training large-scale models.

* **Are there tasks for which the dataset should not be used? If so, please provide a description.**
No.

* **Any other comments?**
No.

## Distribution


* **Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? If so, please provide a description.**
No. 

* **How will the dataset will be distributed (e.g., tarball on website, API, GitHub)? Does the dataset have a digital object identifier (DOI)?**
N/A

* **When will the dataset be distributed?**
No.

* **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.**
No.

* **Have any third parties imposed IP-based or other restrictions on the data associated with the instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.**
No.

* **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.**
N/A

* **Any other comments?**
No.

## Maintenance

* **Who is supporting/hosting/maintaining the dataset?**
FAIR (Fundamental Artificial Intelligence Research)

* **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**
Refer to the main document.

* **Is there an erratum? If so, please provide a link or other access point.**
N/A

* **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?**
No plan for updating.

* **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)? If so, please describe these limits and explain how they will be enforced.**
N/A

* **Will older versions of the dataset continue to be supported/hosted/maintained? If so, please describe how. If not, please describe how its obsolescence will be communicated to users.**
N/A

* **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? If so, please provide a description. Will these contributions be validated/ verified? If so, please describe how. If not, why not? Is there a process for communicating/ distributing these contributions to other users? If so, please provide a description.**
No.

* **Any other comments?**
No.

## References
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2019. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. arXiv:1506.06724.

Sebastian Nagel. 2016. Cc-news. http: //web.archive.org/save/http: //commoncrawl.org/2016/10/news-dataset-available.

Aaron Gokaslan and Vanya Cohen. 2019. Openwebtext corpus. http://web.archive.org/save/http://Skylion007.github.io/OpenWebTextCorpus

Trieu H Trinh and Quoc V Le. 2018. A simple method for commonsense reasoning. arXiv preprint arXiv:1806.02847.

Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave. 2020. CCNet: Extracting high quality monolingual datasets from web crawl data. In Proceedings of the 12th Language Resources and Evaluation Conference, pages 4003–4012, Marseille, France. European Language Resources Association.

