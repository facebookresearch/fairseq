# Wikipedia Biographies of Women


## Training:

The training dataset is created based on WikiSum, a dataset created from the paper [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/pdf/1801.10198.pdf). The dataset needs to be generated following the instructions in this [Github Repository](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/data_generators/wikisum).

### How is the WikiSum dataset structured?

Overall, the task in WikiSum was to generate the entire Wikipedia article based on the contents of the top 10 Google Search Results. The authors provide a way for people to recreate their work. In the WikiSum Github, there are two options for the dataset recreation --- the first is to use CommonCrawl (a static, open source crawl of the web) and the second to do Live Web Fetches. The second has higher coverage, but the content is subject to change and difficult to fetch. We used the static, Commoncrawl version. This can be downloaded following the Github repo instructions, though note it will require usage of Google Cloud. 

Note: in our experience, it also requires requesting that the resource limit of the Google Cloud instance be raised, which requires emailing. 

Note: Having higher coverage in the training dataset would be expected to improve the model quality. There are many instances in the dataset where the training input (web evidence) does not contain sufficient content for producing the desired Wikipedia article. This may harm the model's ability to learn to retrieve, look at the input evidence, and overall could contribute to increased challenges in generating verifiable Wikipedia biographies. 

### How do you go from WikiSum dataset to Biography dataset?

The WikiSum dataset is for Wikipedia in general, not just biographies. We do this by querying WikiData to see if the Wikipedia article has an occupation, with the thought that all articles with occupations are probably biographies.


## Evaluation:

You can download the dataset and baseline model with the following command:

```
wget -N 'https://dl.fbaipublicfiles.com/fairseq/womenbios_dataset.zip'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
```

We provide the full text Wikipedia articles split into four categories:
- Women in Africa
- Women in Asia
- Women in Science 
- Women 
We note that these are not exhaustive intersectional categories and mainly stem from personal interest. 

We also provide the URL of the Wikipedia article. Note that Wikipedia articles are constantly being improved, edited, and changed. Thus, it's completely possible that the Wikipedia article on Wikipedia has been lovingly improved by other Wikipedia editors. 

To get the occupations of each biographical subject, we use WikiData. We provide a sample script to do this. We also provide the raw output of this query. 

The final part of the evaluation dataset is to query web evidence for each of the biographical subjects. This is the part of the evaluation dataset that requires the most improvement. As we discuss in our paper, one of the major reasons why it is difficult to write biographies for sometimes very well qualified women is that there is not information online about them. Further, the search engine may not find it. We encourage others to improve upon this part of the data, as even re-querying again on the internet may find new, updated sources of information as the web is constantly evolving. 

We use the search engine from [Internet-Augmented Dialogue Generation](https://arxiv.org/abs/2107.07566), see [project URL](https://parl.ai/projects/sea/) to do the search queries. Note: we remove wikipedia site sources from our query (or we'd query the data itself). However, it's possible Wikipedia information can be copied around in multiple forms on the web, linked with edits, etc. 


## Section by Section Generation:

Wikipedia articles are split into sections, which are usually separated by headings. These headings can be separated in the article text by looking for these equal signs (==), where the number of equal signs usually signals if you are looking at a toplevel heading or a subheading, etc. An example regex that you can use is:

`
section_header_re = re.compile(r"(?<!=)==([^=]+)==(?!=)")
`


## List of Notes:
- People can have multiple occupations, and we keep all occupations that we query from WikiData


## List of Possible Improvement Areas:
Using a larger generative pre-trained model, larger-scale retrieval, a retrieval encoder specialized to Wikipedia (or biographies), tuning all of the training & generation parameters exhaustively --- and the like --- would most likely be very useful. Overall, we hope that this is a starting point for others who might be interested in focusing on how we can help address the gender gap on Wikipedia.


## Interested in Wikipedia and Gender Gap? 
You might want to check out:
- https://humaniki.wmcloud.org/
- https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Women_in_Red and https://wikimediafoundation.org/news/2018/10/18/women-in-red-wikiproject/ 
- https://meta.wikimedia.org/wiki/Whose_Knowledge%3F/VisibleWikiWomen 
- https://www.ted.com/talks/jess_wade_a_voice_for_diversity_in_science 

and thanks again to all of the Wikipedia editors and the entire community that is already working so hard to write amazing articles for diverse groups of people. 


# LICENSE
This is licensed under CC-BY-NC, however portions of the dataset are available under separate license terms: text sourced from Wikipedia is licensed under CC-BY-SA.





