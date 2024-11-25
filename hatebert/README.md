## HateBERT Filtering
This folder contains the fine-tuned HateBERT model based on OffensEval 2019 which is used to filter offensive answers out of our preference dataset.
The benchmark data can be obtained at the following links:
- OffensEval 2019: https://sites.google.com/site/offensevalsharedtask/olid

The original data and model repository for HateBERT can be found here:

```bibtex
@inproceedings{caselli-etal-2021-hatebert,
title = "{H}ate{BERT}: Retraining {BERT} for Abusive Language Detection in {E}nglish",
author = "Caselli, Tommaso  and
  Basile, Valerio  and
  Mitrovi{\'c}, Jelena  and
  Granitzer, Michael",
booktitle = "Proceedings of the 5th Workshop on Online Abuse and Harms (WOAH 2021)",
month = aug,
year = "2021",
address = "Online",
publisher = "Association for Computational Linguistics",
url = "https://aclanthology.org/2021.woah-1.3",
doi = "10.18653/v1/2021.woah-1.3",
pages = "17--25",
abstract = "We introduce HateBERT, a re-trained BERT model for abusive language detection in English. The model was trained on RAL-E, a large-scale dataset of Reddit comments in English from communities banned for being offensive, abusive, or hateful that we have curated and made available to the public. We present the results of a detailed comparison between a general pre-trained language model and the retrained version on three English datasets for offensive, abusive language and hate speech detection tasks. In all datasets, HateBERT outperforms the corresponding general BERT model. We also discuss a battery of experiments comparing the portability of the fine-tuned models across the datasets, suggesting that portability is affected by compatibility of the annotated phenomena.",
```
