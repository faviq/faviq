# FaVIQ

This repository contains the data and code for the paper
[FaVIQ: Fact Verification from Information seeking Questions]()
by Jungsoo Park, Sewon Min, Jaewoo Kang, Luke Zettlemoyer, Hannaneh Hajishirzi.

* Visit the website [Website]()
* Read the website [paper]()
* Download the dataset: [FaVIQ A set]() / [FaVIQ D set]()

## Contents
1. [Dataset Contents](#dataset-contents)
    * [AmbigNQ format](#ambignq)
    * [NQ-open format](#nq-open)
    * [Additional resources](#additional-resources)
2. [Citation](#citation)

### FaVIQ

FaVIQ consists of the A set and the D set where the former is constructed based on AmbigQA and the latter from NQ.

The A set contains
- train.jsonl ()
- dev.jsonl ()

The D set contains
- train.jsonl ()
- dev.jsonl ()
- test.jsonl ()

`{train,dev,test}.jsonl` files contains a list of dictionary that represents a single instance, with the following keys

- `id` (string): an identifier for the unique claim.
- `claim` (string): a claim. the claims are all lower cased since the questions from NQ-Open and AmbigQA are all low-cased.
- `label` (string): factuality of the claim which is either 'SUPPORTS' or 'REFUTES'.
- `positive_evidence` (dictionary): the top passage that contains the answer to the original question that is retrieved form querying original question (which is used to generate the claim during the data creation process) to TF-IDF.
   - id (string): id of the positive passage.
   - title (string): title of the positive passage.
   - text (string): text of the positive passage.
- `positive_evidence` (dictionary): the top passage that does not contain the answer to the original question that is retrieved form querying original question (which is used to generate the claim during the data creation process) to TF-IDF.
   - id (string): id of the negative passage.
   - title (string): title of the negative passage.
   - text (string): text of the negative passage.

### Resources

- `docs.db`: sqlite db that is consistent with [DrQA](https://github.com/facebookresearch/DrQA); containing plain text only, no disambiguation pages

## Citation

If you find the FaVIQ dataset useful, please cite our paper:

```
@inproceedings{,
    title={FaVIQ: Fact Verification from Information seeking Questions},
    author={ Jungsoo Park, Sewon Min, Jaewoo Kang, Luke Zettlemoyer, Hannaneh Hajishirzi },
    year={2021}
}
```

Please also make sure to credit and cite the creators of AmbigQA and Natural Questions,
the dataset which we built ours off of:

```
@inproceedings{min2020ambigqa,
    title={ {A}mbig{QA}: Answering Ambiguous Open-domain Questions },
    author={ Min, Sewon and Michael, Julian and Hajishirzi, Hannaneh and Zettlemoyer, Luke },
    booktitle={ EMNLP },
    year={2020}
}
```

```
@article{ kwiatkowski2019natural,
  title={ Natural questions: a benchmark for question answering research},
  author={ Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia and Collins, Michael and Parikh, Ankur and Alberti, Chris and Epstein, Danielle and Polosukhin, Illia and Devlin, Jacob and Lee, Kenton and others },
  journal={ Transactions of the Association for Computational Linguistics },
  year={ 2019 }
}
```
