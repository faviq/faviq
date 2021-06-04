# FaVIQ

This repository contains the data and code for the paper
[FaVIQ: Fact Verification from Information seeking Questions]()
by Jungsoo Park, Sewon Min, Jaewoo Kang, Luke Zettlemoyer, Hannaneh Hajishirzi.

* Checkout the [website]()
* Read the [paper]()
* Download the FaVIQ: [A set]() / [D set]()
* Download the [wikipedia dump file (2019.08.01)]()

## Dataset

### Data

FaVIQ consists of **A set** and **D set** where the former is constructed based on AmbigQA and the latter is from Natural Questions. We hide the test set from the A set since the test set of AmbigQA (which we build A set upon) is hidden.

The A set contains
- train.jsonl ()
- dev.jsonl ()

The D set contains
- train.jsonl ()
- dev.jsonl ()
- test.jsonl ()

`{train,dev,test}.jsonl` files contains a list of dictionary that represents a single instance, with the following keys

- `id` (string): an identifier for the unique claim.
- `claim` (string): a claim. the claims are all lowercased since the questions from NQ-Open and AmbigQA are all low-cased.
- `label` (string): factuality of the claim which is either 'SUPPORTS' or 'REFUTES'.
- `positive_evidence` (dictionary): the top passage that contains the answer to the original question that is retrieved from querying the original question to TF-IDF.
   - id (string): id of the positive passage mapped to the [wikipedia dump file](#Resource).
   - title (string): title of the positive passage.
   - text (string): text of the positive passage.
- `negative_evidence` (dictionary): the top passage that does not contain the answer to the original question that is retrieved from querying the original question to TF-IDF.
   - id (string): id of the negative passage mapped to the [wikipedia dump file](#Resource).
   - title (string): title of the negative passage.
   - text (string): text of the negative passage.

### Resource

- `wikipedia_20190801.jsonl`: wikipedia database in jsonl format; containing the passages (~26M) with passage id, title, and text.

## Citation

If you find the FaVIQ dataset useful, please cite our paper:

```bibtex
@article{,
    title={FaVIQ: Fact Verification from Information seeking Questions},
    author={ Jungsoo Park, Sewon Min, Jaewoo Kang, Luke Zettlemoyer, Hannaneh Hajishirzi },
    year={2021}
}
```
Please also make sure to credit and cite the creators of AmbigQA and Natural Questions,
the datasets which we built ours off of:

```bibtex
@inproceedings{min2020ambigqa,
    title={ {A}mbig{QA}: Answering Ambiguous Open-domain Questions },
    author={ Min, Sewon and Michael, Julian and Hajishirzi, Hannaneh and Zettlemoyer, Luke },
    booktitle={ EMNLP },
    year={2020}
}
```

```bibtex
@article{ kwiatkowski2019natural,
  title={ Natural questions: a benchmark for question answering research},
  author={ Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia and Collins, Michael and Parikh, Ankur and Alberti, Chris and Epstein, Danielle and Polosukhin, Illia and Devlin, Jacob and Lee, Kenton and others },
  journal={ Transactions of the Association for Computational Linguistics },
  year={ 2019 }
}
```
