# FaVIQ README

This is the repository documenting the paper
[FaVIQ: Fact Verification from Information seeking Questions]()
by Jungsoo Park, Sewon Min, Jaewoo Kang, Luke Zettlemoyer, and Hannaneh Hajishirzi.

* [Website]()
* Read the [paper]()
* Download the dataset: [FaVIQ A set]() / [FaVIQ D set]()

## Contents
1. [Dataset Contents](#dataset-contents)
    * [AmbigNQ format](#ambignq)
    * [NQ-open format](#nq-open)
    * [Additional resources](#additional-resources)
2. [Citation](#citation)

### AmbigNQ

We provide two distributions of our new dataset AmbigNQ: a `full` version with all annotation metadata
and a `light` version with only inputs and outputs.

The full version contains
- train.json (47M)
- dev.json (17M)

The light version contains
- train_light.json (3.3M)
- dev_light.json (977K)

`train.json` and `dev.json` files contains a list of dictionary that represents a single datapoint, with the following keys

- `id` (string): an identifier for the question, consistent with the original NQ dataset.
- `question` (string): a question. This is identical to the question in the original NQ except we postprocess the string to start uppercase and end with a question mark.
- `annotations` (a list of dictionaries): a list of all acceptable outputs, where each output is a dictionary that represents either a single answer or multiple question-answer pairs.
    - `type`: `singleAnswer` or `multipleQAs`
    - (If `type` is `singleAnswer`) `answer`: a list of strings that are all acceptable answer texts
    - (If `type` is `multipleQAs`) `qaPairs`: a list of dictionaries with `question` and `answer`. `question` is a string, and `answer` is a list of strings that are all acceptable answer texts
- `viewed_doc_titles` (a list of strings): a list of titles of Wikipedia pages viewed by crowdworkers during annotations. This is an underestimate, since Wikipedia pages viewed through hyperlinks are not included. Note that this should not be the input to a system. It is fine to use it as extra supervision, but please keep in mind that it is an underestimate.
- `used_queries` (a list of dictionaries): a list of dictionaries containing the search queries and results that were used by crowdworkers during annotations. Each dictionary contains `query` (a string) and `results` (a list of dictionaries containing `title` and `snippet`). Search results are obtained through the Google Search API restricted to Wikipedia (details in the paper). Note that this should not be the input to a system. It is fine to use it as extra supervision.
- `nq_answer` (a list of strings): the list of annotated answers in the original NQ.
- `nq_doc_title` (string): an associated Wikipedia page title in the original NQ.

`{train|dev}_light.json` are formatted the same way, but only contain `id`, `question` and `annotations`.

### NQ-open

We release our split of NQ-open, for comparison and use as weak supervision:

- nqopen-train.json (9.7M)
- nqopen-dev.json (1.1M)
- nqopen-test.json (489K)

Each file contains a list of dictionaries representing a single datapoint, with the following keys

- `id` (string): an identifier that is consistent with the original NQ.
- `question` (string): a question.
- `answer` (a list of strings): a list of acceptable answer texts.


## Citation

If you find the AmbigQA task or AmbigNQ dataset useful, please cite our paper:
```
@inproceedings{ min2020ambigqa,
    title={ {A}mbig{QA}: Answering Ambiguous Open-domain Questions },
    author={ Min, Sewon and Michael, Julian and Hajishirzi, Hannaneh and Zettlemoyer, Luke },
    booktitle={ EMNLP },
    year={2020}
}
```

Please also make sure to credit and cite the creators of Natural Questions,
the dataset which we built ours off of:
```
@article{ kwiatkowski2019natural,
  title={ Natural questions: a benchmark for question answering research},
  author={ Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia and Collins, Michael and Parikh, Ankur and Alberti, Chris and Epstein, Danielle and Polosukhin, Illia and Devlin, Jacob and Lee, Kenton and others },
  journal={ Transactions of the Association for Computational Linguistics },
  year={ 2019 }
}
```


## Dataset Contents


### Additional resources

- `docs.db`: sqlite db that is consistent with [DrQA](https://github.com/facebookresearch/DrQA); containing plain text only, no disambiguation pages
- `docs-html.db`: sqlite db that is consistent with [DrQA](https://github.com/facebookresearch/DrQA), containing html, no disambiguation pages
- (Coming Soon!) Top 100 Wikipedia passages retrieved from Dense Passage Retrieval
