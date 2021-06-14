# <span style="font-variant:small-caps;">FaVIQ</span>

<div>
<span style="font-variant:small-caps;">FaVIQ</span>
</div>

<div class="tip" markdown="1">Have **fun!**</div>
   
This repository contains the data and code for the paper
[\textsc{FaVIQ}: Fact Verification from Information seeking Questions]()
by Jungsoo Park, Sewon Min, Jaewoo Kang, Luke Zettlemoyer, Hannaneh Hajishirzi.

* Checkout the [website]()
* Read the [paper]()
* Download the \textsc{FaVIQ}: [A set]() / [R set]()
* Download the [wikipedia dump file (2019.08.01)]()

## Dataset

### Data

\textsc{FaVIQ} consists of **A set** and **R set** where the former is constructed based on [AmbigQA](https://nlp.cs.washington.edu/ambigqa/) and the latter is from [Natural Questions](https://ai.google.com/research/NaturalQuestions). We hide the test set from the A set since the test set of AmbigQA (which we build A set upon) is hidden. For obtaining the test set, please contact us via email.

#### Statistics

| Split \ Data  | A set       | D set      |
| ----------- | ----------- | ----------- |
| Train       | 17,008      |140,977      |
| Dev         |  4,260      | 15,566      |
| Test        |  4,688      | 5,877       |

#### Contents

`{train,dev,test}.jsonl` files contains a list of dictionary that represents a single instance, with the following keys

- `id` (string): an identifier for the unique claim.
- `claim` (string): a claim. the claims are all lowercased since the questions from NQ-Open and AmbigQA are all low-cased.
- `label` (string): factuality of the claim which is either 'SUPPORTS' or 'REFUTES'.
- `positive_evidence` (dictionary): the top passage that contains the answer to the original question which is retrieved from querying the original question to TF-IDF.
   - id (string): id of the positive passage mapped to the [wikipedia dump file](#Resource).
   - title (string): title of the positive passage.
   - text (string): text of the positive passage.
- `negative_evidence` (dictionary): the top passage that does not contain the answer to the original question that is retrieved from querying the original question to TF-IDF.
   - id (string): id of the negative passage mapped to the [wikipedia dump file](#Resource).
   - title (string): title of the negative passage.
   - text (string): text of the negative passage.

### Resource

- `wikipedia_20190801.jsonl`: wikipedia database in jsonl format; containing the passages (~26M) with keys of passage id, title, and text. We take the plain text and lists provided by [KILT](https://arxiv.org/abs/2009.02252) and created a collection of passages where each passage has approximately 100 tokens. Note that for training baselines ([FEVER](https://fever.ai/) and \textsc{FaVIQ}) in the paper, we concatenated the title with the passage and regarded it as the passage.

## Citation

If you find the \textsc{FaVIQ} dataset useful, please cite our paper:

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

## Contact

Please contact Jungsoo Park `jungsoopark.1993@gmail.com` or Sewon Min `sewon@cs.washington.edu` if you have any questions.
