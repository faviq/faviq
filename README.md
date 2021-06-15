# <span style="font-variant:small-caps;">FaVIQ</span> 
  
This repository contains the data and code for our paper:
*Jungsoo Park\*, Sewon Min\*, Jaewoo Kang, Luke Zettlemoyer, Hannaneh Hajishirzi. "FaVIQ: FAct Verification from Information seeking Questions".*


* Checkout the [website](https://faviq.github.io)
* Read the [paper]()
* Download the <span style="font-variant:small-caps;">FaVIQ</span>: [A set]() / [R set]()
* Download the [wikipedia dump file (2019.08.01)]()

## Dataset

### Data

The data consists of **A set** and **R set**.
**A set** is our main dataset, consisting of 26k claims converted from ambiguous questions and their disambiguations.
**R set** is an additional dataset, consisting of 188k claims converted from regular question-answer pairs. Please refer to README for the detailed data format. Visit [Explorer](https://faviq.github.io/explorer.html) to see some samples!


#### Statistics

| Split \ Data  | A set       | R set      |
| ----------- | ----------- | ----------- |
| Train       | 17,008      |140,977      |
| Dev         |  4,260      | 15,566      |
| Test        |  4,688      | 5,877       |

#### Contents

`{train,dev,test}.jsonl` files contains a list of dictionary that represents a single instance, with the following keys

- `id` (string): an identifier for the unique claim.
- `claim` (string): a claim. the claims are all lowercased since the questions from NQ-Open and AmbigQA are all low-cased.
- `label` (string): factuality of the claim which is either 'SUPPORTS' or 'REFUTES'.

As additional resources, we provide `positive_evidence` and `negative_evidence` associated with each claim, that could be useful for training baselines (e.g. training DPR).
- `positive_evidence` (dictionary): the top passage that contains the answer to the original question which is retrieved from querying the original question to TF-IDF.
   - id (string): id of the positive passage mapped to the [wikipedia dump file](#Resource).
   - title (string): title of the positive passage.
   - text (string): text of the positive passage.
- `negative_evidence` (dictionary): the top passage that does not contain the answer to the original question that is retrieved from querying the original question to TF-IDF.
   - id (string): id of the negative passage mapped to the [wikipedia dump file](#Resource).
   - title (string): title of the negative passage.
   - text (string): text of the negative passage.

### Request test data

Test data of our main dataset (A set) is hidden. This is to prevent overfitting on the test data, and also because the test data of AmbigQA, our source data, is hidden.
In order to obtain the test data, please email Jungsoo Park `jungsoopark.1993@gmail.com` and Sewon Min `sewon@cs.washington.edu` with the following information:

* Your name and affiliation
* Purpose of the data (e.g. publication)
* Model predictions and accuracy on the dev data of your model. This is to ensure that you evaluate the model on the test data only after you have finished developing the model on the dev data.

### Resource

- `wikipedia_20190801.jsonl`: wikipedia database in jsonl format; containing the passages (~26M) with keys of passage id, title, and text. We take the plain text and lists provided by [KILT](https://ai.facebook.com/tools/kilt/) and created a collection of passages where each passage has approximately 100 tokens. Note that for training baselines ([FEVER](https://fever.ai/) and <span style="font-variant:small-caps;">FaVIQ</span>) in the paper, we concatenated the title with the passage and regarded it as the passage.

## Citation

If you find the <span style="font-variant:small-caps;">FaVIQ</span> dataset useful, please cite our paper:

```bibtex
@article{ park2021faviq,
    title={ {F}a{VIQ}: Fact Verification from Information seeking Questions },
    author={ Park, Jungsoo and Min, Sewon and Kang, Jaewoo and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2021 }
}
```
Please also make sure to credit and cite the creators of AmbigQA and Natural Questions,
the datasets which we built ours off of:

```bibtex
@inproceedings{ min2020ambigqa,
    title={ {A}mbig{QA}: Answering Ambiguous Open-domain Questions },
    author={ Min, Sewon and Michael, Julian and Hajishirzi, Hannaneh and Zettlemoyer, Luke },
    booktitle={ EMNLP },
    year={ 2020 }
}
```

```bibtex
@article{ kwiatkowski2019natural,
  title={ {N}atural {Q}uestions: a benchmark for question answering research },
  author={ Kwiatkowski, Tom and Palomaki, Jennimaria and Redfield, Olivia and Collins, Michael and Parikh, Ankur and Alberti, Chris and Epstein, Danielle and Polosukhin, Illia and Devlin, Jacob and Lee, Kenton and others },
  journal={ Transactions of the Association for Computational Linguistics },
  year={ 2019 }
}
```

## Contact

Please contact Jungsoo Park `jungsoopark.1993@gmail.com` and Sewon Min `sewon@cs.washington.edu`, or leave Github issues for any questions.
