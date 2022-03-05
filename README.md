# <span style="font-variant:small-caps;">FaVIQ</span> 
  
This repository contains the data and code for our paper:
*Jungsoo Park\*, Sewon Min\*, Jaewoo Kang, Luke Zettlemoyer, Hannaneh Hajishirzi. "FaVIQ: FAct Verification from Information seeking Questions".*


* Checkout the [website](https://faviq.github.io)
* Read the [paper](https://arxiv.org/pdf/2107.02153.pdf)
* Download the <span style="font-variant:small-caps;">FaVIQ</span>: [A set](https://nlp.cs.washington.edu/ambigqa/data/faviq_a_set_v1.2.zip) / [R set](https://nlp.cs.washington.edu/ambigqa/data/faviq_r_set_v1.2.zip)
* Download the [wikipedia dump file (2019.08.01)](https://nlp.cs.washington.edu/ambigqa/data/wikipedia_20190801.jsonl)
* Download the [wikipedia dump file (2019.08.01) in sqlite DB format](https://drive.google.com/file/d/1g0NA9_j1iDC8_E1zbW9IanzZaC9K1DAc/view?usp=sharing)
* Download the <span style="font-variant:small-caps;">FaVIQ retrieval predictions</span>: [TF_IDF](https://drive.google.com/file/d/1tZrj0y-0FS6T7o4oLfBk8K-Bt_ZUvzTz/view?usp=sharing) / [DPR](https://drive.google.com/file/d/1_h3OpuqvUVEuadH2F0my7W6nJfjXK4vy/view?usp=sharing)
* Download the <span style="font-variant:small-caps;">FaVIQ in closed domain format (retrieval predictions augmented) </span>: [A set](https://drive.google.com/file/d/1_Zg5vw8jcfLHhoLuI6q_j9JLXl_5k4oD/view?usp=sharing) / [R set](https://drive.google.com/file/d/1ca0pokFxcUwS1VcYteveFRthL6kfpLDM/view?usp=sharing)
* Download the <span style="font-variant:small-caps;">FaVIQ in fact correction task format</span>: [A set](https://nlp.cs.washington.edu/ambigqa/data/fact_correction_a_set.zip) / [R set](https://nlp.cs.washington.edu/ambigqa/data/fact_correction_r_set.zip)

**Update (Fev 2022)**: We provide FaVIQ in closed domain format (retrieval predictions augmented)  
**Update (Nov 2021)**: We updated the positive/negative silver evidence for the refute samples. 

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

- `wikipedia_20190801.db`: wikipedia database in sqlite format; containing the passages (~26M) with keys of passage id and values as passages. Note that we concatenated the title with the passage from Wikipedia and use it as the passage.
 
- `FaVIQ retrieval predictions`: TF-IDF and DPR predictions of FaVIQ dataset. The predictions are retrieved passages' indexes from Wikipedia database (sqlite format).

- `FaVIQ in closed domain format (retrieval predictions augmented)`: FaVIQ dataset with retrieval prediction augmented. Top-3 retrieval predictions from TF-IDF and DPR are augmented in the original FaVIQ dataset in text format. The retrieved texts are in `tf_idf_evidence`, `dpr_evidence` as a list format (containing texts).
 
- `fact_correction_{a_set, r_set}.zip`: FaVIQ modified to fact correction task format; containing the instances with keys of input(refuted claim), and output(a list of corrected claims accordingly). We release the resource to aid in future works studying the fact correction, a task recently studied in this [paper](https://arxiv.org/pdf/2012.15788.pdf).
## Citation

If you find the <span style="font-variant:small-caps;">FaVIQ</span> dataset useful, please cite our paper:

```bibtex
@article{ park2021faviq,
    title={ {F}a{VIQ}: Fact Verification from Information seeking Questions },
    author={ Park, Jungsoo and Min, Sewon and Kang, Jaewoo and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2021 },
    journal={arXiv preprint arXiv:2107.02153}
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
