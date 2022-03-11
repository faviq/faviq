# Baselines on FaVIQ

This section reproduces the baseline experiments from the original paper (Section 4.1).

## Requirements

```bash
$ conda create -n fv python=3.6
$ conda activate fv
$ conda install pytorch=1.7.1 cudatoolkit=11.1 -c pytorch
$ pip install transformers==4.3.0
```
Note that Pytorch has to be installed depending on the version of CUDA.

## Wikipedia passage retrieval

For training models on the fact verification task, you will need the Wikipedia knowledge source and relevant passages retrieved from it.
You can download the Wikipedia passage file in sqlite3 format and retrieved passages' indexes (using TF_IDF or DPR) by running the code below.
This will download and unzip files into `./data` directory.

```bash
bash download_wiki.sh
```

## Quick Guideline for using retrieval data
First, download the Wikipedia file and TF-IDF/DPR retrieval predictions through `bash download_wiki.sh`. Then, you can use the following code to use retrieved passages for each claim.
Note that, if you are using our code, this feature is already incorporated in
the [data.py](https://github.com/faviq/faviq/blob/main/codes/data.py#L116-L125) so you can skip this part.
This code is only for those who will not use our code but want to use our retrieval data.

```python
import json
from tqdm import tqdm
from utils import DocDB

# top k passages you want to use (up to 100)
top_k_passages = 100

# path to wiki_db file
wiki_db_file = "data/wikipedia_20190801.db"

# path to retrieval prediction file
# - replace `dpr` to `tfidf` if you want to use TF-IDF retrieval instead of DPR
# - replace `faviq_a_set` to `faviq_r_set` for the R set instead of A set
# - replace `dev.jsonl` to `train.jsonl` or `test.jsonl` for the train/test data instead of the dev data
retrieval_prediction_file = "data/dpr/faviq_a_set/dev.jsonl"

with open(retrieval_prediction_file, "r") as f:
    retrieved_idxs = json.load(f)

db = DocDB(wiki_db_file)

retrieval_documents = [] # the i-th item of the list contains top-K retrived passages for the i-th claim
for instance_retrieved_idxs in tqdm(retrieved_idxs):
    instance_retrieval_documents = []
    for topk_passages_pred in instance_retrieved_idxs[:top_k_passages]:
        text = db.get_doc_text(topk_passages_pred)
        title = db.get_doc_title(topk_passages_pred)
        instance_retrieval_documents.append({'title':title, 'passage':text})
    retrieval_documents.append(instance_retrieval_documents)
```

## Train

The following example fine-tunes DPR + BART-Large model on FaVIQ (A set and R set).
Please make sure that the FaVIQ dataset (folders of `faviq_r_set`, `faviq_a_set`) is in `data` directory.
The model that we report on the paper was trained with `train_batch_size=32` using eight 32G GPUs.

```bash
CUDA=0,1,2,3,4,5,6,7
TRAIN_DIR=data/faviq_r_set/train.jsonl,data/faviq_a_set/train.jsonl
DEV_DIR=data/faviq_r_set/dev.jsonl,data/faviq_a_set/dev.jsonl
RETRIEVAL_DIR=data/dpr/faviq_r_set/,data/dpr/faviq_a_set/
WIKI_DB=data/wikipedia_20190801.db
OUTPUT_DIR=out/bart_large_dpr

CUDA_VISIBLE_DEVICES=${CUDA} python cli.py \
	--train_file ${TRAIN_DIR} \
	--dev_file ${DEV_DIR} \
	--output_dir ${OUTPUT_DIR} \
	--do_train \
	--train_batch_size 32 \
	--predict_batch_size 32 \
	--max_input_length 1024 \
	--max_output_length 5 \
	--model_name 'facebook/bart-large' \
	--retrieved_pred_dir ${RETRIEVAL_DIR} \
    	--wiki_db_file ${WIKI_DB}
```

## Evaluation

We provide checkpoints of trained models on FaVIQ.
Download the checkpoints by running the code below which will download and unzip models' checkpoints into `./out` directory.

```bash
bash download_ckpt.sh
```

The following example evaluates our trained DPR + BART-Large model on FaVIQ (R set only).

```bash

CUDA=0
DEV_DIR=data/faviq_r_set/dev.jsonl
TEST_DIR=data/faviq_r_set/test.jsonl
OUTPUT_DIR=out/bart_large_dpr
RETRIEVAL_DIR=data/dpr/faviq_r_set/
WIKI_DB=data/wikipedia_20190801.db

CUDA_VISIBLE_DEVICES=${CUDA} python cli.py \
	--dev_file ${DEV_DIR} \
	--test_file ${TEST_DIR} \
	--output_dir ${OUTPUT_DIR} \
	--do_predict \
	--predict_batch_size 32 \
	--max_input_length 1024 \
	--max_output_length 5 \
	--model_name 'facebook/bart-large' \
	--retrieved_pred_dir ${RETRIEVAL_DIR} \
    	--wiki_db_file ${WIKI_DB}
```

## Citations
```bibtex
@inproceedings{ park2022faviq,
    title={ {F}a{VIQ}: Fact Verification from Information seeking Questions },
    author={ Park, Jungsoo and Min, Sewon and Kang, Jaewoo and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2022 },
    booktitle={ ACL },
}
```
