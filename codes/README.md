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

## Wikipedia Passage Retrieval

For training models on the fact verification task, you will need the Wikipedia knowledge source and relevant passages retrieved from it.
You can download the Wikipedia passage file in sqlite3 format and retrieved passages' indexes (using TF_IDF or DPR) by running the code below.
This will download and unzip files into `./data` directory.

```bash
bash download_wiki.sh
```

Note that we concatenated the title with the passage and regarded it as the passage.

## Train

The following example fine-tunes DPR + BART-Large model on FaVIQ (A set and R set).
Please make sure that the FaVIQ dataset (folders of `faviq_r_set`, `faviq_a_set`) is in `data` directory. 
The model that we report on the paper was trained with `train_batch_size=32` using eight 32G GPUs.

```bash
CUDA=0,1,2,3,4,5,6,7
TRAIN_DIR=data/faviq_r_set/train.jsonl,data/faviq_a_set/train.jsonl
DEV_DIR=data/faviq_r_set/dev.jsonl,data/faviq_a_set/dev.jsonl
OUTPUT_DIR=out/bart_large_dpr
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
@article{ park2021faviq,
    title={ {F}a{VIQ}: Fact Verification from Information seeking Questions },
    author={ Park, Jungsoo and Min, Sewon and Kang, Jaewoo and Zettlemoyer, Luke and Hajishirzi, Hannaneh },
    year={ 2021 },
    journal={arXiv preprint arXiv:2107.02153}
}
```
