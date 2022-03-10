import os
import json
import re
import string
import numpy as np
import pickle as pkl
import sqlite3

from tqdm import tqdm
import torch
import unicodedata

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from utils import *

class FVdata(object):

    def __init__(self, logger, args, data_path, is_training):
        self.data_path = data_path
        self.data_lens = []

        data_paths = self.data_path.split(',') if ',' in data_path else [self.data_path]
            
        self.data = []
        for data_path in data_paths:
            data_chunk = []
            with open(data_path, "r") as f:
                for i in f:
                    data_chunk.append(json.loads(i))
            self.data_lens.append(len(data_chunk))
            self.data += data_chunk
        
        self.is_training = is_training
        self.logger = logger
        self.args = args
        
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.cache = None
        
        self.ids = [data['id'] for data in self.data]
        
        if "test" in self.data_path:
            self.data_type = "test"
        elif "dev" in self.data_path:
            self.data_type = "dev"
        elif "train" in self.data_path:
            self.data_type = "train"
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).lower()

    def decode_batch(self, tokens):
        return [self.decode(_tokens) for _tokens in tokens]

    def load_dataset(self, tokenizer, do_return=False):
        self.tokenizer = tokenizer
        self.logger.info("Preprocessing {} file".format(self.data_type))
        input_ids, attention_mask = None, None
        decoder_input_ids, decoder_attention_mask = None, None
        preprocessed_path = None

        encoder_inputs = []
        targets = []

        retrieved_path = self.args.retrieved_pred_dir + '/{}.jsonl'.format(self.data_type)
        postfix = tokenizer.__class__.__name__.replace("zer", "zed")

        if "," in retrieved_path:
            retrieved_paths = self.args.retrieved_pred_dir.split(",")
            
            shared_retrieved_path = ""
            i = 0
            while np.all([_path1[i]==_path2[i] for _path1 in retrieved_paths for _path2 in retrieved_paths]):
                i += 1
            shared_retrieved_path =  retrieved_paths[0][:i]
            if not shared_retrieved_path.endswith('/'):
                shared_retrieved_path = '/'.join(shared_retrieved_path.split('/')[:-1]) + '/'
            shared_retrieved_path = shared_retrieved_path[:-1]
            self.logger.info("shared retrieved path: %s" % shared_retrieved_path)
            preprocessed_path = shared_retrieved_path + "/{}-{}.jsonl".format(self.data_type, postfix)
        else:
            preprocessed_path = retrieved_path.replace(".jsonl", "-{}.jsonl".format(postfix))

        self.logger.info("Preprocessed path: %s" % preprocessed_path)

        if os.path.exists(preprocessed_path):
            with open(preprocessed_path, "rb") as f:
                preprocessed = pkl.load(f)
                input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = preprocessed[:4]
                if len(preprocessed)==5:
                    self.targets = preprocessed[4]
        else:
            if self.args.topk_passages !=0:
                if ',' in self.args.retrieved_pred_dir:
                    retrieved_pred_dirs = self.args.retrieved_pred_dir.split(',')
                    retrieved_pred_dirs = [retrieved_pred_dirs[k] for k in range(len(retrieved_pred_dirs))]
                    retrieved_idxs = []
                    for retrieved_pred_dir in retrieved_pred_dirs:
                        with open(retrieved_pred_dir + '/{}.jsonl'.format(self.data_type), "r") as f:
                            data_chunk = json.load(f)
                        retrieved_idxs += data_chunk
                else:
                    with open(self.args.retrieved_pred_dir + '/{}.jsonl'.format(self.data_type)) as f:
                        retrieved_idxs = json.load(f)
                                    
                assert len(self.data)==len(retrieved_idxs)

            if self.args.topk_passages != 0:
                db = DocDB(self.args.wiki_db_file)
                retrieved_passages = []
                for instance_retrieved_idxs in tqdm(retrieved_idxs):
                    instance_retrieved_passages = ''
                    for topk_passages_pred in instance_retrieved_idxs[:self.args.topk_passages]:
                        text = db.get_doc_text(topk_passages_pred)
                        title = db.get_doc_title(topk_passages_pred)
                        text = title + ' ' + text
                        if self.args.do_lowercase:
                            text = text.lower()
                        instance_retrieved_passages += ' <SEP> ' + text
                    retrieved_passages.append(instance_retrieved_passages.strip())
                
                assert len(retrieved_passages) == len(self.data)

            for i, data in enumerate(self.data):
                claim = data['claim'].lower() if self.args.do_lowercase else data['claim']
                instance = claim + ' ' + retrieved_passages[i] if self.args.topk_passages != 0 else claim
                encoder_inputs.append(instance)
                targets.append(map_fever_label_to_single_token(data['label']))

        if input_ids is None and attention_mask is None and decoder_input_ids is None \
                and decoder_attention_mask is None:
            self.logger.info("Tokenize data...")
            if self.is_training:
                assert len(encoder_inputs) == len(targets)
            self.targets = targets
            self.encoder_inputs = encoder_inputs

            if self.args.do_lowercase:
                targets = [t.lower() for t in targets]

            encoder_inputs = tokenizer.batch_encode_plus(encoder_inputs,
                                                         padding='max_length',
                                                         truncation=True,
                                                         max_length=self.args.max_input_length,
                                                         return_overflowing_tokens=True)

            targets = tokenizer.batch_encode_plus(targets,
                                                  padding='max_length',
                                                  truncation=True,
                                                  max_length=self.args.max_output_length,
                                                  return_overflowing_tokens=True)

            input_ids, attention_mask = encoder_inputs["input_ids"], encoder_inputs["attention_mask"]
            decoder_input_ids, decoder_attention_mask = targets["input_ids"], targets["attention_mask"]

            if preprocessed_path is not None:
                with open(preprocessed_path, "wb") as f:
                    pkl.dump([input_ids, attention_mask,
                              decoder_input_ids, decoder_attention_mask,
                              self.targets], f)

            n_truncated = 0
            for i, num_truncated_tokens in enumerate(encoder_inputs['overflowing_tokens']):
                if len(num_truncated_tokens) >= 1:
                    n_truncated += 1

            self.logger.info('{}/{} truncated in encoder inputs'.format(int(n_truncated), int(len(input_ids))))

            n_truncated = 0
            for i, num_truncated_tokens in enumerate(targets['overflowing_tokens']):
                if len(num_truncated_tokens) >= 1:
                    n_truncated += 1

            self.logger.info('{}/{} truncated in targets'.format(int(n_truncated), int(len(decoder_input_ids))))

        assert len(input_ids)==len(attention_mask)==len(decoder_input_ids)==len(decoder_attention_mask)

        if self.args.append_another_bos and 'bart' in self.args.model_name:
            input_ids = [[0] + _ids[:-1] for _ids in input_ids]
            attention_mask = [[1] + mask[:-1] for mask in attention_mask]
            decoder_input_ids = [[0] + _ids[:-1] for _ids in decoder_input_ids]
            decoder_attention_mask = [[1] + mask[:-1] for mask in decoder_attention_mask]

        if self.is_training:
            indices = np.random.permutation(range(len(input_ids)))
            input_ids = [input_ids[i] for i in indices]
            attention_mask = [attention_mask[i] for i in indices]
            decoder_input_ids = [decoder_input_ids[i] for i in indices]
            decoder_attention_mask = [decoder_attention_mask[i] for i in indices]
            self.logger.info("Shuffled training data")

        self.dataset = fvdataset(input_ids, attention_mask,
                                   decoder_input_ids, decoder_attention_mask,
                                   in_metadata=None, out_metadata=None,
                                   is_training=self.is_training)
        self.logger.info("Loaded {} examples from {} data".format(len(self.dataset), self.data_type))

        if do_return:
            return self.dataset

    def load_dataloader(self, do_return=False):
        self.dataloader = fvdataloader(self.args, self.dataset, self.is_training)
        if do_return:
            return self.dataloader

    def evaluate(self, predictions):

        assert len(predictions)==len(self.targets), (len(predictions), len(self.targets))
        ems = []
        for (prediction, target) in zip(predictions, self.targets):
            ems.append(get_exact_match(prediction, target))

        if ',' in self.data_path:
            ems_1 = ems[:self.data_lens[0]]
            ems_2 = ems[self.data_lens[0]:]
            assert len(ems_2) == self.data_lens[1]
            return [np.mean(ems_1), np.mean(ems_2), np.mean(ems)]
            
        return np.mean(ems)

    def save_predictions(self, predictions):
        assert len(predictions)==len(self.targets), (len(predictions), len(self.targets))

        prediction_dict = {dp["id"]:map_single_token_to_fever_label(prediction) for dp, prediction in zip(self.data, predictions)}
        save_path = os.path.join(self.args.output_dir, "{}_predictions.json".format(self.args.prefix))
        with open(save_path, "w") as f:
            json.dump(prediction_dict, f)
        self.logger.info("Saved prediction in {}".format(save_path))

class fvdataset(Dataset):
    def __init__(self,
                 input_ids, attention_mask,
                 decoder_input_ids, decoder_attention_mask,
                 in_metadata=None, out_metadata=None,
                 is_training=False):
        self.input_ids = torch.LongTensor(input_ids)
        self.attention_mask = torch.LongTensor(attention_mask)
        self.decoder_input_ids = torch.LongTensor(decoder_input_ids)
        self.decoder_attention_mask = torch.LongTensor(decoder_attention_mask)
        self.in_metadata = list(zip(range(len(input_ids)), range(1, 1+len(input_ids)))) \
            if in_metadata is None else in_metadata
        self.out_metadata = list(zip(range(len(decoder_input_ids)), range(1, 1+len(decoder_input_ids)))) \
            if out_metadata is None else out_metadata
        self.is_training = is_training

        assert len(self.input_ids)==len(self.attention_mask)==self.in_metadata[-1][-1]
        assert len(self.decoder_input_ids)==len(self.decoder_attention_mask)==self.out_metadata[-1][-1]

    def __len__(self):
        return len(self.in_metadata)

    def __getitem__(self, idx):
        if not self.is_training:
            idx = self.in_metadata[idx][0]
            return self.input_ids[idx], self.attention_mask[idx], self.decoder_input_ids[idx], self.decoder_attention_mask[idx]

        in_idx = np.random.choice(range(*self.in_metadata[idx]))
        out_idx = np.random.choice(range(*self.out_metadata[idx]))
        return self.input_ids[in_idx], self.attention_mask[in_idx], \
            self.decoder_input_ids[out_idx], self.decoder_attention_mask[out_idx]

class fvdataloader(DataLoader):

    def __init__(self, args, dataset, is_training):
        if is_training:
            sampler=RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler=SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(fvdataloader, self).__init__(dataset, sampler=sampler, batch_size=batch_size)
