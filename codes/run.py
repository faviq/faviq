import os
import json
import numpy as np

import torch
import torch.nn.functional as F

from collections import Counter

from transformers import Adafactor, AdamW, get_linear_schedule_with_warmup
from transformers import BartTokenizer, BartForConditionalGeneration

from data import FVdata
from tqdm import tqdm

def run(args, logger):

    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(["<SEP>"])

    if args.dev_file:
        dev_data = FVdata(logger, args, args.dev_file, False)
        dev_data.load_dataset(tokenizer)
        dev_data.load_dataloader()

    if args.test_file:
        test_data = FVdata(logger, args, args.test_file, False)
        test_data.load_dataset(tokenizer)
        test_data.load_dataloader()

    if args.do_train:
        train_data = FVdata(logger, args, args.train_file, True)
        train_data.load_dataset(tokenizer)
        train_data.load_dataloader()

        if args.checkpoint is not None:
            def convert_to_single_gpu(state_dict):
                def _convert(key):
                    if key.startswith('module.'):
                        return key[7:]
                    return key
                return {_convert(key):value for key, value in state_dict.items()}
            model = BartForConditionalGeneration.from_pretrained(args.model_name)
            model.resize_token_embeddings(len(tokenizer))
            ckpt = convert_to_single_gpu(torch.load(args.checkpoint))
            model.load_state_dict(ckpt)
        else:
            model = BartForConditionalGeneration.from_pretrained(args.model_name)
            model.resize_token_embeddings(len(tokenizer))

        if args.n_gpu>1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        scheduler = None
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=100000)
        train(args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_train or args.do_predict:
        checkpoint = os.path.join(args.output_dir, 'best-model.pt')
        def convert_to_single_gpu(state_dict):
            def _convert(key):
                if key.startswith('module.'):
                    return key[7:]
                return key
            return {_convert(key):value for key, value in state_dict.items()}

        model = BartForConditionalGeneration.from_pretrained(args.model_name)
        model.resize_token_embeddings(len(tokenizer))
        ckpt = convert_to_single_gpu(torch.load(checkpoint))
        model.load_state_dict(ckpt)

        logger.info("Loading checkpoint from {}".format(checkpoint))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()

        def print_result(split, data):
            ems = inference(args, model, data, save_predictions=True)
            if ',' not in data.data_path:
                logger.info("Accuracy on the %s dataset: %.2f%%" % (split, np.mean(ems)*100))
            else:
                logger.info("Accuracy on the %s dataset [all: %.2f%% %s: %.2f%% %s: %.2f%%]" % \
                            (split, ems[2]*100, "first", ems[0]*100, "second", ems[1]*100))

        if args.dev_file:
            print_result("dev", dev_data)
        if args.test_file:
            print_result("test", test_data)


def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    best_accuracy = -1
    stop_training=False

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in train_data.dataloader:
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]

            outputs = model(input_ids=batch[0], attention_mask=batch[1],
                            labels=batch[2], decoder_attention_mask=batch[3])

            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training=True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()   
                model.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            if global_step % args.eval_period == 0:
                model.eval()
                curr_em = inference(args, model if args.n_gpu==1 else model.module, dev_data)
                if ',' not in dev_data.data_path:
                    logger.info("Step %d Train loss %.2f accuracy %.2f%% on epoch=%d" % (
                        global_step,
                        np.mean(train_losses),
                        curr_em*100.0,
                        epoch))
                else:
                    logger.info("Step %d Train loss %.2f Accuracy [overall %.2f%% first datset %.2f%% second dataset %.2f%%] epoch=%d" % (
                            global_step,
                            np.mean(train_losses),
                            curr_em[2]*100.0,
                            curr_em[0]*100.0,
                            curr_em[1]*100.0,
                            epoch))
                    curr_em = curr_em[2]
                        
                train_losses = []
                if best_accuracy < curr_em:
                    model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
                    torch.save(model_state_dict, os.path.join(args.output_dir, "best-model.pt"))
                    logger.info("Saving model with best accuracy: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" % \
                            (best_accuracy*100.0, curr_em*100.0, epoch, global_step))
                    best_accuracy = curr_em
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                model.train()
        if stop_training:
            break


def inference(args, model, dev_data, save_predictions=False):
    prediction_counter = Counter()
    predictions = []
    logits = []
    for i, batch in enumerate(tqdm(dev_data.dataloader)):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 max_length=dev_data.args.max_output_length,
                                 )
        for input_, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            prediction_counter[pred] += 1
            predictions.append(pred)
 
    dev_data.logger.info(str(prediction_counter))
    ems = dev_data.evaluate(predictions)

    if save_predictions:
        dev_data.save_predictions(predictions)

    return ems




