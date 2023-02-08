import argparse
import os
import torch
import numpy as np
import datasets

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

from tabulate import tabulate
import nltk
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, load_metric
from sklearn.preprocessing import LabelEncoder
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('-f')

parser.add_argument("--model_name", default="bart-base", type=str,
                    help="in [bart-base, bart-large, bart-base-cnn, bart-large-cnn]")

parser.add_argument("--model_dir", default="bart-base-v1", type=str)

parser.add_argument("--save_dir", default="dir", type=str)

parser.add_argument("--save_name", default="bart-base-v1", type=str)

parser.add_argument("--test_name", default="tacred", type=str, 
                    help='in [tacred, tacredrev]')

args = parser.parse_args()


args.model_dir = '/save model/seq2se2_tacred_saved_model/' + args.model_dir


args.save_dir = '/training log/seq2seq_tacred/'

args.save_name = args.save_dir + args.test_name + '-' + args.save_name + '.txt'

def get_f1(key, prediction, none_id):
    correct_by_relation = ((key == prediction) & (prediction != none_id)).astype(np.int32).sum()
    guessed_by_relation = (prediction != none_id).astype(np.int32).sum()
    gold_by_relation = (key != none_id).astype(np.int32).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro

def batch_tokenize_preprocess(batch, tokenizer, max_source_length, max_target_length):

    source, target = batch["text"], batch["summary"]
    source_tokenized = tokenizer(
        source, padding="max_length", truncation=True, max_length=max_source_length
    )
    target_tokenized = tokenizer(
        target, padding="max_length", truncation=True, max_length=max_target_length
    )

    batch = {k: v for k, v in source_tokenized.items()}
    # Ignore padding in the loss
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in l]
        for l in target_tokenized["input_ids"]
    ]
    return batch

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

#generate datasets for seq2seq models
train_file = '/summary_tacred/tacred_train_augmented.csv'
dev_file = '/summary_tacred/tacred_dev_augmented.csv'
test_file = '/summary_tacred/tacred_test_augmented.csv'

data_files = {}

data_files["train"] = train_file
extension = train_file.split(".")[-1]

if args.test_name == 'tacred':
    data_files["test"] = test_file
    extension = test_file.split(".")[-1]
else:
    data_files["test"] = test_2_file
    extension = test_2_file.split(".")[-1]

raw_datasets = load_dataset(extension, data_files=data_files)

nltk.download("punkt", quiet=True)
metric = datasets.load_metric("rouge")

train_data_txt = raw_datasets['train']
dev_data_txt = raw_datasets['dev']
test_data_txt = raw_datasets['test']


#initialize pre-trained model

print('using BART-based model: %s'%args.model_name)
tokenizer = AutoTokenizer.from_pretrained('facebook/' + args.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/' + args.model_name)

encoder_max_length = 256  
decoder_max_length = 32

#preprocess datasets
train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)

dev_data = test_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=dev_data_txt.column_names,
)

test_data = test_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=test_data_txt.column_names,
)


training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=10,  # demo
    per_device_eval_batch_size=50,
#     per_gpu_train_batch_size = 10,
#     per_gpu_eval_batch_size = 10,
    # learning_rate=3e-05,
    warmup_steps=500,
    weight_decay=0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=50,
    save_total_limit=3,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def get_score():
    
    pred = trainer.predict(
            test_data, metric_key_prefix="predict", max_length=32, num_beams=4
        )

    label_seq2seq = []
    pred_seq2seq = []
    
    print('start generate pred_seq2seq')

    for k, d in tqdm(enumerate(test_data)):

        tt = d['labels']
        temp_label = tokenizer.decode(tt[:np.sum(np.array(tt) != -100)], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        temp_pred = tokenizer.decode(pred[0][k], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        label_seq2seq.append(temp_label)
        pred_seq2seq.append(temp_pred)

    print('*****finish predict*****')
    def func(x):
        if x in label_seq2seq:
            return x
        else:
            return 'no relaion'
        
    pred_seq2seq = [func(x) for x in pred_seq2seq]
    
    df = pd.DataFrame()

    df['label'] = label_seq2seq
    df['pred'] = pred_seq2seq
    print('*****finish df*****')
    
    lb = LabelEncoder()
    lb.fit(list(df['label']))
    label_lb = lb.transform(list(df['label']))
    pred_lb = lb.transform(list(df['pred']))

    print('*****finish encode*****')

    P, R, F1 = get_f1(label_lb, pred_lb, lb.transform(['no relation'])[0])
    
    return P, R, F1


last_file = None
exisited_id = []

while True:

    file_list = os.listdir(args.model_dir)
    print(file_list)
    
    if len(file_list) != 0:
        
        if last_file is None:
            
            new_file = file_list[-1]
        
            print('new_file :%s'%new_file)
            print('last_file :%s'%last_file)

            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir+ '/' + new_file)

                print('*****do evalation v1*****')
                trainer = Seq2SeqTrainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train_data,
                    eval_dataset=test_data,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                )
                print('*****start predicting v1*****')
                P, R, F1 = get_score()
                print('{}:, P:{}, R:{}, F1 :{} \n'.format(new_file, P, R, F1))

                with open(args.save_name, 'a') as f: 
                    f.write('{}:, P:{}, R:{}, F1 :{} \n'.format(new_file, P, R, F1))

                last_file = new_file
                exisited_id.append(last_file)

            except:
                print('incomplete saved model, back to the loop v1')
                print('sleep 5s...')
                time.sleep(5)

        else:
            for x in exisited_id:
                try:
                    file_list.remove(x)
                except:
                    continue
            if len(file_list) != 0:

                new_file = file_list[-1]
                print('new_file :%s'%new_file)
                print('last_file :%s'%last_file)
                
                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir+ '/' + new_file)
                    
                    print('*****do evaluation v2*****')
                
                    trainer = Seq2SeqTrainer(
                        model=model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=train_data,
                        eval_dataset=test_data,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics,
                    )
                    print('*****start predicting v2*****')
                    P, R, F1 = get_score()
                    print('{}:, P:{}, R:{}, F1 :{} \n'.format(new_file, P, R, F1))

                    with open(args.save_name, 'a') as f: 
                        f.write('{}:, P:{}, R:{}, F1 :{} \n'.format(new_file, P, R, F1))

                    last_file = new_file
                    exisited_id.append(last_file)

                except:
                    print('incomplete saved model, back to the loop v2' )
                    print('sleep 5s...')
                    time.sleep(5)
    
            else:
                print('There is no new saved model.')
                print('sleep 60s...')
                time.sleep(60)

    else:
        print('There is no saved model yet.')
        print('sleep 60s...')
        time.sleep(60)


