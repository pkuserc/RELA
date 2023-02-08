import argparse
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
from datasets import load_dataset, load_metric
from tabulate import tabulate
import nltk
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, load_metric
from sklearn.preprocessing import LabelEncoder
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f')

parser.add_argument("--model_name", default="bart-base", type=str,
                    help="in [bart-base, bart-large, bart-base-cnn, bart-large-cnn, t5-small, t5-base, t5-large, t5-3b, t5-11b]")
parser.add_argument("--save_dir", default="dir", type=str)
parser.add_argument("--save_name", default="bart-base-v1", type=str)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--lr", default=1e-5, type=float)
parser.add_argument("--warmup_ratio", default=0.2, type=float)
parser.add_argument("--label_smoothing_factor", default=0.1, type=float)
parser.add_argument("--saved_steps", default=2000, type=int)

args = parser.parse_args()


args.save_dir = 'save model/seq2se2_tacred_saved_model/' + args.save_name

tokenizer = AutoTokenizer.from_pretrained('facebook/'+args.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/'+args.model_name)

encoder_max_length = 256  
decoder_max_length = 32

train_file = 'summary_tacred/tacred_train_augmented.csv'
test_file = 'summary_tacred/tacred_test_augmented.csv'

data_files = {}

data_files["train"] = train_file
extension = train_file.split(".")[-1]

data_files["validation"] = test_file
extension = test_file.split(".")[-1]

# data_files["test"] = test_file
# extension = test_file.split(".")[-1]
raw_datasets = load_dataset(extension, data_files=data_files)

train_data_txt = raw_datasets['train']
validation_data_txt = raw_datasets['validation']
print(train_data_txt[0])
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


train_data = train_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=train_data_txt.column_names,
)

validation_data = validation_data_txt.map(
    lambda batch: batch_tokenize_preprocess(
        batch, tokenizer, encoder_max_length, decoder_max_length
    ),
    batched=True,
    remove_columns=validation_data_txt.column_names,
)

nltk.download("punkt", quiet=True)
metric = datasets.load_metric("rouge")

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


training_args = Seq2SeqTrainingArguments(
    output_dir=args.save_dir,
    num_train_epochs=args.epochs,  # demo
    do_train=True,
    do_eval=False,
    per_device_train_batch_size=args.batch_size,  # demo
    per_device_eval_batch_size=50,
#     per_gpu_train_batch_size = 10,
#     per_gpu_eval_batch_size = 10,
    save_strategy = 'steps',
    learning_rate=args.lr,
    warmup_ratio=args.warmup_ratio,
    weight_decay=0.1,
    label_smoothing_factor=args.label_smoothing_factor,
    predict_with_generate=True,
    save_steps=args.saved_steps,
    logging_dir="logs",
    logging_steps=200,
    save_total_limit=1,
    report_to = 'none',
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=validation_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()