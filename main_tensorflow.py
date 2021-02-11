# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 19:51:08 2021

@author: MARIJA
"""

import json
from pathlib import Path
from transformers import DistilBertForQuestionAnswering
import torch
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import AdamW
import os
import re
import string
import numpy as np
import keras
#import tensorflow as tf
#import tensorflow_hub as hub
#from tensorflow import keras
#from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from transformers import TFDistilBertForQuestionAnswering



#print(train.head(4))
def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


#print(train_contexts[0])
#print(train_questions[0])
#print(train_answers[0])
def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length

        # if end position is None, the 'char_to_token' function points to the space before the correct token - > add + 1
        if end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] + 1)
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})







if __name__ == '__main__':
    train_path = keras.utils.get_file("train.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")
    eval_path = keras.utils.get_file("eval.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json")
    train = pd.read_json(train_path)
    valid = pd.read_json(eval_path)
    
    train_contexts, train_questions, train_answers = read_squad(train_path)
    val_contexts, val_questions, val_answers = read_squad(eval_path)
    
    add_end_idx(train_answers, train_contexts)
    add_end_idx(val_answers, val_contexts)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
    
    add_token_positions(train_encodings, train_answers)
    add_token_positions(val_encodings, val_answers)
    train_dataset = tf.data.Dataset.from_tensor_slices((
    {key: train_encodings[key] for key in ['input_ids', 'attention_mask']},
    {key: train_encodings[key] for key in ['start_positions', 'end_positions']}
          ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
    {key: val_encodings[key] for key in ['input_ids', 'attention_mask']},
    {key: val_encodings[key] for key in ['start_positions', 'end_positions']}
       ))

    
    
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model = TFDistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
   
    #model.to(device)
    #model.train()
   

    #train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    #optim = AdamW(model.parameters(), lr=5e-5)
   
# Keras will expect a tuple when dealing with labels
    train_dataset = train_dataset.map(lambda x, y: (x, (y['start_positions'], y['end_positions'])))

# Keras will assign a separate loss for each output and add them together. So we'll just use the standard CE loss
# instead of using the built-in model.compute_loss, which expects a dict of outputs and averages the two terms.
# Note that this means the loss will be 2x of when using TFTrainer since we're adding instead of averaging them.
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.distilbert.return_dict = False # if using 🤗 Transformers >3.02, make sure outputs are tuples

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=loss) # can also use any keras loss fn
    model.fit(train_dataset.shuffle(1000).batch(16), epochs=2, batch_size=16)
    model.save_weights("./weights.h5")
   
   