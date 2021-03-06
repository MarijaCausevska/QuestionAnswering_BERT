# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:22:53 2021

@author: MARIJA
"""

import json
import pandas as pd
import os
import re
import string
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tokenizers import BertWordPieceTokenizer
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ============================================= PREPARING DATASET ======================================================
class Sample:
    def __init__(self, question, context, start_char_idx=None, answer_text=None, all_answers=None):
        self.question = question
        self.context = context
        self.start_char_idx = start_char_idx
        self.answer_text = answer_text
        self.all_answers = all_answers
        self.skip = False
        self.start_token_idx = -1
        self.end_token_idx = -1

    def preprocess(self):
        context = " ".join(str(self.context).split())
        question = " ".join(str(self.question).split())
        tokenized_context = tokenizer.encode(context)
        tokenized_question = tokenizer.encode(question)
        if self.answer_text is not None:
            answer = " ".join(str(self.answer_text).split())
            end_char_idx = self.start_char_idx + len(answer)
            if end_char_idx >= len(context):
                self.skip = True
                return
            is_char_in_ans = [0] * len(context)
            for idx in range(self.start_char_idx, end_char_idx):
                is_char_in_ans[idx] = 1
            ans_token_idx = []
            for idx, (start, end) in enumerate(tokenized_context.offsets):
                if sum(is_char_in_ans[start:end]) > 0:
                    ans_token_idx.append(idx)
            if len(ans_token_idx) == 0:
                self.skip = True
                return
            self.start_token_idx = ans_token_idx[0]
            self.end_token_idx = ans_token_idx[-1]
        input_ids = tokenized_context.ids + tokenized_question.ids[1:]
        token_type_ids = [0] * len(tokenized_context.ids) + [1] * len(tokenized_question.ids[1:])
        attention_mask = [1] * len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)
        elif padding_length < 0:
            self.skip = True
            return
        self.input_word_ids = input_ids
        self.input_type_ids = token_type_ids
        self.input_mask = attention_mask
        self.context_token_to_char = tokenized_context.offsets
        
def create_squad_examples(raw_data):
    squad_examples = []
    for item in raw_data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                if "answers" in qa:
                    answer_text = qa["answers"][0]["text"]
                    all_answers = [_["text"] for _ in qa["answers"]]
                    start_char_idx = qa["answers"][0]["answer_start"]
                    squad_eg = Sample(question, context, start_char_idx, answer_text, all_answers)
                else:
                    squad_eg = Sample(question, context)
                squad_eg.preprocess()
                squad_examples.append(squad_eg)
    return squad_examples
def create_inputs_targets(squad_examples):
    dataset_dict = {
        "input_word_ids": [],
        "input_type_ids": [],
        "input_mask": [],
        "start_token_idx": [],
        "end_token_idx": [],
    }
    for item in squad_examples:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])
    x = [dataset_dict["input_word_ids"],
         dataset_dict["input_mask"],
         dataset_dict["input_type_ids"]]
    y = [dataset_dict["start_token_idx"], dataset_dict["end_token_idx"]]
    return x, y

# =================================================== TRAINING =========================================================


class ValidationCallback(keras.callbacks.Callback):

    def normalize_text(self, text):
        text = text.lower()
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        text = re.sub(regex, " ", text)
        text = " ".join(text.split())
        return text

    def __init__(self, x_eval, y_eval):
        self.x_eval = x_eval
        self.y_eval = y_eval

    def on_epoch_end(self, epoch, logs=None):
        pred_start, pred_end = self.model.predict(self.x_eval)
        count = 0
        eval_examples_no_skip = [_ for _ in eval_squad_examples if _.skip == False]
        for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
            squad_eg = eval_examples_no_skip[idx]
            offsets = squad_eg.context_token_to_char
            start = np.argmax(start)
            end = np.argmax(end)
            if start >= len(offsets):
                continue
            pred_char_start = offsets[start][0]
            if end < len(offsets):
                pred_char_end = offsets[end][1]
                pred_ans = squad_eg.context[pred_char_start:pred_char_end]
            else:
                pred_ans = squad_eg.context[pred_char_start:]
            normalized_pred_ans = self.normalize_text(pred_ans)
            normalized_true_ans = [self.normalize_text(_) for _ in squad_eg.all_answers]
            if normalized_pred_ans in normalized_true_ans:
                count += 1
        acc = count / len(self.y_eval[0])
        print(f"\nepoch={epoch + 1}, exact match score={acc:.2f}")
        
if __name__ == '__main__':
    train_path = "./train-v1.1.json"
    eval_path = "./val-v1.1.json"
    raw_train_data = pd.read_json(train_path)
    raw_eval_data = pd.read_json(eval_path)
    #train_path = keras.utils.get_file("train.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json")
    #eval_path = keras.utils.get_file("eval.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")
    #with open(train_path) as f: raw_train_data = json.load(f)
    #with open(eval_path) as f: raw_eval_data = json.load(f)
    max_seq_length = 384
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
    bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy().decode("utf-8")
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = BertWordPieceTokenizer(vocab=vocab_file, lowercase=True)
    train_squad_examples = create_squad_examples(raw_train_data)
    x_train, y_train = create_inputs_targets(train_squad_examples)
    print(f"{len(train_squad_examples)} training points created.")
    eval_squad_examples = create_squad_examples(raw_eval_data)
    x_eval, y_eval = create_inputs_targets(eval_squad_examples)
    print(f"{len(eval_squad_examples)} evaluation points created.")
    start_logits = layers.Dense(1, name="start_logit", use_bias=False)(sequence_output)
    start_logits = layers.Flatten()(start_logits)
    end_logits = layers.Dense(1, name="end_logit", use_bias=False)(sequence_output)
    end_logits = layers.Flatten()(end_logits)
    start_probs = layers.Activation(keras.activations.softmax)(start_logits)
    end_probs = layers.Activation(keras.activations.softmax)(end_logits)
    model = keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=[start_probs, end_probs])
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    model.compile(optimizer=optimizer, loss=[loss, loss])
    model.summary()
    model.fit(x_train, y_train, epochs=2, batch_size=8, callbacks=[ValidationCallback(x_eval, y_eval)])
    model.save_weights("./weights.h5")
    
    
    # ==================================================== TESTING =========================================================
data = {"data":
    [
        {"title": "Project Apollo",
         "paragraphs": [
             {
                 "context":  "Nikola Tesla was a Serbian-American inventor, electrical engineer, mechanical engineer,"
                             "and futurist best known for his contributions to the design of the modern alternating"
                             "current (AC) electricity supply system."
                             "Born and raised in the Austrian Empire, Tesla studied engineering and physics in the 1870s"
                             "without receiving a degree, gaining practical experience in the early 1880s working in"
                             "telephony and at Continental Edison in the new electric power industry. In 1884 he emigrated"
                             "to the United States, where he became a naturalized citizen."
                              "He worked for a short time at the Edison Machine Works in New York City before he struck"
                              "out on his own. With the help of partners to finance and market his ideas, Tesla set up"
                              "laboratories and companies in New York to develop a range of electrical and mechanical devices."
                              "His alternating current (AC) induction motor and related polyphase AC patents,"
                             "licensed by Westinghouse Electric in 1888, earned him a considerable amount of money"
                             "and became the cornerstone of the polyphase system which that company eventually marketed.",
                 "qas": [
                     {"question": "Where was Nikola Tesla born and raised?",
                      "id": "Q1"
                      },
                     {"question": "Who was Nikola Tesla?",
                      "id": "Q2"
                      },
                     {"question": "When was he studied engineering and physics?",
                      "id": "Q3"
                      },
                     {"question": "What was he studied?",
                      "id": "Q4"
                      },
                     {"question": "Where was Nikola Tesla worked?",
                      "id": "Q5"
                      }
                 ]}]}]}

test_samples = create_squad_examples(data)
x_test, _ = create_inputs_targets(test_samples)
pred_start, pred_end = model.predict(x_test)
for idx, (start, end) in enumerate(zip(pred_start, pred_end)):
    test_sample = test_samples[idx]
    offsets = test_sample.context_token_to_char
    start = np.argmax(start)
    end = np.argmax(end)
    pred_ans = None
    if start >= len(offsets):
        continue
    pred_char_start = offsets[start][0]
    if end < len(offsets):
        pred_ans = test_sample.context[pred_char_start:offsets[end][1]]
    else:
        pred_ans = test_sample.context[pred_char_start:]
    print("Q: " + test_sample.question)
    print("A: " + pred_ans)
