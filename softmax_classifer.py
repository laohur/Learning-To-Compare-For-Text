import os
import re
from time import time
import random

from torch import optim

import Constants
import sys
import torch
import json
from Util import *
from Comparator import *
from time import time
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import numpy as np
from Encoder import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
FEATURE_DIM = 128
RELATION_DIM = 100
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 5
BATCH_NUM_PER_CLASS = 15
EPISODE = 100000
TEST_EPISODE = 10
LEARNING_RATE = 0.01
HIDDEN_UNIT = 10


def main():
    path = "data/toutiao.txt"
    doc = open(path, "r", encoding="utf-8").read().splitlines()
    counter = count_doc(doc)
    # with open("data/counter.json", "w", encoding="utf-8") as f:
    #     json.dump(counter, f, ensure_ascii=False)
    word2index, index2word = counter2dict(counter=counter, min_freq=3)
    print(word2index, index2word)

    # weights = load_weights(word2index, "../../data/wordvec/merge_sgns_bigram_char300.txt")
    # weights = load_weights(word2index, "../../data/wordvec/merge_sgns_bigram_char300.txt","data/myembed.txt")
    weights = load_weights(word2index, "../../data/char_vector.txt")
    config = {
        "BATCH_NUM_PER_CLASS": BATCH_NUM_PER_CLASS,
        "SAMPLE_NUM_PER_CLASS": SAMPLE_NUM_PER_CLASS,
        "CLASS_NUM": CLASS_NUM,
        "TEST_EPISODE": TEST_EPISODE,
        "FEATURE_DIM": FEATURE_DIM,
        "emb_dim": 300,
        "lstm_hid_dim": 64,
        "d_a": 64,
        "r": 64,
        "max_len": 10,
        "n_classes": 5,
        "dropout": 0.1,
        "use_pretrained_embeddings": True,
        "embeddings": weights,
        "epochs": 200,
        "vocab_size": len(word2index)
    }
    config["word2index"] = word2index
    config['index2word'] = index2word

    dict_data = {}
    for line in doc:
        tokens = line.split("\t")
        if len(tokens) != 2:
            print(line)
            continue
        y, x = tokens[0], tokens[1]
        if y in dict_data:
            dict_data[y].append(x)
        else:
            dict_data[y] = [x]
    keys = list(dict_data.keys())
    # labels = {key: i for i, key in keys}
    labels = {}
    for i in range(len(keys)):
        labels[keys[i]] = i;
    # for line in doc:
    #     name = line.split("\t")[0]
    #     if name not in labels:
    #         labels[name] = len(labels)
    print("标签类别数量", len(labels))
    dict_data1 = {}
    for k, v in labels.items():
        dict_data1[v] = dict_data[k]

    encoder_model = StructuredSelfAttention(config).to(device)
    classer = Classifier(128, len(labels)).to(device)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(classer.parameters(), lr=0.01)
    loss = 0.0
    for epoch in range(100000):
        # for k, v in dict_data1.items():
        #     target = torch.full(size=(v.size()[0],1), fill_value=k,dtype=torch.long)
        #     target=target.squeeze().to(device)  # 真是变态，多维赋值，一维输入
        #     classer.zero_grad()
        #     x = encoder_model(v.to(device))
        #     log_probs = classer(x)
        #     loss = loss_function(log_probs, target)
        #     loss.backward()
        #     optimizer.step()
        # print(loss.item())

        samples, sample_labels, batches, batch_labels, labels = \
            few_data(dict_data1, n_class=config["CLASS_NUM"], n_support=config["SAMPLE_NUM_PER_CLASS"], n_batch=config["BATCH_NUM_PER_CLASS"],
                     word2index=config['word2index'], index2word=config['index2word'], max_len=config["max_len"])
        for i in range(sample_labels.size(0)):
            sample_labels[i]=labels[sample_labels[i]]
        for i in range(batch_labels.size()[0]):
            batch_labels[i]=labels[ batch_labels[i]]

        classer.zero_grad()
        x = encoder_model(samples.to(device))
        log_probs = classer(x)
        loss = loss_function(log_probs, sample_labels.to(device))
        loss.backward()
        optimizer.step()
        print(loss.item())

        classer.zero_grad()
        x = encoder_model(batches.to(device))
        log_probs = classer(x)
        loss = loss_function(log_probs, batch_labels.to(device))
        loss.backward()
        optimizer.step()
        print(loss.item())

# 不够随机


if __name__ == "__main__":
    t0 = time()
    main()
    print("耗时", time() - t0)
