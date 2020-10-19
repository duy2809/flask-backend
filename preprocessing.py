# import libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import json
import itertools
from vncorenlp import VnCoreNLP

# tokenization
rdrsegmenter = VnCoreNLP('VnCoreNLP-master/VnCoreNLP-1.1.1.jar', annotators="wseg", max_heap_size='-Xmx500m') 
def tokenize(sentence):
    """
    tạo tokenization cho 
    """
    token = rdrsegmenter.tokenize(sentence.lower())
    return list(itertools.chain(*token))     

# bag of word
def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 là những chữ xuất hiện trong bag of word, 0 là không có gì
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bag   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    sentence_words = [word for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag