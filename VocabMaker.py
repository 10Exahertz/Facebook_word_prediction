#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:48:13 2019

@author: stevenalsheimer
"""


from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline

import os
import requests
import io #codecs
from nltk.tokenize.treebank import TreebankWordDetokenizer

##How to install the various language models
from nltk.lm import Lidstone, MLE
from nltk.lm.models import InterpolatedLanguageModel, KneserNeyInterpolated, Laplace, WittenBellInterpolated
from nltk.lm.api import LanguageModel, Smoothing
from nltk.lm.smoothing import KneserNey, WittenBell
import time

from nltk.util import bigrams
start_time = time.time()

try: # Use the default NLTK tokenizer.
    from nltk import word_tokenize, sent_tokenize 
    word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0])
except: # Use a naive sentence tokenizer and toktok.
    import re
    from nltk.tokenize import ToktokTokenizer
    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)
    toktok = ToktokTokenizer()
    word_tokenize = word_tokenize = toktok.tokenize
n=1 
with io.open('Steven_setup/Bobby_train.txt', encoding='utf8') as fin:
    text = fin.read()
tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(text)]
# Preprocess the tokenized text for 3-grams language modelling

train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)


train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
model = MLE(n) #only order and discount needed, WB only order
print(n,model)
model.fit(train_data, padded_sents)
print(model.vocab)

vocab_1 = []
for word in model.vocab:
    vocab_1.append(word)
#print(vocab_1)
with io.open('Steven_setup/Rina_train.txt', encoding='utf8') as fin:
    text = fin.read()
tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(text)]
# Preprocess the tokenized text for 3-grams language modelling

train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)


train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
model = MLE(n) #only order and discount needed, WB only order
print(n,model)
model.fit(train_data, padded_sents)
print(model.vocab)
for word in model.vocab:
    if word not in vocab_1:
        vocab_1.append(word)
print(len(vocab_1))
outF = open("Vocab.vocab", "w")
for line in vocab_1:
  # write line to output file
  outF.write(line)
  outF.write("\n")
outF.close()