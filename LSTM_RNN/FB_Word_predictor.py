#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:11:44 2019

@author: stevenalsheimer
"""

#from nltk.util import everygrams
#from nltk.lm.preprocessing import pad_both_ends
#from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline

import io #codecs
from nltk.tokenize.treebank import TreebankWordDetokenizer

##How to install the various language models
from nltk.lm.models import KneserNeyInterpolated
import time

#from nltk.util import bigrams
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

detokenize = TreebankWordDetokenizer().detokenize
def generate_sent(model, num_words):
    content = []
    for token in model.generate(num_words):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)
def generate_sent_text_seed(model, num_words, words):
    content = []
    for token in model.generate(num_words, text_seed=words, random_seed = 3):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)


with io.open('Joshin_train.txt', encoding='utf8') as fin:
    text = fin.read()
n = 4

tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(text)]

train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
model = KneserNeyInterpolated(n, discount = 0.88) #only order and discount needed, WB only order

#print(DF)
print(n,model)
model.fit(train_data, padded_sents)
print(model.vocab)
vocab_list = []
for word in model.vocab:
    vocab_list.append(word)
#print(vocab_list)
print("value",model. score('<UNK>'))
#print(generate_sent_text_seed(model, 60, words=['<s>','thicc']))
start_time = time.time()
for i in range(1000):
    print(i)
    with open("Joshin_created_messages.txt", "a",encoding="utf-8") as myfile:
        myfile.write(generate_sent(model, 20)+"\n")

print("--- %s seconds ---" % (time.time() - start_time))








