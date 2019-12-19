#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:23:13 2019

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

detokenize = TreebankWordDetokenizer().detokenize
def generate_sent(model, num_words, random_seed=42):
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)
def generate_sent_text_seed(model, num_words, random_seed="the"):
    content = []
    for token in model.generate(num_words, text_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)
def KN_best_discount(n):
        perp = []
        for dis in range(80,100,2):
            dis = dis/100
            print(dis)
            train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
            model = KneserNeyInterpolated(n, discount = dis) #only order and discount needed, WB only order
            print(n,model)
            model.fit(train_data, padded_sents)
            print(model.vocab)
            vocab_list = []
            for word in model.vocab:
                vocab_list.append(word)
            #print(vocab_list)
            print("value",model. score('<UNK>'))
            #print(generate_sent_text_seed(model, 30, random_seed=['thicc']))

            #print(generate_sent(model, 50, random_seed = 30))
            entropy_fin = 0
            lense = 100
            i = 0
            for z in range(lense):
                #print(contents[i])
                tokenized_test = [list(map(str.lower, word_tokenize(contents[i])))]
                if len(tokenized_test[0]) > 0:
                    for g in range(len(tokenized_test[0])):
                        if tokenized_test[0][g] not in vocab_list:
                            tokenized_test[0][g] = '<UNK>'
                    test_text_pad = list(flatten(pad_both_ends(sent, n) for sent in tokenized_test))
                    test_text_everygram = list(everygrams(test_text_pad, max_len=n))
                    #print(test_text_everygram)
                    #test_data, padded_sents_test = padded_everygram_pipeline(n, tokenized_test)
                    #print(i)
                    #print(model.entropy(test_text_bigram))
                    #print(model.entropy(test_text_everygram))
                    entropy_fin += model.entropy(test_text_everygram)
                i += 1
            print(entropy_fin)
            avg_entr = entropy_fin/lense
            print("perplexity",2**avg_entr)
            perp.append([dis,2**avg_entr])
        import pandas as pd
        DF = pd.DataFrame(perp)
        return DF
def model_iterator(n):
        perp = []
        n = n+1
        for n in range(1,n):
            print(n)
            train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
            #model = MLE(n)
            #model = Laplace(n) #only add-one smoothing here
            #model = Lidstone(0.1,n) #Lidstones second number is Gamma/Alpha/Delta
            #model = WittenBellInterpolated(n)
            model = KneserNeyInterpolated(n, discount = 0.88) #only order and discount needed, WB only order
            print(n,model)
            model.fit(train_data, padded_sents)
            print(model.vocab)
            vocab_list = []
            for word in model.vocab:
                vocab_list.append(word)
            #print(vocab_list)
            print("value",model. score('<UNK>'))
            #print(generate_sent_text_seed(model, 30, random_seed=['thicc']))

            #print(generate_sent(model, 50, random_seed = 30))
            entropy_fin = 0
            lense = 1000
            i = 0
            for z in range(lense):
                #print(contents[i])
                tokenized_test = [list(map(str.lower, word_tokenize(contents[i])))]
                if len(tokenized_test[0]) > 0:
                    for g in range(len(tokenized_test[0])):
                        if tokenized_test[0][g] not in vocab_list:
                            tokenized_test[0][g] = '<UNK>'
                    test_text_pad = list(flatten(pad_both_ends(sent, n) for sent in tokenized_test))
                    test_text_everygram = list(everygrams(test_text_pad, max_len=n))
                    #print(test_text_everygram)
                    #test_data, padded_sents_test = padded_everygram_pipeline(n, tokenized_test)
                    #print(i)
                    #print(model.entropy(test_text_bigram))
                    #print(model.entropy(test_text_everygram))
                    entropy_fin += model.entropy(test_text_everygram)
                i += 1
            print(entropy_fin)
            avg_entr = entropy_fin/lense
            print("perplexity",2**avg_entr)
            perp.append([n,2**avg_entr])
        import pandas as pd
        DF = pd.DataFrame(perp)
        return DF
with io.open('filename.txt', encoding='utf8') as fin:
    text = fin.read()
#with io.open('Steven_setup/Bobby_test.txt', encoding='utf8') as fin:
#    test_text = fin.read()
f= open("Steven_setup/Joshin_test.txt","r")
contents = f.readlines()
#print(test_text)   
# Tokenize the text.
n = 4

tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(text)]
# Preprocess the tokenized text for 3-grams language modelling

train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

##Test data must be padded with this technique, to match PDE-pipline above
#test_text = "dam, i asked a girl who majored in chem and applied for thing ."
#tokenized_test = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(test_text)]
#test_text_pad = list(flatten(pad_both_ends(sent, n) for sent in tokenized_test))
#test_text_everygram = list(everygrams(test_text_pad, max_len=n))
#test_data, padded_sents_test = padded_everygram_pipeline(n, tokenized_test)
#print(test_text_everygram)




#print(KN_best_discount(4))
#print(model_iterator(5))

train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
model = KneserNeyInterpolated(n, discount = 0.75) #only order and discount needed, WB only order
print(n,model)
model.fit(train_data, padded_sents)
print(model.vocab)
#print(generate_sent(model, 30, random_seed=42))

#print("Test Perplexity",model.perplexity(test_text_everygram))

print("--- %s seconds ---" % (time.time() - start_time))

###to save model
import dill as pickle 
#
with open('Steven_Rina_KN0.75_4gram_model.pkl', 'wb') as fout:
    pickle.dump(model, fout)
#with open('kilgariff_ngram_model.pkl', 'rb') as fin:
#    model_loaded = pickle.load(fin)


























