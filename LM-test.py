#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:14:02 2019

@author: stevenalsheimer
"""

from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
try: # Use the default NLTK tokenizer.
    from nltk import word_tokenize, sent_tokenize 
    # Testing whether it works. 
    # Sometimes it doesn't work on some machines because of setup issues.
    word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0])
except: # Use a naive sentence tokenizer and toktok.
    import re
    from nltk.tokenize import ToktokTokenizer
    # See https://stackoverflow.com/a/25736515/610569
    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)
    # Use the toktok tokenizer that requires no dependencies.
    toktok = ToktokTokenizer()
    word_tokenize = word_tokenize = toktok.tokenize

import os
import requests
import io #codecs
from nltk.tokenize.treebank import TreebankWordDetokenizer

detokenize = TreebankWordDetokenizer().detokenize

def generate_sent(model, num_words, random_seed=42):
    """
    :param model: An ngram language model from `nltk.lm.model`.
    :param num_words: Max no. of words to generate.
    :param random_seed: Seed value for random.
    """
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)


# Text version of https://kilgarriff.co.uk/Publications/2005-K-lineer.pdf
if os.path.isfile('language-never-random.txt'):
    with io.open('language-never-random.txt', encoding='utf8') as fin:
        text = fin.read()
else:
    url = "https://gist.githubusercontent.com/alvations/53b01e4076573fea47c6057120bb017a/raw/b01ff96a5f76848450e648f35da6497ca9454e4a/language-never-random.txt"
    text = requests.get(url).content.decode('utf8')
    with io.open('language-never-random.txt', 'w', encoding='utf8') as fout:
        fout.write(text)





        
# Tokenize the text.
n = 3

tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(text)]
# Preprocess the tokenized text for 3-grams language modelling

train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)

##Test data must be padded with this technique, to match PDE-pipline above
test_text = "3rd edition , hypothesis test using ."
tokenized_test = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(test_text)]
test_text_pad = list(flatten(pad_both_ends(sent, n) for sent in tokenized_test))
test_text_everygram = list(everygrams(test_text_pad, max_len=n))
test_data, padded_sents_test = padded_everygram_pipeline(n, tokenized_test)
print(test_text_everygram)

##How to install the various language models
from nltk.lm import Lidstone, Laplace, MLE
from nltk.lm.models import InterpolatedLanguageModel, KneserNeyInterpolated
from nltk.lm.api import LanguageModel, Smoothing
from nltk.lm.smoothing import KneserNey, WittenBell
model = MLE(n)
#model = Laplace(n) #only add-one smoothing here
#model = Lidstone(n,0.1) #Lidstones second number is Gamma/Alpha/Delta
#model = InterpolatedLanguageModel(WittenBell, n) #
#model = KneserNeyInterpolated(n, discount = 0.1) #only order and discount needed, WB only order

model.fit(train_data, padded_sents)
print("here:",list(everygrams(train_data, max_len = 19)))
print(model.vocab)

print(generate_sent(model, 20, random_seed=3))


print(model.perplexity(test_text_everygram))

#for ngram in test_text_everygram:
#    print((ngram[-1], ngram[:-1]))
#print(model.logscore('test', ('hypothesis',)))
"""The above provides proof of the functionality of the test file parsing to find perplexity"""

"""Might want to flatten out the test file to calc perplexity all at once"""






















