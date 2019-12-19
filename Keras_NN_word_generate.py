# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Sat Dec 14 17:00:54 2019

@author: EPCOT
"""

'''generate text.
'''



print("import librairies...")

from keras.models import load_model
import numpy as np
import os
import collections
from six.moves import cPickle
from nltk.tokenize import TweetTokenizer
tokenize = TweetTokenizer()
save_dir = 'save' # directory where model is stored
seq_length = 15 # sequence length
words_number = 400 #number of words to generate
#seed_sentences = "large Armies for " #sentence for seed generation

#load vocabulary
print("loading vocabulary...")
#vocab_file = os.path.join(save_dir, "words_vocab.pkl")

with open('Keras_word_based_NN/words_vocab.pkl', 'rb') as f:
        words, vocab, vocabulary_inv = cPickle.load(f)

vocab_size = len(words)
print("vocab_size", vocab_size)

# load the model
print("loading model...")
model = load_model('Keras_word_based_NN/my_model.h5')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#initiate sentences
def generate(seed_sentences):
    generated = ''
    sentence = []
    for i in range (seq_length):
        sentence.append("a")

    seed = seed_sentences.split()

    for i in range(len(seed)):
        sentence[seq_length-i-1]=seed[len(seed)-i-1]

    generated += ' '.join(sentence)
    #print('Generating text with the following seed: "' ,' '.join(sentence) + '"')

    #print ()

    #generate the text
    for i in range(words_number):
        #create the vector
        x = np.zeros((1, seq_length, vocab_size))
        for t, word in enumerate(sentence):
            x[0, t, vocab[word]] = 1.

        #calculate next word
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 1)
        next_word = vocabulary_inv[next_index]

        #add the next word to the text
        generated += " " + next_word
        # shift the sentence by one, and and the next word at its end
        sentence = sentence[1:] + [next_word]

    return generated
def three_choices(seed_sentence):
    seed_sentence = seed_sentence.lower()
    options = []
    while len(options) < 3:
        text = generate(seed_sentence)
        text_token = tokenize.tokenize(text)
        next_token = text_token[seq_length]
        if next_token not in options:
            options.append(next_token)
    return options
    
    
#print(generate("yerr "))
print(three_choices("Yerrr "))