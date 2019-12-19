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
from itertools import groupby
import pandas as pd

tokenize = TweetTokenizer()
save_dir = 'save' # directory where model is stored
seq_length = 5 # sequence length
words_number = 30 #number of words to generate
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
model = load_model('Keras_word_based_NN/my_model_Steven_bobby_s_5.h5')

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
            if word in vocab:
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
#    seed_sentence_tok = tokenize.tokinize(seed_sentence)
#    for word in seed_sentence_tok:
#        if word in vocab:
    seed_sentence = seed_sentence.lower()
    options = []
    wordssss = []
    for i in range(250):
        text = generate(seed_sentence)
        text_token = tokenize.tokenize(text)
        next_token = text_token[seq_length]
        #print(next_token)
        if next_token != "," or next_token != '.' or next_token != ':' or next_token != ';':
            options.append(next_token)
    choices = []
    for word in options:
        countN = options.count(word)
        if word not in wordssss:
            choices.append([countN, word])
            wordssss.append(word)
    choicesDF = pd.DataFrame(choices)
    FinalChoicesDF = choicesDF.nlargest(3, 0)
    FinalChoicesDF = FinalChoicesDF.reset_index()
    #print(FinalChoicesDF)
    FinalChoices = []
    for i in range(len(FinalChoicesDF)):
        #print(FinalChoicesDF.loc[i, 1])
        FinalChoices.append(FinalChoicesDF.loc[i, 1])
    return FinalChoices
def document_word_test(document, output_file):
    DF = pd.read_csv(document,delimiter = " - ", header = None)
    print(DF)
    correct_n = 0
    total = len(DF)
    for i in range(total):
        vere = False
        textt = DF.loc[i,0]
        n_word = DF.loc[i,1]
        predicted_word = three_choices(textt)
        for choice in predicted_word:
            if choice == n_word:
                correct_n += 1
                vere = True
        if vere == False:
            print(i)
        pred_str = str(predicted_word)
        vere_str = str(vere)
        n_wordStr = str(n_word)
        with open(output_file, "a",encoding="utf-8") as myfile:
            myfile.write(n_wordStr + ','+ pred_str+',' + vere_str+"\n")
    score = correct_n/total
    return score
    
print(generate("yerr bro i'm"))
#print(three_choices("Yerr bro i'm "))
#print(document_word_test('Steven_Rina_ext_messages.txt', 'Steven__Rina_ext_NN_test.txt'))