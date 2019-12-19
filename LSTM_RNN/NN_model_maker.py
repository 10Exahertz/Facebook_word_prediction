# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 18:50:06 2019

@author: EPCOT
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:21:29 2019

@author: stevenalsheimer
"""

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding, Activation, TimeDistributed, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
import dill as pickle 
tweet_tokenizer = TweetTokenizer()
f = open("Bobby_train.txt", "r", encoding = 'utf-8')
data_text = f.read()
#print(data_text)
SET = 2
import re

def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    # remove punctuations
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    long_words=[]
    # remove short word
    for i in newString.split():
        if len(i)>=3:                  
            long_words.append(i)
    return (" ".join(long_words)).strip()

# preprocess the text
data_new = text_cleaner(data_text)

def create_seq(text):
    length = 30
    sequences = list()
    for i in range(length, len(text)):
        # select sequence of tokens
        seq = text[i-length:i+1]
        # store
        sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    return sequences

# create sequences   
sequences = create_seq(data_new)

# create a character mapping index
chars = sorted(list(set(data_new)))
mapping = dict((c, i) for i, c in enumerate(chars))

def encode_seq(seq):
    sequences = list()
    for line in seq:
        # integer encode line
        encoded_seq = [mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)
    return sequences

# encode the sequences
sequences = encode_seq(sequences)

from sklearn.model_selection import train_test_split

# vocabulary size
vocab = len(mapping)
sequences = np.array(sequences)
# create X and y
X, y = sequences[:,:-1], sequences[:,-1]
# one hot encode y
y = to_categorical(y, num_classes=vocab)
# create train and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print('Train shape:', X_tr.shape, 'Val shape:', X_val.shape)

model = Sequential()
model.add(Embedding(vocab, 250, input_length=30, trainable=True))
model.add(LSTM(250, return_sequences=True))
#model.add(LSTM(50, return_sequences=True))
model.add(GRU(450, recurrent_dropout=0.1, dropout=0.1))
model.add(Dense(vocab, activation='softmax'))
print(model.summary())

#model = Sequential()
#model.add(Embedding(vocab, 50, input_length=30))
#model.add(LSTM(50, return_sequences=True))
#model.add(LSTM(50, return_sequences=True))
##if use_dropout:
##    model.add(Dropout(0.5))
#model.add(TimeDistributed(Dense(vocab)))
#model.add(Activation('softmax'))
#print(model.summary())
# compile the model
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
# fit the model
model.fit(X_tr, y_tr, epochs=SET, verbose=2, validation_data=(X_val, y_val))

#with open('NN_test.pkl', 'rb') as fin:
#    model_loaded = pickle.load(fin)

# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	# generate a fixed number of characters
	for _ in range(n_chars):
		# encode the characters as integers
		encoded = [mapping[char] for char in in_text]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict character
		yhat = model.predict_classes(encoded, verbose=0)
		# reverse map integer to character
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		# append to input
		in_text += char
	return in_text
inp = 'large armies for '
def next_word_NN(input_, model, length_of_seq, n_chars):
    word_token = tweet_tokenizer.tokenize(inp)
    generated = generate_seq(model, mapping, length_of_seq, inp.lower(), n_chars)
    generate_token = tweet_tokenizer.tokenize(generated)
    return generated, generate_token[len(word_token)]

pred = next_word_NN(inp, model, 30, 20)
print(pred)
with open('NN_test.pkl', 'wb') as fout:
    pickle.dump(model, fout)
    

