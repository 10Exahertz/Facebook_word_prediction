#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 10:17:54 2019

@author: stevenalsheimer
"""
import dill as pickle
import time
import pandas as pd
from nltk.util import pad_sequence
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenize = TreebankWordDetokenizer().detokenize
try: # Use the default NLTK tokenizer.
    from nltk import word_tokenize, sent_tokenize 
    word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0])
except: # Use a naive sentence tokenizer and toktok.
    import re
    from nltk.tokenize import ToktokTokenizer
    sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)
    toktok = ToktokTokenizer()
    word_tokenize = word_tokenize = toktok.tokenize
start_time = time.time()    
    
    
with open('model.pkl', 'rb') as fin:
    model_loaded = pickle.load(fin)
def next_word(model, text):
    tokenized_test = [list(map(str.lower, word_tokenize(sent))) for sent in sent_tokenize(text)]
    context = list(pad_sequence(tokenized_test[0],pad_left=True,left_pad_symbol="<s>",pad_right=False,right_pad_symbol="</s>",n=2))
    best_score = -200
    best_word = '<unk>'
    second_word = '<unk>'
    second_score = -202
    third_word = '<unk>'
    third_score = -203
    for word in model.vocab:
        score  = model.logscore(word, context)
        if score > best_score and word != '<s>' and word != ','and word != ';'and word != ':' and word != "’":
            third_score = second_score
            third_word = second_word
            second_score = best_score
            second_word = best_word
            best_score = score
            best_word = word
    choices = [best_word, second_word,third_word]
    #return "First word",best_score, best_word, "Second word",second_score, second_word, "Third word", third_score, third_word
    return choices      
def generate_sent(model, num_words, random_seed=42):
    content = []
    for token in model.generate(num_words, random_seed=random_seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    return detokenize(content)       
def document_word_test(document, model, output_file):
    DF = pd.read_csv(document,delimiter = " - ", header = None)
    correct_n = 0
    total = len(DF)
    for i in range(total):
        vere = False
        textt = DF.loc[i,0]
        n_word = DF.loc[i,1]
        predicted_word = next_word(model, textt)
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
def Gboard_add(Gboard_file, other_file, output_file):
    DF = pd.read_csv(Gboard_file,delimiter = ",", header = None)
    DF2 = pd.read_csv(other_file,delimiter = ",", header = None)
    print(DF2)
    count = 0
    for i in range(len(DF)):
        one = DF.loc[i,1]
        two = DF2.loc[i,2]
        if one == ' True' or two == True:
            count += 1
        with open(output_file, "a",encoding="utf-8") as myfile:
            strone = str(one)
            strtwo = str(two)
            print(strone + ','+ strtwo+"\n")
            myfile.write(strone + ','+ strtwo+"\n")
    score = count/(len(DF))
    return score
            
        
#text = "the new joker movie is really good u should"
#print(text)
#print(next_word(model_loaded,text))
#print(document_word_test("Steven_setup/Steven_Rina_ext_messages.txt", model_loaded, 'Steven_setup/Steven_Rina_Extrinsic_test0.75.txt'))
#print(next_word(model_loaded, text))
            
print(Gboard_add('Steven_setup/Steven_Rina_ext_GBoard_test.txt','Steven_setup/Steven_Rina_Extrinsic_testGPT2.txt', 'Gboard_GPT2_S_R_test.txt'))

print("--- %s seconds ---" % (time.time() - start_time))
































