#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:43:24 2019

@author: stevenalsheimer
"""

# Import required libraries
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from nltk import TweetTokenizer
import pandas as pd
tkn = TweetTokenizer()
# Load pre-trained model tokenizer (vocabulary)
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('finetune/GPTS_R')
#tokenizer = GPT2Tokenizer.from_pretrained('finetune/B_S_GPT2/models/117M')
# Encode a text inputs
text = "Yerrr, bro I saw Sam's story. Yo u "
# Load pre-trained model (weights)
#model = GPT2LMHeadModel.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('finetune/GPTS_R')

# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()


# Print the predicted word
def Next_word(text):
    indexed_tokens = tokenizer.encode(text)
    
    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])

    tokens_tensor = tokens_tensor.to()
    model.to()
        
        # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    # Get the predicted next sub-word
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    
    text_token = tkn.tokenize(text)
    predicted_text_token = tkn.tokenize(predicted_text)
    #print(predicted_text)
    if len(predicted_text_token)>len(text_token):
        next_word = predicted_text_token[len(text_token)]
    else:
        next_word = 'nanan'
    return next_word
def document_word_test(document, output_file):
    DF = pd.read_csv(document,delimiter = " - ", header = None)
    print(DF)
    correct_n = 0
    total = len(DF)
    for i in range(total):
        vere = False
        textt = DF.loc[i,0]
        n_word = DF.loc[i,1]
        predicted_word = [Next_word(textt)]
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

#print(Next_word(text))
print(document_word_test("Steven_setup/Steven_Rina_ext_messages.txt", 'Steven_setup/Steven_Rina_Extrinsic_testGPT2.txt'))
#print(next_word(model_loaded, text))