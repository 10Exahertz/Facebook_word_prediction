#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:18:56 2019

@author: stevenalsheimer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:00:56 2019

@author: stevenalsheimer
"""
import os
import csv
import re
import time
start_time = time.time()
#lists = []
#f = open("rina_created_messages.txt", "r")
#for line in f:
#    lists.append(line)
#print(lists)
def BOW_features_dict(vocab_file):
    lines = open(vocab_file, 'r', encoding = 'utf8')
    dict_of_features = {}
    index = 0
    V_list = []
    for line in lines:
        V_list.append(line.strip().split('\n'))
    for i in range(len(V_list)):
        for token in V_list[i]:
            if token not in dict_of_features:
                dict_of_features[token] = index
                index += 1
    return dict_of_features

#print(BOW_features_dict('Practice/practice.vocab'))
#print(BOW_features_dict('movie-review-HW2/aclimdb/imdb.vocab'))

def BOW_training_vector_doc(File1_name, File2_name,class1_name,class2_name,output_file_dir, vocab_dir):
    fd = open(output_file_dir, "w+")
    fd.close()
    features_dict = BOW_features_dict(vocab_dir)
    F1 = open(File1_name, "r")
    F2 = open(File2_name, "r")
    for line in F1:
        s = line
        out = re.findall(r"[\w']+|[!?]", s)
        listt = []
        for word in out:
            listt.append(word.lower())
        BOW_vec = (len(features_dict)+1)*[0]
        BOW_vec[0] = class1_name
        for word in listt:
            if word in features_dict:
                index = features_dict[word]
                index = index +1
                BOW_vec[index] += 1
        with open(output_file_dir, 'a') as csvFile:
            writer = csv.writer(csvFile,escapechar=' ',quoting = csv.QUOTE_NONE)
            writer.writerow(BOW_vec)      
    for line in F2:

        s = line
        out = re.findall(r"[\w']+|[!?]", s)
        listt = []
        for word in out:
            listt.append(word.lower())
        BOW_vec = (len(features_dict)+1)*[0]
        BOW_vec[0] = class2_name
        for word in listt:
            if word in features_dict:
                index = (features_dict[word]+1)
                BOW_vec[index] += 1
        with open(output_file_dir, 'a') as csvFile:
            writer = csv.writer(csvFile,escapechar=' ',quoting = csv.QUOTE_NONE)
            writer.writerow(BOW_vec)
    return output_file_dir

BOW_training_vector_doc("Joshin_created_messages.txt","Rina_created_messages.txt", "Joshin", "Rina", "Megadoc_NLP_create_U23.txt", "vocab.vocab")
BOW_training_vector_doc("Joshin_train.txt","Rina_train.txt", "Joshin", "Rina", "megadoc_train_U23.txt", "vocab.vocab")
print("--- %s seconds ---" % (time.time() - start_time))






















