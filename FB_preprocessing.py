#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:35:36 2019

@author: stevenalsheimer
"""
from nltk.tokenize import word_tokenize
#text = "Yea, so the first half you have this mysterious setting where it’s filled with mystery. Then the second half we find out exactly what is talking to him, which I think with certain dialogue we can make the reader know it’s AI without telling them. Also, he def shouldn’t be the one who made it. Cause like how does it go from him creating nova to all of a sudden they’re totally sentient and taking notes on whether to kill him or not. And robot kills its creator has been done too many times. Imo it’s way more interesting that hes just a regular software guy that’s now obsolete. Cause with us not knowing who/when these things were created it’s like oh shit maybe they’ve been doing this for years, going to ppl and killing/maybe even replacing them. -you wouldn’t say any of this in the story but it lets the reader imagine that if you don’t explain everything too much"
f= open("input.txt","r")
contents = f.readlines()
User = "name"
#print(contents)
sentence_1 = []
i = 0
for z in range(len(contents)):
    i += 1
    if contents[i] == "User\n":
        string_print = str(contents[i+1])
        quartent = 1
        if contents[i] == "User\n" and contents[i+3] == "User\n":
            string_4 = str(contents[i+4])
            string_4 = string_4.replace("\n","")
            string_print = string_4+", "+str(contents[i+1])
            quartent = 2
            if contents[i] == "User\n" and contents[i+3] == "User\n" and contents[i+6] == "User\n":
                string_4 = str(contents[i+4])
                string_4 = string_4.replace("\n","")
                
                string_7 = str(contents[i+7])
                string_7 = string_7.replace("\n","")
                string_print = string_7+", "+string_4+", "+str(contents[i+1])
                quartent = 3
                if contents[i] == "User\n" and contents[i+3] == "User\n" and contents[i+6] == "User\n" and contents[i+9] == "User\n":
                    string_4 = str(contents[i+4])
                    string_4 = string_4.replace("\n","")
                    
                    string_7 = str(contents[i+7])
                    string_7 = string_7.replace("\n","")
                    
                    string_10 = str(contents[i+10])
                    string_10 = string_10.replace("\n","")
                    string_print = string_10+", "+string_7+", "+string_4+", "+str(contents[i+1])
                    quartent = 4
                    if contents[i] == "User\n" and contents[i+3] == "User\n" and contents[i+6] == "User\n" and contents[i+9] == "User\n" and contents[i+12] == "User\n":
                        string_4 = str(contents[i+4])
                        string_4 = string_4.replace("\n","")
                        
                        string_7 = str(contents[i+7])
                        string_7 = string_7.replace("\n","")
                        
                        string_10 = str(contents[i+10])
                        string_10 = string_10.replace("\n","")
                        
                        string_13 = str(contents[i+13])
                        string_13 = string_13.replace("\n","") 
                        string_print = string_13+", "+string_10+", "+string_7+", "+string_4+", "+str(contents[i+1])
                        quartent = 5

                        if contents[i] == "User\n" and contents[i+3] == "User\n" and contents[i+6] == "User\n" and contents[i+9] == "User\n" and contents[i+12] == "User\n" and contents[i+15] == "User\n":
                            string_4 = str(contents[i+4])
                            string_4 = string_4.replace("\n","")
                            
                            string_7 = str(contents[i+7])
                            string_7 = string_7.replace("\n","")
                            
                            string_10 = str(contents[i+10])
                            string_10 = string_10.replace("\n","")
                            
                            string_13 = str(contents[i+13])
                            string_13 = string_13.replace("\n","") 
                            
                            string_16 = str(contents[i+16])
                            string_16 = string_16.replace("\n","") 
                            string_print = string_16+", "+string_13+", "+string_10+", "+string_7+", "+string_4+", "+str(contents[i+1])
                            quartent = 6
        sentence_1.append(string_print)
        if quartent == 1:
            continue
        if quartent == 2:
            i += 5
        if quartent == 3:
            i += 8
        if quartent == 4:
            i += 11
        if quartent == 5:
            i += 14
        if quartent == 6:
            i += 17
    if i == len(contents)-1:
        break
output_list = []
for message in sentence_1:
    tokens = word_tokenize(message)
    sentence_line = str("")
    for token in tokens:
        string_new = str(token+" ")
        sentence_line = sentence_line + string_new
        if token == ".":#separating sentences since they are separate ideas
            output_list.append(sentence_line)
            sentence_line = str("")
    output_list.append(sentence_line)
#print(output_list)       
for line in output_list:
    print(line)
    continue
outF = open("output.txt", "w")
for line in output_list:
  # write line to output file
  outF.write(line)
  outF.write("\n")
outF.close()
