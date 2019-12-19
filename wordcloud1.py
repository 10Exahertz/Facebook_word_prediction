#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:30:22 2019

@author: stevenalsheimer
"""
# Start with loading all necessary libraries
#from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

text = "do me baby one more more fuck time"
file1 = open("Steven_setup/Rina_train.txt","r+")
#print(file1.read())
text = file1.read()
file1.close()

# Create and generate a word cloud image:
wordcloud = WordCloud(width=1900, height=1080, max_words=400, background_color="black").generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file("Steven_rina.png")
#wordcloud = WordCloud(width=800, height=400).generate(text)