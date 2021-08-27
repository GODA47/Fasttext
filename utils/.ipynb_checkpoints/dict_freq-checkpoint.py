import os
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
import re
import nltk
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords 

from .replace_dict import rep
from .config import *

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def make_freq_dict(tokenized, freq_dict):
    for w in tokenized:
        if not w in freq_dict:
            freq_dict[w] = 0
        freq_dict[w] += 1
    return freq_dict

def get_freq_dict(dataset):
    freq_dict = {}
    for w in tqdm(dataset):
        text = word_tokenize(w['input'])
        tknn = []
        tk = []
        for idx in range(len(text)):
            temp = text[idx].replace("\\n", " ")
            if '//' not in temp and 'http' not in temp:
                tk.append(temp)
        for w in tk:
            rep_esc = map(re.escape, rep)
            pattern = re.compile("|".join(rep_esc))        
            te = pattern.sub(lambda match: rep[match.group(0)], w)
            te = re.sub('[^0-9a-zA-Z]+', " ", te)
            if stopword and te.lower() in stop_words:
                continue
            te = lem.lemmatize(te,'v')
    #         if te[0].isupper():
    #             te = "^" + te
            tknn.append(te)
        freq_dict = make_freq_dict(word_tokenize(" ".join(tknn).lower()), freq_dict)
    return freq_dict