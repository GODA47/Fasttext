import os
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
import re
import nltk
import math
import torch
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords 

from .replace_dict import rep
from .config import *

lem = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

dev = 'cpu'
if torch.cuda.is_available():
    dev = "cuda:0"

class subwordhash:
    def __init__(self, dataset):
        word_num, hash_len, sample_len = self.average_subword_num(dataset)
        self.word_num = word_num
        self.max_hash = hash_len
        self.max_sample = sample_len
        
    def __call__(self, word):
        return self.subword_hashes(word, max_hash_num = self.max_hash)
    
    def fnv1a(self, txt, K = int(2e6)):
        # 64 bit fnv-1a
        txt = bytes(txt, 'utf-8')
        hval = 0xcbf29ce484222325
        fnv_prime = 0x100000001b3
        for c in txt:
            hval = hval ^ c
            hval = (hval * fnv_prime) % K
        return hval + 1        

    def subword_hashes(self, word, max_hash_num = None, get_len = False):
        sub_hash = []
        tword = '<' + word + '>'
        sub_hash.append(self.fnv1a(tword))
        for n in range(3,7):
            for i in range(len(tword)-n+1):
                sub_hash.append(self.fnv1a(tword[i:i+n]))
                if len(sub_hash) == max_hash_num:
                    return np.array(sub_hash[:max_hash_num])
        if max_hash_num is not None:
            sub_hash.extend([0]*(max_hash_num - len(sub_hash)))
        if get_len:
            return len(sub_hash)
        return np.array(sub_hash)

    def average_subword_num(self, dataset):
        max_sample_len = 0
        hash_len_dist = {}
        len_dist = {}
        for sample in tqdm(dataset):
#             print('AAAAAA')
            tokens = sample[:]
            if len(tokens) not in len_dist:
                len_dist[len(tokens)] = 0
            len_dist[len(tokens)] += 1
            max_sample_len = max(max_sample_len, len(tokens))
            
        for L in list(len_dist):
            hash_len_dist[self.subword_hashes('a'*L, get_len = True)] = len_dist[L]
        
        max_hash_len = max(list(hash_len_dist))
        
        total = 0
        weighted_hash_len = []
        for L in list(hash_len_dist):
            total += hash_len_dist[L]
            weighted_hash_len.append(hash_len_dist[L]*L)
        avg = sum(weighted_hash_len)/total
#         avg = max_hash_len

        return int(total), int(avg), max_sample_len

class Word_Preprocessor:
    def __init__(self, freq_dict = {}, train = False, subsampling = False):
        self.freq_dict = freq_dict
        self.train = train
        self.subsampling = subsampling
        self.total = sum(list(freq_dict.values()))
        
    def __call__(self, sample):
        text = sample["input"]
        text = self.prep_text(text)
        tokenized = word_tokenize(text.lower())
        if self.train:
            tokenized = self.cull(tokenized)
            if self.subsampling:
                tokenized = self.subsample(tokenized)
        return [w for w in tokenized]
    
    def prep_text(self, text):
        tknzd = word_tokenize(text)
        tk = []
        tknn = []
        for idx in range(len(tknzd)):
            temp = tknzd[idx].replace("\\n", " ")
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
#             if te[0].isupper():
#                 te = "^" + te
            tknn.append(te)
        return " ".join(tknn).lower()
    
    def cull(self, tokenized):
        culled = []
        for w in tokenized:
            if w in self.freq_dict:
                if self.freq_dict[w] >= min_freq:
                    culled.append(w)
        return culled

    def subsample(self, tokenized):
        subsamplled = []
        for word in tokenized:
            freq = self.freq_dict[word]/self.total
            p_keep = math.sqrt(threshold/(freq))
            keep = np.random.choice([0,1], p=[1-p_keep, p_keep])
            if keep == 1:
                subsamplled.append(word)
        return subsamplled
                

class Hash_Preprocessor(Word_Preprocessor):
    def __init__(self,
                 max_sw_hash_len,
                 max_sample_len,
                 subword_hashes,
                 device,
                 freq_dict = {},
                 train = False,
                 subsampling = False):
        super().__init__(freq_dict, train, subsampling)
        self.max_sw_hash_len = max_sw_hash_len
        self.max_sample_len = max_sample_len
        self.subword_hashes = subword_hashes
        self.device = device
        
    def __call__(self, sample):
        text = sample["input"]
        text = self.prep_text(text)
        tokenized = word_tokenize(text.lower())
        if self.train:
            tokenized = self.cull(tokenized)
            if self.subsampling:
                tokenized = self.subsample(tokenized)
        tokenized_hashes = self.hash_tokenize(tokenized)
        output_id = self.padding(tokenized_hashes, padding_idx=0)
        
        return {"input": output_id, "target": sample['target']-1}
    

    def hash_tokenize(self, data):
        tokenized_id = [self.subword_hashes(w) for w in data]
        return tokenized_id
    
    def padding(self, data, padding_idx=0):
        if len(data) >= self.max_sample_len:
            return torch.tensor(data[:max_sample_len], dtype = torch.long).to(self.device)
        data.extend(np.array([[padding_idx]*self.max_sw_hash_len]*(self.max_sample_len - len(data))))
        return torch.tensor(data, dtype = torch.long).to(self.device)
    
    
