import numpy as np
from tqdm import tqdm
from nltk import word_tokenize

class subwordhash:
    def __init__(self, dataset):
        word_num, hash_len, sample_len = self.average_subword_num(dataset)
        self.word_num = word_num
        self.max_hash = hash_len
        self.max_sample = sample_len
        
    def __call__(self, word):
        return self.subword_hashes(word, max_hash_num = self.max_hash)
    
    def fnv1a(self, txt, K = int(2e6 + 1)):
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
            tokens = word_tokenize(sample["input"])
            if len(tokens) not in len_dist:
                len_dist[len(tokens)] = 0
            len_dist[len(tokens)] += 1
            max_sample_len = max(max_sample_len, len(tokens))
            
        for L in list(len_dist):
            hash_len_dist[self.subword_hashes('a'*L, get_len = True)] = len_dist[L]
        
        total = 0
        weighted_hash_len = []
        for L in list(hash_len_dist):
            total += hash_len_dist[L]
            weighted_hash_len.append(hash_len_dist[L]*L)
        avg = sum(weighted_hash_len)/total
        
        return int(total), int(avg), max_sample_len
        