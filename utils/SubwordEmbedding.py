import torch
from torch.nn import Module, Embedding

class subwordembedding(Module):
    def __init__(self, 
                 num_embeddings, 
                 embedding_dim, 
                 device = torch.device("cpu"),
                 padding_idx = 0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device =  device
        self.padding_idx = padding_idx
        
        self.subword_embedding = Embedding(num_embeddings = num_embeddings, 
                                           embedding_dim = embedding_dim, 
                                           padding_idx = padding_idx)
    def forward(self, token_ids):
        # token_ids: (batch_size, word_num, hash_size)
        # return: (batch_size, word_num, embedding_dim)
        debug = False
        
        subword_embed = self.subword_embedding(token_ids)
        # (batch_size, word_num, hash_size, embedding_dim)
        if debug: print("subword_embed.shape: ", subword_embed.shape)
        
        word_embed = subword_embed.sum(dim = len(subword_embed.shape) -2).to(self.device)
        # (batch_size, word_num, embedding_dim)
        if debug: print("word_embed.shape: ", word_embed.shape)
        
        if debug: print("\n########################################\n")
        return word_embed
        