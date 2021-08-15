import torch
from torch.nn import Module, Embedding

class subwordembedding(Module):
    def __init__(self, 
                 num_embeddings, 
                 embedding_dim, 
                 device = torch.device("cpu"),
                 padding_idx = 0,
                 sumfirst = True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device =  device
        self.padding_idx = padding_idx
        self.sumfirst = sumfirst
        self.subword_embedding = Embedding(num_embeddings = num_embeddings, 
                                           embedding_dim = embedding_dim, 
                                           padding_idx = padding_idx)
    def forward(self, token_ids):
        # token_ids: (batch_size, hash_size)
        
        subword_embed = self.subword_embedding(token_ids)
        # (batch_size, ngram_size, embedding_dim)
        
        if self.sumfirst:
            word_embed = subword_embed.sum(dim = len(subword_embed.shape) -2).to(self.device)
            # (batch_size, embedding_dim)
        else:
            word_embed = subword_embed.to(self.device)
            # (batch_size, ngram_size, embedding_dim)
        return word_embed
        