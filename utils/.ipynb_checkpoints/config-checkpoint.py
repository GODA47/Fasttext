max_samples = 10000
batch_size = 8
emb_dim = 50
num_emb = int(2e6+1)
context_size = 3
neg_num = 5
uniform = True
max_epochs = 100

classify_epochs = 50
class_size = 10
misspell_freq = 0.5

if uniform: dist = 'uniform'
else: dist = 'noisedist'
    
epoch = f'{max_epochs}e'
emb_path = f"../Trained_Models/SubwordEmbedding/trained_model/trained_model_{emb_dim}d_{dist}_{epoch}_{context_size}w"