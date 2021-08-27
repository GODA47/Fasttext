max_samples = 10000
batch_size = 8
num_emb = int(2e6+1)
min_freq = 1
cull = True
subsampling = False
threshold = 1e-5
noURL = True
lemmatize = True
stopword = False

emb_dim = 100
context_size = 3
neg_num = 5
uniform = False
max_epochs = 100

classify_epochs = 50
class_size = 10
misspell_freq = 0.5
lower = False
if uniform: dist = 'uniform'
else: dist = 'noisedist'

epo = f'{max_epochs}e'
emb_add = f''
emb_path = f"../Trained_Models/SubwordEmbedding/trained_model/"
emb_add += f"trained_model_{emb_dim}d_{dist}_{epo}_{context_size}w"
emb_add += f'_{max_samples}sample'
emb_add += f'_min{min_freq}'

if cull:
    emb_add += '_culled'
if subsampling:
    emb_add += '_subsampling'
if noURL:
    emb_add += '_noURL'
if lemmatize:
    emb_add += '_lemmatized'
if stopword:
    emb_add += '_stopword'
    
emb_path += emb_add