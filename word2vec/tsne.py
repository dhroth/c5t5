import numpy as np
from sklearn.manifold import TSNE

npz = np.load("word2vec.big.vecs.npz")
vecs = npz["vecs"]
words = npz["words"]

tsne = TSNE(n_jobs=-1)

embedded = tsne.fit_transform(vecs)

np.savez("word2vec.big.tsne.npz", words=words, embedded=embedded)
