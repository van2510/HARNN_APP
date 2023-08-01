from gensim.models import KeyedVectors
from collections import OrderedDict

wv = KeyedVectors.load('/content/drive/MyDrive/2022-2023/HARNN/Hierarchical-Multi-Label-Text-Classification-master/data/word2vec_100.kv', mmap='r')

word2idx = OrderedDict({"_UNK": 0})
embedding_size = wv.vector_size
for k, v in wv.vocab.items():
    print(k, v)
    word2idx[k] = v.index + 1
vocab_size = len(word2idx)
print(vocab_size)