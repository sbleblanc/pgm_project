import pickle, gzip, os
import numpy as np
from collections import namedtuple

class Mnist(object):
    def __init__(self):
        with gzip.open('datasets/mnist.pkl.gz', 'rb') as f:
            train, valid, test = pickle.load(f, encoding='latin1')
        Datasets = namedtuple('Datasets', ['x','y'])
        self.train = Datasets(x=train[0], y=train[1])
        self.valid = Datasets(x=valid[0], y=valid[1])
        self.test = Datasets(x=test[0], y=test[1])
        
class Glove(object):
    def __init__(self, which, limit=None):
        with open('datasets/glove.6B/glove.6B.{}d.txt'.format(which), 'r') as f:
             self.words, self.embeddings = zip(*[self.__split_word_vec(f.readline()) for n in range(limit)])
        self.words = list(self.words)
        self.embeddings = np.array(self.embeddings)
        
    def __split_word_vec(self, lines):
        split = lines[:-1].split(' ')
        return split[0], np.array(split[1:], dtype=float)
        
    def __getitem__(self, item):
        return self.words.__getitem__(item), self.embeddings.__getitem__(item)