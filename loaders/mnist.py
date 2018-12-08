import pickle, gzip, os
import numpy as np
import numpy.random as rnd
from collections import namedtuple

class Mnist(object):
    def __init__(self, num_unlabelled=0):
        with gzip.open('datasets/mnist.pkl.gz', 'rb') as f:
            train, valid, test = pickle.load(f, encoding='latin1')
        Datasets = namedtuple('Datasets', ['x','y'])
        self.train = Datasets(x=train[0], y=train[1])
        self.valid = Datasets(x=valid[0], y=valid[1])
        self.test = Datasets(x=test[0], y=test[1])
        if num_unlabelled > 0:
            to_keep = set()
            split = int((len(self.train.y) - num_unlabelled) / 10)
            for i in range(10):
                lbl_indices = np.argwhere(self.train.y == i)
                to_keep.update(lbl_indices[:split].flatten())
            to_unlabel = set(np.arange(0, len(self.train.y))).difference(to_keep)
            self.train.y[list(to_unlabel)] = 10

        
    def batch(self, bs, which='train', limit=None, seed=0):
        rng = np.random.RandomState(seed=seed)
        x, y = self.__getattribute__(which)
        p = rng.permutation(range(len(x)))
        x = x[p]
        y = y[p]
        if limit is not None:
            x = x[:limit]
            y = y[:limit]
        limit = len(x)//bs*bs
        x = x[:limit].reshape(limit//bs, bs, 784)
        y = y[:limit].reshape(limit//bs, bs)
        for a, b in zip(x, y): yield a, b 
