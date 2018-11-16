import torch
from torch import nn

def MLP(struct, non_linearity=nn.ReLU, batch_norm=True):
    layers = list()
    for ni, no in zip(struct[:-2], struct[1:-1]):
        if batch_norm:
            layers.append(nn.Linear(ni, no, bias=None))
            layers.append(nn.BatchNorm1d(no))
            layers.append(non_linearity())
        else:
            layers.append(nn.Linear(ni, no))
            layers.append(non_linearity()) 
    layers.append(nn.Linear(struct[-2], struct[-1]))
    return nn.Sequential(*layers)
