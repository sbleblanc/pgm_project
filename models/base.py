import torch
from torch import nn

def MLP(struct, non_linearity=nn.ReLU, batch_norm=True, gaussian_init=False):
    layers = list()
    for ni, no in zip(struct[:-2], struct[1:-1]):
        if batch_norm:
            layers.append(nn.Linear(ni, no, bias=None))
            layers.append(nn.BatchNorm1d(no))
            layers.append(non_linearity())
        else:
            l = nn.Linear(ni, no)
            if gaussian_init:
                nn.init.normal_(l.weight, std=0.01)
            layers.append(l)
            layers.append(non_linearity())
    l = nn.Linear(struct[-2], struct[-1])
    if gaussian_init:
        nn.init.normal_(l.weight, std=0.01)
    layers.append(l)
    return nn.Sequential(*layers)
