import torch
from torch import nn

def MLP(struct, non_linearity=nn.ReLU, batch_norm=True, w_bound=None):
    w_bound = w_bound**.5
    layers = list()
    for ni, no in zip(struct[:-2], struct[1:-1]):
        if batch_norm:
            l = nn.Linear(ni, no, bias=None)
            layers.append(l)
            layers.append(nn.BatchNorm1d(no))
            layers.append(non_linearity())
        else:
            l = nn.Linear(ni, no)
            nn.init.constant_(l.bias, 0)
            layers.append(l)
            layers.append(non_linearity())
        if w_bound is not None: nn.init.uniform_(l.weight, -w_bound, w_bound)
    l = nn.Linear(struct[-2], struct[-1])
    nn.init.constant_(l.bias, 0)
    if w_bound is not None: nn.init.uniform_(l.weight, -w_bound, w_bound)
    layers.append(l)
    return nn.Sequential(*layers)
