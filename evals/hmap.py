import torch
import numpy as np

def hmap(fct, xlim=(-1,1), ylim=(-1,1), xdelta=0.1, ydelta=0.1):
    x = np.arange(*xlim, xdelta)
    y = np.arange(*ylim, ydelta)
    X, Y = np.meshgrid(x, y)
    inp = np.stack([X.flatten(), Y.flatten()], axis=1)
    inp = torch.tensor(inp, dtype=torch.float32)
    if torch.cuda.is_available(): inp = inp.cuda()
    out = torch.sigmoid(fct(inp))
    return out.detach().cpu().numpy().reshape(X.shape)


def hmap_y(fct, lbl, xlim=(-1,1), ylim=(-1,1), xdelta=0.1, ydelta=0.1):
    x = np.arange(*xlim, xdelta)
    y = np.arange(*ylim, ydelta)
    X, Y = np.meshgrid(x, y)
    inp = np.stack([X.flatten(), Y.flatten()], axis=1)
    fct_y = np.repeat(np.eye(1, 11, lbl), inp.shape[0], axis=0)
    inp = torch.tensor(np.hstack([inp, fct_y]), dtype=torch.float32)
    if torch.cuda.is_available(): inp = inp.cuda()
    out = torch.sigmoid(fct(inp))
    return out.detach().cpu().numpy().reshape(X.shape)


# rec_lr={0:0.,200:0.001,500:0.0001} adv_lr={0:0.01,200:0.05,500:0.005}