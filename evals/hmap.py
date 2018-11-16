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
