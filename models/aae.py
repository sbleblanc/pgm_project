import torch
import numpy as np
logsigmoid = torch.nn.functional.logsigmoid

class AAE(object):
    def __init__(self, enc, dec, dis, adv):
        self.enc = enc
        self.dec = dec
        self.dis = dis
        self.adv = adv

    def rec_loss(self, x, eval=False):
        if eval:
            self.enc.eval()
            self.dec.eval()
        else:
            self.enc.train()
            self.dec.train()
        x_hat = torch.sigmoid(self.dec(self.enc(x)))
        return ((x_hat-x)**2).sum(dim=1).mean()

    def gen_loss(self, x, eval=False, register_hook=None):
        if eval: self.enc.eval()
        else: self.enc.train()
        self.adv.eval()
        z = self.enc(x)
        if register_hook is not None:
            z.register_hook(lambda grad: register_hook.append(grad.cpu().numpy()))
        s = -logsigmoid(self.adv(z)[:,0])
        return s.mean()

    def adv_loss(self, x, eval=False):
        self.enc.eval()
        if eval: self.adv.eval()
        else: self.adv.train()
        l = len(x)
        z0 = self.enc(x)
        z1 = torch.tensor(self.dis(l), dtype=torch.float32)
        if torch.cuda.is_available(): z1 = z1.cuda()
        z = torch.cat([z0, z1], dim=0)
        o = self.adv(z)[:,0]
        s0 = -logsigmoid(-o[:l])
        s1 = -logsigmoid(o[l:])
        return (s0.mean() + s1.mean())/2


class AAERegularized(AAE):

    def __init__(self, enc, dec, dis, adv):
        super(AAERegularized, self).__init__(enc, dec, dis, adv)

    def one_hot(self, y):
        oh_vec = []
        for lbl in y:
            oh_vec.append(np.eye(1, 11, lbl))
        return np.stack(oh_vec, axis=1)[0]

    def gen_loss(self, x, y, eval=False, register_hook=None):
        if eval: self.enc.eval()
        else: self.enc.train()
        self.adv.eval()
        oh = torch.tensor(self.one_hot(y), dtype=torch.float32)
        if torch.cuda.is_available(): oh = oh.cuda()
        z = self.enc(x)
        z = torch.cat([z, oh], 1)
        if register_hook is not None:
            z.register_hook(lambda grad: register_hook.append(grad.cpu().numpy()))
        s = -logsigmoid(self.adv(z)[:,0])
        return s.mean()

    def adv_loss(self, x, y, eval=False):
        self.enc.eval()
        if eval: self.adv.eval()
        else: self.adv.train()
        oh = torch.tensor(self.one_hot(y), dtype=torch.float32)
        if torch.cuda.is_available(): oh = oh.cuda()
        l = len(x)
        z0 = self.enc(x)
        z0 = torch.cat([z0, oh], 1)
        z1, d_y = self.dis(l, classify=True, petals=y)
        z1 = np.hstack([z1, self.one_hot(d_y)])
        z1 = torch.tensor(z1, dtype=torch.float32)
        if torch.cuda.is_available(): z1 = z1.cuda()
        z = torch.cat([z0, z1], dim=0)
        o = self.adv(z)[:,0]
        s0 = -logsigmoid(-o[:l])
        s1 = -logsigmoid(o[l:])
        return (s0.mean() + s1.mean())/2
