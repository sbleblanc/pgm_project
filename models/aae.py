import torch
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
            z.register_hook(lambda grad: register_hook.append(grad))
        s = -logsigmoid(-self.adv(z)[:,0])
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
        s0 = -logsigmoid(o[:l])
        s1 = -logsigmoid(-o[l:])
        return (s0.mean() + s1.mean())/2
