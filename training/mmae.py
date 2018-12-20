"""
This code can only do (those are only the ones that makes sense:
Vanilla for Gaussian
Label for Flower
Style for Gaussian
"""

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from loaders import $dataset=Mnist$ as Dataset
from datetime import datetime as dt
from dist import $dis=Gaussian$ as Dis
from models.base import MLP
from models.mmae import MMAE
from evals.hmap import hmap
from training import State

def one_hot(y, n):
    oh = torch.zeros((len(y), n))
    oh[range(len(y)),y] = 1
    return oh

device = 'gpu' if torch.cuda.is_available() else 'cpu'
print('Training on {}'.format(device))
save_path = '$path=states/aae_mnist.tar$'


type = '$type=vanilla$' #vanilla, style, label, labelclass
classwise = $classwise=True$
rates = $rates=[1,1,0.05]$
nepoch = $nepoch=100$
batch_size = $bs=100$
rec_lr = $rec_lr={0:0.008}$
adv_lr = $adv_lr={0:0.0008}$
input_noise = $input_noise=0.3$

z_dim = $z_dim=2$
hidden = $h=[1000, 1000]$
batch_norm = $bn=True$

xlim = $xlim=(-20,20)$
ylim = $ylim=(-20,20)$
xdelta = $xdelta=0.5$
ydelta = $ydelta=0.5$

dataset = Dataset()
dis = Dis()

state_dict = dict()
state_dict['enc'] = enc = MLP([784] + hidden + [z_dim], batch_norm=batch_norm)
state_dict['dec'] = dec = MLP([z_dim+10 if type=='style' else z_dim] + hidden + [784], batch_norm=batch_norm)

if torch.cuda.is_available():
    enc.cuda()
    dec.cuda()

rec_params = list(enc.parameters()) + list(dec.parameters())
state_dict['rec_opt'] = rec_opt = torch.optim.SGD(rec_params, lr=rec_lr[0], momentum=0.9)
state_dict['gen_opt'] = gen_opt = torch.optim.SGD(enc.parameters(), lr=adv_lr[0], momentum=0.1)

results = {'rec':[], 'gen':[], 'x_rec':[], 'z_space':[], 'z_grad':[], 'llh':[]}
state = State(state_dict, {'epoch':0, 'batch':0, 'results':results})
state.load(save_path)

def Hloss(x): return torch.min(x**2/2, torch.max(.5*torch.ones_like(x), torch.abs(x)-1/2))
def get_cov(data): return (data[:,:,None]*data[:,None,:]).mean(dim=0)#(data.transpose(1,0) @ data)/len(data)
def get_skew(data): return (data[:,:,None,None]*data[:,None,:,None]*data[:,None,None,:]).mean(dim=0)
def mu_moment_gen(mu): return lambda z: Hloss(z.mean(dim=0)-mu).mean()
def cov_moment_gen(mu, cov): return lambda z: Hloss(get_cov(z-mu)-cov).mean()
def skew_moment_gen(mu): return lambda z: Hloss(get_skew(z-mu)).mean()

if type=='vanilla':
    classwise = None
    mu = torch.tensor([0,0], dtype=torch.float32)
    cov = dis.noise**2*torch.eye(2)
    if torch.cuda.is_available:
        mu = mu.cuda()
        cov = cov.cuda()
    moments = [mu_moment_gen(mu), cov_moment_gen(mu, cov), skew_moment_gen(mu)]

elif type=='style':
    classwise = 10
    mu = torch.tensor([0 for _ in range(z_dim)], dtype=torch.float32)
    cov = dis.noise**2*torch.eye(z_dim)
    if torch.cuda.is_available:
        mu = mu.cuda()
        cov = cov.cuda()
    mu_moments = {i:mu_moment_gen(mu) for i in range(10)}
    cov_moments = {i:cov_moment_gen(mu, cov) for i in range(10)}
    skew_moments = {i:skew_moment_gen(mu) for i in range(10)}
    moments = [mu_moments, cov_moments, skew_moments]

elif type=='label':
    classwise = 10
    mu_moments = {i:mu_moment_gen(torch.tensor(dis.mus[i], dtype=torch.float32).cuda()) for i in range(10)}
    cov_moments = {i:cov_moment_gen(torch.tensor(dis.mus[i], dtype=torch.float32).cuda(), torch.tensor(dis.covs[i], dtype=torch.float32).cuda()) for i in range(10)}
    moments = [mu_moments, cov_moments]

else: raise ValueError('type must be: vanilla, style or label. Got {}'.format(type))
    

rec_label = type=='style'
mmae = MMAE(enc, dec, moments, rates, rec_label=rec_label, classwise=classwise)
for k in range(nepoch):
    state['epoch'] += 1
    epoch = state['epoch']
    now = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    print('{} - Epoch {}'.format(now, epoch), end='\r')
    #if epoch==10: mmae.rates = [1,1,0.01]
    if epoch in rec_lr:
        state_dict['rec_opt'] = rec_opt = torch.optim.SGD(rec_params, lr=rec_lr[epoch], momentum=0.9)
    if epoch in adv_lr:
        state_dict['gen_opt'] = gen_opt = torch.optim.SGD(enc.parameters(), lr=adv_lr[epoch], momentum=0.1)
    for x, y in dataset.batch(batch_size, seed=epoch):
        x += np.random.normal(0, input_noise, x.shape)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        enc.zero_grad()
        dec.zero_grad()
        if state['batch']%2==0:
            loss = mmae.rec_loss(x, y)
            loss.backward()
            clip_grad_norm_(rec_params, 5)
            rec_opt.step()
        if state['batch']%2==1:
            loss = mmae.gen_loss(x, y)
            loss.backward()
            clip_grad_norm_(dec.parameters(), 5)
            gen_opt.step()
        state['batch'] += 1

    x = dataset.valid.x[:1000]
    y = dataset.valid.y[:1000]
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    enc.eval()
    dec.eval()
    state['results']['rec'].append(float(mmae.rec_loss(x, y, eval=True)))
    state['results']['gen'].append(float(mmae.gen_loss(x, y, eval=True)))
    z = enc(x)
    if rec_label:
        a = one_hot(y[:5], 10)
        if torch.cuda.is_available(): a = a.cuda()
        z = torch.cat([z[:5], a], dim=1)
    state['results']['x_rec'].append(torch.sigmoid(dec(z)).detach().cpu().numpy())
    state['results']['z_space'].append(z.detach().cpu().numpy())
    mmae.gen_loss(x, y, eval=True, register_hook=state['results']['z_grad']).backward()
    state['results']['llh'].append(dis.log_likelihood(z.detach().cpu().numpy()).mean())
    state.save(save_path)

print()
