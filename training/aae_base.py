import numpy as np
import torch
from loaders import $dataset=Mnist$ as Dataset
from datetime import datetime as dt
from dist import $dis=Gaussian$ as Dis
from models.base import MLP
from models.aae import AAE
from evals.hmap import hmap
from training import State

device = 'gpu' if torch.cuda.is_available() else 'cpu'
print('Training on {}'.format(device))
save_path = '$path=states/aae_mnist.tar$'

nepoch = $nepoch=100$
batch_size = $bs=100$
lr = $lr=0.01$
input_noise = $input_noise=0.3$

z_dim = $z_dim=2$
hidden = $h=[500, 500]$
batch_norm = $bn=False$

xlim = $xlim=(-20,20)$
ylim = $ylim=(-20,20)$
xdelta = $xdelta=0.5$
ydelta = $ydelta=0.5$

dataset = Dataset()
dis = Dis()

state_dict = dict()
state_dict['enc'] = enc = MLP([784] + hidden + [z_dim], batch_norm=batch_norm)
state_dict['dec'] = dec = MLP([z_dim] + hidden + [784], batch_norm=batch_norm)
state_dict['adv'] = adv = MLP([z_dim] + hidden + [1], batch_norm=batch_norm)

if torch.cuda.is_available():
    enc.cuda()
    dec.cuda()
    adv.cuda()

rec_params = list(enc.parameters()) + list(dec.parameters())
state_dict['rec_opt'] = rec_opt = torch.optim.SGD(rec_params, lr=lr, momentum=0.9)
state_dict['gen_opt'] = gen_opt = torch.optim.SGD(enc.parameters(), lr=lr, momentum=0.1)
state_dict['adv_opt'] = adv_opt = torch.optim.SGD(adv.parameters(), lr=lr, momentum=0.1)

results = {'rec':[], 'gen':[], 'adv':[], 'x_rec':[], 'z_hmap':[], 'z_space':[], 'z_grad':[], 'llh':[]}
state = State(state_dict, {'epoch':0, 'batch':0, 'results':results})
state.load(save_path)

aae = AAE(enc, dec, dis, adv)
for k in range(nepoch):
    state['epoch'] += 1
    now = dt.now().strftime('%Y-%m-%d %H:%M:%S')
    print('{} - Epoch {}'.format(now, state['epoch']), end='\r')
    for x, y in dataset.batch(batch_size):
        x += np.random.normal(0, input_noise, x.shape)
        x = torch.tensor(x, dtype=torch.float32)
        if torch.cuda.is_available(): x = x.cuda()
        enc.zero_grad()
        dec.zero_grad()
        adv.zero_grad()
        if state['batch']%3==0:
            loss = aae.rec_loss(x)
            loss.backward()
            rec_opt.step()
        if state['batch']%3==1:
            loss = aae.adv_loss(x)
            loss.backward()
            adv_opt.step()
        if state['batch']%3==2:
            loss = aae.gen_loss(x)
            loss.backward()
            gen_opt.step()
        state['batch'] += 1

    x = dataset.valid.x[:1000]
    x = torch.tensor(x, dtype=torch.float32)
    if torch.cuda.is_available(): x = x.cuda()
    enc.eval()
    dec.eval()
    adv.eval()
    state['results']['rec'].append(float(aae.rec_loss(x, eval=True)))
    state['results']['gen'].append(float(aae.gen_loss(x, eval=True)))
    state['results']['adv'].append(float(aae.adv_loss(x, eval=True)))
    state['results']['x_rec'].append(torch.sigmoid(dec(enc(x[:5]))).detach().cpu().numpy())
    state['results']['z_hmap'].append(hmap(adv, xlim, ylim, xdelta, ydelta))
    state['results']['z_space'].append(enc(x).detach().cpu().numpy())
    aae.gen_loss(x, eval=True, register_hook=state['results']['z_grad']).backward()
    state['results']['llh'].append(dis.log_likelihood(enc(x).detach().cpu().numpy()).mean())
    state.save(save_path)

print()
