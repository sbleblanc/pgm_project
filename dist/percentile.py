import numpy as np
from scipy.special import ndtr, ndtri

def mean_block(a, b, approx_n=100000):
    xs = np.arange(a,b,(b-a)/approx_n)
    ps = ndtr(xs) #get cdf
    p_block = ps[1:] - ps[:-1] #get block cdf
    mean_xs = (xs[:-1] + xs[1:])/2 #get mean position
    return sum(p*a for p,a in zip(p_block, mean_xs))/(ndtr(b)-ndtr(a))

def get_blocks(n=10, min_p=0.000001, max_p=0.999999):
    ps = [min_p] + [i/n for i in range(1,n)] + [max_p]
    inv_ps = ndtri(ps)
    return inv_ps[:-1], inv_ps[1:]

def get_mean_blocks(n=10, min_p=0.000001, max_p=0.999999, approx_n=100000):
    A, B = get_blocks(n=n,min_p=min_p,max_p=max_p)
    return [mean_block(a,b,approx_n) for a,b in zip(A,B)]

def get_2d_mean_block(n=10, min_p=0.000001, max_p=0.999999, approx_n=100000):
    mb = get_mean_blocks(n=n, min_p=min_p, max_p=max_p, approx_n=approx_n)
    return np.stack(np.meshgrid(mb,mb), axis=2).reshape(n**2,2)

def gaussian_percentile(n, mu, K):
    mb = get_2d_mean_block(10, approx_n=1000)
    return mu + mb @ K.T
