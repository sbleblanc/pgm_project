%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

def spiral(t, a):
    r = t*a
    return r*np.cos(t), r*np.sin(t)

def spiral_sampling(n, alpha=1, n_turns=1, noise=0.1):
    t = np.random.uniform(0, n_turns*2*np.pi, n)
    r = alpha*t + np.random.normal(0, noise, n)
    return np.stack([r*np.cos(t), r*np.sin(t)], axis=1)

def rotate(rad, v):
    c = np.cos(rad)
    s = np.sin(rad)
    rot = np.array([[c, s], [-s, c]])
    return v.dot(rot)

def gaussian2d(mu, s1, s2, n):
    x1 = np.random.normal(mu[0], s1, n)
    x2 = np.random.normal(mu[1], s2, n)
    return np.stack([x1, x2], axis=1)

def flower_sampling(n_petals, n, r_noise=0.8, t_noise=0.1, mu_dist=2):
    rad = 2*np.pi/n_petals
    gs = gaussian2d([mu_dist, 0], r_noise, t_noise, n)
    which = np.random.randint(0, n_petals, n)
    return np.array([rotate(which[i]*rad, gs[i]) for i in range(n)])