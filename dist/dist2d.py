#TODO
#Add classify=None to __call__

import numpy as np

class Gaussian(object):
    def __init__(self, noise=5, rng=np.random):
        self.noise = noise
        self.rng = rng
        
    def __call__(self, n):
        return self.rng.normal(0, self.noise, (n,2))
    
    def log_likelihood(self, data):
        llh = -(data/self.noise)**2/2 - np.log(self.noise) - np.log(2*np.pi)/2
        return llh.sum(axis=1)

class Spiral(object):
    def __init__(self, alpha=1, n_turns=2.25, noise=0.2, rng=np.random):
        self.alpha = alpha
        self.n_turns = n_turns
        self.noise = noise
        self.rng = rng
        
    def __call__(self, n):
        t = self.rng.uniform(0, self.n_turns*2*np.pi, n)
        r = self.alpha*t + self.rng.normal(0, self.noise, n)
        return np.stack([r*np.cos(t), r*np.sin(t)], axis=1)
    
    def log_likelihood(self, data):
        #TODO
        #some maths is needed here
        return 0

class Flower(object):
    def __init__(self, n_petals=10, r_noise=3, t_noise=0.45, mu_dist=8, rng=np.random):
        self.n_petals = n_petals
        self.r_noise = r_noise
        self.t_noise = t_noise
        self.mu_dist = mu_dist
        self.rng = rng

    def rotate(self, rad, v):
        c = np.cos(rad)
        s = np.sin(rad)
        rot = np.array([[c, s], [-s, c]])
        return v.dot(rot)

    def gaussian2d(self, mu, s1, s2, n):
        x1 = self.rng.normal(mu[0], s1, n)
        x2 = self.rng.normal(mu[1], s2, n)
        return np.stack([x1, x2], axis=1)

    def __call__(self, n):
        rad = 2*np.pi/self.n_petals
        gs = self.gaussian2d([self.mu_dist, 0], self.r_noise, self.t_noise, n)
        which = self.rng.randint(0, self.n_petals, n)
        return np.array([self.rotate(w*rad, g) for w,g in zip(which, gs)])

    def log_likelihood(self, data):
        #TODO
        #compute all petals mu
        #compute all petals sigma
        #compute log likelihood for all data for each petals (Gaussian log likelihoods)
        #compute numerically stable log likelihood (standard log ( sum_i exp(y_i) ) tricks)
        return 0

