import numpy as np
from utils.solver import inverse_solver #for spiral classification
from dist.percentile import gaussian_percentile

class Gaussian(object):
    def __init__(self, noise=5, rng=np.random):
        self.noise = noise
        self.rng = rng
        
    def __call__(self, n):
        return self.rng.normal(0, self.noise, (n,2))

    def gaussian_percentile(self, n):
        mu = np.array([0,0])
        K = self.noise*np.eye(2)
        return gaussian_percentile(n, mu, K)
    
    def log_likelihood(self, data):
        llh = -(data/self.noise)**2/2 - np.log(self.noise) - np.log(2*np.pi)/2
        return llh.sum(axis=1)

class Spiral(object):
    def __init__(self, alpha=1, n_turns=2.25, noise=0.2, n_class=10, rng=np.random):
        self.alpha = alpha
        self.n_turns = n_turns
        self.noise = noise
        self.n_class = n_class
        self.rng = rng
        
    def length_arc(self, t):
        sqrt = np.sqrt(t**2 + 1)
        return self.alpha * .5 * (np.log(t + sqrt) + t*sqrt)
    
    def random_curve(self, t):
        r = self.alpha*t + self.rng.normal(0, self.noise, len(t))
        return np.stack([r*np.cos(t), r*np.sin(t)], axis=1)
    
    def length_arc_curve(self, s):
        t = inverse_solver(self.length_arc, s, 0, self.n_turns*2*np.pi)
        return self.random_curve(t)
    
    def classify(self, t):
        max_t = self.n_turns*2*np.pi
        total_length = self.length_arc(max_t)
        length_marks = [i*total_length/self.n_class for i in range(1, self.n_class+1)]
        t_marks = inverse_solver(self.length_arc, length_marks, 0, max_t)
        return np.array([(a < t_marks).argmax() for a in t])

    def get_splits(self):
        total_length = self.length_arc(self.n_turns * 2 * np.pi)
        length_marks = [i * total_length / self.n_class for i in range(1, self.n_class + 1)]
        return inverse_solver(self.length_arc, length_marks, 0, total_length)

    def __call__(self, n, classify=False, sections=None):
        if not classify:
            t = self.rng.uniform(0, self.n_turns * 2 * np.pi, n)
            data = self.random_curve(t)
            return data
        else:
            which = self.rng.randint(0, self.n_class, n)
            y_swap = np.argwhere(sections != 10)
            which[y_swap] = sections[y_swap]
            splits = np.hstack([np.array([0]), self.get_splits()])
            t = np.zeros(n)
            for i in range(n):
                s = self.rng.uniform(splits[which[i]], splits[which[i] + 1], 1)
                t[i] = s
            # r = self.alpha * t + self.rng.normal(0, self.noise, n)
            return self.random_curve(t), which
    
    def log_likelihood(self, data):
        #TODO
        #some maths is needed here
        return np.array([0])

class Flower(object):
    def __init__(self, n_petals=10, r_noise=3, t_noise=0.45, mu_dist=8, rng=np.random):
        self.n_petals = n_petals
        self.r_noise = r_noise
        self.t_noise = t_noise
        self.mu_dist = mu_dist
        self.rng = rng
        self.init_normal()
    
    def get_rotation(self, rad):
        c = np.cos(rad)
        s = np.sin(rad)
        return np.array([[c, -s], [s, c]])
        
    def init_normal(self):
        mu0 = np.array([self.mu_dist, 0])
        cov0 = np.array([[self.r_noise, 0], [0, self.t_noise]])
        Rs = [self.get_rotation(i*2*np.pi/self.n_petals) for i in range(self.n_petals)]
        self.mus = [mu0 @ R.T for R in Rs]
        self.covs = [R @ cov0 @ R.T for R in Rs]
        
    def sample_petal(self, petal, n):
        mu = self.mus[petal]
        cov = self.covs[petal]
        return self.rng.multivariate_normal(mu, cov, n)
    
    def __call__(self, n=None, classify=False, which=None):
        if n is None and which is None:
            raise ValueError('Either n or which must be define')
        if n is not None and which is not None and len(which)!=n:
            raise ValueError('len(which) must be equal to n')
        if which is None: which = np.array([self.n_petals for _ in range(n)])
        else: which = np.array(which)
        index = np.argwhere(which==self.n_petals)
        which[index] = self.rng.randint(0, self.n_petals, (len(index),1))
        data = np.zeros((len(which), 2))
        for i in range(self.n_petals):
            index = which==i
            data[index] = self.sample_petal(i, sum(index))
        if classify: return data, which
        else: return data
        
    def log_normal(self, x, mu, sig):
        c = x-mu
        a = (c.dot(np.linalg.inv(sig))*c).sum(axis=1)
        b = len(mu)*np.log(2*np.pi) + np.log(np.linalg.det(sig))
        return -(a + b)*.5
    
    def log_petals(self, x):
        out = np.zeros((len(x), self.n_petals))
        for i in range(self.n_petals):
            out[:,i] = self.log_normal(x, self.mus[i], self.covs[i])
        return out
    
    def ls(self, x, axis=None):
        off = np.max(x, axis=axis, keepdims=True)
        return off + np.log(np.sum(np.exp(x - off), axis=axis, keepdims=True))
    
    def log_likelihood(self, data):
        log_petals = self.log_petals(data)
        data_ll = self.ls(log_petals, axis=1)
        return data_ll.mean() - np.log(self.n_petals)
    
    def gaussian_percentile(self, petal, n):
        rad = 2*np.pi*petal/self.n_petals
        R = self.get_rotation(rad)
        S = np.array([[np.sqrt(self.r_noise),0],[0,np.sqrt(self.t_noise)]])
        return gaussian_percentile(n, self.mus[petal], R @ S)
