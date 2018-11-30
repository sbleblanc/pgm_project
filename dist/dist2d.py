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

    def rotate_covariance(self, rad, cov):
        c = np.cos(rad)
        s = np.sin(rad)
        R = np.array([[c, -s], [s, c]])
        return R @ cov @ R.T

    def gaussian2d(self, mu, s1, s2, n):
        x1 = self.rng.normal(mu[0], s1, n)
        x2 = self.rng.normal(mu[1], s2, n)
        return np.stack([x1, x2], axis=1)

    def gaussian2d_M(self, mu, cov, n):
        return np.random.multivariate_normal(mu, cov, n)


    def simple_gaussian(self, n):
        rad = (2 * np.pi / self.n_petals) * 2
        cov = np.square(np.diag([self.r_noise, self.t_noise]))
        c = np.cos(rad)
        s = np.sin(rad)
        R = np.array([[c, s], [-s, c]])
        cov = R @ cov @ R.T
        return self.gaussian2d_M([self.mu_dist, 0], cov, n)


    def __call__(self, n, classify=False):
        rad = 2*np.pi/self.n_petals
        gs = self.gaussian2d([self.mu_dist, 0], self.r_noise, self.t_noise, n)
        which = self.rng.randint(0, self.n_petals, n)
        data = np.array([self.rotate(w*rad, g) for w,g in zip(which, gs)])
        if classify: return data, which
        return data

    def multivariate_gaussian_ll(self, x, mu, sigma):
        diff = x - mu
        num = -0.5 * diff.T @ np.linalg.lstsq(sigma, diff)[0]
        den = 0.5 * np.log((2*np.pi)**self.n_petals * np.linalg.det(sigma))
        return num - den

    def ls(self, x, axis=None):
        off = np.max(x, axis=axis, keepdims=True)
        return off + np.log(np.sum(np.exp(x - off), axis=axis, keepdims=True))

    def log_likelihood(self, data, f=False):
        mus = []
        sigmas = []
        ll = 0.
        base_mu = np.array([self.mu_dist, 0])
        base_cov = np.square(np.diag([self.r_noise, self.t_noise]))
        rad = 2 * np.pi / self.n_petals
        for i in range(self.n_petals):
            mus.append(self.rotate(rad * i, base_mu))
            sigmas.append(self.rotate_covariance(rad * i, base_cov))

        if f:
            self.multivariate_gaussian_ll(data, mus[i], sigmas[i])

        for d in data:
            d_ll = np.zeros(self.n_petals)
            for i in range(self.n_petals):
                d_ll[i] = np.log(1./self.n_petals) + self.multivariate_gaussian_ll(d, mus[i], sigmas[i])
            ll += self.ls(d_ll, axis=0)
        return ll
        #TODO
        #compute all petals mu
        #compute all petals sigma
        #compute log likelihood for all data for each petals (Gaussian log likelihoods)
        #compute numerically stable log likelihood (standard log ( sum_i exp(y_i) ) tricks)
        # return 0

