#!/usr/bin/env python3


import numpy as np
from scipy import stats
import scipy as sp
from scipy import optimize
import matplotlib.pyplot as plt
import json


def test(tau, mu1, sigma1, mu2, sigma2): 
    N = 10000
    x1 = np.random.normal(mu1, sigma1, size = (int(N * tau),))
    x2 = np.random.normal(mu2, sigma2, size = (int(N * (1 - tau)),))
    x = np.concatenate([x1, x2])
    return x

def probability(x, alpha):
    tau, mu1, sigma1, mu2, sigma2 = alpha
    p1 = stats.norm.pdf(x, loc=mu1, scale=np.abs(sigma1))
    p2 = stats.norm.pdf(x, loc=mu2, scale=np.abs(sigma2))
    
    prob_x = tau * p1 + (1 - tau) * p2
    
    return prob_x, p1, p2

def f_likelihood(alpha, x):
    prob_x, p1, p2 = probability(x, alpha)
    return -np.sum(np.log(np.abs(prob_x)))
    
def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol = 1e-3):
    answer = optimize.minimize(f_likelihood, 
                            np.array([tau, mu1, sigma1, mu2, sigma2]),
                            args = x, tol = rtol, bounds = ((0, 1), (-np.inf, np.inf), (0, np.inf),
(-np.inf, np.inf), (0, np.inf)))
    return answer.x
   
def EM_iteration(alpha, x):
    tau, mu1, sigma1, mu2, sigma2 = alpha
    prob_x, p1, p2 = probability(x, alpha)
    p1 = tau * p1 / prob_x
    p2 = (1 - tau) * p2 / prob_x
    
    sigma1 = (np.sqrt((np.sum(p1 * (x - mu1)**2)) / np.sum(p1)))
    sigma2 = (np.sqrt((np.sum(p2 * (x - mu2)**2)) / np.sum(p2)))
    mu1 = (np.sum(p1 * x) / np.sum(p1))
    mu2 = (np.sum(p2 * x) / np.sum(p2))
    tau = (np.sum(p1) / x.size)
    
    return tau, mu1, sigma1, mu2, sigma2
    
def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    new = tau, mu1, sigma1, mu2, sigma2
    crit = 0
    while crit == 0:
        old = new
        new = EM_iteration(old, x)
        crit = np.allclose(new, old, rtol = rtol, atol = 0)
    return np.asarray(new)


def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2,
                      rtol=1e-5):
    pass


if __name__ == "__main__":
    pass


