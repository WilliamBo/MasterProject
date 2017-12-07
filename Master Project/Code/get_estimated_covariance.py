import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import svd
from functions import *

def get_estimated_covariance(data, r, delta, delta2, bounds, bounds2):
    n, N = data.shape
    trunc_cov = get_trunc_cov(data,bounds2)
    P = get_P(N,delta2)
    tmp = dog(r,P,trunc_cov)[1].reshape((N,-1))
    est = tmp @ tmp.T
    return est
