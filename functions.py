import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import svd



def generate_data(n,r,t,X,mu,sds):
    data = np.zeros((len(t),1))
    for j in range(n):
        rv = np.random.normal(0,1,size=(r,1))
        rv = np.multiply(rv, sds)  
        bla = mu(t) + X(t,rv)
        #plt.plot(t,bla)
        data = np.c_[data,bla]
    data = data[:,1:].T
    return data


def get_real_cov(sds,N,g,t):
    r = len(sds)
    realK = np.sum(np.array([sds[i]**2 * g(t,i).reshape((N,1)) @ g(t,i).reshape((1,N)) for i in range(r)]), axis=0)
    return realK

def get_bounds(N, delta, n):
    #delta2 = delta - 0.1
    delta2 = delta
    s = np.random.randint(0,N,n)
    s = s - 0.5 * delta * N
    tt = s+delta*N
    s[s<0] = 0
    tt[tt>=1000] = 1000
    bounds = np.c_[s,tt]
    bounds2 = np.copy(bounds)
    bounds2[:,0]+= (delta-delta2) * N
    bounds2[:,1]-= (delta-delta2) * N
    bounds = np.array(bounds, dtype='int')
    bounds2 = np.array(bounds2, dtype='int')
    return bounds, bounds2

def hmu(data,i,bounds):
    n = len(data)
    U = np.array([(bounds[l][0] <= i) and (i < bounds[l][1]) for l in  range(n)])
    I = int(sum(U)>0)
    if I == 0:
        return 0
    return (I / sum(U)) * sum(U * data[:,i])

def hr(data,i,j,bounds2):
    n = len(data)
    i,j = min(i,j),max(i,j)
    U = np.array([(bounds2[l][0] <= i) and (j < bounds2[l][1]) for l in range(n)])
    I = int(sum(U)>0)
    if I == 0:
        return 0
    mu1 = (I / sum(U)) * sum(U * data[:,i])
    mu2 = (I / sum(U)) * sum(U * data[:,j]) 
    ans = 0
    for k in range(n):
        ans += U[k] * (data[k,i]-mu1)*(data[k,j]-mu2)
    return ans/sum(U)

def get_trunc_cov(data,bounds2):
    N = data.shape[1]
    bla = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            bla[i,j] = hr(data,i,j,bounds2)
    return bla

def get_P(N,delta2):
    P = np.zeros((N,N))
    for i in np.arange(-np.int(N * delta2) + 2, np.int(N * delta2) - 1):
        P = P + np.diag(np.ones(N),i)[:N,:N]
    return P

def get_obj(P,trunc_cov):
    N = P.shape[0]
    def obj(g):
        a = g.reshape((N,-1))
        u = a @ a.T 
        u = u * P
        return np.linalg.norm(trunc_cov-u)**2
    return obj

def get_grad(P,trunc_cov):
    N = P.shape[0]
    def grad(g):
        a = g.reshape((N,-1))
        u = a @ a.T 
        u = u * P
        M = u - trunc_cov
        return (2 * (P * M + P.T * M.T) @ a).reshape((-1,))
    return grad


def obj(g):
    a = g.reshape((N,-1))
    u = a @ a.T 
    u = u * P
    return np.linalg.norm(trunc_cov-u)**2

def grad(g):
    a = g.reshape((N,-1))
    u = a @ a.T 
    u = u * P
    M = u - trunc_cov
    return (2 * (P * M + P.T * M.T) @ a).reshape((-1,))

def dog(i, P, trunc_cov):
    obj = get_obj(P,trunc_cov)
    grad = get_grad(P,trunc_cov)
    ssvd = svd(trunc_cov)
    x0 = (ssvd[0][:,:i] @ np.diag(ssvd[1][:i])).reshape((-1,))
    #res = minimize(obj, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    res = minimize(obj, x0, method='BFGS',jac = grad, options={'gtol': 1e-6, 'disp': False})
    return res.fun, res.x 