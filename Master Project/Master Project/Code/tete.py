import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import svd
from functions import *
from get_estimated_covariance import *
import scipy.stats
from scipy.interpolate import UnivariateSpline

n = 200
r1 = 20
r2 = 20
N = 50
t = np.linspace(0,1,N)
#sds1 = np.append(np.array([1.5,0.5,0.1,0.05]), np.abs(np.random.normal(0,0.001,r1-4))).reshape(r1,1)
sds1 = np.append(np.array([1,1]), 1/(np.arange(30,r1+28)**2)).reshape(r1,1)
sds1 = np.sqrt(sds1)
#sds2 = sds1 * np.random.chisquare(1000,sds1.shape)/1000
sds2 = np.append(np.array([0.01,0.01,1]), 1/(np.arange(30,r2+27)**2)).reshape(r2,1)
sds2 = np.sqrt(sds2)

def mu1(t):
    return np.zeros(t.shape)
    #return np.exp(t)/100
def mu2(t):
    #return np.sin(2*np.pi*t)/10
    #return -np.log(t + 1)/100
    return np.zeros_like(t)

def g1(x,n):
    if n == 0:
        return np.array([1]* len(x))
    else:
        return np.sqrt(2)*np.cos(2* n * np.pi * x)
    
def g2(x,n):
    return g1(x,n)

def X1(t,rv):
    return np.sum(np.multiply(np.repeat(rv,N).reshape(r1,N),np.array([g1(t,i) for i in range(r1)])),axis = 0)
def X2(t,rv):
    return np.sum(np.multiply(np.repeat(rv,N).reshape(r2,N),np.array([g2(t,i) for i in range(r2)])),axis = 0)

data1 = generate_data(n, r1, t, X1, mu1, sds1)
data2 = generate_data(n, r2, t, X2, mu2, sds2)

from mpl_toolkits.mplot3d import Axes3D

K1 = np.cov(data1, rowvar=False, bias=True)
blar = np.linalg.svd(K1)[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = np.append(data1 @ blar[:,0]/100, data2 @ blar[:,0]/100)
ys = np.append(data1 @ blar[:,1]/100, data2 @ blar[:,1]/100)
zs = np.append(data1 @ blar[:,2]/100, data2 @ blar[:,2]/100)

ax.scatter(xs, ys, zs, c=np.append(np.zeros(n),np.ones(n)))

plt.show()


K2 = np.cov(data1, rowvar=False, bias=True)
blar = np.linalg.svd(K2)[0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = np.append(data1 @ blar[:,0]/100, data2 @ blar[:,0]/100)
ys = np.append(data1 @ blar[:,1]/100, data2 @ blar[:,1]/100)
zs = np.append(data1 @ blar[:,2]/100, data2 @ blar[:,2]/100)

ax.scatter(xs, ys, zs, c=np.append(np.zeros(n),np.ones(n)))

fig.show()

zt = data1 @ np.linalg.svd(K1)[0][:,:3]
zt = np.r_[zt, data2 @ np.linalg.svd(K2)[0][:,:3]]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(zt[:,0], zt[:,1], zt[:,2], c=np.append(np.zeros(n),np.ones(n)))
fig.show()