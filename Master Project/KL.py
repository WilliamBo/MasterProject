import numpy as np
import matplotlib.pyplot as plt

def var(k):
    return  np.sqrt(2/((k-0.5)**2 * np.pi**2))

def eigen(k,x):
    return np.sin((k-0.5)*np.pi*x)

x = np.linspace(0,1,1000)

k = 10000

rvs = [np.random.normal(0,var(i)) for i in range(1,k+1)]

def B(x):
    ans = 0
    for i in range(k):
        ans += rvs[i] * eigen(i+1,x)
    return ans

y = [B(r) for r in x]
plt.plot(x,y)
plt.show()
