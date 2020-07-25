import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def rtoggle(n, alpha1, alpha2, beta1, beta2, mu, sigma, gamma, tmax=300, u0=10, v0=10): 
    u = np.zeros((n, tmax+1))
    v = np.zeros((n, tmax+1))

    u[:,0] = u0
    v[:,0] = v0

    for t in range(0, tmax):
        u[:,t+1] = u[:,t] + alpha1 / (1 + v[:,t]**beta1) - (1 + 0.03 * u[:,t]) 
        v[:,t+1] = v[:,t] + alpha2 / (1 + u[:,t]**beta2) - (1 + 0.03 * v[:,t]) 

        u[:,t+1] += 0.5 * scipy.stats.truncnorm.rvs(-u[:,t+1]/0.5, np.inf, size=n)
        v[:,t+1] += 0.5 * scipy.stats.truncnorm.rvs(-v[:,t+1]/0.5, np.inf, size=n)
        
    lb = -(u[:,tmax] + mu) / (mu * sigma * (u[:,tmax]**gamma))
    y  = u[:,tmax] + mu + mu * sigma * scipy.stats.truncnorm.rvs(a = lb, b=np.inf, size=n) / (u[:,tmax]**gamma)

    return y

