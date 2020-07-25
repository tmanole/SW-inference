import numpy as np
import auxiliary as aux
import matplotlib.pyplot as plt
def w(x, y, r=2, delta=0.1, nq=1000):
    """ Delta-Trimmed r-Wasserstein distance between the empirical measures of two
        one-dimensional samples.
    
        Parameters
        ----------
        x : np.ndarray (n,) 
            sample from P
        y : np.ndarray (m,)
            sample from Q
        r : int, optional
            order of the Wasserstein distance
        delta : float, optional
            trimming constant, between 0 and 0.5.
        nq : int, optional
            number of quantiles to use in Monte Carlo integral approximations

        Returns
        -------
        W : float
            delta-trimmed r-Wasserstein distance
    """        
    n = x.size
    m = y.size
    
    us = np.linspace(delta, 1-delta, nq)

    x_quant = aux._sample_quantile(x, us)
    y_quant = aux._sample_quantile(y, us)
 
    integ = np.mean(np.float_power(np.abs(x_quant - y_quant), r)) 
    
    return np.float_power( ((1/(1-2*delta)) * integ), 1/r)


def sw(x, y, r=2, delta=0.1, N=1000, nq=1000, theta=None):
    """ Delta-trimmed r-Sliced Wasserstein distance between the empirical measures
        of two samples.
    
        Parameters
        ----------
        x : np.ndarray (n,) 
            sample from P
        y : np.ndarray (m,)
            sample from Q
        r : int, optional
            order of the Wasserstein distance
        delta : float, optional
            trimming constant, between 0 and 0.5.
        N : int, optional
            number of Monte Carlo draws from the unit sphere
        nq : int, optional
            number of quantiles to use in Monte Carlo integral approximations
        theta: np.ndarray (N, d), optional
            sample from the unit sphere to be used, if specified

        Returns
        -------
        SW : float
             delta-trimmed r-Sliced Wasserstein distance
    """
    SW = 0

    n = x.shape[0]
    d = x.shape[1]

    th_x = np.empty([n])
    th_y = np.empty([n])
  
    if theta is None:       
        theta = np.random.multivariate_normal(np.repeat(0, d), np.identity(d), size=N)
        theta = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, theta)

    for i in range(N):
        thetas = np.broadcast_to(theta[i,:], [n,d])

        x_proj = np.einsum('ij, ij->i', x, thetas)
        y_proj = np.einsum('ij, ij->i', y, thetas)

        SW += np.float_power(w(x_proj, y_proj, r=r, delta=delta, nq=nq), r)

    return np.float_power(SW/N, 1/r)


