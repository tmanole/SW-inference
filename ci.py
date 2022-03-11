import numpy as np
import distances as dist
import auxiliary as aux
from sklearn.neighbors import KernelDensity
from auxiliary import *
import matplotlib.pyplot as plt

""" One-dimensional CIs. """


def exact_1d(x, y, r=2, delta=0.1, alpha=0.05, mode="DKW", nq=1000):
    """ Confidence intervals for W_{r,delta}(P, Q) in one dimension.
    
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
        alpha : float, optional
            number between 0 and 1, such that 1-alpha is the level of the confidence interval
        mode : str, optional
            either "DKW" to use a confidence interval based on the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality [1,2]
            or "rel_VC" to use a confidence interval based on the relative Vapnik-Chervonenkis (VC) inequality [3]
        nq : int, optional
            number of quantiles to use in Monte Carlo integral approximations

        Returns
        -------
        l : float
            lower confidence limit

        u : float
            upper confidence limit

        References
        ----------

        .. [1] Dvoretzky, Aryeh, Jack Kiefer, and Jacob Wolfowitz. 
               "Asymptotic minimax character of the sample distribution function and 
               of the classical multinomial estimator." The Annals of Mathematical Statistics (1956): 642-669.

        .. [2] Massart, Pascal. "The tight constant in the Dvoretzky-Kiefer-Wolfowitz inequality." The annals of Probability (1990): 1269-1283.

        .. [3] Vapnik, V., Chervonenkis, A.: On the uniform convergence of relative frequencies of events to
               their probabilities. Theory of Probability and its Applications 16 (1971) 264–280.

    """       
    us = np.linspace(delta, 1-delta, nq)

    if mode == "DKW":
        try:
            Lx, Ux = aux._dkw(x, us, alpha)
            Ly, Uy = aux._dkw(y, us, alpha)

        except OverflowError:
            return (0, np.Inf)

    elif mode == "rel_VC":
        try:
            Lx, Ux = aux._rel_vc(x, us, alpha)
            Ly, Uy = aux._rel_vc(y, us, alpha)

        except OverflowError:
            return (0, np.Inf)

    elif mode == "sequential":
        Lx, Ux = aux._quantile_seq(x, us, delta=alpha)[-1,:]
        Ly, Uy = aux._quantile_seq(y, us, delta=alpha)[-1,:]

    else:
        raise Exception("Mode unrecognized.")

    low = np.repeat(0, nq)
    up  = np.repeat(0, nq)

    low = np.fmax(Lx - Uy, Ly - Ux)
    low = np.power( np.fmax(low, np.repeat(0, nq)), r)
    up  = np.power( np.fmax(Ux - Ly, Uy - Lx),  r)

    lower_final = np.power( (1/(1-2*delta)) * np.mean(low),  1/r)
    upper_final = np.power( (1/(1-2*delta)) * np.mean(up ),  1/r)

    return lower_final, upper_final


def bootstrap_1d(x, y, r=2, delta=0.1, alpha=0.05, B=1000, nq=1000):
    """ Bootstrap confidence intervals for W_{r,delta}(P, Q) in one dimension.
    
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
        alpha : float, optional
            number between 0 and 1, such that 1-alpha is the level of the confidence interval
        B : int, optional
            number of bootstrap replications
        nq : int, optional
            number of quantiles to use in Monte Carlo integral approximations

        Returns
        -------
        l : float
            lower confidence limit

        u : float
            upper confidence limit
    """
    n = x.shape[0]
    m = y.shape[0]

    W = []
    What = np.power(dist.w(x, y, r=r, delta=delta, nq=nq), r)

    for b in range(B):
        I  = np.random.choice(n, n)
        xx = x[I]
        I  = np.random.choice(m, m)
        yy = y[I]

        W.append(np.power(dist.w(xx, yy, r=r, delta=delta, nq=nq), r) - What)

    q1 = np.quantile(W, alpha/2)
    q2 = np.quantile(W, 1-alpha/2)

    Wlower = np.max([What - q2,0])
    Wupper = What - q1

    if Wupper < 0:
        return 0, 0

    return np.power(Wlower, 1/r), np.power(Wupper, 1/r)



""" Sliced Wasserstein distance confidence intervals. """


def mc_sw(x, y, r=2, delta=0.1, alpha=0.05, N=500, nq=500, theta=None):
    """ Monte Carlo confidence interval for SW_{r,delta}(P, Q).
    
        Parameters
        ----------
        x : np.ndarray (n,d) 
            sample from P
        y : np.ndarray (m,d)
            sample from Q
        r : int, optional
            order of the Wasserstein distance
        delta : float, optional
            trimming constant, between 0 and 0.5.
        alpha : float, optional
            number between 0 and 1, such that 1-alpha is the level of the confidence interval
        N : int, optional
            number of Monte Carlo draws from the unit sphere
        nq : int, optional
            number of quantiles to use in Monte Carlo integral approximations
        theta: np.ndarray (N, d), optional
            sample from the unit sphere to be used, if specified

        Returns
        -------
        l : float
            lower confidence limit

        u : float
            upper confidence limit
    """
    SW = 0

    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]

    ws = []
    low = []
    up = []
    if theta is None:
        theta = np.random.multivariate_normal(np.repeat(0, d), np.identity(d), size=N)
        theta = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, theta)

    for i in range(N):
        thetas_x = np.broadcast_to(theta[i,:], [n, d])
        thetas_y = np.broadcast_to(theta[i,:], [m, d])

        x_proj = np.einsum('ij, ij->i', x, thetas_x)
        y_proj = np.einsum('ij, ij->i', y, thetas_y)

        l, u = exact_1d(x_proj, y_proj, r=r, delta=delta, alpha=alpha/N, nq=nq)

        low.append(np.power(l,r))
        up.append(np.power(u,r))

    left  = np.power(np.mean(low), 1/r)
    right = np.power(np.mean(up), 1/r)

    return left, right

def bootstrap_sw(x, y, r=2, delta=0.1, alpha=0.05, B=1000, N=500, nq=500, N_fit=2000, theta=None):
    """ Bootstrap confidence interval for SW_{r,delta}(P, Q).
    
        Parameters
        ----------
        x : np.ndarray (n,d) 
            sample from P
        y : np.ndarray (m,d)
            sample from Q
        r : int, optional
            order of the Wasserstein distance
        delta : float, optional
            trimming constant, between 0 and 0.5.
        alpha : float, optional
            number between 0 and 1, such that 1-alpha is the level of the confidence interval
        B : int, optional
            number of bootstrap replications
        N : int, optional
            number of Monte Carlo draws from the unit sphere
        nq : int, optional
            number of quantiles to use in Monte Carlo integral approximations
        theta: np.ndarray (N, d), optional
            sample from the unit sphere to be used, if specified

        Returns
        -------
        l : float
            lower confidence limit

        u : float
            upper confidence limit
    """
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]

    resample = False
    NN=N
    if theta is None:
        resample = True
        NN = N_fit

    boot = []
    SW_hat = np.power(dist.sw(x, y, r=r, delta=delta, N=NN, nq=nq, theta=theta), r)

    for b in range(B):
        if resample:
            theta = np.random.multivariate_normal(np.repeat(0, d), np.identity(d), size=N)
            theta = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, theta)

        x_ind  = np.random.choice(n, n)
        xx = x[x_ind,:]

        y_ind  = np.random.choice(m, m)
        yy = y[y_ind,:]

        boot.append(np.power(dist.sw(xx, yy, r=r, delta=delta, N=N, nq=nq, theta=theta), r) - SW_hat)

    q1 = np.quantile(boot, alpha/2)
    q2 = np.quantile(boot, 1-alpha/2)

    SW_lower = np.max([SW_hat - q2, 0])
    SW_upper = SW_hat - q1

    return np.power(SW_lower, 1/r), np.power(SW_upper, 1/r)


def pretest(x, y, r=2, delta=0.1, alpha=0.05, mode="DKW", N=500, B=500, nq=1000, theta = None):
    """ Pretesting confidence interval, using a combination of a two-sample 
        and a test for the maximum gap between sample points.
    
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
        alpha : float, optional
            real number between 0 and 1, such that 1-alpha is the level of the confidence interval
        mode : str, optional
            either "DKW" to use a confidence interval based on the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality [1,2]
            or "rel_VC" to use a confidence interval based on the relative Vapnik-Chervonenkis (VC) inequality [3]
        N : int, optional
            number of Monte Carlo replications to use for the unit sphere integral approximation
        B : int, optional
            number of bootstrap replications
        nq : int, optional
            number of quantiles to use in Monte Carlo integral approximations

        Returns
        -------
        l : float
            lower confidence limit

        u : float
            upper confidence limit

        References
        ----------

        .. [1] Dvoretzky, Aryeh, Jack Kiefer, and Jacob Wolfowitz. 
               "Asymptotic minimax character of the sample distribution function and 
               of the classical multinomial estimator." The Annals of Mathematical Statistics (1956): 642-669.

        .. [2] Massart, Pascal. "The tight constant in the Dvoretzky-Kiefer-Wolfowitz inequality." The Annals of Probability (1990): 1269-1283.

        .. [3] Vapnik, V., Chervonenkis, A.: On the uniform convergence of relative frequencies of events to
               their probabilities. Theory of Probability and its Applications 16 (1971) 264–280.

    """
    n = x.shape[0]
    m = y.shape[0]

    if x.ndim == 1:
        C_exact = exact_1d(x, y, r=r, delta=delta, alpha=alpha/2.0, mode=mode, nq=nq)
        arr_x = np.array(x).reshape([-1,1])
        arr_y = np.array(y).reshape([-1,1])

    else:
        C_exact = mc_sw(x, y, r=r, delta=delta, alpha=alpha/2.0, N=N, nq=nq, theta=theta)
        arr_x = x
        arr_y = y

    if C_exact[0] == 0:
        print("null")
        return C_exact

    print("boot")
    if x.ndim == 1:
        return bootstrap_1d(x, y, r=r, delta=delta, alpha=alpha/2.0, B=B, nq=nq)
    
    return bootstrap_sw(x, y, r=r, delta=delta, alpha=alpha/2.0, B=B, N=N, nq=nq, theta=theta)


#for i in range(15):
#    #x = np.random.beta(a=2, b=3, size=400)
#    #x = np.concatenate([np.repeat(1, 50), np.repeat(2, 330)]) #np.random.uniform(1, 2, size=400)
#    #y = np.random.uniform(1, 2, size=400)
#
#    
#    x1 = np.concatenate([np.repeat(1, 500), np.repeat(2, 500)]) #np.random.uniform(1, 2, size=400)
#    x2 = np.concatenate([np.repeat(3, 500), np.repeat(5, 500)])
#
#    x = np.random.multivariate_normal([0,0 ], np.identity(2), size=800)#np.vstack([x1,x2]).reshape([1000, 2])
#
#    y = np.random.multivariate_normal([0,0], np.identity(2), size=800)
#
#
#    print("Pretest: ", pretest(x, y, B=5))


