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
