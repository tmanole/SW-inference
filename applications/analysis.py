import numpy as np
from generate import *

import sys
sys.path.insert(0,"..")

from ci import *

import time
import multiprocessing as mp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ns', '--nsim', default=10000, type=int, help='Number of simulated points.')
parser.add_argument('-n', '--n', default=2000, type=int, help='Sample size.')
parser.add_argument('-np','--nproc', default=8, type=int, help='Number of processes to run in parallel.')
parser.add_argument('-ms','--misspec', default=False, type=int, help='Misspecified model?')
args = parser.parse_args()

misspecified = args.misspec

if not misspecified:
    n_sim = args.nsim
    n = args.n

    alpha1_range = np.linspace(18, 26, 15)    
    alpha2_range = np.linspace(2, 22, 15)     
    beta1_range  = np.linspace(4, 4, 1)       
    beta2_range  = np.linspace(4.5, 4.5, 1)   
    mu_range     = np.linspace(250, 400, 101) 
    sigma_range  = np.linspace(0.1, 0.4, 4)   
    gamma_range  = np.linspace(0.15, 0.15, 1) 

else:
    n_sim = 10000
    n = 2000

    alpha1_range = np.linspace(19, 28, 15)    
    alpha2_range = np.linspace(15, 30, 15)    
    beta1_range  = np.linspace(2, 4, 1)       
    beta2_range  = np.linspace(2, 4.5, 1)     
    mu_range     = np.linspace(250, 400, 101) 
    sigma_range  = np.linspace(0.1, 0.4, 4)   
    gamma_range  = np.linspace(0.15, 0.15, 1) 


temp = np.meshgrid(alpha1_range, alpha2_range, beta1_range, beta2_range, mu_range, sigma_range, gamma_range)

theta_grid = np.array(temp).T.reshape(-1, 7)

grid_size = theta_grid.shape[0]
print("Grid Size: ", grid_size)

# Generate true sample. 
y = rtoggle(n, alpha1=22,
               alpha2=12,
               beta1 =4,
               beta2 =4.5,
               mu    =325,
               sigma =0.25,
               gamma =0.15)
   
start_time = time.time()

def process_chunk(bound):   
    cs = np.empty((bound[1]-bound[0], 3))

    # Generate samples over a grid. 
    for j in range(bound[0], bound[1]):
        sim = rtoggle(n_sim, 
                       alpha1=theta_grid[j, 0],
                       alpha2=theta_grid[j, 1],
                       beta1 =theta_grid[j, 2],
                       beta2 =theta_grid[j, 3],
                       mu    =theta_grid[j, 4],
                       sigma =theta_grid[j, 5],
                       gamma =theta_grid[j, 6])

        # Confidence interval for the one-dimensional 1-Wasserstein distance.
        c = exact_1d(y, sim, r=1)

        cs[j-bound[0],0] = j
        cs[j-bound[0],1] = c[0]
        cs[j-bound[0],2] = c[1]
        print(j-bound[0])

    return cs

n_proc = 8
proc_chunks = []

Del = grid_size // n_proc

for i in range(n_proc):
    if i == n_proc-1:
        proc_chunks.append(( (n_proc-1) * Del, grid_size) )

    else:
        proc_chunks.append(( (i*Del, (i+1)*Del ) ))


with mp.Pool(processes=n_proc) as pool:
    proc_results = [pool.apply_async(process_chunk,
                                     args=(chunk,))
                    for chunk in proc_chunks]

    result_chunks = [r.get() for r in proc_results]

final_cs = result_chunks[0]

for i in range(1, n_proc):
    print(result_chunks[i])
    final_cs = np.vstack((final_cs, result_chunks[i]))

print(final_cs)

if not misspecified:
    np.save("well_specified_inds_n" + str(n) + "_m" + str(n_sim) + + "_time" + str(time.time() - start_time) ".npy", np.array(final_cs))

else:
    np.save("misspecified_inds_n" + str(n) + "_m" + str(n_sim) + + "_time" + str(time.time() - start_time) ".npy", np.array(final_cs))

