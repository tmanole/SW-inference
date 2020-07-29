import numpy as np
from models import *
import sys
import multiprocessing as mp
import argparse
import datetime
import pathlib
import time

sys.path.insert(0, "..")
import ci 
import distances 

from models import *

parser = argparse.ArgumentParser()
parser.add_argument('-asymp', '--asymp', default=1, type=int, help='Asymptotic experiment index.')
parser.add_argument('-J', '--finiteJ', default=False, type=bool, help='Is J(P) finite?')
parser.add_argument('-np','--nproc', default=8, type=int, help='Number of processes to run in parallel.')
parser.add_argument('-del', '--delta', default=0.1, type=float, help='Trimming constant.')
parser.add_argument('-eps', '--epsilon', default=0.1, type=float, help='Away-from-null parameter.')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Trimming constant.')
parser.add_argument('-r', '--r', default=2, type=int, help='Order of the Wasserstein distance.')
parser.add_argument('-nq', '--nq', default=2500, type=int, help='Monte Carlo replications over [delta,1-delta].')
parser.add_argument('-reps', '--reps', default=200, type=int, help='Number of simulation replications.')
args = parser.parse_args()

print("Args: ", args)

asymp = args.asymp
finiteJ = args.finiteJ
n_proc = args.nproc
r = args.r
nq = args.nq
delta = args.delta
epsilon = args.epsilon
alpha = args.alpha
reps = args.reps


spec = "infiniteJ"
if finiteJ:
    spec = "finiteJ"

if asymp == 1:
    r = 2
    path = "results/asymp1/" + spec + "/epsilon_" + str(epsilon) 

elif asymp == 2:
    epsilon = 0
    path = "results/asymp2/" + spec + "/r_" + str(r) 

else:
    sys.exit("Experiment index not recognized.")

ns = [250, 500, 750, 1000, 2000, 5000, 10000, 25000, 50000]

if finiteJ:
    def generate_x(n):
        return np.random.uniform(-5, 5, n)
    
    def generate_y(n):
        x = np.empty([n, 1])
    
        for i in range(n):
            u = np.random.uniform(size=1)
    
            if u < 0.5 + epsilon:
                x[i] = np.random.uniform(-5, 0, size=1)
    
            else:
                x[i] = np.random.uniform(0, 5, size=1)
    
        return x

else:
    def generate_x(n):
        x = np.empty(n)

        for i in range(n):
            u = np.random.uniform(0, 1, 1)

            if u < 0.5:
                x[i] = -5

            else:
                x[i] = 5

        return x

    def generate_y(n):
        y = np.empty(n)

        for i in range(n):
            u = np.random.uniform(0, 1, 1)

            if u < 0.5+epsilon:
                y[i] = -5

            else:
                y[i] = 5

        return y


def process_chunk(ran, n, truth):
    """ Helper function for running a given range of simulations, for a given method and sample size,
        in parallel. """
    low = ran[0]
    high = ran[1]

    coverage = np.full((high-low), -1.0)
    lengths  = np.full((high-low), -1.0)
    elapsed  = np.full((high-low), -1.0)

    for rep in range(low, high):
        start_time = time.time()

        np.random.seed(rep)
        x = generate_x(n)

        np.random.seed(rep)
        y = generate_y(n)

        np.random.seed(rep)
        C = ci.exact_1d(x, y, r=r, delta=delta, alpha=alpha, mode="DKW", nq=nq)

        coverage[rep-low] = C[0] <= truth and C[1] >= truth
        lengths [rep-low] = C[1] - C[0]

        if rep % 30 == 0:
            print("Repetition ", rep, ", Covered: ", C[0] <= truth and C[1] >= truth, ";  Interval is: ", C)

        elapsed[rep-low] = time.time() - start_time

    return coverage, lengths, elapsed


print("----------------------------------Starting-------------------------------------")
print("asymp:", asymp, ", r:", r, ", epsilon:", epsilon, ", finiteJ:", finiteJ)
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

x = generate_x(500000) 
y = generate_y(500000) 

np.random.seed(0)
truth = distances.w(x, y, r=r, nq=10000, delta=delta)

print("Truth: ", truth, "\n")

n_num = len(ns)

coverage = np.full((reps, n_num), -1.0)
lengths  = np.full((reps, n_num), -1.0)
elapsed  = np.full((reps, n_num), -1.0)

proc_chunks = []

for i in range(n_num):
    n = ns[i]

    print("---------------------------------------------------------------------------")
    print("Starting n=" + str(n))

    Del = reps // n_proc
    for j in range(n_proc):
        if j == n_proc-1:
            proc_chunks.append(( (n_proc-1) * Del, reps) )

        else:
            proc_chunks.append(( (j*Del, (j+1)*Del ) ))

    with mp.Pool(processes=n_proc) as pool:
        proc_results = [pool.apply_async(process_chunk, args=(chunk, ns[i], truth))
                        for chunk in proc_chunks]
        result_chunks = [r.get() for r in proc_results]

    for j in range(n_proc):
        if j == n_proc-1:
            coverage[((n_proc-1)*Del):reps,i] = result_chunks[j][0]
            lengths [((n_proc-1)*Del):reps,i] = result_chunks[j][1]
            elapsed [((n_proc-1)*Del):reps,i] = result_chunks[j][2]

        else:
            coverage[(j*Del):((j+1)*Del),i] = result_chunks[j][0]
            lengths [(j*Del):((j+1)*Del),i] = result_chunks[j][1]
            elapsed [(j*Del):((j+1)*Del),i] = result_chunks[j][2]

    print("Coverage for n=", n)
    print(np.sum(coverage, axis=0)/reps)

    np.save(path + "/coverage.npy", coverage)
    np.save(path + "/lengths.npy", lengths)
    np.save(path + "/elapsed.npy", elapsed)

