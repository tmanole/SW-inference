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
parser.add_argument('-mod', '--model', default=1, type=int, help='Model number.')
parser.add_argument('-meth', '--method', default="exact_DKW", type=str, help='Method name.')
parser.add_argument('-np','--nproc', default=8, type=int, help='Number of processes to run in parallel.')
parser.add_argument('-del', '--delta', default=0.1, type=float, help='Trimming constant.')
parser.add_argument('-a', '--alpha', default=0.1, type=float, help='Trimming constant.')
parser.add_argument('-B', '--B', default=1000, type=int, help='Number of bootstrap replications.')
parser.add_argument('-r', '--r', default=2, type=int, help='Order of the Wasserstein distance.')
parser.add_argument('-N', '--N', default=500, type=int, help='Monte Carlo replications over unit sphere.')
parser.add_argument('-nq', '--nq', default=500, type=int, help='Monte Carlo replications over [delta,1-delta].')
parser.add_argument('-reps', '--reps', default=200, type=int, help='Number of simulation replications.')
args = parser.parse_args()

print("Args: ", args)

model = args.model
method = args.method
n_proc = args.nproc
r = args.r
B = args.B
N = args.N
nq = args.nq
delta = args.delta
alpha = args.alpha
reps = args.reps

gen_x_list = [generate_x_model1, generate_x_model2, generate_x_model3, generate_x_model4, generate_x_model5]
gen_y_list = [generate_y_model1, generate_y_model2, generate_y_model3, generate_y_model4, generate_y_model5]

generate_x = gen_x_list[model-1]
generate_y = gen_y_list[model-1]

if model in [1,5]:
    d = 2

elif model == 3:
    d = 3

else:
    d = 1

ns = [600, 900, 1200, 1500]

def process_chunk(ran, n, truth):
    """ Helper function for running a given range of simulations, for a given method and sample size,
        in parallel. """
    low = ran[0]
    high = ran[1]

    coverage = np.full((high-low), -1.0)
    lengths  = np.full((high-low), -1.0)
    elapsed  = np.full((high-low), -1.0)

    for rep in range(low, high):
        print(rep)
        start_time = time.time()

#        if fix:
#            theta = np.random.multivariate_normal(np.repeat(0, d), np.identity(d), size=N)
#            theta = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, theta)
#
#        else:
#            theta = None
#
        np.random.seed(rep)
        x = generate_x(n)

        np.random.seed(rep)
        y = generate_y(n)

        if d == 1:

            if method == "exact":
                np.random.seed(rep)
                C = ci.exact_1d(x, y, r=r, delta=delta, alpha=alpha, mode="DKW", nq=nq)

            elif method == "pretest":
                np.random.seed(rep)
                C = ci.pretest(x, y, r=r, delta=delta, alpha=alpha, mode="DKW", B=B, nq=nq)

            elif method == "boot":
                np.random.seed(rep)
                C = ci.bootstrap_1d(x, y, r=r, delta=delta, alpha=alpha, B=B, nq=nq)

            else:
                raise Exception("Method not found for this dimension.")

        else:
            if method == "exact":
                np.random.seed(rep)
                C = ci.mc_sw(x, y, r=r, delta=delta, alpha=alpha, N=N, nq=nq, theta=None)

            elif method == "pretest":
                np.random.seed(rep)
                C = ci.pretest_sw(x, y, r=r, delta=delta, alpha=alpha, B=B, N=N, nq=nq, theta=None)

            elif method == "boot":
                np.random.seed(rep)
                C = ci.bootstrap_sw(x, y, r=r, delta=delta, alpha=alpha, B=B, N=N, nq=nq, theta=None)

            else:
                raise Exception("Method not found for this dimension.")

        coverage[rep-low] = C[0] <= truth and C[1] >= truth
        lengths [rep-low] = C[1] - C[0]

        if rep % 20 == 0:
            print("Repetition ", rep, ", Covered: ", C[0] <= truth and C[1] >= truth, ";  Interval is: ", C)

        elapsed[rep-low] = time.time() - start_time

    return coverage, lengths, elapsed


#def do_sim(generate_x, generate_y, d, method, path=None, truth=None, sim=1, r=2, N=500, B=500, nq=500, ns=[300, 600, 900, 1200, 1500], delta=0.1, alpha=0.05, reps=200, n_proc=8, fix=False):
print("----------------------------------Starting-------------------------------------")
print("Chose method " + str(method) + " with dimension " + str(d))

path = "results/model" + str(model) + "/" + method
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

x = generate_x(50000) 
y = generate_y(50000) 

if d==1:
    np.random.seed(0)
    truth = distances.w(x, y, r=r, nq=10000, delta=delta)

else:
    np.random.seed(0)
    truth = distances.sw(x, y, r=r, N=10000, nq=10000, delta=delta) 

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

