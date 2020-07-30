import numpy as np
import matplotlib.pyplot as plt
import alphashape
from descartes import PolygonPatch
from scipy.spatial import ConvexHull, convex_hull_plot_2d

import matplotlib
matplotlib.rcParams.update({'font.size': 17})

matplotlib.rc('xtick', labelsize=17)
matplotlib.rc('ytick', labelsize=17)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

lw = 2            # Line width
grid_alpha = 0.3 # Grid transparency

n_sim = 20000

fig, ax = plt.subplots()

col=["C0", "C1", "C2", "C3", "C4"]
#labs = ["m=5,000", "m=10,000", "m=20,000"]

alpha1_range = np.linspace(18, 28, 15)
alpha2_range = np.linspace(14, 26, 15)
beta1_range  = np.linspace(2, 4, 1)
beta2_range  = np.linspace(2, 4.5, 1)
mu_range     = np.linspace(250, 400, 101)
sigma_range  = np.linspace(0.1, 0.4, 4)
gamma_range  = np.linspace(0.15, 0.15, 1)

temp = np.meshgrid(alpha1_range, alpha2_range, beta1_range, beta2_range, mu_range, sigma_range, gamma_range)

theta_grid = np.array(temp).T.reshape(-1, 7)

inds = np.load("misspecified_inds_n2000_m10000_time33887.491714.npy")

inds_keep = []

grid_size = inds.shape[0]

epsilons=[5, 10, 20, 50]

itr = -1 

for eps in epsilons:

    itr += 1

    for k in range(grid_size):
        if inds[k,1] <= eps:
            inds_keep.append(int(inds[k,0]))

    points = theta_grid[inds_keep,0:2]
    hull = ConvexHull(points)


    j = 0
    for simplex in hull.simplices:
        if j == 0:
            plt.plot(points[simplex, 0], points[simplex, 1], c=col[itr], ls='dashed', label=str(eps), lw=lw)#, 'k-')
            j = -1

        else:
            plt.plot(points[simplex, 0], points[simplex, 1], c=col[itr], ls='dashed', lw=lw)

    


plt.plot([22], [12], "o", c="red")#, label="$\\theta_0$")
plt.plot(theta_grid[37221,0], theta_grid[37221,1], "o", c="blue")#, label="$\\theta_0$")

plt.legend(loc="lower right", title="$\epsilon$")
plt.grid(True, alpha=grid_alpha)
plt.xlim([19, 28])
plt.ylim([11, 25])
plt.xlabel(r"$\alpha_1$")
plt.ylabel(r"$\alpha_2$")

    
plt.savefig("misspecified.pdf")
    
