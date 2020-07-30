import numpy as np
import matplotlib.pyplot as plt
import alphashape
from descartes import PolygonPatch
from scipy.spatial import ConvexHull, convex_hull_plot_2d

import matplotlib
matplotlib.rcParams.update({'font.size': 13}) 

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

alpha1_range = np.linspace(18, 26, 17)
alpha2_range = np.linspace(2, 22, 21)
beta1_range  = np.linspace(4, 4, 1)
beta2_range  = np.linspace(4.5, 4.5, 1)
mu_range     = np.linspace(250, 400, 101)
sigma_range  = np.linspace(0.1, 0.4, 4)
gamma_range  = np.linspace(0.15, 0.15, 1)

n_sims = [5000, 10000, 20000]

fig, ax = plt.subplots()

col=["C0", "C1", "C2"]
labs = ["5,000", "10,000", "20,000"]
n_i = -1


for n_sim in n_sims:
    n_i += 1

    temp = np.meshgrid(alpha1_range, alpha2_range, beta1_range, beta2_range, mu_range, sigma_range, gamma_range)

    theta_grid = np.array(temp).T.reshape(-1, 7)

    if n_i == 0:
        mats = np.load("results/well_specified_inds_n2000_m5000_time28277.654217.npy")

    elif n_i == 1:
        mats = np.load("results/well_specified_inds_n2000_m20000_time105523.319089.npy")    

    else:
        mats = np.load("results/well_specified_inds_n2000_m10000_time54734.667304.npy")

    wh = mats[:,1] == 0

    inds = mats[wh,0].tolist()
    inds = [int(ind) for ind in inds]

    alpha1 = theta_grid[inds,0]
    alpha2 = theta_grid[inds,1]
   
    points = []
    
    for i in range(alpha1.shape[0]):
        points.append((alpha1[i], alpha2[i]))
    
    alpha_shape = alphashape.alphashape(points, -2)

    print(alpha_shape)

    points = theta_grid[inds,0:2]
    hull = ConvexHull(points)

    j = 0
    for simplex in hull.simplices:
        if j == 0:
            plt.plot(points[simplex, 0], points[simplex, 1], c=col[n_i], ls='dashed', label=labs[n_i], lw=lw)#, 'k-')
            j = -1

        else:
            plt.plot(points[simplex, 0], points[simplex, 1], c=col[n_i], ls='dashed', lw=lw)

#    plt.title("0.95 Confidence Interval")
    


plt.plot([22], [12], "o", c="red")#, label="$\\theta_0$")


plt.legend(loc="lower left", title="$m$")#, ncol=4)
plt.grid(True, alpha=grid_alpha)
plt.xlim([19.5, 24])
plt.ylim([5, 22])
plt.xlabel(r"$\alpha_1$")
plt.ylabel(r"$\alpha_2$")

    
plt.savefig("results/well_specified.pdf")
    
