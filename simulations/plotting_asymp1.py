import numpy as np
from plotting_params_asymp import *

from matplotlib.ticker import FormatStrFormatter

label = "False"
x = []

i = 0

labs = [0.0, 0.1, 0.2, 0.3, 0.4]
ns = [250, 500, 750, 1000, 2000, 5000, 10000, 25000, 50000]

for model in ["finiteJ", "infiniteJ"]:
    plt.figure(figsize=(5, 5))
    for i in range(len(labs)):
        M = np.load("results/asymp1/" + model + "/epsilon_" + str(labs[i]) + "/lengths.npy")

        x = np.sum(M, axis=0)/100

#        plt.plot(ns, x, label=str(l))

        yerr = np.std(M, axis=0)
        plt.errorbar(ns, x, label=labs[i], yerr=yerr, linewidth=lw)

    plt.xlabel("$n$")
    plt.ylabel("Average Length")
    plt.legend(loc="upper right", title="$\Delta$")

    ax = plt.axes()

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#    plt.title("Average Length")
    plt.ylim(bottom=0)
    plt.grid(True, alpha=grid_alpha)
    plt.savefig("plots/asymp1_" + model + ".pdf")



#b.set_aspect("equal")

#plt.savefig("../../../workshop/plot.pdf")
