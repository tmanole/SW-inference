import numpy as np

from plotting_params import *

from matplotlib.ticker import FormatStrFormatter

#import matplotlib
#import matplotlib.pyplot as plt
#
#matplotlib.rcParams.update({'font.size': 17})
#
#matplotlib.rc('xtick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=20) 
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#
#matplotlib.rcParams['mathtext.fontset'] = 'stix'
#matplotlib.rcParams['font.family'] = 'STIXGeneral'


ns = [600, 900, 1200, 1500]
methods = ["pretest", "boot", "exact"]
legend = [True, False]
n_models= 5

"""
lw = 4
lw_thick =9 
grid_alpha = 0.3
"""


reps = 100


plt.figure(figsize=(10, 10))

for i in range(1, n_models+1):
    ax_i = i - 1
    plot_time = False #np.any((model == "model6", model == "model5"))

    if plot_time:
        numplots = 3

    for metric in ["coverage", "lengths", "elapsed"]:
        for method in methods:
            mat = np.load("results/model" + str(i) + "/" + method + "/" + metric + ".npy")
            metric_average = np.sum(mat, axis=0)/reps

            if method == "exact":
                this_lw = lw
                lab = "Finite Sample"
                col = "C0"

            if method == "boot":
                this_lw = lw
                lab = "Bootstrap"
                col = "C2"

            if method == "pretest":
                this_lw = lw_thick
                lab = "Hybrid"
                col = "C1"
            
    #        yerr = np.std(mat/500, axis=0) #np.quantile(mat/500, q=0.1, axis=0)

    #        plt.errorbar(ns, coverage, label=method, yerr=yerr)

            if metric == "coverage":
                yerr = np.std(mat/reps, axis=0)

            else:
                yerr = np.std(mat, axis=0)

            print(method, metric, metric_average, yerr)
        
            if method == "pretest":
                plt.errorbar(ns, metric_average, label=lab, linewidth=this_lw, yerr=yerr, color=col)
        
            else:
                plt.errorbar(ns, metric_average, label=lab, linewidth=this_lw, yerr=yerr, color=col)

        plt.grid(True, alpha=grid_alpha)

        plt.xlabel("$n$")

        if metric == "coverage":
            loc = "center right"
            plt.ylabel("Coverage (\%)")
            plt.plot(ns, np.repeat(0.95, len(ns)), linestyle="--", linewidth=lw_thick, color = "C3")
            plt.ylim([0.74, 1.03])
    
        if metric == "lengths":
            loc = "upper right"
            plt.ylabel("Average Length")
            plt.ylim(bottom=0)    

            if i == 2:
                plt.ylim(top=1.2)


        if metric == "elapsed":
            loc = "center right"
            plt.ylabel("Average Runtime (Seconds)")

            if i in [2, 4]:
                plt.ylim(bottom=-0.05)

            else:
                plt.ylim(bottom=-10)

        ax = plt.axes()
        ax.set_xticks(ns)

        if metric == "length":
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        plt.savefig("plots/model" + str(i) + "_" + metric + ".pdf")
     
        plt.legend(loc=loc, title="Methods")#, framealpha=1)

        plt.savefig("plots/model" + str(i) + "_" + metric + "_leg.pdf")
    

 
        plt.clf()




"""
    for method in methods:
        mat = np.load("results/model" + str(i)  + "/" + method + "/lengths.npy")

        length = np.nansum(mat, axis=0)/reps 

        if method == "exact":
            lab = "DKW-Wr"

        if method == "boot":
            lab = "Bootstrap"

        if method == "pretest":
            lab = "Hybrid"


        yerr = np.std(mat, axis=0)
        #yerr = np.quantile(mat, q=0.65, axis=0)
        print(yerr)
        print(length)
        plt.errorbar(ns, length, label=lab, yerr=yerr, linewidth=lw)
      
    plt.ylim(bottom=0)  
    plt.xlabel("n")
    plt.ylabel("Average Length")
#    plt.title("Average Length")
    plt.legend(loc="upper right", title="Methods", framealpha=1)
    plt.grid(True)

    if plot_time:
        c = plt.subplot(1, numplots, 3)

        for method in methods:
            mat = np.load(model + "/" + method + "/time.npy")
       
            time = np.nansum(mat, axis=0)/reps
            plt.plot(ns, time, linewidth=lw)

        plt.xlabel("n")
        plt.title("Time (seconds)")
        
    plt.savefig("plot.pdf")
    plt.clf()
#plt.show()

#plt.savefig("../../../workshop/plot.pdf")


"""
