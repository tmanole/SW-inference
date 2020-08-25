import matplotlib
import matplotlib.pyplot as plt

fs = 15

matplotlib.rcParams.update({'font.size': fs})

matplotlib.rc('xtick', labelsize=fs)
matplotlib.rc('ytick', labelsize=fs)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


lw = 2
grid_alpha = 0.3

plt.figure(figsize=(10, 10))


