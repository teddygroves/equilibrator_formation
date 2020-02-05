import pandas as pd
import arviz as az
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.style.use('sparse.mplstyle')

INFD_PATH = 'infd.nd'

def load_infd():
    return az.from_netcdf(INFD_PATH)


    
def do_scatter():
    f, ax = plt.subplots(figsize=[20, 20])
    ax.scatter(d[0.5], d['obs'], marker='+')
    ax.vlines(d[0.5], d[0.05], d[0.95], zorder=0, color='tab:orange')
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    return f, ax
