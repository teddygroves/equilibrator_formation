import pandas as pd
import arviz as az
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.style.use('sparse.mplstyle')
from run_model import get_infd, prepare_toy_data, PATHS
import os

    
def do_scatter():
    f, ax = plt.subplots(figsize=[20, 20])
    ax.scatter(d[0.5], d['obs'], marker='+')
    ax.vlines(d[0.5], d[0.05], d[0.95], zorder=0, color='tab:orange')
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    return f, ax


def main():
    infd = az.from_netcdf('model_nc.nd')

    log_sigma = infd.posterior['log_sigma'].to_series()
    log_tau = infd.posterior['log_tau'].to_series()
    fe = infd.posterior['fe_z'].to_series().unstack()
    gfe = infd.posterior['gfe_z'].to_series().unstack()
    mu_gfe = infd.posterior['mu_gfe'].to_series()
    sd_gfe = infd.posterior['mu_gfe'].to_series()

    post = (
        gfe
        .join(fe, rsuffix='_fe', lsuffix='_gfe')
        .join(log_tau, rsuffix='_log_tau')
        .join(log_sigma, rsuffix='_log_sigma')
        .join(mu_gfe, rsuffix='_mu_gfe')
        .join(sd_gfe, rsuffix='sd_gfe')
    )
