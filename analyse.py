from equilibrator_cache import create_compound_cache_from_quilt
import quilt
import pandas as pd
import arviz as az
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from itertools import combinations
plt.style.use('sparse.mplstyle')
import numpy as np
import os
from prepare_data import get_S, get_G, INPUT_DIR
import json


SAMPLE_DIR = 'samples/campfire/'
DATA_IN_JSON = 'data/input_data_non_default_ionic_strength.json'
MEASUREMENT_PATH = 'data/measurements_non_default_ionic_strength.csv'


def tidy_zeros(df):
    return df.mask(df == 0).stack().unstack().fillna(0)


def main():
    data_in = json.load(open(DATA_IN_JSON, 'r'))
    measurements = pd.read_csv(MEASUREMENT_PATH)
    compounds = pd.read_csv('data/compounds_cc.csv', index_col=0)
    groups = pd.read_csv('data/groups_cc.csv', index_col=0)
    S_in = pd.read_csv(os.path.join(INPUT_DIR, 'stoichiometry_cc.csv'), index_col=0)
    G_in = pd.read_csv(os.path.join(INPUT_DIR, 'group_incidence_cc.csv'), index_col=0)
    S_in.columns = map(int, S_in.columns)
    G_in.columns = map(int, G_in.columns)
    S = get_S(S_in, measurements)
    G = get_G(G_in, S)
    csvs = [
        os.path.join(SAMPLE_DIR, f)
        for f in os.listdir(SAMPLE_DIR) if f.endswith('.csv')
    ]
    infd = az.from_cmdstan(
        csvs,
        log_likelihood='log_lik',
        posterior_predictive='y_rep',
        coords={
            'compound': S.index,
            'reaction': S.columns,
            'group': G.columns,
        },
        dims={
            'fe_c_z': ['compound'],
            'compound_formation_energy': ['compound'],
            'group_formation_energy': ['group'],
            'standard_delta_g': ['reaction']
        }
    )
    loo_results = az.loo(infd, pointwise=True, scale='log')
    fe = pd.DataFrame({
        'low': infd.posterior['compound_formation_energy'].quantile(0.05, dim=['chain', 'draw']).values,
        'med': infd.posterior['compound_formation_energy'].quantile(0.5, dim=['chain', 'draw']).values,
        'high': infd.posterior['compound_formation_energy'].quantile(0.95, dim=['chain', 'draw']).values
    }, index=S.index)
    compounds = compounds.join(fe, on='compound_id')
    gfe = pd.DataFrame({
        'low': infd.posterior['group_formation_energy'].quantile(0.05, dim=['chain', 'draw']).to_series(),
        'med': infd.posterior['group_formation_energy'].quantile(0.5, dim=['chain', 'draw']).to_series(),
        'high': infd.posterior['group_formation_energy'].quantile(0.95, dim=['chain', 'draw']).to_series(),
    })
    gfe.index.name = 'group_id'
    groups = groups.join(gfe, on='group_id')
    d = pd.DataFrame({
        'loo': loo_results.loo_i,
        'khat': loo_results.pareto_k,
        'low': infd.posterior_predictive['y_rep'].quantile(0.025, dim=['chain', 'draw']).values,
        'med': infd.posterior_predictive['y_rep'].quantile(0.5, dim=['chain', 'draw']).values,
        'high': infd.posterior_predictive['y_rep'].quantile(0.975, dim=['chain', 'draw']).values
    }, index=measurements.index)
    measurements = measurements.join(d)

    # plot predictions
    f, axes = plt.subplots(1, 2, figsize=[12.5, 7])
    for ax, df in zip(
            axes,
            [measurements.loc[measurements['eval'] != 'formation'], measurements]
    ):
        sctr = ax.scatter(df['standard_dg'], df['med'],
                          label='posterior median', c=df['khat'], cmap=plt.cm.viridis)
        line = ax.plot(df['standard_dg'], df['standard_dg'], color='r', label='y=x')
        ax.vlines(df['standard_dg'], df['low'], df['high'],
                  color='grey', zorder=0, label='5-95% interval')
        ax.set(xlabel='Observed Value', ylabel='Predicted Value')
    cb = plt.colorbar(sctr, ax=ax)
    cb.set_label('khat statistic')
    axes[1].set_title('All measurements', y=0.86)
    axes[0].set_title('Measurements of non-formation reactions', y=0.86)
    axes[1].legend(frameon=False, loc='lower right')
    f.suptitle("Posterior predictive distributions", fontsize=16, x=0.07,
               horizontalalignment='left')
    plt.tight_layout()
    plt.savefig('plots/standard_dg_predictions.png', facecolor=f.get_facecolor())

    # plot differences with component contribution
    f, axes = plt.subplots(1, 2, figsize=[12.5, 7])
    for ax, df, y in zip(
            axes,
            [compounds, groups],
            ['formation_energy_cc', 'formation_energy_gc']):
        sctr = ax.scatter(
            df[y],
            df['med'],
            label='posterior median',
        )
        ax.plot(df[y], df[y], color='r', label='y=x')
        vlines = ax.vlines(
            df[y],
            df['low'],
            df['high'],
            color='grey',
            zorder=0,
            label='90% interval'
        )
        ax.set(
            ylabel='Formation Energy (Structured Bayesian Regression)'
        )
    for i, row in groups.iterrows():
        y = row['formation_energy_gc']
        if (y > row['high'] + 300) or (y < row['low'] - 300):
            axes[1].text(row['formation_energy_gc'], row['med'], row['name'],
                         fontsize=7)
    axes[1].legend(frameon=False, loc='lower right')
    axes[0].set_xlabel('Formation Energy (Component Contribution)')
    axes[1].set_xlabel('Formation Energy (Group Contribution)')
    axes[0].set_title('Compounds', y=0.86)
    axes[1].set_title('Groups', y=0.86)
    f.suptitle("Comparison of Formation energy estimates: Structured Bayesian regression vs Component Contribution")
    plt.savefig('plots/formation_energy_comparison.png', facecolor=f.get_facecolor())

if __name__ == '__main__':
    main()
