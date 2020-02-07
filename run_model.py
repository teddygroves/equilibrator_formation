import arviz as az
import pandas as pd
import numpy as np
import quilt
import cmdstanpy
from collections import namedtuple
from typing import Dict, Any, Union, Iterable
import os

PATHS = {
    'measurements': 'data/measurements.csv',
    'stoichiometry': 'data/stoichiometry.csv',
    'group_incidence': 'data/group_incidence.csv',
    'model': 'model_nc.stan'
}


def standardise(s):
    """Subtract a pd.Series's mean and divide by the standard deviation."""
    return s.subtract(s.mean()).div(s.std())

def prepare_data(measurements, S, G, likelihood=True):
    """Get full model input."""
    return {
        'N_measurement': len(measurements),
        'N_reaction': S.shape[1],
        'N_compound': S.shape[0],
        'N_group': G.shape[1],
        'y': measurements['standard_dg'].pipe(standardise).values,
        'rxn_ix': measurements['reaction_code'].values,
        'likelihood': int(likelihood),
        'S': S.values.tolist(),
        'G': G.values.tolist()
    }


def prepare_toy_data(measurements_in, S_in, G_in, likelihood=True):
    """
    Get model input for the first twenty reaction in the stoichiometric
    matrix.
    """
    S = S_in.iloc[:, :20].mask(lambda df: df == 0).stack().unstack().fillna(0).copy()
    S.columns = map(int, S.columns)
    G_in.index = map(int, G_in.index)
    G = G_in.loc[S.index]
    measurements = measurements_in.loc[lambda df: df['reaction_code'].isin(S.columns)].copy()
    new_rxn_codes = dict(zip(S.columns, range(1, len(S.columns) + 1)))
    measurements['new_reaction_code'] = measurements['reaction_code'].map(new_rxn_codes)
    return {
        'N_measurement': len(measurements),
        'N_reaction': S.shape[1],
        'N_compound': S.shape[0],
        'N_group': G.shape[1],
        'y': measurements['standard_dg'].pipe(standardise).values,
        'rxn_ix': measurements['new_reaction_code'].values,
        'likelihood': int(likelihood),
        'S': S.values.tolist(),
        'G': G.values.tolist()
    }


def get_latest_model_stem(model_path):
    """Find the filename up to '-n.csv' of the most recent cmdstan output."""
    model_name = model_path.split('.')[0]
    csv_filepaths = [
        i for i in os.listdir('.')
        if i[-4:] == '.csv' and i[:len(model_name)] == model_name
    ]
    timestamp = str(max(map(lambda s: int(s.split('-')[1]), csv_filepaths)))
    return model_name + '-' + timestamp


def get_infd(model_path, data_in):
    """Get an arviz InferenceData object."""
    model_stem = get_latest_model_stem(model_path)
    paths = [f'{model_stem}-{str(i)}.csv' for i in range(1, 5)]
    return az.from_cmdstan(
        posterior=paths,
        observed_data='./input_data.rdump',
        observed_data_var='y',
        coords={
            'reaction_id': range(1, len(data_in['S'][0]) + 1),
            'compound_id': range(1, len(data_in['S']) + 1),
            'group_id': range(1, len(data_in['G'][0]) + 1)
        },
        dims={
            'fe_z': ['compound_id'],
            'gfe_z': ['group_id'],
            'formation_energy': ['compound_id'],
            'standard_delta_g': ['reaction_id'],
        }
    )


if __name__ == '__main__':
    # load data
    measurements = pd.read_csv(PATHS['measurements'], index_col=0)
    S = pd.read_csv(PATHS['stoichiometry'], index_col=0)
    G = pd.read_csv(PATHS['group_incidence'], index_col=0)
    # get model input
    data = prepare_toy_data(measurements, S, G, likelihood=True)
    # save model input
    cmdstanpy.utils.jsondump('input_data.json', data)
    # compile model
    model = cmdstanpy.CmdStanModel(PATHS['model'])
    # sample
    fit = model.sample(
        data,
        output_dir='.',
        save_warmup=True,
        warmup_iters=2000,
        sampling_iters=500,
        max_treedepth=15,
        metric='dense'
    )
    # diagnose
    print(fit.diagnose())
    # get and save InferenceData
    infd = get_infd(PATHS['model'], data)
    infd.to_netcdf('infd.nd')

