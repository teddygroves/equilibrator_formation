import arviz as az
import pandas as pd
import numpy as np
import quilt
from equilibrator_api import ccache
import cmdstanpy
from collections import namedtuple
from typing import Dict, Any, Union, Iterable


class InputData():
    def __init__(
        self,
        normalised_measurements,
        stoichiometry,
        group_incidence_matrix,
        qs,
        rs,
        rinvs,
        qg,
        rg,
        rinvg
    ):
        self.normalised_measurements = normalised_measurements
        self.stoichiometry = stoichiometry
        self.group_incidence_matrix = group_incidence_matrix
        self.qs = qs
        self.rs = rs
        self.rinvs = rinvs
        self.qg = qg
        self.rg = rg
        self.rinvg = rinvg
        self.N_compound, self.N_reaction = self.stoichiometry.shape
        _, self.N_group = group_incidence_matrix.shape
        

def get_data(install=False):
    if install:
        quilt.install("equilibrator/component_contribution", force=True)
    pkg = quilt.load("equilibrator/component_contribution")
    S = pkg.parameters.train_S()
    G = pkg.parameters.train_G()
    qs, rs = np.linalg.qr(S.T)
    qg, rg = np.linalg.qr(G)
    rpinvs = np.linalg.pinv(rs)
    rpinvg = np.linalg.pinv(rg)
    rpinvs_approx = np.where(rpinvs > 0.000000001, rpinvs, np.zeros(rpinvs.shape))
    rpinvg_approx = np.where(rpinvg > 0.000000001, rpinvg, np.zeros(rpinvg.shape))
    normalised_measurements = pd.Series(pkg.parameters.train_b(), index=S.columns)
    return InputData(
        normalised_measurements,
        S,
        G,
        qs,
        rs,
        rpinvs_approx,
        qg,
        rg,
        rpinvg_approx
    )


def get_stan_input(
        data: InputData,
        likelihood: bool = True
) -> Dict[str, Union[int, float, np.array]]:
    return {
        'N_reaction': data.N_reaction,
        'N_compound': data.N_compound,
        'N_group': data.N_group,
        'standard_delta_g_measured': data.normalised_measurements.values / 100,
        'ST': data.stoichiometry.T.values,
        'G': data.group_incidence_matrix.values,
        'QST': data.qs,
        'QG': data.qg,
        'RinvS': data.rinvs,
        'RinvG': data.rinvg,
        'likelihood': int(likelihood),
    }
    

def fit_model(data, likelihood=True, model_name='model'):
    stan_input = get_stan_input(data)
    cmdstanpy.utils.rdump('input_data.rdump', stan_input)
    model = cmdstanpy.CmdStanModel(model_name + '.stan')
    fit = model.sample(
        stan_input,
        output_dir='.',
        save_warmup=True,
        warmup_iters=200,
        sampling_iters=200
    )
    return fit


def get_latest_model_stem(model_name=None):
    if model_name is None:
        model_name = 'model'
    csv_filepaths = [i for i in os.listdir('.') if i[-4:] == '.csv']
    timestamp = str(max(map(lambda s: int(s.split('-')[1]), csv_filepaths)))
    return model_name + '-' + timestamp


def get_infd(model_stem):
    paths = [f'{model_stem}-{str(i)}.csv' for i in range(1, 5)]
    return az.from_cmdstan(
        posterior=paths,
        posterior_predictive='standard_delta_g',
        observed_data='./input_data.rdump',
        observed_data_var='standard_delta_g_measured',
        coords={
            'reaction_id': data.stoichiometry.columns,
            'compound_id': data.stoichiometry.index
        },
        dims={
            'formation_energy': ['compound_id'],
            'standard_delta_g': ['reaction_id'],
            'standard_delta_g_measured': ['reaction_id']
        }
    )

if __name__ == '__main__':
    model_name = 'model'
    likelihood = True
    data = get_data()
    fit = fit_model(data, likelihood=likelihood, model_name=model_name)
    print(fit.diagnose())
    infd = get_infd(get_latest_model_stem())
    infd.to_netcdf('infd.nd')

