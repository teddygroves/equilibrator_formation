import pandas as pd
from cmdstanpy.utils import jsondump
from typing import Dict, Union, Iterable

LIKELIHOOD = True
N_RXN = 20
PATHS = {
    'measurements': 'data/measurements.csv',
    'stoichiometry': 'data/stoichiometry.csv',
    'group_incidence': 'data/group_incidence.csv',
}


def standardise(s: pd.Series) -> pd.Series:
    """Subtract a pd.Series's mean and divide by the standard deviation."""
    return s.subtract(s.mean()).div(s.std())


def prepare_data(
    measurements_in: pd.DataFrame,
    S_in: pd.DataFrame,
    G_in: pd.DataFrame,
    likelihood: bool = True,
    n_rxn: int = None
) -> Dict[str, Union[int, float, Iterable]]:
    """Get model input from input tables.
    
    :param measurements_in: table of measurements
    :param S_in: stoichiometric matrix
    :param G_in: group incidence matrix
    :param likelihood: should the `likelihood` field be 1 (True) or 0 (False)
    :param n_rxn: only the first n_rxn columns of S will be used in the model
    """
    if n_rxn is None:
        n_rxn = S_in.shape[1]
    S = (
        S_in.iloc[:, :n_rxn]
        .mask(lambda df: df == 0)
        .stack()
        .unstack()
        .fillna(0)
        .copy()
    )
    S.columns = map(int, S.columns)
    G_in.index = map(int, G_in.index)
    G = G_in.loc[S.index]
    measurements = (
        measurements_in.
        loc[lambda df: df['reaction_code'].isin(S.columns)]
        .copy()
    )
    rxn_codes = dict(zip(S.columns, range(1, len(S.columns) + 1)))
    measurements['new_reaction_code'] = measurements['reaction_code'].map(rxn_codes)
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


if __name__ == '__main__':
    # load dataframes
    measurements = pd.read_csv(PATHS['measurements'], index_col=0)
    S = pd.read_csv(PATHS['stoichiometry'], index_col=0)
    G = pd.read_csv(PATHS['group_incidence'], index_col=0)
    # turn dataframes into Stan input
    data = prepare_data(measurements, S, G, likelihood=LIKELIHOOD, n_rxn=N_RXN)
    # write output
    outfile_name = 'input_data.json' if N_RXN is None else 'input_data_toy.json'
    jsondump(outfile_name, data)

    
