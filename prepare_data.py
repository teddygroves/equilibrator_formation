from cmdstanpy.utils import jsondump, rdump
import os
import numpy as np
import pandas as pd

INPUT_DIR = 'data'
OUTPUT_DIR = 'data'
JSON_OUTPUT_FILENAME = 'input_data_non_default_ionic_strength.json'
CSV_OUTPUT_FILENAME = 'measurements_non_default_ionic_strength.csv'
GROUP_INCIDENCE_FILE = 'data/group_incidence.csv'
LIKELIHOOD = True


# Filter functions

def is_good_measurement(measurements: pd.DataFrame) -> pd.Series:
    ok_evals = ['A', 'formation', 'redox']
    return measurements['eval'].isin(ok_evals)

def is_formation(measurements: pd.DataFrame) -> pd.Series:
    return measurements['method'] == 'formation'


def has_standard_dg(measurements: pd.DataFrame) -> pd.Series:
    return measurements['standard_dg'].notnull()


def has_default_ionic_strength(measurements: pd.DataFrame) -> pd.Series:
    return measurements['default_ionic_strength']

def has_default_standard_dg(measurements: pd.DataFrame) -> pd.Series:
    return measurements['standard_dg_default'].notnull()


def filter_measurements(measurements: pd.DataFrame) -> pd.Series:
    return (
        # is_formation(measurements)
        is_good_measurement(measurements) &
        ~has_default_ionic_strength(measurements)
    )


def tidy_zeros(df):
    return df.mask(df == 0).stack().unstack().fillna(0)


def get_S(S_in, measurements):
    reaction_ids = measurements['reaction_id'].unique()
    return S_in[reaction_ids].pipe(tidy_zeros).copy()


def get_G(G_in, S):
    return G_in.loc[S.index].pipe(tidy_zeros).copy()


def main():
    measurements_in = pd.read_csv(os.path.join(INPUT_DIR, 'measurements_cc.csv'), index_col=0)
    S_in = pd.read_csv(os.path.join(INPUT_DIR, 'stoichiometry_cc.csv'), index_col=0)
    G_in = pd.read_csv(os.path.join(INPUT_DIR, 'group_incidence_cc.csv'), index_col=0)
    compounds = pd.read_csv(os.path.join(INPUT_DIR, 'compounds_cc.csv'))
    reactions = pd.read_csv(os.path.join(INPUT_DIR, 'reactions_cc.csv'))
    S_in.columns = map(int, S_in.columns)
    S_in.columns.name = 'reaction_id'
    G_in.columns.name = 'group_id'

    measurements = measurements_in.loc[lambda df: filter_measurements(df)].copy()
    S = get_S(S_in, measurements)
    G = get_G(G_in, S)

    group_codes = dict(zip(G.columns, range(1, len(G.columns) + 1)))
    reaction_codes = dict(zip(S.columns, range(1, len(S.columns) + 1)))
    measurement_types = measurements['eval'].unique()
    eval_codes = dict(zip(measurement_types, range(1, len(measurement_types) + 1)))
    measurements['reaction_id_stan'] = measurements['reaction_id'].map(reaction_codes.get).values
    measurements['eval_stan'] = measurements['eval'].map(eval_codes.get).values
    stan_input = {
        'N_measurement': len(measurements),
        'N_reaction': S.shape[1],
        'N_compound': S.shape[0],
        'N_group': G.shape[1],
        'G': G.values.tolist(),
        'N_measurement_type': len(measurement_types),
        'y': measurements['standard_dg'].values,
        'rxn_ix': measurements['reaction_id_stan'].values,
        'measurement_type': measurements['eval'].map(eval_codes).values,
        'likelihood': int(LIKELIHOOD),
        'S': S.values.tolist(),
    }
    jsondump(os.path.join(OUTPUT_DIR, JSON_OUTPUT_FILENAME), stan_input)
    rdump(os.path.join(OUTPUT_DIR, JSON_OUTPUT_FILENAME.replace('json', 'R')), stan_input)
    measurements.to_csv(os.path.join(OUTPUT_DIR, CSV_OUTPUT_FILENAME))


if __name__ == '__main__':
    main()
