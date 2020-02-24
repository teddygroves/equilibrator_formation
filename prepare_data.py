from cmdstanpy.utils import jsondump, rdump
import os
import numpy as np
import pandas as pd

INPUT_DIR = 'data'
OUTPUT_DIR = 'data'
JSON_OUTPUT_FILENAME = 'input_data_formation_only.json'
CSV_OUTPUT_FILENAME = 'measurements_standard_dg.csv'
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


def has_default_standard_dg(measurements: pd.DataFrame) -> pd.Series:
    return measurements['standard_dg_default'].notnull()


def filter_measurements(measurements: pd.DataFrame) -> pd.Series:
    return (
        # is_formation(measurements)
        is_good_measurement(measurements) &
        has_standard_dg(measurements)
    )


def get_stoichiometric_matrix(measurements: pd.DataFrame) -> pd.DataFrame:
    stoichs = measurements.groupby('reaction_id')['stoichiometry'].first()
    return (
        pd.DataFrame.from_records(stoichs.values, index=stoichs.index)
        .T
        .fillna(0)
        .rename_axis('compound_id')
    )


def get_group_incidence_matrix(
        measurements: pd.DataFrame,
        Graw: pd.DataFrame
) -> pd.DataFrame:
    S = get_stoichiometric_matrix(measurements)
    return (
        Graw
        .loc[S.index]
        .replace(0, np.nan)
        .stack()
        .unstack()
        .fillna(0)
        .copy()
    )




def main():
    measurements = pd.read_csv(os.path.join(INPUT_DIR, 'measurements.csv'))
    measurements['stoichiometry'] = measurements['stoichiometry'].apply(eval)
    measurements = measurements.loc[lambda df: filter_measurements(df)]
    compounds = pd.read_csv(os.path.join(INPUT_DIR, 'compounds.csv'))

    S = get_stoichiometric_matrix(measurements)

    Graw = pd.read_csv(GROUP_INCIDENCE_FILE, index_col=0)
    G = get_group_incidence_matrix(measurements, Graw)
    group_codes = dict(zip(G.columns, range(1, len(G.columns) + 1)))
    reaction_codes = dict(zip(S.columns, range(1, len(S.columns) + 1)))
    measurement_types = measurements['eval'].unique()
    eval_codes = dict(zip(measurement_types, range(1, len(measurement_types) + 1)))
    measurements['reaction_id_stan'] = measurements['reaction_id'].map(reaction_codes.get).values
    measurements['eval_stan'] = measurements['eval'].map(eval_codes.get).values
    stan_input = {
        'N_measurement_train': len(measurements),
        'N_measurement_test': len(measurements),
        'N_reaction_train': S.shape[1],
        'N_reaction_test': S.shape[1],
        'N_compound': S.shape[0],
        'N_group': G.shape[1],
        'G': G.values.tolist(),
        'N_measurement_type': len(measurement_types),
        'y_train': measurements['standard_dg'].values,
        'y_test': measurements['standard_dg'].values,
        'rxn_ix_train': measurements['reaction_id_stan'].values,
        'rxn_ix_test': measurements['reaction_id_stan'].values,
        'measurement_type_train': measurements['eval'].map(eval_codes).values,
        'measurement_type_test': measurements['eval'].map(eval_codes).values,
        'likelihood': int(LIKELIHOOD),
        'S_train': S.values.tolist(),
        'S_test': S.values.tolist(),
    }
    jsondump(os.path.join(OUTPUT_DIR, JSON_OUTPUT_FILENAME), stan_input)
    rdump(os.path.join(OUTPUT_DIR, JSON_OUTPUT_FILENAME.replace('json', 'R')), stan_input)
    measurements.to_csv(os.path.join(OUTPUT_DIR, CSV_OUTPUT_FILENAME))


if __name__ == '__main__':
    main()
