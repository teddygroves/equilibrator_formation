"""Fetch component contribution data from quilt and put it in local csv files."""


import pandas as pd
import quilt
from component_contribution.linalg import LINALG

MEASUREMENT_COLS = [
    'method',
    'eval',
    'EC',
    'K',
    'K_prime',
    'temperature',
    'ionic_strength',
    'p_h',
    'p_mg'
]


def one_encode(s: pd.Series) -> dict:
    code_map = dict(zip(s.unique(), range(1, len(s.unique()) + 1)))
    return code_map, s.map(code_map)


def main():
    # load
    pkg = quilt.load("equilibrator/component_contribution")
    Sraw = pkg.parameters.train_S()
    tecrdb = pkg.train.TECRDB().reindex(Sraw.columns).copy()
    train_G = pkg.parameters.train_G()

    # transform
    S_uniq, P_col = LINALG._col_uniq(Sraw.values)
    rxn_codes = range(1, S_uniq.shape[1] + 1)
    rxn_ix = pd.Series(
        [i + 1 for r in P_col for i, x in enumerate(r) if x != 0],
        name='reaction_code'
    )
    cpd_to_code, cpd_codes = one_encode(Sraw.index)
    cpd_codes = pd.Series(cpd_codes, name='compound_code')
    compounds = pd.Series(cpd_to_code).reset_index()
    compounds.columns = ['equilibrator_id', 'stan_code']
    S_uniq = pd.DataFrame(S_uniq, index=cpd_codes, columns=rxn_codes)
    msts = tecrdb[MEASUREMENT_COLS].copy()
    msts['reaction_code'] = rxn_ix.values
    msts['standard_dg'] = pkg.parameters.train_b()
    msts['standard_dg_cc'] = Sraw.T @ pkg.parameters.dG0_cc()
    code_to_rxn = tecrdb.groupby(msts['reaction_code'])['reaction'].first()
    reactions = pd.DataFrame({
        'stan_code': rxn_codes,
        'equilibrator_id': code_to_rxn.values
    })
    group_to_code, group_codes = one_encode(train_G.columns)
    group_codes = pd.Series(group_codes, name='group_code')
    group_incidence = pd.DataFrame(
        train_G.values, index=S_uniq.index, columns=group_codes
    )
    groups = pd.Series(group_to_code).reset_index()
    groups.columns = ['equilibrator id', 'stan_code']

    # write csvs
    compounds.to_csv('data/compounds.csv')
    reactions.to_csv('data/reactions.csv')
    S_uniq.to_csv('data/stoichiometry.csv')
    msts.to_csv('data/measurements.csv')
    group_incidence.to_csv('data/group_incidence.csv')
    groups.to_csv('data/groups.csv')
    

if __name__ == '__main__':
    main()
