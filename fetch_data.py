from equilibrator_cache import create_compound_cache_from_quilt, Reaction, R, Q_, FARADAY
from equilibrator_cache.exceptions import MissingDissociationConstantsException
from component_contribution.training_data import FullTrainingData
from component_contribution import ComponentContributionTrainer
from prepare_data import get_stoichiometric_matrix
import numpy as np
import os
import pandas as pd
import pint
import quilt
import warnings

QUANTITY_COLS = [
    'temperature',
    'ionic_strength',
    'p_h',
    'p_mg',
    'standard_dg_prime',
    'standard_dg'
]
OUTPUT_DIR = 'data'
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Q_([])


def tidy_zeros(df):
    return df.mask(df == 0).stack().unstack().fillna(0)


def lookup_compound(cid: str, ccache):
    r, rid  = cid.split(':')
    return ccache.get_compound_from_registry(r.lower(), rid).id


def transform_dg(row, ccache):
    # rxn = Reaction.parse_formula(ccache.get_compound, row['formula'])
    try:
        ddg_over_rt = row['reaction'].transform(
            p_h=Q_(row['p_h']),
            ionic_strength=Q_(row['ionic_strength'], "M"),
            temperature=Q_(row['temperature'], "K"),
        )
        return row['standard_dg_prime'] - (ddg_over_rt * R * row['temperature']).m
    except MissingDissociationConstantsException:
        return np.nan

        

def get_stoichiometry_from_formula(reaction: str, ccache):
    def parse_stoich(s, is_substrate):
        split = s.strip().split(' ')
        stoich = float(split[0]) if len(split) == 2 else 1.0
        if is_substrate:
            stoich *= -1
        return lookup_compound(split[-1], ccache), stoich
    out = {}
    for side_str, is_substrate in zip(reaction.split('='), [True, False]):
        cmpds = side_str.split('+')
        for cmpd in cmpds:
            ccid, stoich = parse_stoich(cmpd, is_substrate)
            out[ccid] = stoich
    return out


def main():
    pkg = quilt.load('equilibrator/component_contribution')
    ccache = create_compound_cache_from_quilt()
    td = FullTrainingData(ccache)

    group_df = pkg.parameters.group_definitions()
    G = ComponentContributionTrainer.group_incidence_matrix(td, group_df)
    G.index = map(lambda c: c.id, G.index)
    G.index.name = 'compound_id'

    m = td.reaction_df.copy()
    m['stoichiometry'] = m.apply(lambda row: {k.id: v for k, v in row['reaction'].items()}, axis=1)
    m['reaction_id'] = m['reaction'].apply(hash)
    for qcol in QUANTITY_COLS:
        m[qcol] = m[qcol].apply(lambda q: q.magnitude)
    S = td.stoichiometric_matrix.copy()
    S.columns = m['reaction_id']
    S.index = map(lambda c: c.id, S.index)
    S.index.name = 'compound_id'
    S_unique = S.T.groupby(level=0).first().T

    compounds = pd.DataFrame({
        'compound_id': map(lambda c: c.id, td.compounds),
        'inchi_key': map(lambda c: c.inchi_key, td.compounds),
        'mass': map(lambda c: c.mass, td.compounds),
        'formation_energy_cc': pkg.parameters.dG0_cc(),
        'formation_energy_rc': pkg.parameters.dG0_rc(),

    })
    reactions = pd.DataFrame({
        'reaction_id': map(hash, m['reaction'].unique()),
        'stoichiometry': map(lambda r: {
            k.id: v for k, v in r.sparse.items()
        }, m['reaction'].unique())
    })
    groups = pd.DataFrame({
        'name': [n  if type(n) == str else 'compound:' + str(n.id) for n in G.columns],
        'group_id': range(1, len(G.columns) + 1),
        'formation_energy_gc': pkg.parameters.dG0_gc(),
    })
    G.columns = groups['group_id']
    G.columns.name = 'group_id'

    m['standard_dg_cc'] = (S.T @ compounds['formation_energy_cc'].values).values
    m['standard_dg_rc'] = (S.T @ compounds['formation_energy_rc'].values).values
    m['standard_dg_gc'] = (S.T @ G @ groups['formation_energy_gc'].values).values

    
    S_unique.to_csv(os.path.join(OUTPUT_DIR, 'stoichiometry_cc.csv'))
    G.to_csv(os.path.join(OUTPUT_DIR, 'group_incidence_cc.csv'))
    m.to_csv(os.path.join(OUTPUT_DIR, 'measurements_cc.csv'))
    compounds.to_csv(os.path.join(OUTPUT_DIR, 'compounds_cc.csv'))
    reactions.to_csv(os.path.join(OUTPUT_DIR, 'reactions_cc.csv'))
    groups.to_csv(os.path.join(OUTPUT_DIR, 'groups_cc.csv'))


if __name__ == "__main__":
    main()
