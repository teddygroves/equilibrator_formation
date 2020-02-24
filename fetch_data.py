from equilibrator_cache import create_compound_cache_from_quilt, Reaction, R, Q_, FARADAY
from equilibrator_cache.exceptions import MissingDissociationConstantsException
from prepare_data import get_stoichiometric_matrix
import numpy as np
import os
import pandas as pd
import pint
import quilt
import warnings

MEASUREMENT_COLS = [
    'method',
    'eval',
    'enzyme_name',
    'temperature',
    'ionic_strength',
    'p_h',
    'p_mg',
    'formula',
    'stoichiometry',
    'reference',
    'K_prime',
    'standard_dg',
    'standard_dg_default',
    'standard_dg_prime'
]
OUTPUT_DIR = 'data/attempt_2'
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
    redox_measurements = (
        pkg.train.redox()
        .copy()
        .assign(
            method='redox',
            eval='redox',
            ccid_ox=lambda df: df['CID_ox'].apply(ccache.get_compound).apply(lambda c: int(c.id)),
            ccid_red=lambda df: df['CID_red'].apply(ccache.get_compound).apply(lambda c: int(c.id)),
            reaction=lambda df: df.apply(lambda row: Reaction({
                ccache.get_compound(row['CID_ox']): -1,
                ccache.get_compound(row['CID_red']): 1
            }), axis=1),
            stoichiometry=lambda df: df.apply(lambda row: {
                row['ccid_ox']: -1, row['ccid_red']: 1
            }, axis=1),
            formula=lambda df: df['CID_ox'].astype(str) + ' = ' + df['CID_red'].astype(str),
            delta_e=lambda df: df['nH_red'] - df['nH_ox'] - df['charge_red'] + df['charge_ox'],
            standard_dg_prime=lambda df: -FARADAY.m * df['standard_E_prime'] * df['delta_e'],
        )
    )
    formation_measurements = (
        pkg.train.formation_energies_transformed()
        .dropna(subset=['standard_dg_prime'])
        .copy()
        .assign(
            method='formation',
            eval='formation',
            formula=lambda df: '+ ' + df['cid'],
            ccid=lambda df: df['cid'].apply(lookup_compound, ccache=ccache),
            reaction=lambda df:
            df['ccid'].apply(
                lambda c: Reaction({ccache.get_compound_by_internal_id(c): 1})
            ),
            stoichiometry=lambda df: [{int(ccid): 1} for ccid in df['ccid'].values]
        )
    )
    tecrdb_measurements = (
        pkg.train.TECRDB().copy()
        .replace('nan', np.nan)
        .assign(
            formula=lambda df: df['reaction'],
            reaction=lambda df: df['formula'].apply(
                lambda f: Reaction.parse_formula(ccache.get_compound, f)
            ),
            stoichiometry=lambda df:
            df['formula'].apply(get_stoichiometry_from_formula, ccache=ccache),
            standard_dg_prime=lambda df: -R*df['temperature'] * np.log(df['K_prime'])
        )
    )
    measurements = (
        pd.concat(
            [redox_measurements, formation_measurements, tecrdb_measurements], ignore_index=True
        )
        .assign(
            standard_dg=lambda df: df.apply(transform_dg, ccache=ccache, axis=1),
            standard_dg_default=lambda df:
                df.fillna({'ionic_strength': 0.25})
                .apply(transform_dg, ccache=ccache, axis=1),
        )
        [MEASUREMENT_COLS]
    )
    cccompounds = list(map(
        ccache.get_compound_by_internal_id,
        measurements['stoichiometry'].map(lambda d: list(d.keys())).explode().unique()
    ))
    formation_energy_cc = pd.Series(
        pkg.parameters.dG0_cc(),
        index=pkg.parameters.train_S().index,
        name='formation_energy_cc'
    )
    compounds = pd.DataFrame.from_records(
        {'mass': c.mass,
         'compound_id': c.id,
         'inchi_key': c.inchi_key}
        for c in cccompounds
    ).join(formation_energy_cc, on='compound_id')
    reactions = (
        measurements.groupby('formula')['stoichiometry'].first()
        .reset_index()
        .assign(reaction_id=lambda df: range(1, len(df) + 1))
    )
    measurements = (
        measurements.merge(reactions[['formula', 'reaction_id']], on='formula')
    )
    S = get_stoichiometric_matrix(measurements)
    standard_dg_cc = (S.T @ formation_energy_cc.reindex(S.index)).rename('standard_dg_cc')
    measurements = measurements.join(standard_dg_cc, on='reaction_id')
    measurements.to_csv(os.path.join(OUTPUT_DIR, 'measurements.csv'))
    compounds.to_csv(os.path.join(OUTPUT_DIR, 'compounds.csv'))
    reactions.to_csv(os.path.join(OUTPUT_DIR, 'reactions.csv'))


if __name__ == '__main__':
    main()
