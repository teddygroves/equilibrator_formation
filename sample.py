"""Get samples from the model at MODEL_PATH and the data at DATA_PATH."""

import arviz as az
import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel
from typing import List
import os

MODEL_PATH = 'stan_code/model_nc.stan'
DATA_PATH = 'data/input_data_toy.json'
OUTPUT_PATH = './samples/toy_data'
SAMPLE_KWARGS = {
    'output_dir': './samples/toy_data',
    'warmup_iters': 20,
    'sampling_iters': 10,
    'max_treedepth': 10,
    'save_warmup': True
}
DELETE_PREVIOUS_OUTPUTS = True


def delete_outputs(target_path: str, files_to_keep: List[str]):
    """Delete output files from target_path."""
    print("Deleting old output files:")
    for filename in os.listdir(target_path):
        if filename not in files_to_keep and filename[-4:] in ['.txt', '.csv']:
            print('\t' + filename)
            os.remove(os.path.join(target_path, filename))


if __name__ == '__main__':
    model = CmdStanModel(MODEL_PATH)
    fit = model.sample(DATA_PATH, **SAMPLE_KWARGS)
    print(fit.diagnose())
    if DELETE_PREVIOUS_OUTPUTS:
        files_to_keep = list(map(os.path.basename, fit.runset.csv_files))
        delete_outputs(OUTPUT_PATH, files_to_keep)

