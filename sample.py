"""Get samples from the model at MODEL_PATH and the data at DATA_PATH."""


from cmdstanpy import CmdStanModel
from typing import List
import os

MODEL_PATH = 'stan_code/model_simple.stan'
DATA_PATH = 'data/input_data_non_default_ionic_strength.json'
OUTPUT_DIR = './samples/standard_dg'
SAMPLE_KWARGS = {
    'output_dir': OUTPUT_DIR,
    'warmup_iters': 50,
    'chains': 2,
    'sampling_iters': 50,
    'max_treedepth': 15,
    'save_warmup': True,
    'adapt_delta': 0.9,
    # 'metric': 'dense'
}
DELETE_PREVIOUS_OUTPUTS = False


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
        delete_outputs(OUTPUT_DIR, files_to_keep)

