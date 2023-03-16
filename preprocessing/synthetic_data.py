import numpy as np
import torch
from preprocessing.utils import calc_hrtf
import scipy.signal as sps

PI_4 = np.pi / 4

def interpolate_synthetic_fft(config, features, cube, fs_original, edge_len):

    # Resample data so that training and validation sets are created at the same fs ('config.hrir_samplerate').
    number_of_samples = round(np.shape(features)[-1] * float(config.hrir_samplerate) / fs_original)
    features_resampled = sps.resample(np.array(features).T, number_of_samples).T

    magnitudes, _ = calc_hrtf(features_resampled)

    # create empty list of lists of lists and initialize counter
    magnitudes_raw = [[[[] for _ in range(edge_len)] for _ in range(edge_len)] for _ in range(5)]
    count = 0

    for panel, x, y in cube:
        # based on cube coordinates, get indices for magnitudes list of lists
        i = panel - 1
        j = round(edge_len * (x - (PI_4 / edge_len) + PI_4) / (np.pi / 2))
        k = round(edge_len * (y - (PI_4 / edge_len) + PI_4) / (np.pi / 2))

        # add to list of lists of lists and increment counter
        magnitudes_raw[i][j][k] = magnitudes[count]
        count += 1

    return torch.tensor(np.array(magnitudes_raw))