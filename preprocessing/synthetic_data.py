import numpy as np
import torch
from preprocessing.utils import calc_hrtf
import scipy.signal as sps

PI_4 = np.pi / 4

def interpolate_synthetic_fft(config, features, cube, fs_original, edge_len):

    fs_output = config.hrir_samplerate

    # Resample data so that training and validation sets are created at the same fs ('config.hrir_samplerate').
    features_resampled = []
    for feature in features:
        number_of_samples = round(len(feature) * float(fs_output) / fs_original)
        features_resampled.append(sps.resample(feature, number_of_samples))

    magnitudes, _ = calc_hrtf(features)

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