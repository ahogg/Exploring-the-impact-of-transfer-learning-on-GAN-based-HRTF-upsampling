import numpy as np

import torch
import scipy

from os import listdir
from os.path import isfile, join


def spectral_distortion_inner(input_spectrum, target_spectrum):
    numerator = target_spectrum
    denominator = input_spectrum
    return torch.mean((20 * torch.log10(numerator / denominator)) ** 2)


path = '/home/ahogg/PycharmProjects/HRTF-GAN/fabio_results/'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and 'orginal' in f]
total_sd_metric = 0

for ori_file in onlyfiles:
    gen_file = ori_file.replace('orginal_one_80_2', 'generated_one_1280_2')
    # print('Original file: %s' % ori_file)
    # print('Generated file: %s' % gen_file)

    ori_hrir = torch.from_numpy(np.load(path + ori_file))
    gen_hrir = torch.from_numpy(np.load(path + gen_file))

    nbins_hrtf = 256
    total_positions = len(gen_hrir)
    total_all_positions = 0

    for ori_ir, gen_ir in zip(ori_hrir, gen_hrir):
        ori_tf = scipy.fft.rfft(np.array(ori_ir), nbins_hrtf * 2)[1:]
        gen_tf = scipy.fft.rfft(np.array(gen_ir), nbins_hrtf * 2)[1:]
        average_over_frequencies = spectral_distortion_inner(torch.from_numpy(abs(gen_tf)), torch.from_numpy(abs(ori_tf)))
        total_all_positions += torch.sqrt(average_over_frequencies)
        # print('Log SD (for 1 position): %s' % torch.sqrt(average_over_frequencies))

    sd_metric = total_all_positions / total_positions
    total_sd_metric += sd_metric

    print('Log SD (across all positions): %s' % float(sd_metric))

print('Mean LSD Error: %s' % float(total_sd_metric/len(onlyfiles)))
