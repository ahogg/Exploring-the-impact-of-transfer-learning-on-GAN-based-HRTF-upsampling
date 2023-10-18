import numpy as np
import os

import torch
import scipy
import pickle
from config import Config

from os import listdir
from os.path import isfile, join
import sofar as sf

from hrtfdata.full import CIPIC, ARI, Listen, BiLi, ITA, HUTUBS, SADIE2, ThreeDThreeA, CHEDAR, Widespread, SONICOM
from pathlib import Path
import matplotlib.pyplot as plt
import numpy.ma as ma

PI_4 = np.pi / 4

def add_itd(az, el, hrir, side, fs=48000, r=0.0875, c=343):

    az = np.radians(az)
    el = np.radians(el)
    interaural_azimuth = np.arcsin(np.sin(az) * np.cos(el))
    delay_in_sec = (r / c) * (interaural_azimuth + np.sin(interaural_azimuth))
    fractional_delay = delay_in_sec * fs

    sample_delay = int(abs(fractional_delay))

    if (delay_in_sec > 0 and side == 'right') or (delay_in_sec < 0 and side == 'left'):
        N = len(hrir)
        delayed_hrir = np.zeros(N)
        delayed_hrir[sample_delay:] = hrir[0:N - sample_delay]
        sofa_delay = sample_delay
    else:
        sofa_delay = 0
        delayed_hrir = hrir

    return delayed_hrir, sofa_delay


def gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count, left_phase=None, right_phase=None):
    el = np.degrees(sphere_coords[count][0])
    az = np.degrees(sphere_coords[count][1])
    source_position = [az + 360 if az < 0 else az, el, 1.2]

    left_hrir = left_hrtf
    right_hrir = right_hrtf

    # if left_phase is None:
    #     left_hrtf[left_hrtf == 0.0] = 1.0e-08
    #     left_phase = np.imag(-hilbert(np.log(np.abs(left_hrtf))))
    # if right_phase is None:
    #     right_hrtf[right_hrtf == 0.0] = 1.0e-08
    #     right_phase = np.imag(-hilbert(np.log(np.abs(right_hrtf))))
    #
    # left_hrir = scipy.fft.irfft(np.concatenate((np.array([0]), np.abs(left_hrtf[:config.nbins_hrtf-1]))) * np.exp(1j * left_phase))[:config.nbins_hrtf]
    # right_hrir = scipy.fft.irfft(np.concatenate((np.array([0]), np.abs(right_hrtf[:config.nbins_hrtf-1]))) * np.exp(1j * right_phase))[:config.nbins_hrtf]

    left_hrir, left_sample_delay = add_itd(az, el, left_hrir, side='left')
    right_hrir, right_sample_delay = add_itd(az, el, right_hrir, side='right')

    full_hrir = [left_hrir, right_hrir]
    delay = [left_sample_delay, right_sample_delay]

    return source_position, full_hrir, delay


def save_sofa(clean_hrtf, config, cube_coords, sphere_coords, sofa_path_output, phase=None):
    full_hrirs = []
    source_positions = []
    delays = []
    left_full_phase = None
    right_full_phase = None
    if cube_coords is None:
        left_full_hrtf = clean_hrtf[:, :config.nbins_hrtf]
        right_full_hrtf = clean_hrtf[:, config.nbins_hrtf:]

        if phase is not None:
            left_full_phase = phase[:, :config.nbins_hrtf]
            right_full_phase = phase[:, config.nbins_hrtf:]

        for count in range(len(sphere_coords)):
            left_hrtf = np.array(left_full_hrtf[count])
            right_hrtf = np.array(right_full_hrtf[count])

            if phase is None:
                source_position, full_hrir, delay = gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count)
            else:
                left_phase = np.array(left_full_phase[count])
                right_phase = np.array(right_full_phase[count])
                source_position, full_hrir, delay = gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count, left_phase, right_phase)

            full_hrirs.append(full_hrir)
            source_positions.append(source_position)
            delays.append(delay)

    else:
        left_full_hrtf = clean_hrtf[:, :, :, :config.nbins_hrtf]
        right_full_hrftf = clean_hrtf[:, :, :, config.nbins_hrtf:]

        count = 0
        for panel, x, y in cube_coords:
            # based on cube coordinates, get indices for magnitudes list of lists
            i = panel - 1
            j = round(config.hrtf_size * (x - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
            k = round(config.hrtf_size * (y - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))

            left_hrtf = np.array(left_full_hrtf[i, j, k])
            right_hrtf = np.array(right_full_hrtf[i, j, k])
            source_position, full_hrir, delay = gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count)
            full_hrirs.append(full_hrir)
            source_positions.append(source_position)
            delays.append(delay)
            count += 1

    sofa = sf.Sofa("SimpleFreeFieldHRIR")
    sofa.Data_IR = full_hrirs
    sofa.Data_SamplingRate = config.hrir_samplerate
    sofa.Data_Delay = delays
    sofa.SourcePosition = source_positions
    sf.write_sofa(sofa_path_output, sofa)


def spectral_distortion_inner(input_spectrum, target_spectrum):
    numerator = target_spectrum
    denominator = input_spectrum
    return torch.mean((20 * torch.log10(numerator / denominator)) ** 2)


results_path = '/home/ahogg/PycharmProjects/HRTF-GAN/fabio_results/'
path = '/home/ahogg/Documents/FabioHassan/output/80to1280/'
# onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and 'orginal' in f]
onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and 'original__02_orginal' in f]
total_sd_metric = 0


base_dir = Path('/home/ahogg/Documents/HRTF Datasets/')
domain = 'time'
side = 'both'
ds = SONICOM(base_dir / 'SONICOM',  features_spec={'hrirs': {'side': side, 'domain': domain}}, target_spec={'side': {}}, group_spec={'subject': {}})

for subject_idx in range(0, len(ds), 2):
    subject_left = ds[subject_idx]
    subject_right = ds[subject_idx+1]
    subject_id = ds.subject_ids[subject_idx]

    hrirs_left = ma.getdata(subject_left['features'])
    hrirs_right = ma.getdata(subject_right['features'])

    target_hrirs = []
    for row_idx, row_left in enumerate(hrirs_left):
        row_right = hrirs_right[row_idx]
        row_pos = ds.row_angles[row_idx]
        for column_idx, column_left in enumerate(row_left):
            column_right = row_right[column_idx]
            column_pos = ds.column_angles[column_idx]
            target_hrir = np.array(list(column_left[0])+list(column_right[0])+[row_pos, column_pos])
            target_hrirs.append(target_hrir)

    with open(f'/home/ahogg/Documents/FabioHassan/AidanDataset/{subject_id}.pickle', 'wb') as handle:
        pickle.dump(target_hrirs, handle, protocol=pickle.HIGHEST_PROTOCOL)

for ori_file in onlyfiles:
    # gen_file = ori_file.replace('orginal_one_80_2', 'generated_one_1280_2')
    gen_file = ori_file.replace('original__02_orginal', 'Upsample__02_orginal')
    # print('Original file: %s' % ori_file)
    # print('Generated file: %s' % gen_file)

    ori_hrir = torch.from_numpy(np.load(path + ori_file))
    gen_hrir = torch.from_numpy(np.load(path + gen_file))

    tag = None
    config = Config(tag, using_hpc=False)
    config.nbins_hrtf = 256

    # need to use protected member to get this data, no getters
    config.dataset = 'ARI'
    projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
    with open(projection_filename, "rb") as file:
        cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

    sofa_output_gen = results_path + 'test_gen.sofa'
    sofa_output_ori = results_path + 'test_ori.sofa'

    # DEBUGGING - plot impulse response
    position_index = 10
    plt.figure()
    plt.plot(ori_hrir[position_index])
    plt.plot(gen_hrir[position_index])
    plt.show()
    ###################################

    save_sofa(gen_hrir, config, None, sphere, sofa_output_gen, phase=None)
    save_sofa(ori_hrir, config, None, sphere, sofa_output_ori, phase=None)

    total_positions = len(gen_hrir)
    total_all_positions = 0

    for ori_ir, gen_ir in zip(ori_hrir, gen_hrir):
        ori_tf = scipy.fft.rfft(np.array(ori_ir), config.nbins_hrtf * 2)[1:]
        gen_tf = scipy.fft.rfft(np.array(gen_ir), config.nbins_hrtf * 2)[1:]
        average_over_frequencies = spectral_distortion_inner(torch.from_numpy(abs(gen_tf)), torch.from_numpy(abs(ori_tf)))
        total_all_positions += torch.sqrt(average_over_frequencies)
        # print('Log SD (for 1 position): %s' % torch.sqrt(average_over_frequencies))

    sd_metric = total_all_positions / total_positions
    total_sd_metric += sd_metric

    print('Log SD (across all positions): %s' % float(sd_metric))

print('Mean LSD Error: %s' % float(total_sd_metric/len(onlyfiles)))
