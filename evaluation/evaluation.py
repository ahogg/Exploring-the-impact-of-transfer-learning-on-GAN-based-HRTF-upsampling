from model.util import spectral_distortion_metric
from model.dataset import downsample_hrtf
from preprocessing.utils import convert_to_sofa

from preprocessing.convert_coordinates import convert_single_panel_indices_to_spherical, convert_cube_indices_to_spherical

import shutil
from pathlib import Path

import glob
import torch
import pickle
import os
import re
import numpy as np

import matlab.engine

def replace_nodes(config, sr_dir, file_name, spectral_distortion_metric=False):
    # Overwrite the generated points that exist in the original data

    def calc_lsd(hr, sr):
        sum_on_band = 0.0
        for j in range(10, 128):
            sum_on_band += (20.0 * np.log10(abs(hr[j]) / abs(sr[j]))) ** 2
        sum_on_band /= 118
        sum_on_band_1 = 0.0
        for j in range(138, 256):
            sum_on_band_1 += (20.0 * np.log10(abs(hr[j]) / abs(sr[j]))) ** 2
        sum_on_band_1 /= 118
        sum_on_band = np.sqrt(sum_on_band.numpy())
        sum_on_band_1 = np.sqrt(sum_on_band_1.numpy())
        error_lsd = (sum_on_band + sum_on_band_1) / 2.0
        return error_lsd

    with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
        hr_hrtf = pickle.load(f)

    with open(sr_dir + file_name, "rb") as f:
        sr_hrtf = pickle.load(f)

    lr_hrtf = torch.moveaxis(
        downsample_hrtf(torch.moveaxis(hr_hrtf, -1, 0), config.hrtf_size, config.upscale_factor, config.panel),
        0, -1)

    lr = lr_hrtf.detach().cpu()
    xy = []
    errors = []

    if len(lr.size()) == 3:  # single panel
            for w in range(config.hrtf_size*4):
                for h in range(int(config.hrtf_size*1.5)):
                    if hr_hrtf[w, h] in lr:
                        sr_hrtf[w, h] = hr_hrtf[w, h]
                        xy.append({'x': w, 'y': h, 'original': True})
                    else:
                        xy.append({'x': w, 'y': h, 'original': False})
                        if spectral_distortion_metric:
                            errors.append(calc_lsd(sr_hrtf[w, h], hr_hrtf[w, h]))

            generated = torch.permute(sr_hrtf[:, None], (1, 3, 0, 2))
            target = torch.permute(hr_hrtf[:, None], (1, 3, 0, 2))
    else:
        for p in range(5):
            for w in range(config.hrtf_size):
                for h in range(config.hrtf_size):
                    xy_spherical = convert_cube_indices_to_spherical(p, w, h, config.hrtf_size)
                    x_spherical = np.degrees(xy_spherical[1])
                    y_spherical = np.degrees(xy_spherical[0])
                    if hr_hrtf[p, w, h] in lr:
                        sr_hrtf[p, w, h] = hr_hrtf[p, w, h]
                        xy.append({'x': x_spherical, 'y': y_spherical, 'original': True})
                    else:
                        xy.append({'x': x_spherical, 'y': y_spherical, 'original': False})
                        if spectral_distortion_metric:
                            errors.append(calc_lsd(sr_hrtf[p, w, h], hr_hrtf[p, w, h]))

        generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3))
        target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))

    return target, generated, errors, xy

def run_lsd_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None):

    file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext

    if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
        lsd_errors = []
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_hrtf_merge_dir, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]

        for file_name in valid_data_file_names:
            # Overwrite the generated points that exist in the original data
            with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
                hr_hrtf = pickle.load(f)

            with open(f'{sr_dir}/{hrtf_selection}.pickle', "rb") as f:
                sr_hrtf = pickle.load(f)

            generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3))
            target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))

            error = spectral_distortion_metric(generated, target)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append([subject_id,  float(error.detach())])
            print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))
    else:
        sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
        sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

        lsd_errors = []
        for file_name in sr_data_file_names:
            target, generated, errors, xy = replace_nodes(config, sr_dir, file_name, spectral_distortion_metric=True)
            # error = spectral_distortion_metric(generated, target)
            error = sum(errors) / len(xy)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append({'subject_id': subject_id, 'total_error': error, 'errors': errors, 'coordinates': xy})
            print('LSD Error of subject %s: %0.4f' % (subject_id, error))

    try:
        with open(f'{config.path}/{file_ext}', "wb") as file:
            pickle.dump(lsd_errors, file)
    except OSError:
        print(f"Unable to load {config.path}/{file_ext} successfully.")
        return
    print('Mean LSD Error: %0.3f' % np.mean([error['total_error'] for error in lsd_errors]))


def run_localisation_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None):

    file_ext = 'loc_errors.pickle' if file_ext is None else file_ext

    if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
        nodes_replaced_path = sr_dir
        hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(config.valid_hrtf_merge_dir + '/sofa_min_phase')]
    else:
        sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
        sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

        # Clear/Create directories
        nodes_replaced_path = sr_dir + '/nodes_replaced'
        shutil.rmtree(Path(nodes_replaced_path), ignore_errors=True)
        Path(nodes_replaced_path).mkdir(parents=True, exist_ok=True)

        for file_name in sr_data_file_names:
            target, generated = replace_nodes(config, sr_dir, file_name)

            with open(nodes_replaced_path + file_name, "wb") as file:
                pickle.dump(torch.permute(generated[0], (1, 2, 3, 0)), file)

        projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
        with open(projection_filename, "rb") as file:
            cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        convert_to_sofa(nodes_replaced_path, config, cube, sphere)
        print('Created valid sofa files')

        hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(nodes_replaced_path + '/sofa_min_phase')]

    eng = matlab.engine.start_matlab()
    s = eng.genpath(config.amt_dir)
    eng.addpath(s, nargout=0)
    s = eng.genpath(config.data_dirs_path)
    eng.addpath(s, nargout=0)

    file_path = config.path
    if not os.path.exists(file_path):
        raise Exception(f'File path does not exist or does not have write permissions ({file_path})')

    loc_errors = []
    for file in hrtf_file_names:
        target_sofa_file = config.valid_hrtf_merge_dir + '/sofa_min_phase/' + file
        if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
            generated_sofa_file = f'{nodes_replaced_path}/sofa_min_phase/{hrtf_selection}.sofa'
        else:
            generated_sofa_file = nodes_replaced_path+'/sofa_min_phase/' + file

        print(f'Target: {target_sofa_file}')
        print(f'Generated: {generated_sofa_file}')
        [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
        subject_id = ''.join(re.findall(r'\d+', file))
        loc_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
        print('pol_acc1: %s' % pol_acc1)
        print('pol_rms1: %s' % pol_rms1)
        print('querr1: %s' % querr1)

    print('Mean ACC Error: %0.3f' % np.mean([error[1] for error in loc_errors]))
    print('Mean RMS Error: %0.3f' % np.mean([error[2] for error in loc_errors]))
    print('Mean QUERR Error: %0.3f' % np.mean([error[3] for error in loc_errors]))
    with open(f'{file_path}/{file_ext}', "wb") as file:
        pickle.dump(loc_errors, file)


def run_target_localisation_evaluation(config):

    eng = matlab.engine.start_matlab()
    s = eng.genpath(config.amt_dir)
    eng.addpath(s, nargout=0)
    s = eng.genpath(config.data_dirs_path)
    eng.addpath(s, nargout=0)

    file_path = f'{config.data_dirs_path}{config.data_dir}'
    if not os.path.exists(file_path):
        raise Exception(f'File path does not exist or does not have write permissions ({file_path})')

    loc_target_errors = []
    target_sofa_path = config.valid_hrtf_merge_dir + '/sofa_min_phase'
    hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(target_sofa_path)]
    for file in hrtf_file_names:
        target_sofa_file = target_sofa_path + '/' + file
        generated_sofa_file = target_sofa_file
        print(f'Target: {target_sofa_file}')
        print(f'Generated: {generated_sofa_file}')
        [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
        subject_id = ''.join(re.findall(r'\d+', file))
        loc_target_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
        print('pol_acc1: %s' % pol_acc1)
        print('pol_rms1: %s' % pol_rms1)
        print('querr1: %s' % querr1)

    print('Mean ACC Error: %0.3f' % np.mean([error[1] for error in loc_target_errors]))
    print('Mean RMS Error: %0.3f' % np.mean([error[2] for error in loc_target_errors]))
    print('Mean QUERR Error: %0.3f' % np.mean([error[3] for error in loc_target_errors]))
    with open(f'{file_path}/{config.dataset}_loc_target_valid_errors.pickle', "wb") as file:
        pickle.dump(loc_target_errors, file)
