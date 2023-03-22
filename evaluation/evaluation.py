from model.util import spectral_distortion_metric
from model.dataset import downsample_hrtf
from preprocessing.utils import convert_to_sofa

import shutil
from pathlib import Path

import glob
import torch
import pickle
import os
import re
import numpy as np

import matlab.engine

def replace_nodes(config, sr_dir, file_name):
    # Overwrite the generated points that exist in the original data
    with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
        hr_hrtf = pickle.load(f)

    with open(sr_dir + file_name, "rb") as f:
        sr_hrtf = pickle.load(f)

    lr_hrtf = torch.permute(
        downsample_hrtf(torch.permute(hr_hrtf, (3, 0, 1, 2)), config.hrtf_size, config.upscale_factor),
        (1, 2, 3, 0))

    lr = lr_hrtf.detach().cpu()
    for p in range(5):
        for w in range(config.hrtf_size):
            for h in range(config.hrtf_size):
                if hr_hrtf[p, w, h] in lr:
                    sr_hrtf[p, w, h] = hr_hrtf[p, w, h]

    generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3))
    target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))

    return target, generated

def run_lsd_evaluation(config, sr_dir):
    sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
    sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

    lsd_errors = []
    for file_name in sr_data_file_names:
        target, generated = replace_nodes(config, sr_dir, file_name)
        error = spectral_distortion_metric(generated, target)
        subject_id = ''.join(re.findall(r'\d+', file_name))
        lsd_errors.append([subject_id,  float(error.detach())])
        print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))
    print('Mean LSD Error: %0.3f' % np.mean([error[1] for error in lsd_errors]))
    with open(f'{config.path}/lsd_errors.pickle', "wb") as file:
        pickle.dump(lsd_errors, file)

def run_localisation_evaluation(config, sr_dir, run_target=False):
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

    with open(config.projection_filename, "rb") as file:
        cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

    convert_to_sofa(nodes_replaced_path, config, cube, sphere)
    print('Created valid sofa files')

    eng = matlab.engine.start_matlab()
    s = eng.genpath('/home/aos13/AMT/amt_code')
    eng.addpath(s, nargout=0)
    s = eng.genpath(config.data_dirs_path)
    eng.addpath(s, nargout=0)

    loc_errors = []
    loc_target_errors = []
    hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(nodes_replaced_path+'/sofa_min_phase')]
    for file in hrtf_file_names:
        target_sofa_file = config.valid_hrtf_merge_dir + '/sofa_min_phase/' + file
        generated_sofa_file = nodes_replaced_path+'/sofa_min_phase/' + file
        print(f'Target: {target_sofa_file}')
        print(f'Generated: {generated_sofa_file}')
        [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
        subject_id = ''.join(re.findall(r'\d+', file_name))
        loc_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
        print('pol_acc1: %s' % pol_acc1)
        print('pol_rms1: %s' % pol_rms1)
        print('querr1: %s' % querr1)

        if run_target:
            target_sofa_file = config.valid_hrtf_merge_dir + '/sofa_min_phase/' + file
            generated_sofa_file = target_sofa_file
            print(f'Target: {target_sofa_file}')
            print(f'Generated: {generated_sofa_file}')
            [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            loc_target_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
            print('pol_acc1: %s' % pol_acc1)
            print('pol_rms1: %s' % pol_rms1)
            print('querr1: %s' % querr1)

    print('Mean ACC Error: %0.3f' % np.mean([error[1] for error in loc_errors]))
    print('Mean RMS Error: %0.3f' % np.mean([error[2] for error in loc_errors]))
    print('Mean QUERR Error: %0.3f' % np.mean([error[3] for error in loc_errors]))
    with open(f'{config.path}/loc_errors.pickle', "wb") as file:
        pickle.dump(loc_errors, file)

    if run_target:
        print('Mean ACC Error: %0.3f' % np.mean([error[1] for error in loc_target_errors]))
        print('Mean RMS Error: %0.3f' % np.mean([error[2] for error in loc_target_errors]))
        print('Mean QUERR Error: %0.3f' % np.mean([error[3] for error in loc_target_errors]))
        with open(f'{config.path}/loc_target_errors.pickle', "wb") as file:
            pickle.dump(loc_target_errors, file)
