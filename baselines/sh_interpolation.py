import pickle
import os
import glob
import numpy as np
import torch
import shutil
from pathlib import Path

import matlab.engine

from model.dataset import downsample_hrtf
from preprocessing.convert_coordinates import convert_cube_to_sphere

PI_4 = np.pi / 4


def run_sh_interpolation(config, sh_output_path, subject_file=None):

    if subject_file is None:
        valid_data_paths = glob.glob('%s/%s_*' % (config.valid_hrtf_merge_dir, config.dataset))
        valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]
    else:
        valid_data_file_names = ['/' + subject_file]

    # Clear/Create directory
    shutil.rmtree(Path(sh_output_path), ignore_errors=True)
    Path(sh_output_path).mkdir(parents=True, exist_ok=True)

    eng = matlab.engine.start_matlab()
    s = eng.genpath(config.supdeq_dir)
    eng.addpath(s, nargout=0)
    s = eng.genpath(config.data_dirs_path)
    eng.addpath(s, nargout=0)

    projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
    with open(projection_filename, "rb") as f:
        (cube_coords, sphere_coords, euclidean_sphere_triangles, euclidean_sphere_coeffs) = pickle.load(f)

    for file_name in valid_data_file_names:
        with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)

        if config.upscale_factor == config.hrtf_size/2:
            lr_hrtf = hr_hrtf[:, 0:16:6, 0:16:6, :]
        else:
            lr_hrtf = torch.permute(downsample_hrtf(torch.permute(hr_hrtf, (3, 0, 1, 2)), config.hrtf_size, config.upscale_factor),(1, 2, 3, 0))

        HRTF_L = []
        HRTF_R = []
        sphere_coords_lr = []
        sphere_coords_lr_index = []
        sphere_coords_hr = []
        for panel, x, y in cube_coords:
            # based on cube coordinates, get indices for magnitudes list of lists
            i = panel - 1
            j = round(config.hrtf_size * (x - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
            k = round(config.hrtf_size * (y - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
            sphere_coords_hr.append(convert_cube_to_sphere(panel, x, y))
            if hr_hrtf[i, j, k] in lr_hrtf:
                HRTF_L.append(np.array(hr_hrtf[i, j, k][:int(len(hr_hrtf[i, j, k])/2)]).tolist())
                HRTF_R.append(np.array(hr_hrtf[i, j, k][int(len(hr_hrtf[i, j, k])/2):]).tolist())
                sphere_coords_lr.append(convert_cube_to_sphere(panel, x, y))
                sphere_coords_lr_index.append([int(i), int(j / config.upscale_factor), int(k / config.upscale_factor)])

        interpHRTF_sh = eng.supdeq_baseline(matlab.double(sphere_coords_lr), matlab.double(sphere_coords_hr), config.hrir_samplerate, matlab.double(HRTF_L), matlab.double(HRTF_R))

        index = 0
        sr_hrtf = torch.tensor([[[[np.nan]*np.shape(hr_hrtf)[3]]*np.shape(hr_hrtf)[2]]*np.shape(hr_hrtf)[1]]*np.shape(hr_hrtf)[0])
        for panel, x, y in cube_coords:
            # based on cube coordinates, get indices for magnitudes list of lists
            i = panel - 1
            j = round(config.hrtf_size * (x - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
            k = round(config.hrtf_size * (y - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
            sr_hrtf[i, j, k] = torch.tensor(np.concatenate((np.abs(interpHRTF_sh['HRTF_L'])[index], np.abs(interpHRTF_sh['HRTF_R'])[index])))
            index += 1

        with open(sh_output_path + file_name, "wb") as file:
            pickle.dump(sr_hrtf, file)

        print('Created SH baseline %s' % file_name.replace('/', ''))

    return cube_coords, sphere_coords
