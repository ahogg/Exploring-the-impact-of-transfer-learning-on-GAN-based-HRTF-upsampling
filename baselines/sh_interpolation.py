import pickle
import os
import glob
import numpy as np
import torch
import shutil
import scipy
import math
from pathlib import Path

import matlab.engine

from model.dataset import downsample_hrtf
from preprocessing.convert_coordinates import convert_cube_indices_to_spherical
from preprocessing.utils import calc_itd_r

PI_4 = np.pi / 4


def run_sh_interpolation(config, sh_output_path, subject_file=None):

    if config.lap_factor:

        valid_lap_original_hrtf_paths = glob.glob('%s/%s_*' % (config.valid_lap_original_hrtf_merge_dir, config.dataset))
        valid_lap_original_file_names = ['/' + os.path.basename(x) for x in valid_lap_original_hrtf_paths if 'mag' in x]

        shutil.rmtree(Path(sh_output_path), ignore_errors=True)
        Path(sh_output_path).mkdir(parents=True, exist_ok=True)

        eng = matlab.engine.start_matlab()
        s = eng.genpath(config.supdeq_dir)
        eng.addpath(s, nargout=0)
        s = eng.genpath(config.data_dirs_path)
        eng.addpath(s, nargout=0)

        original_coordinates_filename = f'{config.projection_dir}/{config.dataset}_original'
        with open(original_coordinates_filename, "rb") as f:
            sphere_original = pickle.load(f)
        sphere_coords = sphere_original

        edge_len = int(int(config.hrtf_size) / int(config.upscale_factor))
        projection_filename = f'{config.projection_dir}/{config.dataset}_projection_lap_{config.lap_factor}_{edge_len}'

        with open(projection_filename, "rb") as file:
            cube_lap, sphere_lap, sphere_triangles_lap, sphere_coeffs_lap, measured_coords_lap = pickle.load(file)

        for file_name in valid_lap_original_file_names:
            with open(config.valid_lap_original_hrtf_merge_dir + file_name, "rb") as f:
                orginal_hrir = pickle.load(f)

            orginal_hrir_left = orginal_hrir[:, :config.nbins_hrtf]
            orginal_hrir_right = orginal_hrir[:, config.nbins_hrtf:]

            rs = []
            for ir_index, measured_coord_lap in enumerate(measured_coords_lap):
                if not math.isclose(measured_coord_lap[0], np.pi/2, rel_tol=np.pi/4) and not math.isclose(measured_coord_lap[1], -np.pi, rel_tol=np.pi/8) \
                        and not math.isclose(measured_coord_lap[1], np.pi, rel_tol=np.pi/8) and not math.isclose(measured_coord_lap[1], 0, abs_tol=np.pi/8):
                    rs.append(calc_itd_r(config, orginal_hrir_left[ir_index], orginal_hrir_right[ir_index], az=measured_coords_lap[ir_index][1], el=measured_coords_lap[ir_index][0]))
            config.head_radius = np.mean(rs)

            orginal_hrtf_left = np.abs(scipy.fft.rfft(np.array(orginal_hrir_left), config.nbins_hrtf * 2)[:, 1:])
            orginal_hrtf_right = np.abs(scipy.fft.rfft(np.array(orginal_hrir_right), config.nbins_hrtf * 2)[:, 1:])

            #(0=front, 90=left, 180=back, 270=right)
            #(0=North Pole, 90=front, 180=South Pole)
            # samplingGridInterp = np.array(
            #     [[360 + x[1], (90 - x[0])] if x[1] < 0 else [x[1], (90 - x[0])] for x in np.degrees(sphere_coords)])
            # samplingGrid = np.array(
            #     [[360 + x[1], (90 - x[0])] if x[1] < 0 else [x[1], (90 - x[0])] for x in np.degrees(measured_coords_lap)])
            samplingGridInterp = np.array(
                [[360 + x[1], (90 - x[0])] if x[1] < 0 else [x[1], (90 - x[0])] for x in np.degrees(sphere_coords)])
            samplingGrid = np.array(
                [[360 + x[1], (90 - x[0])] if x[1] < 0 else [x[1], (90 - x[0])] for x in np.degrees(measured_coords_lap)])


            interpHRTF_sh = eng.supdeq_baseline(matlab.double(samplingGrid), matlab.double(samplingGridInterp),
                                                config.hrir_samplerate, matlab.double(np.double(orginal_hrtf_left)), matlab.double(np.double(orginal_hrtf_right)))

            sh_hr_merged = torch.tensor(
                np.concatenate((np.abs(interpHRTF_sh['HRTF_L']), np.abs(interpHRTF_sh['HRTF_R'])), axis=1))

            with open(sh_output_path + file_name, "wb") as file:
                pickle.dump(sh_hr_merged, file)

            print('Created SH baseline %s' % file_name.replace('/', ''))

        cube_coords = cube_lap

    else:
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

            # if config.upscale_factor == config.hrtf_size/2:
            #     # lr_hrtf = hr_hrtf[:, 0:16:6, 0:16:6, :]
            #     lr_hrtf_sub = hr_hrtf[:, (0, 12), :, :]
            #     lr_hrtf = lr_hrtf_sub[:, :, (0, 12), :]
            # else:
            lr_hrtf = torch.permute(downsample_hrtf(torch.permute(hr_hrtf, (3, 0, 1, 2)), config.hrtf_size, config.upscale_factor),(1, 2, 3, 0))

            HRTF_L = []
            HRTF_R = []
            sphere_coords_lr = []
            sphere_coords_lr_index = []
            for panel, x, y in cube_coords:
                # based on cube coordinates, get indices for magnitudes list of lists
                i = panel - 1
                j = round(config.hrtf_size * (x - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
                k = round(config.hrtf_size * (y - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
                if hr_hrtf[i, j, k] in lr_hrtf:
                    HRTF_L.append(np.array(hr_hrtf[i, j, k][:int(len(hr_hrtf[i, j, k])/2)]).tolist())
                    HRTF_R.append(np.array(hr_hrtf[i, j, k][int(len(hr_hrtf[i, j, k])/2):]).tolist())
                    sphere_coords_lr.append(convert_cube_indices_to_spherical(i, j, k, config.hrtf_size))
                    sphere_coords_lr_index.append([int(i), int(j / config.upscale_factor), int(k / config.upscale_factor)])

            if config.barycentric_postprocessing:
                original_coordinates_filename = f'{config.projection_dir}/{config.dataset}_original'
                with open(original_coordinates_filename, "rb") as f:
                    sphere_original = pickle.load(f)

                samplingGridInterp = np.array([[360 + x[1], (90 - x[0])] if x[1] < 0 else [x[1], (90 - x[0])] for x in np.degrees(sphere_original)])
                samplingGrid = np.array([[360 + x[1], (90 - x[0])] if x[1] < 0 else [x[1], (90 - x[0])] for x in np.degrees(sphere_coords_lr)])

                interpHRTF_sh = eng.supdeq_baseline(matlab.double(np.degrees(samplingGrid)), matlab.double(samplingGridInterp),
                                                    config.hrir_samplerate, matlab.double(HRTF_L), matlab.double(HRTF_R))

                sh_hr_merged = torch.tensor(np.concatenate((np.abs(interpHRTF_sh['HRTF_L']), np.abs(interpHRTF_sh['HRTF_R'])), axis=1))

                with open(sh_output_path + file_name, "wb") as file:
                    pickle.dump(sh_hr_merged, file)

            else:
                samplingGridInterp = np.array([[360+x[1], (90-x[0])] if x[1] < 0 else [x[1], (90-x[0])] for x in np.degrees(sphere_coords)])
                samplingGrid = np.array([[360+x[1], (90-x[0])] if x[1] < 0 else [x[1], (90-x[0])] for x in np.degrees(sphere_coords_lr)])

                interpHRTF_sh = eng.supdeq_baseline(matlab.double(samplingGrid), matlab.double(samplingGridInterp), config.hrir_samplerate, matlab.double(HRTF_L), matlab.double(HRTF_R))

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
