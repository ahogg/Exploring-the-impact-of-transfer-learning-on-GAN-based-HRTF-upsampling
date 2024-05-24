from model.util import spectral_distortion_metric
from model.dataset import downsample_hrtf
from preprocessing.utils import convert_to_sofa

from preprocessing.convert_coordinates import convert_single_panel_indices_to_cube_indices, convert_cube_indices_to_spherical, convert_cube_to_sphere
from preprocessing.barycentric_calcs import get_triangle_vertices, calc_barycentric_coordinates
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.utils import interpolate_fft
from preprocessing.utils import calc_hrtf

import matplotlib.pyplot as plt

import shutil
from pathlib import Path

import glob
import torch
import pickle
import os
import re
import numpy as np

import matlab.engine

from model.util import spectral_distortion_inner

def replace_nodes(config, sr_dir, file_name, calc_spectral_distortion=False, barycentric_postprocessing=True):
    # Overwrite the generated points that exist in the original data

    with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
        hr_hrtf = pickle.load(f)

    with open(sr_dir + file_name, "rb") as f:
        sr_hrtf = pickle.load(f)

    with open(config.valid_original_hrtf_merge_dir + file_name, "rb") as f:
        orig_hrtf = pickle.load(f)

    original_coordinates_filename = f'{config.projection_dir}/{config.dataset}_original'
    with open(original_coordinates_filename, "rb") as f:
        sphere_original = pickle.load(f)

    projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
    with open(projection_filename, "rb") as f:
        (cube_coords, sphere_coords, euclidean_sphere_triangles, euclidean_sphere_coeffs) = pickle.load(f)

    lr_hrtf = torch.moveaxis(
        downsample_hrtf(torch.moveaxis(hr_hrtf, -1, 0), config.hrtf_size, config.upscale_factor, config.panel),
        0, -1)

    lr = lr_hrtf.detach().cpu()
    xy = []
    errors = []

    if len(np.shape(sr_hrtf)) == 2:
        errors = []
        for i in np.arange(len(sr_hrtf)):
            if calc_spectral_distortion:
                errors.append(spectral_distortion_inner(sr_hrtf[i], orig_hrtf[i]))

        xy_postprocessing = []
        for sphere_coord_idx, sphere_coord in enumerate(sphere_original):
            xy_postprocessing.append({'x': np.degrees(sphere_coord[1]), 'y': np.degrees(sphere_coord[0]), 'original': False})

        generated = sr_hrtf
        target = orig_hrtf
        return target, generated, errors, xy_postprocessing

    elif len(lr.size()) == 3:  # single panel
        sr_hrtf_cube = np.empty((5, config.hrtf_size, config.hrtf_size, config.nbins_hrtf*2))
        hr_hrtf_cube = np.empty((5, config.hrtf_size, config.hrtf_size, config.nbins_hrtf*2))
        for w in range(config.hrtf_size * 4):
            if (w < config.hrtf_size) or (2 * config.hrtf_size <= w < 3 * config.hrtf_size):
                height = config.hrtf_size
            else:
                height = int(config.hrtf_size * 1.5)
            for h in range(height):
                panel, i, j = convert_single_panel_indices_to_cube_indices(w, h, config.hrtf_size)
                xy_spherical = convert_cube_indices_to_spherical(panel, i, j, config.hrtf_size)
                x_spherical = np.degrees(xy_spherical[1])
                y_spherical = np.degrees(xy_spherical[0])

                if hr_hrtf[w, h] in lr:
                    sr_hrtf[w, h] = hr_hrtf[w, h]
                    sr_hrtf_cube[panel, i, j] = hr_hrtf[w, h]
                    hr_hrtf_cube[panel, i, j] = hr_hrtf[w, h]
                    xy.append({'x': x_spherical, 'y': y_spherical, 'original': True})
                else:
                    sr_hrtf_cube[panel, i, j] = sr_hrtf[w, h]
                    hr_hrtf_cube[panel, i, j] = hr_hrtf[w, h]
                    xy.append({'x': x_spherical, 'y': y_spherical, 'original': False})
                    if calc_spectral_distortion:
                        errors.append(spectral_distortion_inner(sr_hrtf[w, h], hr_hrtf[w, h]))

        generated =torch.permute(torch.from_numpy(sr_hrtf_cube)[:, None], (1, 4, 0, 2, 3))
        target = torch.permute(torch.from_numpy(hr_hrtf_cube)[:, None], (1, 4, 0, 2, 3))
    else:
        sphere_coords_lr = []
        sphere_coords_lr_index = []
        for p in range(5):
            for w in range(config.hrtf_size):
                for h in range(config.hrtf_size):
                    xy_spherical = convert_cube_indices_to_spherical(p, w, h, config.hrtf_size)
                    sphere_coords_lr.append((xy_spherical[0], xy_spherical[1]))
                    sphere_coords_lr_index.append([p, w, h])
                    x_spherical = np.degrees(xy_spherical[1])
                    y_spherical = np.degrees(xy_spherical[0])
                    if hr_hrtf[p, w, h] in lr:
                        sr_hrtf[p, w, h] = hr_hrtf[p, w, h]
                        xy.append({'x': x_spherical, 'y': y_spherical, 'original': True})
                    else:
                        xy.append({'x': x_spherical, 'y': y_spherical, 'original': False})
                        if calc_spectral_distortion:
                            errors.append(spectral_distortion_inner(sr_hrtf[p, w, h], hr_hrtf[p, w, h]))

        if barycentric_postprocessing:

            cs = CubedSphere(sphere_coords=sphere_coords_lr, indices=sphere_coords_lr_index)

            postprocessing_projection_filename = f'{config.postprocessing_dir}/{config.dataset}_postprocessing_projection_{config.hrtf_size}'
            if os.path.exists(postprocessing_projection_filename):
                with open(postprocessing_projection_filename, "rb") as f:
                    (euclidean_sphere_triangles, euclidean_sphere_coeffs, xy_postprocessing) = pickle.load(f)
            else:
                euclidean_sphere_triangles = []
                euclidean_sphere_coeffs = []
                xy_postprocessing = []
                for sphere_coord_idx, sphere_coord in enumerate(sphere_original):
                    xy_postprocessing.append({'x': np.degrees(sphere_coord[1]), 'y': np.degrees(sphere_coord[0]), 'original': False})
                    # based on cube coordinates, get indices for magnitudes list of lists
                    # print(f'Calculating Barycentric coefficient {sphere_coord_idx} of {len(sphere_original)}')
                    triangle_vertices = get_triangle_vertices(elevation=sphere_coord[0], azimuth=sphere_coord[1],
                                                              sphere_coords=sphere_coords_lr)
                    coeffs = calc_barycentric_coordinates(elevation=sphere_coord[0], azimuth=sphere_coord[1],
                                                          closest_points=triangle_vertices)
                    euclidean_sphere_triangles.append(triangle_vertices)
                    euclidean_sphere_coeffs.append(coeffs)

                with open(postprocessing_projection_filename, "wb") as f:
                    pickle.dump((euclidean_sphere_triangles, euclidean_sphere_coeffs, xy_postprocessing), f)

            sr_hrtf_left = sr_hrtf[:, :, :, :config.nbins_hrtf]
            sr_hrtf_right = sr_hrtf[:, :, :, config.nbins_hrtf:]

            barycentric_sr_left = interpolate_fft(config, cs, sr_hrtf_left, sphere_original, euclidean_sphere_triangles,
                                                  euclidean_sphere_coeffs, cube_coords, edge_len=config.hrtf_size, cs_output=False)
            barycentric_sr_right = interpolate_fft(config, cs, sr_hrtf_right, sphere_original, euclidean_sphere_triangles,
                                                   euclidean_sphere_coeffs, cube_coords, edge_len=config.hrtf_size, cs_output=False)

            barycentric_sr_merged = torch.tensor(np.concatenate((barycentric_sr_left, barycentric_sr_right), axis=1))

            errors_postprocessing = []
            for i in np.arange(len(barycentric_sr_merged)):
                if calc_spectral_distortion:
                    errors_postprocessing.append(spectral_distortion_inner(barycentric_sr_merged[i], orig_hrtf[i]))

        generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3))
        target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))

    if barycentric_postprocessing:
        generated_postprocessing = barycentric_sr_merged
        target_postprocessing = orig_hrtf

        return target_postprocessing, generated_postprocessing, errors_postprocessing, xy_postprocessing
    else:
        return target, generated, errors, xy


def run_lsd_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None, file_ext_postprocessing=None):

    file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext
    file_ext_postprocessing = 'lsd_errors_postprocessing.pickle' if file_ext_postprocessing is None else file_ext_postprocessing

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

            if len(hr_hrtf.size()) == 3:  # single panel
                generated = torch.permute(sr_hrtf[:, None], (1, 3, 0, 2))
                target = torch.permute(hr_hrtf[:, None], (1, 3, 0, 2))
            else:
                generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3))
                target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))

            error, errors, xy = spectral_distortion_metric(generated, target, full_errors=True)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append({'subject_id': subject_id, 'total_error': float(error), 'errors': [float(error) for error in errors], 'coordinates': xy})
            print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))
    else:
        # Clear/Create directories
        nodes_replaced_path = sr_dir + '/nodes_replaced'
        shutil.rmtree(Path(nodes_replaced_path), ignore_errors=True)
        Path(nodes_replaced_path).mkdir(parents=True, exist_ok=True)

        if config.barycentric_postprocessing:
            original_coordinates_path = sr_dir + '/original_coordinates'
            Path(original_coordinates_path).mkdir(parents=True, exist_ok=True)

        sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
        sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

        lsd_errors = []
        for file_name in sr_data_file_names:
            target, generated, errors, xy = replace_nodes(config, sr_dir, file_name, calc_spectral_distortion=True, barycentric_postprocessing=False)
            error = sum(errors) / len(xy)
            subject_id = ''.join(re.findall(r'\d+', file_name))
            lsd_errors.append({'subject_id': subject_id, 'total_error': error, 'errors': errors, 'coordinates': xy})
            print('LSD Error of subject %s: %0.4f' % (subject_id, error))

            with open(nodes_replaced_path + file_name, "wb") as file:
                pickle.dump(torch.permute(generated[0], (1, 2, 3, 0)), file)

        print('Mean LSD Error: %0.3f' % np.mean([error['total_error'] for error in lsd_errors]))

        projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
        with open(projection_filename, "rb") as file:
            cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        convert_to_sofa(nodes_replaced_path, config, cube, sphere)

        print('Created valid sofa files')

        try:
            with open(f'{config.path}/{file_ext}', "wb") as file:
                pickle.dump(lsd_errors, file)
        except OSError:
            print(f"Unable to load {config.path}/{file_ext} successfully.")
            return

        if config.barycentric_postprocessing:
            lsd_errors_postprocessing = []
            for file_name in sr_data_file_names:
                target_postprocessing, generated_postprocessing, errors_postprocessing, xy_postprocessing = replace_nodes(config, sr_dir, file_name, calc_spectral_distortion=True,
                                                              barycentric_postprocessing=True)
                error_postprocessing = sum(errors_postprocessing) / len(xy_postprocessing)
                subject_id = ''.join(re.findall(r'\d+', file_name))
                lsd_errors_postprocessing.append({'subject_id': subject_id, 'total_error': error_postprocessing, 'errors': errors_postprocessing, 'coordinates': xy_postprocessing})
                print('LSD Error (with barycentric postprocessing) of subject %s: %0.4f' % (subject_id, error_postprocessing))

                with open(original_coordinates_path + file_name, "wb") as file:
                    pickle.dump(generated_postprocessing, file)
            print('Mean LSD Error (with barycentric postprocessing): %0.3f' % np.mean([error_postprocessing['total_error'] for error_postprocessing in lsd_errors_postprocessing]))

            sphere_original_filename = f'{config.projection_dir}/{config.dataset}_original'
            with open(sphere_original_filename, "rb") as file:
                sphere_original = pickle.load(file)

            convert_to_sofa(original_coordinates_path, config, cube=None, sphere=sphere_original)

            try:
                with open(f'{config.path}/{file_ext_postprocessing}', "wb") as file:
                    pickle.dump(lsd_errors_postprocessing, file)
            except OSError:
                print(f"Unable to load {config.path}/{file_ext_postprocessing} successfully.")
                return


def run_localisation_evaluation(config, sr_dir, file_ext=None, hrtf_selection=None, baseline=False):

    file_ext = 'loc_errors.pickle' if file_ext is None else file_ext

    if baseline:
        nodes_replaced_path = sr_dir
        target_sofa_path = config.valid_hrtf_merge_dir.replace('single_panel', 'cube_sphere')  # For single panel use cube sphere
        hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(target_sofa_path + '/sofa_min_phase')]
    else:
        sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
        sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

        # Clear/Create directories
        if config.barycentric_postprocessing:
            nodes_replaced_path = sr_dir + '/original_coordinates'
        else:
            nodes_replaced_path = sr_dir + '/nodes_replaced'
        shutil.rmtree(Path(nodes_replaced_path), ignore_errors=True)
        Path(nodes_replaced_path).mkdir(parents=True, exist_ok=True)

        for file_name in sr_data_file_names:
            # target, generated, _, _ = replace_nodes(config, sr_dir, file_name)
            target, generated, _, _ = replace_nodes(config, sr_dir, file_name, calc_spectral_distortion=True,
                                                          barycentric_postprocessing=config.barycentric_postprocessing)

            with open(nodes_replaced_path + file_name, "wb") as file:
                if config.barycentric_postprocessing:
                    pickle.dump(generated, file)
                else:
                    pickle.dump(torch.permute(generated[0], (1, 2, 3, 0)), file)

            print(f'Created valid pickle file for {file_name}')


        if config.barycentric_postprocessing:
            sphere_original_filename = f'{config.projection_dir}/{config.dataset}_original'
            with open(sphere_original_filename, "rb") as file:
                sphere_original = pickle.load(file)

            convert_to_sofa(nodes_replaced_path, config, cube=None, sphere=sphere_original)
        else:
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
        target_sofa_file = target_sofa_file.replace('single_panel', 'cube_sphere')  # For single panel use cube sphere
        if baseline:
            nodes_replaced_path = nodes_replaced_path.replace('single_panel', 'cube_sphere')  # For single panel use cube sphere
            if hrtf_selection == 'minimum' or hrtf_selection == 'maximum':
                generated_sofa_file = f'{nodes_replaced_path}/sofa_min_phase/{hrtf_selection}.sofa'
            else:
                generated_sofa_file = f'{nodes_replaced_path}/sofa_min_phase/{file}'
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
