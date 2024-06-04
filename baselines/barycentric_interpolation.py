import pickle
import os
import glob
import numpy as np
import torch
import shutil
from pathlib import Path

from model.dataset import downsample_hrtf
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.utils import interpolate_fft
from preprocessing.convert_coordinates import convert_cube_to_sphere, convert_single_panel_indices_to_cube_indices, convert_cube_indices_to_spherical
from preprocessing.barycentric_calcs import get_triangle_vertices, calc_barycentric_coordinates

PI_4 = np.pi / 4

def run_barycentric_interpolation(config, barycentric_output_path, subject_file=None):

    if config.lap_factor:
        valid_lap_original_hrtf_paths = glob.glob('%s/%s_*' % (config.valid_lap_original_hrtf_merge_dir, config.dataset))
        valid_lap_original_file_names = ['/' + os.path.basename(x) for x in valid_lap_original_hrtf_paths]

        # Clear/Create directory
        shutil.rmtree(Path(barycentric_output_path), ignore_errors=True)
        Path(barycentric_output_path).mkdir(parents=True, exist_ok=True)

        euclidean_sphere_triangles = []
        euclidean_sphere_coeffs = []

        original_coordinates_filename = f'{config.projection_dir}/{config.dataset}_original'
        with open(original_coordinates_filename, "rb") as f:
            sphere_original = pickle.load(f)
        sphere_coords = sphere_original

        edge_len = int(int(config.hrtf_size) / int(config.upscale_factor))
        projection_filename = f'{config.projection_dir}/{config.dataset}_projection_lap_{config.lap_factor}_{edge_len}'

        with open(projection_filename, "rb") as file:
            cube_lap, sphere_lap, sphere_triangles_lap, sphere_coeffs_lap, measured_coords_lap = pickle.load(file)

        for sphere_coord_idx, sphere_coord in enumerate(sphere_coords):
            # based on cube coordinates, get indices for magnitudes list of lists
            # print(f'Calculating Barycentric coefficient {sphere_coord_idx} of {len(sphere_coords)}')
            triangle_vertices = get_triangle_vertices(elevation=sphere_coord[0], azimuth=sphere_coord[1],
                                                          sphere_coords=measured_coords_lap)
            coeffs = calc_barycentric_coordinates(elevation=sphere_coord[0], azimuth=sphere_coord[1],
                                                      closest_points=triangle_vertices)
            euclidean_sphere_triangles.append(triangle_vertices)
            euclidean_sphere_coeffs.append(coeffs)

        for file_name in valid_lap_original_file_names:
            with open(config.valid_lap_original_hrtf_merge_dir + file_name, "rb") as f:
                orginal_hrtf = pickle.load(f)

            orginal_hrtf_left = orginal_hrtf[:, :config.nbins_hrtf]
            orginal_hrtf_right = orginal_hrtf[:, config.nbins_hrtf:]

            cs_lap = CubedSphere(sphere_coords=measured_coords_lap, indices=[[x] for x in np.arange(int(config.lap_factor))])

            barycentric_hr_left = interpolate_fft(config, cs_lap, orginal_hrtf_left, sphere_coords,
                                                    euclidean_sphere_triangles,
                                                    euclidean_sphere_coeffs, cube_lap, edge_len=config.hrtf_size,
                                                    cs_output=False)
            barycentric_hr_right = interpolate_fft(config, cs_lap, orginal_hrtf_right, sphere_coords,
                                                    euclidean_sphere_triangles,
                                                    euclidean_sphere_coeffs, cube_lap, edge_len=config.hrtf_size,
                                                    cs_output=False)

            barycentric_hr_merged = torch.tensor(np.concatenate((barycentric_hr_left, barycentric_hr_right), axis=1))

            with open(barycentric_output_path + file_name, "wb") as file:
                pickle.dump(barycentric_hr_merged, file)

            print('Created LAP barycentric baseline %s' % file_name.replace('/', ''))

        cube_coords = cube_lap

    else:
        if subject_file is None:
            valid_data_paths = glob.glob('%s/%s_*' % (config.valid_hrtf_merge_dir, config.dataset))
            valid_data_file_names = ['/' + os.path.basename(x) for x in valid_data_paths]
        else:
            valid_data_file_names = ['/' + subject_file]

        # Clear/Create directory
        shutil.rmtree(Path(barycentric_output_path), ignore_errors=True)
        Path(barycentric_output_path).mkdir(parents=True, exist_ok=True)

        projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
        with open(projection_filename, "rb") as f:
            (cube_coords, sphere_coords, euclidean_sphere_triangles, euclidean_sphere_coeffs) = pickle.load(f)

        for file_name in valid_data_file_names:
            with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
                hr_hrtf = pickle.load(f)

            sphere_coords_lr = []
            sphere_coords_lr_index = []
            if len(hr_hrtf.size()) == 3:  # single panel
                lr_hrtf = torch.permute(downsample_hrtf(torch.permute(hr_hrtf, (2, 0, 1)), config.hrtf_size, config.upscale_factor), (1, 2, 0))
                for x in np.arange(hr_hrtf.size()[0]):
                    if (x < config.hrtf_size) or (2 * config.hrtf_size <= x < 3 * config.hrtf_size):
                        height = config.hrtf_size
                    else:
                        height = int(config.hrtf_size * 1.5)
                    for y in np.arange(height):
                        if len(hr_hrtf.size()) == 3:  # single panel
                            if hr_hrtf[x, y] in lr_hrtf:
                                cube_indices = convert_single_panel_indices_to_cube_indices(x, y, config.hrtf_size)
                                sphere_coords_lr.append(convert_cube_indices_to_spherical(cube_indices[0], cube_indices[1], cube_indices[2], config.hrtf_size))
                                sphere_coords_lr_index.append([y/config.upscale_factor, x/config.upscale_factor])
            else:
                lr_hrtf = torch.permute(downsample_hrtf(torch.permute(hr_hrtf, (3, 0, 1, 2)), config.hrtf_size, config.upscale_factor), (1, 2, 3, 0))
                for panel, x, y in cube_coords:
                    # based on cube coordinates, get indices for magnitudes list of lists
                    i = panel - 1
                    j = round(config.hrtf_size * (x - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
                    k = round(config.hrtf_size * (y - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
                    if hr_hrtf[i, j, k] in lr_hrtf:
                        sphere_coords_lr.append(convert_cube_to_sphere(panel, x, y))
                        sphere_coords_lr_index.append([int(i), int(j / config.upscale_factor), int(k / config.upscale_factor)])

            euclidean_sphere_triangles = []
            euclidean_sphere_coeffs = []

            if config.barycentric_postprocessing:
                original_coordinates_filename = f'{config.projection_dir}/{config.dataset}_original'
                with open(original_coordinates_filename, "rb") as f:
                    sphere_original = pickle.load(f)
                sphere_coords = sphere_original

            for sphere_coord_idx, sphere_coord in enumerate(sphere_coords):
                # based on cube coordinates, get indices for magnitudes list of lists
                # print(f'Calculating Barycentric coefficient {sphere_coord_idx} of {len(sphere_coords)}')
                triangle_vertices = get_triangle_vertices(elevation=sphere_coord[0], azimuth=sphere_coord[1],
                                                          sphere_coords=sphere_coords_lr)
                coeffs = calc_barycentric_coordinates(elevation=sphere_coord[0], azimuth=sphere_coord[1],
                                                      closest_points=triangle_vertices)
                euclidean_sphere_triangles.append(triangle_vertices)
                euclidean_sphere_coeffs.append(coeffs)

            cs = CubedSphere(sphere_coords=sphere_coords_lr, indices=sphere_coords_lr_index)

            if len(hr_hrtf.size()) == 3:  # single panel
                lr_hrtf_left = lr_hrtf[:, :, :config.nbins_hrtf]
                lr_hrtf_right = lr_hrtf[:, :, config.nbins_hrtf:]
            else:
                lr_hrtf_left = lr_hrtf[:, :, :, :config.nbins_hrtf]
                lr_hrtf_right = lr_hrtf[:, :, :, config.nbins_hrtf:]

            if config.barycentric_postprocessing:
                barycentric_hr_left = interpolate_fft(config, cs, lr_hrtf_left, sphere_coords, euclidean_sphere_triangles,
                                                      euclidean_sphere_coeffs, cube_coords, edge_len=config.hrtf_size,
                                                      cs_output=False)
                barycentric_hr_right = interpolate_fft(config, cs, lr_hrtf_right, sphere_coords, euclidean_sphere_triangles,
                                                       euclidean_sphere_coeffs, cube_coords, edge_len=config.hrtf_size,
                                                       cs_output=False)
            else:
                barycentric_hr_left = interpolate_fft(config, cs, lr_hrtf_left, sphere_coords, euclidean_sphere_triangles,
                                                 euclidean_sphere_coeffs, cube_coords, edge_len=config.hrtf_size)
                barycentric_hr_right = interpolate_fft(config, cs, lr_hrtf_right, sphere_coords, euclidean_sphere_triangles,
                                                      euclidean_sphere_coeffs, cube_coords, edge_len=config.hrtf_size)

            if len(hr_hrtf.size()) == 3:  # single panel
                barycentric_hr_merged = torch.tensor(np.concatenate((barycentric_hr_left, barycentric_hr_right), axis=2))
            else:
                if config.barycentric_postprocessing:
                    barycentric_hr_merged = torch.tensor(np.concatenate((barycentric_hr_left, barycentric_hr_right), axis=1))
                else:
                    barycentric_hr_merged = torch.tensor(np.concatenate((barycentric_hr_left, barycentric_hr_right), axis=3))

            with open(barycentric_output_path + file_name, "wb") as file:
                pickle.dump(barycentric_hr_merged, file)

            print('Created barycentric baseline %s' % file_name.replace('/', ''))

    return cube_coords, sphere_coords
