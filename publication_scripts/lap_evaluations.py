import argparse
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from spatialaudiometrics import lap_challenge as lap
from model.util import load_dataset

from task2_create_sparse_hrtf import create_sparse_hrtf

plt.rcParams['legend.fancybox'] = False


cm_data_vb = [[0,    0,         0],
    [0.0275,         0,    0.0667],
    [0.0549,         0,    0.1294],
    [0.0824,         0,    0.1961],
    [0.1137,         0,    0.2627],
    [0.1412,         0,    0.3294],
    [0.1686,         0,    0.3922],
    [0.1961,         0,    0.4588],
    [0.2235,         0,    0.5255],
    [0.2510,         0,    0.5882],
    [0.2824,         0,    0.6549],
    [0.3137,    0.0118,    0.6431],
    [0.3490,    0.0275,    0.6118],
    [0.3804,    0.0431,    0.5843],
    [0.4157,    0.0588,    0.5569],
    [0.4471,    0.0745,    0.5255],
    [0.4824,    0.0902,    0.4980],
    [0.5137,    0.1059,    0.4667],
    [0.5490,    0.1216,    0.4392],
    [0.5843,    0.1373,    0.4118],
    [0.6157,    0.1529,    0.3804],
    [0.6510,    0.1686,    0.3529],
    [0.6824,    0.1843,    0.3216],
    [0.7176,    0.2000,    0.2941],
    [0.7529,    0.2157,    0.2667],
    [0.7843,    0.2314,    0.2353],
    [0.8196,    0.2471,    0.2078],
    [0.8510,    0.2627,    0.1765],
    [0.8863,    0.2784,    0.1490],
    [0.9176,    0.2941,    0.1216],
    [0.9529,    0.3098,    0.0902],
    [0.9882,    0.3255,    0.0627],
    [1.0000,    0.3451,    0.0471],
    [1.0000,    0.3725,    0.0471],
    [1.0000,    0.4000,    0.0431],
    [1.0000,    0.4275,    0.0431],
    [1.0000,    0.4549,    0.0392],
    [1.0000,    0.4824,    0.0392],
    [1.0000,    0.5098,    0.0353],
    [1.0000,    0.5373,    0.0353],
    [1.0000,    0.5647,    0.0314],
    [1.0000,    0.5922,    0.0314],
    [1.0000,    0.6196,    0.0275],
    [1.0000,    0.6471,    0.0275],
    [1.0000,    0.6745,    0.0235],
    [1.0000,    0.7020,    0.0235],
    [1.0000,    0.7294,    0.0196],
    [1.0000,    0.7569,    0.0157],
    [1.0000,    0.7843,    0.0157],
    [1.0000,    0.8118,    0.0118],
    [1.0000,    0.8392,    0.0118],
    [1.0000,    0.8667,    0.0078],
    [1.0000,    0.8941,    0.0078],
    [1.0000,    0.9216,    0.0039],
    [1.0000,    0.9490,    0.0039],
    [1.0000,    0.9765,         0],
    [1.0000,    0.9882,    0.0863],
    [1.0000,    0.9882,    0.2157],
    [1.0000,    0.9922,    0.3451],
    [1.0000,    0.9922,    0.4784],
    [1.0000,    0.9961,    0.6078],
    [1.0000,    0.9961,    0.7373],
    [1.0000,    1.0000,    0.8706],
    [1.0000,    1.0000,    1.0000]]

parula = LinearSegmentedColormap.from_list('parula', cm_data_vb[::-1])

import pickle
import os
import shutil
import numpy as np
import sofar as sf
import torch

from preprocessing.utils import get_hrtf_from_ds

from hartufo import CollectionSpec, SideSpec, SubjectSpec, HrirSpec, AnthropometrySpec, ImageSpec, Sonicom

from preprocessing.barycentric_calcs import get_triangle_vertices, calc_barycentric_coordinates
from preprocessing.convert_coordinates import convert_cube_indices_to_spherical

from preprocessing.utils import interpolate_fft, convert_to_sofa,  remove_itd, calc_hrtf, merge_left_right_hrtfs
from preprocessing.cubed_sphere import CubedSphere

from publication_scripts.config_forum_acusticum import Config

from model.model import Generator

from model.util import spectral_distortion_inner


def run_lap(hpc):

    lap_factors = ['100', '19', '5', '3']
    sub_ids = [1, 2, 3]

    # lap_factors = ['3']
    # sub_ids = [1]

    report_evaluation = False

    targets = []

    # Create directory
    settings = Config(tag=None, using_hpc=hpc)
    Path(settings.lap_dir).mkdir(parents=True, exist_ok=True)

    data_dir = settings.raw_hrtf_dir / (settings.dataset + '_LAP')
    ds = Sonicom(data_dir, features_spec=HrirSpec(domain='time', side='both', length=settings.nbins_hrtf * 2,
                                                  samplerate=settings.hrir_samplerate, variant='minphase_compensated'))

    create_preprocess_data = False
    if create_preprocess_data:
        for lap_factor in lap_factors:
            settings = Config(tag=None, using_hpc=hpc, lap_factor = lap_factor)
            cs = CubedSphere(mask=ds[0]['features'].mask, row_angles=ds.fundamental_angles, column_angles=ds.orthogonal_angles)

            # need to use protected member to get this data, no getters
            projection_filename = f'{settings.projection_dir}/{settings.dataset}_projection_{settings.hrtf_size}'
            with open(projection_filename, "rb") as file:
                cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

            projected_dir = '/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/lap_full_pickle_' + str(settings.hrtf_size)
            projected_dir_lap = '/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/lap_full_pickle_' + str(settings.lap_factor) + '_' + str(settings.hrtf_size)
            projected_dir_original = '/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/lap_full_pickle_original_' + str(settings.lap_factor)

            Path(projected_dir).mkdir(parents=True, exist_ok=True)
            Path(projected_dir_lap).mkdir(parents=True, exist_ok=True)
            Path(projected_dir_original).mkdir(parents=True, exist_ok=True)

            for i in range(len(ds)):
                features = ds[i]['features'].data.reshape(*ds[i]['features'].shape[:-2], -1)
                clean_hrtf = interpolate_fft(settings, cs, features, sphere, sphere_triangles, sphere_coeffs,
                                             cube, edge_len=settings.hrtf_size)

                subject_id = str(ds.subject_ids[i])
                side = ds.sides[i]


                hrir_original, _ = get_hrtf_from_ds(settings, ds, i, domain='time')
                hrtf_original, phase_original, sphere_original = get_hrtf_from_ds(settings, ds, i, domain='mag')
                sphere_original_full = []
                sphere_original_full_hrtf = []
                for index, coordinates in enumerate(sphere_original):
                    position = {'coordinates': coordinates, 'IR': hrir_original[index]}
                    sphere_original_full.append(position)
                    position_hrtf = {'coordinates': coordinates, 'TF': hrtf_original[index], 'phase': phase_original[index]}
                    sphere_original_full_hrtf.append(position_hrtf)

                sphere_original_selected = []
                sphere_original_selected_hrtf = []

                edge_len = int(int(settings.hrtf_size) / int(settings.upscale_factor))
                projection_filename_lap = f'{settings.projection_dir}/{settings.dataset}_projection_lap_{settings.lap_factor}_{edge_len}'

                with open(projection_filename_lap, "rb") as file:
                    cube_lap, sphere_lap, sphere_triangles_lap, sphere_coeffs_lap, measured_coords_lap = pickle.load(file)

                for measured_coord_lap in measured_coords_lap:
                    try:
                        index = [tuple([x['coordinates'][0], x['coordinates'][1]]) for x in sphere_original_full].index(
                            measured_coord_lap)
                    except ValueError as e:
                        print(e)
                    else:
                        sphere_original_selected.append(sphere_original_full[index])
                        sphere_original_selected_hrtf.append(sphere_original_full_hrtf[index])

                cs_lap = CubedSphere(sphere_coords=[tuple(x['coordinates']) for x in sphere_original_selected],
                                     indices=[[x] for x in np.arange(int(settings.lap_factor))])
                hrtf_lap = interpolate_fft(settings, cs_lap, np.array([np.array(x['IR']) for x in sphere_original_selected]),
                                           sphere_lap, sphere_triangles_lap, sphere_coeffs_lap, cube_lap, edge_len=edge_len)

                with open('%s/%s_mag_%s%s.pickle' % (projected_dir, settings.dataset, subject_id, side), "wb") as file:
                    pickle.dump(clean_hrtf, file)

                with open('%s/%s_mag_%s%s.pickle' % (projected_dir_lap, settings.dataset, subject_id, side), "wb") as file:
                    pickle.dump(hrtf_lap, file)

                with open('%s/%s_mag_%s%s.pickle' % (projected_dir_original, settings.dataset, subject_id, side), "wb") as file:
                    pickle.dump(np.array([np.array(x['IR']) for x in sphere_original_selected]), file)

            projected_dir_merge = '/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/lap_full_pickle_' + str(settings.hrtf_size) + '_merge'
            merge_left_right_hrtfs(projected_dir, projected_dir_merge)

            projected_dir_lap_merge = '/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/lap_full_pickle_' + str(settings.lap_factor) + '_' + str(settings.hrtf_size) + '_merge'
            merge_left_right_hrtfs(projected_dir_lap, projected_dir_lap_merge)

    # projected_dir_merge = '/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/lap_full_pickle_' + str(settings.hrtf_size) + '_merge'
    # projected_dir_lap_merge = '/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/lap_full_pickle_' + str(lap_factor) + '_' + str(settings.hrtf_size) + '_merge'
    # projected_dir_original = '/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/lap_full_pickle_original_' + str(lap_factor)
    #
    # filename = projected_dir_merge + '/SONICOM_mag_213.pickle'
    # with open(filename, 'rb') as f:
    #     hr_hrtf = pickle.load(f)
    #
    # filename = projected_dir_lap_merge + '/SONICOM_mag_213.pickle'
    # with open(filename, 'rb') as f:
    #     sr_hrtf = pickle.load(f)
    #
    # filename = projected_dir_original + '/SONICOM_mag_213left.pickle'
    # with open(filename, 'rb') as f:
    #     original_hrtf = pickle.load(f)

    # errors = []
    # for p in range(5):
    #     for w in range(settings.hrtf_size):
    #         for h in range(settings.hrtf_size):
    #             errors.append(spectral_distortion_inner(sr_hrtf[p, w, h], hr_hrtf[p, w, h]))
    # print(f'ERROR: {np.mean(errors)}')

    sphere_coords_hr = []
    sphere_coords_hr_index = []
    for p in range(5):
        for w in range(settings.hrtf_size):
            for h in range(settings.hrtf_size):
                xy_spherical = convert_cube_indices_to_spherical(p, w, h, settings.hrtf_size)
                sphere_coords_hr.append((xy_spherical[0], xy_spherical[1]))
                sphere_coords_hr_index.append([p, w, h])

    cs = CubedSphere(sphere_coords=sphere_coords_hr, indices=sphere_coords_hr_index)

    projection_filename = f'{settings.projection_dir}/{settings.dataset}_projection_{settings.hrtf_size}'
    with open(projection_filename, "rb") as file:
        cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

    for lap_factor in lap_factors:

        settings = Config(tag=None, using_hpc=hpc, lap_factor=lap_factor)

        edge_len = int(int(settings.hrtf_size) / int(settings.upscale_factor))
        projection_filename_lap = f'{settings.projection_dir}/{settings.dataset}_projection_lap_{settings.lap_factor}_{edge_len}'

        with open(projection_filename_lap, "rb") as file:
            cube_lap, sphere_lap, sphere_triangles_lap, sphere_coeffs_lap, measured_coords_lap = pickle.load(file)

        if report_evaluation:
            sub_ids = [201, 202, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213]

        for sub_id in sub_ids:
            if report_evaluation:
                projected_dir_original = '/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/lap_full_pickle_original_' + str(lap_factor)
                filename = projected_dir_original + f'/SONICOM_mag_{sub_id}left.pickle'
                with open(filename, 'rb') as f:
                    hrirs_left = pickle.load(f)
                filename = projected_dir_original + f'/SONICOM_mag_{sub_id}right.pickle'
                with open(filename, 'rb') as f:
                    hrirs_right = pickle.load(f)
                print(projected_dir_original + f'/SONICOM_mag_{sub_id}.pickle')

            else:
                file = f'{settings.data_dirs_path}/lap_data/LAP_Task2_Sparse_HRTFs/LAPtask2_{settings.lap_factor}_{sub_id}.sofa'
                print(f'LAP file: {file}')
                sofa = sf.read_sofa(file)
                hrirs = sofa.Data_IR
                hrirs_left = hrirs[:, 0, :]
                hrirs_right = hrirs[:, 1, :]

                for id in [201, 202, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213]:
                    sofa_fn = f'{settings.data_dirs_path}/lap_data/LAP_Task2_Full_HRTFs/P0{id}_FreeFieldCompMinPhase_48kHz.sofa'
                    sofa_ds = create_sparse_hrtf(sofa_fn, int(settings.lap_factor))
                    if np.all(sofa.Data_IR == sofa_ds.Data_IR):
                        print(f'Input: {file}')
                        print(f'Target: {sofa_fn}')
                        targets.append({'Input': {'id': sub_id, 'factor': settings.lap_factor}, 'Target': id})


            cs_lap = CubedSphere(sphere_coords=measured_coords_lap,
                                 indices=[[x] for x in np.arange(len(measured_coords_lap))])
            hrtf_left_lap = interpolate_fft(settings, cs_lap, hrirs_left,
                                       sphere_lap, sphere_triangles_lap, sphere_coeffs_lap, cube_lap, edge_len=edge_len)
            hrtf_right_lap = interpolate_fft(settings, cs_lap, hrirs_right,
                                       sphere_lap, sphere_triangles_lap, sphere_coeffs_lap, cube_lap, edge_len=edge_len)

            hrtf_lap = torch.tensor(np.concatenate((hrtf_left_lap, hrtf_right_lap), axis=3))

            # for id in [201, 202, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213]:
            #     # filename = projected_dir_merge + '/SONICOM_mag_'+ str(id) +'.pickle'
            #     # with open(filename, 'rb') as f:
            #     #     hr_hrtf = pickle.load(f)
            #
            #     # filename = projected_dir_lap_merge + '/SONICOM_mag_'+ str(id) +'.pickle'
            #     # with open(filename, 'rb') as f:
            #     #     sr_hrtf = pickle.load(f)
            #
            #     # filename = projected_dir_original + '/SONICOM_mag_'+ str(id) +'left.pickle'
            #     # with open(filename, 'rb') as f:
            #     #     original_hrtf = pickle.load(f)
            #
            #     file = f'{settings.data_dirs_path}/lap_data/LAP_Task2_Full_HRTFs/P0201_FreeFieldCompMinPhase_48kHz.sofa'
            #     print(f'LAP file: {file}')
            #     sofa = sf.read_sofa(file)
            #     hrirs_full = sofa.Data_IR
            #     hrirs_left_full = hrirs_full[:, 1, :]
            #     hrirs_right_full = hrirs_full[:, 0, :]
            #
            #     hrirs_full = torch.tensor(np.concatenate((hrirs_left_full, hrirs_right_full), axis=1))
            #
            #     errors = []
            #     for p in range(len(hrirs_full)):
            #         errors.append(spectral_distortion_inner(barycentric_sr_merged[p], hrirs_full[p]))
            #     print(f'ERROR: {np.mean(errors)}')
            #
            #     # errors = []
            #     # for p in range(5):
            #     #     for w in range(settings.hrtf_size):
            #     #         for h in range(settings.hrtf_size):
            #     #             errors.append(spectral_distortion_inner(hrtf_lap[p, w, h], hr_hrtf[p, w, h]))
            #     # print(f'ERROR: {np.mean(errors)}')


            # ==================================
            # need to use protected member to get this data, no getters
            # projection_filename_hr = f'{settings.projection_dir}/{settings.dataset}_projection_{settings.hrtf_size}'
            # with open(projection_filename_hr, "rb") as file:
            #     cube_hr, sphere_hr, sphere_triangles_hr, sphere_coeffs_hr = pickle.load(file)
            #
            # hr_file = f'/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/LAP_Task2_Full_HRTFs/P0201_FreeFieldCompMinPhase_48kHz.sofa'
            # print(f'HR file: {hr_file}')
            # sofa = sf.read_sofa(hr_file)
            # hrirs_hr = sofa.Data_IR
            # hrirs_hr_left = hrirs_hr[:, 1, :]
            # hrirs_hr_right = hrirs_hr[:, 0, :]
            #
            # coords_clean = [tuple([np.radians(x[1]), np.radians(x[0]-180)]) for x in sofa.SourcePosition]
            #
            # cs_hr = CubedSphere(sphere_coords=coords_clean,
            #                     indices=[[x] for x in np.arange(int(len(coords_clean)))])
            #
            # clean_hrtf_left = interpolate_fft(settings, cs_hr, hrirs_hr_left, sphere_hr, sphere_triangles_hr, sphere_coeffs_hr,
            #                              cube_hr, edge_len=edge_len)
            #
            # clean_hrtf_right = interpolate_fft(settings, cs_hr, hrirs_hr_right, sphere_hr, sphere_triangles_hr, sphere_coeffs_hr,
            #                              cube_hr, edge_len=edge_len)
            #
            # clean_hrtf = torch.tensor(np.concatenate((clean_hrtf_left, clean_hrtf_right), axis=3))

            # ==================================

            ngpu = settings.ngpu
            device = torch.device(settings.device_name if (torch.cuda.is_available() and ngpu > 0) else "cpu")

            nbins = settings.nbins_hrtf
            if settings.merge_flag:
                nbins = settings.nbins_hrtf * 2

            model = Generator(upscale_factor=settings.upscale_factor, nbins=nbins).to(device)
            print('Using SRGAN model')

            # Load super-resolution model weights (always uses the CPU due to HPC having long wait times)
            try:
                tag = f"pub-prep-upscale-{settings.dataset}-LAP-{settings.lap_factor}-{int(settings.hrtf_size/settings.upscale_factor)}/Gen.pt".replace('_', '-')
                print(f'Weights: {tag}')
                model.load_state_dict(torch.load(f'{settings.data_dirs_path}{settings.runs_folder}/{tag}', map_location=torch.device('cpu')))
            except OSError:
                print(f"Unable to load SRGAN model weights `{tag}` successfully.")
                return
            print(f"Load SRGAN model weights `{tag}` successfully.")

            lr = torch.from_numpy(np.array([torch.moveaxis(hrtf_lap.float(), -1, 0)])).to(device)

            with torch.no_grad():
                sr = model(lr)

            sr = torch.moveaxis(sr[0], 0, -1).detach().cpu()

            sphere_original_filename = f'{settings.projection_dir}/{settings.dataset}_original'
            with open(sphere_original_filename, "rb") as file:
                sphere_original = pickle.load(file)

            postprocessing_projection_filename = f'{settings.postprocessing_dir}/{settings.dataset}_postprocessing_projection_{settings.hrtf_size}'
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
                                                              sphere_coords=sphere_original)
                    coeffs = calc_barycentric_coordinates(elevation=sphere_coord[0], azimuth=sphere_coord[1],
                                                          closest_points=triangle_vertices)
                    euclidean_sphere_triangles.append(triangle_vertices)
                    euclidean_sphere_coeffs.append(coeffs)

                with open(postprocessing_projection_filename, "wb") as f:
                    pickle.dump((euclidean_sphere_triangles, euclidean_sphere_coeffs, xy_postprocessing), f)

            sr_hrtf_left = sr[:, :, :, :settings.nbins_hrtf]
            sr_hrtf_right = sr[:, :, :, settings.nbins_hrtf:]

            barycentric_sr_left = interpolate_fft(settings, cs, sr_hrtf_left, sphere_original, euclidean_sphere_triangles,
                                                  euclidean_sphere_coeffs, cube, edge_len=settings.hrtf_size, cs_output=False, time_domain_flag=False)
            barycentric_sr_right = interpolate_fft(settings, cs, sr_hrtf_right, sphere_original, euclidean_sphere_triangles,
                                                   euclidean_sphere_coeffs, cube, edge_len=settings.hrtf_size, cs_output=False, time_domain_flag=False)

            barycentric_sr_merged = torch.tensor(np.concatenate((barycentric_sr_left, barycentric_sr_right), axis=1))

            replace_original_nodes = True
            if replace_original_nodes:
                edge_len = int(int(settings.hrtf_size) / int(settings.upscale_factor))
                projection_filename = f'{settings.projection_dir}/{settings.dataset}_projection_lap_{settings.lap_factor}_{edge_len}'

                with open(projection_filename, "rb") as file:
                    cube_lap, sphere_lap, sphere_triangles_lap, sphere_coeffs_lap, measured_coords_lap = pickle.load(
                        file)

                orginal_hrir = torch.tensor(np.concatenate((hrirs_left, hrirs_right), axis=1))

                for measured_coord_lap_index, measured_coord_lap in enumerate(measured_coords_lap):
                    sphere_index = sphere_original.index(list(measured_coord_lap))

                    hrir_temp_merge = orginal_hrir[measured_coord_lap_index]
                    hrir_temp_left = hrir_temp_merge[:settings.nbins_hrtf]
                    hrir_temp_right = hrir_temp_merge[settings.nbins_hrtf:]

                    hrtf_temp_left, _ = calc_hrtf(settings, [
                        remove_itd(hrir_temp_left, int(len(hrir_temp_left) * 0.04), len(hrir_temp_left))])
                    hrtf_temp_right, _ = calc_hrtf(settings, [
                        remove_itd(hrir_temp_right, int(len(hrir_temp_right) * 0.04), len(hrir_temp_right))])

                    orginal_hrtf = torch.tensor(np.concatenate((hrtf_temp_left[0], hrtf_temp_right[0]), axis=0))
                    barycentric_sr_merged[sphere_index] = orginal_hrtf

            if report_evaluation:
                file_name = f'/SONICOM_{lap_factor}_{sub_id}.pickle'
            else:
                file_name = f'/LAPtask2_{settings.lap_factor}_{sub_id}.pickle'
            with open(settings.lap_dir + file_name, "wb") as file:
                 pickle.dump(barycentric_sr_merged, file)

    convert_to_sofa(settings.lap_dir, settings, cube=None, sphere=sphere_original, lap_factor=True)

    print('SOFA Files Created')

    print(targets)

def run_baseline_plots(hpc):

    # baselines = ['gan', 'barycentric', 'sh', 'lap', 'lap_reports']
    lap_factors = ['100', '19', '5', '3']

    baselines = ['lap']
    # lap_factors = ['5']

    lap_folder = '0.8_16_lap'
    print(f'LAP Version: {lap_folder}')
    # lap_reports_folder = '0.8_16_lap_reports'
    lap_folder = ''

    config = Config(None, using_hpc=hpc)
    Path(config.data_dirs_path + '/lap_plots').mkdir(parents=True, exist_ok=True)

    targets = [{'Input': {'id': 1, 'factor': '100'}, 'Target': 201}, {'Input': {'id': 2, 'factor': '100'}, 'Target': 205}, {'Input': {'id': 3, 'factor': '100'}, 'Target': 210}, {'Input': {'id': 1, 'factor': '19'}, 'Target': 202}, {'Input': {'id': 2, 'factor': '19'}, 'Target': 206}, {'Input': {'id': 3, 'factor': '19'}, 'Target': 211}, {'Input': {'id': 1, 'factor': '5'}, 'Target': 203}, {'Input': {'id': 2, 'factor': '5'}, 'Target': 207}, {'Input': {'id': 3, 'factor': '5'}, 'Target': 212}, {'Input': {'id': 1, 'factor': '3'}, 'Target': 204}, {'Input': {'id': 2, 'factor': '3'}, 'Target': 208}, {'Input': {'id': 3, 'factor': '3'}, 'Target': 213}]

    # for target in targets:
    #         sr = f'/home/ahogg/PycharmProjects/HRTF-GAN/lap_results/sofa_min_phase/LAPtask2_{target["Input"]["factor"]}_{target["Input"]["id"]}.sofa'
    #         # hr = f'/home/ahogg/Documents/HRTF_Test/{hr_sub}/HRTF/48kHz/{hr_sub}_FreeFieldComp_48kHz.sofa'
    #         # hr = f'/home/ahogg/Downloads/SONICOM/{hr_sub}/HRTF/HRTF/48kHz/{hr_sub}_FreeFieldComp_48kHz.sofa'
    #         hr = f'/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/LAP_Task2_Full_HRTFs/P0{target["Target"]}_FreeFieldCompMinPhase_48kHz.sofa'
    #         print(f'Gen: {sr}')
    #         print(f'Target: {hr}')
    #         metrics, threshold_bool, df = lap.calculate_task_two_metrics(hr, sr)

    barycentric_errors = []
    sh_errors = []
    gan_errors = []
    lap_errors = []
    lap_reports_errors = []
    for baseline in baselines:
        for lap_factor in lap_factors:

            dataset = 'SONICOM'
            tag = {'tag': None}
            config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset, lap_factor=lap_factor)

            if baseline == 'barycentric':
                output_path = config.barycentric_hrtf_dir + '/barycentric_interpolated_data_lap_' + lap_factor
            elif baseline == 'sh':
                output_path = config.sh_hrtf_dir + '/sh_interpolated_data_lap_' + config.lap_factor
            elif baseline == 'gan':
                output_path = f'{config.data_dirs_path}/runs-pub-fa/pub-prep-upscale-{config.dataset}-LAP-{config.lap_factor}-{int(config.hrtf_size/config.upscale_factor)}/valid/original_coordinates'
            elif baseline == 'lap':
                output_path = f'/home/ahogg/PycharmProjects/HRTF-GAN/lap_results/{lap_folder}'
            elif baseline == 'lap_reports':
                output_path = f'/home/ahogg/PycharmProjects/HRTF-GAN/lap_results/{lap_reports_folder}'

            file_path = output_path + '/sofa_min_phase'

            hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(file_path) if '.sofa' in hrtf_file_name]
            if not os.path.exists(file_path):
                raise Exception(f'File path does not exist or does not have write permissions ({file_path})')

            # Calculate LSD
            errors = []
            for file in hrtf_file_names:

                # target_sofa_file = config.valid_lap_original_hrtf_merge_dir + '/sofa_min_phase/' + file
                if baseline == 'lap_reports':
                    sub_id = int(file.split('_')[-1].replace('.sofa', ''))
                    factor = int(file.split('_')[-2].replace('.sofa', ''))
                    if str(factor) != lap_factor:
                        continue
                    target_sofa_file = f'/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/LAP_Task2_Full_HRTFs/P0{sub_id}_FreeFieldCompMinPhase_48kHz.sofa'

                elif baseline == 'lap':
                    sub_id = int(file.split('_')[-1].replace('.sofa', ''))
                    factor = int(file.split('_')[-2].replace('.sofa', ''))
                    if str(factor) != lap_factor:
                        continue
                    target = next(item for item in targets if item["Input"] == {'id': int(sub_id), 'factor': str(factor)})
                    target_sofa_file = f'/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/LAP_Task2_Full_HRTFs/P0{target["Target"]}_FreeFieldCompMinPhase_48kHz.sofa'
                else:
                    sub_id = int(file.split('_')[-1].replace('.sofa', ''))
                    target_sofa_file = f'{config.raw_hrtf_dir}/{config.dataset}/P{str(sub_id).zfill(4)}/HRTF/HRTF/48kHz/P{str(sub_id).zfill(4)}_FreeFieldComp_48kHz.sofa'
                generated_sofa_file = file_path + '/' + file
                metrics, threshold_bool, df = lap.calculate_task_two_metrics(target_sofa_file, generated_sofa_file)

                idt_error = metrics[0]
                ild_error = metrics[1]
                lsd_error = metrics[2]

                errors.append({'subject_id': sub_id, 'total_itd_error': idt_error, 'total_ild_error': ild_error,
                                'total_lsd_error': lsd_error})


            print('Mean ITD Error: %0.3f' % np.mean([error['total_itd_error'] for error in errors]))
            print('Mean ILD Error: %0.3f' % np.mean([error['total_ild_error'] for error in errors]))
            print('Mean LSD Error: %0.3f' % np.mean([error['total_lsd_error'] for error in errors]))

            if baseline == 'barycentric':
                barycentric_errors.append({'lap_factor': lap_factor, 'errors': errors})
            elif baseline == 'sh':
                sh_errors.append({'lap_factor': lap_factor, 'errors': errors})
            elif baseline == 'gan':
                gan_errors.append({'lap_factor': lap_factor, 'errors': errors})
            elif baseline == 'lap':
                lap_errors.append({'lap_factor': lap_factor, 'errors': errors})
            elif baseline == 'lap_reports':
                lap_reports_errors.append({'lap_factor': lap_factor, 'errors': errors})

    error_types = ['total_itd_error', 'total_ild_error', 'total_lsd_error']
    error_units = ['(µs)', '(dB)', '(dB)']
    x_limits = [(20, 200), (0, 11.5), (2.8, 11)]
    error_thresholds = [100, 4.4, 7.4]
    for baseline in baselines:
        for error_index, error_type in enumerate(error_types):

            if baseline == 'barycentric':
                plot_errors = barycentric_errors
                title = 'Barycentric Interpolation'
            elif baseline == 'sh':
                plot_errors = sh_errors
                title = 'Spherical Harmonics Interpolation'
            elif baseline == 'gan':
                plot_errors = gan_errors
                title = 'SRGAN'
            elif baseline == 'lap':
                plot_errors = lap_errors
                title = 'SRGAN LAP'
            elif baseline == 'lap_reports':
                plot_errors = lap_reports_errors
                title = 'SRGAN LAP Reports'

            plot_errors_3 = np.array([[y[error_type] for y in x['errors']] for x in plot_errors if x['lap_factor'] == '3']).flatten()
            plot_errors_5 = np.array([[y[error_type] for y in x['errors']] for x in plot_errors if x['lap_factor'] == '5']).flatten()
            plot_errors_19 = np.array([[y[error_type] for y in x['errors']] for x in plot_errors if x['lap_factor'] == '19']).flatten()
            plot_errors_100 = np.array([[y[error_type] for y in x['errors']] for x in plot_errors if x['lap_factor'] == '100']).flatten()

            fig, ax = plt.subplots(figsize=(8, 2.5))
            data = pd.DataFrame(np.array([plot_errors_100, plot_errors_19, plot_errors_5, plot_errors_3]).T, columns = lap_factors)
            x_label = error_type.replace('_', ' ').replace('total ', '').upper().replace('ERROR', 'Error') + ' ' + error_units[error_index]
            sns.boxplot(data=data, orient="h", whis=[0, 100], width=.8, palette="vlag", gap=.1)
            ax.set(ylabel='Sparsity Level', xlabel=x_label)
            ax.set_title(title)
            ax.axvline(x=error_thresholds[error_index], linewidth=2, color='r', linestyle='--')
            ax.set_xlim(x_limits[error_index])
            ax.xaxis.grid(True)  # Show the vertical gridlines
            fig.tight_layout()
            fig.savefig(f'lap_plots/{baseline}_{error_type}.png')
            plt.close()

    return

def run_evaluation(hpc, experiment_id, type, test_id=None):
    print(f'Running {type} experiment {experiment_id}')
    config_files = []
    if experiment_id == 1:
        upscale_factors = [5]
        datasets = ['SONICOM']
        for dataset in datasets:
            for upscale_factor in upscale_factors:
                tags = [{'tag': f'pub-prep-upscale-{dataset}-LAP-{upscale_factor}'}]
                for tag in tags:
                    config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset, lap_factor='lap_100')
                    if upscale_factor == '5':
                        config.upscale_factor = 2
                        config.hrtf_size = 16
                    config_files.append(config)
    else:
        print('Experiment does not exist')
        return

    print(f'{len(config_files)} config files created successfully.')
    if test_id is not None:
        if test_id.isnumeric():
            test_id = int(test_id)
            config_files = [config_files[test_id]]
        else:
            for config in config_files:
                if config.tag == test_id:
                    config_files = [config]
                    break

    print(f'Running a total of {len(config_files)} config files')
    for config in config_files:
        if type == 'lsd':
            _, test_prefetcher = load_dataset(config, mean=None, std=None)
            print("Loaded all datasets successfully.")
            print('TO DO')
        elif type == 'loc':
            _, test_prefetcher = load_dataset(config, mean=None, std=None)
            print("Loaded all datasets successfully.")

            file_ext = 'loc_errors.pickle'
            if config.barycentric_postprocessing:
                file_original_path = f'{config.data_dirs_path}/runs-pub-fa/pub-prep-upscale-{config.dataset}-LAP-5/valid/original_coordinates/sofa_min_phase'
                hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(file_original_path) if '.sofa' in hrtf_file_name]
                if not os.path.exists(file_original_path):
                    raise Exception(f'File path does not exist or does not have write permissions ({file_original_path})')
            else:
                file_path = f'{config.data_dirs_path}/runs-pub-fa/pub-prep-upscale-{config.dataset}-LAP-100/valid/nodes_replaced/sofa_min_phase'
                hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(file_path) if '.sofa' in hrtf_file_name]
                if not os.path.exists(file_path):
                    raise Exception(f'File path does not exist or does not have write permissions ({file_path})')

            loc_errors = []
            for file in hrtf_file_names:
                if config.barycentric_postprocessing:
                    from_database = True
                    if from_database:
                        sub_id = int(file.split('_')[-1].replace('.sofa', ''))
                        target_sofa_original_file = f'/home/ahogg/Downloads/SONICOM/P{str(sub_id).zfill(4)}/HRTF/HRTF/48kHz/P{str(sub_id).zfill(4)}_FreeFieldComp_48kHz.sofa'
                        # target_sofa_original_file = f'/home/ahogg/Downloads/SONICOM/P{str(sub_id).zfill(4)}/HRTF/HRTF/48kHz/P{str(sub_id).zfill(4)}_FreeFieldCompMinPhase_48kHz.sofa'
                    else:
                        target_sofa_original_file = config.valid_original_hrtf_merge_dir + '/sofa_with_phase/' + file
                        target_sofa_original_file = target_sofa_original_file.replace('single_panel', 'cube_sphere')  # For single panel use cube sphere

                    generated_sofa_original_file = file_original_path + '/' + file
                    metrics, threshold_bool, df = lap.calculate_task_two_metrics(target_sofa_original_file, generated_sofa_original_file)
                else:
                    target_sofa_file = config.valid_hrtf_merge_dir + '/sofa_min_phase/' + file
                    target_sofa_file = target_sofa_file.replace('single_panel', 'cube_sphere')  # For single panel use cube sphere

                    generated_sofa_file = file_path + '/' + file
                    metrics, threshold_bool, df = lap.calculate_task_two_metrics(target_sofa_file, generated_sofa_file)

            #     eng = matlab.engine.start_matlab()
            #     s = eng.genpath(config.amt_dir)
            #     eng.addpath(s, nargout=0)
            #     s = eng.genpath(config.data_dirs_path)
            #     eng.addpath(s, nargout=0)
            #
            #     print(f'Target: {target_sofa_file}')
            #     print(f'Generated: {generated_sofa_file}')
            #     [pol_acc1, pol_rms1, querr1] = eng.calc_loc(generated_sofa_file, target_sofa_file, nargout=3)
            #     subject_id = ''.join(re.findall(r'\d+', file))
            #     loc_errors.append([subject_id, pol_acc1, pol_rms1, querr1])
            #     print('pol_acc1: %s' % pol_acc1)
            #     print('pol_rms1: %s' % pol_rms1)
            #     print('querr1: %s' % querr1)
            #
            #     eng.quit()
            #
            # print('Mean ACC Error: %0.3f' % np.mean([error[1] for error in loc_errors]))
            # print('Mean RMS Error: %0.3f' % np.mean([error[2] for error in loc_errors]))
            # print('Mean QUERR Error: %0.3f' % np.mean([error[3] for error in loc_errors]))
            # with open(f'{file_path}/{file_ext}', "wb") as file:
            #     pickle.dump(loc_errors, file)


        else:
            print(f'Type ({type}) does not exist')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--hpc")
    parser.add_argument("--exp")
    parser.add_argument("--type")
    parser.add_argument("--test")
    args = parser.parse_args()

    if args.hpc == "True":
        hpc = True
    elif args.hpc == "False":
        hpc = False
    else:
        raise RuntimeError("Please enter 'True' or 'False' for the hpc tag (-c/--hpc)")





    # run_lap(hpc)
    run_baseline_plots(hpc)

    # lap_factors = ['100', '19', '5', '3']
    # sub_ids = [1, 2, 3]

    # lap_factors = ['100']
    # sub_ids = [1]
    # targets = [{'Input': {'id': 1, 'factor': '100'}, 'Target': 201}, {'Input': {'id': 2, 'factor': '100'}, 'Target': 205}, {'Input': {'id': 3, 'factor': '100'}, 'Target': 210}, {'Input': {'id': 1, 'factor': '19'}, 'Target': 202}, {'Input': {'id': 2, 'factor': '19'}, 'Target': 206}, {'Input': {'id': 3, 'factor': '19'}, 'Target': 211}, {'Input': {'id': 1, 'factor': '5'}, 'Target': 203}, {'Input': {'id': 2, 'factor': '5'}, 'Target': 207}, {'Input': {'id': 3, 'factor': '5'}, 'Target': 212}, {'Input': {'id': 1, 'factor': '3'}, 'Target': 204}, {'Input': {'id': 2, 'factor': '3'}, 'Target': 208}, {'Input': {'id': 3, 'factor': '3'}, 'Target': 213}]
    #
    # for target in targets:
    #         sr = f'/home/ahogg/PycharmProjects/HRTF-GAN/lap_results/sofa_min_phase/LAPtask2_{target["Input"]["factor"]}_{target["Input"]["id"]}.sofa'
    #         # hr = f'/home/ahogg/Documents/HRTF_Test/{hr_sub}/HRTF/48kHz/{hr_sub}_FreeFieldComp_48kHz.sofa'
    #         # hr = f'/home/ahogg/Downloads/SONICOM/{hr_sub}/HRTF/HRTF/48kHz/{hr_sub}_FreeFieldComp_48kHz.sofa'
    #         hr = f'/home/ahogg/PycharmProjects/HRTF-GAN/lap_data/LAP_Task2_Full_HRTFs/P0{target["Target"]}_FreeFieldCompMinPhase_48kHz.sofa'
    #         print(f'Gen: {sr}')
    #         print(f'Target: {hr}')
    #         metrics, threshold_bool, df = lap.calculate_task_two_metrics(hr, sr)


