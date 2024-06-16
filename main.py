import argparse
import os
import pickle
import torch
import numpy as np
import importlib
import re

from hartufo import CollectionSpec, SideSpec, SubjectSpec, HrirSpec, AnthropometrySpec, ImageSpec
from operator import itemgetter
from preprocessing.convert_coordinates import convert_sphere_to_cube
import sofar as sf

from config import Config
from model.train import train
from model.test import test
from model.util import load_dataset
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.utils import interpolate_fft, generate_euclidean_cube, convert_to_sofa, \
     merge_files, gen_sofa_preprocess, get_hrtf_from_ds, clear_create_directories, remove_itd, calc_hrtf
from model import util
from baselines.barycentric_interpolation import run_barycentric_interpolation
from baselines.sh_interpolation import run_sh_interpolation
from baselines.hrtf_selection import run_hrtf_selection
from evaluation.evaluation import run_lsd_evaluation, run_localisation_evaluation

from spatialaudiometrics import lap_challenge as lap

PI_4 = np.pi / 4

# Random seed to maintain reproducible results
torch.manual_seed(0)
np.random.seed(0)


def main(config, mode):
    # Initialise Config object
    data_dir = config.raw_hrtf_dir / config.dataset
    print(os.getcwd())
    print(config.dataset)

    imp = importlib.import_module('hartufo')
    load_function = getattr(imp, config.dataset.title())

    if mode == 'generate_projection':
        # Must be run in this mode once per dataset, finds barycentric coordinates for each point in the cubed sphere
        # No need to load the entire dataset in this case
        if config.lap_factor is None:
            ds = ds = load_function(data_dir, features_spec=HrirSpec(domain='time', side='left', samplerate=config.hrir_samplerate), subject_ids='first')
            # need to use protected member to get this data, no getters
            cs = CubedSphere(mask=ds[0]['features'].mask, row_angles=ds.fundamental_angles, column_angles=ds.orthogonal_angles)
            generate_euclidean_cube(config, cs.get_sphere_coords(), edge_len=config.hrtf_size)

            _, sphere_original = get_hrtf_from_ds(config, ds, 0, domain='time')
            filename = f'{config.projection_dir}/{config.dataset}_original'
            with open(filename, "wb") as file:
                pickle.dump(sphere_original, file)
        elif config.lap_factor == '100' or config.lap_factor == '19' or config.lap_factor == '5' or config.lap_factor == '3':
            edge_len = int(int(config.hrtf_size) / int(config.upscale_factor))
            sofa = sf.read_sofa(f'{config.data_dirs_path}/lap_data/LAPtask2_{config.lap_factor}_1.sofa')
            generate_euclidean_cube(config, [tuple([np.radians(x[1]), np.radians(x[0]-180)]) for x in sofa.SourcePosition], edge_len=edge_len,
                                    filename=f'lap_{config.lap_factor}_{edge_len}', output_measured_coords=True)
        else:
            raise Exception('LAP factor not found')

    elif mode == 'preprocess':
        # Interpolates data to find HRIRs on cubed sphere, then FFT to obtain HRTF, finally splits data into train and
        # val sets and saves processed data
        ds = load_function(data_dir, features_spec=HrirSpec(domain='time', side='both', length=config.nbins_hrtf*2, samplerate=config.hrir_samplerate))
        cs = CubedSphere(mask=ds[0]['features'].mask, row_angles=ds.fundamental_angles, column_angles=ds.orthogonal_angles)

        # need to use protected member to get this data, no getters
        projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
        with open(projection_filename, "rb") as file:
            cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        # Clear/Create directories
        clear_create_directories(config)

        # Split data into train and test sets
        train_size = int(len(set(ds.subject_ids)) * config.train_samples_ratio)
        train_sample = np.random.choice(list(set(ds.subject_ids)), train_size, replace=False)

        # collect all train_hrtfs to get mean and sd
        train_hrtfs = []
        # for i in range(len(ds)):
        for i in range(20):
            if i % 10 == 0:
                print(f"HRTF {i} out of {len(ds)} ({round(100 * i / len(ds))}%)")

            # Verification that HRTF is valid
            if np.isnan(ds[i]['features']).any():
                print(f'HRTF (Subject ID: {i}) contains nan values')
                continue

            hrir_original, _ = get_hrtf_from_ds(config, ds, i, domain='time')
            hrtf_original, phase_original, sphere_original = get_hrtf_from_ds(config, ds, i, domain='mag')

            edge_len = int(int(config.hrtf_size) / int(config.upscale_factor))
            cube_lap = None
            sphere_lap = None


            ####################LAP########################
            ###############################################

            if config.lap_factor:
                projection_filename = f'{config.projection_dir}/{config.dataset}_projection_lap_{config.lap_factor}_{edge_len}'

                with open(projection_filename, "rb") as file:
                    cube_lap, sphere_lap, sphere_triangles_lap, sphere_coeffs_lap, measured_coords_lap = pickle.load(file)

                sphere_original_full = []
                sphere_original_full_hrtf = []
                for index, coordinates in enumerate(sphere_original):
                    position = {'coordinates': coordinates, 'IR': hrir_original[index]}
                    sphere_original_full.append(position)
                    position_hrtf = {'coordinates': coordinates, 'TF': hrtf_original[index], 'phase': phase_original[index]}
                    sphere_original_full_hrtf.append(position_hrtf)

                sphere_original_selected = []
                sphere_original_selected_hrtf = []
                for measured_coord_lap in measured_coords_lap:
                    try:
                        index = [tuple([x['coordinates'][0], x['coordinates'][1]]) for x in sphere_original_full].index(measured_coord_lap)
                    except ValueError as e:
                        print(e)
                    else:
                        sphere_original_selected.append(sphere_original_full[index])
                        sphere_original_selected_hrtf.append(sphere_original_full_hrtf[index])

                cs_lap = CubedSphere(sphere_coords=[tuple(x['coordinates']) for x in sphere_original_selected], indices=[[x] for x in np.arange(int(config.lap_factor))])
                hrtf_lap = interpolate_fft(config, cs_lap, np.array([np.array(x['IR']) for x in sphere_original_selected]), sphere_lap, sphere_triangles_lap, sphere_coeffs_lap, cube_lap, edge_len=edge_len)

                hrir_original_lap = torch.tensor(np.array([np.array(x['IR']) for x in sphere_original_selected]))

            ###############################################

            else:
                features = ds[i]['features'].data.reshape(*ds[i]['features'].shape[:-2], -1)
                clean_hrtf = interpolate_fft(config, cs, features, sphere, sphere_triangles, sphere_coeffs,
                                             cube, edge_len=config.hrtf_size)
                train_hrtfs.append(clean_hrtf)

            # save cleaned hrtfdata
            if ds.subject_ids[i] in train_sample:
                projected_dir = config.train_hrtf_dir
                projected_dir_original = config.train_original_hrtf_dir
                if config.lap_factor:
                    projected_dir_lap_original = config.train_lap_original_hrtf_dir
                    projected_dir_lap = config.train_lap_dir
            else:
                projected_dir = config.valid_hrtf_dir
                projected_dir_original = config.valid_original_hrtf_dir
                if config.lap_factor:
                    projected_dir_lap_original = config.valid_lap_original_hrtf_dir
                    projected_dir_lap = config.valid_lap_dir

            subject_id = str(ds.subject_ids[i])
            side = ds.sides[i]

            if config.lap_factor:
                with open('%s/%s_mag_%s%s.pickle' % (projected_dir_lap, config.dataset, subject_id, side), "wb") as file:
                    pickle.dump(hrtf_lap, file)

                with open('%s/%s_mag_%s%s.pickle' % (projected_dir_lap_original, config.dataset, subject_id, side), "wb") as file:
                    pickle.dump(hrir_original_lap, file)
            else:
                with open('%s/%s_mag_%s%s.pickle' % (projected_dir, config.dataset, subject_id, side), "wb") as file:
                    pickle.dump(clean_hrtf, file)

                with open('%s/%s_mag_%s%s.pickle' % (projected_dir_original, config.dataset, subject_id, side),
                          "wb") as file:
                    pickle.dump(hrtf_original, file)

                with open('%s/%s_phase_%s%s.pickle' % (projected_dir_original, config.dataset, subject_id, side),
                          "wb") as file:
                    pickle.dump(phase_original, file)

                # save dataset mean and standard deviation for each channel, across all HRTFs in the training data
                mean = torch.mean(torch.from_numpy(np.array(train_hrtfs)), [0, 1, 2, 3])
                std = torch.mean(torch.from_numpy(np.array(train_hrtfs)), [0, 1, 2, 3])
                min_hrtf = torch.min(torch.from_numpy(np.array(train_hrtfs)))
                max_hrtf = torch.max(torch.from_numpy(np.array(train_hrtfs)))
                mean_std_filename = config.mean_std_filename
                with open(mean_std_filename, "wb") as file:
                    pickle.dump((mean, std, min_hrtf, max_hrtf), file)

        print(f"All HRTFs created (100%)")

        if config.merge_flag:
            merge_files(config)
            print(f"Merged HRTFs created")

        if config.gen_sofa_flag:
            gen_sofa_preprocess(config, cube, sphere, sphere_original, edge_len=edge_len,cube_lap=cube_lap, sphere_lap=sphere_lap)
            print(f"SOFA files created")

    elif mode == 'train':
        # Trains the GANs, according to the parameters specified in Config
        # mean_std_filename = config.mean_std_filename
        # with open(mean_std_filename, "rb") as file:
        #     mean, std, min_hrtf, max_hrtf = pickle.load(file)
        # train_prefetcher, _ = load_dataset(config, mean=torch.cat((mean, mean)), std=torch.cat((std, std)))
        train_prefetcher, _ = load_dataset(config,  mean=None, std=None)
        print("Loaded all datasets successfully.")

        util.initialise_folders(config, overwrite=True)
        train(config, train_prefetcher)

    elif mode == 'test':
        _, test_prefetcher = load_dataset(config, mean=None, std=None)
        print("Loaded all datasets successfully.")

        test(config, test_prefetcher)

        run_lsd_evaluation(config, config.valid_path)
        run_localisation_evaluation(config, config.valid_path)

    elif mode == 'barycentric_baseline':
        print('Barycentric Baseline')

        if config.lap_factor is not None:
            print(f'Dataset: {config.dataset}, LAP Factor: {config.lap_factor}')
            barycentric_output_path = config.barycentric_hrtf_dir + '/barycentric_interpolated_data_lap_' + config.lap_factor
        else:
            print(f'Dataset: {config.dataset}, Upscale Factor: {config.upscale_factor}')
            barycentric_data_folder = f'/barycentric_interpolated_data_{config.upscale_factor}'
            barycentric_output_path = config.barycentric_hrtf_dir + barycentric_data_folder

        cube, sphere = run_barycentric_interpolation(config, barycentric_output_path)

        if config.gen_sofa_flag:
            convert_to_sofa(barycentric_output_path, config, cube, sphere)
            print('Created barycentric baseline sofa files')

        if config.lap_factor is not None:
            file_path = barycentric_output_path + '/sofa_min_phase'
            hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(file_path) if '.sofa' in hrtf_file_name]
            if not os.path.exists(file_path):
                raise Exception(f'File path does not exist or does not have write permissions ({file_path})')

            # Calculate LSD
            errors = []
            for file in hrtf_file_names:
                # target_sofa_file = config.valid_lap_original_hrtf_merge_dir + '/sofa_min_phase/' + file
                sub_id = int(file.split('_')[-1].replace('.sofa', ''))
                target_sofa_file = f'{config.raw_hrtf_dir}/{config.dataset}/P{str(sub_id).zfill(4)}/HRTF/HRTF/48kHz/P{str(sub_id).zfill(4)}_FreeFieldComp_48kHz.sofa'
                generated_sofa_file = file_path + '/' + file
                metrics, threshold_bool, df = lap.calculate_task_two_metrics(target_sofa_file, generated_sofa_file)

                idt_error = metrics[0]
                ild_error = metrics[1]
                lsd_error = metrics[2]

                errors.append({'subject_id': sub_id, 'total_itd_error': idt_error, 'total_ild_error': ild_error, 'total_lsd_error': lsd_error})

            print('Mean ITD Error: %0.3f' % np.mean([error['total_itd_error'] for error in errors]))
            print('Mean ILD Error: %0.3f' % np.mean([error['total_ild_error'] for error in errors]))
            print('Mean LSD Error: %0.3f' % np.mean([error['total_lsd_error'] for error in errors]))

        else:
            config.path = config.barycentric_hrtf_dir

            file_ext = f'lsd_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle'
            run_lsd_evaluation(config, barycentric_output_path, file_ext)

            file_ext = f'loc_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle'
            run_localisation_evaluation(config, barycentric_output_path, file_ext, baseline=True)

    elif mode == 'sh_baseline':
        print('SH Baseline')

        if config.lap_factor is not None:
            print(f'Dataset: {config.dataset}, LAP Factor: {config.lap_factor}')
            sh_output_path = config.sh_hrtf_dir + '/sh_interpolated_data_lap_' + config.lap_factor
        else:
            print(f'Dataset: {config.dataset}, Upscale Factor: {config.upscale_factor}')
            sh_data_folder = f'/sh_interpolated_data_{config.upscale_factor}'
            sh_output_path = config.sh_hrtf_dir + sh_data_folder

        cube, sphere = run_sh_interpolation(config, sh_output_path)

        if config.gen_sofa_flag:
            convert_to_sofa(sh_output_path, config, cube, sphere)
            print('Created sh baseline sofa files')

        if config.lap_factor is not None:
            file_path = sh_output_path + '/sofa_min_phase'
            hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(file_path) if '.sofa' in hrtf_file_name]
            if not os.path.exists(file_path):
                raise Exception(f'File path does not exist or does not have write permissions ({file_path})')

            # Calculate LSD
            errors = []
            for file in hrtf_file_names:
                # target_sofa_file = config.valid_lap_original_hrtf_merge_dir + '/sofa_min_phase/' + file
                sub_id = int(file.split('_')[-1].replace('.sofa', ''))
                target_sofa_file = f'{config.raw_hrtf_dir}/{config.dataset}/P{str(sub_id).zfill(4)}/HRTF/HRTF/48kHz/P{str(sub_id).zfill(4)}_FreeFieldComp_48kHz.sofa'
                generated_sofa_file = file_path + '/' + file
                metrics, threshold_bool, df = lap.calculate_task_two_metrics(target_sofa_file, generated_sofa_file)

                idt_error = metrics[0]
                ild_error = metrics[1]
                lsd_error = metrics[2]

                errors.append({'subject_id': sub_id, 'total_itd_error': idt_error, 'total_ild_error': ild_error, 'total_lsd_error': lsd_error})

            print('Mean ITD Error: %0.3f' % np.mean([error['total_itd_error'] for error in errors]))
            print('Mean ILD Error: %0.3f' % np.mean([error['total_ild_error'] for error in errors]))
            print('Mean LSD Error: %0.3f' % np.mean([error['total_lsd_error'] for error in errors]))

        else:
            config.path = config.sh_hrtf_dir

            file_ext = f'lsd_errors_sh_interpolated_data_{config.upscale_factor}.pickle'
            run_lsd_evaluation(config, sh_output_path, file_ext)

            file_ext = f'loc_errors_sh_interpolated_data_{config.upscale_factor}.pickle'
            run_localisation_evaluation(config, sh_output_path, file_ext, baseline=True)

    elif mode == 'hrtf_selection_baseline':

        run_hrtf_selection(config, config.hrtf_selection_dir)

        projection_filename = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
        with open(projection_filename, "rb") as f:
            (cube, sphere, _, _) = pickle.load(f)

        if config.gen_sofa_flag:
            convert_to_sofa(config.hrtf_selection_dir, config, cube, sphere)
            print('Created hrtf selection baseline sofa files')

        config.path = config.hrtf_selection_dir

        file_ext = f'lsd_errors_hrtf_selection_minimum_data.pickle'
        run_lsd_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='minimum')
        # file_ext = f'loc_errors_hrtf_selection_minimum_data.pickle'
        # run_localisation_evaluation(config, config.hrtf_selection_dir, file_ext, baseline=True, hrtf_selection='minimum')

        file_ext = f'lsd_errors_hrtf_selection_maximum_data.pickle'
        run_lsd_evaluation(config, config.hrtf_selection_dir, file_ext, hrtf_selection='maximum')
        # file_ext = f'loc_errors_hrtf_selection_maximum_data.pickle'
        # run_localisation_evaluation(config, config.hrtf_selection_dir, file_ext, baseline=True, hrtf_selection='maximum')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-t", "--tag")
    parser.add_argument("-c", "--hpc")
    args = parser.parse_args()

    if args.hpc == "True":
        hpc = True
    elif args.hpc == "False":
        hpc = False
    else:
        raise RuntimeError("Please enter 'True' or 'False' for the hpc tag (-c/--hpc)")

    if args.tag:
        tag = args.tag
    else:
        tag = None

    config = Config(tag, using_hpc=hpc)
    main(config, args.mode)
