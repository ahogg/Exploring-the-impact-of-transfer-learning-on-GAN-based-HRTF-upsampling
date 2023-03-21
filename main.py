import argparse
import os
import pickle
import torch
import numpy as np
import importlib

from config import Config
from model.train import train
from model.test import test
from model.util import load_dataset
from preprocessing.cubed_sphere import CubedSphere
from preprocessing.utils import interpolate_fft, generate_euclidean_cube, gen_sofa_baseline, \
    load_data, merge_files, gen_sofa_preprocess, get_hrtf_from_ds, clear_create_directories, convert_to_sofa
from preprocessing.synthetic_data import interpolate_synthetic_fft
from model import util
from baselines.barycentric_interpolation import run_barycentric_interpolation
from evaluation.lsd_metric_evaluation import run_lsd_evaluation

PI_4 = np.pi / 4

# Random seed to maintain reproducible results
torch.manual_seed(0)
np.random.seed(0)


def main(config, mode):
    # Initialise Config object
    data_dir = config.raw_hrtf_dir / config.dataset
    print(os.getcwd())
    print(config.dataset)

    imp = importlib.import_module('hrtfdata.torch.full')
    load_function = getattr(imp, config.dataset)

    if mode == 'generate_projection':
        # Must be run in this mode once per dataset, finds barycentric coordinates for each point in the cubed sphere

        if config.dataset == 'SONICOMSynthetic':
            load_function = getattr(imp, 'SONICOM')
            data_dir = config.raw_hrtf_dir / 'SONICOM'

        # No need to load the entire dataset in this case
        ds: load_function = load_data(data_folder=data_dir, load_function=load_function, domain='time', side='left', subject_ids='first')
        # need to use protected member to get this data, no getters
        cs = CubedSphere(sphere_coords=ds._selected_angles)
        generate_euclidean_cube(cs.get_sphere_coords(), config.projection_filename, edge_len=config.hrtf_size)

    elif mode == 'preprocess':
        # Interpolates data to find HRIRs on cubed sphere, then FFT to obtain HRTF, finally splits data into train and
        # val sets and saves processed data

        if config.dataset == 'SONICOMSynthetic':
            load_function = getattr(imp, 'SONICOM')
            data_dir_sonicom = config.raw_hrtf_dir / 'SONICOM'
            ds: load_function = load_data(data_folder=data_dir_sonicom, load_function=load_function, domain='time', side='left', subject_ids='first')
            cs = CubedSphere(sphere_coords=ds._selected_angles)

            load_function = getattr(imp, config.dataset)
            ds: load_function = load_data(data_folder=data_dir, load_function=load_function, domain='time', side='both')
        else:
            ds: load_function = load_data(data_folder=data_dir, load_function=load_function, domain='time', side='both')
            cs = CubedSphere(sphere_coords=ds._selected_angles)

        # need to use protected member to get this data, no getters
        with open(config.projection_filename, "rb") as file:
            cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        # Clear/Create directories
        clear_create_directories(config)

        # Split data into train and test sets
        train_size = int(len(set(ds.subject_ids)) * config.train_samples_ratio)
        train_sample = np.random.choice(list(set(ds.subject_ids)), train_size, replace=False)

        # collect all train_hrtfs to get mean and sd
        train_hrtfs = torch.empty(size=(2 * train_size, 5, config.hrtf_size, config.hrtf_size, 128))
        j = 0
        for i in range(len(ds)):
            if i % 10 == 0:
                print(f"HRTF {i} out of {len(ds)} ({round(100 * i / len(ds))}%)")

            if config.dataset == 'SONICOMSynthetic':
                clean_hrtf = interpolate_synthetic_fft(config, ds[i]['features'], cube, fs_original=ds.hrir_samplerate,
                                                       edge_len=16)
                sphere_original = None
            else:
                clean_hrtf = interpolate_fft(config, cs, ds[i]['features'], sphere, sphere_triangles, sphere_coeffs,
                                             cube, fs_original=ds.hrir_samplerate, edge_len=config.hrtf_size)
                hrtf_original, phase_original, sphere_original = get_hrtf_from_ds(ds, i)

            # save cleaned hrtfdata
            if ds[i]['group'] in train_sample:
                projected_dir = config.train_hrtf_dir
                projected_dir_original = config.train_original_hrtf_dir
                train_hrtfs[j] = clean_hrtf
                j += 1
            else:
                projected_dir = config.valid_hrtf_dir
                projected_dir_original = config.valid_original_hrtf_dir

            subject_id = str(ds[i]['group'])
            side = ds[i]['target']
            with open('%s/%s_mag_%s%s.pickle' % (projected_dir, config.dataset, subject_id, side), "wb") as file:
                pickle.dump(clean_hrtf, file)

            if config.dataset != 'SONICOMSynthetic':
                with open('%s/%s_mag_%s%s.pickle' % (projected_dir_original, config.dataset, subject_id, side), "wb") as file:
                    pickle.dump(hrtf_original, file)

                with open('%s/%s_phase_%s%s.pickle' % (projected_dir_original, config.dataset, subject_id, side), "wb") as file:
                    pickle.dump(phase_original, file)

        if config.merge_flag:
            merge_files(config)

        if config.gen_sofa_flag:
            gen_sofa_preprocess(config, cube, sphere, sphere_original)

        # save dataset mean and standard deviation for each channel, across all HRTFs in the training data
        mean = torch.mean(train_hrtfs, [0, 1, 2, 3])
        std = torch.std(train_hrtfs, [0, 1, 2, 3])
        min_hrtf = torch.min(train_hrtfs)
        max_hrtf = torch.max(train_hrtfs)
        mean_std_filename = config.mean_std_filename
        with open(mean_std_filename, "wb") as file:
            pickle.dump((mean, std, min_hrtf, max_hrtf), file)

    elif mode == 'train':
        # Trains the GANs, according to the parameters specified in Config
        train_prefetcher, test_prefetcher = load_dataset(config, mean=None, std=None)
        print("Loaded all datasets successfully.")

        util.initialise_folders(config, overwrite=True)
        train(config, train_prefetcher)

    elif mode == 'test':
        _, test_prefetcher = load_dataset(config, mean=None, std=None)
        print("Loaded all datasets successfully.")

        test(config, test_prefetcher)

        run_lsd_evaluation(config, config.valid_path)

        with open(config.projection_filename, "rb") as file:
            cube, sphere, sphere_triangles, sphere_coeffs = pickle.load(file)

        if config.gen_sofa_flag:
            convert_to_sofa(config.valid_path, config, cube, sphere)
            print('Created valid sofa files')

    elif mode == 'baseline':
        no_nodes = str(int(5 * (config.hrtf_size / config.upscale_factor) ** 2))
        no_full_nodes = str(int(5 * config.hrtf_size ** 2))

        barycentric_data_folder = '/barycentric_interpolated_data_%s_%s' % (no_nodes, no_full_nodes)
        cube, sphere = run_barycentric_interpolation(config, barycentric_data_folder)

        if config.gen_sofa_flag:
            gen_sofa_baseline(config, barycentric_data_folder, cube, sphere)
            print('Created barycentric baseline sofa files')

        barycentric_output_path = config.barycentric_hrtf_dir + barycentric_data_folder
        run_lsd_evaluation(config, barycentric_output_path)


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
