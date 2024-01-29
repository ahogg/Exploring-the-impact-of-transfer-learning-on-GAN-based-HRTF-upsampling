import argparse
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from prettytable import PrettyTable


from publication_scripts.config_forum_acusticum import Config
from model.test import test
from model.util import load_dataset
from evaluation.evaluation import run_lsd_evaluation, run_localisation_evaluation, run_target_localisation_evaluation
from main import main

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
import re
import numpy as np

import matlab.engine

def run_evaluation(hpc, experiment_id, type, test_id=None):
    print(f'Running {type} experiment {experiment_id}')
    config_files = []
    if experiment_id == 1:
        upscale_factors = [216]
        datasets = ['SONICOM']
        for dataset in datasets:
            for upscale_factor in upscale_factors:
                tags = [{'tag': f'pub-prep-upscale-{dataset}'}]
                for tag in tags:
                    config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset)
                    config.upscale_factor = upscale_factor
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
            file_path = f'{config.data_dirs_path}/xuyi_jian_results/{config.dataset}_{config.upscale_factor}'

            hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(file_path)]

            eng = matlab.engine.start_matlab()
            s = eng.genpath(config.amt_dir)
            eng.addpath(s, nargout=0)
            s = eng.genpath(config.data_dirs_path)
            eng.addpath(s, nargout=0)

            if not os.path.exists(file_path):
                raise Exception(f'File path does not exist or does not have write permissions ({file_path})')

            loc_errors = []
            for file in hrtf_file_names:
                target_sofa_file = config.valid_hrtf_merge_dir + '/sofa_min_phase/' + file
                target_sofa_file = target_sofa_file.replace('single_panel',
                                                            'cube_sphere')  # For single panel use cube sphere

                generated_sofa_file = file_path + '/' + file

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

    run_evaluation(hpc, int(args.exp), args.type, args.test)

