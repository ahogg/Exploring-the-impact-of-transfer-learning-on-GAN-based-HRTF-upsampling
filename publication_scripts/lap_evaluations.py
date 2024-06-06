import argparse
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from prettytable import PrettyTable

from spatialaudiometrics import lap_challenge as lap
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

def run_baseline_plots():

    # baselines = ['barycentric', 'sh', 'gan']
    lap_factors = ['100', '19', '5', '3']
    baselines = ['gan']
    # lap_factors = ['5']

    barycentric_errors = []
    sh_errors = []
    gan_errors = []
    for baseline in baselines:
        for lap_factor in lap_factors:

            dataset = 'SONICOM'
            tag = {'tag': f'pub-prep-upscale-{dataset}-LAP-{lap_factor}'}
            config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset, lap_factor=lap_factor)

            if baseline == 'barycentric':
                output_path = config.barycentric_hrtf_dir + '/barycentric_interpolated_data_lap_' + lap_factor
            elif baseline == 'sh':
                output_path = config.valid_lap_merge_dir + '/sh_interpolated_data_lap_' + config.lap_factor
            elif baseline == 'gan':
                output_path = f'{config.data_dirs_path}/runs-pub-fa/pub-prep-upscale-{config.dataset}-LAP-{config.lap_factor}/valid/original_coordinates'

            file_path = output_path + '/sofa_min_phase'

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

    error_types = ['total_itd_error', 'total_ild_error', 'total_lsd_error']
    error_units = ['(µs)', '(dB)', '(dB)']
    x_limits = [(20, 85), (0, 11.5), (2.8, 11)]
    error_thresholds = [62.5, 4.4, 7.4]
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
            fig.savefig(f'{baseline}_{error_type}.png')
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



    run_baseline_plots()

    # run_evaluation(hpc, int(args.exp), args.type, args.test)

