import argparse
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import pickle
import numpy as np
import itertools
import sys, os
from matplotlib.colors import LinearSegmentedColormap
from prettytable import PrettyTable

sys.path.append('/rds/general/user/aos13/home/HRTF-upsampling-with-a-generative-adversarial-network-using-a-gnomonic-equiangular-projection/')

from publication_scripts.config_single_node_upsampling import Config
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

def get_means(full_results):
    upsample_means = []
    upsample_stds = []
    for full_result in full_results:
        upsample_means.insert(0, np.mean(full_result))
        upsample_stds.insert(0, np.std(full_result))

    return upsample_means, upsample_stds


def create_table(legend, full_results, side_title=None, units=None):
    factors = [2, 4, 8, 16]
    single_line = r"\hhline{-~----}" if side_title is None else r"\hhline{~-~----}"
    double_lines = r"\hhline{=~====}" if side_title is None else r"\hhline{~=~====}"
    title_lines = r"\hhline{~~----}" if side_title is None else r"\hhline{~~~----}"
    extra_column_1 = r"" if side_title is None else r" & "
    extra_column_2 = r"" if side_title is None else r"c"

    ticks = [r' \textbf{%s} $\,\rightarrow$ \textbf{1280}' % int((16 / factor) ** 2 * 5) for factor in factors]

    if units is not None:
        units = f' {units.strip()}'
    else:
        units = ''

    print(r"\begin{tabular}{%s|c|c @{\hspace{-0.3\tabcolsep}}|c|c|c|c|}" % extra_column_2)
    print(single_line)
    print(extra_column_1 +
        r"\multirow{2}{*}{\textbf{Method}} & & \multicolumn{4}{c|}{\textbf{Upsample Factor (No. original  $\,\rightarrow$ upsampled)"+units+"}} \\ " + title_lines)
    print(extra_column_1 + r"& & \multicolumn{1}{c|}{" + ticks[0] + r"} & \multicolumn{1}{c|}{" + ticks[
        1] + r"} & \multicolumn{1}{c|}{" + ticks[2] + r"} & \multicolumn{1}{c|}{" + ticks[3] + r"} \\ " + double_lines)
    if side_title is not None:
        print(r"\parbox[t]{3.5mm}{\multirow{%s}{*}{\rotatebox[origin=c]{90}{\textbf{%s}}}}" % (len(full_results), side_title))
    for idx, full_result in enumerate(full_results):
        if full_result is not None:
            full_result_means, full_result_stds = get_means(full_result)
            if len(full_result_means) == 4:
                print(extra_column_1 +
                    r"\textbf{%s} & & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) & %.2f (%.2f) \\ %s" % tuple(
                        [legend[idx]] + [val for pair in zip(full_result_means, full_result_stds) for val in pair] + [single_line]))
            if len(full_result_means) == 1:
                print(extra_column_1 + r"\textbf{%s} & & \multicolumn{4}{c|}{%.2f (%.2f)} \\ %s" % tuple(
                        [legend[idx]] + [val for pair in zip(full_result_means, full_result_stds) for val in pair] + [single_line]))

    print(r"\end{tabular}")
    print('\n')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color, linewidth=0.5)
    plt.setp(bp['whiskers'], color=color, linewidth=0.5)
    plt.setp(bp['caps'], color=color, linewidth=0.5)
    plt.setp(bp['medians'], color=color, linewidth=0.5)


def plot_boxplot(config, name, ylabel, full_results, legend, colours, ticks, xlabel=None, hrtf_selection_results=None):
    plt.rc('font', family='serif', serif='Times New Roman')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    fig, ax = plt.subplots()

    full_results = list(filter(lambda item: item is not None, full_results))

    for idx, full_result in enumerate(full_results):
        # Append nans to results to make them of equal length
        maxlen = np.max([len(i) for i in full_result])
        for result in full_result:
            if (maxlen - len(result)) > 0:
                result[:] = [np.NaN] * (maxlen - len(result)) + result

        data = np.vstack(full_result)

        for i, d in enumerate(data):
            filtered_data = d[~np.isnan(d)]
            blp = plt.boxplot(filtered_data, positions=[(i * 1.0) - np.linspace(0.15 * (len(full_results) / 2), -0.15 * (len(full_results) / 2), len(full_results))[idx]],
                              flierprops=dict(marker='x', markeredgecolor=colours[idx], markersize=4), widths=0.12)

            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(blp[element], color=colours[idx], linewidth=0.7)

        plt.plot([], c=colours[idx], label=legend[idx])

    if hrtf_selection_results is not None:
        ticks += ['Sel']
        c = ['#00FFFF', '#FFC300']
        l = ['Selection-1', 'Selection-2']
        for idx, hrtf_selection_result in enumerate(hrtf_selection_results):
            blp = plt.boxplot(hrtf_selection_result, positions=[(len(data) * 1.0) - np.linspace(0.15 * (len(full_results) / 2),
                                                                                        -0.15 * (len(full_results) / 2),
                                                                                        len(full_results))[idx+1]],
                              flierprops=dict(marker='x', markeredgecolor=c[idx], markersize=4), widths=0.12)
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(blp[element], color=c[idx], linewidth=0.7)

            plt.plot([], c=c[idx], label=l[idx])

    if len(full_results) > 2:
        if hrtf_selection_results is not None:
            plt.axvline(range(0, (len(ticks) - 1), 1)[-1] + 0.5, color='#a6a6a6', linewidth=1)
            [plt.axvline(x + 0.5, color='#a6a6a6', linestyle='--', linewidth=0.5) for x in range(0, (len(ticks) - 2), 1)]
        else:
            [plt.axvline(x + 0.5, color='#a6a6a6', linestyle='--', linewidth=0.5) for x in range(0, (len(ticks) - 1), 1)]
    else:
        if hrtf_selection_results is not None:
            plt.axvline(range(0, (len(ticks) - 1) * 2, 2)[-1] + 1, color='#a6a6a6', linewidth=1)
            [plt.axvline(x + 1, color='#a6a6a6', linestyle='--', linewidth=0.5) for x in range(0, (len(ticks) - 2) * 2, 2)]
        else:
            [plt.axvline(x + 1, color='#a6a6a6', linestyle='--', linewidth=0.5) for x in range(0, (len(ticks) - 1) * 2, 2)]

    leg = ax.legend(prop={'size': 7}, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower right",  # mode="expand",
                        borderaxespad=0, ncol=2, handlelength=1.06)

    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor('k')

    ax.yaxis.grid(zorder=0, linewidth=0.4)

    if xlabel == None:
        plt.xlabel(
            'Upsample Factor\n' + r'(No. of original nodes$\ {\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}}$ No. of upsampled nodes)')
    else:
        plt.xlabel(xlabel)

    plt.ylabel(ylabel)
    if len(full_results) > 2:
        plt.xticks(range(0, len(ticks), 1), ticks)
        plt.xlim(-0.5, len(ticks) - 0.5)
    else:
        plt.xticks(range(0, len(ticks) * 2, 2), ticks)

    ymin = np.nanmin(full_results) - 0.1 * abs(np.nanmax(full_results) - np.nanmin(full_results))
    ymax = np.nanmax(full_results) + 0.1 * abs(np.nanmax(full_results) - np.nanmin(full_results))
    ax.set_ylim((ymin, ymax))
    ax.yaxis.set_label_coords(-0.12, 0.5)

    # w = 2.974
    # h = w / 1.5
    w = 5
    h = w / 2.5
    fig.set_size_inches(w, h)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    fig.savefig(config.data_dirs_path + '/plots/' + name, bbox_inches='tight')


def plot_lsd_plot(config, full_lsd_plot_results):
    plt.rc('font', family='serif', serif='Times New Roman')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    subject_id = 0

    max_node_lsd = max(filter(None.__ne__, np.array(full_lsd_plot_results)[:,:,:,-1].flatten()))
    for upsampling_idx in [0, 1, 2, 3]:

        fig, ax = plt.subplots(1, 1, sharey=False)

        coordinates = [coordinate for coordinate in full_lsd_plot_results[upsampling_idx][subject_id] if coordinate[2] is not None]
        coordinates_original = [coordinate for coordinate in full_lsd_plot_results[upsampling_idx][subject_id] if coordinate[2] is None]
        plt.scatter([coordinate_original[0] for coordinate_original in coordinates_original], [coordinate_original[1] for coordinate_original in coordinates_original], s=5, facecolors='none', edgecolors='k', linewidth=0.5)
        plt.scatter([coordinate[0] for coordinate in coordinates], [coordinate[1] for coordinate in coordinates], c=[coordinate[2] for coordinate in coordinates], cmap=parula, s=5)

        cbar = plt.colorbar(pad=0.01)
        cbar.set_label('Average SD error [dB]', rotation=270, labelpad=14)
        #     plt.title("SD per node -- HR vs SR "+"avg: "+str(round(sum(diff_left)/1280,5)))
        plt.clim(0, max_node_lsd)
        plt.xlabel(r"Azimuth [$^\circ$]")
        plt.ylabel(r"Elevation [$^\circ$]")
        #     plt.text(0.5, 0.5, 'Average SD error', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        width = 2.874
        height = width / 2.5
        fig.set_size_inches(width * 1.5, height * 1.5)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        fig.savefig(f'{config.data_dirs_path}/plots/SD_node_{upsampling_idx}.pdf', dpi=300)
    return


def get_results(tag, mode, upscale_factors=[16, 8, 4, 2], file_ext=None, runs_folder=None):
    full_results = []
    full_plot_results = []
    for upscale_factor in upscale_factors:
        config = Config(tag + str(upscale_factor), using_hpc=hpc, runs_folder=runs_folder)
        if mode == 'lsd' or mode == 'baseline_lsd':
            if mode == 'lsd':
                file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext
                file_path = f'{config.path}/{file_ext}'
            elif mode == 'baseline_lsd':
                file_path = f'{tag}/{file_ext}{upscale_factor}.pickle'

            try:
                with open(file_path, 'rb') as file:
                    lsd_id_errors = pickle.load(file)
            except OSError:
                print(f"Unable to load {file_path} successfully.")
                return None, None

            total_lsd_errors = [lsd_error['total_error'] if not np.isinf(lsd_error['total_error']) else np.nan for lsd_error in lsd_id_errors]

            errors = [lsd_error['errors'] if not np.isinf(lsd_error['errors']).any() else np.nan for lsd_error in lsd_id_errors]
            coordinates = [[lsd_errors for lsd_errors in subject_lsd_errors['coordinates']] for subject_lsd_errors in lsd_id_errors]
            lsd_plot_results = []
            for sub_idx, subject in enumerate(coordinates):
                error_idx = 0
                lsd_plot_result = []
                for pos_idx, position in enumerate(subject):
                    if position['original']:
                        lsd_plot_result.append((position['x'], position['y'], None))
                    else:
                        if 'x' in position and 'y' in position:
                            lsd_plot_result.append((position['x'], position['y'], errors[sub_idx][error_idx]))
                        else:
                            lsd_plot_result.append((position['p'], position['h'], position['w'], errors[sub_idx][error_idx]))
                        error_idx += 1
                lsd_plot_results.append(lsd_plot_result)

            print(f'Loading: {file_path}')
            print('Mean (STD) LSD: %0.3f (%0.3f)' % (np.mean(total_lsd_errors),  np.std(total_lsd_errors)))
            full_results.append(total_lsd_errors)
            full_plot_results.append(lsd_plot_results)
        elif mode == 'loc' or mode == 'target' or mode == 'baseline_loc':
            file_ext = 'loc_errors.pickle' if file_ext is None else file_ext
            if mode == 'loc':
                file_path = f'{config.path}/{file_ext}'
            elif mode == 'target':
                file_path = tag + '/' + file_ext
            elif mode == 'baseline_loc':
                file_path = f'{tag}/{file_ext}{upscale_factor}.pickle'

            with open(file_path, 'rb') as file:
                loc_id_errors = pickle.load(file)
            pol_acc1 = [loc_error[1] for loc_error in loc_id_errors]
            pol_rms1 = [loc_error[2] for loc_error in loc_id_errors]
            querr1 = [loc_error[3] for loc_error in loc_id_errors]
            print(f'Loading: {file_path}')
            print('Mean (STD) ACC Error: %0.3f (%0.3f)' % (np.mean(pol_acc1), np.std(pol_acc1)))
            print('Mean (STD) RMS Error: %0.3f (%0.3f)' % (np.mean(pol_rms1), np.std(pol_rms1)))
            print('Mean (STD) QUERR Error: %0.3f (%0.3f)' % (np.mean(querr1), np.std(querr1)))
            full_results.append([pol_acc1, pol_rms1, querr1])

            if mode == 'target':
                break

    return full_results, full_plot_results

def run_projection(hpc, dataset_id=None):
    print(f'Running projection')
    config_files = []
    datasets = ['ARI', 'SONICOM', 'SONICOMSynthetic']
    for dataset in datasets:
        config = Config(tag=None, using_hpc=hpc, dataset=dataset)
        config.hrtf_size = 16
        config_files.append(config)

    print(f'{len(config_files)} config files created successfully.')
    if dataset_id is not None:
        if dataset_id.isnumeric():
            test_id = int(dataset_id)
            config_files = [config_files[test_id]]
        else:
            for config in config_files:
                if config.dataset == dataset_id:
                    config_files = [config]
                    break

    print(f'Running a total of {len(config_files)} config files')
    for config in config_files:
        main(config, 'generate_projection')

def run_preprocess(hpc, type, dataset_id=None):
    print(f'Running projection')
    config_files = []
    datasets = ['ARI', 'SONICOM', 'SONICOMSynthetic']
    for dataset in datasets:
        if type == 'base':
            config = Config(tag=None, using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset)
            config.train_samples_ratio = 0.8
        elif type == 'tl':
            config = Config(tag=None, using_hpc=hpc, dataset=dataset, data_dir='/data-transfer-learning/' + dataset)
            config.train_samples_ratio = 1.0
        config.hrtf_size = 16
        config_files.append(config)

    print(f'{len(config_files)} config files created successfully.')
    if dataset_id is not None:
        if dataset_id.isnumeric():
            test_id = int(dataset_id)
            config_files = [config_files[test_id]]
        else:
            for config in config_files:
                if config.dataset == dataset_id:
                    config_files = [config]
                    break

    print(f'Running a total of {len(config_files)} config files')
    for config in config_files:
        main(config, 'preprocess')

def run_train(hpc, type, test_id=None, tuning=None):
    print(f'Running training')
    config_files = []
    # upscale_factors = [2, 4, 8, 16, 40]
    upscale_factors = [2, 4, 8, 16]
    double_panels = []
    # double_panels = [[0, 2], [1, 3], [0, 1], [2, 3]]
    datasets = ['ARI', 'SONICOM', 'SONICOMSynthetic']
    if type == 'tl' or type == 'base':
        datasets.remove('SONICOMSynthetic')
    for dataset in datasets:
        other_dataset = 'ARI' if dataset == 'SONICOM' else 'SONICOM'
        for upscale_factor in upscale_factors:
            tags = []
            if type == 'base':
                if upscale_factor == 80:
                    for panel in [0, 1, 2, 3, 4]:
                        tags.append({'tag': f'pub-prep-upscale-{dataset}-{upscale_factor}-{panel}'})
                elif upscale_factor == 40:
                    for panel in double_panels:
                        tags.append({'tag': f'pub-prep-upscale-{dataset}-{upscale_factor}-{panel[0]}-{panel[1]}'})
                else:
                    tags = [{'tag': f'pub-prep-upscale-{dataset}-{upscale_factor}'}]
            elif type == 'base-tl':
                if upscale_factor == 80:
                    for panel in [0, 1, 2, 3, 4]:
                        tags.append({'tag': f'pub-prep-upscale-{dataset}-tl-{upscale_factor}-{panel}'})
                elif upscale_factor == 40:
                    for panel in double_panels:
                        tags.append({'tag': f'pub-prep-upscale-{dataset}-tl-{upscale_factor}-{panel[0]}-{panel[1]}'})
                else:
                    tags = [{'tag': f'pub-prep-upscale-{dataset}-tl-{upscale_factor}'}]
            elif type == 'tl':
                if upscale_factor == 40:
                    for panel in double_panels:
                        tags.append({'tag': f'pub-prep-upscale-{dataset}-{other_dataset}-tl-{upscale_factor}-{panel[0]}-{panel[1]}',
                                     'existing_model_tag': f'pub-prep-upscale-{other_dataset}-tl-{upscale_factor}-{panel[0]}-{panel[1]}'})
                        tags.append({'tag': f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-{upscale_factor}-{panel[0]}-{panel[1]}',
                                     'existing_model_tag': f'pub-prep-upscale-SONICOMSynthetic-tl-{upscale_factor}-{panel[0]}-{panel[1]}'})
                elif upscale_factor == 80:
                    for panel in [0, 1, 2, 3, 4]:
                        tags.append({'tag': f'pub-prep-upscale-{dataset}-{other_dataset}-tl-{upscale_factor}-{panel}',
                                     'existing_model_tag': f'pub-prep-upscale-{other_dataset}-tl-{upscale_factor}-{panel}'})
                        tags.append({'tag': f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-{upscale_factor}-{panel}',
                                     'existing_model_tag': f'pub-prep-upscale-SONICOMSynthetic-tl-{upscale_factor}-{panel}'})
                else:
                    tags = [{'tag': f'pub-prep-upscale-{dataset}-{other_dataset}-tl-{upscale_factor}', 'existing_model_tag': f'pub-prep-upscale-{other_dataset}-tl-{upscale_factor}'},
                            {'tag': f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-{upscale_factor}', 'existing_model_tag': f'pub-prep-upscale-SONICOMSynthetic-tl-{upscale_factor}'}]
            else:
                print("Type not valid. Please use 'base' or 'tl'")

            for tag in tags:

                if upscale_factor == 80:
                    runs_folder = '/runs-hpc-single-node'
                elif upscale_factor == 40:
                    runs_folder = '/runs-hpc-double-node'
                else:
                    runs_folder ='/runs-hpc'


                if tuning == True:
                    content_weight_grid_search = [0.1, 0.01, 0.001]
                    adversarial_weight_grid_search = [0.1, 0.01, 0.001]
                    lr_gen_grid_search = [0.0002, 0.0004, 0.0006, 0.0008]
                    lr_dis_grid_search = [0.0000015]
                    grid_search = list(itertools.product(content_weight_grid_search, adversarial_weight_grid_search, lr_gen_grid_search, lr_dis_grid_search))
                    for search_index, hyperparameters in enumerate(grid_search):
                        label = tag['tag'] + f'-search-{search_index}'
                        temporary_runs_path = '/rds/general/ephemeral/project/sonicom/ephemeral/tuning_GAN'
                        if type == 'base':
                            config = Config(label, using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset, runs_folder=runs_folder, temporary_runs_path=temporary_runs_path)
                        elif type == 'base-tl':
                            config = Config(label, using_hpc=hpc, dataset=dataset, data_dir='/data-transfer-learning/' + dataset, runs_folder=runs_folder, temporary_runs_path=temporary_runs_path)
                        elif type == 'tl':
                            existing_model_label = tag['existing_model_tag'] + f'-search-{search_index}'
                            config = Config(label, using_hpc=hpc, dataset=dataset, existing_model_tag=existing_model_label, data_dir='/data/' + dataset, runs_folder=runs_folder, temporary_runs_path=temporary_runs_path)

                        config.upscale_factor = upscale_factor
                        config.content_weight = hyperparameters[0]
                        config.adversarial_weight = hyperparameters[1]
                        config.lr_gen = hyperparameters[2]
                        config.lr_dis = hyperparameters[3]
                        config_files.append(config)
                else:
                    if type == 'base':
                        config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset, runs_folder=runs_folder)
                    elif type == 'base-tl':
                        config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data-transfer-learning/' + dataset, runs_folder=runs_folder)
                    elif type == 'tl':
                        config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, existing_model_tag=tag['existing_model_tag'], data_dir='/data/' + dataset, runs_folder=runs_folder)
                    config.upscale_factor = upscale_factor
                    config.lr_gen = 0.0002
                    config.lr_dis = 0.0000015

                    if upscale_factor == 2:
                        config.content_weight = 0.1
                        config.adversarial_weight = 0.001
                    elif upscale_factor == 4:
                        config.content_weight = 0.01
                        config.adversarial_weight = 0.1
                    elif upscale_factor == 8:
                        config.content_weight = 0.001
                        config.adversarial_weight = 0.001
                    elif upscale_factor == 16:
                        config.content_weight = 0.01
                        config.adversarial_weight = 0.01
                    elif upscale_factor == 40:
                        config.content_weight = 0.01
                        config.adversarial_weight = 0.01
                        config.panel = [int(config.tag[-3]), int(config.tag[-1])]
                    elif upscale_factor == 80:
                        config.content_weight = 0.01
                        config.adversarial_weight = 0.01
                        config.panel = int(config.tag[-1])

                    config_files.append(config)

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
            if len(config_files) != 1:
                print(f'{test_id} not found... running all config files')

    print(f'Running a total of {len(config_files)} config files')
    for config in config_files:
        main(config, 'train')


def get_tuning_results(hpc, test_id=None):
    if hpc:
        temporary_runs_path = '/rds/general/ephemeral/project/sonicom/ephemeral/tuning_GAN/' + test_id
    else:
        temporary_runs_path = '/home/ahogg/PycharmProjects/HRTF-GAN/tuning_results/' + test_id

    run_paths = sorted([temporary_runs_path+'/'+name for name in os.listdir(temporary_runs_path) if os.path.isfile(f'{temporary_runs_path}/{name}/train_losses.pickle')])

    results = []
    for run_path in run_paths:

        content_weight_grid_search = [0.1, 0.01, 0.001]
        adversarial_weight_grid_search = [0.1, 0.01, 0.001]
        lr_gen_grid_search = [0.0002, 0.0004, 0.0006, 0.0008]
        lr_dis_grid_search = [0.0000015]
        grid_search = list(itertools.product(content_weight_grid_search, adversarial_weight_grid_search, lr_gen_grid_search,lr_dis_grid_search))

        with (open(f'{run_path}/train_losses.pickle', "rb")) as run_file:
            while True:
                try:
                    (train_losses_G, train_losses_G_adversarial, train_losses_G_content,
                     train_losses_D, train_losses_D_hr, train_losses_D_sr, train_SD_metric) = pickle.load(run_file)
                    loss = np.mean(train_SD_metric[-20:-1])
                    search_name = os.path.basename(run_path)
                    dataset = search_name.split('-')[-4]
                    upscale_factor = int(search_name.split('-')[-3])
                    search_idx = int(search_name.split('-')[-1])
                    content_weight = grid_search[search_idx][0]
                    adversarial_weight = grid_search[search_idx][1]
                    lr_gen = grid_search[search_idx][2]
                    lr_dis = grid_search[search_idx][3]

                    results.append([search_name, dataset, upscale_factor, content_weight, adversarial_weight, lr_gen, lr_dis, loss])
                except EOFError:
                    break

    datasets = ['ARI', 'SONICOM']
    upscale_factors = [2, 4, 8, 16]
    for dataset in datasets:
        table = PrettyTable(['File', 'Dataset', 'Factor', 'Content', 'Adversarial', 'LR Generator', 'LR Discriminator', 'Loss'])
        best_table = PrettyTable(['File', 'Dataset', 'Factor', 'Content', 'Adversarial', 'LR Generator', 'LR Discriminator', 'Loss'])
        for upscale_factor in upscale_factors:
            best_result = [np.inf]
            for result in results:
                if result[1] == dataset and result[2] == upscale_factor:
                    table.add_row(result)
                    if result[-1] < best_result[-1]:
                        best_result = result
            if best_result[-1] != np.inf:
                best_table.add_row(best_result)
        print(f'Get all tuning results for {dataset}:')
        print(table)
        print(f'Get best tuning results for {dataset}:')
        print(best_table)
    return

def run_evaluation(hpc, experiment_id, type, test_id=None):
    print(f'Running {type} experiment {experiment_id}')
    config_files = []
    if experiment_id == 1:
        upscale_factors = [2, 4, 8, 16]
        datasets = ['ARI', 'SONICOM']
        for dataset in datasets:
            for upscale_factor in upscale_factors:
                tags = [{'tag': f'pub-prep-upscale-{dataset}-{upscale_factor}'}]
                for tag in tags:
                    config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset)
                    config.upscale_factor = upscale_factor
                    config_files.append(config)
    elif experiment_id == 2:
        upscale_factors = [2, 4, 8, 16]
        datasets = ['ARI', 'SONICOM']
        for dataset in datasets:
            other_dataset = 'ARI' if dataset == 'SONICOM' else 'SONICOM'
            for upscale_factor in upscale_factors:
                tags = [{'tag': f'pub-prep-upscale-{dataset}-{upscale_factor}'},
                        {'tag': f'pub-prep-upscale-{dataset}-{other_dataset}-tl-{upscale_factor}'},
                        {'tag': f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-{upscale_factor}'}]
                for tag in tags:
                    config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset)
                    config.upscale_factor = upscale_factor
                    config_files.append(config)
    elif experiment_id == 3:
        datasets = ['ARI', 'SONICOM']
        for dataset in datasets:
            tag = None
            config = Config(tag, using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset)
            config_files.append(config)
    elif experiment_id == 4:
        # upscale_factors = [2, 4, 8, 16, 40, 80]
        upscale_factors = [2, 4, 8, 16]
        panels = [0, 1, 2, 3, 4]
        double_panels = [[0, 2], [1, 3], [0, 1], [2, 3]]
        datasets = ['ARI', 'SONICOM']
        for dataset in datasets:
            other_dataset = 'ARI' if dataset == 'SONICOM' else 'SONICOM'
            for upscale_factor in upscale_factors:
                if upscale_factor == 80:
                    runs_folder = '/runs-hpc-single-node'
                    for panel in panels:
                        tags = [
                                {'tag': f'pub-prep-upscale-{dataset}-{upscale_factor}-{panel}'},
                                {'tag': f'pub-prep-upscale-{dataset}-{other_dataset}-tl-{upscale_factor}-{panel}'},
                                {'tag': f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-{upscale_factor}-{panel}'}
                        ]
                        for tag in tags:
                            config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset,
                                            runs_folder=runs_folder)
                            config.upscale_factor = upscale_factor
                            config.panel = panel
                            config_files.append(config)
                elif upscale_factor == 40:
                    runs_folder = '/runs-hpc-double-node'
                    for panel in double_panels:
                        tags = [
                            {'tag': f'pub-prep-upscale-{dataset}-{upscale_factor}-{panel[0]}-{panel[1]}'},
                            {'tag': f'pub-prep-upscale-{dataset}-{other_dataset}-tl-{upscale_factor}-{panel[0]}-{panel[1]}'},
                            {'tag': f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-{upscale_factor}-{panel[0]}-{panel[1]}'}
                        ]
                        for tag in tags:
                            config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset,
                                            runs_folder=runs_folder)
                            config.panel = panel
                            config.upscale_factor = upscale_factor
                            config_files.append(config)
                else:
                    runs_folder = '/runs-hpc'
                    tags = [{'tag': f'pub-prep-upscale-{dataset}-{upscale_factor}'},
                            {'tag': f'pub-prep-upscale-{dataset}-{other_dataset}-tl-{upscale_factor}'},
                            {'tag': f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-{upscale_factor}'}]
                    for tag in tags:
                        config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset,
                                        runs_folder=runs_folder)
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
        if experiment_id == 3:
            run_target_localisation_evaluation(config)
        elif type == 'lsd':
            _, test_prefetcher = load_dataset(config, mean=None, std=None)
            print("Loaded all datasets successfully.")
            test(config, test_prefetcher)
            run_lsd_evaluation(config, config.valid_path)
        elif type == 'loc':
            _, test_prefetcher = load_dataset(config, mean=None, std=None)
            print("Loaded all datasets successfully.")
            test(config, test_prefetcher)
            run_localisation_evaluation(config, config.valid_path)
        else:
            print(f'Type ({type}) does not exist')

def plot_evaluation(hpc, experiment_id, mode):
    tag = None
    config = Config(tag, using_hpc=hpc)

    if experiment_id == 1:
        datasets = ['ARI', 'SONICOM']
        for dataset in datasets:
            if mode == 'lsd':
                full_results_LSD_dataset, _ = get_results(f'pub-prep-upscale-{dataset}-', mode)
                full_results_LSD_dataset_sonicom_synthetic_tl, _ = get_results(f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-', mode)
                legend = ['SRGAN', 'SRGAN TL (Synthetic)']
                colours = ['#0047a4', '#af211a', 'g', '#6C0BA9', '#E67E22']
                plot_boxplot(config, f'LSD_boxplot_ex_{experiment_id}_{dataset}', f'{dataset.upper()} LSD error [dB]', [full_results_LSD_dataset, full_results_LSD_dataset_sonicom_synthetic_tl], legend, colours)
            elif mode == 'loc':
                full_results_loc_dataset, _ = get_results(f'pub-prep-upscale-{dataset}-', mode)
                full_results_loc_dataset_sonicom_synthetic_tl, _ = get_results(f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-', mode)
                types = ['ACC', 'RMS', 'QUERR']
                labels = [r'Polar accuracy error [$^\circ$]', r'Polar RMS error [$^\circ$]', 'Quadrant error [\%]']
                for i in np.arange(np.shape(full_results_loc_dataset)[1]):
                    legend = ['SRGAN', 'SRGAN TL (Synthetic)']
                    colours = ['#0047a4', '#af211a', 'g', '#6C0BA9', '#E67E22']
                    plot_boxplot(config, f'{types[i]}_boxplot_ex_{experiment_id}_{dataset}', labels[i], [np.array(full_results_loc_dataset)[:, i, :],
                                np.array(full_results_loc_dataset_sonicom_synthetic_tl)[:, i, :]], legend, colours)

    elif experiment_id == 2:
        datasets = ['ARI', 'SONICOM']
        # datasets = ['ARI']
        for dataset in datasets:
            other_dataset = 'ARI' if dataset == 'SONICOM' else 'SONICOM'
            factors = [2, 4, 8, 16]
            ticks = [
                r'$%s \,{\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}} %s$' % (
                    int((16 / factor) ** 2 * 5), int(config.hrtf_size ** 2 * 5)) for factor in factors]
            if mode == 'lsd':
                full_results_dataset, full_lsd_plot_results_dataset = get_results(f'pub-prep-upscale-{dataset}-', mode)
                # full_results_dataset_sonicom_synthetic_tl, full_lsd_plot_results_sonicom_synthetic_tl = get_results(f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-', mode)
                # full_results_dataset_dataset_tl, full_lsd_plot_results_dataset_tl = get_results(f'pub-prep-upscale-{dataset}-{other_dataset}-tl-', mode)
                full_results_dataset_baseline, full_lsd_plot_results_baseline = get_results(f'{config.data_dirs_path}/baseline_results/{dataset.upper()}/single_panel/barycentric/valid',mode=f'baseline_{mode}', file_ext=f'{mode}_errors_barycentric_interpolated_data_')
                full_results_dataset_baseline_hrtf_selection, _ = get_results(
                    f'{config.data_dirs_path}/baseline_results/{dataset.upper()}/single_panel/hrtf_selection/valid',
                    mode=f'baseline_{mode}', upscale_factors=['minimum_data', 'maximum_data'],
                    file_ext=f'{mode}_errors_hrtf_selection_')
                legend = ['SRGAN', 'Barycentric', 'HRTF Selection-1', 'HRTF Selection-2']
                colours = ['#0047a4', '#af211a', 'g', '#6C0BA9', '#E67E22']
                # remove baseline results at upscale-16
                # full_results_dataset_baseline[0] = np.full(shape=len(full_results_dataset_baseline[-1]), fill_value=np.nan).tolist()
                #######################################
                plot_lsd_plot(config, full_lsd_plot_results_dataset)
                create_table(legend, [full_results_dataset, full_results_dataset_baseline, [full_results_dataset_baseline_hrtf_selection[0]], [full_results_dataset_baseline_hrtf_selection[1]]], dataset.upper(), units='[dB]')
                # plot_boxplot(config, f'LSD_boxplot_ex_{experiment_id}_{dataset}', f'{dataset.upper()} \n LSD error [dB]', [full_results_dataset, full_results_dataset_baseline], legend, colours, ticks, hrtf_selection_results=full_results_dataset_baseline_hrtf_selection)
            elif mode == 'loc':
                types = ['ACC', 'RMS', 'QUERR']
                labels = [r'Polar ACC error [$^\circ$]', r'Polar RMS error [$^\circ$]', 'Quadrant error [\%]']
                labels = [f'{dataset.upper()} \n' + label for label in labels]
                units = [r'[$^\circ$]', r'[$^\circ$]', '[\%]']
                legend = ['SRGAN', 'Baseline', 'Target']
                colours = ['#0047a4', '#af211a', 'g', '#6C0BA9', '#E67E22']
                # remove baseline results at upscale-16
                # full_results_dataset_baseline[0] = np.full(shape=(np.shape(full_results_dataset_baseline[-1])), fill_value=np.nan).tolist()
                #######################################
                full_results_dataset = get_results(f'pub-prep-upscale-{dataset}-', mode)[0]
                # full_results_dataset_sonicom_synthetic_tl = get_results(f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-', mode)[0]
                # full_results_dataset_dataset_tl = get_results(f'pub-prep-upscale-{dataset}-{other_dataset}-tl-', mode)[0]
                full_results_dataset_baseline = get_results(f'{config.data_dirs_path}/baseline_results/{dataset.upper()}/cube_sphere/barycentric/valid',mode=f'baseline_{mode}', file_ext=f'{mode}_errors_barycentric_interpolated_data_')[0]

                full_results_dataset_target_tl = get_results(f'{config.data_dirs_path}/data/{dataset.upper()}/cube_sphere', 'target', file_ext=f'{dataset.upper()}_loc_target_valid_errors.pickle')[::2][0]*4
                for i in np.arange(np.shape(full_results_dataset)[1]):
                    plot_boxplot(config, f'{types[i]}_boxplot_ex_{experiment_id}_{dataset}', labels[i], [np.array(full_results_dataset)[:, i, :],
                                                                                                         np.array(full_results_dataset_baseline)[:, i, :], np.array(full_results_dataset_target_tl)[:, i, :]], legend, colours, ticks, hrtf_selection_results=full_results_dataset_baseline_hrtf_selection)
                    print(f'Generate table containing {types[i]} errors for the {dataset.upper()} dataset: \n')
                    create_table(legend, [np.array(full_results_dataset)[:, i, :], np.array(full_results_dataset_baseline)[:, i, :], [np.array(full_results_dataset_target_tl)[0, i, :]]], dataset.upper(), units=units[i])
    elif experiment_id == 4:
        datasets = ['ARI', 'SONICOM']
        for dataset in datasets:
            other_dataset = 'ARI' if dataset == 'SONICOM' else 'SONICOM'
            full_results_dataset_single_node, _ = get_results(f'pub-prep-upscale-{dataset}-80-', mode, upscale_factors=[0, 1, 2, 3, 4], runs_folder='/runs-hpc-single-node')
            full_results_dataset_double_node, _ = get_results(f'pub-prep-upscale-{dataset}-40-', mode, upscale_factors=['0-2', '1-3', '0-1', '2-3'], runs_folder='/runs-hpc-double-node')
            full_results_dataset_sonicom_synthetic_tl_single_node, _ = get_results(f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-80-', mode,  upscale_factors=[0, 1, 2, 3, 4], runs_folder='/runs-hpc-single-node')
            full_results_dataset_sonicom_synthetic_tl_double_node, _ = get_results(f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-40-', mode, upscale_factors=['0-2', '1-3', '0-1', '2-3'], runs_folder='/runs-hpc-double-node')
            full_results_dataset_dataset_tl_single_node, _ = get_results(f'pub-prep-upscale-{dataset}-{other_dataset}-tl-80-', mode, upscale_factors=[0, 1, 2, 3, 4], runs_folder='/runs-hpc-single-node')
            full_results_dataset_dataset_tl_double_node, _ = get_results(f'pub-prep-upscale-{dataset}-{other_dataset}-tl-40-', mode, upscale_factors=['0-2', '1-3', '0-1', '2-3'], runs_folder='/runs-hpc-double-node')
            full_results_dataset, _ = get_results(f'pub-prep-upscale-{dataset}-', mode, upscale_factors=[2, 4, 8, 16], runs_folder='/runs-hpc')
            full_results_dataset_sonicom_synthetic_tl, _ = get_results(f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-',
                                                                    mode, upscale_factors=[2, 4, 8, 16], runs_folder='/runs-hpc')
            full_results_dataset_dataset_tl, _ = get_results(f'pub-prep-upscale-{dataset}-{other_dataset}-tl-', mode, upscale_factors=[2, 4, 8, 16], runs_folder='/runs-hpc')
            full_results_dataset_baseline, _ = get_results(
                f'{config.data_dirs_path}/baseline_results/{dataset.upper()}/barycentric/valid',
                mode=f'baseline_{mode}', upscale_factors=[2, 4, 8, 16], file_ext=f'{mode}_errors_barycentric_interpolated_data_')
            full_results_dataset_baseline_hrtf_selection, _ = get_results(
                f'{config.data_dirs_path}/baseline_results/{dataset.upper()}/hrtf_selection/valid',
                mode=f'baseline_{mode}', upscale_factors=['minimum_data', 'maximum_data'],
                file_ext=f'{mode}_errors_hrtf_selection_')
            factors = [2, 4, 8, 16]
            basline_ticks = [
                r'$%s $' % (
                    int((16 / factor) ** 2 * 5)) for factor in factors]
            double_panels = [[0, 2], [1, 3], [0, 1], [2, 3]]
            double_ticks = [
                r'$%s(%s,%s)$' % (
                    int(2), panel[0], panel[1]) for panel in double_panels]
            panels = [0, 1, 2, 3, 4]
            ticks = basline_ticks + double_ticks + [
                r'$%s(%s)$' % (
                    int(1), panel) for panel in panels]
            xlabel = 'Upsample Factor\n' + r'No. of original nodes (Panel No.)'
            if mode == 'lsd':
                legend = ['SRGAN', 'TL (Synthetic)', f'TL ({other_dataset})', 'Baseline']
                colours = ['#0047a4', '#af211a', 'g', '#6C0BA9', '#E67E22']
                # remove baseline results at upscale-16
                # full_results_dataset_baseline[0] = np.full(shape=len(full_results_dataset_baseline[-1]), fill_value=np.nan).tolist()
                #######################################
                pad_no = 9
                full_results_dataset_baseline = np.concatenate((full_results_dataset_baseline, [np.full(shape=len(full_results_dataset_baseline[-1]), fill_value=np.nan).tolist()]*pad_no))
                # full_results_dataset_sonicom_synthetic_tl = np.concatenate((full_results_dataset_sonicom_synthetic_tl, full_results_dataset_sonicom_synthetic_tl_single_node))
                # full_results_dataset_dataset_tl = np.concatenate((full_results_dataset_dataset_tl, full_results_dataset_dataset_tl_single_node))

                # full_results_dataset_sonicom_synthetic_tl = np.concatenate((full_results_dataset_sonicom_synthetic_tl, [np.full(shape=len(full_results_dataset_single_node[-1]), fill_value=np.nan).tolist()]*pad_no))
                full_results_dataset_sonicom_synthetic_tl = np.concatenate((full_results_dataset_sonicom_synthetic_tl, full_results_dataset_sonicom_synthetic_tl_double_node, full_results_dataset_sonicom_synthetic_tl_single_node))
                # full_results_dataset_dataset_tl = np.concatenate((full_results_dataset_dataset_tl, [np.full(shape=len(full_results_dataset_single_node[-1]), fill_value=np.nan).tolist()]*pad_no))
                full_results_dataset_dataset_tl = np.concatenate((full_results_dataset_dataset_tl, full_results_dataset_dataset_tl_single_node, full_results_dataset_dataset_tl_double_node))

                # full_results_dataset_sonicom_synthetic_tl = np.concatenate((full_results_dataset_sonicom_synthetic_tl, [np.full(shape=len(full_results_dataset_sonicom_synthetic_tl[-1]), fill_value=np.nan).tolist()]*pad_no))
                # full_results_dataset_dataset_tl = np.concatenate((full_results_dataset_dataset_tl, [np.full(shape=len(full_results_dataset_dataset_tl[-1]), fill_value=np.nan).tolist()]*pad_no))
                #######################################
                create_table(legend, [full_results_dataset+full_results_dataset_single_node], dataset.upper(),
                             units='[dB]')
                plot_boxplot(config, f'LSD_boxplot_ex_{experiment_id}_{dataset}',
                             f'{dataset.upper()} \n LSD error [dB]',
                             [full_results_dataset+full_results_dataset_double_node+full_results_dataset_single_node, full_results_dataset_sonicom_synthetic_tl, full_results_dataset_dataset_tl, full_results_dataset_baseline], legend, colours, ticks, xlabel, hrtf_selection_results=full_results_dataset_baseline_hrtf_selection)
            elif mode == 'loc':
                types = ['ACC', 'RMS', 'QUERR']
                labels = [r'Polar ACC error [$^\circ$]', r'Polar RMS error [$^\circ$]', 'Quadrant error [\%]']
                labels = [f'{dataset.upper()} \n' + label for label in labels]
                units = [r'[$^\circ$]', r'[$^\circ$]', '[\%]']
                legend = ['SRGAN', 'TL (Synthetic)', f'TL ({other_dataset})', 'Baseline', 'Target']
                colours = ['#0047a4', '#af211a', 'g', '#6C0BA9', '#E67E22']
                # remove baseline results at upscale-16
                # full_results_dataset_baseline[0] = np.full(shape=(np.shape(full_results_dataset_baseline[-1])), fill_value=np.nan).tolist()
                #######################################
                full_results_dataset_target_tl, _ = get_results(config.data_dirs_path + '/data/' + dataset.upper(),
                                                             'target',
                                                             file_ext=f'{dataset.upper()}_loc_target_valid_errors.pickle') * 9
                for i in np.arange(np.shape(full_results_dataset)[1]):
                    plot_boxplot(config, f'{types[i]}_boxplot_ex_{experiment_id}_{dataset}', labels[i],
                                 [np.concatenate((np.array(full_results_dataset_baseline)[::-1, i, :], np.array(full_results_dataset)[:, i, :])),
                                  np.array(full_results_dataset_target_tl)[:, i, :]], legend, colours, ticks, xlabel)
                    print(f'Generate table containing {types[i]} errors for the {dataset.upper()} dataset: \n')
                    create_table(legend, [np.array(full_results_dataset)[:, i, :],
                                          np.array(full_results_dataset_baseline)[:, i, :],
                                          [np.array(full_results_dataset_target_tl)[0, i, :]]], dataset.upper(),
                                 units=units[i])
    else:
        print('Experiment does not exist')


def run_baseline(hpc, test_id=None):
    print(f'Running training')
    config_files = []
    upscale_factors = [2, 4, 8, 16]
    datasets = ['ARI', 'SONICOM']
    for dataset in datasets:
        if args.mode == 'barycentric_baseline':
            for upscale_factor in upscale_factors:
                config = Config(tag=None, using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset)
                config.upscale_factor = upscale_factor
                config.dataset = dataset
                config_files.append(config)
        elif args.mode == 'hrtf_selection_baseline':
            config = Config(tag=None, using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset)
            config.dataset = dataset
            config_files.append(config)

    print(f'{len(config_files)} config files created successfully.')
    if test_id is not None:
        if test_id.isnumeric():
            test_id = int(test_id)
            config_files = [config_files[test_id]]
        else:
            print(f'{test_id} not found')

    print(f'Running a total of {len(config_files)} config files')
    for config in config_files:
        main(config, args.mode)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
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

    if args.mode == 'projection':
        run_projection(hpc, args.test)
    elif args.mode == 'preprocess':
        run_preprocess(hpc, args.type, args.test)
    elif args.mode == 'train':
        run_train(hpc, args.type, args.test)
    elif args.mode == 'tuning':
        run_train(hpc, args.type, args.test, tuning=True)
    elif args.mode == 'tuning_results':
        get_tuning_results(hpc, args.test)
    elif args.mode == 'evaluation':
        run_evaluation(hpc, int(args.exp), args.type, args.test)
    elif args.mode == 'plot':
        plot_evaluation(hpc, int(args.exp), args.type)
    elif args.mode == 'barycentric_baseline' or args.mode == 'hrtf_selection_baseline':
        run_baseline(hpc, args.test)
    else:
        print('Please specify a valid mode')