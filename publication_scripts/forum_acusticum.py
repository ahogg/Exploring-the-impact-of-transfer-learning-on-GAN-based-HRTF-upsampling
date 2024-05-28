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

    for idx, full_result in enumerate(full_results):
        # Append nans to results to make them of equal length
        maxlen = np.max([len(i) for i in full_result])
        for result in full_result:
            if (maxlen - len(result)) > 0:
                result[:] = [np.NaN] * (maxlen - len(result)) + result

        data = np.vstack(full_result)

        for i, d in enumerate(data):
            filtered_data = d[~np.isnan(d)]
            if len(filtered_data) > 5:
                blp = plt.boxplot(filtered_data, positions=[(i * 1.0) - np.linspace(0.15 * (len(full_results) / 2), -0.15 * (len(full_results) / 2), len(full_results))[idx]],
                                  flierprops=dict(marker='x', markeredgecolor=colours[idx], markersize=4), widths=0.12)

                for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                    plt.setp(blp[element], color=colours[idx], linewidth=0.7)

        plt.plot([], c=colours[idx], label=legend[idx])

    if hrtf_selection_results is not None:
        ticks += ['HRTF\n Selection']
        c = ['#FF8D1C', '#00999E']
        l = ['Selection-1', 'Selection-2']
        if legend[-1] == 'Target':
            hrtf_selection_results = np.concatenate((hrtf_selection_results, np.array([filtered_data])))

        hrtf_selection_results_len = len(hrtf_selection_results)-1
        for idx, hrtf_selection_result in enumerate(hrtf_selection_results):
            if idx == hrtf_selection_results_len and legend[-1] == 'Target':
                i = len(full_results) - 1
                blp = plt.boxplot(hrtf_selection_result,
                                  positions=[(len(data) * 1.0) - np.linspace(0.15 * (len(full_results) / 2),
                                                                             -0.15 * (len(full_results) / 2),
                                                                             len(full_results))[idx + 1]-0.1],
                                  flierprops=dict(marker='x', markeredgecolor=colours[i], markersize=4), widths=0.12)
                for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                    plt.setp(blp[element], color=colours[i], linewidth=0.7)

            else:
                blp = plt.boxplot(hrtf_selection_result, positions=[(len(data) * 1.0) - np.linspace(0.15 * (len(full_results) / 2),
                                                                                            -0.15 * (len(full_results) / 2),
                                                                                            len(full_results))[idx+1]-0.1],
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

    leg = ax.legend(prop={'size': 7}, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower right", # mode="expand",
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
    if hrtf_selection_results is not None:
        ymin_hrtf_selection = np.nanmin(hrtf_selection_results) - 0.1 * abs(np.nanmax(hrtf_selection_results) - np.nanmin(hrtf_selection_results))
        ymax_hrtf_selection = np.nanmax(hrtf_selection_results) + 0.1 * abs(np.nanmax(hrtf_selection_results) - np.nanmin(hrtf_selection_results))
        ax.set_ylim((np.min([ymin, ymin_hrtf_selection]), np.max([ymax, ymax_hrtf_selection])))

    ax.yaxis.set_label_coords(-0.12, 0.5)
    ax.xaxis.set_label_coords(0.4, -0.2)

    w = 2.974
    h = w / 1.5
    # w = 5
    # h = w / 2.5
    fig.set_size_inches(w, h)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    fig.savefig(config.data_dirs_path + '/plots/' + name, bbox_inches='tight')


# def plot_boxplot(config, name, ylabel, full_results, legend, colours):
#     plt.rc('font', family='serif', serif='Times New Roman')
#     plt.rc('text', usetex=True)
#     plt.rc('xtick', labelsize=8)
#     plt.rc('ytick', labelsize=8)
#     plt.rc('axes', labelsize=8)
#
#     fig, ax = plt.subplots()
#
#     factors = [2, 4, 8, 16]
#     ticks = [
#         r'$%s \,{\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}} %s$' % (int(
#             (16 / factor) ** 2 * 5), int(config.hrtf_size ** 2 * 5)) for factor in factors]
#
#     for idx, full_result in enumerate(full_results):
#         data = np.vstack((full_result[3], full_result[2], full_result[1], full_result[0]))
#
#         blp = plt.boxplot(data.T, positions=np.array(range(len(data))) * 1.0 - np.linspace(0.15*(len(full_results)/2), -0.15*(len(full_results)/2), len(full_results))[idx],
#                                                 flierprops=dict(marker='x', markeredgecolor=colours[idx], markersize=4), widths=0.12)
#
#         for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
#             plt.setp(blp[element], color=colours[idx], linewidth=0.7)
#
#         plt.plot([], c=colours[idx], label=legend[idx])
#
#     if len(full_results) > 2:
#         [plt.axvline(x + 0.5, color='#a6a6a6', linestyle='--', linewidth=0.5) for x in range(0, (len(ticks) - 1), 1)]
#     else:
#         [plt.axvline(x + 1, color='#a6a6a6', linestyle='--', linewidth=0.5) for x in range(0, (len(ticks) - 1) * 2, 2)]
#
#     leg = ax.legend(prop={'size': 7}, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower right",  # mode="expand",
#                         borderaxespad=0, ncol=2, handlelength=1.06)
#
#     leg.get_frame().set_linewidth(0.5)
#     leg.get_frame().set_edgecolor('k')
#
#     ax.yaxis.grid(zorder=0, linewidth=0.4)
#     plt.xlabel(
#         'Upsample Factor\n' + r'(No. of original nodes$\ {\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}}$ No. of upsampled nodes)')
#
#     plt.ylabel(ylabel)
#     if len(full_results) > 2:
#         plt.xticks(range(0, len(ticks), 1), ticks)
#         plt.xlim(-0.5, len(ticks) - 0.5)
#     else:
#         plt.xticks(range(0, len(ticks) * 2, 2), ticks)
#
#     # Append nans to results to make them of equal length
#     maxlen = np.max([[len(j) for j in i] for i in full_results])
#     for full_result in full_results:
#         for result in full_result:
#             if (maxlen - len(result)) > 0:
#                 result[:] = [np.nan] * (maxlen - len(result)) + result
#
#     ymin = np.nanmin(full_results) - 0.1 * abs(np.nanmax(full_results) - np.nanmin(full_results))
#     ymax = np.nanmax(full_results) + 0.1 * abs(np.nanmax(full_results) - np.nanmin(full_results))
#     ax.set_ylim((ymin, ymax))
#     ax.yaxis.set_label_coords(-0.12, 0.5)
#
#     w = 2.974
#     h = w / 1.5
#     fig.set_size_inches(w, h)
#     fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
#     fig.savefig(config.data_dirs_path + '/plots/' + name, bbox_inches='tight')


# def get_results(tag, mode, file_ext=None):
#     full_results = []
#     upscale_factors = [16, 8, 4, 2]
#     for upscale_factor in upscale_factors:
#         config = Config(tag + str(upscale_factor), using_hpc=hpc)
#         if mode == 'lsd' or mode == 'baseline_lsd':
#             if mode == 'lsd':
#                 file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext
#                 file_path = f'{config.path}/{file_ext}'
#             elif mode == 'baseline_lsd':
#                 file_path = f'{tag}/{file_ext}{upscale_factor}.pickle'
#             with open(file_path, 'rb') as file:
#                 lsd_id_errors = pickle.load(file)
#             lsd_errors = [lsd_error[1] for lsd_error in lsd_id_errors]
#             print(f'Loading: {file_path}')
#             print('Mean (STD) LSD: %0.3f (%0.3f)' % (np.mean(lsd_errors),  np.std(lsd_errors)))
#             full_results.append(lsd_errors)
#         elif mode == 'loc' or mode == 'target' or mode == 'baseline_loc':
#             file_ext = 'loc_errors.pickle' if file_ext is None else file_ext
#             if mode == 'loc':
#                 file_path = f'{config.path}/{file_ext}'
#             elif mode == 'target':
#                 file_path = tag + '/' + file_ext
#             elif mode == 'baseline_loc':
#                 file_path = f'{tag}/{file_ext}{upscale_factor}.pickle'
#
#             with open(file_path, 'rb') as file:
#                 loc_id_errors = pickle.load(file)
#             pol_acc1 = [loc_error[1] for loc_error in loc_id_errors]
#             pol_rms1 = [loc_error[2] for loc_error in loc_id_errors]
#             querr1 = [loc_error[3] for loc_error in loc_id_errors]
#             print(f'Loading: {file_path}')
#             print('Mean (STD) ACC Error: %0.3f (%0.3f)' % (np.mean(pol_acc1), np.std(pol_acc1)))
#             print('Mean (STD) RMS Error: %0.3f (%0.3f)' % (np.mean(pol_rms1), np.std(pol_rms1)))
#             print('Mean (STD) QUERR Error: %0.3f (%0.3f)' % (np.mean(querr1), np.std(querr1)))
#             full_results.append([pol_acc1, pol_rms1, querr1])
#
#             if mode == 'target':
#                 break
#
#     return full_results

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
        # plt.clim(0, max_node_lsd)
        plt.clim(0, 17.5)
        plt.xlabel(r"Azimuth [$^\circ$]")
        plt.ylabel(r"Elevation [$^\circ$]")
        #     plt.text(0.5, 0.5, 'Average SD error', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        width = 2.874
        height = width / 1.4
        fig.set_size_inches(width*1.4, height*1.4)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        fig.savefig(f'{config.data_dirs_path}/plots/SD_node_{upsampling_idx}.png', dpi=300)
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

# def get_results(tag, mode, upscale_factors=[2, 4, 8, 16], file_ext=None, runs_folder=None):
#     full_results = []
#     for upscale_factor in upscale_factors:
#         config = Config(tag + str(upscale_factor), using_hpc=hpc, runs_folder=runs_folder)
#         if mode == 'lsd' or mode == 'baseline_lsd':
#             if mode == 'lsd':
#                 file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext
#                 file_path = f'{config.path}/{file_ext}'
#             elif mode == 'baseline_lsd':
#                 file_path = f'{tag}/{file_ext}{upscale_factor}.pickle'
#             with open(file_path, 'rb') as file:
#                 lsd_id_errors = pickle.load(file)
#             lsd_errors = [lsd_error['total_error'] if not np.isinf(lsd_error['total_error']) else np.nan for lsd_error in lsd_id_errors]
#             print(f'Loading: {file_path}')
#             print('Mean (STD) LSD: %0.3f (%0.3f)' % (np.mean(lsd_errors),  np.std(lsd_errors)))
#             full_results.append(lsd_errors)
#         elif mode == 'loc' or mode == 'target' or mode == 'baseline_loc':
#             file_ext = 'loc_errors.pickle' if file_ext is None else file_ext
#             if mode == 'loc':
#                 file_path = f'{config.path}/{file_ext}'
#             elif mode == 'target':
#                 file_path = tag + '/' + file_ext
#             elif mode == 'baseline_loc':
#                 file_path = f'{tag}/{file_ext}{upscale_factor}.pickle'
#
#             with open(file_path, 'rb') as file:
#                 loc_id_errors = pickle.load(file)
#             pol_acc1 = [loc_error[1] for loc_error in loc_id_errors]
#             pol_rms1 = [loc_error[2] for loc_error in loc_id_errors]
#             querr1 = [loc_error[3] for loc_error in loc_id_errors]
#             print(f'Loading: {file_path}')
#             print('Mean (STD) ACC Error: %0.3f (%0.3f)' % (np.mean(pol_acc1), np.std(pol_acc1)))
#             print('Mean (STD) RMS Error: %0.3f (%0.3f)' % (np.mean(pol_rms1), np.std(pol_rms1)))
#             print('Mean (STD) QUERR Error: %0.3f (%0.3f)' % (np.mean(querr1), np.std(querr1)))
#             full_results.append([pol_acc1, pol_rms1, querr1])
#
#             if mode == 'target':
#                 break
#
#     return full_results


def run_projection(hpc, dataset_id=None, lap_factor=None):
    print(f'Running projection')
    config_files = []
    datasets = ['ARI', 'SONICOM', 'SONICOMSynthetic']
    for dataset in datasets:
        config = Config(tag=None, using_hpc=hpc, dataset=dataset, lap_factor=lap_factor)
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

def run_preprocess(hpc, type, dataset_id=None, lap_factor=None):
    print(f'Running preprocess')
    config_files = []
    datasets = ['ARI', 'SONICOM', 'SONICOMSynthetic']
    for dataset in datasets:
        if type == 'base':
            config = Config(tag=None, using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset, lap_factor=lap_factor)
            config.train_samples_ratio = 0.8
        elif type == 'tl':
            config = Config(tag=None, using_hpc=hpc, dataset=dataset, data_dir='/data-transfer-learning/' + dataset, lap_factor=lap_factor)
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

def run_train(hpc, type, test_id=None, lap_factor=None):
    print(f'Running training')
    config_files = []
    tags = []
    if lap_factor == '100':
        upscale_factors = [2]
    elif lap_factor == '19':
        upscale_factors = [4]
    else:
        upscale_factors = [2, 4, 8, 16]
    datasets = ['ARI', 'SONICOM', 'SONICOMSynthetic']
    if type == 'tl' or type == 'base':
        datasets.remove('SONICOMSynthetic')
    for dataset in datasets:
        other_dataset = 'ARI' if dataset == 'SONICOM' else 'SONICOM'
        for upscale_factor in upscale_factors:
            if lap_factor is not None:
                tags = [{'tag': f'pub-prep-upscale-{dataset}-LAP-{lap_factor}'.replace('_','-')}]
            else:
                if type == 'base':
                    tags = [{'tag': f'pub-prep-upscale-{dataset}-{upscale_factor}'}]
                elif type == 'base-tl':
                    tags = [{'tag': f'pub-prep-upscale-{dataset}-tl-{upscale_factor}'}]
                elif type == 'tl':
                    tags = [{'tag': f'pub-prep-upscale-{dataset}-{other_dataset}-tl-{upscale_factor}', 'existing_model_tag': f'pub-prep-upscale-{other_dataset}-tl-{upscale_factor}'},
                            {'tag': f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-{upscale_factor}', 'existing_model_tag': f'pub-prep-upscale-SONICOMSynthetic-tl-{upscale_factor}'}]
                else:
                    print("Type not valid. Please use 'base' or 'tl'")

            for tag in tags:
                if type == 'base':
                    config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset, lap_factor=lap_factor)
                elif type == 'base-tl':
                    config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data-transfer-learning/' + dataset, lap_factor=lap_factor)
                elif type == 'tl':
                    config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, existing_model_tag=tag['existing_model_tag'], data_dir='/data/' + dataset, lap_factor=lap_factor)
                config.upscale_factor = upscale_factor
                config.lr_gen = 0.0002
                config.lr_dis = 0.0000015
                if lap_factor == '100':
                    config.content_weight = 0.1
                    config.adversarial_weight = 0.001
                elif lap_factor == '19':
                    config.content_weight = 0.01
                    config.adversarial_weight = 0.1
                else:
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
                else:
                    print(f'{test_id} not found')

    print(f'Running a total of {len(config_files)} config files')
    for config in config_files:
        main(config, args.mode)


def run_evaluation(hpc, experiment_id, type, test_id=None, lap_flag=None):
    print(f'Running {type} experiment {experiment_id}')
    lap = 'lap_100' if lap_flag else False
    config_files = []
    if lap=='lap_100':
        datasets = ['SONICOM']
        upscale_factor = 2
        for dataset in datasets:
            tags = [{'tag': f'pub-prep-upscale-{dataset}-{lap.upper()}'.replace('_', '-')}]
            for tag in tags:
                config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset, lap=lap)
                config.upscale_factor = upscale_factor
                config_files.append(config)
    else:
        if experiment_id == 1:
            upscale_factors = [2, 4, 8, 16]
            datasets = ['ARI']
            for dataset in datasets:
                for upscale_factor in upscale_factors:
                    tags = [{'tag': f'pub-prep-upscale-{dataset}-{upscale_factor}'}]
                    for tag in tags:
                        config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset)
                        config.upscale_factor = upscale_factor
                        config_files.append(config)
        elif experiment_id == 2:
            upscale_factors = [2, 4, 8, 16]
            datasets = ['ARI', 'SONICOM', ]
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
            upscale_factors = [2]
            datasets = ['SONICOM']
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
        # elif experiment_id == 4:
        #     upscale_factors = [2, 4, 8, 16]
        #     datasets = ['ARI', 'SONICOM']
        #     for dataset in datasets:
        #         for upscale_factor in upscale_factors:
        #             tag = None
        #             config = Config(tag, using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset)
        #             config.upscale_factor = upscale_factor
        #             config.valid_path = f'{config.data_dirs_path}/baseline_results/{config.dataset}/barycentric/valid/barycentric_interpolated_data_{upscale_factor}'
        #             config.path = f'{config.data_dirs_path}/baseline_results/{config.dataset}/barycentric/valid'
        #             config_files.append(config)
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
        # elif experiment_id == 4:
        #     if type == 'lsd':
        #         run_lsd_evaluation(config, config.valid_path, f'lsd_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle')
        #     elif type == 'loc':
        #         run_localisation_evaluation(config, config.valid_path, f'loc_errors_barycentric_interpolated_data_{config.upscale_factor}.pickle')
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
                full_results_LSD_dataset = get_results(f'pub-prep-upscale-{dataset}-', mode)
                full_results_LSD_dataset_sonicom_synthetic_tl = get_results(f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-', mode)
                legend = ['SRGAN', 'SRGAN TL (Synthetic)']
                colours = ['#0047a4', '#af211a', 'g', '#6C0BA9', '#E67E22']
                plot_boxplot(config, f'LSD_boxplot_ex_{experiment_id}_{dataset}', f'{dataset.upper()} LSD error [dB]', [full_results_LSD_dataset, full_results_LSD_dataset_sonicom_synthetic_tl], legend, colours)
            elif mode == 'loc':
                full_results_loc_dataset = get_results(f'pub-prep-upscale-{dataset}-', mode)
                full_results_loc_dataset_sonicom_synthetic_tl = get_results(f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-', mode)
                types = ['ACC', 'RMS', 'QUERR']
                labels = [r'Polar accuracy error [$^\circ$]', r'Polar RMS error [$^\circ$]', 'Quadrant error [\%]']
                for i in np.arange(np.shape(full_results_loc_dataset)[1]):
                    legend = ['SRGAN', 'SRGAN TL (Synthetic)']
                    colours = ['#0047a4', '#af211a', 'g', '#6C0BA9', '#E67E22']
                    plot_boxplot(config, f'{types[i]}_boxplot_ex_{experiment_id}_{dataset}', labels[i], [np.array(full_results_loc_dataset)[:, i, :],
                                np.array(full_results_loc_dataset_sonicom_synthetic_tl)[:, i, :]], legend, colours)

    elif experiment_id == 2:
        datasets = ['ARI', 'SONICOM']
        for dataset in datasets:
            other_dataset = 'ARI' if dataset == 'SONICOM' else 'SONICOM'
            full_results_dataset = get_results(f'pub-prep-upscale-{dataset}-', mode)
            full_results_dataset_sonicom_synthetic_tl = get_results(f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-', mode)
            full_results_dataset_dataset_tl = get_results(f'pub-prep-upscale-{dataset}-{other_dataset}-tl-', mode)
            full_results_dataset_baseline = get_results(f'{config.data_dirs_path}/baseline_results/{dataset.upper()}/cube_sphere/barycentric/valid', mode=f'baseline_{mode}', file_ext=f'{mode}_errors_barycentric_interpolated_data_')
            full_results_dataset_sh_baseline = get_results(f'{config.data_dirs_path}/baseline_results/{dataset.upper()}/cube_sphere/sh/valid', mode=f'baseline_{mode}', file_ext=f'{mode}_errors_sh_interpolated_data_')

            factors = [2, 4, 8, 16]
            ticks = [
                r'$%s \,{\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}} %s$' % (
                    int((16 / factor) ** 2 * 5), int(config.hrtf_size ** 2 * 5)) for factor in factors]

            if mode == 'lsd':
                # remove baseline results at upscale-16
                # full_results_dataset_baseline[0] = np.full(shape=len(full_results_dataset_baseline[-1]), fill_value=np.nan).tolist()
                legend = ['SRGAN (No TL)', 'TL (Synthetic)', f'TL ({other_dataset})', 'Baseline']
                colours = ['#0047a4', '#af211a', 'g', '#6C0BA9', '#E67E22']
                create_table(legend, [full_results_dataset, full_results_dataset_sonicom_synthetic_tl, full_results_dataset_dataset_tl, full_results_dataset_baseline], dataset.upper(), units='[dB]')
                plot_boxplot(config, f'LSD_boxplot_ex_{experiment_id}_{dataset}', f'{dataset.upper()} \n LSD error [dB]', [full_results_dataset, full_results_dataset_sonicom_synthetic_tl, full_results_dataset_dataset_tl, full_results_dataset_baseline], legend, colours, ticks)
            elif mode == 'loc':
                types = ['ACC', 'RMS', 'QUERR']
                labels = [r'Polar ACC error [$^\circ$]', r'Polar RMS error [$^\circ$]', 'Quadrant error [\%]']
                labels = [f'{dataset.upper()} \n' + label for label in labels]
                units = [r'[$^\circ$]', r'[$^\circ$]', '[\%]']
                legend = ['SRGAN (No TL)', 'TL (Synthetic)', f'TL ({other_dataset})', 'Baseline', 'Target']
                colours = ['#0047a4', '#af211a', 'g', '#6C0BA9', '#E67E22']
                # remove baseline results at upscale-16
                # full_results_dataset_baseline[0] = np.full(shape=(np.shape(full_results_dataset_baseline[-1])), fill_value=np.nan).tolist()
                #######################################
                full_results_dataset_target_tl = get_results(config.data_dirs_path + '/data/' + dataset.upper() + '/cube_sphere',
                                                            'target',
                                                            file_ext=f'{dataset.upper()}_loc_target_valid_errors.pickle') * 4
                for i in np.arange(np.shape(full_results_dataset)[1]):
                    plot_boxplot(config, f'{types[i]}_boxplot_ex_{experiment_id}_{dataset}', labels[i], [np.array(full_results_dataset)[:, i, :],
                                np.array(full_results_dataset_sonicom_synthetic_tl)[:, i, :], np.array(full_results_dataset_dataset_tl)[:, i, :], np.array(full_results_dataset_baseline)[:, i, :], np.array(full_results_dataset_target_tl)[:, i, :]], legend, colours, ticks)
                    print(f'Generate table containing {types[i]} errors for the {dataset.upper()} dataset: \n')
                    create_table(legend, [np.array(full_results_dataset)[:, i, :],
                                np.array(full_results_dataset_sonicom_synthetic_tl)[:, i, :], np.array(full_results_dataset_dataset_tl)[:, i, :], np.array(full_results_dataset_baseline)[:, i, :], [np.array(full_results_dataset_target_tl)[0, i, :]]], dataset.upper(), units=units[i])
    elif experiment_id == 4:
        datasets = ['ARI', 'SONICOM']
        for dataset in datasets:
            full_results_dataset, _ = get_results(f'pub-prep-upscale-{dataset}-', mode, upscale_factors=[2, 4, 8, 16], runs_folder='/runs-hpc')
            # full_results_dataset_sonicom_synthetic_tl = get_results(f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-',
            #                                                         mode, upscale_factors=[2, 4, 8, 16], runs_folder='/runs-hpc')
            full_results_dataset_baseline, full_lsd_plot_results_baseline = get_results(
                f'{config.data_dirs_path}/baseline_results/{dataset.upper()}/cube_sphere/barycentric/valid',
                mode=f'baseline_{mode}', upscale_factors=[2, 4, 8, 16], file_ext=f'{mode}_errors_barycentric_interpolated_data_')
            full_results_dataset_sh_baseline, full_lsd_plot_results_sh_baseline  = get_results(
                f'{config.data_dirs_path}/baseline_results/{dataset.upper()}/cube_sphere/sh/valid',
                mode=f'baseline_{mode}', file_ext=f'{mode}_errors_sh_interpolated_data_')
            full_results_dataset_baseline_hrtf_selection, _ = get_results(
                f'{config.data_dirs_path}/baseline_results/{dataset.upper()}/cube_sphere/hrtf_selection/valid',
                mode=f'baseline_{mode}', upscale_factors=['minimum_data', 'maximum_data'],
                file_ext=f'{mode}_errors_hrtf_selection_')
            full_results_dataset_baseline_hrtf_selection = [[j for j in i if j != 0.0] for i in full_results_dataset_baseline_hrtf_selection]
            factors = [2, 4, 8, 16]
            xlabel = 'Upsample Factor'
            if mode == 'lsd':
                basline_ticks = [
                    r'$%s \,{\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}} %s$' % (
                        int((16 / factor) ** 2 * 5), int(config.hrtf_size ** 2 * 5)) for factor in factors]
                # remove baseline results at upscale-16
                # full_results_dataset_baseline[0] = np.full(shape=len(full_results_dataset_baseline[-1]), fill_value=np.nan).tolist()
                legend = ['SRGAN', f'SH Baseline', 'Barycentric Baseline', 'Selection-1', 'Selection-2']
                colours = ['#0047a4', '#af211a', 'g', '#FFA500', '#E67E22']
                plot_lsd_plot(config, full_lsd_plot_results_sh_baseline)
                create_table(legend, [full_results_dataset[::-1], full_results_dataset_sh_baseline, full_results_dataset_baseline[::-1], [np.array(full_results_dataset_baseline_hrtf_selection)[0, :]],
                                          [np.array(full_results_dataset_baseline_hrtf_selection)[1, :]]], units='[dB]')
                plot_boxplot(config, f'LSD_boxplot_ex_{experiment_id}_{dataset}', f'LSD error [dB]', [full_results_dataset, full_results_dataset_sh_baseline, full_results_dataset_baseline], legend, colours, basline_ticks, xlabel, hrtf_selection_results=np.array(full_results_dataset_baseline_hrtf_selection))
            elif mode == 'loc':
                types = ['ACC', 'RMS', 'QUERR']
                labels = [r'Polar ACC error [$^\circ$]', r'Polar RMS error [$^\circ$]', 'Quadrant error [\%]']
                # labels = [f'{dataset.upper()} \n' + label for label in labels]
                units = [r'[$^\circ$]', r'[$^\circ$]', '[\%]']
                colours = ['#0047a4', '#af211a', 'g', '#6C0BA9']
                # remove baseline results at upscale-16
                # full_results_dataset_baseline[0] = np.full(shape=(np.shape(full_results_dataset_baseline[-1])), fill_value=np.nan).tolist()
                #######################################
                full_results_dataset_target_tl = get_results(config.data_dirs_path + '/data/' + dataset.upper() + '/cube_sphere',
                                                            'target',
                                                            file_ext=f'{dataset.upper()}_loc_target_valid_errors.pickle') * 4

                for i in np.arange(np.shape(full_results_dataset)[1]):
                    legend = ['SRGAN', f'SH Baseline', 'Barycentric Baseline', 'Target']
                    basline_ticks = [
                        r'$%s \,{\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}} %s$' % (
                            int((16 / factor) ** 2 * 5), int(config.hrtf_size ** 2 * 5)) for factor in factors]
                    plot_boxplot(config, f'{types[i]}_boxplot_ex_{experiment_id}_{dataset}', labels[i],
                                 [np.array(full_results_dataset)[:, i, :], np.array(full_results_dataset_sh_baseline)[:, i, :], np.array(full_results_dataset_baseline)[:, i, :],
                                  np.array(full_results_dataset_target_tl)[:, i, :]], legend, colours, basline_ticks, xlabel, hrtf_selection_results=np.array(full_results_dataset_baseline_hrtf_selection)[:, i, :])
                    print(f'Generate table containing {types[i]} errors for the {dataset.upper()} dataset: \n')
                    legend = ['SRGAN', f'SH Baseline', 'Barycentric Baseline', 'Selection-1', 'Selection-2', 'Target']
                    create_table(legend, [np.array(full_results_dataset)[::-1, i, :],
                                          np.array(full_results_dataset_sh_baseline)[::-1, i, :],
                                          np.array(full_results_dataset_baseline)[::-1, i, :],
                                          [np.array(full_results_dataset_baseline_hrtf_selection)[0, i, :]],
                                          [np.array(full_results_dataset_baseline_hrtf_selection)[1, i, :]],
                                          [np.array(full_results_dataset_target_tl)[0, i, :]]],
                                 units=units[i])


                # for i in np.arange(np.shape(full_results_dataset)[1]):
                #     plot_boxplot(config, f'{types[i]}_boxplot_ex_{experiment_id}_{dataset}', labels[i],
                #                  [np.array(full_results_dataset)[:, i, :],
                #                   # np.array(full_results_dataset_sonicom_synthetic_tl)[:, i, :],
                #                   np.array(full_results_dataset_sh_baseline)[:, i, :],
                #                   np.array(full_results_dataset_baseline)[:, i, :],
                #                   np.array(full_results_dataset_target_tl)[:, i, :]], legend, colours, ticks)
                #     print(f'Generate table containing {types[i]} errors for the {dataset.upper()} dataset: \n')
                #     create_table(legend, [np.array(full_results_dataset)[:, i, :],
                #                           # np.array(full_results_dataset_sonicom_synthetic_tl)[:, i, :],
                #                           np.array(full_results_dataset_sh_baseline)[:, i, :],
                #                           np.array(full_results_dataset_baseline)[:, i, :],
                #                           [np.array(full_results_dataset_target_tl)[0, i, :]]], dataset.upper(),
                #                  units=units[i])

    elif experiment_id == 5:

        def get_trasfer_function(hrtf_file, position):
            hrtf = []
            with (open(hrtf_file, 'rb')) as openfile:
                while True:
                    try:
                        hrtf.append(pickle.load(openfile))
                    except EOFError:
                        break
            return hrtf[0][position[0]][position[1]][position[2]][0:128]


        plt.rc('font', family='serif', serif='Times New Roman')
        plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('axes', labelsize=8)

        factors = [2, 4, 8, 16]
        positions = [(3, 7, 7), (3, 15, 15)]
        subject = 853

        for pos_idx, position in enumerate(positions):

            fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)

            for factor_idx, factor in enumerate(factors):
                srgan = f'{config.data_dirs_path}{config.runs_folder}/pub-prep-upscale-ARI-{factor}/valid'
                ground_truth = f'{config.data_dirs_path}/data/ARI/cube_sphere/hr_merge/valid'
                barycentic = f'{config.data_dirs_path}/baseline_results/ARI/cube_sphere/barycentric/valid/barycentric_interpolated_data_{factor}'
                sh = f'{config.data_dirs_path}/baseline_results/ARI/cube_sphere/sh/valid/sh_interpolated_data_{factor}'
                tags = [srgan, sh, barycentic, ground_truth]

                labels = ['SRGAN', 'SH Baseline', 'Barycentic Baseline', 'Ground Truth']
                c = ['#0047a4', '#af211a', 'g', 'black']

                for idx, tag in enumerate(tags):
                    hrtf_file = f'{tag}/ARI_mag_{subject}.pickle'
                    hrtf_tf = get_trasfer_function(hrtf_file, position)
                    axs[factor_idx].plot(np.linspace(0, config.hrir_samplerate/2, len(hrtf_tf)),20 * np.log10(hrtf_tf) ** 2, c=c[idx], label=labels[idx], linewidth=1)
                    # plt.plot(get_trasfer_function(hrtf_file))

                title = r'$%s \,{\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}} %s$' % (int((16 / factor) ** 2 * 5), int(config.hrtf_size ** 2 * 5))
                axs[factor_idx].set_ylabel(f'Factor: {title} \n Manitude [dB]')

                if factor_idx == 0:
                    axs[factor_idx].set_title(f'Location - Panel: {position[0]}, Position: ({position[1]},{position[2]})', fontsize=8)
                    leg = axs[factor_idx].legend(prop={'size': 7}, loc="upper left",
                                                 # mode="expand",  bbox_to_anchor=(0, 1.02, 1, 0.2), borderaxespad=0,
                                                 ncol=1, handlelength=1.06)

                    leg.get_frame().set_linewidth(0.5)
                    leg.get_frame().set_edgecolor('k')


                if factor_idx != len(factors)-1:
                    axs[factor_idx].tick_params( axis='x',          # changes apply to the x-axis
                                        which='both',      # both major and minor ticks are affected
                                        bottom=False,      # ticks along the bottom edge are off
                                        top=False,         # ticks along the top edge are off
                                        labelbottom=False) # labels along the bottom edge are off
                else:
                    axs[factor_idx].set_xlabel(f'Frequency [Hz]')

                axs[factor_idx].yaxis.grid(zorder=0, linewidth=0.4)
                axs[factor_idx].xaxis.grid(zorder=0, linewidth=0.4)
                axs[factor_idx].set_ylim(bottom=0)
                axs[factor_idx].set_xscale('log')

            w = 2.974
            h = w * 2
            fig.set_size_inches(w, h)
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
            fig.savefig(f'{config.data_dirs_path}/plots/freq_dependent/freq_dependent_{position[0]}_{position[1]}_{position[2]}.png', dpi=600,
                            bbox_inches='tight')

        ##################################################

        tag = f'/pub-prep-upscale-ARI-8/train_losses.pickle'
        training_loss_file = config.data_dirs_path+config.runs_folder+tag
        losses = []
        with (open(training_loss_file, 'rb')) as openfile:
            while True:
                try:
                    losses.append(pickle.load(openfile))
                except EOFError:
                    break

        fig, ax1 = plt.subplots()

        ax1.plot([x*100 for x in losses[0][0]], '#af211a', label='Generator loss')
        ax1.plot([x*0.1 for x in losses[0][3]], '#0047a4', label='Discriminator loss')

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')

        w = 2.974
        h = w / 2
        leg1 = ax1.legend(prop={'size': 7}, loc="upper right",  # mode="expand",  bbox_to_anchor=(0, 1.02, 1, 0.2), borderaxespad=0,
                        ncol=1, handlelength=1.06)

        leg1.get_frame().set_linewidth(0.5)
        leg1.get_frame().set_edgecolor('k')
        ax1.yaxis.grid(zorder=0, linewidth=0.4)
        ax1.xaxis.grid(zorder=0, linewidth=0.4)

        ax1.set_ylim((0, 0.6))
        ax1.set_xlim((0, 300))

        fig.set_size_inches(w, h)
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        fig.savefig(config.data_dirs_path + '/plots/upsample_8_pub-prep-arrayjob_8_loss_curves_pub_2_axis', bbox_inches='tight')
    else:
        print('Experiment does not exist')


def run_baseline(hpc, test_id=None):
    print(f'Running training')
    config_files = []
    upscale_factors = [2, 4, 8, 16]
    datasets = ['ARI', 'SONICOM']
    for dataset in datasets:
        if args.mode == 'barycentric_baseline' or args.mode == 'sh_baseline':
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
    parser.add_argument("--lap")
    args = parser.parse_args()

    if args.hpc == "True":
        hpc = True
    elif args.hpc == "False":
        hpc = False
    else:
        raise RuntimeError("Please enter 'True' or 'False' for the hpc tag (-c/--hpc)")

    if args.mode == 'projection':
        run_projection(hpc, args.test, args.lap)
    elif args.mode == 'preprocess':
        run_preprocess(hpc, args.type, args.test,  args.lap)
    elif args.mode == 'train':
        run_train(hpc, args.type, args.test, args.lap)
    elif args.mode == 'evaluation':
        run_evaluation(hpc, int(args.exp), args.type, args.test, args.lap)
    elif args.mode == 'plot':
        plot_evaluation(hpc, int(args.exp), args.type)
    elif args.mode == 'barycentric_baseline' or args.mode == 'hrtf_selection_baseline' or args.mode == 'sh_baseline':
        run_baseline(hpc, args.test)
    else:
        print('Please specify a valid mode')