import argparse
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import pickle
import numpy as np

from config import Config
from model.test import test
from model.util import load_dataset
from evaluation.evaluation import run_lsd_evaluation, run_localisation_evaluation, run_target_localisation_evaluation

plt.rcParams['legend.fancybox'] = False


def get_means(full_results):
    upsample_means = []
    upsample_stds = []
    for full_result in full_results:
        upsample_means.insert(0, np.mean(full_result))
        upsample_stds.insert(0, np.std(full_result))

    return upsample_means, upsample_stds

def create_table(legend, full_results, side_title=None):
    factors = [2, 4, 8, 16]
    single_line = r"\hhline{-~----}" if side_title is None else r"\hhline{~-~----}"
    double_lines = r"\hhline{=~====}" if side_title is None else r"\hhline{~=~====}"
    title_lines = r"\hhline{~~----}" if side_title is None else r"\hhline{~~~----}"
    extra_column_1 = r"" if side_title is None else r" & "
    extra_column_2 = r"" if side_title is None else r"c"


    ticks = [r' \textbf{%s} $\,\rightarrow$ \textbf{1280}' % int((16 / factor) ** 2 * 5) for factor in factors]

    print(r"\begin{tabular}{%s|c|c @{\hspace{-0.3\tabcolsep}}|c|c|c|c|}" % extra_column_2)
    print(single_line)
    print(extra_column_1 +
        r"\multirow{2}{*}{\textbf{Method}} & & \multicolumn{4}{c|}{\textbf{Upsample Factor [No. orginal  $\,\rightarrow$ No. upsampled]}} \\" + title_lines)
    print(extra_column_1 + r"& & \multicolumn{1}{c|}{" + ticks[0] + r"} & \multicolumn{1}{c|}{" + ticks[
        1] + r"} & \multicolumn{1}{c|}{" + ticks[2] + r"} & \multicolumn{1}{c|}{" + ticks[3] + r"} \\ " + double_lines)
    if side_title is not None:
        print(r"\parbox[t]{3.5mm}{\multirow{%s}{*}{\rotatebox[origin=c]{90}{\textbf{%s}}}}" % (len(full_results), side_title))
    for idx, full_result in enumerate(full_results):
        full_result_means, full_result_stds = get_means(full_result)
        if len(full_result_means) == 4:
            print(extra_column_1 +
                r"\textbf{%s} & & %.2f (%.2f) & %.2f (%.2f)  & %.2f (%.2f)  & %.2f (%.2f)   \\ %s" % tuple(
                    [legend[idx]] + [val for pair in zip(full_result_means, full_result_stds) for val in pair] + [single_line]))
        if len(full_result_means) == 1:
            print(extra_column_1 + r"\textbf{%s} & & \multicolumn{4}{c|}{%.2f (%.2f)}   \\ %s" % tuple(
                    [legend[idx]] + [val for pair in zip(full_result_means, full_result_stds) for val in pair] + [single_line]))

    print(r"\end{tabular}")
    print('\n')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color, linewidth=0.5)
    plt.setp(bp['whiskers'], color=color, linewidth=0.5)
    plt.setp(bp['caps'], color=color, linewidth=0.5)
    plt.setp(bp['medians'], color=color, linewidth=0.5)


def plot_boxplot(config, name, ylabel, full_results, legend):
    plt.rc('font', family='serif', serif='Times New Roman')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    fig, ax = plt.subplots()

    factors = [2, 4, 8, 16]
    ticks = [
        r'$%s \,{\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}} 1280$' % int(
            (16 / factor) ** 2 * 5) for factor in factors]

    colours = ['#0047a4', '#af211a', 'g', '#6C0BA9']
    for idx, full_result in enumerate(full_results):
        data = np.vstack((full_result[3], full_result[2], full_result[1], full_result[0]))

        blp = plt.boxplot(data.T, positions=np.array(range(len(data))) * 1.0 - np.linspace(0.15*(len(full_results)/2), -0.15*(len(full_results)/2), len(full_results))[idx],
                                                flierprops=dict(marker='x', markeredgecolor=colours[idx], markersize=4), widths=0.17)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(blp[element], color=colours[idx], linewidth=0.7)

        plt.plot([], c=colours[idx], label=legend[idx])

    if len(full_results) > 2:
        [plt.axvline(x + 0.5, color='#a6a6a6', linestyle='--', linewidth=0.5) for x in range(0, (len(ticks) - 1), 1)]
    else:
        [plt.axvline(x + 1, color='#a6a6a6', linestyle='--', linewidth=0.5) for x in range(0, (len(ticks) - 1) * 2, 2)]

    leg = ax.legend(prop={'size': 7}, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower right",  # mode="expand",
                        borderaxespad=0, ncol=2, handlelength=1.06)

    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor('k')

    ax.yaxis.grid(zorder=0, linewidth=0.4)
    plt.xlabel(
        'Upsampling \n' + r'[No. of orginal nodes$\ {\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}}$ No. of upsampled nodes]')

    plt.ylabel(ylabel)
    if len(full_results) > 2:
        plt.xticks(range(0, len(ticks), 1), ticks)
        plt.xlim(-0.5, len(ticks) - 0.5)
    else:
        plt.xticks(range(0, len(ticks) * 2, 2), ticks)

    w = 2.974
    h = w / 1.5
    fig.set_size_inches(w, h)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    fig.savefig(config.data_dirs_path + '/plots/' + name)


def get_results(tag, mode, file_ext=None):
    full_results = []
    upscale_factors = [16, 8, 4, 2]
    for upscale_factor in upscale_factors:
        config = Config(tag + str(upscale_factor), using_hpc=hpc)
        if mode == 'lsd':
            file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext
            file_path = f'{config.path}/{file_ext}'
            with open(file_path, 'rb') as file:
                lsd_id_errors = pickle.load(file)
            lsd_errors = [lsd_error[1] for lsd_error in lsd_id_errors]
            print(f'Loading: {file_path}')
            print('Mean LSD: %s' % np.mean(lsd_errors))
            print('STD LSD: %s' % np.std(lsd_errors))
            full_results.append(lsd_errors)
        elif mode == 'localisation' or 'target':
            file_ext = 'loc_errors.pickle' if file_ext is None else file_ext
            if mode == 'localisation':
                file_path = f'{config.path}/{file_ext}'
            elif mode == 'target' :
                file_path = tag + '/' + file_ext

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

    return full_results

def run_evaluation(hpc, experiment_id, type, test_id=None):

    print(f'Running {type} experiment {experiment_id}')
    config_files = []
    if experiment_id == 1:
        upscale_factors = [2, 4, 8, 16]
        datasets = ['ari', 'sonicom']
        for dataset in datasets:
            for upscale_factor in upscale_factors:

                tags = [f'pub-prep-upscale-{dataset}-{upscale_factor}',
                        f'pub-prep-upscale-{dataset}-sonicom-synthetic-tl-{upscale_factor}']
                for tag in tags:
                    config = Config(tag, using_hpc=hpc)
                    config.upscale_factor = upscale_factor
                    config.dataset = dataset.upper()
                    config.valid_hrtf_merge_dir = f'{config.data_dirs_path}/data/{config.dataset}/hr_merge/valid'
                    config_files.append(config)
                    print(f'{len(config_files)} config files created successfully.')

    elif experiment_id == 2:
        upscale_factors = [2, 4, 8, 16]
        datasets = ['ari', 'sonicom']
        for dataset in datasets:
            other_dataset = 'ari' if dataset == 'sonicom' else 'sonicom'
            for upscale_factor in upscale_factors:
                tags = [f'pub-prep-upscale-{dataset}-{upscale_factor}',
                        f'pub-prep-upscale-{dataset}-{other_dataset}-tl-{upscale_factor}',
                        f'pub-prep-upscale-{dataset}-sonicom-synthetic-tl-{upscale_factor}']
                for tag in tags:
                    config = Config(tag, using_hpc=hpc)
                    config.upscale_factor = upscale_factor
                    config.dataset = dataset.upper()
                    config.valid_hrtf_merge_dir = f'{config.data_dirs_path}/data/{config.dataset}/hr_merge/valid'
                    config_files.append(config)
        print(f'{len(config_files)} config files created successfully.')

    elif experiment_id == 3:
        datasets = ['ari', 'sonicom']
        for dataset in datasets:
            tag = None
            config = Config(tag, using_hpc=hpc)
            config.dataset = dataset.upper()
            config.data_dir = '/data/' + config.dataset
            config.valid_hrtf_merge_dir = config.data_dirs_path + config.data_dir + '/hr_merge/valid'

            config_files.append(config)
        print(f'{len(config_files)} config files created successfully.')

    else:
        print('Experiment does not exist')
        return

    if test_id is not None:
        test_id = int(test_id)
        config_files = [config_files[test_id]]

    print(f'Running a total of {len(config_files)} config files')
    for config in config_files:
        if experiment_id == 3:
            run_target_localisation_evaluation(config)
        elif type == 'lsd':
            _, test_prefetcher = load_dataset(config, mean=None, std=None)
            print("Loaded all datasets successfully.")
            test(config, test_prefetcher)
            run_lsd_evaluation(config, config.valid_path)
        elif type == 'localisation':
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
        datasets = ['ari', 'sonicom']
        for dataset in datasets:
            if mode == 'lsd':
                full_results_LSD_dataset = get_results(f'pub-prep-upscale-{dataset}-', mode)
                full_results_LSD_dataset_sonicom_synthetic_tl = get_results(f'pub-prep-upscale-{dataset}-sonicom-synthetic-tl-', mode)
                legend = ['SRGAN', 'SRGAN TL (Synthetic)']
                plot_boxplot(config, f'LSD_boxplot_ex_1_{dataset}', 'LSD error [dB]', [full_results_LSD_dataset, full_results_LSD_dataset_sonicom_synthetic_tl], legend)
            elif mode == 'localisation':
                full_results_loc_dataset = get_results(f'pub-prep-upscale-{dataset}-', mode)
                full_results_loc_dataset_sonicom_synthetic_tl = get_results(f'pub-prep-upscale-{dataset}-sonicom-synthetic-tl-', mode)
                types = ['ACC', 'RMS', 'QUERR']
                labels = [r'Polar accuracy error [$^\circ$]', r'Polar RMS error [$^\circ$]', 'Quadrant error [\%]']
                for i in np.arange(np.shape(full_results_loc_dataset)[1]):
                    legend = ['SRGAN', 'SRGAN TL (Synthetic)']
                    plot_boxplot(config, f'{types[i]}_boxplot_ex_1_{dataset}', labels[i], [np.array(full_results_loc_dataset)[:, i, :],
                                np.array(full_results_loc_dataset_sonicom_synthetic_tl)[:, i, :]], legend)

    elif experiment_id == 2:
        datasets = ['ari', 'sonicom']
        for dataset in datasets:
            other_dataset = 'ari' if dataset == 'sonicom' else 'sonicom'
            full_results_dataset = get_results(f'pub-prep-upscale-{dataset}-', mode)
            full_results_dataset_sonicom_synthetic_tl = get_results(f'pub-prep-upscale-{dataset}-sonicom-synthetic-tl-', mode)
            full_results_dataset_dataset_tl = get_results(f'pub-prep-upscale-{dataset}-{other_dataset}-tl-', mode)
            if mode == 'lsd':
                legend = ['SRGAN', 'TL (Synthetic)', 'TL (Real)']
                plot_boxplot(config, f'LSD_boxplot_ex_2_{dataset}', 'LSD error [dB]', [full_results_dataset, full_results_dataset_sonicom_synthetic_tl, full_results_dataset_dataset_tl], legend)
                create_table(legend, [full_results_dataset, full_results_dataset_sonicom_synthetic_tl, full_results_dataset_dataset_tl])
            elif mode == 'localisation':
                types = ['ACC', 'RMS', 'QUERR']
                labels = [r'Polar accuracy error [$^\circ$]', r'Polar RMS error [$^\circ$]', 'Quadrant error [\%]']
                legend = ['SRGAN', 'TL (Synthetic)', 'TL (Real)', 'Target']
                full_results_dataset_target_tl = get_results(config.data_dirs_path + '/data/' + dataset.upper(), 'target', f'{dataset.upper()}_loc_target_valid_errors.pickle')*4
                for i in np.arange(np.shape(full_results_dataset)[1]):
                    plot_boxplot(config, f'{types[i]}_boxplot_ex_1_{dataset}', labels[i], [np.array(full_results_dataset)[:, i, :],
                                np.array(full_results_dataset_sonicom_synthetic_tl)[:, i, :], np.array(full_results_dataset_dataset_tl)[:, i, :], np.array(full_results_dataset_target_tl)[:, i, :]], legend)
                    print(f'Generate table containing {types[i]} errors for the {dataset} dataset: \n')
                    create_table(legend, [np.array(full_results_dataset)[:, i, :],
                                np.array(full_results_dataset_sonicom_synthetic_tl)[:, i, :], np.array(full_results_dataset_dataset_tl)[:, i, :], [np.array(full_results_dataset_target_tl)[0, i, :]]], dataset.upper())
    else:
        print('Experiment does not exist')


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

    # Note that experiment_id=3 is always of localisation type
    if args.mode == 'evaluation':
        run_evaluation(hpc, int(args.exp), args.type, args.test)
    elif args.mode == 'plot':
        plot_evaluation(hpc, int(args.exp), args.type)
    else:
        print('Please specify a valid mode')