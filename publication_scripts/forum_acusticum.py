import argparse
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import pickle
import numpy as np

from publication_scripts.config_forum_acusticum import Config
from model.test import test
from model.util import load_dataset
from evaluation.evaluation import run_lsd_evaluation, run_localisation_evaluation, run_target_localisation_evaluation
from main import main

plt.rcParams['legend.fancybox'] = False


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


def plot_boxplot(config, name, ylabel, full_results, legend, colours):
    plt.rc('font', family='serif', serif='Times New Roman')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

    fig, ax = plt.subplots()

    factors = [2, 4, 8, 16]
    ticks = [
        r'$%s \,{\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}} %s$' % (int(
            (16 / factor) ** 2 * 5), int(config.hrtf_size ** 2 * 5)) for factor in factors]

    for idx, full_result in enumerate(full_results):
        data = np.vstack((full_result[3], full_result[2], full_result[1], full_result[0]))

        blp = plt.boxplot(data.T, positions=np.array(range(len(data))) * 1.0 - np.linspace(0.15*(len(full_results)/2), -0.15*(len(full_results)/2), len(full_results))[idx],
                                                flierprops=dict(marker='x', markeredgecolor=colours[idx], markersize=4), widths=0.12)

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
        'Upsample Factor\n' + r'(No. of original nodes$\ {\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}}$ No. of upsampled nodes)')

    plt.ylabel(ylabel)
    if len(full_results) > 2:
        plt.xticks(range(0, len(ticks), 1), ticks)
        plt.xlim(-0.5, len(ticks) - 0.5)
    else:
        plt.xticks(range(0, len(ticks) * 2, 2), ticks)

    # Append nans to results to make them of equal length
    maxlen = np.max([[len(j) for j in i] for i in full_results])
    for full_result in full_results:
        for result in full_result:
            if (maxlen - len(result)) > 0:
                result[:] = [np.nan] * (maxlen - len(result)) + result

    ymin = np.nanmin(full_results) - 0.1 * abs(np.nanmax(full_results) - np.nanmin(full_results))
    ymax = np.nanmax(full_results) + 0.1 * abs(np.nanmax(full_results) - np.nanmin(full_results))
    ax.set_ylim((ymin, ymax))
    ax.yaxis.set_label_coords(-0.12, 0.5)

    w = 2.974
    h = w / 1.5
    fig.set_size_inches(w, h)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    fig.savefig(config.data_dirs_path + '/plots/' + name, bbox_inches='tight')


def get_results(tag, mode, file_ext=None):
    full_results = []
    upscale_factors = [16, 8, 4, 2]
    for upscale_factor in upscale_factors:
        config = Config(tag + str(upscale_factor), using_hpc=hpc)
        if mode == 'lsd' or mode == 'baseline_lsd':
            if mode == 'lsd':
                file_ext = 'lsd_errors.pickle' if file_ext is None else file_ext
                file_path = f'{config.path}/{file_ext}'
            elif mode == 'baseline_lsd':
                file_path = f'{tag}/{file_ext}{upscale_factor}.pickle'
            with open(file_path, 'rb') as file:
                lsd_id_errors = pickle.load(file)
            lsd_errors = [lsd_error[1] for lsd_error in lsd_id_errors]
            print(f'Loading: {file_path}')
            print('Mean (STD) LSD: %0.3f (%0.3f)' % (np.mean(lsd_errors),  np.std(lsd_errors)))
            full_results.append(lsd_errors)
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

    return full_results

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

def run_train(hpc, type, test_id=None):
    print(f'Running training')
    config_files = []
    tags = []
    upscale_factors = [2, 4, 8, 16]
    datasets = ['ARI', 'SONICOM', 'SONICOMSynthetic']
    if type == 'tl' or type == 'base':
        datasets.remove('SONICOMSynthetic')
    for dataset in datasets:
        other_dataset = 'ARI' if dataset == 'SONICOM' else 'SONICOM'
        for upscale_factor in upscale_factors:
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
                    config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data/' + dataset)
                elif type == 'base-tl':
                    config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, data_dir='/data-transfer-learning/' + dataset)
                elif type == 'tl':
                    config = Config(tag['tag'], using_hpc=hpc, dataset=dataset, existing_model_tag=tag['existing_model_tag'], data_dir='/data/' + dataset)
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


def run_evaluation(hpc, experiment_id, type, test_id=None):
    print(f'Running {type} experiment {experiment_id}')
    config_files = []
    if experiment_id == 1:
        upscale_factors = [2, 4, 8, 16]
        datasets = ['ARI', 'SONICOM']
        for dataset in datasets:
            for upscale_factor in upscale_factors:
                tags = [{'tag': f'pub-prep-upscale-{dataset}-{upscale_factor}'},
                        {'tag': f'pub-prep-upscale-{dataset}-SONICOMSynthetic-tl-{upscale_factor}'}]
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
            full_results_dataset_baseline = get_results(f'{config.data_dirs_path}/baseline_results/{dataset.upper()}/barycentric/valid', mode=f'baseline_{mode}', file_ext=f'{mode}_errors_barycentric_interpolated_data_')
            if mode == 'lsd':
                legend = ['SRGAN (No TL)', 'TL (Synthetic)', f'TL ({other_dataset})', 'Baseline']
                colours = ['#0047a4', '#af211a', 'g', '#6C0BA9', '#E67E22']
                # remove baseline results at upscale-16
                # full_results_dataset_baseline[0] = np.full(shape=len(full_results_dataset_baseline[-1]), fill_value=np.nan).tolist()
                #######################################
                create_table(legend, [full_results_dataset, full_results_dataset_sonicom_synthetic_tl, full_results_dataset_dataset_tl, full_results_dataset_baseline], dataset.upper(), units='[dB]')
                plot_boxplot(config, f'LSD_boxplot_ex_{experiment_id}_{dataset}', f'{dataset.upper()} \n LSD error [dB]', [full_results_dataset, full_results_dataset_sonicom_synthetic_tl, full_results_dataset_dataset_tl, full_results_dataset_baseline], legend, colours)
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
                full_results_dataset_target_tl = get_results(config.data_dirs_path + '/data/' + dataset.upper(), 'target', f'{dataset.upper()}_loc_target_valid_errors.pickle')*4
                for i in np.arange(np.shape(full_results_dataset)[1]):
                    plot_boxplot(config, f'{types[i]}_boxplot_ex_{experiment_id}_{dataset}', labels[i], [np.array(full_results_dataset)[:, i, :],
                                np.array(full_results_dataset_sonicom_synthetic_tl)[:, i, :], np.array(full_results_dataset_dataset_tl)[:, i, :], np.array(full_results_dataset_baseline)[:, i, :], np.array(full_results_dataset_target_tl)[:, i, :]], legend, colours)
                    print(f'Generate table containing {types[i]} errors for the {dataset.upper()} dataset: \n')
                    create_table(legend, [np.array(full_results_dataset)[:, i, :],
                                np.array(full_results_dataset_sonicom_synthetic_tl)[:, i, :], np.array(full_results_dataset_dataset_tl)[:, i, :], np.array(full_results_dataset_baseline)[:, i, :], [np.array(full_results_dataset_target_tl)[0, i, :]]], dataset.upper(), units=units[i])
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
    elif args.mode == 'evaluation':
        run_evaluation(hpc, int(args.exp), args.type, args.test)
    elif args.mode == 'plot':
        plot_evaluation(hpc, int(args.exp), args.type)
    elif args.mode == 'barycentric_baseline' or args.mode == 'hrtf_selection_baseline' or args.mode == 'sh_baseline':
        run_baseline(hpc, args.test)
    else:
        print('Please specify a valid mode')