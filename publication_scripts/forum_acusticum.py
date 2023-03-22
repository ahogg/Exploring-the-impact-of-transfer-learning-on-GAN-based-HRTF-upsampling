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

def create_table(full_results_SD_SRGAN, full_results_SD_SRGAN_synthetic, full_results_SD_SRGAN_Real):
    upsample_SD_SRGAN_means, upsample_SD_SRGAN_stds = get_means(full_results_SD_SRGAN)
    upsample_SD_SRGAN_synthetic_means, upsample_SD_SRGAN_synthetic_stds = get_means(full_results_SD_SRGAN_synthetic)
    upsample_SD_SRGAN_real_means, upsample_SD_SRGAN_real_stds = get_means(full_results_SD_SRGAN_Real)

    factors = [2, 4, 8, 16]
    ticks = [r' \textbf{%s} $\,\rightarrow$ \textbf{1280}' % int((16 / factor) ** 2 * 5) for factor in factors]

    print(r"\begin{tabular}{|c|c @{\hspace{-0.3\tabcolsep}}|c|c|c|c|}")
    print(r"\hhline{-~----}")
    print(
        r"\multirow{2}{*}{\textbf{Method}} & & \multicolumn{4}{c|}{\textbf{Upsample Factor [No. orginal  $\,\rightarrow$ No. upsampled]}}                                                               \\ \cline{3-6}")
    print(r"                        & & \multicolumn{1}{c|}{" + ticks[0] + r"} & \multicolumn{1}{c|}{" + ticks[
        1] + r"} & \multicolumn{1}{c|}{" + ticks[2] + r"} & \multicolumn{1}{c|}{" + ticks[3] + r"} \\ \hhline{=~====}")
    print(
        r"\textbf{SRGAN}             & & %.2f (%.2f) & %.2f (%.2f)  & %.2f (%.2f)  & %.2f (%.2f)   \\ \hhline{-~----}" % tuple(
            [val for pair in zip(upsample_SD_SRGAN_means, upsample_SD_SRGAN_stds) for val in pair]))
    print(
        r"\textbf{SRGAN TL (synthetic)}       & & %.2f (%.2f) & %.2f (%.2f)  & %.2f (%.2f)  & %.2f (%.2f)   \\ \hhline{=~====}" % tuple(
            [val for pair in zip(upsample_SD_SRGAN_synthetic_means, upsample_SD_SRGAN_synthetic_stds) for val in pair]))
    print(
        r"\textbf{SRGAN TL (real)}       & & %.2f (%.2f) & %.2f (%.2f)  & %.2f (%.2f)  & %.2f (%.2f)   \\ \hhline{=~====}" % tuple(
            [val for pair in zip(upsample_SD_SRGAN_real_means, upsample_SD_SRGAN_real_stds) for val in pair]))

    print(r"\end{tabular}")
    print('\n')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color, linewidth=0.5)
    plt.setp(bp['whiskers'], color=color, linewidth=0.5)
    plt.setp(bp['caps'], color=color, linewidth=0.5)
    plt.setp(bp['medians'], color=color, linewidth=0.5)


def plot_boxplot(config, name, ylabel, full_results_SRGAN, full_results_Barycentric, full_results_Raw=None):
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
    # print(ticks)
    # ticks = ['2','4','8','16']

    upsample_16 = full_results_SRGAN[0]
    upsample_8 = full_results_SRGAN[1]
    upsample_4 = full_results_SRGAN[2]
    upsample_2 = full_results_SRGAN[3]

    upsample_16_b = full_results_Barycentric[0]
    upsample_8_b = full_results_Barycentric[1]
    upsample_4_b = full_results_Barycentric[2]
    upsample_2_b = full_results_Barycentric[3]

    if full_results_Raw != None:
        upsample_16_r = full_results_Raw[0]
        upsample_8_r = full_results_Raw[1]
        upsample_4_r = full_results_Raw[2]
        upsample_2_r = full_results_Raw[3]

    data_a = np.vstack((upsample_2, upsample_4, upsample_8, upsample_16))

    data_b = np.vstack((upsample_2_b, upsample_4_b, upsample_8_b, upsample_16_b))

    if full_results_Raw != None:
        data_c = np.vstack((upsample_2_r, upsample_4_r, upsample_8_r, upsample_16_r))

    colour_2 = '#0047a4'
    colour_1 = '#af211a'
    colour_3 = 'g'

    if full_results_Raw == None:
        bpl = plt.boxplot(data_a.T, positions=np.array(range(len(data_a))) * 1.0 - 0.15,
                          flierprops=dict(marker='x', markeredgecolor=colour_1, markersize=4), widths=0.17)
        bpm = plt.boxplot(data_b.T, positions=np.array(range(len(data_b))) * 1.0 + 0.15,
                          flierprops=dict(marker='x', markeredgecolor=colour_2, markersize=4), widths=0.17)
    else:
        bpl = plt.boxplot(data_a.T, positions=np.array(range(len(data_a))) * 2.0 - 0.4,
                          flierprops=dict(marker='x', markeredgecolor=colour_1, markersize=4), widths=0.3)
        bpm = plt.boxplot(data_b.T, positions=np.array(range(len(data_b))) * 2.0,
                          flierprops=dict(marker='x', markeredgecolor=colour_2, markersize=4), widths=0.3)
        bpr = plt.boxplot(data_c.T, positions=np.array(range(len(data_c))) * 2.0 + 0.4,
                          flierprops=dict(marker='x', markeredgecolor=colour_3, markersize=4), widths=0.3)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bpl[element], color=colour_1, linewidth=0.7)
        plt.setp(bpm[element], color=colour_2, linewidth=0.7)
        if full_results_Raw != None:
            plt.setp(bpr[element], color=colour_3, linewidth=0.7)

    if full_results_Raw == None:
        [plt.axvline(x + 0.5, color='#a6a6a6', linestyle='--', linewidth=0.5) for x in range(0, (len(ticks) - 1), 1)]
    else:
        [plt.axvline(x + 1, color='#a6a6a6', linestyle='--', linewidth=0.5) for x in range(0, (len(ticks) - 1) * 2, 2)]

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c=colour_1, label='SRGAN')
    plt.plot([], c=colour_2, label='SRGAN TL (Synthetic)')
    if full_results_Raw != None:
        plt.plot([], c=colour_3, label='SRGAN TL (Real)')

    if full_results_Raw == None:
        leg = ax.legend(prop={'size': 7}, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower right",  # mode="expand",
                        borderaxespad=0, ncol=3, handlelength=1.06)
    else:
        leg = ax.legend(prop={'size': 7}, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower right",  # mode="expand",
                        borderaxespad=0, ncol=2, handlelength=1.06)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor('k')

    ax.yaxis.grid(zorder=0, linewidth=0.4)
    plt.xlabel(
        'Upsampling \n' + r'[No. of orginal nodes$\ {\mathrel{\vcenter{\hbox{\rule[-.2pt]{4pt}{.4pt}}} \mkern-4mu\hbox{\usefont{U}{lasy}{m}{n}\symbol{41}}}}$ No. of upsampled nodes]')

    plt.ylabel(ylabel)
    if full_results_Raw == None:
        plt.xticks(range(0, len(ticks), 1), ticks)
        plt.xlim(-0.5, len(ticks) - 0.5)
    else:
        plt.xticks(range(0, len(ticks) * 2, 2), ticks)

    w = 2.974
    h = w / 1.5
    fig.set_size_inches(w, h)
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    fig.savefig(config.data_dirs_path + '/plots/' + name)


def get_results(tag, mode):
    full_results = []
    upscale_factors = [16, 8, 4, 2]
    for upscale_factor in upscale_factors:
        config = Config(tag + str(upscale_factor), using_hpc=hpc)
        if mode == 'lsd':
            file_path = f'{config.path}/lsd_errors.pickle'
            with open(file_path, 'rb') as file:
                lsd_id_errors = pickle.load(file)
            lsd_errors = [lsd_error[1] for lsd_error in lsd_id_errors]
            print(f'Loading: {file_path}')
            print('Mean LSD: %s' % np.mean(lsd_errors))
            print('STD LSD: %s' % np.std(lsd_errors))
            full_results.append(lsd_errors)
        elif mode == 'localisation':
            file_path = f'{config.path}/loc_errors.pickle'
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

    return full_results




def run_evaluation(hpc, experiment_id, mode, test_id=None):

    print(f'Running {mode} experiment {experiment_id}')
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
                    config.dataset = 'ARI'
                    config.valid_hrtf_merge_dir = f'{config.data_dirs_path}/data/{config.dataset}/hr_merge/valid'
                    config_files.append(config)
        print(f'{len(config_files)} config files created successfully.')

    elif experiment_id == 3:
        datasets = ['ari', 'sonicom']
        for dataset in datasets:
            tag = None
            config = Config(tag, using_hpc=hpc)
            config.dataset = dataset
            config.valid_hrtf_merge_dir = f'{config.data_dirs_path}/data/{config.dataset}/hr_merge/valid'
            config_files.append(config)
        print(f'{len(config_files)} config files created successfully.')

    else:
        print('Experiment does not exist')
        return

    if test_id is not None:
        config_files = [config_files[test_id]]

    print(f'Running a total of {len(config_files)} config files')
    for config in config_files:
        if experiment_id == 3:
            run_target_localisation_evaluation(config, config.valid_path)
        elif mode == 'lsd':
            _, test_prefetcher = load_dataset(config, mean=None, std=None)
            print("Loaded all datasets successfully.")
            test(config, test_prefetcher)
            run_lsd_evaluation(config, config.valid_path)
        elif mode == 'localisation':
            _, test_prefetcher = load_dataset(config, mean=None, std=None)
            print("Loaded all datasets successfully.")
            test(config, test_prefetcher)
            run_localisation_evaluation(config, config.valid_path)



def plot_evaluation(hpc, experiment_id, mode):
    tag = None
    config = Config(tag, using_hpc=hpc)

    if experiment_id == 1:
        datasets = ['ari', 'sonicom']
        for dataset in datasets:
            if mode == 'lsd':
                full_results_LSD_dataset = get_results(f'pub-prep-upscale-{dataset}-', mode)
                full_results_LSD_dataset_sonicom_synthetic_tl = get_results(f'pub-prep-upscale-{dataset}-sonicom-synthetic-tl-', mode)
                plot_boxplot(config, f'LSD_boxplot_ex_1_{dataset}', 'LSD error [dB]', full_results_LSD_dataset, full_results_LSD_dataset_sonicom_synthetic_tl)
            elif mode == 'localisation':
                full_results_loc_dataset = get_results(f'pub-prep-upscale-{dataset}-', mode)
                full_results_loc_dataset_sonicom_synthetic_tl = get_results(f'pub-prep-upscale-{dataset}-sonicom-synthetic-tl-', mode)
                types = ['ACC', 'RMS', 'QUERR']
                labels = ['Polar accuracy', 'Polar RMS', 'Quadrant']
                for i in np.arange(np.shape(full_results_loc_dataset)[1]):
                    plot_boxplot(config, f'{types[i]}_boxplot_ex_1_{dataset}', f'{labels[i]} error [dB]', np.array(full_results_loc_dataset)[:, i, :],
                                np.array(full_results_loc_dataset_sonicom_synthetic_tl)[:, i, :])

    elif experiment_id == 2:
        if mode == 'lsd':
            datasets = ['ari', 'sonicom']
            for dataset in datasets:
                other_dataset = 'ari' if dataset == 'sonicom' else 'sonicom'
                full_results_LSD_dataset = get_results(f'pub-prep-upscale-{dataset}-', mode)
                full_results_LSD_dataset_sonicom_synthetic_tl = get_results(f'pub-prep-upscale-{dataset}-sonicom-synthetic-tl-', mode)
                full_results_LSD_dataset_dataset_tl = get_results(f'pub-prep-upscale-{dataset}-{other_dataset}-tl-', mode)
                plot_boxplot(config, f'LSD_boxplot_ex_2_{dataset}', 'LSD error [dB]', full_results_LSD_dataset, full_results_LSD_dataset_sonicom_synthetic_tl, full_results_LSD_dataset_dataset_tl)
                create_table(full_results_LSD_dataset, full_results_LSD_dataset_sonicom_synthetic_tl, full_results_LSD_dataset_dataset_tl)


    else:
        print('Experiment does not exist')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-c", "--hpc")
    parser.add_argument("exp")
    parser.add_argument("type")
    parser.add_argument("test")
    args = parser.parse_args()

    if args.hpc == "True":
        hpc = True
    elif args.hpc == "False":
        hpc = False
    else:
        raise RuntimeError("Please enter 'True' or 'False' for the hpc tag (-c/--hpc)")


    # Note that experiment_id=3 does not have a mode
    if args.mode == 'Evaluation':
        run_evaluation(hpc, args.exp, args.type, args.test)
    elif args.mode == 'Plot':
        plot_evaluation(hpc, args.exp, args.type)
    else:
        print('Please specify a valid mode')