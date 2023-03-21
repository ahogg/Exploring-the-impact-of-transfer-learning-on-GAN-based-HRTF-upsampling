from model.util import spectral_distortion_metric
from model.dataset import downsample_hrtf
import glob
import torch
import pickle
import os
import re
import numpy as np

def run_lsd_evaluation(config, sr_dir, plot_fig=False):
    sr_data_paths = glob.glob('%s/%s_*' % (sr_dir, config.dataset))
    sr_data_file_names = ['/' + os.path.basename(x) for x in sr_data_paths]

    lsd_errors = []
    for file_name in sr_data_file_names:
        with open(config.valid_hrtf_merge_dir + file_name, "rb") as f:
            hr_hrtf = pickle.load(f)

        with open(sr_dir + file_name, "rb") as f:
            sr_hrtf = pickle.load(f)

        lr_hrtf = torch.permute(
            downsample_hrtf(torch.permute(hr_hrtf, (3, 0, 1, 2)), config.hrtf_size, config.upscale_factor),
            (1, 2, 3, 0))

        lr = lr_hrtf.detach().cpu()
        for p in range(5):
            for w in range(config.hrtf_size):
                for h in range(config.hrtf_size):
                    if hr_hrtf[p, w, h] in lr:
                        sr_hrtf[p, w, h] = hr_hrtf[p, w, h]

        generated = torch.permute(sr_hrtf[:, None], (1, 4, 0, 2, 3))
        target = torch.permute(hr_hrtf[:, None], (1, 4, 0, 2, 3))

        error = spectral_distortion_metric(generated, target)
        subject_id = ''.join(re.findall(r'\d+', file_name))
        lsd_errors.append([subject_id,  float(error.detach())])
        print('LSD Error of subject %s: %0.4f' % (subject_id, float(error.detach())))
    print('Mean LSD Error: %0.3f' % np.mean([error[1] for error in lsd_errors]))
    with open(f'{config.path}/lsd_errors.pickle', "wb") as file:
        pickle.dump(lsd_errors, file)

