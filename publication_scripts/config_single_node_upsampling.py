import json
from pathlib import Path


class Config:
    """Config class

    Set using HPC to true in order to use appropriate paths for HPC
    """

    def __init__(self, tag, using_hpc, dataset=None, existing_model_tag=None, data_dir=None, runs_folder=None, temporary_runs_path=None):

        # overwrite settings with arguments provided
        self.tag = tag if tag is not None else 'pub-prep-upscale-sonicom-sonicom-synthetic-tl-2'
        self.dataset = dataset if dataset is not None else 'SONICOM'
        self.data_dir = data_dir if data_dir is not None else '/data/' + self.dataset
        self.runs_folder = runs_folder if runs_folder is not None else '/runs-hpc'

        if existing_model_tag is not None:
            self.start_with_existing_model = True
        else:
            self.start_with_existing_model = False

        self.existing_model_tag = existing_model_tag if existing_model_tag is not None else None

        # Data processing parameters
        self.merge_flag = True
        self.gen_sofa_flag = False
        self.nbins_hrtf = 128  # make this a power of 2
        self.hrtf_size = 16
        self.panel = 1  # panel used to select point when upscale_factor is 80
        self.upscale_factor = 1  # can only take values: 2, 4 ,8, 16
        self.train_samples_ratio = 0.8
        self.hrir_samplerate = 48000.0
        self.single_panel = False

        # Data dirs
        if using_hpc:
            self.ngpu = 1
            # HPC data dirs
            self.data_dirs_path = '/rds/general/user/aos13/home/HRTF-upsampling-with-a-generative-' \
                                  'adversarial-network-using-a-gnomonic-equiangular-projection'
            self.raw_hrtf_dir = Path('/rds/general/project/sonicom/live/HRTF Datasets')
            self.amt_dir = '/rds/general/user/aos13/home/HRTF-GANs-27Sep22-prep-for-publication/thirdParty'
        else:
            self.ngpu = 0  # Don't use GPU locally
            # local data dirs
            self.data_dirs_path = '/home/ahogg/PycharmProjects/HRTF-GAN'
            self.raw_hrtf_dir = Path('/home/ahogg/Documents/HRTF Datasets')
            self.amt_dir = '/home/ahogg/PycharmProjects/HRTF-GAN/thirdParty'

        self.baseline_dir = '/baseline_results/' + self.dataset

        if self.single_panel == True:
            self.data_dir += '/single_panel'
            self.baseline_dir += '/single_panel'
            self.runs_folder += '-single-panel'
        else:
            self.data_dir += '/cube_sphere'
            self.baseline_dir += '/cube_sphere'

        if temporary_runs_path is None:
            self.path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}'
            self.existing_model_path = f'{self.data_dirs_path}{self.runs_folder}/{self.existing_model_tag}'

            self.valid_path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}/valid'
            self.model_path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}'
        else:
            self.path = f'{temporary_runs_path}{self.runs_folder}/{self.tag}'
            self.existing_model_path = f'{temporary_runs_path}{self.runs_folder}/{self.existing_model_tag}'

            self.valid_path = f'{temporary_runs_path}{self.runs_folder}/{self.tag}/valid'
            self.model_path = f'{temporary_runs_path}{self.runs_folder}/{self.tag}'

        self.projection_dir = f'{self.data_dirs_path}/projection_coordinates'

        self.train_hrtf_dir = self.data_dirs_path + self.data_dir + '/hr/train'
        self.valid_hrtf_dir = self.data_dirs_path + self.data_dir + '/hr/valid'
        self.train_original_hrtf_dir = self.data_dirs_path + self.data_dir + '/original/train'
        self.valid_original_hrtf_dir = self.data_dirs_path + self.data_dir + '/original/valid'

        self.train_hrtf_merge_dir = self.data_dirs_path + self.data_dir + '/hr_merge/train'
        self.valid_hrtf_merge_dir = self.data_dirs_path + self.data_dir + '/hr_merge/valid'
        self.train_original_hrtf_merge_dir = self.data_dirs_path + self.data_dir + '/merge_original/train'
        self.valid_original_hrtf_merge_dir = self.data_dirs_path + self.data_dir + '/merge_original/valid'

        self.mean_std_filename = self.data_dirs_path + self.data_dir + '/mean_std_' + self.dataset
        self.barycentric_hrtf_dir = self.data_dirs_path + self.baseline_dir + '/barycentric/valid'
        self.hrtf_selection_dir = self.data_dirs_path + self.baseline_dir + '/hrtf_selection/valid'

        # Training hyperparams
        self.batch_size = 1
        self.num_workers = 1
        self.num_epochs = 300  # was originally 250
        self.lr_gen = 0.0002
        self.lr_dis = 0.0000015
        # how often to train the generator
        self.critic_iters = 4

        # Loss function weight
        self.content_weight = 0.01
        self.adversarial_weight = 0.01

        # betas for Adam optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999

        if self.ngpu > 0:
            self.device_name = "cuda:0"
        else:
            self.device_name = 'cpu'

    def save(self):
        j = {}
        for k, v in self.__dict__.items():
            j[k] = v
        with open(f'{self.path}/config.json', 'w') as f:
            json.dump(list(j), f)

    def load(self):
        with open(f'{self.path}/config.json', 'r') as f:
            j = json.load(f)
            for k, v in j.items():
                setattr(self, k, v)

    def get_train_params(self):
        return self.batch_size, self.beta1, self.beta2, self.num_epochs, self.lr_gen, self.lr_dis, self.critic_iters
