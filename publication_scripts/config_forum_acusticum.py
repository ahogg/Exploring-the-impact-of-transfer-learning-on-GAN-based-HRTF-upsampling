import json
from pathlib import Path


class Config:
    """Config class

    Set using HPC to true in order to use appropriate paths for HPC
    """

    def __init__(self, tag, using_hpc, dataset=None, existing_model_tag=None):


        self.dataset = 'SONICOMSynthetic'
        # self.dataset = 'ARI'
        self.tag = 'pub-prep-upscale-sonicom-sonicom-synthetic-tl-2'

        self.start_with_existing_model = False
        self.existing_model_tag = 'pub-prep-upscale-sonicom-synthetic'

        # overwrite settings with arguments provided
        if tag is not None:
            self.tag = tag
        if dataset is not None:
            self.dataset = dataset
        if existing_model_tag is not None:
            self.start_with_existing_model = True
            self.existing_model_tag = existing_model_tag

        # Data processing parameters
        self.merge_flag = True
        self.gen_sofa_flag = True
        self.hrtf_size = 16
        self.upscale_factor = 2  # can only take values: 2, 4 ,8, 16
        self.train_samples_ratio = 0.8
        self.hrir_samplerate = 48000.0

        # Data dirs
        if using_hpc:
            # HPC data dirs
            self.data_dirs_path = '/rds/general/user/aos13/home/HRTF-upsampling-with-a-generative-' \
                                  'adversarial-network-using-a-gnomonic-equiangular-projection'
            self.raw_hrtf_dir = Path('/rds/general/project/sonicom/live/HRTF Datasets')
        else:
            # local data dirs
            self.data_dirs_path = '/home/aos13/HRTF-GANs-27Sep22-prep-for-publication'
            self.raw_hrtf_dir = Path('/home/aos13/HRTF_datasets')

        self.runs_folder = '/runs-hpc'
        self.path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}'
        self.existing_model_path = f'{self.data_dirs_path}{self.runs_folder}/{self.existing_model_tag}'

        self.valid_path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}/valid'
        self.model_path = f'{self.data_dirs_path}{self.runs_folder}/{self.tag}'

        self.projection_filename = f'{self.data_dirs_path}/projection_coordinates/{self.dataset}_projection_{self.hrtf_size}'
        self.data_dir = '/data/' + self.dataset
        self.baseline_dir = '/baseline_results/' + self.dataset
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

        # Training hyperparams
        self.batch_size = 4
        self.num_workers = 4
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

        self.ngpu = 1
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