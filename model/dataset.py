import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from preprocessing.convert_coordinates import convert_cube_indices_to_single_panel_indices


# based on https://github.com/Lornatang/SRGAN-PyTorch/blob/7292452634137d8f5d4478e44727ec1166a89125/dataset.py
def downsample_hrtf(hr_hrtf, hrtf_size, upscale_factor, panel=0):
    # downsample hrtf

    if len(hr_hrtf.size()) == 3:  # Single panel

        # selected_cube_indices = []
        # for panel in np.arange(0, 5):
        #     for p in np.arange(0, 16, upscale_factor):
        #         for q in np.arange(0, 16, upscale_factor):
        #             selected_cube_indices.append([panel, p, q])
        #
        # single_panel_indices = convert_cube_indices_to_single_panel_indices(selected_cube_indices, hrtf_size)

        if upscale_factor == 80:
            lr_hrtf = torch.nn.functional.interpolate(hr_hrtf[None, :], scale_factor=1 / 16)[0][:, panel, :, None]
        elif upscale_factor == 40:
            lr_hrtf = torch.nn.functional.interpolate(hr_hrtf[None, :], scale_factor=1 / 16)[0][:, panel, :]
        else:
            # lr_hrtf = torch.nn.functional.interpolate(hr_hrtf[None, :], scale_factor=1 / upscale_factor)[0]
            lr_hrtf = torch.from_numpy(
                np.moveaxis(np.array(torch.from_numpy(np.moveaxis(np.array(hr_hrtf), 0, -1))[np.ix_(
                    np.arange((upscale_factor/2)-1, np.shape(hr_hrtf)[1], upscale_factor),
                    np.arange((upscale_factor/2-1), np.shape(hr_hrtf)[2], upscale_factor))]), -1, 0))
    else:
        if upscale_factor == hrtf_size*5:
            mid_pos = int(hrtf_size / 2)
            lr_hrtf = hr_hrtf[:, panel, mid_pos, mid_pos, None, None, None]
        elif upscale_factor == hrtf_size*2.5:
            mid_pos = int(hrtf_size / 2)
            lr_hrtf = torch.tensor(np.concatenate((hr_hrtf[:, panel[0], mid_pos, mid_pos, None, None, None], hr_hrtf[:, panel[1], mid_pos, mid_pos, None, None, None]), axis=1))
        elif upscale_factor == hrtf_size:
            mid_pos = int(hrtf_size / 2)
            lr_hrtf = hr_hrtf[:, :, mid_pos, mid_pos, None, None]
        else:
            lr_hrtf = torch.nn.functional.interpolate(hr_hrtf, scale_factor=1 / upscale_factor)

    return lr_hrtf

class TrainValidHRTFDataset(Dataset):
    """Define training/valid dataset loading methods.
    Args:
        hrtf_dir (str): Train/Valid dataset address.
        hrtf_size (int): High resolution hrtf size.
        upscale_factor (int): hrtf up scale factor.
        transform (callable): A function/transform that takes in an HRTF and returns a transformed version.
    """

    def __init__(self, hrtf_dir: str, hrtf_size: int, upscale_factor: int, panel: int, transform=None, run_validation=True) -> None:
        super(TrainValidHRTFDataset, self).__init__()
        # Get all hrtf file names in folder
        self.hrtf_file_names = [os.path.join(hrtf_dir, hrtf_file_name) for hrtf_file_name in os.listdir(hrtf_dir)
                                if os.path.isfile(os.path.join(hrtf_dir, hrtf_file_name))]

        if run_validation:
            print('Running Validation')
            valid_hrtf_file_names = []
            for hrtf_file_name in self.hrtf_file_names:
                file = open(hrtf_file_name, 'rb')
                hrtf = pickle.load(file)
                if not np.isnan(np.sum(hrtf.cpu().data.numpy())):
                    valid_hrtf_file_names.append(hrtf_file_name)
                #     if all(map(lambda i: isinstance(i, np.floating), np.array(hrtf.cpu().data.numpy().ravel()))):
                #         if np.logical_and(hrtf.cpu().data.numpy() > 0, hrtf.cpu().data.numpy() < 2).all():
                #             valid_hrtf_file_names.append(hrtf_file_name)
                #         else:
                #             count = 0
                #             err_count = 0
                #             for i in range(hrtf.cpu().data.numpy().shape[0]):
                #                 for j in range(hrtf.cpu().data.numpy().shape[1]):
                #                     for k in range(hrtf.cpu().data.numpy().shape[2]):
                #                         count += 1
                #                         if not np.logical_and(hrtf.cpu().data.numpy()[i, j, k, :] > 0, hrtf.cpu().data.numpy()[i, j, k, :] < 2).all():
                #                             err_count += 1
                #                             print(f'Number of elements of of range: {np.logical_and(hrtf.cpu().data.numpy()[i, j, k, :] > 0, hrtf.cpu().data.numpy()[i, j, k, :] < 2).sum()}')
                #             print(f'Number of HRTFs in total: {count}')
                #             print(f'Number of HRTFs out of range: {err_count}')
                #             print(f'Minimum value: {min(np.array(hrtf.cpu().data.numpy().ravel()))}')
                #             print(f'Maximum value: {max(np.array(hrtf.cpu().data.numpy().ravel()))}')
                #             print(f'{hrtf_file_name} discarded due to impulse response not being in range')
                #     else:
                #         print(f'{hrtf_file_name} discarded due to impulse response not being stored as floats')
                # else:
                #     print(f'{hrtf_file_name} discarded due to nan')

            self.hrtf_file_names = valid_hrtf_file_names

        # Specify the high-resolution hrtf size, with equal length and width
        self.hrtf_size = hrtf_size
        # How many times the high-resolution hrtf is the low-resolution hrtf
        self.upscale_factor = upscale_factor
        # transform to be applied to the data
        self.transform = transform
        # panel used to select point when upscale_factor is 80
        self.panel = panel

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of hrtf data
        with open(self.hrtf_file_names[batch_index], "rb") as file:
            hrtf = pickle.load(file)

        # hrtf processing operations
        if self.transform is not None:
            # If using a transform, treat panels as batch dim such that dims are (panels, channels, X, Y)
            hr_hrtf = torch.permute(hrtf, (0, 3, 1, 2))
            # Then, transform hr_hrtf to normalize and swap panel/channel dims to get channels first
            hr_hrtf = torch.permute(self.transform(hr_hrtf), (1, 0, 2, 3))
        else:
            # If no transform, go directly to (channels, ..., X, Y)
            hr_hrtf = torch.moveaxis(hrtf, -1, 0)

        # downsample hrtf
        lr_hrtf = downsample_hrtf(hr_hrtf, self.hrtf_size, self.upscale_factor,  self.panel)

        return {"lr": lr_hrtf, "hr": hr_hrtf, "filename": self.hrtf_file_names[batch_index]}

    def __len__(self) -> int:
        return len(self.hrtf_file_names)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
