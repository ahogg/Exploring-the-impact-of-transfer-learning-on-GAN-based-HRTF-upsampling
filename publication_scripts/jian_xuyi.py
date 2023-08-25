import os
import numpy as np
import importlib
import sys
import torch
from decimal import *
getcontext().prec = 28
import scipy

from preprocessing.utils import remove_itd

import matplotlib.pyplot as plt

import torch.nn.functional as F
from model.util import spectral_distortion_metric

from scipy.special import sph_harm

import jax.numpy as jp
import jax
jax.config.update("jax_enable_x64", True)

sys.path.append('/rds/general/user/aos13/home/HRTF-upsampling-with-a-generative-adversarial-network-using-a-gnomonic-equiangular-projection/')

from publication_scripts.config_single_node_upsampling import Config


def spectral_distortion_inner(input_spectrum, target_spectrum):
    numerator = target_spectrum
    denominator = input_spectrum
    return np.mean((20 * np.log10(numerator / denominator)) ** 2)


def plot_Y(order, degree, sph_harmonics, grid, selection_mask):
    # Create a 2-D meshgrid of (theta, phi) angles.
    masked_grid = np.ma.array(grid, mask=np.column_stack((selection_mask, selection_mask, selection_mask))).reshape(
        selection_mask.shape[0], selection_mask.shape[1], 3)
    theta = masked_grid[:, :, 1]
    phi = masked_grid[:, :, 0]
    r = masked_grid[:, :, 2]

    # Calculate the Cartesian coordinates of each point in the mesh.
    xyz = np.array(r * [np.sin(phi) * np.cos(theta),
                        np.sin(phi) * np.sin(theta),
                        -np.cos(phi)])

    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')

    """Plot the spherical harmonic of degree el and order m on Axes ax."""

    Y = sph_harmonics[:, (order ** 2) + degree].reshape(xyz.shape[1:])
    Yx, Yy, Yz = np.abs(Y) * xyz

    # Colour the plotted surface according to the sign of Y.
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'))
    cmap.set_clim(-0.5, 0.5)

    ax.plot_surface(Yx, Yy, Yz,
                    facecolors=cmap.to_rgba(Y.real),
                    rstride=2, cstride=2)
    # ax.scatter(xyz[0], xyz[1], xyz[2])
    # ax.scatter(Yx, Yy, Yz)

    # Draw a set of x, y, z axes for reference.
    ax_lim = 0.5
    ax.plot([-ax_lim, ax_lim], [0, 0], [0, 0], c='0.5', lw=1, zorder=10)
    ax.plot([0, 0], [-ax_lim, ax_lim], [0, 0], c='0.5', lw=1, zorder=10)
    ax.plot([0, 0], [0, 0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    # Set the Axes limits and title, turn off the Axes frame.
    ax.set_title(r'$Y_{{{},{}}}$'.format(order, degree))
    ax_lim = 0.5
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis('off')
    plt.show()


class SphericalHarmonicsTransform:

    def __init__(self, max_degree, row_angles, column_angles, radii, selection_mask, coordinate_system='spherical', PLOT_FLAG=False):

        # row_angles = np.linspace(-180, 180, 100, endpoint=False)
        # column_angles = np.linspace(-90, 90, 100, endpoint=False)
        # selection_mask = np.zeros((len(row_angles), len(column_angles), 1), dtype=bool)

        self.grid = np.stack(np.meshgrid(row_angles, column_angles, radii, indexing='ij'), axis=-1)
        if coordinate_system == 'spherical':
            # elevations, azimuths, radii -> azimuths, elevations, radii
            self.grid[..., 0], self.grid[..., 1] = np.deg2rad(self.grid[..., 1]), np.deg2rad(self.grid[..., 0])
        elif coordinate_system == 'interaural':
            # lateral, vertical, radius -> azimuths, elevations, radii
            self.grid[..., 0], self.grid[..., 1], self.grid[..., 2] = interaural2spherical(self.grid[..., 0], self.grid[..., 1], self.grid[..., 2],
                                                                            out_angles_as_degrees=False)
        else:
            # X, Y, Z -> azimuths, elevations, radii
            self.grid[..., 0], self.grid[..., 1], self.grid[..., 2] = cartesian2spherical(self.grid[..., 0], self.grid[..., 1], self.grid[..., 2],
                                                                           out_angles_as_degrees=False)
        # Convert elevations to zeniths, azimuths, elevations, radii
        self.grid[..., 1] = np.pi + self.grid[..., 1]
        self.grid[..., 0] = np.pi / 2 + self.grid[..., 0]


        self.selected_angles = self.grid[~selection_mask]
        # self._harmonics = np.column_stack(
        #     [np.real(sph_harm(order, degree, self.selected_angles[:, 1], self.selected_angles[:, 0])) for degree in
        #      np.arange(max_degree + 1) for order in np.arange(-degree, degree + 1)])
        self._harmonics = np.transpose(
            [np.real(sph_harm(order, degree, self.selected_angles[:, 1], self.selected_angles[:, 0]))
             if order == 0 else
             np.sqrt(2) * (-1.0)**order * np.real(sph_harm(order, degree, self.selected_angles[:, 1], self.selected_angles[:, 0]))
             if order > 0 else
             np.sqrt(2) * (-1.0)**order * np.imag(sph_harm(order, degree, self.selected_angles[:, 1], self.selected_angles[:, 0]))
             for degree in
             np.arange(max_degree + 1) for order in np.arange(-degree, degree + 1)])

        # Plot single harmonic
        if PLOT_FLAG:
            [plot_Y(order, degree, self._harmonics, self.grid,  selection_mask) for order in np.arange(max_degree + 1) for degree in np.arange(2*order + 1)]


        self._valid_mask = ~selection_mask

    def __call__(self, hrirs):
        # max_degree = int(np.sqrt(np.shape(self._harmonics)[1]) - 1)
        # coef = np.linalg.lstsq(self._harmonics, hrirs[self._valid_mask].data, rcond=None)[0]
        self._harmonics_inv = scipy.linalg.pinv(self._harmonics)
        coef = self._harmonics_inv @ hrirs[self._valid_mask].data
        # coef = self._harmonics_inv @ np.swapaxes(hrirs.data, 1, 0)[np.swapaxes(self._valid_mask, 1, 0)].data
        #

        # [plot_Y(order, degree, self._harmonics, self.grid, ~self._valid_mask) for order in
        #  np.arange(max_degree + 1) for
        #  degree in np.arange(2 * order + 1)]
        return coef

    def inverse(self, coefficients):
        return self._harmonics @ coefficients

    def get_harmonics(self):
        return self._harmonics

    def get_grid(self):
        return self.grid

    def get_selected_angles(self):
        return self.selected_angles


dataset = 'SONICOM'
config = Config('debug', using_hpc=False, dataset=dataset, data_dir='/data/' + dataset)

PI_4 = np.pi / 4

data_dir = config.raw_hrtf_dir / config.dataset
print(os.getcwd())
print(config.dataset)

imp = importlib.import_module('hrtfdata.full')
load_function = getattr(imp, config.dataset)


data_dir = config.raw_hrtf_dir / config.dataset
print(os.getcwd())
print(config.dataset)

imp = importlib.import_module('hrtfdata.full')
load_function = getattr(imp, config.dataset)

domain = 'time'

left_hrtf = load_function(data_dir, features_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                            'side': 'left', 'domain': domain}},  subject_ids='first')

right_hrtf = load_function(data_dir, features_spec={'hrirs': {'samplerate': config.hrir_samplerate,
                                                             'side': 'right', 'domain': domain}},  subject_ids='first')
left_ids = left_hrtf.subject_ids
right_ids = right_hrtf.subject_ids

valid_dir = config.valid_path
# valid_gt_dir = config.valid_gt_path/
# shutil.rmtree(Path(valid_dir), ignore_errors=True)
# Path(valid_dir).mkdir(parents=True, exist_ok=True)
# shutil.rmtree(Path(valid_gt_dir), ignore_errors=True)
# Path(valid_gt_dir).mkdir(parents=True, exist_ok=True)
# min_list = []
# for sample_id in list(left_ids):
#     sample_id -= 1
#     left = left_hrtf[sample_id]['features'][:, :, :, 1:]
#     right = right_hrtf[sample_id]['features'][:, :, :, 1:]
#     merge = np.ma.concatenate([left, right], axis=3)
#     merge = torch.from_numpy(merge.data)
#     min_list.append(torch.min(merge))
# print(min_list)
# print("avg min: ", np.average(min_list))

for order in [80]:

    sample_id = 0
    if domain == 'time':
        left = left_hrtf[sample_id]['features'][:, :, :, :]
        right = right_hrtf[sample_id]['features'][:, :, :, :]
        left = np.array([[[remove_itd(x[0], int(len(x[0]) * 0.04), len(x[0]))] for x in y] for y in left])
        right = np.array([[[remove_itd(x[0], int(len(x[0]) * 0.04), len(x[0]))] for x in y] for y in right])
    else:
        left = left_hrtf[sample_id]['features'][:, :, :, 1:]
        right = right_hrtf[sample_id]['features'][:, :, :, 1:]

    merge = np.ma.concatenate([left, right], axis=3)
    # merge = right

    # super_threshold_indices = merge < 0.001
    # merge[super_threshold_indices] = 0.001
    # merge = merge[:, :, :, 0, None]
    mask = np.ones((len(left_hrtf.row_angles), len(left_hrtf.column_angles), 1), dtype=bool)
    original_mask = np.all(np.ma.getmaskarray(left), axis=3)

    row_ratio = 3
    col_ratio = 3
    for i in range(len(left_hrtf.row_angles) // row_ratio):
        for j in range(len(left_hrtf.column_angles) // col_ratio):
            # print(f'index: ({row_ratio * i}, {col_ratio * j})')
            mask[row_ratio * i, col_ratio * j, :] = original_mask[row_ratio * i, col_ratio * j, :]

    # for (i, j) in [(x, y) for x in range(0, 72) for y in range(0, 12)]:
    # for (i, j) in [(x, 0) for x in range(0, 72)]:
    #     mask[i, j, :] = original_mask[i, j, :]
    #     print(f'angle - row: {left_hrtf.row_angles[i]}, col: {left_hrtf.column_angles[j]}')

    # for TF in merge[~mask].data:
    #     # ir_id = int(ir_id)
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    #     ax1.plot(TF)
    #     # ax1.plot(left.reshape(864, 256)[ir_id])
    #     # ax1.set_title(f'left (ID: {ir_id})')
    #     # ax2.plot(right.reshape(864, 256)[ir_id])
    #     # ax2.set_title(f'right (ID: {ir_id})')
    #     # ax3.plot(merge.reshape(864, 512)[ir_id])
    #     # ax3.set_title(f'merge (ID: {ir_id})')
    #     plt.show()

    print(f'Number of points in input: {len(merge[~mask].data)}')
    print(f'Number of points in output: {len(merge[~original_mask].data)}')

    # SHT
    order = int(np.ceil(np.sqrt(len(merge[~mask].data)))) - 1
    print(f'Order: {order}')
    SHT = SphericalHarmonicsTransform(order, left_hrtf.row_angles, left_hrtf.column_angles, left_hrtf.radii, mask)
    sh_coef = SHT(merge)
    print("coef: ", sh_coef.shape, sh_coef.dtype)

    # inverse SHT
    SHT_orig = SphericalHarmonicsTransform(order, left_hrtf.row_angles, left_hrtf.column_angles, left_hrtf.radii, original_mask, PLOT_FLAG=False)
    harmonics_orig = SHT_orig.get_harmonics()
    harmonics = SHT.get_harmonics()
    print("harmonics shape: ", harmonics_orig.shape, harmonics_orig.dtype)
    inverse_orig = harmonics_orig @ sh_coef
    inverse = harmonics @ sh_coef
    print("inverse: ", inverse.shape)

    ori_hrtf = merge[~original_mask].data
    recon_hrtf = inverse_orig

    def plot_tf(ir_id):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.plot(ori_hrtf[ir_id])
        ax1.set_title(f'original (ID: {ir_id})')
        ax2.plot(recon_hrtf[ir_id])
        ax2.set_title(f'recon (ID {ir_id})')

        plt.show()

    total_positions = len(ori_hrtf)
    total_all_positions = 0

    total_sd_metric = 0

    ir_id = 0
    max_value = None
    max_id = None
    min_value = None
    min_id = None

    for ori, gen in zip(ori_hrtf, recon_hrtf):
        if domain == 'magnitude_db':
            ori = 10 ** (ori/20)
            gen = 10 ** (gen/20)

        if domain == 'magnitude_db' or domain == 'magnitude':
            average_over_frequencies = spectral_distortion_inner(abs(gen), abs(ori))
            total_all_positions += np.sqrt(average_over_frequencies)
        elif domain == 'time':
            nbins = 128
            ori_tf_left = abs(scipy.fft.rfft(ori[:nbins], nbins*2)[1:])
            ori_tf_right = abs(scipy.fft.rfft(ori[nbins:], nbins*2)[1:])
            gen_tf_left = abs(scipy.fft.rfft(gen[:nbins], nbins*2)[1:])
            gen_tf_right = abs(scipy.fft.rfft(gen[nbins:], nbins*2)[1:])

            ori_tf = np.ma.concatenate([ori_tf_left, ori_tf_right])
            gen_tf = np.ma.concatenate([gen_tf_left, gen_tf_right])

            average_over_frequencies = spectral_distortion_inner(gen_tf, ori_tf)
            total_all_positions += np.sqrt(average_over_frequencies)

        # print('Log SD (for %s position): %s' % (ir_id, np.sqrt(average_over_frequencies)))
        if max_value is None or np.sqrt(average_over_frequencies) > max_value:
            max_value = np.sqrt(average_over_frequencies)
            max_id = ir_id
        if min_value is None or np.sqrt(average_over_frequencies) < min_value:
            min_value = np.sqrt(average_over_frequencies)
            min_id = ir_id
        ir_id += 1

    sd_metric = total_all_positions / total_positions
    total_sd_metric += sd_metric

    print('Min Log SD (for %s position): %s' % (min_id, min_value))
    print('Max Log SD (for %s position): %s' % (max_id, max_value))

    PLOT_FLAG = True
    if PLOT_FLAG == True:
        plot_tf(min_id)
        plot_tf(max_id)


    print('Log SD (across all positions): %s' % float(sd_metric))




