import cmath
import pickle
import os

import sofar as sf
import numpy as np
import torch
import scipy
import math
from scipy.signal import hilbert
import scipy.signal as sps
import shutil
from pathlib import Path
import re
import scipy.signal as sn


from preprocessing.barycentric_calcs import calc_barycentric_coordinates, get_triangle_vertices
from preprocessing.convert_coordinates import convert_cube_to_sphere
from preprocessing.KalmanFilter import KalmanFilter

PI_4 = np.pi / 4


def clear_create_directories(config):
    """Clear/Create directories"""
    if config.lap_factor:
        shutil.rmtree(Path(config.train_lap_dir), ignore_errors=True)
        shutil.rmtree(Path(config.valid_lap_dir), ignore_errors=True)
        Path(config.train_lap_dir).mkdir(parents=True, exist_ok=True)
        Path(config.valid_lap_dir).mkdir(parents=True, exist_ok=True)
        shutil.rmtree(Path(config.train_lap_original_hrtf_dir), ignore_errors=True)
        shutil.rmtree(Path(config.valid_lap_original_hrtf_dir), ignore_errors=True)
        Path(config.train_lap_original_hrtf_dir).mkdir(parents=True, exist_ok=True)
        Path(config.valid_lap_original_hrtf_dir).mkdir(parents=True, exist_ok=True)
    else:
        shutil.rmtree(Path(config.train_hrtf_dir), ignore_errors=True)
        shutil.rmtree(Path(config.valid_hrtf_dir), ignore_errors=True)
        Path(config.train_hrtf_dir).mkdir(parents=True, exist_ok=True)
        Path(config.valid_hrtf_dir).mkdir(parents=True, exist_ok=True)
        shutil.rmtree(Path(config.train_original_hrtf_dir), ignore_errors=True)
        shutil.rmtree(Path(config.valid_original_hrtf_dir), ignore_errors=True)
        Path(config.train_original_hrtf_dir).mkdir(parents=True, exist_ok=True)
        Path(config.valid_original_hrtf_dir).mkdir(parents=True, exist_ok=True)


def merge_left_right_hrtfs(input_dir, output_dir):
    # Clear/Create directory
    shutil.rmtree(Path(output_dir), ignore_errors=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    hrtf_file_names = [os.path.join(input_dir, hrtf_file_name) for hrtf_file_name in os.listdir(input_dir)
                       if os.path.isfile(os.path.join(input_dir, hrtf_file_name))]

    data_dict_left = {}
    data_dict_right = {}
    for f in hrtf_file_names:

        file_ext = re.findall(re.escape(input_dir) + '/(.*)_[0-9]*[a-z]*.pickle$', f)[0]

        with open(f, "rb") as file:
            data = pickle.load(file)

        # add to dict for right ears
        if re.search(re.escape(input_dir)+'/.*_[0-9]*right.pickle$', f):
            subj_id = int(re.findall(re.escape(input_dir)+'/.*_([0-9]*)right.pickle$', f)[0])
            if file_ext not in data_dict_right:
                data_dict_right[file_ext] = {}
            data_dict_right[file_ext][subj_id] = data
        # add to dict for left ears
        elif re.search(re.escape(input_dir)+'/.*_[0-9]*left.pickle$', f):
            subj_id = int(re.findall(re.escape(input_dir)+'/.*_([0-9]*)left.pickle$', f)[0])
            if file_ext not in data_dict_left:
                data_dict_left[file_ext] = {}
            data_dict_left[file_ext][subj_id] = data

    for file_ext in data_dict_right.keys():
        missing_subj_ids =  list(set(data_dict_right[file_ext].keys()) - set(data_dict_left[file_ext].keys()))
        if len(missing_subj_ids) > 0:
            print('Excluding subject IDs where both ears do not exist (IDs: %s)' % ', '.join(map(str, missing_subj_ids)))
            for missing_subj_id in missing_subj_ids:
                data_dict_right[file_ext].pop(missing_subj_id, None)
                data_dict_left[file_ext].pop(missing_subj_id, None)

        for subj_id in data_dict_right[file_ext].keys():
            hrtf_r = data_dict_right[file_ext][subj_id]
            hrtf_l = data_dict_left[file_ext][subj_id]
            dimension = hrtf_r.ndim-1
            hrtf_merged = torch.cat((hrtf_l, hrtf_r), dim=dimension)
            with open('%s/%s_%s.pickle' % (output_dir, file_ext, subj_id), "wb") as file:
                pickle.dump(hrtf_merged, file)


def merge_files(config):
    if config.lap_factor:
        merge_left_right_hrtfs(config.train_lap_dir, config.train_lap_merge_dir)
        merge_left_right_hrtfs(config.valid_lap_dir, config.valid_lap_merge_dir)
        merge_left_right_hrtfs(config.train_lap_original_hrtf_dir, config.train_lap_original_hrtf_merge_dir)
        merge_left_right_hrtfs(config.valid_lap_original_hrtf_dir, config.valid_lap_original_hrtf_merge_dir)
    else:
        merge_left_right_hrtfs(config.train_hrtf_dir, config.train_hrtf_merge_dir)
        merge_left_right_hrtfs(config.valid_hrtf_dir, config.valid_hrtf_merge_dir)
        merge_left_right_hrtfs(config.train_original_hrtf_dir, config.train_original_hrtf_merge_dir)
        merge_left_right_hrtfs(config.valid_original_hrtf_dir, config.valid_original_hrtf_merge_dir)


def get_hrtf_from_ds(config, ds, index, domain='mag'):
    coordinates = ds.fundamental_angles, ds.orthogonal_angles
    position_grid = np.stack(np.meshgrid(*coordinates, indexing='ij'), axis=-1)

    sphere_temp = []
    hrir_temps = []
    hrir_temps_no_ITD = []
    for row_idx, row in enumerate(ds.fundamental_angles):
        for column_idx, column in enumerate(ds.orthogonal_angles):
            if not any(np.ma.getmaskarray(ds[index]['features'][row_idx][column_idx].flatten())):
                az_temp = np.radians(position_grid[row_idx][column_idx][0])
                el_temp = np.radians(position_grid[row_idx][column_idx][1])
                sphere_temp.append([el_temp, az_temp])
                hrir_temp = np.ma.getdata(ds[index]['features'][row_idx][column_idx]).flatten()
                hrir_temps.append(hrir_temp)
                hrir_temps_no_ITD.append(remove_itd(hrir_temp, int(len(hrir_temp) * 0.04), len(hrir_temp)))

    if domain == 'mag':
        hrtf_temps_no_ITD, phase_temp = calc_hrtf(config, hrir_temps_no_ITD)
        return torch.tensor(np.array(hrtf_temps_no_ITD)), torch.tensor(np.array(phase_temp)), sphere_temp
    elif domain == 'time':
        return torch.tensor(np.array(hrir_temps)), sphere_temp


def calc_itd_r(config, ir_left, ir_right, az, el, c=343):

    az = az + 2*np.pi if az < 0 else az

    delay_left = remove_itd(ir_left, output_delay=True)
    delay_right = remove_itd(ir_right, output_delay=True)
    itd_samples = abs(delay_left-delay_right)

    itd_in_sec = itd_samples/config.hrir_samplerate
    interaural_azimuth = np.arcsin(np.sin(az) * np.cos(el))

    r = abs((itd_in_sec*c)/(interaural_azimuth + np.sin(interaural_azimuth)))

    return r


def add_itd(az, el, hrir, side, fs=48000, r=0.0875, c=343, onset=0, idt_increase=0):
    az = np.radians(az)
    el = np.radians(el)
    interaural_azimuth = np.arcsin(np.sin(az) * np.cos(el))
    delay_in_sec = (r / c) * (interaural_azimuth + np.sin(interaural_azimuth))

    if (delay_in_sec > 0 and side == 'right') or (delay_in_sec < 0 and side == 'left'):
        fractional_delay = abs(delay_in_sec * fs) + idt_increase + onset
    else:
        fractional_delay = onset

    N = len(hrir)

    B = 2*N
    n = np.arange(B)  # Filter length
    h = np.sinc(n - (B-1)/2 - fractional_delay)  # Compute sinc filter
    h *= np.blackman(B)  # Multiply sinc filter by window
    h /= np.sum(h)  # Normalize to get unity gain.
    delayed_hrir = np.convolve(hrir, h)
    delayed_hrir = delayed_hrir[int((B - 1) / 2):][:N]

    sofa_delay = fractional_delay

    return delayed_hrir, sofa_delay


def gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count, left_phase=None, right_phase=None):
    el = np.degrees(sphere_coords[count][0])
    az = np.degrees(sphere_coords[count][1])
    source_position = [az + 360 if az < 0 else az, el, 1.5]

    if left_phase is None:
        left_hrtf[left_hrtf == 0.0] = 1.0e-08
        left_phase = np.imag(hilbert(-np.log(np.abs(left_hrtf))))
    if right_phase is None:
        right_hrtf[right_hrtf == 0.0] = 1.0e-08
        right_phase = np.imag(hilbert(-np.log(np.abs(right_hrtf))))

    left_hrir = scipy.fft.irfft(np.concatenate((np.array([0]), np.abs(left_hrtf[:config.nbins_hrtf-1]))) * np.exp(1j * left_phase))[:config.nbins_hrtf]
    right_hrir = scipy.fft.irfft(np.concatenate((np.array([0]), np.abs(right_hrtf[:config.nbins_hrtf-1]))) * np.exp(1j * right_phase))[:config.nbins_hrtf]

    left_hrir_delay, left_sample_delay = add_itd(az, el, left_hrir, side='left', fs=config.hrir_samplerate, r=config.head_radius, onset = 10)
    right_hrir_delay, right_sample_delay = add_itd(az, el, right_hrir, side='right', fs=config.hrir_samplerate, r=config.head_radius, onset = 10)



    ##########################################
    desired_delay = int(abs(left_sample_delay - right_sample_delay))

    upper_cut_freq = 3000
    filter_order = 10
    fs = config.hrir_samplerate

    wn = upper_cut_freq / (fs / 2)
    b, a = sn.butter(filter_order, wn)
    loc_l = sn.lfilter(b, a, left_hrir_delay)
    loc_r = sn.lfilter(b, a, right_hrir_delay)

    # Take the maximum absolute value of the cross correlation between the two ears to get the maxiacc
    correlation = sn.correlate(np.abs(sn.hilbert(left_hrir_delay)), np.abs(sn.hilbert(right_hrir_delay)))
    idx_lag = np.argmax(np.abs(correlation))
    lag = abs(idx_lag - len(left_hrir_delay))

    # print(f'desired delay: {desired_delay}')
    # print(f'lag orginal: {lag}')

    if abs(lag) != desired_delay:
        if desired_delay == 0:
            idt_increase = desired_delay - lag
            left_hrir_delay, left_sample_delay = add_itd(az, el, left_hrir, side='left', fs=config.hrir_samplerate,
                                                         r=config.head_radius, onset=10+idt_increase)
            right_hrir_delay, right_sample_delay = add_itd(az, el, right_hrir, side='right', fs=config.hrir_samplerate,
                                                           r=config.head_radius, onset=10)
        else:
            idt_increase =  desired_delay - lag
            left_hrir_delay, left_sample_delay = add_itd(az, el, left_hrir, side='left', fs=config.hrir_samplerate,
                                                   r=config.head_radius, onset=10, idt_increase=idt_increase)
            right_hrir_delay, right_sample_delay = add_itd(az, el, right_hrir, side='right', fs=config.hrir_samplerate,
                                                     r=config.head_radius, onset=10, idt_increase=idt_increase)

    #     print(f'idt increase: {idt_increase}')
    #
    #     # Take the maximum absolute value of the cross correlation between the two ears to get the maxiacc
    #     correlation = sn.correlate(np.abs(sn.hilbert(left_hrir_delay)), np.abs(sn.hilbert(right_hrir_delay)))
    #     idx_lag = np.argmax(np.abs(correlation))
    #     lag = idx_lag - len(left_hrir_delay)
    #
    #     print(f'lag new: {lag}')
    #
    # if abs(lag - desired_delay) > 2:
    #         print('ITD failed')

    ##########################################

    full_hrir = [left_hrir_delay, right_hrir_delay]
    delay = [left_sample_delay, right_sample_delay]

    return source_position, full_hrir, delay


def save_sofa(clean_hrtf, config, cube_coords, sphere_coords, sofa_path_output, phase=None):
    full_hrirs = []
    source_positions = []
    delays = []
    left_full_phase = None
    right_full_phase = None
    if cube_coords is None:
        left_full_hrtf = clean_hrtf[:, :config.nbins_hrtf]
        right_full_hrtf = clean_hrtf[:, config.nbins_hrtf:]

        if phase is not None:
            left_full_phase = phase[:, :config.nbins_hrtf]
            right_full_phase = phase[:, config.nbins_hrtf:]

        for count in range(len(sphere_coords)):
            left_hrtf = np.array(left_full_hrtf[count])
            right_hrtf = np.array(right_full_hrtf[count])

            if phase is None:
                source_position, full_hrir, delay = gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count)
            else:
                left_phase = np.array(left_full_phase[count])
                right_phase = np.array(right_full_phase[count])
                source_position, full_hrir, delay = gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count, left_phase, right_phase)

            full_hrirs.append(full_hrir)
            source_positions.append(source_position)
            delays.append(delay)

    else:
        if len(np.shape(clean_hrtf)) == 2:
            left_full_hrtf = clean_hrtf[:, :config.nbins_hrtf]
            right_full_hrtf = clean_hrtf[:, config.nbins_hrtf:]

            for count in np.arange(len(clean_hrtf)):
                left_hrtf = np.array(left_full_hrtf[count])
                right_hrtf = np.array(right_full_hrtf[count])
                source_position, full_hrir, delay = gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count)
                full_hrirs.append(full_hrir)
                source_positions.append(source_position)
                delays.append(delay)

        else:
            left_full_hrtf = clean_hrtf[:, :, :, :config.nbins_hrtf]
            right_full_hrtf = clean_hrtf[:, :, :, config.nbins_hrtf:]

            count = 0
            for panel, x, y in cube_coords:
                # based on cube coordinates, get indices for magnitudes list of lists
                i = panel - 1
                j = round(config.hrtf_size * (x - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))
                k = round(config.hrtf_size * (y - (PI_4 / config.hrtf_size) + PI_4) / (np.pi / 2))

                left_hrtf = np.array(left_full_hrtf[i, j, k])
                right_hrtf = np.array(right_full_hrtf[i, j, k])
                source_position, full_hrir, delay = gen_sofa_file(config, sphere_coords, left_hrtf, right_hrtf, count)
                full_hrirs.append(full_hrir)
                source_positions.append(source_position)
                delays.append(delay)
                count += 1

    sofa = sf.Sofa("SimpleFreeFieldHRIR")
    sofa.Data_IR = full_hrirs
    sofa.Data_SamplingRate = config.hrir_samplerate
    sofa.Data_Delay = delays
    sofa.SourcePosition = source_positions
    sf.write_sofa(sofa_path_output, sofa)


def convert_to_sofa(hrtf_dir, config, cube, sphere, phase_ext='_phase', use_phase=False, mag_ext='_mag', lap_factor=None):
    if use_phase:
        sofa_path_output = hrtf_dir + '/sofa_with_phase/'
    else:
        sofa_path_output = hrtf_dir + '/sofa_min_phase/'

    hrtf_file_names = [hrtf_file_name for hrtf_file_name in os.listdir(hrtf_dir)
                        if os.path.isfile(os.path.join(hrtf_dir, hrtf_file_name)) and phase_ext not in hrtf_file_name]
    phase_file_names = [phase_file_name for phase_file_name in os.listdir(hrtf_dir)
                        if os.path.isfile(os.path.join(hrtf_dir, phase_file_name)) and phase_ext in phase_file_name]

    # Clear/Create directories
    shutil.rmtree(Path(sofa_path_output), ignore_errors=True)
    Path(sofa_path_output).mkdir(parents=True, exist_ok=True)

    for f in hrtf_file_names:
        with open(os.path.join(hrtf_dir, f), "rb") as hrtf_file:
            hrtf = pickle.load(hrtf_file)
            sofa_filename_output = os.path.basename(hrtf_file.name).replace('.pickle', '.sofa').replace(mag_ext,'')
            sofa_output = sofa_path_output + sofa_filename_output

            if lap_factor:
                if type(lap_factor) is not str:
                    config.lap_factor = re.search('_(.+?)_', f).group(1)

                edge_len = int(int(config.hrtf_size) / int(config.upscale_factor))
                projection_filename = f'{config.projection_dir}/{config.dataset}_projection_lap_{config.lap_factor}_{edge_len}'

                with open(projection_filename, "rb") as file:
                    cube_lap, sphere_lap, sphere_triangles_lap, sphere_coeffs_lap, measured_coords_lap = pickle.load(
                        file)

                if hrtf_dir == '/home/ahogg/PycharmProjects/HRTF-GAN/lap_results':
                    if 'SONICOM' in f:
                        sub_id = int(f.split('_')[-1].replace('.pickle', ''))
                        filename = f'{config.data_dirs_path}/lap_data/lap_full_pickle_original_{config.lap_factor}/SONICOM_mag_{sub_id}left.pickle'
                        with open(filename, 'rb') as f:
                            hrirs_left = pickle.load(f)
                        filename = f'{config.data_dirs_path}/lap_data/lap_full_pickle_original_{config.lap_factor}/SONICOM_mag_{sub_id}right.pickle'
                        with open(filename, 'rb') as f:
                            hrirs_right = pickle.load(f)
                    else:
                        sofa = sf.read_sofa(f'{config.data_dirs_path}/lap_data/{f.replace(".pickle", ".sofa")}')
                        hrirs = sofa.Data_IR
                        hrirs_left = hrirs[:, 1, :]
                        hrirs_right = hrirs[:, 0, :]

                    orginal_hrir = torch.tensor(np.concatenate((hrirs_left, hrirs_right), axis=1))
                else:
                #######################################################
                    with open(config.valid_lap_original_hrtf_merge_dir + '/' + f, "rb") as f:
                        orginal_hrir = pickle.load(f)

                orginal_hrir_left = orginal_hrir[:, :config.nbins_hrtf]
                orginal_hrir_right = orginal_hrir[:, config.nbins_hrtf:]

                rs = []
                for ir_index, measured_coord_lap in enumerate(measured_coords_lap):
                    if not math.isclose(measured_coord_lap[0], np.pi / 2, abs_tol=np.pi / 4) and not math.isclose(
                            measured_coord_lap[1], -np.pi, abs_tol=np.pi / 8) \
                            and not math.isclose(measured_coord_lap[1], np.pi, abs_tol=np.pi / 8) and not math.isclose(
                        measured_coord_lap[1], 0, abs_tol=np.pi / 8):
                        rs.append(calc_itd_r(config, orginal_hrir_left[ir_index], orginal_hrir_right[ir_index],
                                             az=measured_coords_lap[ir_index][1], el=measured_coords_lap[ir_index][0]))
                config.head_radius = np.mean([r for r in rs if math.isclose(r, 0.08, abs_tol=0.02)])

                #######################################################
            else:
                config.head_radius = 0.085


            if use_phase:
                for f_phase in phase_file_names:
                    f_phase = f_phase.replace(phase_ext, mag_ext)
                    if f_phase == f:
                        with open(os.path.join(hrtf_dir, f), "rb") as phase_file:
                            phase = pickle.load(phase_file)
                            save_sofa(hrtf, config, cube, sphere, sofa_output, phase)
            else:
                save_sofa(hrtf, config, cube, sphere, sofa_output)


def gen_sofa_preprocess(config, cube, sphere, sphere_original, edge_len=None, cube_lap=None, sphere_lap=None):
    if config.lap_factor:
        config.hrtf_size = edge_len
        convert_to_sofa(config.train_lap_merge_dir, config, cube_lap, sphere_lap)
        convert_to_sofa(config.valid_lap_merge_dir, config, cube_lap, sphere_lap)
        convert_to_sofa(config.train_lap_original_hrtf_merge_dir, config, cube_lap, sphere_lap)
        convert_to_sofa(config.valid_lap_original_hrtf_merge_dir, config, cube_lap, sphere_lap)
    else:
        convert_to_sofa(config.train_hrtf_merge_dir, config, cube, sphere)
        convert_to_sofa(config.valid_hrtf_merge_dir, config, cube, sphere)
        convert_to_sofa(config.train_original_hrtf_merge_dir, config, cube=None, sphere=sphere_original)
        convert_to_sofa(config.valid_original_hrtf_merge_dir, config, cube=None, sphere=sphere_original)
        convert_to_sofa(config.train_original_hrtf_merge_dir, config, use_phase=True, cube=None, sphere=sphere_original)
        convert_to_sofa(config.valid_original_hrtf_merge_dir, config, use_phase=True, cube=None, sphere=sphere_original)


def generate_euclidean_cube(config, measured_coords, edge_len=16, filename=None, output_measured_coords=False):
    """Calculate barycentric coordinates for projection based on a specified cube sphere edge length and a set of
    measured coordinates, finally save them to the file"""
    cube_coords, sphere_coords = [], []
    for panel in range(1, 6):
        for x in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
            for y in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
                x_i, y_i = x + PI_4 / edge_len, y + PI_4 / edge_len
                cube_coords.append((panel, x_i, y_i))
                sphere_coords.append(convert_cube_to_sphere(panel, x_i, y_i))

    euclidean_sphere_triangles = []
    euclidean_sphere_coeffs = []

    for count, p in enumerate(sphere_coords):
        triangle_vertices = get_triangle_vertices(elevation=p[0], azimuth=p[1], sphere_coords=measured_coords)
        coeffs = calc_barycentric_coordinates(elevation=p[0], azimuth=p[1], closest_points=triangle_vertices)
        euclidean_sphere_triangles.append(triangle_vertices)
        euclidean_sphere_coeffs.append(coeffs)

        print(f"Data point {count+1} out of {len(sphere_coords)} ({round(100 * count / len(sphere_coords))}%)")

    # save euclidean_cube, euclidean_sphere, euclidean_sphere_triangles, euclidean_sphere_coeffs
    Path(config.projection_dir).mkdir(parents=True, exist_ok=True)
    if filename == None:
        filepath = f'{config.projection_dir}/{config.dataset}_projection_{config.hrtf_size}'
    else:
        filepath = f'{config.projection_dir}/{config.dataset}_projection_{filename}'

    if output_measured_coords:
        with open(filepath, "wb") as file:
            pickle.dump((cube_coords, sphere_coords, euclidean_sphere_triangles, euclidean_sphere_coeffs, measured_coords), file)
    else:
        with open(filepath, "wb") as file:
            pickle.dump((cube_coords, sphere_coords, euclidean_sphere_triangles, euclidean_sphere_coeffs), file)


def save_euclidean_cube(edge_len=16):
    """Save euclidean cube as a txt file for use as input to matlab"""
    sphere_coords = []
    for panel in range(1, 6):
        for x in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
            for y in np.linspace(-PI_4, PI_4, edge_len, endpoint=False):
                x_i, y_i = x + PI_4 / edge_len, y + PI_4 / edge_len
                sphere_coords.append(convert_cube_to_sphere(panel, x_i, y_i))
    with open('../projection_coordinates/generated_coordinates.txt', 'w') as f:
        for coord in sphere_coords:
            print(coord)
            f.write(str(coord[0]))
            f.write(", ")
            f.write(str(coord[1]))
            f.write('\n')


def get_feature_for_point(elevation, azimuth, all_coords, subject_features):
    """For a given point (elevation, azimuth), get the associated feature value"""
    all_coords_row = all_coords.query(f'elevation == {elevation} & azimuth == {azimuth}')
    if len(np.shape(subject_features)) == 2:
        index = int(all_coords_row.index.values[0])
        return subject_features[index]
    else:
        azimuth_index = int(all_coords_row.azimuth_index.iloc[0])
        elevation_index = int(all_coords_row.elevation_index.iloc[0])
        return subject_features[azimuth_index][elevation_index]


def get_feature_for_point_tensor(elevation, azimuth, all_coords, subject_features, config=None):
    """For a given point (elevation, azimuth), get the associated feature value"""
    all_coords_row = all_coords.query(f'elevation == {elevation} & azimuth == {azimuth}')
    if len(np.shape(subject_features)) == 2:
        return scipy.fft.irfft(np.concatenate((np.array([0.0]), np.array(subject_features[(
        int(all_coords_row.index.values[0]))]))))
    elif len(np.shape(subject_features)) == 3:  # single panel
        return scipy.fft.irfft(np.concatenate((np.array([0.0]), np.array(subject_features[(int(all_coords_row.azimuth_index.values[0]), int(all_coords_row.elevation_index.values[0]))]))))
    else:
        return scipy.fft.irfft(np.concatenate((np.array([0.0]), np.array(
            subject_features[int(all_coords_row.panel.values[0] - 1)][int(all_coords_row.elevation_index.values[0])][
                int(all_coords_row.azimuth_index.values[0])]))))


def calc_interpolated_feature(time_domain_flag, triangle_vertices, coeffs, all_coords, subject_features, config=None):
    """Calculate the interpolated feature for a given point based on vertices specified by triangle_vertices, features
    specified by subject_features, and barycentric coefficients specified by coeffs"""
    # get features for each of the three closest points, add to a list in order of closest to farthest
    features = []
    for p in triangle_vertices:
        if time_domain_flag:
            features_p = get_feature_for_point(p[0], p[1], all_coords, subject_features)
            features_no_ITD = remove_itd(features_p, int(len(features_p)*0.04), len(features_p))
            features.append(features_no_ITD)
        else:
            features_p = get_feature_for_point_tensor(p[0], p[1], all_coords, subject_features, config)
            features.append(features_p)

    # based on equation 6 in "3D Tune-In Toolkit: An open-source library for real-time binaural spatialisation"
    if len(features) == 3:
        interpolated_feature = coeffs["alpha"] * features[0] + coeffs["beta"] * features[1] + coeffs["gamma"] * features[2]
    else:
        interpolated_feature = features[0]

    return interpolated_feature


def calc_all_interpolated_features(cs, features, euclidean_sphere, euclidean_sphere_triangles, euclidean_sphere_coeffs, config=None, time_domain_flag=True):
    """Essentially a wrapper function for calc_interpolated_features above, calculated interpolated features for all
    points on the euclidean sphere rather than a single point"""
    selected_feature_interpolated = []
    for i, p in enumerate(euclidean_sphere):
        if p[0] is not None:
            features_p = calc_interpolated_feature(time_domain_flag=time_domain_flag,
                                                   triangle_vertices=euclidean_sphere_triangles[i],
                                                   coeffs=euclidean_sphere_coeffs[i],
                                                   all_coords=cs.get_all_coords(),
                                                   subject_features=features,
                                                   config=config)

            selected_feature_interpolated.append(features_p)
        else:
            selected_feature_interpolated.append(None)

    return selected_feature_interpolated


def calc_hrtf(config, hrirs):
    """FFT to obtain HRTF from HRIR"""
    magnitudes = []
    phases = []

    for hrir in hrirs:
        # remove value that corresponds to 0 Hz
        hrtf = scipy.fft.rfft(hrir, config.nbins_hrtf*2)[1:]
        magnitude = abs(hrtf)
        phase = [cmath.phase(x) for x in hrtf]
        magnitudes.append(magnitude)
        phases.append(phase)
    return magnitudes, phases


def interpolate_fft(config, cs, features, sphere, sphere_triangles, sphere_coeffs, cube, edge_len, cs_output=True, time_domain_flag=True):
    """Combine all data processing steps into one function

    :param cs: Cubed sphere object associated with dataset
    :param features: All features for a given subject in the dataset, given by ds[i]['features'] from hrtfdata
    :param sphere: A list of locations of the gridded cubed sphere points to be interpolated,
                    given as (elevation, azimuth)
    :param sphere_triangles: A list of lists of triangle vertices for barycentric interpolation, where each list of
                             vertices defines the triangle for the corresponding point in sphere
    :param sphere_coeffs: A list of barycentric coordinates for each location in sphere, corresponding to the triangles
                          described by sphere_triangles
    :param cube: A list of locations of the gridded cubed sphere points to be interpolated, given as (panel, x, y)
    :param edge_len: Edge length of gridded cube
    """

    # interpolated_hrirs is a list of interpolated HRIRs corresponding to the points specified in load_sphere and
    # load_cube, all three lists share the same ordering
    interpolated_hrirs = calc_all_interpolated_features(cs, features, sphere, sphere_triangles, sphere_coeffs, config, time_domain_flag)

    magnitudes, phases = calc_hrtf(config, interpolated_hrirs)

    if not cs_output:
        return torch.tensor(np.array(magnitudes))

    # create empty list of lists of lists and initialize counter
    magnitudes_raw = [[[[] for _ in range(edge_len)] for _ in range(edge_len)] for _ in range(5)]
    magnitudes_raw_cube = [[[[] for _ in range(edge_len)] for _ in range(edge_len)] for _ in range(5)]
    magnitudes_zero_cube = [[[[] for _ in range(edge_len)] for _ in range(edge_len)] for _ in range(5)]
    count = 0

    for panel, x, y in cube:
        # based on cube coordinates, get indices for magnitudes list of lists
        i = panel - 1
        j = round(edge_len * (x - (PI_4 / edge_len) + PI_4) / (np.pi / 2))
        k = round(edge_len * (y - (PI_4 / edge_len) + PI_4) / (np.pi / 2))


        # add to list of lists of lists and increment counter
        magnitudes_raw[i][j][k] = magnitudes[count]
        magnitudes_raw_cube[i][j][k] = {'magnitude': magnitudes[count], 'cube': (panel, x, y)}
        magnitudes_zero_cube[i][j][k] = {'magnitude': np.array([1e-6 for mag in magnitudes[count]]), 'cube': (panel, x, y)}
        count += 1

    if config.single_panel:
        bottom_strip = np.concatenate(
            (magnitudes_raw_cube[3], magnitudes_raw_cube[0], magnitudes_raw_cube[1], magnitudes_raw_cube[2]))

        top = magnitudes_raw_cube[4]
        top_rotated_cw_90 = list(zip(*top[::-1]))
        top_rotated_cw_180 = list(zip(*top_rotated_cw_90[::-1]))
        top_rotated_ccw_90 = list(zip(*top))[::-1]

        # top_strip = np.concatenate((top_rotated_ccw_90, top, top_rotated_cw_90, top_rotated_cw_180))[:, 0:int(edge_len/2)]
        top_strip = np.concatenate((magnitudes_zero_cube[4], top, magnitudes_zero_cube[4], top_rotated_cw_180))[:,
                    0:int(edge_len / 2)]

        magnitudes_raw_flattened = np.concatenate((bottom_strip, top_strip), axis=1)

        feature = [[y['magnitude'] for y in x] for x in magnitudes_raw_flattened]

        #######################################
        #
        # import matplotlib.pyplot as plt
        #
        # fig, ax = plt.subplots()
        # import matplotlib
        # matplotlib.use('TkAgg')
        #
        # # Format data.
        # x, y = [], []
        # mask = []
        #
        # def getColour(magnitudes_cube):
        #     print(magnitudes_cube['cube'])
        #
        #     panel = magnitudes_cube['cube'][0]
        #     p = magnitudes_cube['cube'][1]
        #     q = magnitudes_cube['cube'][2]
        #
        #     if panel == 1:
        #         x_i, y_i = p, q
        #         colour = 'b'
        #     elif panel == 2:
        #         x_i, y_i = p + np.pi / 2, q
        #         colour = 'g'
        #     elif panel == 3:
        #         x_i, y_i = p + np.pi, q
        #         colour = 'r'
        #     elif panel == 4:
        #         x_i, y_i = p - np.pi / 2, q
        #         colour = 'k'
        #     elif panel == 5:
        #         x_i, y_i = p, q + np.pi / 2
        #         if x_i > 0 and y_i > 1.55:
        #             colour = 'y'
        #         elif x_i > 0 and y_i < 1.55:
        #             colour = 'magenta'
        #         elif x_i < 0 and y_i > 1.55:
        #             colour = 'pink'
        #         elif x_i < 0 and y_i < 1.55:
        #             colour = 'Lavender'
        #         else:
        #             colour = 'Gray'
        #     else:
        #         colour = 'Gray'
        #
        #     return colour
        #
        # for i in range(len(magnitudes_raw_flattened)):
        #     for j in range(len(magnitudes_raw_flattened[i])):
        #         colour = getColour(magnitudes_raw_flattened[i][j])
        #         x.append((i, colour))
        #         y.append((j, colour))
        #
        # colours = np.asarray([j[1] for j in x])
        # x, y = np.asarray([j[0] for j in x]), np.asarray([j[0] for j in y])
        #
        # # Plot the surface.
        # ax.scatter(x, y, c=colours, s=10, linewidth=0, antialiased=False)
        #
        # fig.tight_layout()
        # fig.set_size_inches(9, 4)
        # plt.show()
        #
        # #######################################
        #
        # import matplotlib.pyplot as plt
        # import matplotlib
        #
        # fig, ax = plt.subplots()
        # matplotlib.use('TkAgg')
        #
        # # Format data.
        # x, y = [], []
        # mask = []
        #
        # for panel, p, q in cube:
        #     if not np.isnan(p) and not np.isnan(q):
        #         mask.append(True)
        #
        #         if panel == 1:
        #             x_i, y_i = p, q
        #             colour = 'b'
        #         elif panel == 2:
        #             x_i, y_i = p + np.pi / 2, q
        #             colour = 'g'
        #         elif panel == 3:
        #             x_i, y_i = p + np.pi, q
        #             colour = 'r'
        #         elif panel == 4:
        #             x_i, y_i = p - np.pi / 2, q
        #             colour = 'k'
        #         elif panel == 5:
        #             x_i, y_i = p, q + np.pi / 2
        #             if x_i > 0 and y_i > 1.55:
        #                 colour = 'y'
        #             elif x_i > 0 and y_i < 1.55:
        #                 colour = 'magenta'
        #             elif x_i < 0 and y_i > 1.55:
        #                 colour = 'pink'
        #             elif x_i < 0 and y_i < 1.55:
        #                 colour = 'Lavender'
        #             else:
        #                 colour = 'Gray'
        #         else:
        #             x_i, y_i = p, q - np.pi / 2
        #
        #         x.append((x_i, colour))
        #         y.append((y_i, colour))
        #
        #     else:
        #         mask.append(False)
        #
        # colours = np.asarray([j[1] for j in x])
        # x, y = np.asarray([j[0] for j in x]), np.asarray([j[0] for j in y])
        #
        # # Plot the surface.
        # ax.scatter(x, y, c=colours, s=10, linewidth=0, antialiased=False)
        #
        # # draw lines outlining cube
        # ax.hlines(y=-PI_4, xmin=-3 * PI_4, xmax=5 * PI_4, linewidth=2, color="grey")
        # ax.hlines(y=PI_4, xmin=-3 * PI_4, xmax=5 * PI_4, linewidth=2, color="grey")
        # ax.hlines(y=3 * PI_4, xmin=-PI_4, xmax=PI_4, linewidth=2, color="grey")
        #
        # ax.vlines(x=-3 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")
        # ax.vlines(x=-PI_4, ymin=-PI_4, ymax=3 * PI_4, linewidth=2, color="grey")
        # ax.vlines(x=PI_4, ymin=-PI_4, ymax=3 * PI_4, linewidth=2, color="grey")
        # ax.vlines(x=3 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")
        # ax.vlines(x=5 * PI_4, ymin=-PI_4, ymax=PI_4, linewidth=2, color="grey")
        #
        # fig.tight_layout()
        # fig.set_size_inches(9, 4)
        # plt.show()

        #######################################

    else:
        feature = magnitudes_raw

    # convert list of numpy arrays into a single array, such that converting into tensor is faster
    return torch.tensor(np.array(feature))


def trim_hrir(hrir, start, stop):
    if start < 0:
        hrir_padded = np.pad(hrir, (abs(start), 0), mode='constant')
        trimmed_hrir = hrir_padded[:stop]
    else:
        trimmed_hrir = hrir[start:stop]
    return trimmed_hrir


def remove_itd(hrir, pre_window=None, length=None, output_delay=False):
    """Remove ITD from HRIR using kalman filter"""
    # normalize such that max(abs(hrir)) == 1
    rescaling_factor = 1 / max(np.abs(hrir))
    normalized_hrir = rescaling_factor * hrir

    # initialise Kalman filter
    x = np.array([[0]])  # estimated initial state
    p = np.array([[0]])  # estimated initial variance

    h = np.array([[1]])  # observation model (observation represents internal state directly)

    # r and q may require tuning
    r = np.array([[np.sqrt(400)]])  # variance of the observation noise
    q = np.array([[0.01]])  # variance of the process noise

    hrir_filter = KalmanFilter(x, p, h, q, r)
    f = np.array([[1]])  # F is state transition model
    for i, z in enumerate(normalized_hrir):
        hrir_filter.prediction(f)
        hrir_filter.update(z)
        # find first time post fit residual exceeds some threshold
        if np.abs(hrir_filter.get_post_fit_residual()) > 0.005:
            over_threshold_index = i
            break
    else:
        print("RuntimeWarning: ITD not removed (Kalman filter did not find a time where post fit residual exceeded threshold).")
        return hrir

    if output_delay:
        return over_threshold_index

    # create fade window in order to taper off HRIR towards the beginning and end
    fadeout_len = 50
    fadeout_interval = -1. / fadeout_len
    fadeout = np.arange(1 + fadeout_interval, fadeout_interval, fadeout_interval).tolist()

    fadein_len = 10
    fadein_interval = 1. / fadein_len
    fadein = np.arange(0.0, 1.0, fadein_interval).tolist()

    # trim HRIR based on first time threshold is exceeded
    start = over_threshold_index - pre_window
    if start < 0:
        stop = length
    else:
        stop = start + length

    if len(hrir) >= stop:
        trimmed_hrir = trim_hrir(hrir, start, stop)
        fade_window = fadein + [1] * (length - fadein_len - fadeout_len) + fadeout
        faded_hrir = np.array(trimmed_hrir) * fade_window
    else:
        trimmed_hrir = trim_hrir(hrir, start, -1)
        fade_window = fadein + [1] * (len(trimmed_hrir) - fadein_len - fadeout_len) + fadeout
        faded_hrir = np.array(trimmed_hrir) * fade_window
        zero_pad = [0] * (length - len(trimmed_hrir))
        faded_hrir = np.ma.append(faded_hrir, zero_pad)

    return faded_hrir


