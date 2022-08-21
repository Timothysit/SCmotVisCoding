import os
import glob
import numpy as np
import scipy.stats as sstats
import sklearn

import sklearn.linear_model as sklinear
import sklearn.metrics as sklmetrics
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KernelDensity
import scipy.optimize as sciopt

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import sciplotlib.style as splstyle
import sciplotlib.text as spltext
import pdb  # python debugger

from collections import defaultdict
import pandas as pd
from tqdm import tqdm  # loading bar


# Stats
import statsmodels.api as sm
from pymer4.models import Lmer


def load_data(data_folder, file_types_to_load=['_windowVis'],
              exp_ids=['SS041_2015-04-23', 'SS044_2015-04-28', 'SS045_2015-05-04',
                       'SS045_2015-05-05', 'SS047_2015-11-23', 'SS047_2015-12-03',
                       'SS048_2015-11-09', 'SS048_2015-12-02']):
    """
    Load data to do regression analysis
    Parameters
    -----------

    file_types_to_load : list
        list of files to load in the folder
        _windowVis : 1 x time array with the time (in seconds) of recording in the visual grating experiment
        _widowGray : 1 x time array with the time (in seconds) of recording in the gray screen experiment



    Returns
    -------
    data : dict
        dict where each key is the ID (mouse name and exp date) of an experiment
        and each item is another dict, within that dict, the keys are the different names of the data,
        and the items are the numpy arrays containing the data


    """

    data = dict()

    for exp_id in exp_ids:

        data[exp_id] = dict()

        for file_type in file_types_to_load:

            file_paths = glob.glob(os.path.join(data_folder, '%s*%s*.npy' % (file_type, exp_id)))
            assert len(file_paths) == 1
            file_data = np.load(file_paths[0])
            data[exp_id][file_type] = file_data



    return data


def plot_grating_exp_data(exp_data, zscore_activity=False, fig=None, axs=None,
                          label_size=11):
    """
    Plot visual grating experiment experiment variables (when the gratings were played)
    and observed variables
    Parameters
    ----------
    exp_data : dict

    Returns
    -------
    fig : matplotlib figure object
        figure object to plot in
    axs : array of matplotlib axs object
        axes object to plot in
    zscore_activity : bool
        whether to zscore activity before plotting the heatmap
    label_size : float
        size of the text labels for x and y axis of the plot
    """

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(3, 1, sharex=True)


    vis_exp_times = exp_data['_windowVis'].flatten()
    neural_activity = exp_data['_tracesVis']
    grating_intervals = exp_data['_gratingIntervals']
    vis_exp_saccade_intervals = exp_data['_saccadeIntervalsVis']

    num_neuron = np.shape(neural_activity)[1]
    num_time_points = len(vis_exp_times)
    grating_interval_binary_vec = np.zeros((num_time_points, ))
    saccade_interval_binary_vec = np.zeros((num_time_points, ))

    for onset_time, offset_time in grating_intervals:
        grating_interval_binary_vec[
            (vis_exp_times >= onset_time) &
            (vis_exp_times <= offset_time)
        ] = 1

    for onset_frame, offset_frame in vis_exp_saccade_intervals:
        saccade_interval_binary_vec[int(onset_frame):int(offset_frame)] = 1

    # plot the time of grating presentation (binary vector?)
    axs[0].plot(vis_exp_times, grating_interval_binary_vec, lw=1, color='black')
    axs[0].set_ylabel('Grating', size=label_size)

    axs[1].plot(vis_exp_times, saccade_interval_binary_vec, lw=1, color='black')
    axs[1].set_ylabel('Saccade', size=label_size)

    activity_to_plot = neural_activity.T
    if zscore_activity:
        activity_to_plot = (activity_to_plot - np.mean(activity_to_plot, axis=0)) / np.std(activity_to_plot, axis=0)
        vmin = -2
        vmax = 2
        cmap = 'bwr'
    else:
        vmin = None
        vmax = None
        cmap = 'viridis'

    axs[2].imshow(activity_to_plot, extent=[vis_exp_times[0], vis_exp_times[-1],
                                           1, num_neuron], aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    axs[2].set_xlabel('Time (seconds)', size=label_size)
    axs[2].set_ylabel('Neurons', size=label_size)

    return fig, axs


def get_vis_and_saccade_times(exp_data):

    vis_exp_times = exp_data['_windowVis'].flatten()

    grating_intervals = exp_data['_gratingIntervals']
    grating_onset_times = grating_intervals[:, 0]
    vis_exp_saccade_intervals = exp_data['_saccadeIntervalsVis'].astype(int)
    vis_exp_saccade_onset_times = vis_exp_times[vis_exp_saccade_intervals[:, 0]]
    saccade_dirs = exp_data['_saccadeVisDir'].flatten()
    saccade_dirs[saccade_dirs == 0] = -1

    grating_id_per_trial = exp_data['_gratingIds'] - 1  # matab 1 indexing to python 0 indexing
    id_to_grating_orientations = exp_data['_gratingIdDirections']
    grating_orientation_per_trial = [id_to_grating_orientations[int(x)][0] for x in grating_id_per_trial]

    return vis_exp_times, vis_exp_saccade_onset_times, grating_onset_times, saccade_dirs, grating_orientation_per_trial

def make_X_Y_for_regression(exp_data, feature_set=['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir'],
                            feature_time_windows={'vis_on': [-1.0, 3.0], 'vis_dir': [-1.0, 3.0], 'vis_ori': [-1.0, 3.0],
                                                  'saccade_on': [-1.0, 3.0], 'saccade_dir': [-1.0, 3.0],
                                                  'vis_on_saccade_on': [-1.0, 3.0], 'vis_ori_iterative': [0, 3.0]},
                            neural_preprocessing_steps=['zscore'], check_for_nans=True,
                            train_indices=None, test_indices=None):
    """
    Make feature matrix (or design matrix) X and target matrix Y from experiment data

    Parameters
    ----------
    exp_data : dict
    feature_set : list
    feature_time_windows : dict
    neural_preprocessing_steps : list
        list of pre-processing transformations to perform on the neural activity
    check_for_nans : bool
        whether to check for NaNs in the data, and if detected remove them
    train_indices : list (optional)
    test_indices : list (optional)
    Returns
    -------

    Notes
    ------
    grating duration : 2 seconds

    """


    vis_exp_times = exp_data['_windowVis'].flatten()
    neural_activity = exp_data['_tracesVis']

    grating_intervals = exp_data['_gratingIntervals']
    vis_exp_saccade_intervals = exp_data['_saccadeIntervalsVis'].astype(int)
    vis_exp_saccade_onset_times = vis_exp_times[vis_exp_saccade_intervals[:, 0]]
    saccade_dirs = exp_data['_saccadeVisDir'].flatten()
    saccade_dirs[saccade_dirs == 0] = -1

    grating_id_per_trial = exp_data['_gratingIds'] - 1  # matab 1 indexing to python 0 indexing
    id_to_grating_orientations = exp_data['_gratingIdDirections']
    grating_orientation_per_trial = [id_to_grating_orientations[int(x)][0] for x in grating_id_per_trial]

    if check_for_nans:
        if np.sum(np.isnan(neural_activity)) > 0:
            print('Detected NaNs, assuming this is along the time dimension and removing time frames with NaNs')
            subset_time_frame = ~np.isnan(np.sum(neural_activity, axis=1))
            neural_activity = neural_activity[subset_time_frame, :]
            vis_exp_times = vis_exp_times[subset_time_frame]

    sec_per_time_samples = np.mean(np.diff(vis_exp_times))
    num_time_samples = int(len(vis_exp_times))

    # Make target matrix Y
    Y = neural_activity
    if 'zscore' in neural_preprocessing_steps:
        Y  = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)


    # Make X
    feature_matrices = []

    for feature_name in feature_set:

        if feature_name == 'bias':

            feature_mat = np.zeros((num_time_samples, 1))

        elif feature_name == 'vis_on':
            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            grating_onset_times = grating_intervals[:, 0]
            # grating_offset_times = grating_intervals[:, 1]

            subset_onset_time = grating_onset_times[
                (grating_onset_times + feature_time_window[1]) < vis_exp_times[-1]
                ]

            for onset_time in subset_onset_time:

                # onset_sample = np.argmin(np.abs(vis_exp_times - onset_time))
                start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                feature_mat[row_idx, col_idx] = 1


        elif feature_name == 'vis_dir':
            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            grating_onset_times = grating_intervals[:, 0]
            # grating_offset_times = grating_intervals[:, 1]

            subset_onset_time = grating_onset_times[
                (grating_onset_times + feature_time_window[1]) < vis_exp_times[-1]
                ]

            for n_trial, onset_time in enumerate(subset_onset_time):

                # onset_sample = np.argmin(np.abs(vis_exp_times - onset_time))
                start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                # Grating direction
                if grating_orientation_per_trial[n_trial] == 90:
                    grating_dir_value = 1
                elif grating_orientation_per_trial[n_trial] == 270:
                    grating_dir_value = -1
                else:
                    grating_dir_value = 0

                feature_mat[row_idx, col_idx] = grating_dir_value
        elif feature_name == 'vis_ori':

            unique_ori = np.unique(grating_orientation_per_trial)
            unique_ori = unique_ori[~np.isnan(unique_ori)]
            num_unique_ori = len(unique_ori)

            feature_time_window = feature_time_windows[feature_name]
            num_time_sample_per_ori = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            num_sample_in_window = num_time_sample_per_ori * num_unique_ori
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            grating_onset_times = grating_intervals[:, 0]
            # grating_offset_times = grating_intervals[:, 1]

            subset_trial_idx = np.where((grating_onset_times + feature_time_window[1]) < vis_exp_times[-1])[0]

            for n_trial, trial_idx in enumerate(subset_trial_idx):

                onset_time = grating_onset_times[trial_idx]
                # onset_sample = np.argmin(np.abs(vis_exp_times - onset_time))
                start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                # Grating direction
                trial_grating_ori = grating_orientation_per_trial[trial_idx]

                if ~np.isnan(trial_grating_ori):
                    grating_idx = np.where(unique_ori==trial_grating_ori)[0][0]
                    col_start = int(grating_idx * num_time_sample_per_ori)
                    col_end = int((grating_idx + 1) * num_time_sample_per_ori)
                    col_idx = np.arange(col_start, col_end)

                    if len(row_idx) != len(col_idx):
                        row_idx = row_idx[0:len(col_idx)]

                    feature_mat[row_idx, col_idx] = 1


        elif feature_name == 'saccade_on':
            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            subset_vis_exp_saccade_onset_times = vis_exp_saccade_onset_times[
                (vis_exp_saccade_onset_times + feature_time_window[1] < vis_exp_times[-1]) &
                (vis_exp_saccade_onset_times + feature_time_window[0] > vis_exp_times[0])
                ]

            for n_saccade_time, onset_time in enumerate(subset_vis_exp_saccade_onset_times):

                start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                feature_mat[row_idx, col_idx] = 1


        elif feature_name == 'saccade_dir':
            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            subset_vis_exp_saccade_onset_times = vis_exp_saccade_onset_times[
                (vis_exp_saccade_onset_times + feature_time_window[1] < vis_exp_times[-1]) &
                (vis_exp_saccade_onset_times + feature_time_window[0] > vis_exp_times[0])
                ]

            for n_saccade_time, onset_time in enumerate(subset_vis_exp_saccade_onset_times):

                if onset_time + feature_time_window[1] > vis_exp_times[-1]:
                    print('onset time exceeds experiment time, skipping')
                    continue

                start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                feature_mat[row_idx, col_idx] = saccade_dirs[n_saccade_time]

        elif feature_name == 'vis_on_saccade_on':

            # TODO: this is quick copy and paste, can simplify this
            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            saccade_feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            saccade_binary_vector = np.zeros((num_time_samples, ))

            for n_saccade_time, onset_time in enumerate(vis_exp_saccade_onset_times):

                if onset_time + feature_time_window[1] > vis_exp_times[-1]:
                    print('onset time exceeds experiment time, skipping')
                    continue
                if onset_time + feature_time_window[0] < vis_exp_times[0]:
                    print('onset time is lower than experiment start time, skipping')
                    continue

                start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)
                saccade_binary_vector[row_idx] = 1

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                saccade_feature_mat[row_idx, col_idx] = 1

            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            vis_on_feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            grating_onset_times = grating_intervals[:, 0]
            # grating_offset_times = grating_intervals[:, 1]

            grating_binary_vector = np.zeros((num_time_samples, ))

            for onset_time in grating_onset_times:

                if onset_time + feature_time_window[1] > vis_exp_times[-1]:
                    print('onset time exceeds experiment time, skipping')
                    continue

                # onset_sample = np.argmin(np.abs(vis_exp_times - onset_time))
                start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                grating_binary_vector[row_idx] = 1
                vis_on_feature_mat[row_idx, col_idx] = 1

            saccade_and_vis_on_frames = saccade_binary_vector * grating_binary_vector
            # feature_mat = vis_on_feature_mat * saccade_feature_mat

        elif feature_name == 'vis_ori_iterative':

            # Get orientation per trial and the grating onset times
            grating_orientation_per_trial = np.array(grating_orientation_per_trial)
            unique_ori = np.unique(grating_orientation_per_trial)
            unique_ori = unique_ori[~np.isnan(unique_ori)]
            num_unique_ori = len(unique_ori)

            feature_time_window = feature_time_windows[feature_name]
            num_time_sample_per_ori = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            num_sample_in_window = num_time_sample_per_ori * num_unique_ori
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            grating_onset_times = grating_intervals[:, 0]
            # grating_offset_times = grating_intervals[:, 1]

            subset_trial_idx = np.where((grating_onset_times + feature_time_window[1]) < vis_exp_times[-1])[0]
            grating_onset_times_subset = grating_onset_times[subset_trial_idx]
            grating_orientation_per_trial_subset = grating_orientation_per_trial[subset_trial_idx]

            num_neurons = np.shape(Y)[1]
            Y_aligned_to_vis_onset = np.zeros((num_time_sample_per_ori, num_neurons, len(grating_onset_times_subset)))

            for n_onset, onset_time in enumerate(grating_onset_times_subset):

                start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[1])))
                subset_idx = np.arange(start_sample, end_sample)
                if len(subset_idx) > num_time_sample_per_ori:
                    subset_idx = subset_idx[0:num_time_sample_per_ori]

                Y_aligned_to_vis_onset[:, :, n_onset] = Y[subset_idx, :]

            for train_idx_set in train_indices:
                train_supported_time = vis_exp_times[train_idx_set]
                orientation_activity_matrix = np.zeros((num_time_sample_per_ori, num_neurons, num_unique_ori))
                train_trial_idx = np.where(
                    (grating_onset_times_subset >= train_supported_time[0]) &
                    (grating_onset_times_subset <= train_supported_time[-1])
                )[0]
                train_grating_per_trial = grating_orientation_per_trial_subset[train_trial_idx]

                for n_ori, ori in enumerate(unique_ori):
                    trial_idx_to_get = np.where(train_grating_per_trial == ori)[0]
                    orientation_activity_matrix[:, :, n_ori] = np.mean(
                        Y_aligned_to_vis_onset[:, :, trial_idx_to_get], axis=2
                    )


                # temp_save_path = '/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/testData/iterative_fit_example_test_set.npy'
                # np.save(temp_save_path, orientation_activity_matrix)

            for test_idx_set in test_indices:
                train_supported_time = vis_exp_times[test_idx_set]
                orientation_activity_matrix = np.zeros((num_time_sample_per_ori, num_neurons, num_unique_ori))
                train_trial_idx = np.where(
                    (grating_onset_times_subset >= train_supported_time[0]) &
                    (grating_onset_times_subset <= train_supported_time[-1])
                )[0]
                train_grating_per_trial = grating_orientation_per_trial_subset[train_trial_idx]

                for n_ori, ori in enumerate(unique_ori):
                    trial_idx_to_get = np.where(train_grating_per_trial == ori)[0]
                    orientation_activity_matrix[:, :, n_ori] = np.mean(
                        Y_aligned_to_vis_onset[:, :, trial_idx_to_get], axis=2
                    )

                pdb.set_trace()
                temp_save_path = '/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/testData/iterative_fit_example_test_set.npy'
                np.save(temp_save_path, orientation_activity_matrix)

            # For each

            pdb.set_trace()

        else:
            print('%s is not a valid feature name' % feature_name)

        feature_matrices.append(feature_mat)

    X = np.concatenate(feature_matrices, axis=1)



    return X, Y


def get_ori_activity_from_time_indices(Y_aligned_to_vis_onset, vis_exp_times, grating_onset_times,
                                       indices_to_get, grating_per_trial, num_time_sample_per_ori):
    """

    """

    unique_ori = np.unique(grating_per_trial[~np.isnan(grating_per_trial)])
    num_unique_ori = len(unique_ori)
    num_neurons = np.shape(Y_aligned_to_vis_onset)[1]
    orientation_activity_matrix_list = []

    for idx_set in indices_to_get:
        supported_time = vis_exp_times[idx_set]
        orientation_activity_matrix = np.zeros((num_time_sample_per_ori, num_neurons, num_unique_ori))
        trial_idx = np.where(
            (grating_onset_times >= supported_time[0]) &
            (grating_onset_times <= supported_time[-1])
        )[0]
        grating_per_trial_subset = grating_per_trial[trial_idx]

        for n_ori, ori in enumerate(unique_ori):
            trial_idx_to_get = np.where(grating_per_trial_subset == ori)[0]
            orientation_activity_matrix[:, :, n_ori] = np.mean(
                Y_aligned_to_vis_onset[:, :, trial_idx_to_get], axis=2
            )

        orientation_activity_matrix_list.append(orientation_activity_matrix)


    return orientation_activity_matrix_list


def get_ori_train_test_data(exp_data, time_window,
                            neural_preprocessing_steps=['zscore'], check_for_nans=True,
                            train_indices=None, test_indices=None, method='split_then_align'):


    vis_exp_times = exp_data['_windowVis'].flatten()
    neural_activity = exp_data['_tracesVis']

    grating_intervals = exp_data['_gratingIntervals']
    vis_exp_saccade_intervals = exp_data['_saccadeIntervalsVis'].astype(int)
    vis_exp_saccade_onset_times = vis_exp_times[vis_exp_saccade_intervals[:, 0]]
    saccade_dirs = exp_data['_saccadeVisDir'].flatten()
    saccade_dirs[saccade_dirs == 0] = -1

    grating_id_per_trial = exp_data['_gratingIds'] - 1  # matab 1 indexing to python 0 indexing
    id_to_grating_orientations = exp_data['_gratingIdDirections']
    grating_orientation_per_trial = [id_to_grating_orientations[int(x)][0] for x in grating_id_per_trial]

    if check_for_nans:
        if np.sum(np.isnan(neural_activity)) > 0:
            print('Detected NaNs, assuming this is along the time dimension and removing time frames with NaNs')
            subset_time_frame = ~np.isnan(np.sum(neural_activity, axis=1))
            neural_activity = neural_activity[subset_time_frame, :]
            vis_exp_times = vis_exp_times[subset_time_frame]

    sec_per_time_samples = np.mean(np.diff(vis_exp_times))
    num_time_samples = int(len(vis_exp_times))

    # Make target matrix Y
    Y = neural_activity
    if 'zscore' in neural_preprocessing_steps:
        Y  = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    grating_orientation_per_trial = np.array(grating_orientation_per_trial)
    unique_ori = np.unique(grating_orientation_per_trial)
    unique_ori = unique_ori[~np.isnan(unique_ori)]
    num_unique_ori = len(unique_ori)

    feature_time_window = time_window
    num_time_sample_per_ori = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
    num_sample_in_window = num_time_sample_per_ori * num_unique_ori
    feature_mat = np.zeros((num_time_samples, num_sample_in_window))
    grating_onset_times = grating_intervals[:, 0]
    # grating_offset_times = grating_intervals[:, 1]

    subset_trial_idx = np.where((grating_onset_times + feature_time_window[1]) < vis_exp_times[-1])[0]
    grating_onset_times_subset = grating_onset_times[subset_trial_idx]
    grating_orientation_per_trial_subset = grating_orientation_per_trial[subset_trial_idx]

    num_neurons = np.shape(Y)[1]
    Y_aligned_to_vis_onset = np.zeros((num_time_sample_per_ori, num_neurons, len(grating_onset_times_subset)))

    for n_onset, onset_time in enumerate(grating_onset_times_subset):

        start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[0])))
        end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[1])))
        subset_idx = np.arange(start_sample, end_sample)
        if len(subset_idx) > num_time_sample_per_ori:
            subset_idx = subset_idx[0:num_time_sample_per_ori]

        Y_aligned_to_vis_onset[:, :, n_onset] = Y[subset_idx, :]

    if method == 'split_then_align':
        # train_orientation_activity_matrix = []
        # test_orientation_activity_matrix = []

        """"
        for train_idx_set in train_indices:
            train_supported_time = vis_exp_times[train_idx_set]
            orientation_activity_matrix = np.zeros((num_time_sample_per_ori, num_neurons, num_unique_ori))
            train_trial_idx = np.where(
                (grating_onset_times_subset >= train_supported_time[0]) &
                (grating_onset_times_subset <= train_supported_time[-1])
            )[0]
            train_grating_per_trial = grating_orientation_per_trial_subset[train_trial_idx]

            for n_ori, ori in enumerate(unique_ori):
                trial_idx_to_get = np.where(train_grating_per_trial == ori)[0]
                orientation_activity_matrix[:, :, n_ori] = np.mean(
                    Y_aligned_to_vis_onset[:, :, trial_idx_to_get], axis=2
                )

            train_orientation_activity_matrix.append(orientation_activity_matrix)
        """

        train_orientation_activity_matrix = get_ori_activity_from_time_indices(Y_aligned_to_vis_onset, vis_exp_times,
                                                                               grating_onset_times_subset,
                                                                               indices_to_get=train_indices,
                                                                               grating_per_trial=grating_orientation_per_trial_subset,
                                                                               num_time_sample_per_ori=num_time_sample_per_ori)

        test_orientation_activity_matrix = get_ori_activity_from_time_indices(Y_aligned_to_vis_onset, vis_exp_times,
                                                                               grating_onset_times_subset,
                                                                               indices_to_get=test_indices,
                                                                               grating_per_trial=grating_orientation_per_trial_subset,
                                                                              num_time_sample_per_ori=num_time_sample_per_ori)

        # TODO: tidy this code up
        """
        for test_idx_set in test_indices:
            test_supported_time = vis_exp_times[test_idx_set]
            orientation_activity_matrix = np.zeros((num_time_sample_per_ori, num_neurons, num_unique_ori))
            test_trial_idx = np.where(
                (grating_onset_times_subset >= test_supported_time[0]) &
                (grating_onset_times_subset <= test_supported_time[-1])
            )[0]
            train_grating_per_trial = grating_orientation_per_trial_subset[test_trial_idx]

            for n_ori, ori in enumerate(unique_ori):
                trial_idx_to_get = np.where(train_grating_per_trial == ori)[0]
                orientation_activity_matrix[:, :, n_ori] = np.mean(
                    Y_aligned_to_vis_onset[:, :, trial_idx_to_get], axis=2
                )

            test_orientation_activity_matrix.append(orientation_activity_matrix)
        """

    elif method == 'align_then_split':

        unique_ori, unique_counts = np.unique(grating_orientation_per_trial_subset[
            ~np.isnan(grating_orientation_per_trial_subset)
                                              ], return_counts=True)
        min_counts = np.min(unique_counts)

        orientation_activity_matrix = np.zeros((num_time_sample_per_ori, num_neurons, num_unique_ori, min_counts))

        for n_ori, ori in enumerate(unique_ori):
            trial_idx_to_get = np.where(grating_orientation_per_trial_subset == ori)[0]
            trial_idx_to_get = trial_idx_to_get[0:min_counts]
            orientation_activity_matrix[:, :, n_ori, :] = Y_aligned_to_vis_onset[:, :, trial_idx_to_get]

        train_trial_idx = np.arange(0, int(min_counts/2))
        test_trial_idx = np.arange(int(min_counts/2), min_counts)
        first_half_activity = np.mean(orientation_activity_matrix[:, :, :, train_trial_idx], axis=-1)
        second_half_activity = np.mean(orientation_activity_matrix[:, :, :, test_trial_idx], axis=-1)

        train_orientation_activity_matrix = [first_half_activity, second_half_activity]
        test_orientation_activity_matrix = [second_half_activity, first_half_activity]


    return train_orientation_activity_matrix, test_orientation_activity_matrix


def fit_regression_model(X, Y, model_type='Ridge', train_test_split_method='half', n_cv_folds=10,
                         custom_train_indices=None, custom_test_indices=None):
    """
    Fits regression model

    Parameters
    ----------
    X : numpy ndarray
        feature / design matrix of shape (num_time_points, num_features)
    Y : numpy ndarray
        target matrix of shape (num_time_points, num_neurons)
    model_type : str
        type of (regression) model to fit
    train_test_split_method : str
        how to do train-test split
    n_cv_folds : int
        number of cross-validation folds to do
    Returns
    -------
    regression_result : dict

    """


    regression_result = dict()

    if train_test_split_method == 'half':
        num_time_points = np.shape(Y)[0]
        first_half_indices = np.arange(0, int(num_time_points/2))
        second_half_indices = np.arange(int(num_time_points/2), num_time_points)

        # Make sure they are the same length
        if len(first_half_indices) < len(second_half_indices):
            second_half_indices = second_half_indices[0:len(first_half_indices)]
        else:
            first_half_indices = first_half_indices[0:len(second_half_indices)]

        train_indices = [first_half_indices, second_half_indices]
        test_indices = [second_half_indices, first_half_indices]


    elif train_test_split_method == 'n_fold_cv':
        print('TODO: set up n-fold cross validation')


    if model_type == 'Ridge':
        model = sklinear.Ridge(alpha=1.0, fit_intercept=False)

    num_cv_set = len(train_indices)
    num_neurons = np.shape(Y)[1]
    explained_variance_per_cv_set = np.zeros((num_neurons, num_cv_set))
    Y_test_hat_per_cv_set = []
    Y_test_per_cv_set = []
    test_idx_per_set = []

    for n_cv_set, (train_idx, test_idx) in enumerate(zip(train_indices, test_indices)):

        X_train, X_test = X[train_idx, :], X[test_idx, :]
        Y_train, Y_test = Y[train_idx, :], Y[test_idx, :]
        model.fit(X_train, Y_train)

        Y_test_hat = model.predict(X_test)
        explained_variance = sklmetrics.explained_variance_score(y_true=Y_test, y_pred=Y_test_hat,
                                                                 multioutput='raw_values')
        Y_test_hat_per_cv_set.append(Y_test_hat)
        Y_test_per_cv_set.append(Y_test)
        explained_variance_per_cv_set[:, n_cv_set] = explained_variance
        test_idx_per_set.append(test_idx)

    regression_result['X'] = X
    regression_result['Y'] = Y
    regression_result['test_idx_per_cv_set'] = np.array(test_idx_per_set)
    regression_result['Y_test_hat_per_cv_set'] = Y_test_hat_per_cv_set
    regression_result['explained_variance_per_cv_set'] = explained_variance_per_cv_set
    regression_result['Y_test_per_cv_set'] = np.array(Y_test_per_cv_set)

    return regression_result


def get_aligned_explained_variance(regression_result, exp_data, alignment_time_window=[0, 3]):
    """
    regression_result : dict
        dictionary containing regression results, it should have the following keys
        'Y_test_per_cv_set' : list of numpy arrays
        'test_idx_per_cv_set' : list of numpy arrays

    exp_data : dict
        dictionary containing all used experiment data
    """

    # Get the fitted activity matrix for the entire time trace
    Y = regression_result['Y']
    Y_test_per_cv_set = regression_result['Y_test_per_cv_set']
    Y_test_hat_per_cv_set = np.array(regression_result['Y_test_hat_per_cv_set'])
    test_idx_per_cv_set = regression_result['test_idx_per_cv_set']

    vis_exp_times, vis_exp_saccade_onset_times, grating_onset_times, saccade_dirs, grating_orientation_per_trial = get_vis_and_saccade_times(
        exp_data)

    subset_vis_exp_vis_onset_times = grating_onset_times[
        (grating_onset_times + alignment_time_window[1] < vis_exp_times[-1]) &
        (grating_onset_times + alignment_time_window[0] > vis_exp_times[0])
        ]

    subset_vis_exp_saccade_onset_times = vis_exp_saccade_onset_times[
        (vis_exp_saccade_onset_times + alignment_time_window[1] < vis_exp_times[-1]) &
        (vis_exp_saccade_onset_times + alignment_time_window[0] > vis_exp_times[0])
        ]

    num_time_points = len(vis_exp_times)
    num_neurons = np.shape(Y_test_per_cv_set)[2]
    Y_hat = np.zeros((num_time_points, num_neurons)) + np.nan

    for n_cv, test_idx in enumerate(test_idx_per_cv_set):
        Y_hat[test_idx, :] = Y_test_hat_per_cv_set[n_cv, :, :]

    if np.sum(np.isnan(Y_hat)) != 0:
        print('Something wrong with getting Y_hat, found: %.f nans' % (np.sum(np.isnan(Y_hat))))
        Y_hat[np.isnan(Y_hat)] = np.mean(Y_hat)
        # pdb.set_trace()

    num_saccades = len(subset_vis_exp_saccade_onset_times)
    num_grating_presentations = len(subset_vis_exp_vis_onset_times)

    sec_per_time_samples = np.mean(np.diff(vis_exp_times))
    num_aligned_time_points = int((alignment_time_window[1] - alignment_time_window[0]) / sec_per_time_samples)
    num_trials_to_get = np.min([num_saccades, num_grating_presentations])

    Y_hat_vis_aligned = np.zeros((num_trials_to_get, num_aligned_time_points, num_neurons)) + np.nan
    Y_vis_aligned = np.zeros((num_trials_to_get, num_aligned_time_points, num_neurons)) + np.nan
    Y_hat_saccade_aligned = np.zeros((num_trials_to_get, num_aligned_time_points, num_neurons)) + np.nan
    Y_saccade_aligned = np.zeros((num_trials_to_get, num_aligned_time_points, num_neurons)) + np.nan

    subset_vis_on_times = np.random.choice(subset_vis_exp_vis_onset_times, num_trials_to_get)
    subset_saccade_on_times = np.random.choice(subset_vis_exp_saccade_onset_times, num_trials_to_get)


    # align to visual stimulus
    for n_trial, vis_on_time in enumerate(subset_vis_on_times):

        time_idx = np.where(
            (vis_exp_times >= (vis_on_time + alignment_time_window[0])) &
            (vis_exp_times <= (vis_on_time + alignment_time_window[1]))
        )[0]

        time_idx = time_idx[0:num_aligned_time_points]

        Y_hat_vis_aligned[n_trial, :, :] = Y_hat[time_idx, :]
        Y_vis_aligned[n_trial, :, :] = Y[time_idx, :]

    # align to saccade onset
    for n_trial, saccade_on_time in enumerate(subset_saccade_on_times):
        time_idx = np.where(
            (vis_exp_times >= (saccade_on_time + alignment_time_window[0])) &
            (vis_exp_times <= (saccade_on_time + alignment_time_window[1]))
        )[0]

        time_idx = time_idx[0:num_aligned_time_points]

        Y_hat_saccade_aligned[n_trial, :, :] = Y_hat[time_idx, :]
        Y_saccade_aligned[n_trial, :, :] = Y[time_idx, :]

    Y_hat_vis_aligned_flattened = Y_hat_vis_aligned.reshape(-1, num_neurons)
    Y_vis_aligned_flattened = Y_vis_aligned.reshape(-1, num_neurons)

    Y_hat_saccade_aligned_flattened = Y_hat_saccade_aligned.reshape(-1, num_neurons)
    Y_saccade_aligned_flattened = Y_saccade_aligned.reshape(-1, num_neurons)

    vis_aligned_var_explained = sklmetrics.explained_variance_score(Y_vis_aligned_flattened, Y_hat_vis_aligned_flattened, multioutput='raw_values')
    saccade_aligned_var_explained = sklmetrics.explained_variance_score(Y_saccade_aligned_flattened, Y_hat_saccade_aligned_flattened, multioutput='raw_values')

    regression_result['vis_aligned_var_explained'] = vis_aligned_var_explained
    regression_result['saccade_aligned_var_explained'] = saccade_aligned_var_explained

    return regression_result


def do_iterative_orientation_fit(grating_orientation_per_trial):


    unique_ori = np.unique(grating_orientation_per_trial)
    unique_ori = unique_ori[~np.isnan(unique_ori)]
    num_unique_ori = len(unique_ori)

    feature_time_window = feature_time_windows[feature_name]
    num_time_sample_per_ori = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
    num_sample_in_window = num_time_sample_per_ori * num_unique_ori
    feature_mat = np.zeros((num_time_samples, num_sample_in_window))
    grating_onset_times = grating_intervals[:, 0]

    # TODO: get aligned time trace for each orientation


    return ori_fit_results


def loss_function_1(time_course, actual_activity, ori_scales):
    prediction = np.outer(time_course, ori_scales)
    loss = np.mean(np.sqrt((actual_activity - prediction) ** 2))

    return loss


def loss_function_2(ori_scales, actual_activity, time_course):
    prediction = np.outer(time_course, ori_scales)
    loss = np.mean(np.sqrt((actual_activity - prediction) ** 2))

    return loss



def fit_all_neuron_ori(orientation_activity_matrix, num_iter=5, method='iterative'):

    num_neuron = np.shape(orientation_activity_matrix)[1]
    prediction_per_neuron = np.zeros(np.shape(orientation_activity_matrix))

    for neuron_id in tqdm(np.arange(num_neuron)):

        neuron_orientation_activity_matrix = orientation_activity_matrix[:, neuron_id, :]

        if method == 'iterative':
            prediction = fit_single_neuron_ori_iteratively(neuron_orientation_activity_matrix, num_iter=num_iter)
        elif method == 'svd':
            prediction = fit_single_neuron_ori_w_svd(neuron_orientation_activity_matrix)
        elif method == 'mean':
            # just returning the mean
            prediction = neuron_orientation_activity_matrix

        prediction_per_neuron[:, neuron_id, :] = prediction

    return prediction_per_neuron


def fit_single_neuron_ori_iteratively(neuron_orientation_activity_matrix, num_iter=5):


    actual_activity = neuron_orientation_activity_matrix
    num_ori = np.shape(neuron_orientation_activity_matrix)[1]
    num_time_points = np.shape(neuron_orientation_activity_matrix)[0]
    fitted_ori_scales = np.zeros(num_ori, ) + 1
    fitted_time_course = np.random.normal(0, 1, num_time_points)

    loss_per_iter_step = np.zeros((num_iter * 2 + 1,))
    loss_per_iter = np.zeros((num_iter,))
    prediction_per_iter = np.zeros((num_iter, num_time_points, num_ori))

    init_loss = loss_function_1(fitted_time_course, actual_activity, fitted_ori_scales)
    loss_per_iter_step[0] = init_loss
    step_counter = 1

    time_course_per_iteration = np.zeros((num_iter + 1, num_time_points))
    scales_per_iteration = np.zeros((num_iter + 1, num_ori))
    time_course_per_iteration[0, :] = fitted_time_course
    scales_per_iteration[0, :] = fitted_ori_scales

    for iter_i in np.arange(num_iter):
        # first optimise the time course
        # time_course_loss_func = functools.partial(loss_function, actual_activity=actual_activity, ori_scales=fitted_ori_scales)
        fitted_time_course_opt_result = sciopt.minimize(loss_function_1, fitted_time_course,
                                                        args=(actual_activity, fitted_ori_scales))
        # fitted_time_course_opt_result = sciopt.minimize(time_course_loss_func, fitted_time_course)
        fitted_time_course = fitted_time_course_opt_result['x']
        loss = fitted_time_course_opt_result['fun']
        loss_per_iter_step[step_counter] = loss
        step_counter += 1

        # scales_loss_func = functools.partial(loss_function, actual_activity=actual_activity, time_course=fitted_time_course)
        fitted_ori_scales_opt_result = sciopt.minimize(loss_function_2, fitted_ori_scales,
                                                       args=(actual_activity, fitted_time_course))
        fitted_ori_scales = fitted_ori_scales_opt_result['x']

        loss = fitted_ori_scales_opt_result['fun']
        loss_per_iter[iter_i] = loss

        loss_per_iter_step[step_counter] = loss
        step_counter += 1

        time_course_per_iteration[iter_i + 1, :] = fitted_time_course
        scales_per_iteration[iter_i + 1, :] = fitted_ori_scales

        prediction_per_iter[iter_i, :, :] = np.outer(fitted_time_course, fitted_ori_scales)

    prediction = prediction_per_iter[-1, :, :]

    return prediction


def fit_single_neuron_ori_w_svd(neuron_orientation_activity_matrix, component_idx=0):


    u_left, sigma, v_right = np.linalg.svd(neuron_orientation_activity_matrix)
    prediction = np.matmul(u_left[:, component_idx].reshape(-1, 1) * sigma[component_idx], v_right[component_idx, :].reshape(1, -11))

    return prediction


def plot_neuron_orientation_tuning_fit(activity_matrices, labels=['Observed (train)', 'Observed (test)', 'Fitted'],
                                       colors=['black', 'red', 'gray'],
                                       fig=None, axs=None):

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(4, 3, sharex=True, sharey=True)

    num_labels = len(labels)

    for label_idx in np.arange(num_labels):
        for ori_i in np.arange(0, 12):

            if ori_i == 0:
                label = labels[label_idx]
            else:
                label = None

            axs.flatten()[ori_i].plot(activity_matrices[label_idx, :, ori_i],
                                      color=colors[label_idx], label=label)

    fig.legend(bbox_to_anchor=(1.2, 0.85))

    return fig, axs


def get_num_saccade_per_grating(vis_exp_saccade_onset_times, grating_onset_times, grating_orientation_per_trial):


    unique_ori = np.unique(grating_orientation_per_trial[~np.isnan(grating_orientation_per_trial)])
    num_saccade_per_grating = np.zeros((unique_ori, )) + np.nan

    for n_ori, ori in enumerate(unique_ori):

        subset_trial_idx = np.where(grating_orientation_per_trial == ori)[0]
        subset_grating_onset_times = grating_onset_times[subset_trial_idx]

        saccade_counter = 0

        for grating_onset_t in subset_grating_onset_times:

            pdb.set_trace()


    return num_saccade_per_grating

def main():

    available_processes = ['load_data', 'plot_data', 'fit_regression_model', 'plot_regression_model_explained_var',
                           'plot_regression_model_example_neurons', 'compare_iterative_vs_normal_fit']

    processes_to_run = ['compare_iterative_vs_normal_fit']
    process_params = {
        'load_data': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir']
        ),
        'plot_data': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis'],
            zscore_activity=True,
        ),
        'fit_regression_model': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis'],
            X_sets_to_compare={'bias_only': ['bias'],
                               'vis_on_only': ['bias', 'vis_on'],
                               'vis_ori': ['bias', 'vis_ori'],
                               # 'vis_ori_iterative': ['bias', 'vis_ori_iterative'],
                               # 'vis': ['bias', 'vis_on', 'vis_dir'],
                               'saccade_on_only': ['bias', 'saccade_on'],
                               'saccade': ['bias', 'saccade_on', 'saccade_dir'],
                               # 'vis_and_saccade': ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir'],
                               # 'vis_on_and_saccade_on': ['bias', 'vis_on', 'saccade_on'],
                               # 'vis_on_and_saccade_on_and_interaction': ['bias', 'vis_on', 'saccade_on',
                               #                                           'vis_on_saccade_on']
                               },
            feature_time_windows={'vis_on': [-1.0, 3.0], 'vis_dir': [-1.0, 3.0], 'vis_ori': [-1.0, 3.0],
                                  'saccade_on': [-1.0, 3.0], 'saccade_dir': [-1.0, 3.0],
                                  'vis_on_saccade_on': [-1.0, 3.0], 'vis_ori_iterative': [0, 3.0]},
            dataset='grating',
            neural_preprocessing_steps=['zscore'],  # 'zscore' is optional
        ),
        'plot_regression_model_explained_var': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            X_sets_to_compare=[
                ['vis_on_only', 'vis_ori'],
                ['saccade_on_only', 'saccade'],
                ['vis_on_only', 'saccade_on_only'],
                ['vis_ori', 'saccade_on_only'],
                ['vis_ori', 'saccade'],
                # ['vis_on_only', 'vis'],
                # ['saccade_on_only', 'saccade'],
                # ['vis_on_only', 'saccade_on_only'],
                # ['vis', 'saccade'],
                # ['vis_on_and_saccade_on', 'vis_on_and_saccade_on_and_interaction']
            ],  # options for metrics are : 'vis_aligned_var_explained', 'saccade_aligned_var_explained', '
            metrics_to_compare=np.array([
                 ['vis_aligned_var_explained', 'vis_aligned_var_explained'],
                 ['saccade_aligned_var_explained', 'saccade_aligned_var_explained'],
                 ['vis_aligned_var_explained', 'saccade_aligned_var_explained'],
                 ['vis_aligned_var_explained', 'saccade_aligned_var_explained'],
                 ['vis_aligned_var_explained', 'saccade_aligned_var_explained'],
            ]),
            # metrics_to_compare=np.array([
            #     ['explained_var_per_X_set', 'explained_var_per_X_set'],
            #     ['explained_var_per_X_set', 'explained_var_per_X_set'],
            #     ['explained_var_per_X_set', 'explained_var_per_X_set'],
            #     ['explained_var_per_X_set', 'explained_var_per_X_set'],
            #     ['explained_var_per_X_set', 'explained_var_per_X_set'],
            # ]),
            custom_fig_addinfo='aligned',  # 'original', or 'aligned'
        ),
        'plot_regression_model_example_neurons': dict(
            neuron_type_to_plot='saccade_on',  # 'vis_on', 'saccade_on', 'saccade_dir', 'vis_ori'
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            file_types_to_load=['_windowVis', '_windowGray', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis'],
            models_to_plot=['vis_ori', 'saccade_on_only'],  # 'vis_on_only', 'saccade_on_only', 'vis_ori'
            model_colors = {'vis_on_only': 'orange',
                           'saccade_on_only': 'green',
                            'vis_ori': 'blue'},
            group_by_orientation=True,
        ),
        'fit_iterative_orientation_model': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis'],
            dataset='grating',
            neural_preprocessing_steps=['zscore'],  # 'zscore' is optional
        ),
        'compare_iterative_vs_normal_fit': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis'],
            dataset='grating',
            neural_preprocessing_steps=['zscore'],  # 'zscore' is optional
        ),
        'plot_num_saccade_per_ori': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis'],
            zscore_activity=True,
        ),
        'plot_pupil_data': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis'],
        )
    }

    print('Selected the following processes to run %s' % processes_to_run)

    for process in processes_to_run:

        assert process in available_processes

        if process == 'load_data':

            data = load_data(**process_params[process])

        if process == 'plot_data':

            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            for exp_id, exp_data in data.items():

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plot_grating_exp_data(exp_data, zscore_activity=process_params[process]['zscore_activity'])

                    fig_folder = process_params[process]['fig_folder']
                    fig_name = '%s_vis_experiment_overview' % exp_id
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


        if process == 'fit_regression_model':

            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])
            regression_results_folder = process_params[process]['regression_results_folder']
            neural_preprocessing_steps = process_params[process]['neural_preprocessing_steps']

            X_sets_to_compare = process_params[process]['X_sets_to_compare']
            # feature_set = ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir']

            for exp_id, exp_data in data.items():

                num_X_set = len(X_sets_to_compare.keys())
                num_neurons = np.shape(exp_data['_tracesVis'])[1]
                explained_var_per_X_set = np.zeros((num_neurons, num_X_set))
                vis_aligned_explained_var_per_X_set = np.zeros((num_neurons, num_X_set))
                saccade_aligned_explained_var_per_X_set = np.zeros((num_neurons, num_X_set))

                exp_regression_result = dict()
                exp_regression_result['X_sets_names'] = list(X_sets_to_compare.keys())
                Y_test_hat_per_X_set = []

                for n_X_set, (X_set_name, feature_set) in enumerate(X_sets_to_compare.items()):

                    if 'vis_ori_iterative' in feature_set:
                        train_test_split_method = 'half'
                        if train_test_split_method == 'half':
                            num_time_points = np.shape(Y)[0]
                            first_half_indices = np.arange(0, int(num_time_points / 2))
                            second_half_indices = np.arange(int(num_time_points / 2), num_time_points)

                            # Make sure they are the same length
                            if len(first_half_indices) < len(second_half_indices):
                                second_half_indices = second_half_indices[0:len(first_half_indices)]
                            else:
                                first_half_indices = first_half_indices[0:len(second_half_indices)]

                            train_indices = [first_half_indices, second_half_indices]
                            test_indices = [second_half_indices, first_half_indices]

                        elif train_test_split_method == 'n_fold_cv':
                            print('TODO: set up n-fold cross validation')
                    else:
                        train_indices = None
                        test_indices = None

                    X, Y = make_X_Y_for_regression(exp_data, feature_set=feature_set,
                                                   neural_preprocessing_steps=neural_preprocessing_steps,
                                                   train_indices=train_indices, test_indices=test_indices)

                    regression_result = fit_regression_model(X, Y)
                    regression_result = get_aligned_explained_variance(regression_result, exp_data)
                    explained_var_per_X_set[:, n_X_set] = np.mean(regression_result['explained_variance_per_cv_set'], axis=1)
                    vis_aligned_explained_var_per_X_set[:, n_X_set] = regression_result['vis_aligned_var_explained']
                    saccade_aligned_explained_var_per_X_set[:, n_X_set] = regression_result['saccade_aligned_var_explained']

                    Y_test_hat_per_X_set.append(regression_result['Y_test_hat_per_cv_set'])


                exp_regression_result['explained_var_per_X_set'] = explained_var_per_X_set
                exp_regression_result['Y_test'] = regression_result['Y_test_per_cv_set']
                exp_regression_result['vis_aligned_var_explained'] = vis_aligned_explained_var_per_X_set
                exp_regression_result['saccade_aligned_var_explained'] = saccade_aligned_explained_var_per_X_set
                # if exp_id == 'SS047_2015-12-03':
                #     pdb.set_trace()

                exp_regression_result['Y_test_hat'] = np.array(Y_test_hat_per_X_set)
                exp_regression_result['test_idx_per_cv_set'] = regression_result['test_idx_per_cv_set']


                regression_result_savename = '%s_regression_results.npz' % exp_id
                regression_result_savepath = os.path.join(regression_results_folder, regression_result_savename)
                np.savez(regression_result_savepath, **exp_regression_result)

        if process == 'fit_iterative_orientation_model':

            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])
            regression_results_folder = process_params[process]['regression_results_folder']
            neural_preprocessing_steps = process_params[process]['neural_preprocessing_steps']

            X_sets_to_compare = process_params[process]['X_sets_to_compare']
            # feature_set = ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir']

            for exp_id, exp_data in data.items():

                num_X_set = len(X_sets_to_compare.keys())
                num_neurons = np.shape(exp_data['_tracesVis'])[1]
                explained_var_per_X_set = np.zeros((num_neurons, num_X_set))
                exp_regression_result = dict()
                exp_regression_result['X_sets_names'] = list(X_sets_to_compare.keys())
                Y_test_hat_per_X_set = []

                for n_X_set, (X_set_name, feature_set) in enumerate(X_sets_to_compare.items()):
                    X, Y = make_X_Y_for_regression(exp_data, feature_set=feature_set,
                                                   neural_preprocessing_steps=neural_preprocessing_steps)

                    regression_result = fit_regression_model(X, Y)

                    explained_var_per_X_set[:, n_X_set] = np.mean(regression_result['explained_variance_per_cv_set'],
                                                                  axis=1)
                    Y_test_hat_per_X_set.append(regression_result['Y_test_hat_per_cv_set'])

                exp_regression_result['explained_var_per_X_set'] = explained_var_per_X_set
                exp_regression_result['Y_test'] = regression_result['Y_test_per_cv_set']

                # if exp_id == 'SS047_2015-12-03':
                #     pdb.set_trace()

                exp_regression_result['Y_test_hat'] = np.array(Y_test_hat_per_X_set)
                exp_regression_result['test_idx_per_cv_set'] = regression_result['test_idx_per_cv_set']

                regression_result_savename = '%s_regression_results.npz' % exp_id
                regression_result_savepath = os.path.join(regression_results_folder, regression_result_savename)
                # np.savez(regression_result_savepath, **exp_regression_result)



        if process == 'plot_regression_model_explained_var':

            regression_result_files = glob.glob(os.path.join(process_params[process]['regression_results_folder'],
                                                             '*npz'))
            X_sets_to_compare = process_params[process]['X_sets_to_compare']
            metrics_to_compare = process_params[process]['metrics_to_compare']
            custom_fig_addinfo = process_params[process]['custom_fig_addinfo']

            fig_folder = process_params[process]['fig_folder']
            text_size = 11

            for fpath in regression_result_files:

                regression_result = np.load(fpath)

                X_sets_names = regression_result['X_sets_names']

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plt.subplots(1, len(X_sets_to_compare))
                    fig.set_size_inches(len(X_sets_to_compare)*3, 3)
                    for n_comparison in np.arange(0, len(X_sets_to_compare)):

                        model_a = X_sets_to_compare[n_comparison][0]
                        model_b = X_sets_to_compare[n_comparison][1]
                        model_a_idx = np.where(X_sets_names == model_a)
                        model_b_idx = np.where(X_sets_names == model_b)

                        if process_params[process]['metrics_to_compare'] is not None:
                            model_a_metric = metrics_to_compare[n_comparison, 0]
                            model_b_metric = metrics_to_compare[n_comparison, 1]

                        else:
                            model_a_metric = 'explained_var_per_X_set'
                            model_b_metric = 'explained_var_per_X_set'

                        model_a_explained_var = regression_result[model_a_metric][:, model_a_idx]
                        model_b_explained_var = regression_result[model_b_metric][:, model_b_idx]

                        axs[n_comparison].scatter(model_a_explained_var, model_b_explained_var, color='black',
                                                  s=10)

                        both_model_explained_var = np.concatenate([model_a_explained_var,
                                                                  model_b_explained_var])

                        both_model_min = np.min(both_model_explained_var)
                        both_model_max = np.max(both_model_explained_var)
                        unity_vals = np.linspace(both_model_min, both_model_max, 100)
                        axs[n_comparison].axvline(0, linestyle='--', color='gray', alpha=0.5, lw=0.75, zorder=-1)
                        axs[n_comparison].axhline(0, linestyle='--', color='gray', alpha=0.5, lw=0.75, zorder=-2)
                        axs[n_comparison].plot(unity_vals, unity_vals, linestyle='--', color='gray', alpha=0.5)
                        axs[n_comparison].set_xlabel(model_a, size=text_size)
                        axs[n_comparison].set_ylabel(model_b, size=text_size)

                    exp_id_parts = os.path.basename(fpath).split('.')[0].split('_')
                    subject = exp_id_parts[0]
                    exp_date = exp_id_parts[1]

                    if custom_fig_addinfo is not None:
                        fig_name = '%s_%s_%s_explained_variance_per_X_set_comparison' % (subject, exp_date, custom_fig_addinfo)
                    else:
                        fig_name = '%s_%s_explained_variance_per_X_set_comparison' % (subject, exp_date)
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbbox_inches='tight')

                    plt.close(fig)
        if process == 'plot_regression_model_example_neurons':

            # TODO: currently this assumes the Y_test cv split consists of first half and second half of recording

            regression_result_files = glob.glob(os.path.join(process_params[process]['regression_results_folder'],
                                                             '*npz'))
            fig_folder = process_params[process]['fig_folder']

            models_to_plot = process_params[process]['models_to_plot']
            model_colors = process_params[process]['model_colors']
            group_by_orientation = process_params[process]['group_by_orientation']

            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            neuron_type_to_plot = process_params[process]['neuron_type_to_plot']
            num_example_neurons_to_plot = 10

            for fpath in regression_result_files:

                regression_result = np.load(fpath, allow_pickle=True)

                X_sets_names = regression_result['X_sets_names']
                explained_var_per_X_set = regression_result['explained_var_per_X_set']
                Y_test = regression_result['Y_test']  # num cv x num time points x num neurons
                Y_test_hat = regression_result['Y_test_hat'] # num model x num cv x num time points x num neurons
                Y = np.concatenate([Y_test[1], Y_test[0]])
                try:
                    Y_hat = np.concatenate([Y_test_hat[:, 1, :, :], Y_test_hat[:, 0, :, :]], axis=1)
                except:
                    pdb.set_trace()

                model_results_dict = {}

                for n_X_set in np.arange(0, len(X_sets_names)):
                    X_set_name = X_sets_names[n_X_set]
                    model_results_dict[X_set_name] = explained_var_per_X_set[:, n_X_set]

                model_result_df = pd.DataFrame.from_dict(model_results_dict)

                if neuron_type_to_plot == 'vis_on':
                    vis_on_neuron_df = model_result_df.loc[
                        (model_result_df['vis_on_only'] > 0) &
                        (model_result_df['vis_on_only'] > model_result_df['vis']) &
                        (model_result_df['vis_on_only'] > model_result_df['saccade_on_only'])
                    ]

                    neuron_df_sorted = vis_on_neuron_df.sort_values('vis_on_only')[::-1]
                elif neuron_type_to_plot == 'saccade_on':
                    saccade_on_neuron_df = model_result_df.loc[
                        (model_result_df['saccade_on_only'] > 0) &
                        (model_result_df['saccade_on_only'] > model_result_df['saccade']) &
                        (model_result_df['saccade_on_only'] > model_result_df['vis_on_only'])
                        ]

                    neuron_df_sorted = saccade_on_neuron_df.sort_values('saccade_on_only')[::-1]

                elif neuron_type_to_plot == 'vis_ori':
                    vis_ori_neuron_df = model_result_df.loc[
                        (model_result_df['vis_ori'] > 0) &
                        (model_result_df['vis_ori'] > model_result_df['vis_on_only']) &
                        (model_result_df['vis_ori'] > model_result_df['saccade_on_only'])
                    ]

                    neuron_df_sorted = vis_ori_neuron_df.sort_values('vis_ori')[::-1]

                if len(model_result_df) == 383:
                    pdb.set_trace()
                num_model = len(models_to_plot)
                linewidth = 1

                for n_neuron in np.arange(num_example_neurons_to_plot):

                    neuron_idx = neuron_df_sorted.index.values[n_neuron]

                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        gs = gridspec.GridSpec(2, 2)
                        fig = plt.figure()
                        ax1 = plt.subplot(gs[0, :])
                        ax2 = plt.subplot(gs[1, 0])
                        ax3 = plt.subplot(gs[1, 1])

                        # ax1.plot(Y_test[0, :, neuron_idx], color='black', label='Observed', lw=linewidth)
                        ax1.plot(Y[:, neuron_idx], color='black', label='Observed', lw=linewidth)

                        title_str = []

                        for model in models_to_plot:
                            color = model_colors[model]
                            model_idx = np.where(X_sets_names == model)[0][0]
                            # ax1.plot(Y_test_hat[model_idx, 0, :, neuron_idx], color=color,
                             #        label=model, lw=linewidth)
                            ax1.plot(Y_hat[model_idx, :, neuron_idx], color=color,
                                     label=model, lw=linewidth)

                            title_str.append('%s : %.3f' % (model, explained_var_per_X_set[neuron_idx, model_idx]))

                        ax1.set_title(title_str, size=9)
                        ax1.legend()
                        ax1.set_xlabel('Time', size=11)
                        ax1.set_ylabel('Activity', size=11)


                        # Align to stimulus onset and plot that
                        exp_id_parts = os.path.basename(fpath).split('.')[0].split('_')
                        subject = exp_id_parts[0]
                        exp_date = exp_id_parts[1]
                        exp_data = data['%s_%s' % (subject, exp_date)]
                        grating_intervals = exp_data['_gratingIntervals']
                        vis_exp_saccade_intervals = exp_data['_saccadeIntervalsVis']
                        vis_exp_times = exp_data['_windowVis'].flatten()


                        grating_onset_times = grating_intervals[:, 0]

                        feature_time_windows = {'vis_on': [-1.0, 3.0], 'vis_dir': [-1.0, 3.0],
                                                'saccade_on': [-1.0, 3.0], 'saccade_dir': [-1.0, 3.0]}

                        feature_time_window = feature_time_windows['vis_on']
                        subset_onset_time = grating_onset_times[
                            (grating_onset_times + feature_time_window[1])  < vis_exp_times[-1]
                        ]

                        num_trials = len(subset_onset_time)
                        sec_per_time_samples = np.mean(np.diff(vis_exp_times))
                        num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
                        Y_grating_onset_aligned = np.zeros((num_trials, num_sample_in_window))
                        Y_hat_grating_onset_aligned = np.zeros((num_model, num_trials, num_sample_in_window))

                        grating_id_per_trial = exp_data['_gratingIds'] - 1  # matab 1 indexing to python 0 indexing
                        id_to_grating_orientations = exp_data['_gratingIdDirections']
                        grating_orientation_per_trial = [id_to_grating_orientations[int(x)][0] for x in
                                                         grating_id_per_trial]

                        for n_trial, onset_time in enumerate(subset_onset_time):

                            start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[0])))
                            end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[1])))
                            samples_to_get = np.arange(start_sample, end_sample)
                            samples_to_get = samples_to_get[0:num_sample_in_window]

                            Y_grating_onset_aligned[n_trial, :] = Y[samples_to_get, neuron_idx]

                            for n_X_set, X_set in enumerate(models_to_plot):
                                model_idx = np.where(X_sets_names == X_set)[0][0]
                                Y_hat_grating_onset_aligned[n_X_set, n_trial, :] = Y_hat[model_idx, samples_to_get, neuron_idx]

                        mean_Y_grating_onset_aligned = np.mean(Y_grating_onset_aligned, axis=0)
                        mean_Y_hat_grating_onset_aligned = np.mean(Y_hat_grating_onset_aligned, axis=1)
                        std_Y = np.std(Y_grating_onset_aligned, axis=0)
                        sem_Y = std_Y / np.sqrt(num_trials)
                        peri_event_time = np.linspace(feature_time_window[0], feature_time_window[1], num_sample_in_window)


                        ax2.plot(peri_event_time, mean_Y_grating_onset_aligned, color='black', label='observed', lw=linewidth)
                        y1 = mean_Y_grating_onset_aligned + sem_Y
                        y2 = mean_Y_grating_onset_aligned - sem_Y
                        ax2.fill_between(peri_event_time, y1, y2, color='gray', lw=0, alpha=0.3)
                        ax2.set_title('Visual stimuli onset', size=11)

                        for n_model, model in enumerate(models_to_plot):
                            color = model_colors[model]
                            ax2.plot(peri_event_time, mean_Y_hat_grating_onset_aligned[n_model, :], color=color,
                                     label=model, lw=linewidth)

                        # Align to saccade onset and plot that
                        feature_time_window = feature_time_windows['saccade_on']
                        vis_exp_saccade_onset_frames = vis_exp_saccade_intervals[:, 0].astype(int)

                        vis_exp_saccade_onset_times = vis_exp_times[vis_exp_saccade_onset_frames]
                        subset_vis_exp_saccade_onset_times = vis_exp_saccade_onset_times[
                            (vis_exp_saccade_onset_times + feature_time_window[1] < vis_exp_times[-1]) &
                            (vis_exp_saccade_onset_times + feature_time_window[0] > vis_exp_times[0])
                        ]
                        n_saccade_trials = len(subset_vis_exp_saccade_onset_times)

                        Y_vis_exp_saccade_aligned = np.zeros((n_saccade_trials, num_sample_in_window))
                        Y_hat_vis_exp_saccade_aligned = np.zeros((num_model, n_saccade_trials, num_sample_in_window))

                        for n_saccade_time, onset_time in enumerate(subset_vis_exp_saccade_onset_times):

                            start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[0])))
                            end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[1])))
                            samples_to_get = np.arange(start_sample, end_sample)
                            samples_to_get = samples_to_get[0:num_sample_in_window]

                            Y_vis_exp_saccade_aligned[n_saccade_time, :] = Y[samples_to_get, neuron_idx]

                            for n_X_set, X_set in enumerate(models_to_plot):
                                model_idx = np.where(X_sets_names == X_set)[0][0]
                                Y_hat_vis_exp_saccade_aligned[n_X_set, n_saccade_time, :] = Y_hat[model_idx, samples_to_get, neuron_idx]

                        mean_Y_vis_exp_saccade_aligned = np.mean(Y_vis_exp_saccade_aligned, axis=0)
                        mean_Y_hat_vis_exp_saccade_aligned = np.mean(Y_hat_vis_exp_saccade_aligned, axis=1)
                        std_Y_saccade = np.std(Y_vis_exp_saccade_aligned, axis=0)
                        sem_Y_saccade = std_Y_saccade / np.sqrt(n_saccade_trials)
                        peri_event_time = np.linspace(feature_time_window[0], feature_time_window[1],
                                                      num_sample_in_window)

                        ax3.plot(peri_event_time, mean_Y_vis_exp_saccade_aligned, color='black', label='observed',
                                 lw=linewidth)
                        y1 = mean_Y_vis_exp_saccade_aligned + sem_Y_saccade
                        y2 = mean_Y_vis_exp_saccade_aligned - sem_Y_saccade
                        ax3.fill_between(peri_event_time, y1, y2, color='gray', lw=0, alpha=0.3)

                        for n_model, model in enumerate(models_to_plot):
                            color = model_colors[model]
                            ax3.plot(peri_event_time, mean_Y_hat_vis_exp_saccade_aligned[n_model, :], color=color,
                                     label=model, lw=linewidth)


                        ax3.set_title('Saccade onset', size=11)

                        ax2.set_ylabel('Activity', size=11)
                        ax2.set_xlabel('Peri-event time (s)', size=11)
                        ax3.set_xlabel('Peri-event time (s)', size=11)

                        fig_name = '%s_%s_example_%s_neuron_%.f' % (subject, exp_date, neuron_type_to_plot, neuron_idx)
                        fig.tight_layout()
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


                        plt.close(fig)

                        if group_by_orientation:

                            fig, axs = plt.subplots(3, 4, sharex=True, sharey=True)

                            unique_orientations = np.unique(grating_orientation_per_trial)
                            unique_orientations = unique_orientations[~np.isnan(unique_orientations)]
                            grating_orientation_per_trial = np.array(grating_orientation_per_trial)
                            grating_orientation_per_trial_subset = grating_orientation_per_trial[
                                (grating_onset_times + feature_time_window[1]) < vis_exp_times[-1]
                            ]

                            for n_ori, ori_to_get in enumerate(unique_orientations):

                                subset_trial = np.where(grating_orientation_per_trial_subset == ori_to_get)[0]
                                Y_grating_onset_aligned_given_ori = Y_grating_onset_aligned[subset_trial, :]
                                Y_hat_grating_onset_aligned_given_ori = Y_hat_grating_onset_aligned[:, subset_trial, :]

                                mean_Y_grating_onset_aligned_given_ori = np.mean(Y_grating_onset_aligned_given_ori, axis=0)
                                mean_Y_hat_grating_onset_aligned_given_ori = np.mean(Y_hat_grating_onset_aligned_given_ori, axis=1)

                                axs.flatten()[n_ori].plot(peri_event_time, mean_Y_grating_onset_aligned_given_ori, color='black', label='observed',
                                         lw=linewidth)


                                std_Y = np.std(Y_grating_onset_aligned_given_ori, axis=0)
                                sem_Y = std_Y / np.sqrt(num_trials)
                                y1 = mean_Y_grating_onset_aligned_given_ori + sem_Y
                                y2 = mean_Y_grating_onset_aligned_given_ori - sem_Y

                                axs.flatten()[n_ori].fill_between(peri_event_time, y1, y2, color='gray', lw=0, alpha=0.3)
                                axs.flatten()[n_ori].set_title('Ori: %.f' % ori_to_get, size=11)

                                for n_model, model in enumerate(models_to_plot):
                                    color = model_colors[model]
                                    axs.flatten()[n_ori].plot(peri_event_time,
                                             mean_Y_hat_grating_onset_aligned_given_ori[n_model, :], color=color,
                                             label=model, lw=linewidth)

                            fig.text(0.5, 0, 'Peri-stimulus time (s)', size=11, ha='center')
                            fig.text(0, 0.5, 'Activity', size=11, va='center', rotation=90)

                            fig_name = '%s_%s_example_%s_neuron_%.f_ori_tuning' % (
                            subject, exp_date, neuron_type_to_plot, neuron_idx)
                            fig.tight_layout()
                            fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                            plt.close(fig)
        if process == 'compare_iterative_vs_normal_fit':

            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])
            fig_folder = process_params[process]['fig_folder']
            regression_results_folder = process_params[process]['regression_results_folder']
            neural_preprocessing_steps = process_params[process]['neural_preprocessing_steps']

            for exp_id, exp_data in data.items():

                train_test_split_method = 'half'

                num_time_points = len(exp_data['_windowVis'].flatten())

                if train_test_split_method == 'half':
                    first_half_indices = np.arange(0, int(num_time_points / 2))
                    second_half_indices = np.arange(int(num_time_points / 2), num_time_points)

                    # Make sure they are the same length
                    if len(first_half_indices) < len(second_half_indices):
                        second_half_indices = second_half_indices[0:len(first_half_indices)]
                    else:
                        first_half_indices = first_half_indices[0:len(second_half_indices)]

                    train_indices = [first_half_indices, second_half_indices]
                    test_indices = [second_half_indices, first_half_indices]

                elif train_test_split_method == 'n_fold_cv':
                    print('TODO: set up n-fold cross validation')

                train_orientation_activity_matrix, test_orientation_activity_matrix = get_ori_train_test_data(
                    exp_data, time_window=[0, 3], method='align_then_split',
                                        neural_preprocessing_steps=['zscore'], check_for_nans=True,
                                        train_indices=train_indices, test_indices=test_indices)

                num_cv = len(train_orientation_activity_matrix)
                num_neurons = np.shape(train_orientation_activity_matrix)[2]
                cv_model_per_cv = np.zeros((2, num_neurons, num_cv))

                for cv_idx in np.arange(num_cv):

                    train_set = train_orientation_activity_matrix[cv_idx]
                    test_set = test_orientation_activity_matrix[cv_idx]
                    svd_train_fit = fit_all_neuron_ori(orientation_activity_matrix=train_set, num_iter=3, method='svd')
                    mean_train_fit = fit_all_neuron_ori(orientation_activity_matrix=train_set, num_iter=3, method='mean')

                    test_set_per_neuron = np.swapaxes(test_set, 1, 2).reshape(-1, num_neurons)
                    svd_train_fit_per_neuron = np.swapaxes(svd_train_fit, 1, 2).reshape(-1, num_neurons)
                    mean_train_fit_per_neuron = np.swapaxes(mean_train_fit, 1, 2).reshape(-1, num_neurons)

                    svd_explained_var = sklmetrics.explained_variance_score(
                        test_set_per_neuron, svd_train_fit_per_neuron, multioutput='raw_values'
                    )

                    mean_explained_var = sklmetrics.explained_variance_score(
                        test_set_per_neuron, mean_train_fit_per_neuron, multioutput='raw_values'
                    )

                    cv_model_per_cv[0, :, cv_idx] = mean_explained_var
                    cv_model_per_cv[1, :, cv_idx] = svd_explained_var

                    # Plot some examples
                    activity_matrices = np.stack([
                        train_set,
                        test_set,
                        svd_train_fit
                    ])

                    """
                    for neuron_idx in np.arange(0, 10):
                        neuron_idx = 0
                        fig, axs = plot_neuron_orientation_tuning_fit(activity_matrices[:, :, neuron_idx, :], labels=['Observed (train)', 'Observed (test)', 'Fitted'],
                                           colors=['black', 'red', 'gray'],
                                           fig=None, axs=None)

                        fig_folder = '/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/test_iterative_fits'
                        fig_name = 'neuron_%.f_final_output_SVD_way' % neuron_idx
                        plt.close(fig)
                        # fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                    # axs.flatten()[-1].legend(bbox_to_anchor=(1.13, 0.5))
                    """
                # Plot variance explained by mean model vs. SVD model

                mean_cv_model_per_cv = np.mean(cv_model_per_cv, axis=-1)
                all_model_min = np.min(mean_cv_model_per_cv.flatten())
                all_model_max = np.max(mean_cv_model_per_cv.flatten())
                unity_vals = np.linspace(all_model_min, all_model_max, 100)

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(4, 4)
                    ax.set_xlabel('Mean model', size=11)
                    ax.set_ylabel('SVD rank 1 approximation', size=11)
                    ax.set_title('Variance explained', size=11)
                    ax.plot(unity_vals, unity_vals, linestyle='--', color='gray', zorder=3)
                    ax.axvline(0, linestyle='--', lw=1, color='gray', zorder=1)
                    ax.axhline(0, linestyle='--', lw=1, color='gray', zorder=2)
                    ax.scatter(mean_cv_model_per_cv[0, :], mean_cv_model_per_cv[1, :], s=4, color='black', zorder=4)

                    ax.set_xlim([-1, 1])
                    ax.set_ylim([-1, 1])
                    ax.set_title(exp_id, size=11)
                    fig_name = '%s_svd_vs_mean_model_explained_variance.png' % (exp_id)
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


        if process == 'plot_num_saccade_per_ori':

            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            for exp_id, exp_data in data.items():


                vis_exp_times, vis_exp_saccade_onset_times, grating_onset_times, saccade_dirs, grating_orientation_per_trial = get_vis_and_saccade_times(
                    exp_data)

                num_saccade_per_grating = get_num_saccade_per_grating(vis_exp_saccade_onset_times,
                                                                      grating_onset_times,
                                                                      grating_orientation_per_trial)

        if process == 'plot_pupil_data':


            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            for exp_id, exp_data in data.items():

                fig, ax = plot_pupil_data(data)







if __name__ == '__main__':
    main()