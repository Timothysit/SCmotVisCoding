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

            if len(file_paths) != 1:
                print('0 or more than 1 path found, please debug')
                pdb.set_trace()

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


def get_vis_and_saccade_times(exp_data, exp_type='grating', exclude_saccade_on_vis_exp=False):
    """
    Parameters
    ----------
    exp_data : dict
    exp_type : str
        type of experiment
    """

    if exp_type == 'grating':
        exp_times = exp_data['_windowVis'].flatten()
        neural_activity = exp_data['_tracesVis']
        grating_intervals = exp_data['_gratingIntervals']
        vis_exp_saccade_intervals = exp_data['_saccadeIntervalsVis'].astype(int)
        exp_saccade_onset_times = exp_times[vis_exp_saccade_intervals[:, 0]]
        saccade_dirs = exp_data['_saccadeVisDir'].flatten()
        saccade_dirs[saccade_dirs == 0] = -1
        grating_id_per_trial = exp_data['_gratingIds'] - 1  # matab 1 indexing to python 0 indexing
        id_to_grating_orientations = exp_data['_gratingIdDirections']
        grating_orientation_per_trial = [id_to_grating_orientations[int(x)][0] for x in grating_id_per_trial]
        pupil_size = exp_data['_pupilSizeVis'].flatten()  # size : (nTimePoints, )
        # TODO: use a smoothing imputation method instead
        pupil_size[np.isnan(pupil_size)] = np.nanmean(pupil_size)

        grating_onset_times = grating_intervals[:, 0]

    elif exp_type == 'gray':

        exp_times = exp_data['_windowGray'].flatten()
        neural_activity = exp_data['_tracesGray']
        exp_saccade_intervals = exp_data['_onsetOffset'].astype(int)
        exp_saccade_onset_times = exp_times[exp_saccade_intervals[:, 0]]

        gray_exp_saccade_dirs = exp_data['_trial_Dir'].flatten()
        gray_exp_saccade_dirs[gray_exp_saccade_dirs == 0] = -1
        saccade_dirs = gray_exp_saccade_dirs

        grating_onset_times = np.array([])
        grating_orientation_per_trial = np.array([])
        # TODO: do pupil size


    elif exp_type == 'both':

        # Neural activity
        vis_exp_neural_activity = exp_data['_tracesVis']
        gray_exp_neural_activity = exp_data['_tracesGray']

        neural_activity = np.concatenate([vis_exp_neural_activity, gray_exp_neural_activity], axis=0)

        # Times

        vis_exp_times = exp_data['_windowVis'].flatten()
        gray_exp_times = exp_data['_windowGray'].flatten()
        gray_exp_times_w_offset = gray_exp_times + vis_exp_times[-1]
        exp_times = np.concatenate([vis_exp_times, gray_exp_times_w_offset])

        # Saccade
        vis_exp_saccade_intervals = exp_data['_saccadeIntervalsVis'].astype(int)
        vis_exp_saccade_onset_times = vis_exp_times[vis_exp_saccade_intervals[:, 0]]

        gray_exp_saccade_intervals = exp_data['_onsetOffset'].astype(int)
        gray_exp_saccade_onset_times = gray_exp_times[gray_exp_saccade_intervals[:, 0]]
        gray_exp_saccade_onset_times_w_offset = gray_exp_saccade_onset_times + vis_exp_times[-1]

        if exclude_saccade_on_vis_exp:
            exp_saccade_onset_times = gray_exp_saccade_onset_times_w_offset
        else:
            exp_saccade_onset_times = np.concatenate([vis_exp_saccade_onset_times,
                                                  gray_exp_saccade_onset_times_w_offset])

        vis_saccade_dirs = exp_data['_saccadeVisDir'].flatten()
        vis_saccade_dirs[vis_saccade_dirs == 0] = -1
        gray_exp_saccade_dirs = exp_data['_trial_Dir'].flatten()
        gray_exp_saccade_dirs[gray_exp_saccade_dirs == 0] = -1

        if exclude_saccade_on_vis_exp:
            saccade_dirs = gray_exp_saccade_dirs
        else:
            saccade_dirs = np.concatenate([vis_saccade_dirs, gray_exp_saccade_dirs])


        # Grating
        grating_intervals = exp_data['_gratingIntervals']
        grating_id_per_trial = exp_data['_gratingIds'] - 1  # matab 1 indexing to python 0 indexing
        id_to_grating_orientations = exp_data['_gratingIdDirections']
        grating_orientation_per_trial = [id_to_grating_orientations[int(x)][0] for x in grating_id_per_trial]
        grating_onset_times = grating_intervals[:, 0]

    else:
        print('WARNING: no valid exp_type specified')

    return exp_times, exp_saccade_onset_times, grating_onset_times, saccade_dirs, grating_orientation_per_trial


def make_X_Y_for_regression(exp_data, feature_set=['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir'],
                            feature_time_windows={'vis_on': [-1.0, 3.0], 'vis_dir': [-1.0, 3.0], 'vis_ori': [-1.0, 3.0],
                                                  'saccade_on': [-1.0, 3.0], 'saccade_dir': [-1.0, 3.0],
                                                  'vis_on_saccade_on': [-1.0, 3.0], 'vis_ori_iterative': [0, 3.0]},
                            neural_preprocessing_steps=['zscore'], check_for_nans=True, exp_type='grating',
                            exclude_saccade_on_vis_exp=False,
                            train_indices=None, test_indices=None,
                            return_trial_type=False, pupil_preprocessing_steps=[]):
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

    if exp_type == 'grating':
        vis_exp_times = exp_data['_windowVis'].flatten()
        exp_times = exp_data['_windowVis'].flatten()
        neural_activity = exp_data['_tracesVis']
        grating_intervals = exp_data['_gratingIntervals']
        grating_onset_times = grating_intervals[:, 0]
        vis_exp_saccade_intervals = exp_data['_saccadeIntervalsVis'].astype(int)
        exp_saccade_onset_times = exp_times[vis_exp_saccade_intervals[:, 0]]
        saccade_dirs = exp_data['_saccadeVisDir'].flatten()
        saccade_dirs[saccade_dirs == 0] = -1
        grating_id_per_trial = exp_data['_gratingIds'] - 1  # matab 1 indexing to python 0 indexing
        id_to_grating_orientations = exp_data['_gratingIdDirections']
        grating_orientation_per_trial = [id_to_grating_orientations[int(x)][0] for x in grating_id_per_trial]
        pupil_size = exp_data['_pupilSizeVis'].flatten()  # size : (nTimePoints, )
        pupil_size = impute_time_series(pupil_size, method='interpolate')

        if 'zscore' in pupil_preprocessing_steps:
            print('z-scoring pupil size')
            pupil_size = sstats.zscore(pupil_size)

    elif exp_type == 'gray':
        vis_exp_times = exp_data['_windowVis'].flatten()
        exp_times = exp_data['_windowGray'].flatten()
        neural_activity = exp_data['_tracesGray']
        exp_saccade_intervals = exp_data['_onsetOffset'].astype(int)
        exp_saccade_onset_times = exp_times[exp_saccade_intervals[:, 0]]
        gray_exp_saccade_dirs = exp_data['_trial_Dir'].flatten()
        gray_exp_saccade_dirs[gray_exp_saccade_dirs == 0] = -1
        saccade_dirs = gray_exp_saccade_dirs
        grating_onset_times = []
        grating_orientation_per_trial = []
        pupil_size = exp_data['_pupilSizeGray'].flatten()
        pupil_size = impute_time_series(pupil_size, method='interpolate')
        if 'zscore' in pupil_preprocessing_steps:
            print('z-scoring pupil size')
            pupil_size = sstats.zscore(pupil_size)

    elif exp_type == 'both':

        # Neural activity
        vis_exp_neural_activity = exp_data['_tracesVis']
        gray_exp_neural_activity = exp_data['_tracesGray']

        num_vis_exp_nans = np.sum(np.isnan(vis_exp_neural_activity))
        num_gray_exp_nans = np.sum(np.isnan(gray_exp_neural_activity))
        if num_vis_exp_nans > 0:
            print('WARNING: found %.f NaNs in vis exp data, imputing with mean for now' % num_vis_exp_nans)
            for neuron_idx in np.arange(np.shape(vis_exp_neural_activity)[1]):
                neuron_trace = vis_exp_neural_activity[:, neuron_idx]
                neuron_trace[np.isnan(neuron_trace)] = np.nanmean(neuron_trace)
                vis_exp_neural_activity[:, neuron_idx] = neuron_trace
        if num_gray_exp_nans > 0:
            print('WARNING: found %.f NaNs in gray exp data, imputing with mean for now' % num_gray_exp_nans)
            for neuron_idx in np.arange(np.shape(gray_exp_neural_activity)[1]):
                neuron_trace = gray_exp_neural_activity[:, neuron_idx]
                neuron_trace[np.isnan(neuron_trace)] = np.nanmean(neuron_trace)
                gray_exp_neural_activity[:, neuron_idx] = neuron_trace

        # Times
        vis_exp_times = exp_data['_windowVis'].flatten()
        gray_exp_times = exp_data['_windowGray'].flatten()

        one_sample_time = np.mean(np.diff(vis_exp_times))
        gray_exp_start_time = gray_exp_times[0]

        gray_exp_times_w_offset = gray_exp_times + vis_exp_times[-1] + one_sample_time - gray_exp_start_time
        exp_times = np.concatenate([vis_exp_times, gray_exp_times_w_offset])

        # Saccade
        vis_exp_saccade_intervals = exp_data['_saccadeIntervalsVis'].astype(int)
        vis_exp_saccade_onset_times = vis_exp_times[vis_exp_saccade_intervals[:, 0]]

        gray_exp_saccade_intervals = exp_data['_onsetOffset'].astype(int)
        gray_exp_saccade_onset_times = gray_exp_times[gray_exp_saccade_intervals[:, 0]]
        gray_exp_saccade_onset_times_w_offset = gray_exp_saccade_onset_times + vis_exp_times[-1] + one_sample_time - gray_exp_start_time

        if exclude_saccade_on_vis_exp:
            exp_saccade_onset_times = gray_exp_saccade_onset_times_w_offset
        else:
            exp_saccade_onset_times = np.concatenate([vis_exp_saccade_onset_times,
                                                      gray_exp_saccade_onset_times_w_offset])

        vis_saccade_dirs = exp_data['_saccadeVisDir'].flatten()
        vis_saccade_dirs[vis_saccade_dirs == 0] = -1
        gray_exp_saccade_dirs = exp_data['_trial_Dir'].flatten()
        gray_exp_saccade_dirs[gray_exp_saccade_dirs == 0] = -1
        if exclude_saccade_on_vis_exp:
            saccade_dirs = gray_exp_saccade_dirs
        else:
            saccade_dirs = np.concatenate([vis_saccade_dirs, gray_exp_saccade_dirs])


        # Grating
        grating_intervals = exp_data['_gratingIntervals']
        grating_onset_times = grating_intervals[:, 0]
        grating_id_per_trial = exp_data['_gratingIds'] - 1  # matab 1 indexing to python 0 indexing
        id_to_grating_orientations = exp_data['_gratingIdDirections']
        grating_orientation_per_trial = [id_to_grating_orientations[int(x)][0] for x in grating_id_per_trial]

        # Pupil size
        vis_exp_pupil_size = exp_data['_pupilSizeVis'].flatten()  # size : (nTimePoints, )
        vis_exp_pupil_size = impute_time_series(vis_exp_pupil_size, method='interpolate')

        gray_exp_pupil_size = exp_data['_pupilSizeGray'].flatten()
        gray_exp_pupil_size = impute_time_series(gray_exp_pupil_size, method='interpolate')

        if 'zscore' in pupil_preprocessing_steps:
            print('z-scoring pupil size')
            vis_exp_pupil_size = sstats.zscore(vis_exp_pupil_size)
            gray_exp_pupil_size = sstats.zscore(gray_exp_pupil_size)

        pupil_size = np.concatenate([vis_exp_pupil_size, gray_exp_pupil_size])

        # Neural activity
        if 'z_score_separately' in neural_preprocessing_steps:
            if 'zscore_w_baseline' in neural_preprocessing_steps:
                print('Zscoring using baseline activity separately in each experiment')
                vis_iti_times = vis_exp_times.copy()
                time_after_vis_to_exclude = 3
                time_after_saccade_to_exclude = 2
                for onset_time in grating_onset_times:
                    vis_iti_times[(vis_exp_times >= onset_time) & (vis_exp_times <= onset_time + time_after_vis_to_exclude)] = np.nan
                for saccade_time in vis_exp_saccade_onset_times:
                    vis_iti_times[(vis_exp_times >= saccade_time) & (
                                vis_exp_times <= saccade_time + time_after_saccade_to_exclude)] = np.nan

                vis_iti_times_idx = np.where(~np.isnan(vis_iti_times))[0]
                vis_exp_iti_neural_activity_mu = np.mean(vis_exp_neural_activity[vis_iti_times_idx, :], axis=0)
                vis_exp_iti_neural_activity_std = np.std(vis_exp_neural_activity[vis_iti_times_idx, :], axis=0)
                vis_exp_neural_activity = (vis_exp_neural_activity - vis_exp_iti_neural_activity_mu) / vis_exp_iti_neural_activity_std

                # Gray screen experiment z-scoring using ITI and baseline activity
                gray_exp_iti_times = gray_exp_times.copy()
                for saccade_time in gray_exp_saccade_onset_times:
                    gray_exp_iti_times[(gray_exp_iti_times >= saccade_time) & (
                            gray_exp_iti_times <= saccade_time + time_after_saccade_to_exclude)] = np.nan
                gray_exp_iti_times_idx = np.where(~np.isnan(gray_exp_iti_times))[0]
                gray_exp_iti_neural_activity_mu = np.mean(gray_exp_neural_activity[gray_exp_iti_times_idx, :], axis=0)
                gray_exp_iti_neural_activity_std = np.std(gray_exp_neural_activity[gray_exp_iti_times_idx, :], axis=0)
                gray_exp_neural_activity = (gray_exp_neural_activity - gray_exp_iti_neural_activity_mu) / gray_exp_iti_neural_activity_std

            else:
                vis_exp_neural_activity = sstats.zscore(vis_exp_neural_activity, axis=0)
                gray_exp_neural_activity = sstats.zscore(gray_exp_neural_activity, axis=0)

        neural_activity = np.concatenate([vis_exp_neural_activity, gray_exp_neural_activity], axis=0)


    else:
        print('WARNING: no valid exp_type specified')

    # print('Min np.diff(exp_times) %.4f' % np.min(np.diff(exp_times)))
    # print('Max np.diff(exp_times) %.4f' % np.max(np.diff(exp_times)))

    if check_for_nans:
        if np.sum(np.isnan(neural_activity)) > 0:
            print('Detected NaNs, assuming this is along the time dimension and removing time frames with NaNs')
            subset_time_frame = ~np.isnan(np.sum(neural_activity, axis=1))
            neural_activity = neural_activity[subset_time_frame, :]
            exp_times = exp_times[subset_time_frame]

    sec_per_time_samples = np.mean(np.diff(exp_times))
    num_time_samples = int(len(exp_times))

    # Make target matrix Y
    Y = neural_activity
    if ('zscore' in neural_preprocessing_steps) and ('z_score_separately' not in neural_preprocessing_steps):
        Y  = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    elif ('zscore_w_baseline' in neural_preprocessing_steps) and ('z_score_separately' not in neural_preprocessing_steps):
        pdb.set_trace()

    # TODO: need to z-score with baseline

    # Make X
    feature_matrices = []
    feature_indices = {}

    feature_idx_counter = 0

    for feature_name in feature_set:

        if feature_name == 'bias':

            feature_mat = np.zeros((num_time_samples, 1))

        elif feature_name == 'vis_on':
            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            # grating_onset_times = grating_intervals[:, 0]
            # grating_offset_times = grating_intervals[:, 1]

            if len(grating_onset_times) != 0:
                subset_onset_time = grating_onset_times[
                    (grating_onset_times + feature_time_window[1]) < vis_exp_times[-1]
                    ]
            else:
                subset_onset_time = []

            for onset_time in subset_onset_time:

                # onset_sample = np.argmin(np.abs(vis_exp_times - onset_time))
                start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                feature_mat[row_idx, col_idx] = 1


        elif feature_name == 'vis_dir':
            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            # grating_onset_times = grating_intervals[:, 0]
            # grating_offset_times = grating_intervals[:, 1]

            subset_onset_time = grating_onset_times[
                (grating_onset_times + feature_time_window[1]) < exp_times[-1]
                ]

            for n_trial, onset_time in enumerate(subset_onset_time):

                # onset_sample = np.argmin(np.abs(vis_exp_times - onset_time))
                start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
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
            # grating_onset_times = grating_intervals[:, 0]
            # grating_offset_times = grating_intervals[:, 1]

            # Note we still use vis exp times here because the vis and gray screen experiments are not truncated exactly
            # one after another ... there is some emtpy time in between... not sure if that matters
            if len(grating_onset_times) != 0:
                subset_trial_idx = np.where((grating_onset_times + feature_time_window[1]) < vis_exp_times[-1])[0]
            else:
                subset_trial_idx = []

            for n_trial, trial_idx in enumerate(subset_trial_idx):

                onset_time = grating_onset_times[trial_idx]
                # onset_sample = np.argmin(np.abs(vis_exp_times - onset_time))
                start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
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

            subset_exp_saccade_onset_times = exp_saccade_onset_times[
                (exp_saccade_onset_times + feature_time_window[1] < exp_times[-1]) &
                (exp_saccade_onset_times + feature_time_window[0] > exp_times[0])
                ]

            for n_saccade_time, onset_time in enumerate(subset_exp_saccade_onset_times):

                start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                feature_mat[row_idx, col_idx] = 1


        elif feature_name == 'saccade_dir':
            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            subset_exp_saccade_onset_times = exp_saccade_onset_times[
                (exp_saccade_onset_times + feature_time_window[1] < exp_times[-1]) &
                (exp_saccade_onset_times + feature_time_window[0] > exp_times[0])
                ]

            for n_saccade_time, onset_time in enumerate(subset_exp_saccade_onset_times):

                if onset_time + feature_time_window[1] > exp_times[-1]:
                    print('onset time exceeds experiment time, skipping')
                    continue

                start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
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

            for n_saccade_time, onset_time in enumerate(exp_saccade_onset_times):

                if onset_time + feature_time_window[1] > exp_times[-1]:
                    print('onset time exceeds experiment time, skipping')
                    continue
                if onset_time + feature_time_window[0] < exp_times[0]:
                    print('onset time is lower than experiment start time, skipping')
                    continue

                start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
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

                if onset_time + feature_time_window[1] > exp_times[-1]:
                    print('onset time exceeds experiment time, skipping')
                    continue

                # onset_sample = np.argmin(np.abs(vis_exp_times - onset_time))
                start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
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

            subset_trial_idx = np.where((grating_onset_times + feature_time_window[1]) < exp_times[-1])[0]
            grating_onset_times_subset = grating_onset_times[subset_trial_idx]
            grating_orientation_per_trial_subset = grating_orientation_per_trial[subset_trial_idx]

            num_neurons = np.shape(Y)[1]
            Y_aligned_to_vis_onset = np.zeros((num_time_sample_per_ori, num_neurons, len(grating_onset_times_subset)))

            for n_onset, onset_time in enumerate(grating_onset_times_subset):

                start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
                subset_idx = np.arange(start_sample, end_sample)
                if len(subset_idx) > num_time_sample_per_ori:
                    subset_idx = subset_idx[0:num_time_sample_per_ori]

                Y_aligned_to_vis_onset[:, :, n_onset] = Y[subset_idx, :]

            for train_idx_set in train_indices:
                train_supported_time = exp_times[train_idx_set]
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
                train_supported_time = exp_times[test_idx_set]
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
        elif feature_name == 'pupil_size':

            feature_mat = pupil_size.reshape(-1, 1)


        else:
            print('%s is not a valid feature name' % feature_name)

        feature_matrices.append(feature_mat)

        # Add the feature indices
        feature_indices[feature_name] = np.arange(feature_idx_counter, feature_idx_counter + np.shape(feature_mat)[1])
        feature_idx_counter += np.shape(feature_mat)[1]

    try:
        X = np.concatenate(feature_matrices, axis=1)
    except:
        pdb.set_trace()

    if return_trial_type:
        return X, Y, grating_orientation_per_trial, saccade_dirs, feature_indices
    else:
        return X, Y, feature_indices


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
                         custom_train_indices=None, custom_test_indices=None,
                         performance_metrics=['explained_variance', 'r2']):
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
    performance_metrics : list of str
        list of performance metrics to use
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
        num_time_points = np.shape(Y)[0]
        window_partition_points = np.linspace(0, num_time_points, n_cv_folds+1).astype(int)
        start_indices = window_partition_points[0:n_cv_folds]
        end_indices = window_partition_points[1:n_cv_folds+1]
        all_partition_time_indices = [np.arange(x, y) for (x, y) in zip(start_indices, end_indices)]

        train_indices = []
        test_indices = []
        for partition_to_exclude in np.arange(0, n_cv_folds):
            train_indices.append(
                np.concatenate([x for (n_part, x) in enumerate(all_partition_time_indices) if n_part != partition_to_exclude])
            )
            test_indices.append(all_partition_time_indices[partition_to_exclude])

    if model_type == 'Ridge':
        model = sklinear.Ridge(alpha=1.0, fit_intercept=False)

    num_cv_set = len(train_indices)
    num_neurons = np.shape(Y)[1]
    explained_variance_per_cv_set = np.zeros((num_neurons, num_cv_set))

    if 'r2' in performance_metrics:
        r2_per_cv_set = np.zeros((num_neurons, num_cv_set))

    Y_test_hat_per_cv_set = []
    Y_test_per_cv_set = []
    test_idx_per_set = []
    weights_per_cv_set = []

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

        if 'r2' in performance_metrics:
            r2 = sklmetrics.r2_score(y_true=Y_test, y_pred=Y_test_hat, multioutput='raw_values')
            r2_per_cv_set[:, n_cv_set] = r2

        # Save model weights
        model_weights = model.coef_
        weights_per_cv_set.append(model_weights)

    regression_result['X'] = X
    regression_result['Y'] = Y
    regression_result['test_idx_per_cv_set'] = np.array(test_idx_per_set)
    regression_result['Y_test_hat_per_cv_set'] = Y_test_hat_per_cv_set
    regression_result['explained_variance_per_cv_set'] = explained_variance_per_cv_set
    regression_result['weights_per_cv_set'] = np.array(weights_per_cv_set)

    if 'r2' in performance_metrics:
        regression_result['r2_per_cv_set'] = r2_per_cv_set

    regression_result['Y_test_per_cv_set'] = np.array(Y_test_per_cv_set)

    return regression_result


def get_aligned_explained_variance(regression_result, exp_data, alignment_time_window=[0, 3],
                                   performance_metrics=['explained_variance', 'r2'],
                                   exp_type='grating', exclude_saccade_on_vis_exp=False):
    """
    Parameters
    ----------
    regression_result : dict
        dictionary containing regression results, it should have the following keys
        'Y_test_per_cv_set' : list of numpy arrays
        'test_idx_per_cv_set' : list of numpy arrays
    exp_data : dict
        dictionary containing all used experiment data
    performance_metrics : list

    Returns
    -------
    regression_result : dict

    """

    # Get the fitted activity matrix for the entire time trace
    Y = regression_result['Y']
    Y_test_per_cv_set = regression_result['Y_test_per_cv_set']
    Y_test_hat_per_cv_set = np.array(regression_result['Y_test_hat_per_cv_set'])
    test_idx_per_cv_set = regression_result['test_idx_per_cv_set']

    vis_exp_times, vis_exp_saccade_onset_times, grating_onset_times, saccade_dirs, grating_orientation_per_trial = get_vis_and_saccade_times(
        exp_data, exp_type=exp_type, exclude_saccade_on_vis_exp=exclude_saccade_on_vis_exp)

    subset_vis_exp_vis_onset_times = grating_onset_times[
        (grating_onset_times + alignment_time_window[1] < vis_exp_times[-1]) &
        (grating_onset_times + alignment_time_window[0] > vis_exp_times[0])
        ]

    subset_vis_exp_saccade_onset_times = vis_exp_saccade_onset_times[
        (vis_exp_saccade_onset_times + alignment_time_window[1] < vis_exp_times[-1]) &
        (vis_exp_saccade_onset_times + alignment_time_window[0] > vis_exp_times[0])
        ]

    num_time_points = len(vis_exp_times)
    num_neurons = np.shape(Y_test_per_cv_set[0])[1]

    Y_hat = np.zeros((num_time_points, num_neurons)) + np.nan

    for n_cv, test_idx in enumerate(test_idx_per_cv_set):
        # deal with either unequal time points per cv set or same number
        if Y_test_hat_per_cv_set.ndim == 1:
            Y_hat[test_idx, :] = Y_test_hat_per_cv_set[n_cv]
        else:
            Y_hat[test_idx, :] = Y_test_hat_per_cv_set[n_cv, :, :]

    if np.sum(np.isnan(Y_hat)) != 0:
        print('Something wrong with getting Y_hat, found: %.f nans' % (np.sum(np.isnan(Y_hat))))
        Y_hat[np.isnan(Y_hat)] = np.nanmean(Y_hat)

    num_saccades = len(subset_vis_exp_saccade_onset_times)
    num_grating_presentations = len(subset_vis_exp_vis_onset_times)

    sec_per_time_samples = np.mean(np.diff(vis_exp_times))
    num_aligned_time_points = int((alignment_time_window[1] - alignment_time_window[0]) / sec_per_time_samples)

    if exp_type == 'gray':
        num_trials_to_get = num_saccades
    else:
        num_trials_to_get = np.min([num_saccades, num_grating_presentations])

    Y_hat_vis_aligned = np.zeros((num_trials_to_get, num_aligned_time_points, num_neurons)) + np.nan
    Y_vis_aligned = np.zeros((num_trials_to_get, num_aligned_time_points, num_neurons)) + np.nan
    Y_hat_saccade_aligned = np.zeros((num_trials_to_get, num_aligned_time_points, num_neurons)) + np.nan
    Y_saccade_aligned = np.zeros((num_trials_to_get, num_aligned_time_points, num_neurons)) + np.nan

    if exp_type == 'gray':
        subset_vis_on_times = []
        subset_saccade_on_times = subset_vis_exp_saccade_onset_times
    else:
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

    if np.all(np.isnan(Y_vis_aligned_flattened)):
        print('No vis aligned data found, assuming saccade only experiment and setting vis aligned variance to NaN')
        vis_aligned_var_explained = np.repeat(np.nan, num_neurons)
    else:
        vis_aligned_var_explained = sklmetrics.explained_variance_score(Y_vis_aligned_flattened,
                                                                        Y_hat_vis_aligned_flattened,
                                                                        multioutput='raw_values')

    saccade_aligned_var_explained = sklmetrics.explained_variance_score(Y_saccade_aligned_flattened, Y_hat_saccade_aligned_flattened, multioutput='raw_values')


    if 'r2' in performance_metrics:
        if np.all(np.isnan(Y_vis_aligned_flattened)):
            vis_aligned_r2 = np.repeat(np.nan, num_neurons)
        else:
            vis_aligned_r2 = sklmetrics.r2_score(Y_vis_aligned_flattened, Y_hat_vis_aligned_flattened, multioutput='raw_values')
        saccade_aligned_r2 = sklmetrics.r2_score(Y_saccade_aligned_flattened, Y_hat_saccade_aligned_flattened, multioutput='raw_values')


    regression_result['Y_hat_vis_aligned'] = Y_hat_vis_aligned
    regression_result['Y_vis_aligned'] = Y_vis_aligned
    regression_result['Y_hat_saccade_aligned'] = Y_hat_saccade_aligned
    regression_result['Y_saccade_aligned'] = Y_saccade_aligned
    regression_result['vis_aligned_var_explained'] = vis_aligned_var_explained
    regression_result['saccade_aligned_var_explained'] = saccade_aligned_var_explained

    if 'r2' in performance_metrics:
        regression_result['vis_aligned_r2'] = vis_aligned_r2
        regression_result['saccade_aligned_r2'] = saccade_aligned_r2


    # Get the full Y_vis_aligned and Y_saccade_aligned for later plotting
    Y_vis_aligned_full = np.zeros((len(subset_vis_exp_vis_onset_times), num_aligned_time_points, num_neurons)) + np.nan
    Y_saccade_aligned_full = np.zeros((len(subset_vis_exp_saccade_onset_times), num_aligned_time_points, num_neurons)) + np.nan

    # align to visual stimulus
    for n_trial, vis_on_time in enumerate(subset_vis_exp_vis_onset_times):

        time_idx = np.where(
            (vis_exp_times >= (vis_on_time + alignment_time_window[0])) &
            (vis_exp_times <= (vis_on_time + alignment_time_window[1]))
        )[0]

        time_idx = time_idx[0:num_aligned_time_points]

        Y_vis_aligned_full[n_trial, :, :] = Y[time_idx, :]

    for n_trial, saccade_on_time in enumerate(subset_vis_exp_saccade_onset_times):
        time_idx = np.where(
            (vis_exp_times >= (saccade_on_time + alignment_time_window[0])) &
            (vis_exp_times <= (saccade_on_time + alignment_time_window[1]))
        )[0]
        time_idx = time_idx[0:num_aligned_time_points]
        Y_saccade_aligned_full[n_trial, :, :] = Y[time_idx, :]

    regression_result['Y_vis_aligned_full'] = Y_vis_aligned_full
    regression_result['Y_saccade_aligned_full'] = Y_saccade_aligned_full

    return regression_result

def get_aligned_activity(exp_data, exp_type='grating', aligned_event='saccade', alignment_time_window=[-1, 3],
                         exclude_saccade_on_vis_exp=False):
    """
    Parameters
    ----------

    Returns
    -------

    """


    exp_times, exp_saccade_onset_times, grating_onset_times, saccade_dirs, grating_orientation_per_trial = get_vis_and_saccade_times(
        exp_data, exp_type=exp_type, exclude_saccade_on_vis_exp=exclude_saccade_on_vis_exp)

    neural_activity = exp_data['_tracesVis']
    num_neurons = np.shape(neural_activity)[1]
    time_bin_width = np.mean(np.diff(exp_times))
    trial_type = np.zeros((len(exp_saccade_onset_times), ))
    time_windows = np.arange(alignment_time_window[0], alignment_time_window[1], time_bin_width)
    num_time_bins = len(time_windows)

    min_time = exp_times[0]
    max_time = exp_times[-1]

    if aligned_event == 'saccade':
        subset_exp_saccade_onset_times = exp_saccade_onset_times[
            (exp_saccade_onset_times + alignment_time_window[0] > min_time) &
            (exp_saccade_onset_times + alignment_time_window[1] < max_time)
        ]

        aligned_activity = np.zeros((len(subset_exp_saccade_onset_times), num_time_bins, num_neurons)) + np.nan

        for trial_i, saccade_time in enumerate(subset_exp_saccade_onset_times):

            subset_idx = np.where(
                (exp_times >= (saccade_time + alignment_time_window[0])) &
                (exp_times <= (saccade_time + alignment_time_window[1]))
            )[0]

            aligned_activity[trial_i, :, :] = neural_activity[subset_idx, :]

        return aligned_activity, trial_type, time_windows

    elif aligned_event == 'vis':

        subset_grating_onset_trial_idx = np.where(
             (grating_onset_times + alignment_time_window[0] > min_time) &
             (grating_onset_times + alignment_time_window[1] < max_time)
        )[0]

        subset_grating_onset_times = grating_onset_times[subset_grating_onset_trial_idx]

        aligned_activity = np.zeros((len(subset_grating_onset_times), num_time_bins, num_neurons)) + np.nan

        saccade_time = []
        saccade_dir_during_vis = []

        for trial_i, vis_on_time in enumerate(subset_grating_onset_times):

            # Get saccade information
            saccade_idx = np.where(
                (exp_saccade_onset_times >= vis_on_time + alignment_time_window[0]) &
                (exp_saccade_onset_times <= vis_on_time + alignment_time_window[1])
            )[0]

            if len(saccade_idx) > 0:
                saccade_time.append(exp_saccade_onset_times[saccade_idx] - vis_on_time)
                saccade_dir_during_vis.append(saccade_dirs[saccade_idx])
            else:
                saccade_time.append(np.array([]))
                saccade_dir_during_vis.append(np.array([]))

            # Get aligned activity
            subset_idx = np.where(
                (exp_times >= (vis_on_time + alignment_time_window[0])) &
                (exp_times <= (vis_on_time + alignment_time_window[1]))
            )[0]

            # TODO: this is a bit strange...
            if len(subset_idx) < num_time_bins:
                # print('Not enough time bins, taking one back')
                subset_idx = np.concatenate([np.array([subset_idx[0]-1]), subset_idx])

            aligned_activity[trial_i, :, :] = neural_activity[subset_idx, :]

        grating_id_per_trial = np.array(exp_data['_gratingIds'].flatten()) - 1 # to 0 indexing
        grating_id_per_trial = grating_id_per_trial.astype(int)
        grating_id_to_dir = np.array(exp_data['_gratingIdDirections'].flatten())

        grating_id_per_trial_subset = grating_id_per_trial[subset_grating_onset_trial_idx]

        vis_ori = np.array([grating_id_to_dir[x] for x in grating_id_per_trial_subset])

        return aligned_activity, vis_ori, saccade_dir_during_vis, saccade_time, time_windows


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


def get_num_saccade_per_grating(vis_exp_saccade_onset_times, grating_onset_times,
                                grating_orientation_per_trial, window_rel_grating_onset=[0, 3]):

    if type(grating_orientation_per_trial) is list:
        grating_orientation_per_trial = np.array(grating_orientation_per_trial)

    unique_ori = np.unique(grating_orientation_per_trial[~np.isnan(grating_orientation_per_trial)])
    num_saccade_per_grating = np.zeros((len(unique_ori), )) + np.nan

    for n_ori, ori in enumerate(unique_ori):

        subset_trial_idx = np.where(grating_orientation_per_trial == ori)[0]
        subset_grating_onset_times = grating_onset_times[subset_trial_idx]

        saccade_counter = 0

        for grating_onset_t in subset_grating_onset_times:

            grating_window = [grating_onset_t + window_rel_grating_onset[0],
                              grating_onset_t + window_rel_grating_onset[1]]

            saccade_in_window = np.where(
                (vis_exp_saccade_onset_times >= grating_window[0]) &
                (vis_exp_saccade_onset_times <= grating_window[1])
            )[0]

            saccade_counter += len(saccade_in_window)

        num_saccade_per_grating[n_ori] = saccade_counter

    saccade_outside_grating = len(vis_exp_saccade_onset_times) - np.sum(num_saccade_per_grating)

    return unique_ori, saccade_outside_grating, num_saccade_per_grating


def get_saccade_trials_without_grating(exp_data):

    vis_exp_times = exp_data['_windowVis'].flatten()
    exp_times = exp_data['_windowVis'].flatten()
    grating_intervals = exp_data['_gratingIntervals']
    grating_onset_times = grating_intervals[:, 0]
    vis_exp_saccade_intervals = exp_data['_saccadeIntervalsVis'].astype(int)
    vis_exp_saccade_intervals_sec = vis_exp_times[vis_exp_saccade_intervals]  # usually last less than one second
    exp_saccade_onset_times = exp_times[vis_exp_saccade_intervals[:, 0]]
    saccade_dirs = exp_data['_saccadeVisDir'].flatten()
    saccade_dirs[saccade_dirs == 0] = -1

    subset_trial_indices = []
    for n_trial, saccade_on_time in enumerate(exp_saccade_onset_times):
        in_any_interval = np.sum([(saccade_on_time > x) & (saccade_on_time < y) for (x, y) in zip(grating_intervals[:, 0], grating_intervals[:, 1])])
        if in_any_interval == 0:
            subset_trial_indices.append(n_trial)

    trial_idx = np.array(subset_trial_indices)

    return trial_idx

def plot_pupil_data(exp_data, exp_time_var_name='_windowVis',
                    pupil_size_var_name='pupilSizeVis', highlight_nans=False, run_impute_time_series=False,
                    pupil_preprocessing_steps=[],
                    fig=None, ax=None):
    """
    Parameters
    ----------
    exp_data : dict
    fig : matplotlib figure object
    ax : matplotlib axis object
    exp_time_var_name : str

    highlight_nans : bool
        whether to highlight time points where there are NaNs in the pupil size
    Returns
    -------
    fig : matplotlib figure object
    ax : matplotlib axis object
    """

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 3)


    # Grating experiment
    pupil_size = exp_data['_pupilSizeVis'].flatten()
    exp_time = exp_data[exp_time_var_name].flatten()

    # Gray screen experiment
    gray_pupil_size = exp_data['_pupilSizeGray'].flatten()
    gray_exp_time = exp_data['_windowGray'].flatten()


    # Do imputation
    pupil_size = impute_time_series(pupil_size, method='interpolate')
    gray_pupil_size = impute_time_series(gray_pupil_size, method='interpolate')

    if 'zscore' in pupil_preprocessing_steps:
        pupil_size = sstats.zscore(pupil_size)
        gray_pupil_size = sstats.zscore(gray_pupil_size)

    ax.plot(exp_time, pupil_size, lw=0.5, color='black')
    ax.plot(exp_time[-1] + gray_exp_time, gray_pupil_size, lw=0.5, color='gray')

    if highlight_nans:
        nan_times = exp_time[np.isnan(pupil_size)]
        ax.plot(nan_times, np.repeat(np.max(pupil_size), len(nan_times)) + 1, lw=1.0, color='gray')

    if run_impute_time_series:
        imputed_pupil_size = impute_time_series(pupil_size, method='interpolate')
        ax.plot(exp_time, imputed_pupil_size, lw=0.5, color='red')

    ax.set_xlabel('Time (seconds)', size=11)
    ax.set_ylabel('Pupil size', size=11)

    return fig, ax

def impute_time_series(time_series, method='interpolate'):


    if method == 'interpolate':

        imputed_time_series = pd.Series(time_series).interpolate().values

    # Check imputation worked
    assert np.sum(np.isnan(imputed_time_series)) == 0

    return imputed_time_series




def plot_grating_and_gray_exp_neuron_regression(grating_Y, gray_Y, grating_Y_hat_model,
                                                gray_Y_hat_model, grating_Y_hat_model_2=None, neuron_idx=0, fig=None, axs=None):
    """
    Parameters
    ----------
    grating_Y : numpy ndarray
        neural activity during the grating experiment
        array with shape (numTimePoints, numNeurons)
    gray_Y : numpy ndarray
        neural activity during the gray screen experiment
        array with shape (numTimePoints, numNeurons)
    grating_Y_hat_model : numpy ndarray
    neuron_idx : int
        which neuron to plot
    fig : matplotlib figure object

    Returns
    -------
    """

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(2, 1)

    axs[0].plot(grating_Y[:, neuron_idx], color='black', lw=1)
    axs[0].plot(grating_Y_hat_model[:, neuron_idx], color='blue', lw=1)

    if grating_Y_hat_model_2 is not None:
        axs[0].plot(grating_Y_hat_model_2[:, neuron_idx], color='green', lw=1)

    axs[1].plot(gray_Y[:, neuron_idx], color='black', lw=1)
    axs[1].plot(gray_Y_hat_model[:, neuron_idx], color='green', lw=1)

    axs[0].set_title('Grating exp', size=11)
    axs[1].set_title('Gray screen exp', size=11)

    axs[1].set_xlabel('Time (frames)', size=11)

    return fig, axs


def main():

    available_processes = ['load_data', 'plot_data', 'fit_regression_model', 'plot_regression_model_explained_var',
                           'plot_regression_model_example_neurons', 'compare_iterative_vs_normal_fit',
                           'plot_pupil_data', 'plot_original_vs_aligned_explained_var',
                           'plot_grating_vs_gray_screen_single_cell_fit_performance',
                           'plot_grating_vs_gray_screen_example_neurons', 'compare_saccade_kernels',
                           'plot_sig_vs_nosig_neuron_explained_var', 'plot_saccade_neuron_psth_and_regression',
                           'plot_sig_model_comparison_neurons', 'plot_num_saccade_per_ori',
                           'plot_vis_and_saccade_neuron_individual_trials']

    processes_to_run = ['plot_vis_and_saccade_neuron_individual_trials']
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
            file_types_to_load=['_windowVis', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis',
                                '_windowGray', '_tracesGray', '_pupilSizeGray', '_onsetOffset', '_trial_Dir',  # gray screen experiments
                                ],
            X_sets_to_compare={'bias_only': ['bias'],
                               # 'vis_on_only': ['bias', 'vis_on'],
                               # 'vis_ori': ['bias', 'vis_ori'],
                               # 'vis_ori_iterative': ['bias', 'vis_ori_iterative'],
                               # 'vis': ['bias', 'vis_on', 'vis_dir'],
                               # 'vis_ori_and_pupil': ['bias', 'vis_ori', 'pupil_size'],
                               # 'saccade_on_only': ['bias', 'saccade_on'],
                               'saccade': ['bias', 'saccade_on', 'saccade_dir'],
                               # 'pupil_size_only': ['bias', 'pupil_size'],
                               # 'saccade_and_pupil': ['bias', 'saccade_on', 'saccade_dir', 'pupil_size'],
                               'vis_and_saccade': ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir'],
                               # 'vis_and_saccade_and_pupil': ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir', 'pupil_size'],
                               # 'vis_on_and_saccade_on': ['bias', 'vis_on', 'saccade_on'],
                               # 'vis_on_and_saccade_on_and_interaction': ['bias', 'vis_on', 'saccade_on',
                               #                                           'vis_on_saccade_on']
                               },
            feature_time_windows={'vis_on': [-1.0, 3.0], 'vis_dir': [-1.0, 3.0], 'vis_ori': [-1.0, 3.0],
                                  'saccade_on': [-1.0, 3.0], 'saccade_dir': [-1.0, 3.0],
                                  'vis_on_saccade_on': [-1.0, 3.0], 'vis_ori_iterative': [0, 3.0],
                                  'pupil_size': None},
            performance_metrics=['explained_variance', 'r2'],
            exp_type='grating',  # 'grating', 'gray', 'both'
            exclude_saccade_on_vis_exp=False,
            pupil_preprocessing_steps=['zscore'],
            train_test_split_method='n_fold_cv',   # 'half' or 'n_fold_cv'
            neural_preprocessing_steps=['z_score_separately', 'zscore_w_baseline'],  # 'zscore' is optional, 'z_score_separately', 'zscore_w_baseline'
        ),
        'plot_regression_model_explained_var': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/both-include-vis-saccade-separate-zscore',
            # fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            X_sets_to_compare=[
                # ['vis_on_only', 'vis_ori'],
                # ['saccade_on_only', 'saccade'],
                # ['vis_on_only', 'saccade_on_only'],
                # ['vis_ori', 'saccade_on_only'],
                # ['vis_ori', 'saccade'],
                # ['vis_on_only', 'vis'],
                # ['saccade_on_only', 'saccade'],
                # ['vis_on_only', 'saccade_on_only'],
                # ['vis', 'saccade'],
                # ['vis_on_and_saccade_on', 'vis_on_and_saccade_on_and_interaction']
                ['saccade_and_pupil', 'saccade'],
                ['vis_ori_and_pupil', 'vis_ori'],
                ['vis_ori', 'saccade'],
                ['vis_ori_and_pupil', 'saccade_and_pupil']
            ],  # options for metrics are : 'vis_aligned_var_explained', 'saccade_aligned_var_explained', '
            # metrics_to_compare=np.array([
            #       ['vis_aligned_r2', 'vis_aligned_r2'],
            #       ['saccade_aligned_r2', 'saccade_aligned_r2'],
            #       ['vis_aligned_r2', 'saccade_aligned_r2'],
            #       ['vis_aligned_r2', 'saccade_aligned_r2'],
            #       ['vis_aligned_r2', 'saccade_aligned_r2'],
            # ]),
            # metrics_to_compare=np.array([
            #       ['explained_var_per_X_set', 'explained_var_per_X_set'],
            #       ['explained_var_per_X_set', 'explained_var_per_X_set'],
            #       ['explained_var_per_X_set', 'explained_var_per_X_set'],
            #        ['explained_var_per_X_set', 'explained_var_per_X_set'],
            #       ['explained_var_per_X_set', 'explained_var_per_X_set'],
            # ]),
            metrics_to_compare=np.array([
                  # ['vis_aligned_var_explained', 'vis_aligned_var_explained'],
                  ['saccade_aligned_var_explained', 'saccade_aligned_var_explained'],
                  ['vis_aligned_var_explained', 'vis_aligned_var_explained'],
                  ['vis_aligned_var_explained', 'saccade_aligned_var_explained'],
                  ['vis_aligned_var_explained', 'saccade_aligned_var_explained'],
                  # ['explained_var_per_X_set', 'explained_var_per_X_set'],
                  # ['explained_var_per_X_set', 'explained_var_per_X_set'],
                  # ['explained_var_per_X_set', 'explained_var_per_X_set'],
                  # ['explained_var_per_X_set', 'explained_var_per_X_set']
            ]),
            custom_fig_addinfo='w_without_pupil_aligned_explained_var',  # 'original', or 'aligned', 'aligned_r2', 'original_r2', 'aligned_separate_zscore_using_iti'
            exp_type='both',  # 'grating', 'gray', 'both'
            clip_at_zero=True,
        ),
        'plot_regression_model_example_neurons': dict(
            neuron_type_to_plot='saccade_on',  # 'vis_on', 'saccade_on', 'saccade_dir', 'vis_ori'
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            # fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            # fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/both-include-vis-saccade-separate-zscore',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/include-pupil',
            file_types_to_load=['_windowVis', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis',
                                '_windowGray', '_tracesGray', '_pupilSizeGray', '_onsetOffset', '_trial_Dir',  # gray screen experiments
                                ],
            models_to_plot=['saccade', 'saccade_and_pupil'],  # 'vis_on_only', 'saccade_on_only', 'vis_ori', 'saccade',
            model_colors = {'vis_on_only': 'orange',
                            'saccade': 'green',
                           'saccade_on_only': 'green',
                           'saccade_and_pupil': 'pink',
                           'vis_ori': 'blue',
                           'vis_ori_and_pupil': 'lightblue',
                           },
            exp_type='both',  # 'grating', 'gray', 'both'
            exclude_saccade_on_vis_exp=False,
            group_by_orientation=True,
            custom_fig_addinfo='zscore_using_iti',
            num_example_neurons_to_plot=10,
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
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/num_saccade_per_grating',
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis'],
            zscore_activity=True,
        ),
        'plot_pupil_data': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/pupil',
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis', '_pupilSizeGray'],
            highlight_nans=True,
            pupil_preprocessing_steps=['zscore'],
        ),
        'plot_original_vs_aligned_explained_var': dict(
            neuron_type_to_plot='saccade_on',  # 'vis_on', 'saccade_on', 'saccade_dir', 'vis_ori'
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            file_types_to_load=['_windowVis', '_windowGray', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis'],
            models_to_plot=['vis_ori', 'saccade_on_only'],  # 'vis_on_only', 'saccade_on_only', 'vis_ori'
            model_colors={'vis_on_only': 'orange',
                          'saccade_on_only': 'green',
                          'vis_ori': 'blue'},
            group_by_orientation=True,
            X_set_to_plot='saccade',  # 'vis_ori', or 'saccade'
            plot_single_neuron_examples=False,
            plot_overall_summary=True,
        ),
        'plot_grating_vs_gray_screen_single_cell_fit_performance': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            metrics_to_compare=np.array([
                ['vis_aligned_var_explained', 'vis_aligned_var_explained'],
                ['saccade_aligned_var_explained', 'saccade_aligned_var_explained'],
                ['vis_aligned_var_explained', 'saccade_aligned_var_explained'],
                ['vis_aligned_var_explained', 'saccade_aligned_var_explained'],
                ['vis_aligned_var_explained', 'saccade_aligned_var_explained'],
            ]),
            custom_fig_addinfo='aligned',
        ),
        'plot_grating_vs_gray_screen_example_neurons': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            num_neurons_to_plot=10,
        ),
        'compare_saccade_kernels': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/saccade_kernel_comparison',
            plot_variance_explained_comparison=False,
            gray_exp_model='saccade',
            grating_exp_model='vis_and_saccade',
            num_neurons_to_plot=10,
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            file_types_to_load=['_windowVis', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis',
                                '_windowGray', '_tracesGray', '_pupilSizeGray', '_onsetOffset', '_trial_Dir',
                                # gray screen experiments
                                ],
        ),
        'plot_sig_vs_nosig_neuron_explained_var': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/sig_nosig_explained_var',
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            file_types_to_load=['_windowVis', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis',
                                '_windowGray', '_tracesGray', '_pupilSizeGray', '_onsetOffset', '_trial_Dir',
                                '_sigNeurons', '_sigDirectionNeuron',
                                # gray screen experiments
                                ],
        ),
        'plot_saccade_neuron_psth_and_regression': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/sig_nosig_explained_var',
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            file_types_to_load=['_windowVis', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis',
                                '_windowGray', '_tracesGray', '_pupilSizeGray', '_onsetOffset', '_trial_Dir',
                                '_sigNeurons', '_sigDirectionNeuron',
                                # gray screen experiments
                                ],
            exclude_saccade_on_vis_exp=False,
        ),
        'plot_sig_model_comparison_neurons': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/sig_model_comparison_neurons',
            X_sets_to_compare=[
                ['vis_and_saccade', 'saccade'],
                ['vis_and_saccade', 'vis_ori'],
            ],
            custom_fig_addinfo=None,
            num_neurons_to_plot=5,  # this is per recording I think...
            exp_type='both',
            model_a_explained_var_threshold=0.1,
            model_b_explained_var_threshold=0.1,
            min_model_ev_diff=0.05,
        ),
        'plot_vis_and_saccade_neuron_individual_trials': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/vis_and_saccade_neurons',
            plot_variance_explained_comparison=False,
            gray_exp_model='saccade',
            grating_exp_model='vis_and_saccade',
            num_neurons_to_plot=10,
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            file_types_to_load=['_windowVis', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis',
                                '_windowGray', '_tracesGray', '_pupilSizeGray', '_onsetOffset', '_trial_Dir',
                                # gray screen experiments
                                ],
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
            exclude_saccade_on_vis_exp = process_params[process]['exclude_saccade_on_vis_exp']

            X_sets_to_compare = process_params[process]['X_sets_to_compare']
            performance_metrics = process_params[process]['performance_metrics']
            # feature_set = ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir']

            for exp_id, exp_data in data.items():

                num_X_set = len(X_sets_to_compare.keys())
                num_neurons = np.shape(exp_data['_tracesVis'])[1]
                explained_var_per_X_set = np.zeros((num_neurons, num_X_set))
                vis_aligned_explained_var_per_X_set = np.zeros((num_neurons, num_X_set))
                saccade_aligned_explained_var_per_X_set = np.zeros((num_neurons, num_X_set))

                vis_aligned_Y_hat_per_X_set = []
                saccade_aligned_Y_hat_per_X_set = []
                vis_aligned_Y_per_X_set = []
                saccade_aligned_Y_per_X_set = []

                exp_regression_result = dict()
                exp_regression_result['X_sets_names'] = list(X_sets_to_compare.keys())
                Y_test_hat_per_X_set = []

                if 'r2' in performance_metrics:
                    r2_per_X_set = np.zeros((num_neurons, num_X_set))
                    vis_aligned_r2_per_X_set = np.zeros((num_neurons, num_X_set))
                    saccade_aligned_r2_per_X_set = np.zeros((num_neurons, num_X_set))

                feature_indices_per_X_set = {}
                model_weights_per_X_set = {}

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

                    X, Y, grating_orientation_per_trial, saccade_dirs, feature_indices = make_X_Y_for_regression(exp_data, feature_set=feature_set,
                                                   neural_preprocessing_steps=neural_preprocessing_steps,
                                                   train_indices=train_indices, test_indices=test_indices,
                                                   exp_type=process_params[process]['exp_type'],
                                                   exclude_saccade_on_vis_exp=exclude_saccade_on_vis_exp,
                                                   feature_time_windows=process_params[process]['feature_time_windows'],
                                                   return_trial_type=True,
                                                   pupil_preprocessing_steps=process_params[process]['pupil_preprocessing_steps'])

                    feature_indices_per_X_set[X_set_name] = feature_indices

                    regression_result = fit_regression_model(X, Y, performance_metrics=performance_metrics,
                                                             train_test_split_method=process_params[process]['train_test_split_method'])


                    regression_result = get_aligned_explained_variance(regression_result, exp_data, performance_metrics=performance_metrics,
                                                                       exp_type=process_params[process]['exp_type'],
                                                                       exclude_saccade_on_vis_exp=exclude_saccade_on_vis_exp,
                                                                       alignment_time_window=[0, 3])

                    explained_var_per_X_set[:, n_X_set] = np.mean(regression_result['explained_variance_per_cv_set'], axis=1)
                    vis_aligned_explained_var_per_X_set[:, n_X_set] = regression_result['vis_aligned_var_explained']
                    saccade_aligned_explained_var_per_X_set[:, n_X_set] = regression_result['saccade_aligned_var_explained']

                    if 'r2' in performance_metrics:
                        r2_per_X_set[:, n_X_set] = np.mean(regression_result['r2_per_cv_set'], axis=1)
                        vis_aligned_r2_per_X_set[:, n_X_set] = regression_result['vis_aligned_r2']
                        saccade_aligned_r2_per_X_set[:, n_X_set] = regression_result['saccade_aligned_r2']

                    Y_test_hat_per_X_set.append(regression_result['Y_test_hat_per_cv_set'])

                    vis_aligned_Y_hat_per_X_set.append(regression_result['Y_hat_vis_aligned'])
                    saccade_aligned_Y_hat_per_X_set.append(regression_result['Y_hat_saccade_aligned'])
                    vis_aligned_Y_per_X_set.append(regression_result['Y_vis_aligned'])
                    saccade_aligned_Y_per_X_set.append(regression_result['Y_saccade_aligned'])
                    model_weights_per_X_set[X_set_name] = regression_result['weights_per_cv_set']

                exp_regression_result['explained_var_per_X_set'] = explained_var_per_X_set
                exp_regression_result['Y_test'] = regression_result['Y_test_per_cv_set']
                exp_regression_result['vis_aligned_var_explained'] = vis_aligned_explained_var_per_X_set
                exp_regression_result['saccade_aligned_var_explained'] = saccade_aligned_explained_var_per_X_set
                exp_regression_result['feature_indices_per_X_set'] = feature_indices_per_X_set
                exp_regression_result['model_weights_per_X_set'] = model_weights_per_X_set

                if 'r2' in performance_metrics:
                    exp_regression_result['r2_per_X_set'] = r2_per_X_set
                    exp_regression_result['vis_aligned_r2'] = vis_aligned_r2_per_X_set
                    exp_regression_result['saccade_aligned_r2'] = saccade_aligned_r2_per_X_set

                exp_regression_result['Y_hat_vis_aligned'] = vis_aligned_Y_hat_per_X_set
                exp_regression_result['Y_vis_aligned'] = vis_aligned_Y_per_X_set
                exp_regression_result['Y_hat_saccade_aligned'] = saccade_aligned_Y_hat_per_X_set
                exp_regression_result['Y_saccade_aligned'] = saccade_aligned_Y_per_X_set
                exp_regression_result['Y_test_hat'] = np.array(Y_test_hat_per_X_set)
                exp_regression_result['test_idx_per_cv_set'] = regression_result['test_idx_per_cv_set']
                exp_regression_result['regression_kernel_names'] = np.array([*process_params[process]['feature_time_windows'].keys()])
                exp_regression_result['regression_kernel_windows'] = np.array([*process_params[process]['feature_time_windows'].values()])

                exp_regression_result['grating_orientation_per_trial'] = grating_orientation_per_trial
                exp_regression_result['saccade_dirs'] = saccade_dirs

                exp_regression_result['Y_vis_aligned_full'] = regression_result['Y_vis_aligned_full']
                exp_regression_result['Y_saccade_aligned_full'] = regression_result['Y_saccade_aligned_full']

                regression_result_savename = '%s_%s_regression_results.npz' % (exp_id, process_params[process]['exp_type'])
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
                exp_regression_result['Y_test_hat'] = np.array(Y_test_hat_per_X_set)
                exp_regression_result['test_idx_per_cv_set'] = regression_result['test_idx_per_cv_set']

                regression_result_savename = '%s_regression_results.npz' % exp_id
                regression_result_savepath = os.path.join(regression_results_folder, regression_result_savename)
                # np.savez(regression_result_savepath, **exp_regression_result)


        if process == 'plot_regression_model_explained_var':

            exp_type = process_params[process]['exp_type']
            regression_result_files = glob.glob(os.path.join(process_params[process]['regression_results_folder'],
                                                             '*%s*npz' % exp_type))
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

                        if process_params[process]['clip_at_zero']:
                            both_model_min = -0.1
                        else:
                            both_model_min = np.min(both_model_explained_var)
                        both_model_max = np.max(both_model_explained_var)
                        unity_vals = np.linspace(both_model_min, both_model_max, 100)
                        axs[n_comparison].axvline(0, linestyle='--', color='gray', alpha=0.5, lw=0.75, zorder=-1)
                        axs[n_comparison].axhline(0, linestyle='--', color='gray', alpha=0.5, lw=0.75, zorder=-2)
                        axs[n_comparison].plot(unity_vals, unity_vals, linestyle='--', color='gray', alpha=0.5)
                        axs[n_comparison].set_xlabel(model_a, size=text_size)
                        axs[n_comparison].set_ylabel(model_b, size=text_size)

                        axs[n_comparison].set_xlim([both_model_min, both_model_max])
                        axs[n_comparison].set_ylim([both_model_min, both_model_max])

                    exp_id_parts = os.path.basename(fpath).split('.')[0].split('_')
                    subject = exp_id_parts[0]
                    exp_date = exp_id_parts[1]

                    if custom_fig_addinfo is not None:
                        fig_name = '%s_%s_%s_%s_explained_variance_per_X_set_comparison' % (exp_type, subject, exp_date, custom_fig_addinfo)
                    else:
                        fig_name = '%s_%s_%s_explained_variance_per_X_set_comparison' % (exp_type, subject, exp_date)
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbbox_inches='tight')

                    plt.close(fig)
        if process == 'plot_grating_vs_gray_screen_single_cell_fit_performance':

            print('Plotting grating vs gray screen single cell performance')
            regression_results_folder = process_params[process]['regression_results_folder']
            fig_folder = process_params[process]['fig_folder']
            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % 'grating'))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in regression_result_files]

            for exp_id in exp_ids:
                grating_regression_result = np.load(glob.glob(os.path.join(regression_results_folder, '*%s*grating*.npz' % (exp_id)))[0])
                gray_regression_result = np.load(glob.glob(os.path.join(regression_results_folder, '*%s*gray*.npz' % (exp_id)))[0])


                with plt.style.context(splstyle.get_style('nature-reviews')):

                    fig, axs = plt.subplots(1, 2)
                    fig.set_size_inches(8, 4)


                    # Full variance explained
                    grating_exp_vis_ori_model_idx = np.where(
                        grating_regression_result['X_sets_names'] == 'vis_ori'
                    )[0][0]

                    gray_exp_saccade_dir_model_idx = np.where(
                        gray_regression_result['X_sets_names'] == 'saccade'
                    )[0][0]

                    grating_exp_vis_ori_explained_var = grating_regression_result['explained_var_per_X_set'][:, grating_exp_vis_ori_model_idx]
                    gray_exp_saccade_dir_explained_var = gray_regression_result['explained_var_per_X_set'][:, gray_exp_saccade_dir_model_idx]

                    axs[0].scatter(
                        grating_exp_vis_ori_explained_var,
                        gray_exp_saccade_dir_explained_var, lw=0, color='black', s=10
                    )

                    axs[0].set_xlabel('Grating exp vis ori model', size=11)
                    axs[0].set_ylabel('Gray screen saccade dir model', size=11)
                    axs[0].set_title('Full explained variance', size=11)

                    # Aligned variance explained
                    grating_exp_vis_ori_aligned_explained_var = grating_regression_result['vis_aligned_var_explained'][:,
                                                        grating_exp_vis_ori_model_idx]
                    gray_exp_saccade_dir_aligned_explained_var = gray_regression_result['saccade_aligned_var_explained'][:,
                                                         gray_exp_saccade_dir_model_idx]

                    axs[1].scatter(
                        grating_exp_vis_ori_aligned_explained_var,
                        gray_exp_saccade_dir_aligned_explained_var, lw=0, color='black', s=10
                    )

                    axs[1].set_xlabel('Grating exp vis ori model', size=11)
                    axs[1].set_ylabel('Gray screen saccade dir model', size=11)
                    axs[1].set_title('Aligned explained variance', size=11)


                    # Significance line
                    axs[0].axvline(0, linestyle='--', color='gray', lw=0.5, alpha=0.3)
                    axs[0].axhline(0, linestyle='--', color='gray', lw=0.5, alpha=0.3)
                    axs[1].axvline(0, linestyle='--', color='gray', lw=0.5, alpha=0.3)
                    axs[1].axhline(0, linestyle='--', color='gray', lw=0.5, alpha=0.3)

                    fig_name = '%s_grating_vs_gray_exp_explained_var_per_neuron' % exp_id
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                    plt.close(fig)


        if process == 'plot_regression_model_example_neurons':

            # TODO: currently this assumes the Y_test cv split consists of first half and second half of recording
            exp_type = process_params[process]['exp_type']
            regression_result_files = glob.glob(os.path.join(process_params[process]['regression_results_folder'],
                                                             '*%s*npz' % exp_type))
            exclude_saccade_on_vis_exp = process_params[process]['exclude_saccade_on_vis_exp']

            fig_folder = process_params[process]['fig_folder']

            models_to_plot = process_params[process]['models_to_plot']
            model_colors = process_params[process]['model_colors']
            group_by_orientation = process_params[process]['group_by_orientation']

            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            neuron_type_to_plot = process_params[process]['neuron_type_to_plot']
            num_example_neurons_to_plot = process_params[process]['num_example_neurons_to_plot']

            for fpath in regression_result_files:

                regression_result = np.load(fpath, allow_pickle=True)

                X_sets_names = regression_result['X_sets_names']
                explained_var_per_X_set = regression_result['explained_var_per_X_set']
                Y_test = regression_result['Y_test']  # num cv x num time points x num neurons
                Y_test_hat = regression_result['Y_test_hat'] # num model x num cv x num time points x num neurons

                if len(Y_test) == 2:
                    print('Assuming half test half train')
                    Y = np.concatenate([Y_test[1], Y_test[0]])  # reproduce the entire trace (assume 2 fold cv)
                    try:
                        Y_hat = np.concatenate([Y_test_hat[:, 1, :, :], Y_test_hat[:, 0, :, :]], axis=1)
                    except:
                        pdb.set_trace()
                else:
                    Y = np.concatenate(Y_test)
                    num_x_set = np.shape(Y_test_hat)[0]
                    Y_hat = np.array([np.concatenate(Y_test_hat[x_set_idx, :]) for x_set_idx in np.arange(num_x_set)])
                    all_test_idx = np.concatenate(regression_result['test_idx_per_cv_set'])

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

                num_neurons_to_plot = np.min([num_example_neurons_to_plot, len(neuron_df_sorted)])

                num_model = len(models_to_plot)
                linewidth = 1

                for n_neuron in np.arange(num_neurons_to_plot):

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

                        vis_exp_times = exp_data['_windowVis'].flatten()

                        exp_times, exp_saccade_onset_times, grating_onset_times, saccade_dirs, grating_orientation_per_trial = \
                            get_vis_and_saccade_times(exp_data, exp_type=exp_type, exclude_saccade_on_vis_exp=exclude_saccade_on_vis_exp)

                        print('Exp times length: %.f' % len(exp_times))
                        print('All test idx length: %.f' % len(all_test_idx))
                        exp_times = exp_times[all_test_idx]

                        # grating_onset_times = grating_intervals[:, 0]
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

                        for n_trial, onset_time in enumerate(subset_onset_time):

                            start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                            end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
                            samples_to_get = np.arange(start_sample, end_sample)  # +1 here temp fix 2022-08-30
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
                        # vis_exp_saccade_onset_frames = vis_exp_saccade_intervals[:, 0].astype(int)
                        # vis_exp_saccade_onset_times = vis_exp_times[vis_exp_saccade_onset_frames]
                        subset_exp_saccade_onset_times = exp_saccade_onset_times[
                            (exp_saccade_onset_times + feature_time_window[1] < exp_times[-1]) &
                            (exp_saccade_onset_times + feature_time_window[0] > exp_times[0])
                        ]
                        n_saccade_trials = len(subset_exp_saccade_onset_times)

                        Y_vis_exp_saccade_aligned = np.zeros((n_saccade_trials, num_sample_in_window))
                        Y_hat_vis_exp_saccade_aligned = np.zeros((num_model, n_saccade_trials, num_sample_in_window))


                        for n_saccade_time, onset_time in enumerate(subset_exp_saccade_onset_times):

                            start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                            end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
                            samples_to_get = np.arange(start_sample, end_sample)
                            samples_to_get = samples_to_get[0:num_sample_in_window]

                            try:
                                Y_vis_exp_saccade_aligned[n_saccade_time, :] = Y[samples_to_get, neuron_idx]
                            except:
                                pdb.set_trace()

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

                        fig_name = '%s_%s_%s_example_%s_neuron_%.f' % (exp_type, subject, exp_date, neuron_type_to_plot, neuron_idx)

                        if process_params[process]['custom_fig_addinfo'] is not None:
                            fig_name += process_params[process]['custom_fig_addinfo']

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

                            fig_name = '%s_%s_%s_example_%s_neuron_%.f_ori_tuning' % (
                            exp_type, subject, exp_date, neuron_type_to_plot, neuron_idx)
                            fig.tight_layout()
                            fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                            plt.close(fig)
        if process == 'plot_grating_vs_gray_screen_example_neurons':

            regression_results_folder = process_params[process]['regression_results_folder']
            fig_folder = process_params[process]['fig_folder']
            num_neurons_to_plot = process_params[process]['num_neurons_to_plot']
            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % 'grating'))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in regression_result_files]

            for exp_id in exp_ids:
                grating_regression_result = np.load(glob.glob(os.path.join(regression_results_folder, '*%s*grating*.npz' % (exp_id)))[0], allow_pickle=True)
                gray_regression_result = np.load(glob.glob(os.path.join(regression_results_folder, '*%s*gray*.npz' % (exp_id)))[0], allow_pickle=True)

                grating_Y_test = grating_regression_result['Y_test']
                if len(grating_Y_test) == 2:
                    print('This part is not ready')
                    print('Assuming half test half train')
                    Y = np.concatenate([Y_test[1], Y_test[0]])  # reproduce the entire trace (assume 2 fold cv)
                    try:
                        Y_hat = np.concatenate([Y_test_hat[:, 1, :, :], Y_test_hat[:, 0, :, :]], axis=1)
                    except:
                        pdb.set_trace()
                else:
                    grating_Y = np.concatenate(grating_regression_result['Y_test'])
                    gray_Y = np.concatenate(gray_regression_result['Y_test'])

                    grating_Y_test_hat = grating_regression_result['Y_test_hat']
                    gray_Y_test_hat = gray_regression_result['Y_test_hat']
                    num_x_set = np.shape(grating_Y_test_hat)[0]

                    grating_Y_hat = np.array([np.concatenate(grating_Y_test_hat[x_set_idx, :]) for x_set_idx in np.arange(num_x_set)])
                    gray_Y_hat = np.array([np.concatenate(gray_Y_test_hat[x_set_idx, :]) for x_set_idx in np.arange(num_x_set)])

                    # all_test_idx = np.concatenate(regression_result['test_idx_per_cv_set'])

                grating_exp_vis_ori_model_idx = np.where(
                    grating_regression_result['X_sets_names'] == 'vis_ori'
                )[0][0]

                grating_exp_saccade_dir_model_idx = np.where(
                    grating_regression_result['X_sets_names'] == 'saccade'
                )[0][0]

                gray_exp_saccade_dir_model_idx = np.where(
                    gray_regression_result['X_sets_names'] == 'saccade'
                )[0][0]

                grating_exp_vis_ori_explained_var = grating_regression_result['explained_var_per_X_set'][:,
                                                    grating_exp_vis_ori_model_idx]
                gray_exp_saccade_dir_explained_var = gray_regression_result['explained_var_per_X_set'][:,
                                                     gray_exp_saccade_dir_model_idx]

                saccade_only_neuron_idx = np.where(
                    (grating_exp_vis_ori_explained_var <= 0) &
                    (gray_exp_saccade_dir_explained_var >= 0.1)
                )[0]

                vis_only_neuron_idx = np.where(
                    (gray_exp_saccade_dir_explained_var <= 0) &
                    (grating_exp_vis_ori_explained_var >= 0.1)
                )[0]

                saccade_and_grating_neuron_idx = np.where(
                    (gray_exp_saccade_dir_explained_var >= 0.1) &
                    (grating_exp_vis_ori_explained_var >= 0.1)
                )[0]

                # Plot some example saccade neurons
                grating_Y_hat_model = grating_Y_hat[grating_exp_vis_ori_model_idx, :, :]
                gray_Y_hat_model = gray_Y_hat[gray_exp_saccade_dir_model_idx, :, :]
                grating_Y_hat_model_2 = grating_Y_hat[grating_exp_saccade_dir_model_idx, :, :]

                num_saccade_neurons_to_plot = np.min([len(saccade_only_neuron_idx), num_neurons_to_plot])
                num_visual_neurons_to_plot = np.min([len(vis_only_neuron_idx), num_neurons_to_plot])
                num_both_neurons_to_plot = np.min([len(saccade_and_grating_neuron_idx), num_neurons_to_plot])

                for neuron_idx in saccade_only_neuron_idx[0:num_saccade_neurons_to_plot]:

                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, axs = plt.subplots(2, 1)
                        fig.set_size_inches(8, 4)
                        fig, axs = plot_grating_and_gray_exp_neuron_regression(grating_Y, gray_Y, grating_Y_hat_model,
                                                                           gray_Y_hat_model, grating_Y_hat_model_2,
                                                                           neuron_idx=neuron_idx,
                                                                           fig=fig, axs=axs)

                        fig_name = 'example_saccade_neuron_in_grating_and_gray_exp_%s_neuron_%.f' % (exp_id, neuron_idx)
                        fig.tight_layout()
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                        plt.close(fig)


                # Plot some example visual neurons
                for neuron_idx in vis_only_neuron_idx[0:num_visual_neurons_to_plot]:
                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, axs = plt.subplots(2, 1)
                        fig.set_size_inches(8, 4)
                        fig, axs = plot_grating_and_gray_exp_neuron_regression(grating_Y, gray_Y, grating_Y_hat_model,
                                                                               gray_Y_hat_model, grating_Y_hat_model_2,
                                                                               neuron_idx=neuron_idx,
                                                                               fig=fig, axs=axs)

                        fig_name = 'example_visual_neuron_in_grating_and_gray_exp_%s_neuron_%.f' % (exp_id, neuron_idx)
                        fig.tight_layout()
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                        plt.close(fig)

                # Plot some example both neurons
                for neuron_idx in saccade_and_grating_neuron_idx[0:num_both_neurons_to_plot]:
                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, axs = plt.subplots(2, 1)
                        fig.set_size_inches(8, 4)
                        fig, axs = plot_grating_and_gray_exp_neuron_regression(grating_Y, gray_Y, grating_Y_hat_model,
                                                                               gray_Y_hat_model, grating_Y_hat_model_2,
                                                                               neuron_idx=neuron_idx,
                                                                               fig=fig, axs=axs)

                        fig_name = 'example_both_neuron_in_grating_and_gray_exp_%s_neuron_%.f' % (exp_id, neuron_idx)
                        fig.tight_layout()
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                        plt.close(fig)
        if process == 'compare_saccade_kernels':

            regression_results_folder = process_params[process]['regression_results_folder']
            fig_folder = process_params[process]['fig_folder']
            plot_variance_explained_comparison = process_params[process]['plot_variance_explained_comparison']
            grating_exp_model = process_params[process]['grating_exp_model']
            gray_exp_model = process_params[process]['gray_exp_model']
            # metric_to_plot = 'explained_var_per_X_set'
            metric_to_plot = 'saccade_aligned_var_explained'

            if not os.path.isdir(fig_folder):
                os.makedirs(fig_folder)

            num_neurons_to_plot = process_params[process]['num_neurons_to_plot']
            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % 'grating'))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in
                       regression_result_files]

            if plot_variance_explained_comparison:
                for exp_id in exp_ids:
                    grating_regression_result = np.load(
                        glob.glob(os.path.join(regression_results_folder, '*%s*grating*.npz' % (exp_id)))[0],
                        allow_pickle=True)
                    gray_regression_result = np.load(
                        glob.glob(os.path.join(regression_results_folder, '*%s*gray*.npz' % (exp_id)))[0],
                        allow_pickle=True)

                    # Get the model index in the grating and gray screen experiments
                    grating_exp_saccade_dir_model_idx = np.where(
                        grating_regression_result['X_sets_names'] == 'saccade'
                    )[0][0]

                    gray_exp_saccade_dir_model_idx = np.where(
                        gray_regression_result['X_sets_names'] == 'saccade'
                    )[0][0]

                    grating_exp_saccade_dir_explained_var = grating_regression_result[metric_to_plot][:,
                                                        grating_exp_saccade_dir_model_idx]
                    gray_exp_saccade_dir_explained_var = gray_regression_result[metric_to_plot][:,
                                                         gray_exp_saccade_dir_model_idx]


                    # Plot the explained variance for the saccade model in the grating versus gray screen period
                    fig_name = '%s_saccade_model_ev_in_gray_screen_versus_grating_exp' % exp_id
                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, ax = plt.subplots()
                        fig.set_size_inches(4, 4)
                        both_vals = np.concatenate([gray_exp_saccade_dir_explained_var, grating_exp_saccade_dir_explained_var])
                        both_min = -0.05 # np.min(both_vals)
                        both_max = np.max(both_vals)
                        unity_vals = np.linspace(both_min, both_max, 100)
                        ax.scatter(gray_exp_saccade_dir_explained_var, grating_exp_saccade_dir_explained_var,
                                   lw=0, color='black', s=10)
                        ax.plot(unity_vals, unity_vals, linestyle='--', color='gray', alpha=0.5)
                        ax.set_xlim([both_min, both_max])
                        ax.set_ylim([both_min, both_max])

                        ax.set_xlabel('Gray screen saccade model EV', size=11)
                        ax.set_ylabel('Grating exp saccade model EV', size=11)

                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight', transparent=True)

                    grating_exp_vis_ori_model_idx = np.where(
                        gray_regression_result['X_sets_names'] == 'vis_ori'
                    )[0][0]
                    grating_exp_vis_ori_explained_var = grating_regression_result[metric_to_plot][:,
                                                        grating_exp_vis_ori_model_idx]
                    saccade_only_neuron_idx = np.where(
                         (grating_exp_vis_ori_explained_var <= 0) &
                         (gray_exp_saccade_dir_explained_var >= 0.1)
                    )[0]

                    actual_num_neurons_to_plot = np.min([num_neurons_to_plot, len(saccade_only_neuron_idx)])

                    for neuron_plot_idx in np.arange(actual_num_neurons_to_plot):

                        neuron_idx = saccade_only_neuron_idx[neuron_plot_idx]

                        neuron_grating_exp_saccade_aligned_Y_hat = grating_regression_result['Y_hat_saccade_aligned'][
                                                                   grating_exp_saccade_dir_model_idx, :, :, neuron_idx]
                        neuron_grating_exp_saccade_aligned_Y_hat_mean = np.mean(neuron_grating_exp_saccade_aligned_Y_hat, axis=0)

                        neuron_gray_exp_saccade_aligned_Y_hat = gray_regression_result['Y_hat_saccade_aligned'][
                                                                   gray_exp_saccade_dir_model_idx, :, :, neuron_idx]
                        neuron_gray_exp_saccade_aligned_Y_hat_mean = np.mean(neuron_gray_exp_saccade_aligned_Y_hat, axis=0)

                        # TODO: harded coded for now, need to save window time in regression result files
                        peri_saccade_time = np.linspace(0, 3, len(neuron_grating_exp_saccade_aligned_Y_hat_mean))

                        with plt.style.context(splstyle.get_style('nature-reviews')):
                            fig, ax = plt.subplots()
                            fig.set_size_inches(6, 4)


                            ax.plot(peri_saccade_time, neuron_grating_exp_saccade_aligned_Y_hat_mean, color='black', label='Grating exp')
                            ax.plot(peri_saccade_time, neuron_gray_exp_saccade_aligned_Y_hat_mean, color='gray', label='Gray exp')

                            ax.legend()

                        ax.set_ylabel('Activity (z-scored)', size=11)
                        ax.set_xlabel('Peri-saccade time (s)')
                        fig_name = '%s_neuron_%.f_saccade_kernel_gray_and_grating_exp.png' % (exp_id, neuron_idx)
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                        plt.close(fig)


                    # vis_only_neuron_idx = np.where(
                    #     (gray_exp_saccade_dir_explained_var <= 0) &
                    #   (grating_exp_vis_ori_explained_var >= 0.1)
                    # )[0]

                    # saccade_and_grating_neuron_idx = np.where(
                    #     (gray_exp_saccade_dir_explained_var >= 0.1) &
                    #     (grating_exp_vis_ori_explained_var >= 0.1)
                    # )[0]


            # Load all experiment data
            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            # Compare the temporal profile of the saccade kernels
            for exp_id in exp_ids:

                # Experiment data
                exp_data = data[exp_id]

                grating_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*grating*.npz' % (exp_id)))[0],
                    allow_pickle=True)
                gray_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*gray*.npz' % (exp_id)))[0],
                    allow_pickle=True)

                # Get the model index in the grating and gray screen experiments
                grating_exp_saccade_dir_model_idx = np.where(
                    grating_regression_result['X_sets_names'] == grating_exp_model
                )[0][0]

                gray_exp_saccade_dir_model_idx = np.where(
                    gray_regression_result['X_sets_names'] == gray_exp_model
                )[0][0]

                grating_exp_model_explained_var = grating_regression_result[metric_to_plot][:,
                                                        grating_exp_saccade_dir_model_idx]

                gray_exp_model_explained_var = gray_regression_result[metric_to_plot][:,
                                                     gray_exp_saccade_dir_model_idx]

                gray_exp_model_saccade_aligned_ev = gray_regression_result['saccade_aligned_var_explained'][:, gray_exp_saccade_dir_model_idx]

                """
                grating_exp_Y_hat_saccade_aligned = grating_regression_result['Y_hat_saccade_aligned'][grating_exp_saccade_dir_model_idx, :, :, :]
                gray_exp_Y_hat_saccade_aligned = gray_regression_result['Y_hat_saccade_aligned'][gray_exp_saccade_dir_model_idx, :, :, :]  # Trial x Time x Neurons
                gray_exp_Y_saccade_aligned = gray_regression_result['Y_saccade_aligned'][gray_exp_saccade_dir_model_idx, :, :, :]

                assert len(exp_data['_saccadeVisDir']) == np.shape(grating_exp_Y_hat_saccade_aligned)[0]

                grating_exp_subset_trial_idx = get_saccade_trials_without_grating(exp_data)
                grating_exp_saccade_dir = exp_data['_saccadeVisDir'].flatten() # 0 : nasal, 1 : temporal

                grating_exp_Y_hat_saccade_aligned_subset = grating_exp_Y_hat_saccade_aligned[grating_exp_subset_trial_idx, :, :]
                grating_exp_saccade_dir_subset = grating_exp_saccade_dir[grating_exp_subset_trial_idx]

                grating_exp_saccade_temporal_trials = grating_exp_Y_hat_saccade_aligned_subset[grating_exp_saccade_dir_subset == 1, :, :]
                grating_exp_saccade_nasal_trials = grating_exp_Y_hat_saccade_aligned_subset[grating_exp_saccade_dir_subset == 0, :, :]

                gray_exp_saccade_dir = exp_data['_trial_Dir'].flatten()

                grating_exp_Y_hat_saccade_aligned_temporal = gray_exp_Y_hat_saccade_aligned[gray_exp_saccade_dir == 1, :, :]
                grating_exp_Y_hat_saccade_aligned_nasal = gray_exp_Y_hat_saccade_aligned[gray_exp_saccade_dir == 0, :, :]
                """

                # Obtain the temporal kernel (weights) for the saccade response of each neuron
                grating_exp_model_weights = grating_regression_result['model_weights_per_X_set'].item()[grating_exp_model]
                grating_exp_model_weights_mean = np.mean(grating_exp_model_weights, axis=0)  # mean across the cross validation sets
                grating_exp_saccade_on_weights_index = grating_regression_result['feature_indices_per_X_set'].item()[grating_exp_model]['saccade_on']
                grating_exp_saccade_dir_weights_index = \
                grating_regression_result['feature_indices_per_X_set'].item()[grating_exp_model]['saccade_dir']

                grating_exp_saccade_on_weights = grating_exp_model_weights_mean[:, grating_exp_saccade_on_weights_index]
                grating_exp_saccade_dir_weights = grating_exp_model_weights_mean[:, grating_exp_saccade_dir_weights_index]

                gray_exp_model_weights = gray_regression_result['model_weights_per_X_set'].item()[gray_exp_model]
                gray_exp_model_weights_mean = np.mean(gray_exp_model_weights, axis=0)  # mean across the cross validation sets
                gray_exp_saccade_on_weights_index = \
                    gray_regression_result['feature_indices_per_X_set'].item()[gray_exp_model]['saccade_on']
                gray_exp_saccade_dir_weights_index = \
                    gray_regression_result['feature_indices_per_X_set'].item()[gray_exp_model]['saccade_dir']

                gray_exp_saccade_on_weights = gray_exp_model_weights_mean[:, gray_exp_saccade_on_weights_index]
                gray_exp_saccade_dir_weights = gray_exp_model_weights_mean[:, gray_exp_saccade_dir_weights_index]

                # Calculate the correlation of weights for each neuron
                num_neurons = np.shape(gray_exp_saccade_on_weights)[0]
                saccade_on_corr_per_neuron = np.zeros((num_neurons, ))
                saccade_dir_corr_per_neuron = np.zeros((num_neurons, ))
                for neuron_idx in np.arange(num_neurons):
                    saccade_on_corr_per_neuron[neuron_idx], _ = sstats.pearsonr(grating_exp_saccade_on_weights[neuron_idx, :],
                                                                                gray_exp_saccade_on_weights[neuron_idx, :])

                    saccade_dir_corr_per_neuron[neuron_idx], _ = sstats.pearsonr(grating_exp_saccade_dir_weights[neuron_idx, :],
                                                                                 gray_exp_saccade_dir_weights[neuron_idx, :])


                saccade_on_sort_idx = np.argsort(saccade_on_corr_per_neuron)[::-1]
                saccade_dir_sort_idx = np.argsort(saccade_dir_corr_per_neuron)[::-1]

                # Saccade onset kernel correlation of all neurons
                with plt.style.context(splstyle.get_style('nature-reviews')):

                    fig, axs = plt.subplots(1, 3, sharey=True)
                    fig.set_size_inches(8, 4)

                    all_saccade_on_weights = np.concatenate([gray_exp_saccade_on_weights, grating_exp_saccade_on_weights])
                    vmin = np.percentile(all_saccade_on_weights.flatten(), 10)
                    vmax = np.percentile(all_saccade_on_weights.flatten(), 90)

                    axs[0].imshow(gray_exp_saccade_on_weights[saccade_on_sort_idx, :], aspect='auto', vmin=vmin, vmax=vmax)
                    axs[1].imshow(grating_exp_saccade_on_weights[saccade_on_sort_idx, :], aspect='auto', vmin=vmin, vmax=vmax)
                    axs[0].set_xlabel('Time (frames)', size=11)
                    axs[1].set_xlabel('Time (frames)', size=11)
                    axs[0].set_ylabel('Neurons', size=11)
                    axs[0].set_title('Gray exp', size=11)
                    axs[1].set_title('Grating exp', size=11)

                    axs[2].plot(saccade_on_corr_per_neuron[saccade_on_sort_idx], np.arange(num_neurons))
                    axs[2].set_xlabel('Kernel correlation', size=11)

                    fig.suptitle('%s saccade on kernel' % exp_id, size=11)

                    fig_name = '%s_saccade_on_kernel_raster_corr_sorted' % exp_id
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight', transparent=True)

                plt.close(fig)

                # Saccade direction kernel correlation of all neurons
                with plt.style.context(splstyle.get_style('nature-reviews')):

                    fig, axs = plt.subplots(1, 3, sharey=True)
                    fig.set_size_inches(8, 4)

                    all_saccade_dir_weights = np.concatenate(
                        [gray_exp_saccade_dir_weights, grating_exp_saccade_dir_weights])
                    vmin = np.percentile(all_saccade_dir_weights.flatten(), 10)
                    vmax = np.percentile(all_saccade_dir_weights.flatten(), 90)

                    axs[0].imshow(gray_exp_saccade_dir_weights[saccade_dir_sort_idx, :], aspect='auto', vmin=vmin, vmax=vmax)
                    axs[1].imshow(grating_exp_saccade_dir_weights[saccade_dir_sort_idx, :], aspect='auto', vmin=vmin, vmax=vmax)
                    axs[0].set_xlabel('Time (frames)', size=11)
                    axs[1].set_xlabel('Time (frames)', size=11)
                    axs[0].set_ylabel('Neurons', size=11)
                    axs[0].set_title('Gray exp', size=11)
                    axs[1].set_title('Grating exp', size=11)

                    axs[2].plot(saccade_dir_corr_per_neuron[saccade_dir_sort_idx], np.arange(num_neurons))
                    axs[2].set_xlabel('Kernel correlation', size=11)

                    fig.suptitle('%s saccade dir kernel' % exp_id, size=11)

                    fig_name = '%s_saccade_dir_kernel_raster_corr_sorted' % exp_id
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight', transparent=True)

                plt.close(fig)

                # See if there is correlation between variance explained in gray exp model and correlation
                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plt.subplots(1, 2)
                    fig.set_size_inches(8, 4)

                    axs[0].scatter(gray_exp_model_saccade_aligned_ev, saccade_on_corr_per_neuron, color='black', s=8)
                    axs[1].scatter(gray_exp_model_saccade_aligned_ev, saccade_dir_corr_per_neuron, color='black', s=8)

                    axs[0].set_xlabel('Gray exp explained var', size=11)
                    axs[1].set_xlabel('Gray exp explained var', size=11)

                    axs[0].set_ylabel('Saccade on kernel correlation', size=11)
                    axs[1].set_ylabel('Saccade dir kernel correlation', size=11)

                    # Lines to denote no correlation and zero variance explained
                    [ax.axvline(0, linestyle='--', color='gray', alpha=0.25) for ax in axs]
                    [ax.axhline(0, linestyle='--', color='gray', alpha=0.25) for ax in axs]

                    fig.suptitle('%s' % exp_id, size=11)

                    fig_name = '%s_overall_gray_exp_saccade_aligned_ev_vs_saccade_kernel_corr' % exp_id
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight', transparent=False)

                plt.close(fig)


                # Get the top 10 neurons from each recording, and plot their traces
                gray_exp_model_saccade_aligned_ev_sort_idx = np.argsort(gray_exp_model_saccade_aligned_ev)[::-1]

                for neuron_idx in gray_exp_model_saccade_aligned_ev_sort_idx[0:num_neurons_to_plot]:

                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, axs = plt.subplots(1, 4)
                        fig.set_size_inches(12, 3)

                        # Plot where it is located on the correlation axis
                        axs[0].scatter(gray_exp_model_saccade_aligned_ev, saccade_on_corr_per_neuron, color='black',
                                       s=8)
                        axs[0].scatter(gray_exp_model_saccade_aligned_ev[neuron_idx], saccade_on_corr_per_neuron[neuron_idx], color='red',
                                       s=8)

                        axs[1].scatter(gray_exp_model_saccade_aligned_ev, saccade_dir_corr_per_neuron, color='black',
                                       s=8)
                        axs[1].scatter(gray_exp_model_saccade_aligned_ev[neuron_idx],
                                       saccade_dir_corr_per_neuron[neuron_idx], color='red',
                                       s=8)
                        axs[0].set_xlabel('Gray exp explained var', size=11)
                        axs[1].set_xlabel('Gray exp explained var', size=11)

                        axs[0].set_ylabel('Saccade on kernel correlation', size=11)
                        axs[1].set_ylabel('Saccade dir kernel correlation', size=11)

                        # Lines to denote no correlation and zero variance explained
                        [ax.axvline(0, linestyle='--', color='gray', alpha=0.25) for ax in axs]
                        [ax.axhline(0, linestyle='--', color='gray', alpha=0.25) for ax in axs]

                        # Plot saccade on temporal kernesl
                        axs[2].plot(gray_exp_saccade_on_weights[neuron_idx, :], color='gray', label='Gray')
                        axs[2].plot(grating_exp_saccade_on_weights[neuron_idx, :], color='black', label='Grating')
                        axs[2].set_ylabel('Weights', size=11)
                        axs[2].set_xlabel('Time (frames)', size=11)

                        # Plot saccade direction temporal kernel
                        axs[3].plot(gray_exp_saccade_dir_weights[neuron_idx, :], color='gray', label='Gray')
                        axs[3].plot(grating_exp_saccade_dir_weights[neuron_idx, :], color='black', label='Grating')
                        axs[3].set_ylabel('Weights', size=11)
                        axs[3].set_xlabel('Time (frames)', size=11)
                        axs[3].legend()

                    fig.suptitle('%s neuron %.f' % (exp_id, neuron_idx), size=11)
                    fig_name = '%s_neuron_%.f_saccade_kernel_comparison' % (exp_id, neuron_idx)
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight', transparent=False)
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

            fig_folder = process_params[process]['fig_folder']
            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            for exp_id, exp_data in data.items():

                vis_exp_times, vis_exp_saccade_onset_times, \
                grating_onset_times, saccade_dirs, grating_orientation_per_trial = get_vis_and_saccade_times(
                    exp_data, exp_type='grating')

                unique_ori, saccade_outside_grating, num_saccade_per_grating = get_num_saccade_per_grating(vis_exp_saccade_onset_times,
                                                                      grating_onset_times,
                                                                      grating_orientation_per_trial)

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(6, 4)
                    ax.plot(unique_ori, num_saccade_per_grating)
                    # ax.axhline(saccade_outside_grating, linestyle='-', label='No grating', color='gray')
                    ax.set_xlabel('Grating orientation', size=11)
                    ax.set_ylabel('Number of saccades')

                    ax.set_title('%s' % exp_id, size=11)
                    fig_name = '%s_number_of_saccades_per_grating' % exp_id
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                    plt.close(fig)

        if process == 'plot_pupil_data':

            fig_folder = process_params[process]['fig_folder']
            pupil_preprocessing_steps = process_params[process]['pupil_preprocessing_steps']
            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            for exp_id, exp_data in data.items():

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plot_pupil_data(exp_data, highlight_nans=process_params[process]['highlight_nans'],
                                              pupil_preprocessing_steps=pupil_preprocessing_steps)

                    fig_name = '%s_pupil_trace' % (exp_id)
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


        if process == 'plot_original_vs_aligned_explained_var':

            fig_folder = process_params[process]['fig_folder']
            regression_result_files = glob.glob(os.path.join(process_params[process]['regression_results_folder'],
                                                             '*npz'))
            X_set_to_plot = process_params[process]['X_set_to_plot']

            original_num_sig_neurons = []
            aligned_num_sig_neurons = []
            sig_threshold_val = 0

            for fpath in regression_result_files:

                regression_result = np.load(fpath, allow_pickle=True)
                X_sets_names = regression_result['X_sets_names']
                explained_var_per_X_set = regression_result['explained_var_per_X_set']
                Y_test = regression_result['Y_test']  # num cv x num time points x num neurons
                Y_test_hat = regression_result['Y_test_hat']  # num model x num cv x num time points x num neurons
                Y = np.concatenate([Y_test[1], Y_test[0]])  # reproduce the entire trace (assume 2 fold cv)
                Y_hat = np.concatenate([Y_test_hat[:, 1, :, :], Y_test_hat[:, 0, :, :]], axis=1)

                X_set_idx = np.where(X_sets_names == X_set_to_plot)[0][0]
                original_explained_var = explained_var_per_X_set[:, X_set_idx]
                if X_set_to_plot in ['saccade', 'saccade_on_only']:
                    aligned_explained_var = regression_result['saccade_aligned_var_explained'][:, X_set_idx]
                    Y_aligned = regression_result['Y_saccade_aligned'][X_set_idx]  # which model is this ???
                    Y_hat_aligned = regression_result['Y_hat_saccade_aligned'][X_set_idx]
                elif X_set_to_plot in ['vis_on_only', 'vis_ori']:
                    aligned_explained_var = regression_result['vis_aligned_var_explained'][:, X_set_idx]
                    Y_aligned = regression_result['Y_vis_aligned'][X_set_idx]
                    Y_hat_aligned = regression_result['Y_hat_vis_aligned'][X_set_idx]


                subset_indices = np.where(
                    (original_explained_var > 0)
                )[0]

                original_num_sig_neurons.append(np.sum(original_explained_var > sig_threshold_val))
                aligned_num_sig_neurons.append(np.sum(aligned_explained_var > sig_threshold_val))

                if process_params[process]['plot_single_neuron_examples']:

                    example_neuron_indices = np.arange(0, 10)

                    for ex_idx in example_neuron_indices:

                        neuron_idx = subset_indices[ex_idx]

                        with plt.style.context(splstyle.get_style('nature-reviews')):
                            fig = plt.figure()
                            fig.set_size_inches(12, 6)
                            gs = fig.add_gridspec(2, 2)
                            ax1 = fig.add_subplot(gs[0, 0])
                            ax2 = fig.add_subplot(gs[0, 1])
                            ax3 = fig.add_subplot(gs[1, 1])
                            ax4 = fig.add_subplot(gs[1, 0])

                            all_explained_var = np.concatenate([original_explained_var, aligned_explained_var])
                            all_ev_min = np.min(all_explained_var)
                            all_ev_max = np.max(all_explained_var)
                            unity_vals = np.linspace(all_ev_min, all_ev_max, 100)

                            ax1.scatter(original_explained_var, aligned_explained_var, color='gray')
                            ax1.scatter(original_explained_var[neuron_idx], aligned_explained_var[neuron_idx], color='black')
                            ax1.axvline(0, linestyle='--', color='gray', lw=0.5, alpha=0.5)
                            ax1.axhline(0, linestyle='--', color='gray', lw=0.5, alpha=0.5)
                            ax1.plot(unity_vals, unity_vals, linestyle='--', color='gray', lw=0.5, alpha=0.5)
                            ax1.set_xlabel('Original explained variance', size=11)
                            ax1.set_ylabel('Aligned explained variance', size=11)

                            Y_neuron = Y[:, neuron_idx]
                            Y_hat_neuron = Y_hat[X_set_idx, :, neuron_idx]
                            Y_aligned_neuron = Y_aligned[:, :, neuron_idx].flatten()
                            Y_hat_aligned_neuron = Y_hat_aligned[:, :, neuron_idx].flatten()

                            ax2.plot(Y_neuron, color='black', lw=1)
                            ax2.plot(Y_hat_neuron, color='red', lw=1)

                            var_y_neuron = np.var(Y_neuron)
                            var_y_minus_y_hat = np.var(Y_neuron - Y_hat_neuron)
                            ev_neuron = 1 - (var_y_minus_y_hat / var_y_neuron)

                            ax2.set_title('Original, Var(y) = %.3f, Var(y - y_hat) = %.3f, ev = %.3f' % (
                                 var_y_neuron, var_y_minus_y_hat, ev_neuron,
                            ), size=9)

                            # ax2.set_title('Original', size=9)

                            y_max = np.max(Y_neuron)
                            y_min = np.min(Y_neuron)
                            ax2.set_ylim([y_min, y_max])


                            var_y_neuron = np.var(Y_aligned_neuron)
                            var_y_minus_y_hat = np.var(Y_aligned_neuron - Y_hat_aligned_neuron)
                            ev_neuron = 1 - (var_y_minus_y_hat / var_y_neuron)

                            ax3.plot(Y_aligned_neuron, color='black', lw=1)
                            ax3.plot(Y_hat_aligned_neuron, color='red', lw=1)

                            ax3.set_ylim([y_min, y_max])

                            ax3.set_title('Aligned, Var(y) = %.3f, Var(y - y_hat) = %.3f, ev = %.3f' %
                                           (
                                               var_y_neuron, var_y_minus_y_hat, ev_neuron,
                                          ), size=11)
                            # ax3.set_title('Aligned', size=11)


                            bins = np.linspace(y_min, y_max, 100)
                            ax4.hist(Y_neuron, color='black', bins=bins)
                            ax4.hist(Y_aligned_neuron, color='gray', bins=bins)


                        exp_id_parts = os.path.basename(fpath).split('.')[0].split('_')
                        subject = exp_id_parts[0]
                        exp_date = exp_id_parts[1]

                        fig.tight_layout()
                        fig_name = '%s_%s_neuron_%.f_%s_original_vs_aligned_trace' % (subject, exp_date, neuron_idx, X_set_to_plot)
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                        plt.close(fig)



            if process_params[process]['plot_overall_summary']:

                all_num_sig_neurons = np.concatenate([original_num_sig_neurons, aligned_num_sig_neurons])
                all_min = np.min(all_num_sig_neurons)
                all_max = np.max(all_num_sig_neurons)
                unity_vals = np.linspace(all_min, all_max, 100)

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(4, 4)
                    ax.plot(unity_vals, unity_vals, linestyle='--', color='gray', lw=1, alpha=0.5)
                    ax.scatter(original_num_sig_neurons, aligned_num_sig_neurons, color='black', lw=0)
                    ax.set_xlabel('Original num sig neurons', size=11)
                    ax.set_ylabel('Aligned num sig neurons', size=11)
                    ax.set_xlim([all_min - 10, all_max + 10])
                    ax.set_ylim([all_min - 10, all_max + 10])
                    ax.set_title('%s' % X_set_to_plot, size=11)
                    fig_name = 'all_exp_num_sig_neurons_in_%s_before_after_alignment' % (X_set_to_plot)
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

        if process == 'plot_sig_vs_nosig_neuron_explained_var':
            print('Plotting the explained variance of significant versus non-significant saccade neurons')

            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            regression_results_folder = process_params[process]['regression_results_folder']
            fig_folder = process_params[process]['fig_folder']
            # metric_to_plot = 'explained_var_per_X_set'
            metric_to_plot = 'saccade_aligned_var_explained'



            if not os.path.isdir(fig_folder):
                os.makedirs(fig_folder)

            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % 'grating'))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in
                       regression_result_files]

            for exp_id, exp_data in data.items():

                grating_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*grating*.npz' % (exp_id)))[0],
                    allow_pickle=True)
                gray_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*gray*.npz' % (exp_id)))[0],
                    allow_pickle=True)

                grating_exp_saccade_dir_model_idx = np.where(
                    grating_regression_result['X_sets_names'] == 'saccade'
                )[0][0]

                gray_exp_saccade_dir_model_idx = np.where(
                    gray_regression_result['X_sets_names'] == 'saccade'
                )[0][0]

                grating_exp_saccade_dir_explained_var = grating_regression_result['explained_var_per_X_set'][:,
                                                    grating_exp_saccade_dir_model_idx]
                gray_exp_saccade_dir_explained_var = gray_regression_result['explained_var_per_X_set'][:,
                                                     gray_exp_saccade_dir_model_idx]

                gray_exp_sig_saccade_neurons = data[exp_id]['_sigNeurons'].flatten()
                gray_exp_sig_saccade_neurons = np.array([int(x) for x in gray_exp_sig_saccade_neurons])
                sig_idx = np.where(gray_exp_sig_saccade_neurons == 1)[0]
                non_sig_idx = np.where(gray_exp_sig_saccade_neurons == 0)[0]
                gray_exp_sig_saccade_neuron_type = data[exp_id]['_sigDirectionNeuron'].flatten()
                # gray_exp_sig_saccade_neuron_type = np.array([int(x) for x in gray_exp_sig_saccade_neuron_type])

                sig_neurons_grating_exp_saccade_dir_ev = grating_exp_saccade_dir_explained_var[sig_idx]
                non_sig_neurons_grating_exp_saccade_dir_ev = grating_exp_saccade_dir_explained_var[non_sig_idx]

                sig_neurons_gray_exp_saccade_dir_ev = gray_exp_saccade_dir_explained_var[sig_idx]
                non_sig_neurons_gray_exp_saccade_dir_ev = gray_exp_saccade_dir_explained_var[non_sig_idx]

                max_ev = np.max(gray_exp_saccade_dir_explained_var)

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plt.subplots(1, 2)
                    fig.set_size_inches(8, 4)
                    bins = np.linspace(-0.05, max_ev, 50)
                    axs[0].hist(sig_neurons_gray_exp_saccade_dir_ev, bins=bins, edgecolor='black', fill=False, label='Sig', histtype='step')
                    axs[0].hist(non_sig_neurons_gray_exp_saccade_dir_ev, bins=bins, edgecolor='gray', fill=False, label='Not sig', histtype='step')
                    axs[0].set_ylabel('Neuron count', size=11)

                    axs[1].hist(sig_neurons_grating_exp_saccade_dir_ev, bins=bins, edgecolor='black', fill=False, label='Sig', histtype='step')
                    axs[1].hist(non_sig_neurons_grating_exp_saccade_dir_ev, bins=bins, edgecolor='gray', fill=False, label='Not sig', histtype='step')

                    axs[0].legend()

                    axs[0].set_title('Gray screen', size=11)
                    axs[1].set_title('Grating experiment', size=11)

                fig_name = '%s_sig_saccade_vs_not_sig_saccade_explained_variance_aligned' % (exp_id)
                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight', transparent=True)

        if process == 'plot_saccade_neuron_psth_and_regression':

            print('Plotting the saccade neurons where significance test and regression results differ')

            exclude_saccade_on_vis_exp = process_params[process]['exclude_saccade_on_vis_exp']

            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            regression_results_folder = process_params[process]['regression_results_folder']
            fig_folder = process_params[process]['fig_folder']
            # metric_to_plot = 'explained_var_per_X_set'
            metric_to_plot = 'saccade_aligned_var_explained'

            if not os.path.isdir(fig_folder):
                os.makedirs(fig_folder)

            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % 'grating'))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in
                       regression_result_files]

            for exp_id, exp_data in data.items():
                grating_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*grating*.npz' % (exp_id)))[0],
                    allow_pickle=True)
                gray_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*gray*.npz' % (exp_id)))[0],
                    allow_pickle=True)

                gray_exp_sig_saccade_neurons = exp_data['_sigNeurons'].flatten()
                gray_exp_sig_saccade_neurons = np.array([int(x) for x in gray_exp_sig_saccade_neurons])

                grating_exp_saccade_dir_model_idx = np.where(
                    grating_regression_result['X_sets_names'] == 'saccade'
                )[0][0]

                gray_exp_saccade_dir_model_idx = np.where(
                    gray_regression_result['X_sets_names'] == 'saccade'
                )[0][0]

                grating_exp_saccade_dir_explained_var = grating_regression_result['explained_var_per_X_set'][:,
                                                        grating_exp_saccade_dir_model_idx]
                gray_exp_saccade_dir_explained_var = gray_regression_result['explained_var_per_X_set'][:,
                                                     gray_exp_saccade_dir_model_idx]


                # Get aligned activity of neurons
                aligned_activity, trial_type, time_windows = \
                    get_aligned_activity(exp_data, exp_type='grating', aligned_event='saccade',
                                     alignment_time_window=[-1, 3], exclude_saccade_on_vis_exp=exclude_saccade_on_vis_exp)

                stat_not_sig_model_sig_saccade_neuron = np.where(
                    (gray_exp_saccade_dir_explained_var > 0) &
                    (gray_exp_sig_saccade_neurons == 0)
                )[0]

                num_neurons_to_plot = 10
                actual_num_neurons_to_plot = np.min([num_neurons_to_plot, len(stat_not_sig_model_sig_saccade_neuron)])

                for neuron_i in np.arange(0, actual_num_neurons_to_plot):
                    
                    with plt.style.context(splstyle.get_style('nature-reviews')):

                        fig, axs = plt.subplots(1, 2)
                        neuron_idx = stat_not_sig_model_sig_saccade_neuron[neuron_i]
                        neuron_activity = aligned_activity[:, :, neuron_idx]
                        neuron_mean = np.mean(neuron_activity, axis=0)

                        neuron_Y_hat_saccade_aligned = gray_regression_result['Y_hat_saccade_aligned'][gray_exp_saccade_dir_model_idx, :, :, neuron_idx]
                        neuron_Y_hat_saccade_aligned_mean = np.mean(neuron_Y_hat_saccade_aligned, axis=0)

                        neuron_Y_saccade_aligned = gray_regression_result['Y_saccade_aligned'][
                                                       gray_exp_saccade_dir_model_idx, :, :, neuron_idx]
                        neuron_Y_saccade_aligned_mean = np.mean(neuron_Y_saccade_aligned, axis=0)

                        axs[0].plot(time_windows, neuron_mean)
                        axs[1].plot(neuron_Y_saccade_aligned_mean, color='black')
                        axs[1].plot(neuron_Y_hat_saccade_aligned_mean, color='red')



                    pdb.set_trace()
        if process == 'plot_sig_model_comparison_neurons':

            exp_type = process_params[process]['exp_type']
            regression_result_files = glob.glob(os.path.join(process_params[process]['regression_results_folder'],
                                                             '*%s*npz' % exp_type))
            X_sets_to_compare = process_params[process]['X_sets_to_compare']
            # metrics_to_compare = process_params[process]['metrics_to_compare']
            custom_fig_addinfo = process_params[process]['custom_fig_addinfo']
            num_neurons_to_plot = process_params[process]['num_neurons_to_plot']
            model_a_explained_var_threshold = process_params[process]['model_a_explained_var_threshold']
            model_b_explained_var_threshold = process_params[process]['model_b_explained_var_threshold']
            min_model_ev_diff = process_params[process]['min_model_ev_diff']

            fig_folder = process_params[process]['fig_folder']
            text_size = 11

            for fpath in regression_result_files:
                regression_result = np.load(fpath, allow_pickle=True)
                exp_id = '_'.join(os.path.basename(fpath).split('_')[0:2])
                X_sets_names = regression_result['X_sets_names']

                explained_var_per_X_set = regression_result['explained_var_per_X_set']
                Y_test = regression_result['Y_test']  # num cv x num time points x num neurons
                Y_test_hat = regression_result['Y_test_hat'] # num model x num cv x num time points x num neurons

                if len(Y_test) == 2:
                    print('Assuming half test half train')
                    Y = np.concatenate([Y_test[1], Y_test[0]])  # reproduce the entire trace (assume 2 fold cv)
                    try:
                        Y_hat = np.concatenate([Y_test_hat[:, 1, :, :], Y_test_hat[:, 0, :, :]], axis=1)
                    except:
                        pdb.set_trace()
                else:
                    Y = np.concatenate(Y_test)
                    num_x_set = np.shape(Y_test_hat)[0]
                    Y_hat = np.array([np.concatenate(Y_test_hat[x_set_idx, :]) for x_set_idx in np.arange(num_x_set)])
                    all_test_idx = np.concatenate(regression_result['test_idx_per_cv_set'])

                for model_a, model_b in X_sets_to_compare:
                    model_a_idx = np.where(X_sets_names == model_a)[0][0]
                    model_b_idx = np.where(X_sets_names == model_b)[0][0]

                    model_a_explained_var = explained_var_per_X_set[:, model_a_idx]
                    model_b_explained_var = explained_var_per_X_set[:, model_b_idx]

                    both_model_explained_var = np.concatenate([model_a_explained_var, model_b_explained_var])
                    both_min = np.min(both_model_explained_var)
                    both_max = np.max(both_model_explained_var)
                    unity_vals = np.linspace(both_min, both_max, 100)

                    model_a_Y_hat = Y_hat[model_a_idx, :, :]
                    model_b_Y_hat = Y_hat[model_b_idx, :, :]


                    if min_model_ev_diff > 0:
                        sig_neuron_indices = np.where(
                            ((model_a_explained_var - model_b_explained_var) > min_model_ev_diff) &
                            (model_a_explained_var > model_a_explained_var_threshold) &
                            (model_b_explained_var > model_b_explained_var_threshold)
                        )[0]
                    else:
                        sig_neuron_indices = np.where(
                            (model_a_explained_var > model_b_explained_var) &
                            (model_a_explained_var > model_a_explained_var_threshold) &
                            (model_b_explained_var > model_b_explained_var_threshold)
                        )[0]

                    num_neurons_to_plot = np.min([len(sig_neuron_indices), num_neurons_to_plot])

                    for neuron_plot_idx in np.arange(0, num_neurons_to_plot):

                        neuron_idx = sig_neuron_indices[neuron_plot_idx]

                        with plt.style.context(splstyle.get_style('nature-reviews')):
                            fig = plt.figure()
                            gs = fig.add_gridspec(2, 3)
                            ax1 = fig.add_subplot(gs[0, 0])
                            ax2 = fig.add_subplot(gs[0, 1])
                            ax3 = fig.add_subplot(gs[0, 2])
                            ax4 = fig.add_subplot(gs[1, :])
                            fig.set_size_inches(9, 6)

                        # Plot model comparison
                        ax1.axvline(0, linestyle='--', color='gray', lw=0.5)
                        ax1.axhline(0, linestyle='--', color='gray', lw=0.5)
                        ax1.plot(unity_vals, unity_vals, linestyle='--', color='gray', lw=0.5)

                        ax1.scatter(model_a_explained_var, model_b_explained_var, color='black', s=10)
                        ax1.scatter(model_a_explained_var[neuron_idx], model_b_explained_var[neuron_idx], color='red', s=10)
                        ax1.set_xlabel(model_a, size=11)
                        ax1.set_ylabel(model_b, size=11)
                        ax1.set_xlim([both_min, both_max])
                        ax1.set_ylim([both_min, both_max])

                        ax1.set_title('Explained variance', size=11)

                        # Plot visual aligned activity
                        grating_orientation_per_trial = regression_result['grating_orientation_per_trial']
                        oris_to_plot = np.sort(np.unique(grating_orientation_per_trial[~np.isnan(grating_orientation_per_trial)]))
                        grating_cmap = mpl.cm.get_cmap(name='viridis')
                        grating_colors = [grating_cmap(x/np.max(oris_to_plot)) for x in oris_to_plot]
                        for n_ori, ori in enumerate(oris_to_plot):
                            trial_idx = np.where(grating_orientation_per_trial == ori)[0]
                            ori_aligned_activity = regression_result['Y_vis_aligned_full'][trial_idx, :, neuron_idx]
                            ori_aligned_activity_mean = np.mean(ori_aligned_activity, axis=0)
                            ax2.plot(ori_aligned_activity_mean, color=grating_colors[n_ori])

                        ax2.set_title('Grating', size=11)

                        # Plot saccade aligned activity
                        saccade_dirs = regression_result['saccade_dirs']
                        saccade_dir_colors = {
                            -1 : 'green',  # nasal
                            1: 'purple', # temporal
                        }
                        for s_dir in np.unique(saccade_dirs):
                            trial_idx = np.where(saccade_dirs == s_dir)[0]
                            s_dir_aligned_activity = regression_result['Y_saccade_aligned_full'][trial_idx, :, neuron_idx]
                            s_dir_aligned_activity_mean = np.mean(s_dir_aligned_activity, axis=0)
                            ax3.plot(s_dir_aligned_activity_mean, color=saccade_dir_colors[s_dir], lw=1.5)
                        ax3.set_title('Saccade', size=11)

                        ax4.plot(Y[:, neuron_idx], label='original', lw=1, color='black')
                        ax4.plot(model_a_Y_hat[:, neuron_idx], label='%s' % model_a, lw=1)
                        ax4.plot(model_b_Y_hat[:, neuron_idx], label='%s' % model_b, lw=1)
                        ax4.legend()

                        fig_name = '%s_neuron_%s_model_%s_vs_%s_traces' % (exp_id, neuron_idx, model_a, model_b)
                        fig.tight_layout()
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight', transparent=False)

                        plt.close(fig)

        if process == 'plot_vis_and_saccade_neuron_individual_trials':

            regression_results_folder = process_params[process]['regression_results_folder']
            fig_folder = process_params[process]['fig_folder']
            plot_variance_explained_comparison = process_params[process]['plot_variance_explained_comparison']
            grating_exp_model = process_params[process]['grating_exp_model']
            gray_exp_model = process_params[process]['gray_exp_model']
            # metric_to_plot = 'explained_var_per_X_set'
            metric_to_plot = 'saccade_aligned_var_explained'

            if not os.path.isdir(fig_folder):
                os.makedirs(fig_folder)

            num_neurons_to_plot = process_params[process]['num_neurons_to_plot']
            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % 'grating'))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in
                       regression_result_files]

            # Load all experiment data
            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            # Compare the temporal profile of the saccade kernels
            exp_type = 'both'
            for exp_id in exp_ids:
                # Experiment data
                exp_data = data[exp_id]

                regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*%s*.npz' % (exp_id, exp_type)))[0],
                    allow_pickle=True)

                """
                grating_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*grating*.npz' % (exp_id)))[0],
                    allow_pickle=True)
                gray_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*gray*.npz' % (exp_id)))[0],
                    allow_pickle=True)
                """

                X_sets_names = regression_result['X_sets_names']
                explained_var_per_X_set = regression_result['explained_var_per_X_set']

                model_a = 'vis_ori'
                model_b = 'vis_and_saccade'

                model_a_idx = np.where(X_sets_names == model_a)[0][0]
                model_b_idx = np.where(X_sets_names == model_b)[0][0]

                model_a_explained_var = explained_var_per_X_set[:, model_a_idx]
                model_b_explained_var = explained_var_per_X_set[:, model_b_idx]

                b_minus_a = model_b_explained_var - model_a_explained_var

                subset_idx = np.where(
                    (model_a_explained_var > 0) &
                    (model_b_explained_var > 0) &
                    (model_b_explained_var > model_a_explained_var)
                )[0]

                vis_subset_idx = np.where(
                    (model_a_explained_var > 0)
                )[0]

                b_minus_a_subset_sort_idx = np.argsort(b_minus_a[subset_idx])[::-1]
                b_minus_a_sort_idx = subset_idx[b_minus_a_subset_sort_idx]

                both_model_explained_var = np.concatenate([model_a_explained_var, model_b_explained_var])
                both_min = np.min(both_model_explained_var)
                both_max = np.max(both_model_explained_var)
                unity_vals = np.linspace(both_min, both_max, 100)

                # Get visual aligned activity of all neurons
                vis_aligned_activity, vis_ori, saccade_dir_during_vis, saccade_time, time_windows = get_aligned_activity(
                    exp_data, exp_type='grating', aligned_event='vis',
                    alignment_time_window=[-1, 3],
                    exclude_saccade_on_vis_exp=False)

                saccade_on_trials = np.where([(len(x) > 0) for x in saccade_dir_during_vis])[0]
                saccade_off_trials = np.where([(len(x) == 0) for x in saccade_dir_during_vis])[0]
                saccade_time_saccade_on_trials = [x for x in saccade_time if len(x) > 0]
                saccade_dir_saccade_on_trials = [x for x in saccade_dir_during_vis if len(x) > 0]

                vis_aligned_activity_saccade_off = vis_aligned_activity[saccade_off_trials, :, :]
                vis_aligned_activity_saccade_on = vis_aligned_activity[saccade_on_trials, :, :]

                vis_ori_saccade_off = vis_ori[saccade_off_trials]
                vis_ori_saccade_on = vis_ori[saccade_on_trials]

                vis_ori_saccade_on_sort_idx = np.argsort(vis_ori_saccade_on)
                vis_ori_saccade_off_sort_idx = np.argsort(vis_ori_saccade_off)

                grating_cmap = mpl.cm.get_cmap(name='viridis')
                vis_ori_subset = np.sort(np.unique(vis_ori[~np.isnan(vis_ori)]))
                grating_colors = [grating_cmap(x / np.max(vis_ori_subset)) for x in vis_ori_subset]

                for neuron_idx in b_minus_a_sort_idx[0:num_neurons_to_plot]:

                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, axs = plt.subplots(1, 3)
                        fig.set_size_inches(9, 6)

                        # Plot explained variance
                        axs[0].axvline(0, linestyle='--', color='gray', lw=1, alpha=0.3)
                        axs[0].axhline(0, linestyle='--', color='gray', lw=1, alpha=0.3)
                        axs[0].plot(unity_vals, unity_vals, linestyle='--', lw=1, color='gray', alpha=0.3)
                        axs[0].scatter(model_b_explained_var, model_a_explained_var, color='black', s=8)
                        axs[0].scatter(model_b_explained_var[neuron_idx], model_a_explained_var[neuron_idx], color='red', s=8)
                        axs[0].set_xlabel(model_b, size=11)
                        axs[0].set_ylabel(model_a, size=11)

                        # Plot when there is no saccade
                        y_offset = 0
                        for within_saccade_off_idx in vis_ori_saccade_off_sort_idx:
                            trial_vis_ori = vis_ori_saccade_off[within_saccade_off_idx]

                            if np.isnan(trial_vis_ori):
                                color = 'gray'
                            else:
                                color = grating_colors[np.where(vis_ori_subset == trial_vis_ori)[0][0]]

                            trial_trace = vis_aligned_activity_saccade_off[within_saccade_off_idx, :, neuron_idx]
                            axs[1].plot(time_windows, y_offset + trial_trace, color=color, lw=1)
                            y_offset += np.max(trial_trace)


                        # Plot visual onset aligned traces sorted by orientation and saccade
                        y_offset = 0
                        saccade_color_map = {
                            -1: 'green',  # nasal
                            1: 'purple',  # temporal
                        }
                        for within_saccade_on_idx in vis_ori_saccade_on_sort_idx:
                            trial_vis_ori = vis_ori_saccade_on[within_saccade_on_idx]
                            trial_saccade_time_rel_vis = saccade_time_saccade_on_trials[within_saccade_on_idx]
                            trial_saccade_dirs = saccade_dir_saccade_on_trials[within_saccade_on_idx]
                            trial_saccade_colors = [saccade_color_map[x] for x in trial_saccade_dirs]

                            if np.isnan(trial_vis_ori):
                                color = 'gray'
                            else:
                                color = grating_colors[np.where(vis_ori_subset == trial_vis_ori)[0][0]]

                            trial_trace = vis_aligned_activity_saccade_on[within_saccade_on_idx, :, neuron_idx]
                            axs[2].plot(time_windows, y_offset + trial_trace, color=color, lw=1)
                            saccade_marker_y = np.mean(trial_trace) + y_offset
                            
                            axs[2].scatter(trial_saccade_time_rel_vis, np.repeat(saccade_marker_y, len(trial_saccade_time_rel_vis)), 
                                           color=trial_saccade_colors, marker='v', s=3)
                            y_offset += np.max(trial_trace)

                        # Legend
                        legend_elements = []

                        legend_elements.append(
                            mpl.lines.Line2D([0], [0], color='gray', lw=1, label='blank')
                        )
                        # Flip the order just to match the one in the plot
                        for ori_idx, color in enumerate(grating_colors[::-1]):
                            legend_elements.append(
                                mpl.lines.Line2D([0], [0], color=color, lw=1, label=vis_ori_subset[::-1][ori_idx]),
                            )

                        legend_elements.append(
                            mpl.lines.Line2D([0], [0], marker='v', color='green', lw=0, label='Nasal', markeredgecolor='None')
                        )
                        legend_elements.append(
                            mpl.lines.Line2D([0], [0], marker='v', color = 'purple', lw=0, label='Temporal', markeredgecolor='None')
                        )

                        fig.legend(handles=legend_elements, bbox_to_anchor=(1.04, 0.7))

                        fig.suptitle('%s neuron %.f' % (exp_id, neuron_idx), size=11)
                        fig_name = '%s_neuron_%.f_vis_aligned_ori_sorted_response_with_saccade_time' % (exp_id, neuron_idx)
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                        plt.close(fig)


                # Get the prefer orientation of all neurons
                num_neurons = np.shape(vis_aligned_activity)[2]
                mean_response_per_ori = np.zeros((len(vis_ori_subset), num_neurons))
                for ori_i, ori in enumerate(vis_ori_subset):
                    trial_idx = np.where(vis_ori == ori)[0]
                    ori_activity = np.mean(vis_aligned_activity[trial_idx, :, :], axis=0)
                    mean_response_per_ori[ori_i, :] = np.mean(ori_activity, axis=0)

                pref_idx_per_neuron = np.argmax(mean_response_per_ori, axis=0)
                pref_ori_per_neuron = vis_ori_subset[pref_idx_per_neuron]

                # Plot the prefer orientation of all the "signicicant" neurons
                all_neuron_num_neuron_per_ori = np.array([len(np.where(pref_ori_per_neuron == x)[0]) for x in vis_ori_subset])
                vis_saccade_neuron_num_neuron_per_ori = np.array([len(np.where(pref_ori_per_neuron[subset_idx] == x)[0]) for x in vis_ori_subset])
                vis_neuron_num_neuron_per_ori = np.array([len(np.where(pref_ori_per_neuron[vis_subset_idx] == x)[0]) for x in vis_ori_subset])

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plt.subplots(1, 3, sharex=True)
                    fig.set_size_inches(9, 3)
                    axs[0].bar(vis_ori_subset, all_neuron_num_neuron_per_ori, lw=5, color='black')
                    axs[1].bar(vis_ori_subset, vis_neuron_num_neuron_per_ori, lw=5, color='black')
                    axs[2].bar(vis_ori_subset, vis_saccade_neuron_num_neuron_per_ori, lw=5, color='black')

                    axs[0].set_ylabel('Number of neurons', size=11)
                    axs[0].set_title('All neurons', size=11)
                    axs[1].set_title('Vis neurons', size=11)
                    axs[2].set_title('Vis + Saccade neurons', size=11)

                    fig.text(0.5, 0, 'Preferred orientation', ha='center', size=11)
                    axs[0].set_xticks([0, 90, 180, 270, 360])

                    fig.suptitle('%s' % (exp_id), size=11)
                    fig_name = '%s_neuron_preferred_orientation' % (exp_id)
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                    plt.close(fig)

if __name__ == '__main__':
    main()