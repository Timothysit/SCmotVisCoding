import os
import glob
import time
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

# Plotting

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import sciplotlib.style as splstyle
import sciplotlib.text as spltext
import matplotlib_venn as mpl_venn

# Debugging
import pdb  # python debugger

from collections import defaultdict
import pandas as pd
from tqdm import tqdm  # loading bar


# Stats
import statsmodels.api as sm
from pymer4.models import Lmer
import scipy.optimize as spopt
import scipy.io as spio
import scipy
import cmath  # for circular statistics

# parallel processing
import ray

# Sorting / Clustering
import rastermap

import itertools


def gradient_respecting_bounds(bounds, fun, eps=1e-8):
    """bounds: list of tuples (lower, upper)"""
    def gradient(x):
        fx = fun(x)
        grad = np.zeros(len(x))
        for k in range(len(x)):
            d = np.zeros(len(x))
            d[k] = eps if x[k] + eps <= bounds[k][1] else -eps
            grad[k] = (fun(x + d) - fx) / d[k]
        return grad

    return gradient


def von_mises(x, mu, kappa):
    return np.exp( kappa*np.cos((x - mu)) ) / (2 * np.pi * scipy.special.i0(kappa))

def von_mises2(x, par1, par2, par3, par4, par5):
    """
    Inputs take in

    Parameters
    ----------
    x : numpy ndarray
        the presented stimuli orientation in degrees
    par1 : float
        the peak tuning (in degrees)
    par2 : float
        scales the response at the primary peak
    par3 :
    par4 :
    par5 :
    """

    Dp = par1 * np.pi / 180  # convert from degrees to radians
    Rp = par2
    Rn = par3
    Ro = par4
    kappa = 1 / (par5 * np.pi / 180) # kappa is approximately sigma^2 of the Guassian
    alpha = x * np.pi / 180

    # kappa = 1 / kappa # 1 / kappa is approximately sigma^2 of the Guassian

    s_vm_p = np.exp( kappa * (np.cos(alpha - Dp))) / (2 * np.pi * scipy.special.ive(0, kappa))
    s_vm_n = np.exp( kappa * (np.cos(alpha - (Dp + np.pi)))) / (2 * np.pi * scipy.special.ive(0, kappa))

    s_vm_p = s_vm_p / np.max(s_vm_p)
    s_vm_n = s_vm_n / np.max(s_vm_n)

    r = Rp * s_vm_p + Rn * s_vm_n + Ro

    return r


def von_mises2_loss(params, x, y):

    par1, par2, par3, par4, par5 = params

    y_pred = von_mises2(x, par1, par2, par3, par4, par5)
    mse = np.nanmean((y - y_pred) ** 2)

    return mse


def load_data(data_folder, file_types_to_load=['_windowVis'],
              exp_ids=['SS041_2015-04-23', 'SS044_2015-04-28', 'SS045_2015-05-04',
                       'SS045_2015-05-05', 'SS047_2015-11-23', 'SS047_2015-12-03',
                       'SS048_2015-11-09', 'SS048_2015-12-02']):
    """
    Load data to do regression analysis
    Parameters
    -----------
    data_folder : str
        path to folder with the experiment data
    file_types_to_load : list
        list of files to load in the folder
        _windowVis : 1 x time array with the time (in seconds) of recording in the visual grating experiment
        _widowGray : 1 x time array with the time (in seconds) of recording in the gray screen experiment
    exp_ids : list
        list of experiment IDs to load
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

            if file_type == '_inPerNeuron':
                file_paths = [x for x in file_paths if 'Vis' not in os.path.basename(x)]

            if len(file_paths) != 1:
                print('0 or more than 1 path found, please debug')
                pdb.set_trace()

            file_data = np.load(file_paths[0])

            data[exp_id][file_type] = file_data

    return data


def loadmat(filename, struct_as_record=False):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects

    struct_as_record : if True, you access fields using data['field']
    if False, you access field using dot notation: data.field

    '''

    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                # TS: Make sure numpy array is at least 1 dimensional
                # Otherwise the loop in _tolist() will complain about 0-d arrays.
                elem = np.atleast_1d(elem)
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=struct_as_record, squeeze_me=True)
    return _check_keys(data)

def load_running_data(exp_folder):
    """
    Load running speed data
    exp_folder : str

    """
    # find running data and start plotting
    running_speed_data_path = os.path.join(exp_folder, '_ss_running.speed.npy')
    running_speed_timestamp_data_path = os.path.join(exp_folder, '_ss_running.timestamps.npy')
    gray_screen_interval_data_path = os.path.join(exp_folder, '_ss_recordings.grayScreen_intervals.npy')
    grating_interval_data_path = os.path.join(exp_folder, '_ss_recordings.gratings_intervals.npy')

    running_speed = np.load(running_speed_data_path).flatten()
    running_speed_timestamps = np.load(running_speed_timestamp_data_path).flatten()
    grating_interval = np.load(grating_interval_data_path).flatten()
    gray_interval = np.load(gray_screen_interval_data_path).flatten()

    return running_speed, running_speed_timestamps, grating_interval, gray_interval


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
    exclude_saccade_on_vis_exp : bool
        whether to exclude saccade onset times during visual stimulus experiment
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
                            exclude_saccade_off_screen_times=False,
                            exclude_saccade_any_off_screen_times=False,
                            saccade_off_screen_exclusion_window=[-0.5, 0.5],
                            return_trial_type=False, pupil_preprocessing_steps=[]):
    """
    Make feature matrix (or design matrix) X and target matrix Y from experiment data

    Parameters
    ----------
    exp_data : dict
    feature_set : list
        list of features to use in the regression model 
    feature_time_windows : dict
        dictionary where keys are the feature names, and the items are the start and end times of the kernel support 
        this can be a super-set of feature_set used
    neural_preprocessing_steps : list
        list of pre-processing transformations to perform on the neural activity
    check_for_nans : bool
        whether to check for NaNs in the data, and if detected remove them
    train_indices : list (optional)
    test_indices : list (optional)
    exclude_saccade_off_screen_times : bool
        whether for each neuron to exclude times when the saccade is off the screen of the receptive field of the neuron
    Returns
    -------
    X : numpy ndarray or list of numpy ndarrays
        feature matrix
        if exclude_saccade_off_screen_times = True, then this is a list where each elemeent is a separate X per neuron
        because each neuron will have different times that needs to be excluded
    Y : numpy ndarray
        target matrix
        if exclude_saccade_off_screen_times = True, then this is a list where each elemeent is a separate Y per neuron
        because each neuron will have different times that needs to be excluded
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

        # Saccade
        vis_exp_saccade_intervals = exp_data['_saccadeIntervalsVis'].astype(int)
        vis_exp_saccade_onset_times = vis_exp_times[vis_exp_saccade_intervals[:, 0]]
        vis_exp_neural_activity = exp_data['_tracesVis']

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

    if exp_type in ['gray', 'grating'] and ('z_score_separately' in neural_preprocessing_steps):
        print('Removing z-score separately from pre-processing step because using only one experiment type')
        # remove z-score separately if only using one experiment
        neural_preprocessing_steps.remove('z_score_separately')


    # Make target matrix Y

    Y = neural_activity
    if ('zscore' in neural_preprocessing_steps) and ('z_score_separately' not in neural_preprocessing_steps):
        Y  = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)
    elif ('zscore_w_baseline' in neural_preprocessing_steps) and ('z_score_separately' not in neural_preprocessing_steps):
        if exp_type == 'gray':
            # Gray screen experiment z-scoring using ITI and baseline activity
            gray_exp_iti_times = exp_times.copy()
            time_after_saccade_to_exclude = 2
            for saccade_time in exp_saccade_onset_times:
                gray_exp_iti_times[(gray_exp_iti_times >= saccade_time) & (
                        gray_exp_iti_times <= saccade_time + time_after_saccade_to_exclude)] = np.nan
            gray_exp_iti_times_idx = np.where(~np.isnan(gray_exp_iti_times))[0]
            gray_exp_iti_neural_activity_mu = np.mean(neural_activity[gray_exp_iti_times_idx, :], axis=0)
            gray_exp_iti_neural_activity_std = np.std(neural_activity[gray_exp_iti_times_idx, :], axis=0)
            Y = (neural_activity - gray_exp_iti_neural_activity_mu) / gray_exp_iti_neural_activity_std
        elif exp_type == 'grating':
            # Grating experiment z-scoring using ITI and baseline activity
            vis_iti_times = vis_exp_times.copy()
            time_after_vis_to_exclude = 3
            time_after_saccade_to_exclude = 2
            for onset_time in grating_onset_times:
                vis_iti_times[
                    (vis_exp_times >= onset_time) & (vis_exp_times <= onset_time + time_after_vis_to_exclude)] = np.nan
            for saccade_time in vis_exp_saccade_onset_times:
                vis_iti_times[(vis_exp_times >= saccade_time) & (
                        vis_exp_times <= saccade_time + time_after_saccade_to_exclude)] = np.nan

            vis_iti_times_idx = np.where(~np.isnan(vis_iti_times))[0]
            vis_exp_iti_neural_activity_mu = np.mean(vis_exp_neural_activity[vis_iti_times_idx, :], axis=0)
            vis_exp_iti_neural_activity_std = np.std(vis_exp_neural_activity[vis_iti_times_idx, :], axis=0)
            Y = (vis_exp_neural_activity - vis_exp_iti_neural_activity_mu) / vis_exp_iti_neural_activity_std


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

        elif feature_name == 'saccade_on_shuffled':
            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            subset_exp_saccade_onset_times = exp_saccade_onset_times[
                (exp_saccade_onset_times + feature_time_window[1] < exp_times[-1]) &
                (exp_saccade_onset_times + feature_time_window[0] > exp_times[0])
                ]

            # Shuffle saccade onset times
            saccade_allowed_time_points = [exp_times[0] - feature_time_window[0], exp_times[-1] - feature_time_window[1]]
            num_subset_exp_saccade_onset_times = len(subset_exp_saccade_onset_times)
            np.random.seed(333)
            subset_exp_saccade_onset_times_shuffled = np.sort(np.random.uniform(low=saccade_allowed_time_points[0],
                                                                  high=saccade_allowed_time_points[1],
                                                                  size=num_subset_exp_saccade_onset_times))

            for n_saccade_time, onset_time in enumerate(subset_exp_saccade_onset_times_shuffled):

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

            subset_saccade_onset_idx = np.where(
                (exp_saccade_onset_times + feature_time_window[1] < exp_times[-1]) &
                (exp_saccade_onset_times + feature_time_window[0] > exp_times[0])
            )[0]

            subset_exp_saccade_onset_times = exp_saccade_onset_times[subset_saccade_onset_idx]
            subset_saccade_dirs = saccade_dirs[subset_saccade_onset_idx]

            for n_saccade_time, onset_time in enumerate(subset_exp_saccade_onset_times):

                if onset_time + feature_time_window[1] > exp_times[-1]:
                    print('onset time exceeds experiment time, skipping')
                    continue

                start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                feature_mat[row_idx, col_idx] = subset_saccade_dirs[n_saccade_time]

        elif feature_name == 'saccade_dir_shuffled':

            # currently also requires saccade_on_shuffled, and uses the random times from that

            for n_saccade_time, onset_time in enumerate(subset_exp_saccade_onset_times_shuffled):

                if onset_time + feature_time_window[1] > exp_times[-1]:
                    print('onset time exceeds experiment time, skipping')
                    continue

                start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                feature_mat[row_idx, col_idx] = saccade_dirs[n_saccade_time]

        elif feature_name == 'saccade_nasal':

            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            subset_saccade_onset_idx = np.where(
                (exp_saccade_onset_times + feature_time_window[1] < exp_times[-1]) &
                (exp_saccade_onset_times + feature_time_window[0] > exp_times[0])
            )[0]

            subset_exp_saccade_onset_times = exp_saccade_onset_times[subset_saccade_onset_idx]
            subset_saccade_dirs = saccade_dirs[subset_saccade_onset_idx]

            saccade_nasal_trials = np.where(subset_saccade_dirs == -1)[0]
            saccade_nasal_onset_times = subset_exp_saccade_onset_times[saccade_nasal_trials]

            for n_saccade_time, onset_time in enumerate(saccade_nasal_onset_times):

                if onset_time + feature_time_window[1] > exp_times[-1]:
                    print('onset time exceeds experiment time, skipping')
                    continue

                start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                feature_mat[row_idx, col_idx] = 1

        elif feature_name == 'saccade_temporal':

            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            subset_saccade_onset_idx = np.where(
                (exp_saccade_onset_times + feature_time_window[1] < exp_times[-1]) &
                (exp_saccade_onset_times + feature_time_window[0] > exp_times[0])
            )[0]

            subset_exp_saccade_onset_times = exp_saccade_onset_times[subset_saccade_onset_idx]
            subset_saccade_dirs = saccade_dirs[subset_saccade_onset_idx]

            saccade_temporal_trials = np.where(subset_saccade_dirs == 1)[0]
            saccade_temporal_onset_times = subset_exp_saccade_onset_times[saccade_temporal_trials]

            for n_saccade_time, onset_time in enumerate(saccade_temporal_onset_times):

                if onset_time + feature_time_window[1] > exp_times[-1]:
                    print('onset time exceeds experiment time, skipping')
                    continue

                start_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                feature_mat[row_idx, col_idx] = 1

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

        elif feature_name == 'running':

            # perform linear interpolation to match the running speed time samples to the exp time samples
            running_speed_timestamps = exp_data['running_speed_timestamps'].flatten()
            big_time_gap = 10
            runnning_speed_cutoff_loc = np.where(np.diff(running_speed_timestamps) > big_time_gap)[0]
            running_speed_cutoff_timestamps = running_speed_timestamps[runnning_speed_cutoff_loc]

            first_cut_off_time = running_speed_timestamps[runnning_speed_cutoff_loc[0] + 1]

            grating_interval = exp_data['grating_interval']
            gray_interval = exp_data['gray_interval']

            if exp_type == 'both':
                gray_exp_global_time = gray_exp_times + first_cut_off_time

                vis_exp_running_speed_interpolated = np.interp(x=vis_exp_times, xp=exp_data['running_speed_timestamps'],
                                                               fp=exp_data['running_speed'])
                gray_exp_running_speed_interpolated = np.interp(x=gray_exp_global_time, xp=exp_data['running_speed_timestamps'],
                                                               fp=exp_data['running_speed'])

                running_speed_interpolated = np.concatenate([vis_exp_running_speed_interpolated,
                                                             gray_exp_running_speed_interpolated])

                gray_interval_duration = gray_interval[1] - gray_interval[0]
                gray_exp_time_duration = gray_exp_times[-1] - gray_exp_times[0]

                assert (np.abs(gray_interval_duration - gray_exp_time_duration) < 2)

            elif exp_type == 'grating':

                running_speed_interpolated = np.interp(x=exp_times, xp=exp_data['running_speed_timestamps'],
                                                               fp=exp_data['running_speed'])

            elif exp_type == 'gray':
                gray_exp_global_time = exp_times + first_cut_off_time
                running_speed_interpolated = np.interp(x=gray_exp_global_time,
                                                                xp=exp_data['running_speed_timestamps'],
                                                                fp=exp_data['running_speed'])
            
            feature_mat = running_speed_interpolated.reshape(-1, 1)

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
    
    
    # Do saccade off screen exclusion
    if exclude_saccade_off_screen_times and (exp_type != 'grating'):

        # Note this is only the gray screen saccades (???) 2023-04-02
        gray_exp_saccade_within_screen = exp_data['_inPerNeuron']  # Saccade trial x Neuron, 1 = within screen, 0 = outside screen

        if exp_type == 'both':
            vis_exp_saccade_within_screen = exp_data['_inPerNeuronVis']
            num_vis_saccade = np.shape(vis_exp_saccade_within_screen)[0]
            num_gray_saccade = np.shape(gray_exp_saccade_within_screen)[0]
            assert np.shape(vis_exp_saccade_within_screen)[0] == len(vis_exp_saccade_onset_times)
            assert len(exp_saccade_onset_times) == (num_vis_saccade + num_gray_saccade)

        if exp_type == 'gray':
            assert np.shape(gray_exp_saccade_within_screen)[0] == len(exp_saccade_onset_times)

        if exp_type == 'grating':
            vis_exp_saccade_within_screen = exp_data['_inPerNeuronVis']
            num_vis_saccade = np.shape(vis_exp_saccade_within_screen)[0]

        num_neurons = np.shape(exp_data['_inPerNeuron'])[1]
        # Loop through each neuron to exclude times where saccade was outside of the screen
        X_per_neuron = []
        Y_per_neuron = []
        subset_index_per_neuron = []
        for neuron_idx in np.arange(num_neurons):
            X_neuron = X.copy()
            Y_neuron = Y.copy()[:, neuron_idx]
            gray_exp_saccade_off_screen_trials = np.where(gray_exp_saccade_within_screen[:, neuron_idx] == 0)[0]

            for gray_exp_trial_idx in gray_exp_saccade_off_screen_trials:
                if exp_type == 'gray':

                    time_idx_to_exclude = np.where(
                    (exp_times >= exp_saccade_onset_times[gray_exp_trial_idx] + saccade_off_screen_exclusion_window[0]) &
                    (exp_times <= exp_saccade_onset_times[gray_exp_trial_idx] + saccade_off_screen_exclusion_window[1])
                                                   )[0]

                    Y_neuron[time_idx_to_exclude] = np.nan
                    X_neuron[time_idx_to_exclude, :] = np.nan

                elif (exp_type == 'both') and exclude_saccade_on_vis_exp:

                    # in this case exp_saccade_onset_times is the gray exp saccade with offset, so the same code can apply
                    time_idx_to_exclude = np.where(
                        (exp_times >= exp_saccade_onset_times[gray_exp_trial_idx] + saccade_off_screen_exclusion_window[0]) &
                        (exp_times <= exp_saccade_onset_times[gray_exp_trial_idx] + saccade_off_screen_exclusion_window[1])
                    )[0]

                    Y_neuron[time_idx_to_exclude] = np.nan
                    X_neuron[time_idx_to_exclude, :] = np.nan
                elif exp_type == 'both':
                    # in this case you need to offset the gray saccade trials (since exp_saccade_onset_times is vis, then gray concatenated)
                    time_idx_to_exclude = np.where(
                        (exp_times >= exp_saccade_onset_times[num_vis_saccade + gray_exp_trial_idx] + saccade_off_screen_exclusion_window[0]) &
                        (exp_times <= exp_saccade_onset_times[num_vis_saccade + gray_exp_trial_idx] + saccade_off_screen_exclusion_window[1])
                    )[0]

                    Y_neuron[time_idx_to_exclude] = np.nan
                    X_neuron[time_idx_to_exclude, :] = np.nan

            # grating experiment exclude saccade off screen : here we also exclude the vis exp saccades
            if exp_type == 'both':
                vis_exp_saccade_off_screen_trials = np.where(vis_exp_saccade_within_screen[:, neuron_idx] == 0)[0]
                for vis_exp_trial_idx in vis_exp_saccade_off_screen_trials:
                    time_idx_to_exclude = np.where(
                        (exp_times >= exp_saccade_onset_times[vis_exp_trial_idx] + saccade_off_screen_exclusion_window[0]) &
                        (exp_times <= exp_saccade_onset_times[vis_exp_trial_idx] + saccade_off_screen_exclusion_window[1])
                    )[0]

                    Y_neuron[time_idx_to_exclude] = np.nan
                    X_neuron[time_idx_to_exclude, :] = np.nan


            subset_index = np.where(~np.isnan(np.sum(X_neuron, axis=1)))[0]
            # subset_index = np.where(~np.isnan(X_neuron))[0]
            X_per_neuron.append(X_neuron[subset_index, :])
            Y_per_neuron.append(Y_neuron[subset_index])
            subset_index_per_neuron.append(subset_index)

        if return_trial_type:
            return X_per_neuron, Y_per_neuron, grating_orientation_per_trial, saccade_dirs, feature_indices, subset_index_per_neuron
        else:
            return X_per_neuron, Y_per_neuron, feature_indices, subset_index_per_neuron

    elif exp_type == 'grating' and exclude_saccade_off_screen_times:

        grating_exp_saccade_within_screen = exp_data['_inPerNeuronVis']  # Saccade trial x Neuron, 1 = within screen, 0 = outside screen
        assert np.shape(grating_exp_saccade_within_screen)[0] == len(vis_exp_saccade_onset_times)

        grating_exp_saccade_out_screen = np.sum(1 - grating_exp_saccade_within_screen, axis=1)
        grating_exp_saccade_off_screen_trials = np.where(grating_exp_saccade_out_screen >= 1)[0]
        for grating_exp_trial_idx in grating_exp_saccade_off_screen_trials:
            time_idx_to_exclude = np.where(
                (exp_times >= exp_saccade_onset_times[grating_exp_trial_idx] + saccade_off_screen_exclusion_window[
                    0]) &
                (exp_times <= exp_saccade_onset_times[grating_exp_trial_idx] + saccade_off_screen_exclusion_window[1])
            )[0]

            Y[time_idx_to_exclude, :] = np.nan
            X[time_idx_to_exclude, :] = np.nan

        subset_index = np.where(~np.isnan(np.sum(X, axis=1)))[0]
        Y = Y[subset_index, :]
        X = X[subset_index, :]


    if exclude_saccade_any_off_screen_times and (exp_type != 'grating'):

        gray_exp_saccade_within_screen = exp_data['_inPerNeuron']  # Saccade trial x Neuron, 1 = within screen, 0 = outside screen

        if exp_type == 'gray':
            assert np.shape(gray_exp_saccade_within_screen)[0] == len(exp_saccade_onset_times)

        gray_exp_saccade_out_screen = np.sum(1 - gray_exp_saccade_within_screen, axis=1)
        gray_exp_saccade_off_screen_trials = np.where(gray_exp_saccade_out_screen >= 1)[0]
        for gray_exp_trial_idx in gray_exp_saccade_off_screen_trials:
            if exp_type == 'gray':
                time_idx_to_exclude = np.where(
                    (exp_times >= exp_saccade_onset_times[gray_exp_trial_idx] + saccade_off_screen_exclusion_window[
                        0]) &
                    (exp_times <= exp_saccade_onset_times[gray_exp_trial_idx] + saccade_off_screen_exclusion_window[1])
                )[0]

                Y[time_idx_to_exclude, :] = np.nan
                X[time_idx_to_exclude, :] = np.nan

        subset_index = np.where(~np.isnan(np.sum(X, axis=1)))[0]
        Y = Y[subset_index, :]
        X = X[subset_index, :]


    if return_trial_type:
        return X, Y, grating_orientation_per_trial, saccade_dirs, feature_indices, subset_index
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
    """

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
                         performance_metrics=['explained_variance', 'r2'],
                         exclude_saccade_off_screen_times=False):
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
    exclude_saccade_off_screen_times : bool
        whther the provided X and Y have times where saccade off screen excluded
        in which case they are separate for each neuron and so cross-validation train test indices will be different...

    Returns
    -------
    regression_result : dict
        weights_per_cv_set : numpy ndarray of what shape???

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

    pdb.set_trace()

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
                                   exp_type='grating', exclude_saccade_on_vis_exp=False,
                                   exclude_saccade_off_screen_times=False,
                                   exclude_saccade_any_off_screen_times=False,
                                   saccade_off_screen_exclusion_window=[-0.5, 0.5],
                                   subset_index=None,
                                   neuron_idx=None):
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
    exclude_saccade_off_screen_times : bool 
        whether to exclude times for the neuron where the saccade is off the screen
    Returns
    -------
    regression_result : dict

    """

    # Get the fitted activity matrix for the entire time trace
    Y = regression_result['Y']
    Y_test_per_cv_set = regression_result['Y_test_per_cv_set']
    Y_test_hat_per_cv_set = np.array(regression_result['Y_test_hat_per_cv_set'])
    test_idx_per_cv_set = regression_result['test_idx_per_cv_set']

    vis_exp_times, vis_exp_saccade_onset_times, grating_onset_times, saccade_dirs, grating_orientation_per_trial = \
        get_vis_and_saccade_times(
        exp_data, exp_type=exp_type, exclude_saccade_on_vis_exp=exclude_saccade_on_vis_exp)

    sec_per_time_samples = np.mean(np.diff(vis_exp_times))  # do this before some subsetting below
    # sec_per_time_samples = np.median(np.diff(vis_exp_times))  # switch to mode because if some skipped times sometimes...

    if type(alignment_time_window) is dict:
        vis_alignment_time_window = alignment_time_window['vis']
        saccade_alignment_time_window = alignment_time_window['saccade']
    else:
        vis_alignment_time_window = alignment_time_window
        saccade_alignment_time_window = alignment_time_window

    # single neuron version / all neuron version can use the same code
    if exclude_saccade_off_screen_times or exclude_saccade_any_off_screen_times:
        if subset_index is not None:
            # vis_exp_excluded_times = np.delete(vis_exp_times, subset_index)
            # vis_exp_times = vis_exp_times[subset_index]
            time_idx_included_bool = np.zeros(len(vis_exp_times), )
            time_idx_included_bool[subset_index] = 1

        subset_saccade_trial_idx = []
        for trial_idx in np.arange(len(vis_exp_saccade_onset_times)):

            onset_time = vis_exp_saccade_onset_times[trial_idx]
            # -1 / +1 here just to be a bit more conservative
            start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + saccade_alignment_time_window[0]))) - 1
            end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + saccade_alignment_time_window[1]))) + 1

            if np.all(time_idx_included_bool[start_sample:end_sample]):
                subset_saccade_trial_idx.append(trial_idx)

        vis_exp_saccade_onset_times = vis_exp_saccade_onset_times[subset_saccade_trial_idx]
        saccade_dirs = saccade_dirs[subset_saccade_trial_idx]

        # do the same for grating onsets : ie. exclude grating onset times during saccade off screen
        if exp_type in ['grating', 'both'] and (not exclude_saccade_on_vis_exp):
            subset_grating_trial_idx = []
            for trial_idx in np.arange(len(grating_onset_times)):
                onset_time = grating_onset_times[trial_idx]
                # -1 / +1 here just to be a bit more conservative
                start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + vis_alignment_time_window[0]))) - 1
                end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + vis_alignment_time_window[1]))) + 1

                if np.all(time_idx_included_bool[start_sample:end_sample]):
                    subset_grating_trial_idx.append(trial_idx)

            grating_onset_times = grating_onset_times[subset_grating_trial_idx]
            grating_orientation_per_trial = np.array(grating_orientation_per_trial)[subset_grating_trial_idx]

        # subset experiment times
        vis_exp_times = vis_exp_times[subset_index]

    subset_vis_exp_vis_onset_times = grating_onset_times[
        (grating_onset_times + vis_alignment_time_window[1] < vis_exp_times[-1]) &
        (grating_onset_times + vis_alignment_time_window[0] > vis_exp_times[0])
        ]

    subset_vis_exp_saccade_onset_times = vis_exp_saccade_onset_times[
        (vis_exp_saccade_onset_times + saccade_alignment_time_window[1] < vis_exp_times[-1]) &
        (vis_exp_saccade_onset_times + saccade_alignment_time_window[0] > vis_exp_times[0])
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


    num_vis_aligned_time_points = int((vis_alignment_time_window[1] - vis_alignment_time_window[0]) / sec_per_time_samples)
    num_saccade_aligned_time_points = int((saccade_alignment_time_window[1] - saccade_alignment_time_window[0]) / sec_per_time_samples)


    if exp_type == 'gray':
        num_trials_to_get = num_saccades
    else:
        num_trials_to_get = np.min([num_saccades, num_grating_presentations])

    Y_hat_vis_aligned = np.zeros((num_trials_to_get, num_vis_aligned_time_points, num_neurons)) + np.nan
    Y_vis_aligned = np.zeros((num_trials_to_get, num_vis_aligned_time_points, num_neurons)) + np.nan
    Y_hat_saccade_aligned = np.zeros((num_trials_to_get, num_saccade_aligned_time_points, num_neurons)) + np.nan
    Y_saccade_aligned = np.zeros((num_trials_to_get, num_saccade_aligned_time_points, num_neurons)) + np.nan

    if exp_type == 'gray':
        subset_vis_on_times = []
        subset_saccade_on_times = subset_vis_exp_saccade_onset_times
    else:
        # TODO: make the subset grating id and saccade id as well
        subset_vis_trial_idx = np.random.choice(np.arange(0, len(subset_vis_exp_vis_onset_times)), num_trials_to_get)
        subset_vis_on_times = subset_vis_exp_vis_onset_times[subset_vis_trial_idx]
        subset_vis_ori = grating_orientation_per_trial[subset_vis_trial_idx]

        subset_saccade_trial_idx = np.random.choice(np.arange(0, len(subset_vis_exp_saccade_onset_times)), num_trials_to_get)
        subset_saccade_on_times = subset_vis_exp_saccade_onset_times[subset_saccade_trial_idx]
        subset_saccade_dir = saccade_dirs[subset_saccade_trial_idx]
        # subset_vis_on_times = np.random.choice(subset_vis_exp_vis_onset_times, num_trials_to_get)
        # subset_saccade_on_times = np.random.choice(subset_vis_exp_saccade_onset_times, num_trials_to_get)

    # align to visual stimulus
    for n_trial, vis_on_time in enumerate(subset_vis_on_times):

        time_idx = np.where(
            (vis_exp_times >= (vis_on_time + vis_alignment_time_window[0])) &
            (vis_exp_times <= (vis_on_time + vis_alignment_time_window[1]))
        )[0]

        time_idx = time_idx[0:num_vis_aligned_time_points]

        try:
            Y_hat_vis_aligned[n_trial, :, :] = Y_hat[time_idx, :]
        except:
            pdb.set_trace()

        Y_vis_aligned[n_trial, :, :] = Y[time_idx, :]

    # align to saccade onset
    for n_trial, saccade_on_time in enumerate(subset_saccade_on_times):
        time_idx = np.where(
            (vis_exp_times >= (saccade_on_time + saccade_alignment_time_window[0])) &
            (vis_exp_times <= (saccade_on_time + saccade_alignment_time_window[1]))
        )[0]

        time_idx = time_idx[0:num_saccade_aligned_time_points]

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

    if num_saccades > 0:

        if np.sum(np.isnan(Y_saccade_aligned_flattened)) > 0:
            pdb.set_trace()

        saccade_aligned_var_explained = sklmetrics.explained_variance_score(Y_saccade_aligned_flattened,
                                                                        Y_hat_saccade_aligned_flattened,
                                                                        multioutput='raw_values')
    else:
        print('No saccade found in this neuron / set of neurons, returning NaN')
        saccade_aligned_var_explained = np.repeat(np.nan, num_neurons)

    if 'r2' in performance_metrics:
        if np.all(np.isnan(Y_vis_aligned_flattened)):
            vis_aligned_r2 = np.repeat(np.nan, num_neurons)
        else:
            vis_aligned_r2 = sklmetrics.r2_score(Y_vis_aligned_flattened, Y_hat_vis_aligned_flattened, multioutput='raw_values')
        if num_saccades > 0:
            saccade_aligned_r2 = sklmetrics.r2_score(Y_saccade_aligned_flattened, Y_hat_saccade_aligned_flattened, multioutput='raw_values')
        else:
            saccade_aligned_r2 = np.repeat(np.nan, num_neurons)

    regression_result['Y_hat_vis_aligned'] = Y_hat_vis_aligned
    regression_result['Y_vis_aligned'] = Y_vis_aligned
    regression_result['Y_hat_saccade_aligned'] = Y_hat_saccade_aligned
    regression_result['Y_saccade_aligned'] = Y_saccade_aligned
    regression_result['vis_aligned_var_explained'] = vis_aligned_var_explained
    regression_result['saccade_aligned_var_explained'] = saccade_aligned_var_explained
    regression_result['subset_vis_ori'] = subset_vis_ori
    regression_result['subset_saccade_dir'] = subset_saccade_dir
    regression_result['vis_oris'] = grating_orientation_per_trial
    regression_result['saccade_dirs'] = saccade_dirs

    if 'r2' in performance_metrics:
        regression_result['vis_aligned_r2'] = vis_aligned_r2
        regression_result['saccade_aligned_r2'] = saccade_aligned_r2


    # Get the full Y_vis_aligned and Y_saccade_aligned for later plotting
    # Also get the corresponding Y_hat for these...
    Y_vis_aligned_full = np.zeros((len(subset_vis_exp_vis_onset_times), num_vis_aligned_time_points, num_neurons)) + np.nan
    Y_saccade_aligned_full = np.zeros((len(subset_vis_exp_saccade_onset_times), num_saccade_aligned_time_points, num_neurons)) + np.nan

    Y_hat_vis_aligned_full = np.zeros((len(subset_vis_exp_vis_onset_times), num_vis_aligned_time_points, num_neurons)) + np.nan
    Y_hat_saccade_aligned_full = np.zeros((len(subset_vis_exp_saccade_onset_times), num_saccade_aligned_time_points, num_neurons)) + np.nan

    # align to visual stimulus
    for n_trial, vis_on_time in enumerate(subset_vis_exp_vis_onset_times):

        time_idx = np.where(
            (vis_exp_times >= (vis_on_time + vis_alignment_time_window[0])) &
            (vis_exp_times <= (vis_on_time + vis_alignment_time_window[1]))
        )[0]

        time_idx = time_idx[0:num_vis_aligned_time_points]

        Y_vis_aligned_full[n_trial, :, :] = Y[time_idx, :]
        Y_hat_vis_aligned_full[n_trial, :, :] = Y_hat[time_idx, :]

    for n_trial, saccade_on_time in enumerate(subset_vis_exp_saccade_onset_times):
        time_idx = np.where(
            (vis_exp_times >= (saccade_on_time + saccade_alignment_time_window[0])) &
            (vis_exp_times <= (saccade_on_time + saccade_alignment_time_window[1]))
        )[0]
        time_idx = time_idx[0:num_saccade_aligned_time_points]
        Y_saccade_aligned_full[n_trial, :, :] = Y[time_idx, :]
        Y_hat_saccade_aligned_full[n_trial, :, :] = Y_hat[time_idx, :]

    regression_result['Y_vis_aligned_full'] = Y_vis_aligned_full
    regression_result['Y_saccade_aligned_full'] = Y_saccade_aligned_full
    regression_result['Y_hat_vis_aligned_full'] = Y_hat_vis_aligned_full
    regression_result['Y_hat_saccade_aligned_full'] = Y_hat_saccade_aligned_full

    return regression_result


def get_aligned_activity(exp_data, exp_type='grating', aligned_event='saccade', alignment_time_window=[-1, 3],
                         exclude_saccade_on_vis_exp=False, return_vis_ori_for_saccade=False):
    """
    Parameters
    ----------
    exp_data :
    exp_type : str
        type of experiment data to load

    Returns
    -------

    trial_type (saccade) : numpy ndarray
        -1 : nasal
        1 : temporal
    vis_aligned_activity : numpy ndarray
        matrix of shape (numTrial, numTimePoints, numNeurons)
    """

    assert exp_type in ['grating', 'gray', 'both']
    assert aligned_event in ['saccade', 'vis']

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

        subset_saccade_trial_idx = np.where(
            (exp_saccade_onset_times + alignment_time_window[0] > min_time) &
            (exp_saccade_onset_times + alignment_time_window[1] < max_time))[0]

        subset_exp_saccade_onset_times = exp_saccade_onset_times[subset_saccade_trial_idx]

        aligned_activity = np.zeros((len(subset_exp_saccade_onset_times), num_time_bins, num_neurons)) + np.nan


        # Get visual time (for getting visual identity)
        subset_grating_onset_trial_idx = np.where(
            (grating_onset_times + alignment_time_window[0] > min_time) &
            (grating_onset_times + alignment_time_window[1] < max_time)
        )[0]


        # Get visual orientation each trial
        grating_id_per_trial = np.array(exp_data['_gratingIds'].flatten()) - 1  # to 0 indexing
        grating_id_per_trial = grating_id_per_trial.astype(int)
        grating_id_to_dir = np.array(exp_data['_gratingIdDirections'].flatten())
        grating_id_per_trial_subset = grating_id_per_trial[subset_grating_onset_trial_idx]
        vis_ori_per_trial = np.array([grating_id_to_dir[x] for x in grating_id_per_trial_subset])

        subset_grating_onset_times = grating_onset_times[subset_grating_onset_trial_idx]
        vis_ori_during_saccade = np.zeros((len(subset_exp_saccade_onset_times), ))
        for trial_i, saccade_time in enumerate(subset_exp_saccade_onset_times):

            subset_idx = np.where(
                (exp_times >= (saccade_time + alignment_time_window[0])) &
                (exp_times <= (saccade_time + alignment_time_window[1]))
            )[0]

            aligned_activity[trial_i, :, :] = neural_activity[subset_idx, :]

            # Get the vis orientation presented during each saccade (set to NaN if no visuasl stimuli)
            vis_trial_during_saccade = np.where(
                (saccade_time >= subset_grating_onset_times) &  # saccade occured after grating onset
                (saccade_time <= subset_grating_onset_times + 3)  # saccade occured within 3 seconds after grating onset
            )[0]

            if len(vis_trial_during_saccade) == 0:
                vis_ori_during_saccade[trial_i] = -1
            else:
                vis_ori_during_saccade[trial_i] = vis_ori_per_trial[vis_trial_during_saccade[0]]

        if return_vis_ori_for_saccade:
            trial_type = saccade_dirs[subset_saccade_trial_idx]
            return aligned_activity, trial_type, time_windows, vis_ori_during_saccade
        else:
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

def get_saccade_kernel_from_regression_result(regression_result, model_X_set_to_plot,
                                              kernel_summary_metric='mean-post-saccade'):
    """
    regression_result : numpy item
    model_X_set_to_plot : str
    """

    # Get the kernels
    model_weights_per_X_set = regression_result['model_weights_per_X_set'].item()
    X_set_weights = model_weights_per_X_set[model_X_set_to_plot]  # numCV x numNeuron x numFeatures
    feature_name_and_indices = regression_result['feature_indices_per_X_set'].item()[model_X_set_to_plot]
    regression_kernel_windows = regression_result['regression_kernel_windows']
    saccade_on_kernel_idx = np.where(regression_result['regression_kernel_names'] == 'saccade_on')[0][0]
    saccade_on_window = regression_kernel_windows[saccade_on_kernel_idx]

    saccade_onset_kernel_indices = feature_name_and_indices['saccade_on']
    saccade_dir_kernel_indices = feature_name_and_indices['saccade_dir']

    saccade_on_kernels = np.mean(X_set_weights[:, :, :][:, :, saccade_onset_kernel_indices], axis=0)
    saccade_dir_kernels = np.mean(X_set_weights[:, :, :][:, :, saccade_dir_kernel_indices], axis=0)

    saccade_on_peri_event_window = np.linspace(saccade_on_window[0], saccade_on_window[1],
                                               len(saccade_onset_kernel_indices))
    post_saccade_idx = np.where(saccade_on_peri_event_window >= 0)[0]


    saccade_nasal_kernels = saccade_on_kernels - saccade_dir_kernels


    # get only post-saccade time window
    saccade_temporal_kernels = saccade_on_kernels + saccade_dir_kernels

    if kernel_summary_metric == 'mean-post-saccade':
        saccade_nasal_kernels_summary_val = np.mean(saccade_nasal_kernels[:, post_saccade_idx], axis=1)
        saccade_temporal_kernels_summary_val = np.mean(saccade_temporal_kernels[:, post_saccade_idx],
                                                             axis=1)
    elif kernel_summary_metric == 'peak':

        nasal_peak_index_per_neuron = np.argmax(np.abs(saccade_nasal_kernels), axis=1)
        saccade_nasal_kernels_summary_val = np.zeros((len(nasal_peak_index_per_neuron), ))

        for neuron_idx in np.arange(len(saccade_nasal_kernels_summary_val)):
            saccade_nasal_kernels_summary_val[neuron_idx] = saccade_nasal_kernels[neuron_idx, nasal_peak_index_per_neuron[neuron_idx]]

        temporal_peak_index_per_neuron = np.argmax(np.abs(saccade_temporal_kernels), axis=1)
        saccade_temporal_kernels_summary_val = np.zeros((len(temporal_peak_index_per_neuron),))

        for neuron_idx in np.arange(len(saccade_temporal_kernels_summary_val)):
            saccade_temporal_kernels_summary_val[neuron_idx] = saccade_temporal_kernels[
                neuron_idx, temporal_peak_index_per_neuron[neuron_idx]]


    return saccade_nasal_kernels_summary_val, saccade_temporal_kernels_summary_val

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
    """

    """

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



def get_vis_and_vis_plus_saccade_activity(vis_aligned_activity, vis_aligned_activity_saccade_on,
                                          saccade_on_trials,
                                          vis_ori_saccade_on, vis_ori_saccade_off,
                                          neuron_idx=0, vis_vs_vis_plus_saccade_window_width=[0, 0.5]):
    """
    Parameters
    ----------
    vis_aligned_activity : numpy ndarray
        matrix of shape (numTrial, numTimePoints, numNeurons)
    neuron_idx : int
    vis_vs_vis_plus_saccade_window_width : list

    """

    neuron_vis_aligned_activity_saccade_off = vis_aligned_activity[:, :, neuron_idx]

    all_trial_vis_only_activity = []
    all_trials_vis_plus_saccade_activity = []
    saccade_dir_during_vis = []

    for n_trial, saccade_on_trial_idx in enumerate(saccade_on_trials):
        trial_vis_ori = vis_ori_saccade_on[n_trial]

        if np.isnan(trial_vis_ori):
            continue

        saccade_off_ori_trials = np.where(vis_ori_saccade_off == trial_vis_ori)[
            0]  # get trials in saccade off condition with same orientation as this trial
        neuron_vis_aligned_activity_saccade_on = vis_aligned_activity_saccade_on[n_trial, :, neuron_idx]

        saccade_dir_during_vis_in_this_trial = saccade_dir_saccade_on_trials[n_trial]

        for n_trial_saccade, trial_saccade_times in enumerate(saccade_time_saccade_on_trials[n_trial]):

            if trial_saccade_times > 0:
                time_window_idx = np.where(
                    (time_windows >= trial_saccade_times + vis_vs_vis_plus_saccade_window_width[0]) &
                    (time_windows <= trial_saccade_times + vis_vs_vis_plus_saccade_window_width[1])
                )[0]

                neuron_vis_only_activity = np.mean(
                    neuron_vis_aligned_activity_saccade_off[saccade_off_ori_trials, :][:, time_window_idx])
                neuron_vis_plus_saccade_activity = np.mean(neuron_vis_aligned_activity_saccade_on[time_window_idx])

                # saccade_dir_within_trial = saccade_dir_during_vis_in_this_trial[n_trial_saccade]

                all_trial_vis_only_activity.append(neuron_vis_only_activity)
                all_trials_vis_plus_saccade_activity.append(neuron_vis_plus_saccade_activity)


    return neuron_vis_only_activity, neuron_vis_plus_saccade_activity


def get_ori_grouped_vis_and_saccade_activity(vis_aligned_activity, saccade_on_trials, saccade_off_trials,
                                             vis_ori_saccade_on, vis_ori_saccade_off, saccade_dir_saccade_on_trials,
                                             saccade_time_saccade_on_trials, time_windows,
                                             window_width=[0, 0.5], vis_tuning_window=[0, 1]):
    """
    Parameters
    -----------
    vis_aligned_activity :
    saccade_on_trials : numpy ndarray
    saccade_off_trials : numpy ndarray
    saccade_dir_saccade_on_trials : numpy ndarray
        -1 : nasal
        1 : temporal
    window_width : list
        list with 2 elements, denoting the start time and end time to take the mean activity

    Returns
    --------

    time_windows : list
        time points of the activity aligned to stimulus onset

    saccade_off_ori_activity : numpy ndarray

    nasal_saccade_ori_activity : numpy ndarray
        array with dimensions (numTrial, numOri, numNeurons)
        this contains the activity (observed vis + saccade activity subtracted by mean vis activity of that particular ori and time)
        of all neurons (numNeurons) for individual trials (1, 2, ... numTrial) for each orientation (numOri)
        NaNs represent trials that are missing, since different orientation have different trial counts,
        therefore use np.nanmean and np.nanstd to do summaries
    """

    vis_aligned_activity_saccade_on = vis_aligned_activity[saccade_on_trials, :, :]
    vis_aligned_activity_saccade_off = vis_aligned_activity[saccade_off_trials, :, :]
    num_neuron = np.shape(vis_aligned_activity)[-1]

    unique_ori, unique_ori_counts = np.unique(vis_ori_saccade_on, return_counts=True)  # why saccade on? Can be saccade off as well?
    unique_ori = unique_ori[~np.isnan(unique_ori)]

    all_unique_ori = np.unique(np.concatenate([vis_ori_saccade_on, vis_ori_saccade_off]))
    all_unique_ori = all_unique_ori[~np.isnan(all_unique_ori)]

    _, max_unique_ori_counts = sstats.mode(vis_ori_saccade_off)

    saccade_off_ori_activity = np.zeros((max_unique_ori_counts[0], len(all_unique_ori), num_neuron)) + np.nan

    all_vis_subtracted_activity = []
    saccade_dir_during_vis = []
    ori_during_saccade = []

    for n_trial, saccade_on_trial_idx in enumerate(saccade_on_trials):
        trial_vis_ori = vis_ori_saccade_on[n_trial]

        if np.isnan(trial_vis_ori):
            continue

        saccade_off_ori_trials = np.where(vis_ori_saccade_off == trial_vis_ori)[
            0]  # get trials in saccade off condition with same orientation as this trial

        trial_vis_aligned_activity_saccade_on = vis_aligned_activity_saccade_on[n_trial, :, :]

        saccade_dir_during_vis_in_this_trial = saccade_dir_saccade_on_trials[n_trial]

        for n_trial_saccade, trial_saccade_times in enumerate(saccade_time_saccade_on_trials[n_trial]):

            if trial_saccade_times > 0:
                time_window_idx = np.where(
                    (time_windows >= trial_saccade_times + window_width[0]) &
                    (time_windows <= trial_saccade_times + window_width[1])
                )[0]

                vis_only_activity = np.mean(vis_aligned_activity_saccade_off[saccade_off_ori_trials, :, :][:, time_window_idx, :], axis=(0, 1))
                trial_vis_plus_saccade_activity = np.mean(trial_vis_aligned_activity_saccade_on[time_window_idx, :], axis=0)

                vis_subtracted_activity = trial_vis_plus_saccade_activity - vis_only_activity

                # Append things
                all_vis_subtracted_activity.append(vis_subtracted_activity)
                ori_during_saccade.append(trial_vis_ori)
                saccade_dir_during_vis.append(saccade_dir_during_vis_in_this_trial[n_trial_saccade])

    # Unpack the individual trial vis subtracted saccade activity
    ori_during_saccade = np.array(ori_during_saccade)
    saccade_dir_during_vis = np.array(saccade_dir_during_vis)
    all_vis_subtracted_activity = np.array(all_vis_subtracted_activity)

    ori_during_nasal_saccade = ori_during_saccade[saccade_dir_during_vis == -1]
    ori_during_temporal_saccade = ori_during_saccade[saccade_dir_during_vis == 1]

    nasal_ori_mode, nasal_ori_mode_count = sstats.mode(ori_during_nasal_saccade)
    temporal_ori_mode, temporal_ori_mode_count = sstats.mode(ori_during_temporal_saccade)

    nasal_saccade_ori_activity = np.zeros((nasal_ori_mode_count[0],len(all_unique_ori), num_neuron)) + np.nan
    temporal_saccade_ori_activity = np.zeros((temporal_ori_mode_count[0], len(all_unique_ori), num_neuron)) + np.nan

    vis_time_window_idx = np.where((time_windows >= vis_tuning_window[0]) &
                                   (time_windows <= vis_tuning_window[1])
                                   )[0]

    for ori_idx, ori in enumerate(all_unique_ori):

        nasal_idx = np.where((ori_during_saccade == ori) & (saccade_dir_during_vis == -1))[0]
        nasal_saccade_ori_activity[0:len(nasal_idx), ori_idx, :] = all_vis_subtracted_activity[nasal_idx, :]

        temporal_idx = np.where((ori_during_saccade == ori) & (saccade_dir_during_vis == 1))[0]
        temporal_saccade_ori_activity[0:len(temporal_idx), ori_idx, :] = all_vis_subtracted_activity[temporal_idx, :]

        no_saccade_idx = np.where(vis_ori_saccade_off == ori)[0]
        saccade_off_ori_activity[0:len(no_saccade_idx), ori_idx, :] = np.mean(vis_aligned_activity_saccade_off[no_saccade_idx, :, :][:, vis_time_window_idx, :], axis=1)

    return saccade_off_ori_activity, nasal_saccade_ori_activity, temporal_saccade_ori_activity, unique_ori, all_unique_ori


def transform_ori_data_for_vonmises(vis_response,
                                    ori=np.array([0., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330.]),
                                    vis_response_error=None, scale_data=False):
    """

    """

    vis_response_padded = np.concatenate([vis_response, np.array([vis_response[0]])])
    ori_padded = np.concatenate([ori, np.array([360])])

    if scale_data:
        ydata = (vis_response_padded - np.min(vis_response_padded)) / np.max(vis_response_padded)
    else:
        ydata = vis_response_padded

    xdata = np.linspace(-np.pi, np.pi, len(vis_response_padded))

    if vis_response_error is None:
        return xdata, ydata
    else:
        vis_response_error_padded = np.concatenate([vis_response_error, np.array([vis_response_error[0]])])
        if scale_data:
            ydata_error = (vis_response_error_padded - np.min(vis_response_padded)) / np.max(vis_response_padded)
        else:
            ydata_error = vis_response_error_padded

        return xdata, ydata, ydata_error

def transform_ori_data_for_vonmises2(vis_response,
                                    ori=np.array([0., 30., 60., 90., 120., 150., 180., 210., 240., 270., 300., 330.]),
                                    vis_response_error=None, scale_data=False):
    """
    Parameters
    ----------
    """


    # remove NaNs (orientation with no trials)
    not_nan_idx = np.where(~np.isnan(vis_response))

    vis_response_padded = np.concatenate([vis_response, np.array([vis_response[0]])])
    ori_padded = np.concatenate([ori, np.array([360])])

    if scale_data:
        ydata = (vis_response_padded - np.min(vis_response_padded)) / np.max(vis_response_padded)
    else:
        ydata = vis_response_padded

    xdata = np.linspace(0, 360, len(vis_response_padded))

    if vis_response_error is None:
        return xdata[not_nan_idx], ydata[not_nan_idx]
    else:
        vis_response_error_padded = np.concatenate([vis_response_error, np.array([vis_response_error[0]])])
        if scale_data:
            ydata_error = (vis_response_error_padded - np.min(vis_response_padded)) / np.max(vis_response_padded)
        else:
            ydata_error = vis_response_error_padded

        return xdata[not_nan_idx], ydata[not_nan_idx], ydata_error[not_nan_idx]

##### SOME CIRCULAR STATISTICS FUNCTIONS ########

def circ_mean(angles, deg=True):
    '''Circular mean of angle data(default to degree)
    '''
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    mean = cmath.phase(angles_complex.sum()) % (2 * np.pi)
    return round(np.rad2deg(mean) if deg else mean, 7)

def circ_var(angles, deg=True):
    '''Circular variance of angle data(default to degree)
    0 <= var <= 1
    '''
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    r =abs(angles_complex.sum()) / len(angles)
    return round(1 - r, 7)

def circ_std(angles, deg=True):
    '''Circular standard deviation of angle data(default to degree)
    0 <= std
    '''
    a = np.deg2rad(angles) if deg else np.array(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    r = abs(angles_complex.sum()) / len(angles)
    std = np.sqrt(-2 * np.log(r))
    return round(np.rad2deg(std) if deg else std, 7)

def circ_corrcoef(x, y, deg=True, test=False):
    '''Circular correlation coefficient of two angle data(default to degree)
    Set `test=True` to perform a significance test.
    '''
    convert = np.pi / 180.0 if deg else 1
    sx = np.frompyfunc(np.sin, 1, 1)((x - circ_mean(x, deg)) * convert)
    sy = np.frompyfunc(np.sin, 1, 1)((y - circ_mean(y, deg)) * convert)
    r = (sx * sy).sum() / np.sqrt((sx ** 2).sum() * (sy ** 2).sum())

    if test:
        l20, l02, l22 = (sx ** 2).sum(),(sy ** 2).sum(), ((sx ** 2) * (sy ** 2)).sum()
        test_stat = r * np.sqrt(l20 * l02 / l22)
        p_value = 2 * (1 - sstats.norm.cdf(abs(test_stat)))
        return tuple(round(v, 7) for v in (r, test_stat, p_value))

    return round(r, 7)

@ray.remote
def fit_vis_ori_response(vis_response_matrix, initial_guess=[180, 1, 1, 1, 20]):

    # do leave-one-out
    scale_data = False
    # initial_guess = [180, 1, 1, 1, 6]  # TODO: guess deg from max response, and
    # amplitude from max - min response amplitude,
    # second peak (max - min) / 2
    # baseline guess = minimal response
    # kappa lower bound = 10
    # kappa guess : RP * (2 - rp^2) / (1 - rp^2)
    # Peak Bounds: to 5 times
    param_bounds = ((0, 0, 0, -np.inf, 5),  # the '2' here is the spread, roughly about 30 degrees here
                     (360, np.inf, np.inf, np.inf, 100))

    # param_bounds = ((0, 360), (0, np.inf), (0, np.inf), (-np.inf, np.inf), (10, 1000))
    # param_bounds = ((0, 360), (0, 100), (0, 100), (-100, 100), (10, 1000))

    #  param_bounds = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), (10, 1000))

    num_valid_trials = np.sum(~np.isnan(vis_response_matrix))
    loo_errors = np.zeros((num_valid_trials,)) + np.nan
    loo_predictions = np.zeros(np.shape(vis_response_matrix)) + np.nan
    x_start = 0
    x_end = 330
    n_valid_trial = 0

    for ori_idx in np.arange(np.shape(vis_response_matrix)[1]):
        for trial_idx in np.arange(np.shape(vis_response_matrix)[0]):
            if ~np.isnan(vis_response_matrix[trial_idx, ori_idx]):
                loo_vis_response_matrix = vis_response_matrix.copy()
                loo_vis_response_matrix[trial_idx, ori_idx] = np.nan  # set to NaN to exclude when calculating mean
                loo_vis_response_mean = np.nanmean(loo_vis_response_matrix, axis=0)  # mean across trials
                xdata, ydata = transform_ori_data_for_vonmises2(loo_vis_response_mean,
                                                                vis_response_error=None,
                                                                scale_data=scale_data)

                fitted_params, _ = spopt.curve_fit(von_mises2, xdata, ydata, p0=initial_guess,
                                                        bounds=param_bounds,
                                                        maxfev=100000)  #50000

                # 'dogbox' method leads to maximum number of evaluations excdeed for 100,000

                # Temp hack to reinforce lower bound...
                # if fitted_params[4] < 10:
                #     fitted_params[4] = 10

                # fitted_kappas[fitted_kappas <= 10] = 10
                # fitted_params[:, 4] = fitted_kappas

                # fitted_kappas[fitted_kappas ]

                # L-BFGS-B : does not work
                # SLSQP : also does not work
                # fitted_params_result = spopt.minimize(von_mises2_loss, x0=initial_guess, args=(xdata, ydata),
                #                                bounds=param_bounds, method='L-BFGS-B')

                # fitted_params_result = spopt.minimize(von_mises2_loss, x0=initial_guess, args=(xdata, ydata),
                #                                       bounds=param_bounds,
                 #                                      jac=gradient_respecting_bounds(param_bounds, von_mises2_loss))

                # fitted_params = fitted_params_result['x']

                # except:
                #     fitted_params = initial_guess

                xdata_interpolated = np.arange(x_start, x_end + 1, 30)
                # ydata_predicted = von_mises2(xdata, fitted_params[0], fitted_params[1])
                # ydata_interpolated = von_mises2(xdata_interpolated, fitted_params[0], fitted_params[1])
                ydata_predicted = von_mises2(xdata_interpolated, fitted_params[0], fitted_params[1],
                                             fitted_params[2],
                                             fitted_params[3], fitted_params[4])

                loo_value = vis_response_matrix[trial_idx, ori_idx]
                loo_ori_prediction = ydata_predicted[ori_idx]
                loo_squared_diff = (loo_value - loo_ori_prediction) ** 2
                loo_errors[n_valid_trial] = loo_squared_diff
                n_valid_trial += 1

                loo_predictions[trial_idx, ori_idx] = loo_ori_prediction

    loo_mse = np.mean(loo_errors)

    # Variance explained after averaging over trials with some orientation
    loo_predictions_mean = np.nanmean(loo_predictions, axis=0)
    vis_response_matrix_mean = np.nanmean(vis_response_matrix, axis=0)

    # remove missing orientations
    loo_predictions_mean = loo_predictions_mean[~np.isnan(loo_predictions_mean)]
    vis_response_matrix_mean = vis_response_matrix_mean[~np.isnan(vis_response_matrix_mean)]

    loo_variance_explained = sklmetrics.explained_variance_score(vis_response_matrix_mean, loo_predictions_mean)

    vis_response_matrix_flatten = vis_response_matrix.flatten()
    vis_response_matrix_flatten = vis_response_matrix_flatten[~np.isnan(vis_response_matrix_flatten)]
    loo_predictions_flatten = loo_predictions.flatten()
    loo_predictions_flatten = loo_predictions_flatten[~np.isnan(loo_predictions_flatten)]

    trial_by_trial_variance_explained = sklmetrics.explained_variance_score(vis_response_matrix_flatten, loo_predictions_flatten)

    return loo_mse, loo_predictions_mean, vis_response_matrix_mean, loo_variance_explained, trial_by_trial_variance_explained


def plot_ori_grouped_raster(vis_only_response, nasal_response, temporal_response, vis_ori_sort_idx,
                            vis_only_neuron_pref_ori=None, neuron_pref_ori_during_nasal_saccade=None,
                            neuron_pref_ori_during_temporal_saccade=None, zscore_cmap='bwr', general_cmap='viridis',
                            interpolation='none',
                            zscore_ea_neuron_separately=True, divide_by_max=False, scale_to_unit_range=False,
                            include_histogram=True, gray_nans=True, include_cbar=True,
                            fig=None, axs=None):
    """
    interpolation : str
        imshow interpolation, use 'none' to turn it off
    zscore_cmap : str
        options are: 'bwr', 'viridis'
    """
    
    if zscore_ea_neuron_separately:

        vis_only_response = sstats.zscore(vis_only_response, axis=1, nan_policy='omit')
        nasal_response = sstats.zscore(nasal_response, axis=1, nan_policy='omit')
        temporal_response = sstats.zscore(temporal_response, axis=1, nan_policy='omit')
        if zscore_cmap == 'bwr':
            cmap = mpl.cm.bwr
        elif zscore_cmap == 'viridis':
            cmap = mpl.cm.viridis
    else:
        if general_cmap == 'bwr':
            cmap = mpl.cm.bwr
        elif general_cmap == 'viridis':
            cmap = mpl.cm.viridis

    if divide_by_max:
        vis_only_response = vis_only_response / np.nanmax(np.abs(vis_only_response), axis=1).reshape(-1, 1)
        nasal_response = nasal_response / np.nanmax(np.abs(nasal_response), axis=1).reshape(-1, 1)
        temporal_response = temporal_response / np.nanmax(np.abs(temporal_response), axis=1).reshape(-1, 1)

    if scale_to_unit_range:
        vis_only_response = (vis_only_response - np.nanmin(vis_only_response, axis=1).reshape(-1, 1)) / (
                np.nanmax(vis_only_response, axis=1) - np.nanmin(vis_only_response, axis=1)).reshape(-1, 1)
        nasal_response = (nasal_response - np.nanmin(nasal_response, axis=1).reshape(-1, 1)) / (
                    np.nanmax(nasal_response, axis=1) - np.nanmin(nasal_response, axis=1)).reshape(-1, 1)
        temporal_response = (temporal_response - np.nanmin(temporal_response, axis=1).reshape(-1, 1)) / (
                    np.nanmax(temporal_response, axis=1) - np.nanmin(temporal_response, axis=1)).reshape(-1, 1)

    if gray_nans:
        cmap.set_bad(color='gray')

    title_txt_size = 11
    extent = [0, 360, 1, len(vis_ori_sort_idx)]
    histogram_bins = np.arange(-30, 361, 30)

    if include_histogram:

        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, axs = plt.subplots(2, 3, sharex=True, sharey=False,
                                    gridspec_kw={'height_ratios': [1, 2]})
            fig.set_size_inches(9, 3)

            # Histograms
            axs[0, 0].hist(vis_only_neuron_pref_ori, bins=histogram_bins, lw=1, color='black')
            axs[0, 1].hist(neuron_pref_ori_during_nasal_saccade, bins=histogram_bins, lw=1, color='black')
            axs[0, 2].hist(neuron_pref_ori_during_temporal_saccade, bins=histogram_bins, lw=1,
                           color='black')

            axs[0, 0].get_shared_y_axes().join(axs[0, 0], axs[0, 1])
            axs[0, 0].get_shared_y_axes().join(axs[0, 0], axs[0, 2])

            # Raster sorted by vis ori preference
            aspect = 'auto'
            axs[1, 0].imshow(vis_only_response, extent=extent, aspect=aspect, cmap=cmap, interpolation=interpolation)
            axs[1, 1].imshow(nasal_response, extent=extent, aspect=aspect, cmap=cmap, interpolation=interpolation)
            im = axs[1, 2].imshow(temporal_response, extent=extent, aspect=aspect, cmap=cmap, interpolation=interpolation)

            axs[1, 0].get_shared_y_axes().join(axs[1, 0], axs[1, 1])
            axs[1, 0].get_shared_y_axes().join(axs[1, 0], axs[1, 2])

            axs[1, 0].set_yticks([1, int(len(vis_ori_sort_idx) / 2), len(vis_ori_sort_idx)])
            axs[1, 1].set_yticks([])
            axs[1, 2].set_yticks([])
            axs[1, 0].set_xticks([0, 180, 360])

            axs[0, 0].set_title('Vis only', size=title_txt_size)
            axs[0, 1].set_title('Nasal saccade', size=title_txt_size)
            axs[0, 2].set_title('Temporal saccade', size=title_txt_size)

            if include_cbar:
                cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.48])
                cbar_obj = fig.colorbar(im, cax=cbar_ax)
                cbar_ax.set_ylabel('Scaled response', size=11)


            axs[1, 0].set_ylabel('Cells', size=11)
            fig.text(0.5, -0.04, 'Orientation (deg)', ha='center', size=11)

    else:

        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
            fig.set_size_inches(9, 3)
            if gray_nans:

                current_cmap = matplotlib.cm.get_cmap()
                current_cmap.set_bad(color='gray')
            axs[0].imshow(vis_only_response, extent=extent, aspect='auto', cmap=cmap)
            axs[1].imshow(nasal_response, extent=extent, aspect='auto', cmap=cmap)
            axs[2].imshow(temporal_response, extent=extent, aspect='auto', cmap=cmap)

            axs[0].set_yticks([1, int(len(vis_ori_sort_idx) / 2), len(vis_ori_sort_idx)])
            axs[0].set_xticks([0, 180, 360])

            axs[0].set_title('Vis only', size=title_txt_size)
            axs[1].set_title('Nasal saccade', size=title_txt_size)
            axs[2].set_title('Temporal saccade', size=title_txt_size)



            axs[0].set_ylabel('Cells', size=11)
            fig.text(0.5, -0.04, 'Orientation (deg)', ha='center', size=11)
    
    
    return fig, axs

def get_Anya_raster_sort_idx(temporal_response_per_t_neuron, nasal_response_per_t_neuron,
                             excited_threshold=0.5, inhibited_threshold=-0.5):

    """
    Get indices for sorting neurons based on the same way Anya did it in Figure 2
    Parameters
    ----------
    temporal_response_per_t_neuron : numpy ndarray
    nasal_response_per_t_neuron : numpy ndarray
    excited_threshold : float
    inhibited_threshold : float

    Returns
    -------
    sort_idx : numpy ndarray
    """

    temporal_response_per_neuron = np.mean(temporal_response_per_t_neuron, axis=1)
    nasal_response_per_neuron = np.mean(nasal_response_per_t_neuron, axis=1)

    temporal_e_nasal_null_idx = np.where(
        (temporal_response_per_neuron >= excited_threshold) &
        (np.abs(nasal_response_per_neuron) < excited_threshold)
    )[0]

    # sort the temporal e nasal null idx, by location of maximum response
    temporal_e_nasal_null_sort_idx = np.argsort(np.argmax(temporal_response_per_t_neuron[temporal_e_nasal_null_idx], axis=1))
    temporal_e_nasal_null_idx = temporal_e_nasal_null_idx[temporal_e_nasal_null_sort_idx]

    temporal_i_nasal_null_idx = np.where(
        (temporal_response_per_neuron <= inhibited_threshold) &
        (np.abs(nasal_response_per_neuron) < excited_threshold)
    )[0]

    # sort the temporal i nasal null idx, by location of maximum response
    temporal_i_nasal_null_sort_idx = np.argsort(
        np.argmin(temporal_response_per_t_neuron[temporal_i_nasal_null_idx], axis=1))
    temporal_i_nasal_null_idx = temporal_i_nasal_null_idx[temporal_i_nasal_null_sort_idx]

    temporal_null_nasal_e_idx = np.where(
        (temporal_response_per_neuron < excited_threshold) &
        (nasal_response_per_neuron >= excited_threshold)
    )[0]

    # sort the temporal null nasal e idx
    temporal_null_nasal_e_sort_idx = np.argsort(
        np.argmax(nasal_response_per_t_neuron[temporal_null_nasal_e_idx], axis=1))
    temporal_null_nasal_e_idx = temporal_null_nasal_e_idx[temporal_null_nasal_e_sort_idx]

    # TEMPORAL E NASAL E
    temporal_e_nasal_e_idx = np.where(
        (temporal_response_per_neuron > excited_threshold) &
        (nasal_response_per_neuron > excited_threshold)
    )[0]

    temporal_e_nasal_e_idx_sort_idx = np.argsort(
        np.argmax(nasal_response_per_t_neuron[temporal_e_nasal_e_idx], axis=1))
    temporal_e_nasal_e_idx = temporal_e_nasal_e_idx[temporal_e_nasal_e_idx_sort_idx]

    # TEMPORAL E NASAL I

    temporal_e_nasal_i_idx = np.where(
        (temporal_response_per_neuron > excited_threshold) &
        (nasal_response_per_neuron <= inhibited_threshold)
    )[0]

    temporal_e_nasal_i_sort_idx = np.argsort(
        np.argmin(nasal_response_per_t_neuron[temporal_e_nasal_i_idx], axis=1))
    temporal_e_nasal_i_idx = temporal_e_nasal_i_idx[temporal_e_nasal_i_sort_idx]

    # TEMPORAL I NASAL I

    temporal_i_nasal_i_idx = np.where(
        (temporal_response_per_neuron <= inhibited_threshold) &
        (nasal_response_per_neuron <= inhibited_threshold)
    )[0]

    temporal_i_nasal_i_sort_idx = np.argsort(
        np.argmin(nasal_response_per_t_neuron[temporal_i_nasal_i_idx], axis=1))
    temporal_i_nasal_i_idx = temporal_i_nasal_i_idx[temporal_i_nasal_i_sort_idx]

    # TEMPORAL NULL NASAL I
    temporal_null_nasal_i_idx = np.where(
        (np.abs(temporal_response_per_neuron) < excited_threshold) &
        (nasal_response_per_neuron <= inhibited_threshold)
    )[0]

    temporal_null_nasal_i_idx_sort_idx = np.argsort(
        np.argmin(nasal_response_per_t_neuron[temporal_null_nasal_i_idx], axis=1))
    temporal_null_nasal_i_idx = temporal_null_nasal_i_idx[temporal_null_nasal_i_idx_sort_idx]

    temporal_null_nasal_null_idx = np.where(
        (np.abs(temporal_response_per_neuron) < excited_threshold) &
        (np.abs(nasal_response_per_neuron) < excited_threshold)
    )[0]

    sort_idx = np.concatenate([
        temporal_e_nasal_null_idx,
        temporal_i_nasal_null_idx,
        temporal_e_nasal_i_idx,
        temporal_i_nasal_i_idx,
        temporal_e_nasal_e_idx,
        temporal_null_nasal_e_idx,
        temporal_null_nasal_i_idx,
        temporal_null_nasal_null_idx,
    ])

    return sort_idx

def plot_von_mises_fitted_loc(von_mises_fitted_loc, subset_indices=None, dot_size=20, xlim=[0, 360],
                              ylim=[0, 360], tick_vals=[0, 180, 360],
                              fig=None, axs=None):
    """
    Parameters
    ----------
    von_mises_fitted_loc : numpy ndarray
        array of shape (numNeuron, 3)
        second dimension are the different conditions
        first column : visual only response (no saccade)
        second column : nasal saccade response with stimulus subtracted
        third column : temporal saccade repsonse with stimulus subtracted
    fig : matplotlib figure object
    axs : matplotlib axes object

    """


    with plt.style.context(splstyle.get_style('nature-reviews')):

        if (fig is None) and (axs is None):
            fig, axs = plt.subplots(1, 3)
            fig.set_size_inches(10, 3)
            plt.subplots_adjust(wspace=0.35)

        # No saccade vs. Nasal saccade
        if subset_indices is not None:
            subset_idx = subset_indices[0]
        else:
            subset_idx = np.arange(0, np.shape(von_mises_fitted_loc)[0])

        axs[0].scatter(von_mises_fitted_loc[subset_idx, 0], von_mises_fitted_loc[subset_idx, 1], color='black', lw=0,
                       s=dot_size, clip_on=False)
        axs[0].set_xlabel('No saccade', size=11)
        axs[0].set_ylabel('Nasal saccade', size=11)

        if np.sum(subset_idx) > 2:
            r_val, test_stat, p_value = circ_corrcoef(von_mises_fitted_loc[subset_idx, 0], von_mises_fitted_loc[subset_idx, 1], test=True)
            axs[0].set_title(r'$r = %.2f, p = %.2f$' % (r_val, p_value), size=11)

        # No saccade vs. Temporal saccade
        if subset_indices is not None:
            subset_idx = subset_indices[1]
        else:
            subset_idx = np.arange(0, np.shape(von_mises_fitted_loc)[0])

        axs[1].scatter(von_mises_fitted_loc[subset_idx, 0], von_mises_fitted_loc[subset_idx, 2], color='black', lw=0,
                       s=dot_size, clip_on=False)
        axs[1].set_xlabel('No saccade', size=11)
        axs[1].set_ylabel('Temporal saccade', size=11)

        if np.sum(subset_idx) > 2:
            r_val, test_stat, p_value = circ_corrcoef(von_mises_fitted_loc[subset_idx, 0], von_mises_fitted_loc[subset_idx, 2],
                                                      test=True)
            axs[1].set_title(r'$r = %.2f, p = %.2f$' % (r_val, p_value), size=11)


        # Nasal sasccade vs. Temporal saccade
        if subset_indices is not None:
            subset_idx = subset_indices[2]
        else:
            subset_idx = np.arange(0, np.shape(von_mises_fitted_loc)[0])

        axs[2].scatter(von_mises_fitted_loc[subset_idx, 1], von_mises_fitted_loc[subset_idx, 2], color='black', lw=0,
                       s=dot_size, clip_on=False)
        axs[2].set_xlabel('Nasal saccade', size=11)
        axs[2].set_ylabel('Temporal saccade', size=11)

        if np.sum(subset_idx) > 2:
            r_val, test_stat, p_value = circ_corrcoef(von_mises_fitted_loc[subset_idx, 1], von_mises_fitted_loc[subset_idx, 2],
                                                  test=True)

            axs[2].set_title(r'$r = %.2f, p = %.2f$' % (r_val, p_value), size=11)

        [ax.set_xlim(xlim) for ax in axs]
        [ax.set_ylim(ylim) for ax in axs]
        [ax.set_xticks(tick_vals) for ax in axs]
        [ax.set_yticks(tick_vals) for ax in axs]

    return fig, axs

def plot_running_weight_dist(both_exp_running_weights, gray_exp_running_weights, grating_exp_running_weights,
                             num_bins=100, fig=None, axs=None):

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(1, 3, sharex=True)
        fig.set_size_inches(9, 3)

    hist_color = 'black'

    axs[0].hist(both_exp_running_weights, bins=num_bins, lw=0, color=hist_color)
    axs[0].set_title('Both experiments', size=11)

    axs[1].hist(gray_exp_running_weights, bins=num_bins, lw=0, color=hist_color)
    axs[1].set_title('Gray experiments', size=11)

    axs[2].hist(grating_exp_running_weights, bins=num_bins, lw=0, color=hist_color)
    axs[2].set_title('Grating experiments', size=11)

    axs[0].set_ylabel('Number of cells', size=11)
    fig.text(0.5, -0.04, 'Running weight', size=11, ha='center')

    axs[0].set_xlim([-1, 1])

    return fig, axs


def main():

    available_processes = ['load_data', 'plot_data',
                           'plot_raster_after_saccade_off_screen_exclusion',
                           'plot_running_data',
                           'fit_regression_model',
                           'plot_example_neuron_fit',
                           'plot_regression_model_explained_var',
                           'plot_regression_model_full_vs_aligned_explained_var',
                           'plot_regression_model_example_neurons', 'compare_iterative_vs_normal_fit',
                           'plot_pupil_data', 'plot_original_vs_aligned_explained_var',
                           'plot_grating_vs_gray_screen_single_cell_fit_performance',
                           'plot_grating_vs_gray_screen_example_neurons', 'compare_saccade_kernels',
                           'compare_saccade_triggered_average',
                           'plot_sig_vs_nosig_neuron_explained_var',
                           'plot_saccade_neuron_psth_and_regression',
                           'plot_sig_model_comparison_neurons',
                           'plot_num_saccade_per_ori',
                           'plot_vis_and_saccade_neuron_individual_trials',
                           'fit_saccade_ori_tuning_curves',
                           'plot_saccade_ori_tuning_curves', 'plot_saccade_ori_preferred_ori',
                           'plot_vis_and_saccade_response_sorted_raster',
                           'plot_vis_neurons_vs_vis_saccade_neuron_preferred_ori',
                           'plot_saccade_kernel_and_sign_distribution',
                           'plot_kernel_fit_raster',
                           'plot_kernel_scatter',
                           'plot_running_weights',
                           'plot_kernel_train_test', 
                           'plot_before_after_saccade_exclusion_ev',
                           'compare_exp_times']


    processes_to_run = ['fit_regression_model']
    process_params = {
        'load_data': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis/New',
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir']
        ),
        'plot_data': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis/New',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis'],
            zscore_activity=True,
        ),
        'plot_raster_after_saccade_off_screen_exclusion': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis/New07032023',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_tracesGray', '_onsetOffset', '_pupilSizeGray',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections', '_inPerNeuron',
                                '_saccadeIntervalsVis', '_inPerNeuronVis'],
            exp_type='grating',  # only supports gray screen experiment, no equivalent for grating experiment?
            zscore_activity=True,
        ),
        'compare_exp_times': dict(
            data_folders=['/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis/New',
                          '/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis/New07032023'],
            file_types_to_load=['_windowVis', '_windowGray', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_tracesGray', '_onsetOffset', '_pupilSizeGray',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections', '_inPerNeuron',
                                '_saccadeIntervalsVis', '_inPerNeuronVis'],
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
        ),
        'plot_running_data' : dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/sc_neurons_2p',

        ),
        'fit_regression_model': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis/New07032023',  # New07032023
            running_data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/sc_neurons_2p',
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults/exclude-off-screen',
            file_types_to_load=['_windowVis', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis', '_inPerNeuron', '_inPerNeuronVis',
                                '_windowGray', '_tracesGray', '_pupilSizeGray', '_onsetOffset', '_trial_Dir',  # gray screen experiments
                                ],
            exp_ids=[
                     'SS041_2015-04-23',
                     'SS044_2015-04-28',
                     'SS045_2015-05-04',
                     'SS045_2015-05-05',
                     'SS047_2015-11-23',
                     'SS047_2015-12-03',
                     'SS048_2015-11-09',
                     'SS048_2015-12-02'
                    ],
            X_sets_to_compare={
                               'bias_only': ['bias'],
                               # 'vis_on_only': ['bias', 'vis_on'],
                               # 'vis_ori': ['bias', 'vis_ori'],
                               # 'vis_ori_iterative': ['bias', 'vis_ori_iterative'],
                               # 'vis': ['bias', 'vis_on', 'vis_dir'],
                               # 'vis_ori_and_pupil': ['bias', 'vis_ori', 'pupil_size'],
                               # 'saccade_on_only': ['bias', 'saccade_on'],
                               # 'saccade': ['bias', 'saccade_on', 'saccade_dir'],
                               # 'pupil_size_only': ['bias', 'pupil_size'],
                               # 'saccade_and_pupil': ['bias', 'saccade_on', 'saccade_dir', 'pupil_size'],
                               # 'vis_and_saccade': ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir'],
                               # 'vis_ori_and_saccade': ['bias', 'vis_ori', 'saccade_on', 'saccade_dir'],
                               'vis_and_running': ['bias', 'vis_on', 'vis_dir', 'running'],
                               # 'vis_ori_and_running': ['bias', 'vis_ori', 'running'],
                               # 'vis_and_saccade_and_pupil': ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir', 'pupil_size'],
                               'vis_on_and_saccade_and_running':  ['bias', 'vis_on', 'saccade_on', 'saccade_dir', 'running'],
                               'vis_and_saccade_and_running': ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir', 'running'],
                               'vis_and_saccade_on_and_running': ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'running'],
                               # 'vis_and_saccade_nt_and_running': ['bias', 'vis_on', 'vis_dir', 'saccade_nasal', 'saccade_temporal', 'running'],
                               # 'running': ['bias', 'running'],
                               # 'saccade': ['bias', 'saccade_on', 'saccade_dir'],
                               # 'saccade_and_running': ['bias', 'saccade_on', 'saccade_dir', 'running'],
                               # 'saccade_on_and_running': ['bias', 'saccade_dir', 'running'],
                               # 'saccade_shuffled_and_running': ['bias', 'saccade_on_shuffled', 'saccade_dir_shuffled', 'running'],
                               # 'saccade_nt': ['bias', 'saccade_nasal', 'saccade_temporal'],
                               # 'vis_ori_and_saccade_and_running': ['bias', 'vis_ori', 'saccade_on', 'saccade_dir', 'running'],
                               'vis_and_saccade_shuffled_and_running': ['bias', 'vis_on', 'vis_dir', 'saccade_on_shuffled', 'saccade_dir_shuffled',
                                                                          'running'],
                               # 'vis_ori_and_saccade_shuffled_and_running': ['bias', 'vis_ori', 'saccade_on_shuffled', 'saccade_dir_shuffled',
                               #                                          'running'],
                               # 'vis_and_saccade_and_running_and_pupil': ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir',
                               #                                 'running', 'pupil_size'],
                               # 'vis_ori_and_saccade_and_running_and_pupil': ['']
                               # 'vis_on_and_saccade_on': ['bias', 'vis_on', 'saccade_on'],
                               # 'vis_on_and_saccade_on_and_interaction': ['bias', 'vis_on', 'saccade_on',
                               #                                           'vis_on_saccade_on']
                               },
            feature_time_windows={'vis_on': [-1.0, 3.0], 'vis_dir': [-1.0, 3.0], 'vis_ori': [-1.0, 3.0],
                                  # 'saccade_on': [-1.0, 3.0], 'saccade_dir': [-1.0, 3.0],
                                  'saccade_on': [-0.5, 0.5], 'saccade_dir': [-0.5, 0.5],
                                  'saccade_on_shuffled': [-0.5, 0.5], 'saccade_dir_shuffled': [-0.5, 0.5],
                                  'saccade_nasal': [-0.5, 0.5], 'saccade_temporal': [-0.5, 0.5],
                                  'vis_on_saccade_on': [-1.0, 3.0], 'vis_ori_iterative': [0, 3.0],
                                  'pupil_size': None, 'running': None},
            performance_metrics=['explained_variance', 'r2'],
            aligned_explained_var_time_windows={'vis': [0, 3],  # originally [0, 3]
                                                'saccade': [-0.5, 0.5]},  # originally [0, 0.5]
            exp_type='grating',  # 'grating', 'gray', 'both'
            exclude_saccade_on_vis_exp=False,
            exclude_saccade_off_screen_times=True,
            exclude_saccade_any_off_screen_times=False,
            saccade_off_screen_exclusion_window=[-0.5, 0.5],
            pupil_preprocessing_steps=['zscore'],
            train_test_split_method='n_fold_cv',   # 'half' or 'n_fold_cv'
            n_cv_folds=10,   # default to 10
            neural_preprocessing_steps=['z_score_separately', 'zscore_w_baseline'],  # 'zscore' is optional, 'z_score_separately', 'zscore_w_baseline'
            # for 'both' experiment option, use: ['z_score_separately', 'zscore_w_baseline']
            # for 'gray' / 'grating' experiment option, use : 'zscore_w_baseline'
        ),
        'plot_example_neuron_fit': dict(

        ),
        'plot_regression_model_explained_var': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults/exclude-off-screen/archive2',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/single-neuron-exclude-saccade-off-screen',
            # fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
            exp_ids=[
                'SS041_2015-04-23',
                'SS044_2015-04-28',
                'SS045_2015-05-04',
                'SS045_2015-05-05',
                'SS047_2015-11-23',
                'SS047_2015-12-03',
                'SS048_2015-11-09',
                'SS048_2015-12-02'],
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
                # ['saccade_and_pupil', 'saccade'],
                # ['vis_ori_and_pupil', 'vis_ori'],
               #  ['vis_ori', 'saccade'],
                # ['vis_ori_and_pupil', 'saccade_and_pupil']
                  ['vis_and_running', 'vis_and_saccade_and_running'],
               #  ['vis_and_saccade_shuffled_and_running', 'vis_and_saccade_and_running'],
              #   ['bias_only', 'saccade'],
                # ['saccade', 'saccade_and_running'],
              #   ['running', 'saccade_and_running'],
              #   ['saccade_on_and_running', 'saccade_and_running'],
                # ['saccade_shuffled_and_running', 'saccade_and_running'],
               #  ['vis_and_saccade', 'vis_and_saccade_and_pupil'],
               #  ['vis_and_saccade_and_running', 'vis_and_saccade_and_running_and_pupil']
               #  ['vis_and_saccade_and_running', 'vis_and_saccade_nt_and_running']
               #  ['saccade', 'saccade_nt']
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
                  # ['saccade_aligned_var_explained', 'saccade_aligned_var_explained'],
               #  ['explained_var_per_X_set', 'explained_var_per_X_set'],
               ['saccade_aligned_var_explained', 'saccade_aligned_var_explained'],
               #  ['saccade_aligned_var_explained', 'saccade_aligned_var_explained'],
              #  ['explained_var_per_X_set', 'explained_var_per_X_set'],
               #  ['saccade_aligned_var_explained', 'saccade_aligned_var_explained'],
               #  ['saccade_aligned_var_explained', 'saccade_aligned_var_explained'],
               #  ['saccade_aligned_var_explained', 'saccade_aligned_var_explained'],
                  # ['vis_aligned_var_explained', 'saccade_aligned_var_explained'],
                  # ['vis_aligned_var_explained', 'saccade_aligned_var_explained'],
                 # ['saccade_aligned_var_explained', 'saccade_aligned_var_explained'],
                 #  ['saccade_aligned_var_explained', 'saccade_aligned_var_explained'],
                 #  ['saccade_aligned_var_explained', 'saccade_aligned_var_explained'],
                  # ['explained_var_per_X_set', 'explained_var_per_X_set']
            ]),
            custom_fig_addinfo='full_trace',  # 'original', or 'aligned', 'aligned_r2', 'original_r2', 'aligned_separate_zscore_using_iti'
            exp_type='both',  # 'grating', 'gray', 'both'
            clip_at_zero=True,
            include_exp_name_in_title=True,
            plot_delta_ev_summary=True,
            custom_xlim=[-3, 1]  # set to None to automatically find this
        ),
        'plot_regression_model_full_vs_aligned_explained_var': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/both-include-vis-saccade-separate-zscore',
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
                # ['saccade_and_pupil', 'saccade'],
                # ['vis_ori_and_pupil', 'vis_ori'],
                #  ['vis_ori', 'saccade'],
                # ['vis_ori_and_pupil', 'saccade_and_pupil']
                ['vis_and_running', 'vis_and_saccade_and_running'],
                ['vis_and_saccade_shuffled_and_running', 'vis_and_saccade_and_running'],
                #  ['vis_and_saccade', 'vis_and_saccade_and_pupil'],
                #  ['vis_and_saccade_and_running', 'vis_and_saccade_and_running_and_pupil']
                #  ['vis_and_saccade_and_running', 'vis_and_saccade_nt_and_running']
                #  ['saccade', 'saccade_nt']
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
            custom_fig_addinfo='both_exp_saccade_shuffled_vs_not_shuffled',
            # 'original', or 'aligned', 'aligned_r2', 'original_r2', 'aligned_separate_zscore_using_iti'
            exp_type='both',  # 'grating', 'gray', 'both'
            clip_at_zero=True,
            include_exp_name_in_title=True,
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
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults/exclude-off-screen',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/saccade_kernel_comparison-short-time-window/new/',
            plot_variance_explained_comparison=False,
            gray_exp_model='saccade_and_running',
            gray_exp_null_model='running',
            grating_exp_model='vis_and_saccade_and_running',
            grating_exp_null_model='vis_and_running',
            sig_experiment='both',  # 'gray', 'grating', 'both'
            num_neurons_to_plot=1,  # originally 10
            kernel_summary_metric='peak',  # 'peak', 'mean-post-saccade', 'mean', 'peak-post-saccade', 'peak-pre-saccade',
            metric_to_plot='explained_var_per_X_set',  # saccade_aligned_var_explained
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            file_types_to_load=['_windowVis', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis',
                                '_windowGray', '_tracesGray', '_pupilSizeGray', '_onsetOffset', '_trial_Dir',
                                # gray screen experiments
                                ],
        ),
        'compare_saccade_triggered_average': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/saccade_kernel_comparison-short-time-window',
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
            X_sets_to_compare=[  # note the first one is to more complicated one usually
                # ['vis_and_saccade', 'saccade'],
                # ['vis_and_saccade', 'vis_ori'],
                # ['vis_and_saccade_and_running', 'vis_and_saccade'],
                ['vis_and_saccade_and_running_and_pupil', 'vis_and_saccade_and_running'],
            ],
            custom_fig_addinfo=None,
            num_neurons_to_plot=5,  # this is per recording I think...
            exp_type='both',
            ev_metric='aligned',
            model_a_explained_var_threshold=0,  # previously 0.1
            model_b_explained_var_threshold=0,  # preivously 0.1
            min_model_ev_diff=0.02,  # originally 0.05
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
            plot_prefer_orientation=False,
            plot_tuning_curves=True,
        ),
        'plot_vis_and_saccade_neuron_stim_and_saccade_triggered_average': dict(
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
        ),
        'fit_saccade_ori_tuning_curves': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            save_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/vis_and_saccade_neurons/ori-tuning-curves',
            plot_variance_explained_comparison=False,
            gray_exp_model='saccade',
            grating_exp_model='vis_and_saccade',
            neurons_to_fit='all',  # 'all' or 'vis_and_saccade'
            num_neurons_to_plot=10,
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            file_types_to_load=['_windowVis', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis',
                                '_windowGray', '_tracesGray', '_pupilSizeGray', '_onsetOffset', '_trial_Dir',
                                # gray screen experiments
                                ],
            do_imputation=False,
            run_parallel=True,
        ),
        'plot_saccade_ori_tuning_curves': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/vis_and_saccade_neurons/ori-tuning-curves',
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
            do_imputation=False,
        ),
        'plot_saccade_ori_preferred_ori': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/vis_and_saccade_neurons/ori-tuning-curves',
            fitted_params_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/vis_and_saccade_neurons/ori-tuning-curves',
            fig_exts=['.png', '.svg']
        ),
        'plot_vis_and_saccade_response_sorted_raster': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/vis_and_saccade_neurons/ori-tuning-curves',
            fitted_params_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/vis_and_saccade_neurons/ori-tuning-curves',
            sort_using='vis',  # 'vis', 'nasal' or 'temporal'
            zscore_ea_neuron_separately=False,
            divide_by_max=False,
            scale_to_unit_range=True,
            include_histogram=True,
            only_plot_sig_neurons=True,
            zscore_cmap='viridis',   # 'bwr', 'viridis'
            general_cmap='viridis',
            gray_nans=False,
            plot_fitted_curves=True,
        ),
        'plot_vis_neurons_vs_vis_saccade_neuron_preferred_ori': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/vis_and_saccade_neurons/ori-tuning-curves',
            fitted_params_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/vis_and_saccade_neurons/ori-tuning-curves',
            fig_exts=['.png', '.svg']
        ),
        'plot_saccade_kernel_and_sign_distribution': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/saccade_neurons/weights/',
            plot_variance_explained_comparison=False,
            gray_exp_model='saccade_and_running',
            gray_exp_null_model='running',
            subset_only_better_than_null=True,
            grating_exp_model='vis_and_saccade',
            both_exp_model=None,
            file_types_to_load=['_windowVis', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis',
                                '_windowGray', '_tracesGray', '_pupilSizeGray', '_onsetOffset', '_trial_Dir',
                                '_sigNeurons'
                                # gray screen experiments
                                ],
        ),
        'plot_kernel_fit_raster': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults/exclude-off-screen/archive2',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/weights_raster/exclude-off-screen',
            model_X_set_to_plot='vis_and_saccade_and_running',  # vis_and_saccade_and_running / saccade_and_running for gray screen
            exp_type='both',  # 'gray' or 'both'
            sort_method='Anya-identical',
            kernels_to_include=['vis_dir', 'vis_on', 'saccade_on', 'saccade_dir'],   #'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir'
            transform_dir_to_temporal_nasal=True,
            explained_var_threshold=0.02,
            file_types_to_load=['_windowVis', '_tracesVis', '_trial_Dir', '_saccadeVisDir',
                                '_gratingIntervals', '_gratingIds', '_gratingIdDirections',
                                '_saccadeIntervalsVis', '_pupilSizeVis',
                                '_windowGray', '_tracesGray', '_pupilSizeGray', '_onsetOffset', '_trial_Dir',
                                '_sigNeurons'
                                # gray screen experiments
                                ],
            num_running_kernels_to_duplicate=29,
            num_separating_columns=10,
        ),
        'plot_kernel_scatter': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults/exclude-off-screen/archive2',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/kernel_scatter/exclude-off-screen',
            neuron_subset_condition='better_than_null',  # 'sig_saccade_neurons' or 'better_than_null', 'both_vis_dir_and_saccade_dir_selective'
            model_X_set_to_plot='vis_and_saccade_and_running',  # vis_and_saccade_and_running, saccade_and_running
            null_X_set='vis_and_running',   # 'running', or 'vis_and_running', 'vis_and_saccade_on_and_running', 'saccade_and_running'
            explained_var_threshold=0,
            exp_type='both',  # 'gray', 'grating', 'both'
            x_axis_kernel='saccade_dir_nasal',  # 'vis_dir_nasal' , saccade_dir_nasal, 'vis_dir_diff',
            y_axis_kernel='saccade_dir_temporal',  # 'vis_dir_temporal', 'saccade_dir_temporal', 'saccade_dir_diff',
            kernel_metric='peak-post-saccade',   # 'mean' or 'peak' or 'peak-post-saccade'
            do_stats=True,
            same_x_y_range=True,
            plot_indv_lobf=False,  # line of best fit
        ),
        'plot_kernel_train_test': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults/2-fold',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/kernel_scatter/',
            neuron_subset_condition='better_than_null',
            # 'sig_saccade_neurons' or 'better_than_null', 'both_vis_dir_and_saccade_dir_selective'
            model_X_set_to_plot='vis_and_saccade_and_running',  # vis_and_saccade_and_running, saccade_and_running
            null_X_set='vis_and_running',  # 'running', or 'vis_and_running'
            explained_var_threshold=0,
            exp_type='grating',  # 'gray', 'grating', 'both'
            kernels_to_plot=['saccade_nasal', 'saccade_temporal'],
            kernel_metric='mean',  # 'mean' or 'peak'
            do_stats=True,
            same_x_y_range=False,
            plot_indv_lobf=False,  # line of best fit
        ),
        'plot_running_weights': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/running_weights/',
            both_exp_model='vis_and_saccade_and_running',
            both_exp_null_model='vis_and_saccade',
            grating_exp_model='vis_and_saccade_and_running',
            grating_exp_null_model='vis_and_saccade',
            gray_exp_model='saccade_and_running',
            gray_exp_null_model='saccade',
        ), 
        'plot_before_after_saccade_exclusion_ev': dict(
            data_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/InteractionSacc_Vis',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/before-after-saccade-exclusion',
            before_regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            after_regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults/exclude-off-screen',
            exp_type='gray',
            metric_to_plot='saccade_aligned_var_explained',  # explained_var_per_X_set, saccade_aligned_var_explained
            plot_kernel_traces=True,
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

        if process == 'plot_raster_after_saccade_off_screen_exclusion':

            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])
            fig_folder = process_params[process]['fig_folder']
            exp_type = process_params[process]['exp_type']
            exclude_saccade_off_screen_times = True

            for exp_id, exp_data in data.items():

                """
                X, Y, grating_orientation_per_trial, saccade_dirs, feature_indices, subset_index \
                        = make_X_Y_for_regression(exp_data,
                                        feature_set=['bias'],
                                        feature_time_windows={'vis_on': [-1.0, 3.0], 'vis_dir': [-1.0, 3.0],
                                                              'vis_ori': [-1.0, 3.0],
                                                              'saccade_on': [-1.0, 3.0], 'saccade_dir': [-1.0, 3.0],
                                                              'vis_on_saccade_on': [-1.0, 3.0],
                                                              'vis_ori_iterative': [0, 3.0]},
                                        neural_preprocessing_steps=['zscore'], check_for_nans=True, exp_type=exp_type,
                                        exclude_saccade_on_vis_exp=False,
                                        train_indices=None, test_indices=None,
                                        exclude_saccade_off_screen_times=exclude_saccade_off_screen_times,
                                        return_trial_type=True, pupil_preprocessing_steps=[])

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(6, 4)
                    ax.imshow(Y.T, aspect='auto')
                    fig_name = '%s_%s_raster_with_saccade_off_screen_set_to_nan' % (exp_id, exp_type)
                    fig_path = os.path.join(fig_folder, fig_name)
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print('Saved figure to %s' % fig_path)
                    plt.close(fig)
                """

                # Plot individual trial
                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(6, 4)

                    if exp_type == 'gray':
                        in_per_neuron = exp_data['_inPerNeuron']  # trial x neuron
                    elif exp_type == 'grating':
                        in_per_neuron = exp_data['_inPerNeuronVis']

                    ax.imshow(in_per_neuron, aspect='auto')

                    ax.set_xlabel('Neuron', size=11)
                    ax.set_ylabel('Saccade trial', size=11)

                    num_in_screen_per_neuron = np.sum(in_per_neuron, axis=0)
                    num_1_trial = len(np.where(num_in_screen_per_neuron >= 1)[0])
                    num_2_trial = len(np.where(num_in_screen_per_neuron >= 2)[0])
                    num_5_trial = len(np.where(num_in_screen_per_neuron >= 5)[0])
                    num_10_trial = len(np.where(num_in_screen_per_neuron >= 10)[0])

                    ax.set_title('''%s
                                  Number of neurons with at least 1 usable trial: %.f, 
                                  2 usable trials: %.f 
                                  5 usable trials: %.f  
                                  10 usable trials: %.f'''
                                 % (exp_id, num_1_trial, num_2_trial, num_5_trial, num_10_trial), size=9)

                    fig_name = '%s_%s_inPerNeuron' % (exp_id, exp_type)
                    fig_path = os.path.join(fig_folder, fig_name)
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print('Saved figure to %s' % fig_path)
                    plt.close(fig)


        if process == 'plot_running_data':

            data_folder = process_params[process]['data_folder']
            subject_list = os.listdir(data_folder)

            for subject in subject_list:
                subject_folder = os.path.join(data_folder, subject)

                exp_date_list = os.listdir(subject_folder)
                
                for exp_date in exp_date_list:
                    exp_date_folder = os.path.join(subject_folder, exp_date)
                    exp_num_list = os.listdir(exp_date_folder)
                
                    for exp_num in exp_num_list:
                        
                        exp_folder = os.path.join(exp_date_folder, exp_num)
                        
                        # find running data and start plotting
                        running_speed_data_path = os.path.join(exp_folder, '_ss_running.speed.npy')
                        running_speed_timestamp_data_path = os.path.join(exp_folder, '_ss_running.timestamps.npy')

                        running_speed = np.load(running_speed_data_path)
                        running_speed_timestamps = np.load(running_speed_timestamp_data_path)

                        print('TODO: write function to make plots')


        if process == 'fit_regression_model':

            exp_ids = process_params[process]['exp_ids']
            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'],
                             exp_ids=exp_ids)
            regression_results_folder = process_params[process]['regression_results_folder']
            neural_preprocessing_steps = process_params[process]['neural_preprocessing_steps']
            exclude_saccade_on_vis_exp = process_params[process]['exclude_saccade_on_vis_exp']
            aligned_explained_var_time_windows = process_params[process]['aligned_explained_var_time_windows']
            X_sets_to_compare = process_params[process]['X_sets_to_compare']
            performance_metrics = process_params[process]['performance_metrics']
            n_cv_folds = process_params[process]['n_cv_folds']
            exclude_saccade_off_screen_times = process_params[process]['exclude_saccade_off_screen_times']
            saccade_off_screen_exclusion_window = process_params[process]['saccade_off_screen_exclusion_window']
            exclude_saccade_any_off_screen_times = process_params[process]['exclude_saccade_any_off_screen_times']

            if not os.path.isdir(regression_results_folder):
                os.makedirs(regression_results_folder)

            # feature_set = ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir']

            for exp_id, exp_data in data.items():

                print('Fitting regression model to %s' % exp_id)
                num_X_set = len(X_sets_to_compare.keys())
                num_neurons = np.shape(exp_data['_tracesVis'])[1]
                explained_var_per_X_set = np.zeros((num_neurons, num_X_set))
                vis_aligned_explained_var_per_X_set = np.zeros((num_neurons, num_X_set))
                saccade_aligned_explained_var_per_X_set = np.zeros((num_neurons, num_X_set))

                vis_aligned_Y_hat_per_X_set = []
                saccade_aligned_Y_hat_per_X_set = []
                vis_aligned_Y_per_X_set = []
                saccade_aligned_Y_per_X_set = []

                saccade_dir_per_X_set = []
                vis_ori_per_X_set = []
                full_saccade_dir_per_X_set = []
                full_vis_ori_per_X_set = []

                Y_vis_aligned_full_per_X_set = []
                Y_saccade_aligned_full_per_X_set = []
                Y_hat_vis_aligned_full_per_X_set = []
                Y_hat_saccade_aligned_full_per_X_set = []

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

                    if 'running' in feature_set:

                        running_data_folder = process_params[process]['running_data_folder']
                        subject = exp_id.split('_')[0]
                        exp_date = exp_id.split('_')[1]

                        exp_folder = os.path.join(running_data_folder, subject, exp_date, '001')
                        running_speed, running_speed_timestamps, grating_interval, gray_interval = load_running_data(exp_folder)
                        exp_data['running_speed'] = running_speed
                        exp_data['running_speed_timestamps'] = running_speed_timestamps
                        exp_data['grating_interval'] = grating_interval
                        exp_data['gray_interval'] = gray_interval

                    if not exclude_saccade_off_screen_times:
                        X, Y, grating_orientation_per_trial, saccade_dirs, feature_indices, subset_index = make_X_Y_for_regression(exp_data, feature_set=feature_set,
                                                   neural_preprocessing_steps=neural_preprocessing_steps,
                                                   train_indices=train_indices, test_indices=test_indices,
                                                   exp_type=process_params[process]['exp_type'],
                                                   exclude_saccade_on_vis_exp=exclude_saccade_on_vis_exp,
                                                   feature_time_windows=process_params[process]['feature_time_windows'],
                                                   return_trial_type=True,
                                                   pupil_preprocessing_steps=process_params[process]['pupil_preprocessing_steps'],
                                                   exclude_saccade_off_screen_times=exclude_saccade_off_screen_times,
                                                  exclude_saccade_any_off_screen_times=exclude_saccade_any_off_screen_times,
                                                   saccade_off_screen_exclusion_window=saccade_off_screen_exclusion_window,
                                                   )

                    else:
                        X, Y, grating_orientation_per_trial, saccade_dirs, feature_indices, subset_index_per_neuron = make_X_Y_for_regression(
                            exp_data, feature_set=feature_set,
                            neural_preprocessing_steps=neural_preprocessing_steps,
                            train_indices=train_indices, test_indices=test_indices,
                            exp_type=process_params[process]['exp_type'],
                            exclude_saccade_on_vis_exp=exclude_saccade_on_vis_exp,
                            feature_time_windows=process_params[process]['feature_time_windows'],
                            return_trial_type=True,
                            pupil_preprocessing_steps=process_params[process]['pupil_preprocessing_steps'],
                            exclude_saccade_off_screen_times=exclude_saccade_off_screen_times,
                            saccade_off_screen_exclusion_window=saccade_off_screen_exclusion_window)

                    feature_indices_per_X_set[X_set_name] = feature_indices

                    if not exclude_saccade_off_screen_times:
                        regression_result = fit_regression_model(X, Y, performance_metrics=performance_metrics,
                                                             train_test_split_method=process_params[process]['train_test_split_method'],
                                                             n_cv_folds=n_cv_folds, exclude_saccade_off_screen_times=exclude_saccade_off_screen_times)

                        regression_result = get_aligned_explained_variance(regression_result,
                                                                        exp_data, performance_metrics=performance_metrics,
                                                                       exp_type=process_params[process]['exp_type'],
                                                                       exclude_saccade_on_vis_exp=exclude_saccade_on_vis_exp,
                                                                       alignment_time_window=aligned_explained_var_time_windows,
                                                                       exclude_saccade_off_screen_times=exclude_saccade_off_screen_times,
                                                                       exclude_saccade_any_off_screen_times=exclude_saccade_any_off_screen_times,
                                                                       subset_index=subset_index)

                        # NOTE: doing nanmean here in case some CV set has 0 variance explained because of no saccade...
                        explained_var_per_X_set[:, n_X_set] = np.nanmean(
                            regression_result['explained_variance_per_cv_set'], axis=1)


                        vis_aligned_explained_var_per_X_set[:, n_X_set] = regression_result['vis_aligned_var_explained']
                        saccade_aligned_explained_var_per_X_set[:, n_X_set] = regression_result[
                            'saccade_aligned_var_explained']

                        if 'r2' in performance_metrics:
                            r2_per_X_set[:, n_X_set] = np.mean(regression_result['r2_per_cv_set'], axis=1)
                            vis_aligned_r2_per_X_set[:, n_X_set] = regression_result['vis_aligned_r2']
                            saccade_aligned_r2_per_X_set[:, n_X_set] = regression_result['saccade_aligned_r2']

                        Y_test_hat_per_X_set.append(regression_result['Y_test_hat_per_cv_set'])

                        vis_aligned_Y_hat_per_X_set.append(regression_result['Y_hat_vis_aligned'])
                        saccade_aligned_Y_hat_per_X_set.append(regression_result['Y_hat_saccade_aligned'])
                        vis_aligned_Y_per_X_set.append(regression_result['Y_vis_aligned'])
                        saccade_aligned_Y_per_X_set.append(regression_result['Y_saccade_aligned'])
                        model_weights_per_X_set[X_set_name] = regression_result['weights_per_cv_set']  # numCV x numNeuorn x 1 (what is the 1 doing?)


                    else:
                        # do the fitting per neuron
                        neuron_regression_results = []
                        num_neuron = len(Y)
                        for neuron_idx in np.arange(num_neuron):
                            X_neuron = X[neuron_idx]
                            if X_neuron.ndim == 1:
                                X_neuron = X_neuron.reshape(-1, 1)

                            # time indices with saccade off time excluded
                            subset_index = subset_index_per_neuron[neuron_idx]

                            pdb.set_trace()

                            neuron_regression_result = fit_regression_model(X_neuron, Y[neuron_idx].reshape(-1, 1), performance_metrics=performance_metrics,
                                                             train_test_split_method=process_params[process]['train_test_split_method'],
                                                             n_cv_folds=n_cv_folds, exclude_saccade_off_screen_times=exclude_saccade_off_screen_times)

                            neuron_regression_result = get_aligned_explained_variance(neuron_regression_result, exp_data,
                                                                       performance_metrics=performance_metrics,
                                                                       exp_type=process_params[process]['exp_type'],
                                                                       exclude_saccade_on_vis_exp=exclude_saccade_on_vis_exp,
                                                                       alignment_time_window=aligned_explained_var_time_windows,
                                                                       exclude_saccade_off_screen_times=exclude_saccade_off_screen_times,
                                                                       subset_index=subset_index, neuron_idx=neuron_idx)

                            neuron_regression_results.append(neuron_regression_result)

                        ev_per_neuron = np.array([np.mean(x['explained_variance_per_cv_set']) for x in neuron_regression_results])
                        explained_var_per_X_set[:, n_X_set] = ev_per_neuron

                        vis_ev_per_neuron = np.array([np.mean(x['vis_aligned_var_explained']) for x in neuron_regression_results])
                        vis_aligned_explained_var_per_X_set[:, n_X_set] = vis_ev_per_neuron

                        saccade_ev_per_neuron = np.array([np.mean(x['saccade_aligned_var_explained']) for x in neuron_regression_results])
                        saccade_aligned_explained_var_per_X_set[:, n_X_set] = saccade_ev_per_neuron
            
                        weights_per_cv_set_per_neuron = np.array([x['weights_per_cv_set'][:, 0, :] for x in neuron_regression_results])
                        model_weights_per_X_set[X_set_name] = np.swapaxes(weights_per_cv_set_per_neuron, 0, 1)

                        Y_test_hat_per_cv_set_per_neuron = [x['Y_test_hat_per_cv_set'] for x in neuron_regression_results]
                        Y_test_per_cv_set_per_neuron = [x['Y_test_per_cv_set'] for x in neuron_regression_results]
                        Y_hat_vis_aligned_per_neuron = [x['Y_hat_vis_aligned'] for x in neuron_regression_results]
                        Y_hat_saccade_aligned_per_neuron = [x['Y_hat_saccade_aligned'] for x in neuron_regression_results]
                        Y_vis_aligned_aligned_per_neuron = [x['Y_vis_aligned'] for x in neuron_regression_results]
                        Y_saccade_aligned_per_neuron = [x['Y_saccade_aligned'] for x in neuron_regression_results]
                        Y_test_idx_per_cv_set_per_neuron = [x['test_idx_per_cv_set'] for x in neuron_regression_results]


                        # FULL
                        Y_vis_aligned_full_per_neuron = [x['Y_vis_aligned_full'] for x in neuron_regression_results]
                        Y_saccade_aligned_full_per_neuron = [x['Y_saccade_aligned_full'] for x in neuron_regression_results]
                        Y_hat_vis_aligned_full_per_neuron = [x['Y_hat_vis_aligned_full'] for x in neuron_regression_results]
                        Y_hat_saccade_aligned_full_per_neuron = [x['Y_hat_saccade_aligned_full'] for x in neuron_regression_results]

                        Y_vis_aligned_full_per_X_set.append(Y_vis_aligned_full_per_neuron)
                        Y_saccade_aligned_full_per_X_set.append(Y_saccade_aligned_full_per_neuron)
                        Y_hat_vis_aligned_full_per_X_set.append(Y_hat_vis_aligned_full_per_neuron)
                        Y_hat_saccade_aligned_full_per_X_set.append(Y_hat_saccade_aligned_full_per_neuron)

                        # NOTE: These are the ones used for getting aligned variance explained, may not be super-informative...
                        saccade_dir_per_neuron = [x['subset_saccade_dir'] for x in neuron_regression_results]
                        vis_ori_per_neuron = [x['subset_vis_ori'] for x in neuron_regression_results]

                        # These are all the trials, perhaps more useful
                        full_vis_ori_per_neuron = [x['vis_oris'] for x in neuron_regression_results]
                        full_saccade_dir_per_neuron = [x['saccade_dirs'] for x in neuron_regression_results]
                        
                        Y_test_hat_per_X_set.append(Y_test_hat_per_cv_set_per_neuron)
                        vis_aligned_Y_hat_per_X_set.append(Y_hat_vis_aligned_per_neuron)
                        saccade_aligned_Y_hat_per_X_set.append(Y_hat_saccade_aligned_per_neuron)
                        vis_aligned_Y_per_X_set.append(Y_vis_aligned_aligned_per_neuron)
                        saccade_aligned_Y_per_X_set.append(Y_saccade_aligned_per_neuron)
                        saccade_dir_per_X_set.append(saccade_dir_per_neuron)
                        vis_ori_per_X_set.append(vis_ori_per_neuron)

                        full_saccade_dir_per_X_set.append(full_saccade_dir_per_neuron)
                        full_vis_ori_per_X_set.append(full_vis_ori_per_neuron)

                if not exclude_saccade_off_screen_times:
                    exp_regression_result['Y_test'] = regression_result['Y_test_per_cv_set']
                    exp_regression_result['test_idx_per_cv_set'] = regression_result['test_idx_per_cv_set']
                    exp_regression_result['Y_test_hat'] = np.array(Y_test_hat_per_X_set)
                else:
                    # single neuron version, allow for unequal array sizes contained in a list
                    num_neurons = len(Y_test_per_cv_set_per_neuron)
                    Y_test = np.empty(num_neurons, object)
                    Y_test[:] = Y_test_per_cv_set_per_neuron
                    test_idx_per_cv_set = np.empty(num_neurons, object)
                    test_idx_per_cv_set[:] = Y_test_idx_per_cv_set_per_neuron
                    exp_regression_result['Y_test'] = Y_test
                    exp_regression_result['test_idx_per_cv_set'] = test_idx_per_cv_set
                    exp_regression_result['Y_test_hat'] = Y_test_hat_per_X_set  # list of neurons

                exp_regression_result['explained_var_per_X_set'] = explained_var_per_X_set
                exp_regression_result['vis_aligned_var_explained'] = vis_aligned_explained_var_per_X_set
                exp_regression_result['saccade_aligned_var_explained'] = saccade_aligned_explained_var_per_X_set
                exp_regression_result['feature_indices_per_X_set'] = feature_indices_per_X_set
                exp_regression_result['model_weights_per_X_set'] = model_weights_per_X_set

                if not exclude_saccade_off_screen_times:
                    if 'r2' in performance_metrics:
                        exp_regression_result['r2_per_X_set'] = r2_per_X_set
                        exp_regression_result['vis_aligned_r2'] = vis_aligned_r2_per_X_set
                        exp_regression_result['saccade_aligned_r2'] = saccade_aligned_r2_per_X_set

                exp_regression_result['Y_hat_vis_aligned'] = vis_aligned_Y_hat_per_X_set
                exp_regression_result['Y_vis_aligned'] = vis_aligned_Y_per_X_set
                exp_regression_result['Y_hat_saccade_aligned'] = saccade_aligned_Y_hat_per_X_set
                exp_regression_result['Y_saccade_aligned'] = saccade_aligned_Y_per_X_set

                exp_regression_result['Y_vis_aligned_full_per_X_set'] = Y_vis_aligned_full_per_X_set
                exp_regression_result['Y_saccade_aligned_full_per_X_set'] = Y_saccade_aligned_full_per_X_set
                exp_regression_result['Y_hat_vis_aligned_full_per_X_set'] = Y_hat_vis_aligned_full_per_X_set
                exp_regression_result['Y_hat_saccade_aligned_full_per_X_set'] = Y_hat_saccade_aligned_full_per_X_set

                exp_regression_result['saccade_dir_per_X_set'] = saccade_dir_per_X_set
                exp_regression_result['vis_ori_per_X_set'] = vis_ori_per_X_set
                exp_regression_result['full_saccade_dir_per_X_set'] = full_saccade_dir_per_X_set
                exp_regression_result['full_vis_ori_per_X_set'] = full_vis_ori_per_X_set

                exp_regression_result['regression_kernel_names'] = np.array([*process_params[process]['feature_time_windows'].keys()])
                exp_regression_result['regression_kernel_windows'] = np.array([*process_params[process]['feature_time_windows'].values()])

                exp_regression_result['grating_orientation_per_trial'] = grating_orientation_per_trial
                exp_regression_result['saccade_dirs'] = saccade_dirs

                if not exclude_saccade_off_screen_times:
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
            include_exp_name_in_title = process_params[process]['include_exp_name_in_title']
            plot_delta_ev_summary = process_params[process]['plot_delta_ev_summary']

            fig_folder = process_params[process]['fig_folder']
            if not os.path.isdir(fig_folder):
                os.makedirs(fig_folder)

            text_size = 11

            delta_ev_per_exp = defaultdict(list)


            for n_fpath, fpath in enumerate(regression_result_files):

                regression_result = np.load(fpath)

                X_sets_names = regression_result['X_sets_names']

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plt.subplots(1, len(X_sets_to_compare))
                    fig.set_size_inches(len(X_sets_to_compare)*3, 3)

                    if len(X_sets_to_compare) == 1:
                        axs = [axs]

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

                        model_a_explained_var = regression_result[model_a_metric][:, model_a_idx].flatten()
                        model_b_explained_var = regression_result[model_b_metric][:, model_b_idx].flatten()

                        #if n_fpath == 2:
                        #       pdb.set_trace()

                        # TODO: think about excluding some neurons (negative in both?)
                        subset_idx = np.where(
                           ~((model_b_explained_var < 0) &
                             (model_a_explained_var < 0))
                        )[0]

                        delta_ev = model_b_explained_var - model_a_explained_var
                        delta_ev_per_exp[n_comparison].append(delta_ev[subset_idx])

                        axs[n_comparison].scatter(model_a_explained_var, model_b_explained_var, color='black',
                                                  s=10)

                        both_model_explained_var = np.concatenate([model_a_explained_var,
                                                                  model_b_explained_var])

                        if process_params[process]['clip_at_zero']:
                            both_model_min = -0.1
                        else:
                            both_model_min = np.nanmin(both_model_explained_var)
                        both_model_max = np.nanmax(both_model_explained_var)
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

                    if include_exp_name_in_title:
                        fig.suptitle('%s %s %s exp' % (subject, exp_date, exp_type), size=11)

                    if custom_fig_addinfo is not None:
                        fig_name = '%s_%s_%s_%s_explained_variance_per_X_set_comparison' % (exp_type, subject, exp_date, custom_fig_addinfo)
                    else:
                        fig_name = '%s_%s_%s_explained_variance_per_X_set_comparison' % (exp_type, subject, exp_date)
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbbox_inches='tight')

                    plt.close(fig)


            if plot_delta_ev_summary:
                for n_comparison in np.arange(0, len(X_sets_to_compare)):
                    # version 1 of plots : histogram of all delta ev, then line / triangle to indicate mean
                    n_comparison_delta_ev_per_exp = delta_ev_per_exp[n_comparison]
                    all_exp_delta_ev = np.concatenate(n_comparison_delta_ev_per_exp).flatten()
                    each_exp_mean_delta_ev = [np.nanmean(x) for x in n_comparison_delta_ev_per_exp]

                    x_plot_min = np.nanpercentile(all_exp_delta_ev, 2.5)
                    x_plot_max = np.nanpercentile(all_exp_delta_ev, 97.5)
                    bins = np.linspace(x_plot_min, x_plot_max, 100)

                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, ax = plt.subplots()
                        ax.hist(all_exp_delta_ev, lw=0, color='black', bins=bins, zorder=0)
                        ax_ymin, ax_ymax = ax.set_ylim()

                        for exp_mean_delta_ev in each_exp_mean_delta_ev:
                            ax.scatter(exp_mean_delta_ev, ax_ymax * 0.9, marker='v', lw=0, zorder=1)

                    model_a = X_sets_to_compare[n_comparison][0]
                    model_b = X_sets_to_compare[n_comparison][1]

                    ax.set_title('%s - %s' % (model_b, model_a), size=11)
                    ax.set_xlabel(r'$\Delta$ Explained Variance', size=11)
                    ax.set_ylabel('Number of units', size=11)
                    # ax.set_xlim([x_plot_min, x_plot_max])
                    fig_name = 'all_exp_delta_ev_hist_X_set_comparison_%.f_%s' % (n_comparison, custom_fig_addinfo)
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                    plt.close(fig)

                    # Box plot for each experiment

                    n_comparison_delta_ev_per_exp_filtered = [x[~np.isnan(x)] for x in n_comparison_delta_ev_per_exp]

                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, ax = plt.subplots()
                        fig.set_size_inches(6, 5)
                        ax.boxplot(n_comparison_delta_ev_per_exp_filtered, vert=False)
                        ax.set_title('%s - %s' % (model_b, model_a), size=11)
                        ax.set_xlabel(r'$\Delta$ Explained Variance', size=11)
                        ax.set_ylabel('Experiments')
                        ax.set_yticks([])
                        ax.spines['left'].set_visible(False)

                        if process_params[process]['custom_xlim'] is not None:
                            ax.set_xlim(process_params[process]['custom_xlim'])

                        fig_name = 'all_%s_exp_delta_ev_boxplot_X_set_comparison_%.f_%s_%s' % (exp_type, n_comparison, metrics_to_compare[0], custom_fig_addinfo)
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                        plt.close(fig)

                    
        if process == 'plot_regression_model_full_vs_aligned_explained_var':

            exp_type = process_params[process]['exp_type']
            regression_result_files = glob.glob(os.path.join(process_params[process]['regression_results_folder'],
                                                             '*%s*npz' % exp_type))
            X_sets_to_compare = process_params[process]['X_sets_to_compare']
            # metrics_to_compare = process_params[process]['metrics_to_compare']
            custom_fig_addinfo = process_params[process]['custom_fig_addinfo']
            include_exp_name_in_title = process_params[process]['include_exp_name_in_title']

            subset_only_both_pos_neurons = True

            fig_folder = process_params[process]['fig_folder']
            text_size = 11

            for fpath in regression_result_files:
                regression_result = np.load(fpath)

                X_sets_names = regression_result['X_sets_names']
                exp_id_parts = os.path.basename(fpath).split('.')[0].split('_')
                subject = exp_id_parts[0]
                exp_date = exp_id_parts[1]

                for n_comparison in np.arange(0, len(X_sets_to_compare)):
                    model_a = X_sets_to_compare[n_comparison][0]
                    model_b = X_sets_to_compare[n_comparison][1]
                    model_a_idx = np.where(X_sets_names == model_a)
                    model_b_idx = np.where(X_sets_names == model_b)

                    model_a_explained_var = regression_result['explained_var_per_X_set'][:, model_a_idx]
                    model_b_explained_var = regression_result['explained_var_per_X_set'][:, model_b_idx]

                    if subset_only_both_pos_neurons:
                        subset_idx = np.where(
                            (model_a_explained_var > 0) &
                            (model_b_explained_var > 0)
                        )[0]
                    else:
                        subset_idx = np.arange(np.shape(model_a_explained_var)[0])

                    model_a_saccade_aligned_ev = regression_result['saccade_aligned_var_explained'][:, model_a_idx]
                    model_b_saccade_aligned_ev = regression_result['saccade_aligned_var_explained'][:, model_b_idx]

                    model_b_minus_a_ev = model_b_explained_var[subset_idx] - model_a_explained_var[subset_idx]
                    model_b_minus_a_saccade_aligned_ev = model_b_saccade_aligned_ev[subset_idx] - model_a_saccade_aligned_ev[subset_idx]


                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, ax = plt.subplots()
                        fig.set_size_inches(4, 4)

                        ax.scatter(model_b_minus_a_ev, model_b_minus_a_saccade_aligned_ev, color='black', s=10, lw=0)
                        ax.set_title('%s %s %s' %  (exp_type, subject, exp_date), size=11)
                        ax.set_xlabel(r'$\Delta$ EV', size=11)
                        ax.set_ylabel(r'$\Delta$ saccade aligned EV', size=11)



                    if custom_fig_addinfo is not None:
                        fig_name = '%s_%s_%s_%s_full_EV_vs_saccade_aligned_EV_diff' % (
                        exp_type, subject, exp_date, custom_fig_addinfo)
                    else:
                        fig_name = '%s_%s_%s_full_EV_vs_saccade_aligned_EV_diff' % (exp_type, subject, exp_date)
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
            print('Saving figures to %s' % fig_folder)
            if not os.path.isdir(fig_folder):
                os.makedirs(fig_folder)

            plot_variance_explained_comparison = process_params[process]['plot_variance_explained_comparison']
            grating_exp_model = process_params[process]['grating_exp_model']
            grating_exp_null_model = process_params[process]['grating_exp_null_model']
            gray_exp_model = process_params[process]['gray_exp_model']
            gray_exp_null_model = process_params[process]['gray_exp_null_model']
            metric_to_plot = process_params[process]['metric_to_plot']
            sig_experiment = process_params[process]['sig_experiment']
            kernel_summary_metric = process_params[process]['kernel_summary_metric']

            # metric_to_plot = 'saccade_aligned_var_explained'

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

            all_gray_exp_nasal_saccade_kernel_mean = []
            all_gray_exp_temporal_saccade_kernel_mean = []
            all_grating_exp_nasal_saccade_kernel_mean = []
            all_grating_exp_temporal_saccade_kernel_mean = []

            all_gray_exp_nasal_saccade_kernel_peak = []
            all_gray_exp_temporal_saccade_kernel_peak = []
            all_grating_exp_nasal_saccade_kernel_peak = []
            all_grating_exp_temporal_saccade_kernel_peak = []

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

                grating_exp_null_model_idx = np.where(
                    grating_regression_result['X_sets_names'] == grating_exp_null_model
                )[0][0]

                gray_exp_saccade_dir_model_idx = np.where(
                    gray_regression_result['X_sets_names'] == gray_exp_model
                )[0][0]

                gray_exp_null_model_idx = np.where(
                    gray_regression_result['X_sets_names'] == gray_exp_null_model
                )[0][0]

                grating_exp_model_explained_var = grating_regression_result[metric_to_plot][:,
                                                        grating_exp_saccade_dir_model_idx]

                grating_exp_null_model_explained_var = grating_regression_result[metric_to_plot][:,
                                                  grating_exp_null_model_idx]

                gray_exp_model_explained_var = gray_regression_result[metric_to_plot][:,
                                                     gray_exp_saccade_dir_model_idx]

                gray_exp_null_model_explained_var = gray_regression_result[metric_to_plot][:,
                                               gray_exp_null_model_idx]


                grating_nasal_saccade_kernel, grating_temporal_saccade_kernel = \
                    get_saccade_kernel_from_regression_result(grating_regression_result, grating_exp_model,
                                                              kernel_summary_metric)

                grating_sig_saccade_neurons = np.where(
                    (grating_exp_model_explained_var > grating_exp_null_model_explained_var) &
                    (grating_exp_model_explained_var > 0)
                )[0]

                gray_nasal_saccade_kernel, gray_temporal_saccade_kernel = \
                    get_saccade_kernel_from_regression_result(gray_regression_result, gray_exp_model,
                                                              kernel_summary_metric)

                gray_sig_saccade_neurons = np.where(
                    (gray_exp_model_explained_var > gray_exp_null_model_explained_var) &
                    (gray_exp_model_explained_var > 0)
                )[0]

                if sig_experiment == 'both':

                    subset_idx = np.intersect1d(grating_sig_saccade_neurons,
                                                gray_sig_saccade_neurons)

                elif sig_experiment == 'grating':

                    subset_idx = grating_sig_saccade_neurons

                elif sig_experiment == 'gray':

                    subset_idx = gray_sig_saccade_neurons

                grating_nasal_saccade_kernel_subset = grating_nasal_saccade_kernel[subset_idx]
                grating_temporal_saccade_kernel_subset = grating_temporal_saccade_kernel[subset_idx]
                gray_nasal_saccade_kernel_subset = gray_nasal_saccade_kernel[subset_idx]
                gray_temporal_saccade_kernel = gray_temporal_saccade_kernel[subset_idx]

                all_gray_exp_nasal_saccade_kernel_mean.append(gray_nasal_saccade_kernel_subset)
                all_gray_exp_temporal_saccade_kernel_mean.append(gray_temporal_saccade_kernel)
                all_grating_exp_nasal_saccade_kernel_mean.append(grating_nasal_saccade_kernel_subset)
                all_grating_exp_temporal_saccade_kernel_mean.append(grating_temporal_saccade_kernel_subset)

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

                # Get saccade time window
                saccade_on_kernel_idx = \
                np.where(grating_regression_result['regression_kernel_names'] == 'saccade_on')[0][0]
                peri_saccade_time_window = grating_regression_result['regression_kernel_windows'][saccade_on_kernel_idx]
                num_time_frames = np.shape(gray_exp_saccade_on_weights)[1]
                peri_saccade_time = np.linspace(peri_saccade_time_window[0], peri_saccade_time_window[1], num_time_frames)

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
                        [ax.axvline(0, linestyle='--', color='gray', alpha=0.25) for ax in [axs[0], axs[1]]]
                        [ax.axhline(0, linestyle='--', color='gray', alpha=0.25) for ax in axs]

                        # Plot saccade on temporal kernesl
                        axs[2].plot(peri_saccade_time, gray_exp_saccade_on_weights[neuron_idx, :], color='gray', label='Gray')
                        axs[2].plot(peri_saccade_time, grating_exp_saccade_on_weights[neuron_idx, :], color='black', label='Grating')
                        axs[2].set_ylabel('Weights', size=11)
                        axs[2].set_xlabel('Peri-saccade time (s)', size=11)

                        # Plot saccade direction temporal kernel
                        axs[3].plot(peri_saccade_time, gray_exp_saccade_dir_weights[neuron_idx, :], color='gray', label='Gray')
                        axs[3].plot(peri_saccade_time, grating_exp_saccade_dir_weights[neuron_idx, :], color='black', label='Grating')
                        axs[3].set_ylabel('Weights', size=11)
                        axs[3].set_xlabel('Peri-saccade time (s)', size=11)
                        axs[3].legend()

                    fig.suptitle('%s neuron %.f' % (exp_id, neuron_idx), size=11)
                    fig_name = '%s_neuron_%.f_saccade_kernel_comparison' % (exp_id, neuron_idx)
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight', transparent=False)
                    plt.close(fig)



            # Compare grating and gray screen kernel mean across all experiments
            # TODO: rename, because this can also be the peak based on kernel_summary Metirc
            all_gray_exp_nasal_saccade_kernel_mean = np.concatenate(all_gray_exp_nasal_saccade_kernel_mean)
            all_grating_exp_nasal_saccade_kernel_mean = np.concatenate(all_grating_exp_nasal_saccade_kernel_mean)
            all_gray_exp_temporal_saccade_kernel_mean = np.concatenate(all_gray_exp_temporal_saccade_kernel_mean)
            all_grating_exp_temporal_saccade_kernel_mean = np.concatenate(all_grating_exp_temporal_saccade_kernel_mean)

            with plt.style.context(splstyle.get_style('nature-reviews')):

                fig, axs = plt.subplots(1, 2)
                fig.set_size_inches(8, 4)
                axs[0].scatter(all_gray_exp_nasal_saccade_kernel_mean,
                               all_grating_exp_nasal_saccade_kernel_mean, color='black', lw=0, s=10)

                # Do stats
                pearson_r, pearson_p_val = sstats.pearsonr(all_gray_exp_nasal_saccade_kernel_mean, all_grating_exp_nasal_saccade_kernel_mean)
                axs[0].set_title('r = %.3f, p = %.5f' % (pearson_r, pearson_p_val), size=11)

                axs[0].set_xlabel('Gray exp nasal saccade', size=11)
                axs[0].set_ylabel('Grating exp nasal saccade', size=11)

                axs[1].scatter(all_gray_exp_temporal_saccade_kernel_mean,
                               all_grating_exp_temporal_saccade_kernel_mean, color='black', lw=0, s=10)
                axs[1].set_xlabel('Gray exp temporal saccade', size=11)
                axs[1].set_ylabel('Grating exp temporal saccade', size=11)

                pearson_r, pearson_p_val = sstats.pearsonr(all_gray_exp_temporal_saccade_kernel_mean,
                                                           all_grating_exp_temporal_saccade_kernel_mean)
                axs[1].set_title('r = %.3f, p = %.5f' % (pearson_r, pearson_p_val), size=11)

                fig.suptitle(kernel_summary_metric, size=11)


                fig_name = 'all_exp_gray_vs_grating_saccade_kernels_%s' % kernel_summary_metric
                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                plt.close(fig)




        if process == 'compare_saccade_triggered_average':

            print('Comparing saccade triggered average of neurons')


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
        if process == 'plot_saccade_kernel_and_sign_distribution':

            print('Loading neurons with significant explained variance using saccade model, and looking at the sign '
                  'of their saccade response')

            regression_results_folder = process_params[process]['regression_results_folder']
            fig_folder = process_params[process]['fig_folder']
            gray_exp_model = process_params[process]['gray_exp_model']
            gray_exp_null_model = process_params[process]['gray_exp_null_model']
            subset_only_better_than_null = process_params[process]['subset_only_better_than_null']

            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            # Store the post - pre response of all gray exp sig saccade dir neurons
            all_exp_sig_saccade_dir_neurons_nasal_postpre_diff = []
            all_exp_sig_saccade_dir_neurons_temporal_postpre_diff = []

            all_exp_saccade_on_model_ev = []
            all_exp_exp_saccade_dir_model_ev = []

            all_exp_saccade_nasal_mean = []
            all_exp_saccade_temporal_mean = []

            for exp_id, exp_data in data.items():
                grating_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*grating*.npz' % (exp_id)))[0],
                    allow_pickle=True)
                gray_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*gray*.npz' % (exp_id)))[0],
                    allow_pickle=True)

                both_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*%s*.npz' % (exp_id, 'both')))[0],
                    allow_pickle=True)

                # Get "significant" neurons from test Anja performed
                gray_exp_sig_saccade_neurons = exp_data['_sigNeurons'].flatten()
                gray_exp_sig_saccade_neurons = np.array([int(x) for x in gray_exp_sig_saccade_neurons])

                grating_exp_saccade_dir_model_idx = np.where(
                    grating_regression_result['X_sets_names'] == 'saccade'
                )[0][0]

                gray_exp_saccade_dir_model_idx = np.where(
                    gray_regression_result['X_sets_names'] == 'saccade'
                )[0][0]

                # Get the mean saccade temporal and nasal kernel for significant sacacade neurons in gray exp
                if subset_only_better_than_null:

                    gray_exp_saccade_model_idx = np.where(
                        gray_regression_result['X_sets_names'] == gray_exp_model
                    )[0][0]

                    gray_exp_null_model_idx = np.where(
                        gray_regression_result['X_sets_names'] == gray_exp_null_model
                    )[0][0]

                    gray_exp_explained_var_per_X_set = gray_regression_result['explained_var_per_X_set']
                    model_a_explained_var = gray_exp_explained_var_per_X_set[:, gray_exp_saccade_model_idx]
                    model_b_explained_var = gray_exp_explained_var_per_X_set[:, gray_exp_null_model_idx]

                    subset_neuron_idx = np.where(
                        (model_a_explained_var > model_b_explained_var) &
                        (model_a_explained_var > 0)
                    )[0]

                    pdb.set_trace()



                # TEMP: some code here to compare number of saccade on vs. saccade dir neurons
                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(4, 4)
                    explained_var_metric = 'saccade_aligned_var_explained'
                    # This can be 'saccade_aligned_var_explained', or 'explained_var_per_X_set'

                    saccade_on_only_idx = 1
                    saccade_dir_idx = 2

                    max_val = np.max(gray_regression_result[explained_var_metric])
                    min_val = -0.01  # np.min(gray_regression_result[explained_var_metric])

                    exp_saccade_on_model_ev = gray_regression_result[explained_var_metric][:, saccade_on_only_idx]
                    exp_saccade_dir_model_ev = gray_regression_result[explained_var_metric][:, saccade_dir_idx]

                    all_exp_saccade_on_model_ev.extend(exp_saccade_on_model_ev)
                    all_exp_exp_saccade_dir_model_ev.extend(exp_saccade_dir_model_ev)

                    ax.scatter(exp_saccade_on_model_ev,
                               exp_saccade_dir_model_ev,
                               s=8, color='black', lw=0)
                    unity_vals = np.linspace(min_val, max_val)
                    ax.set_xlim([min_val, max_val + 0.01])
                    ax.set_ylim([min_val, max_val + 0.01])

                    ax.plot(unity_vals, unity_vals, linestyle='--', color='gray', alpha=0.5, lw=1)
                    ax.axvline(0, linestyle='--', color='gray', lw=1, alpha=0.5)
                    ax.axhline(0, linestyle='--', color='gray', lw=1, alpha=0.5)

                    ax.set_xlabel('Saccade only model', size=10)
                    ax.set_ylabel('Saccade dir model', size=10)

                    fig_name = '%s_saccade_on_vs_saccade_dir_aligned_var_expalined_in_gray_exp' % exp_id
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                grating_exp_saccade_dir_explained_var = grating_regression_result['explained_var_per_X_set'][:,
                                                        grating_exp_saccade_dir_model_idx]
                gray_exp_saccade_dir_explained_var = gray_regression_result['explained_var_per_X_set'][:,
                                                     gray_exp_saccade_dir_model_idx]

                gray_exp_sig_saccade_dir_neuron_idx = np.where(gray_exp_saccade_dir_explained_var > 0)[0]

                gray_exp_saccade_aligned_full = gray_regression_result['Y_saccade_aligned_full']
                gray_exp_saccade_dir = gray_regression_result['saccade_dirs']
                nasal_idx = np.where(gray_exp_saccade_dir == -1)[0]
                temporal_idx = np.where(gray_exp_saccade_dir == 1)[0]
                gray_regression_kernel_names = gray_regression_result['regression_kernel_names']
                gray_regression_kernel_windows = gray_regression_result['regression_kernel_windows']



                if len(gray_exp_sig_saccade_dir_neuron_idx) > 0:

                    gray_exp_saccade_model_weights = gray_regression_result['model_weights_per_X_set'].item()['saccade']
                    gray_exp_saccade_model_weights_mean = np.mean(gray_exp_saccade_model_weights, axis=0)
                    gray_exp_saccade_model_feature_indices = gray_regression_result['feature_indices_per_X_set'].item()['saccade']

                    features_to_plot = ['bias', 'saccade_on', 'saccade_dir']

                    for neuron_idx in gray_exp_sig_saccade_dir_neuron_idx:
                        
                        # Get the fitted repsonse profile of nasal and temporal saccade
                        saccade_on_indices = gray_exp_saccade_model_feature_indices['saccade_on']
                        saccade_dir_indices = gray_exp_saccade_model_feature_indices['saccade_dir']

                        fitted_saccade_on_response = gray_exp_saccade_model_weights_mean[neuron_idx, saccade_on_indices]
                        fitted_saccade_dir_response = gray_exp_saccade_model_weights_mean[neuron_idx, saccade_dir_indices]
                        fitted_nasal_saccade_response = fitted_saccade_on_response - fitted_saccade_dir_response
                        fitted_temporal_saccade_response = fitted_saccade_on_response + fitted_saccade_dir_response

                        # This assumes same kernel window for saccade_on and saccade_dir which should be true
                        saccade_feat_idx = np.where(gray_regression_kernel_names == 'saccade_on')[0][0]
                        saccade_kernel_window = gray_regression_kernel_windows[saccade_feat_idx]
                        saccade_kernel_t = np.linspace(saccade_kernel_window[0], saccade_kernel_window[-1], len(saccade_on_indices))
                        pre_saccade_idx = np.where(saccade_kernel_t <= 0)[0]
                        post_saccade_idx = np.where(saccade_kernel_t > 0)[0]

                        n_saccade_post_minus_pre = np.mean(fitted_nasal_saccade_response[post_saccade_idx]) - \
                                                   np.mean(fitted_nasal_saccade_response[pre_saccade_idx])

                        t_saccade_post_minus_pre = np.mean(fitted_temporal_saccade_response[post_saccade_idx]) - \
                                                   np.mean(fitted_temporal_saccade_response[pre_saccade_idx])

                        all_exp_sig_saccade_dir_neurons_nasal_postpre_diff.append(n_saccade_post_minus_pre)
                        all_exp_sig_saccade_dir_neurons_temporal_postpre_diff.append(t_saccade_post_minus_pre)


                        # Set to False to speed up things for wodnwstream code
                        make_plots = False

                        if make_plots:
                            with plt.style.context(splstyle.get_style('nature-reviews')):
                                fig, axs = plt.subplots(1, 4, sharey=True)
                                fig.set_size_inches(12, 3)

                                nasal_saccade_mean = np.mean(gray_exp_saccade_aligned_full[nasal_idx, :, neuron_idx], axis=0)
                                temporal_saccade_mean = np.mean(gray_exp_saccade_aligned_full[temporal_idx, :, neuron_idx],
                                                             axis=0)

                                axs[0].plot(nasal_saccade_mean, color='green')
                                axs[0].plot(temporal_saccade_mean, color='purple')

                                for n_feature, feature_name in enumerate(features_to_plot):
                                    feature_indices = gray_exp_saccade_model_feature_indices[feature_name]
                                    if feature_name == 'bias':
                                        axs[n_feature + 1].axhline(
                                            gray_exp_saccade_model_weights_mean[neuron_idx, feature_indices])
                                    else:
                                        feat_idx = np.where(gray_regression_kernel_names == feature_name)[0][0]
                                        kernel_window = gray_regression_kernel_windows[feat_idx]
                                        kernel_t = np.linspace(kernel_window[0], kernel_window[-1], len(feature_indices))
                                        axs[n_feature + 1].plot(kernel_t,
                                                                gray_exp_saccade_model_weights_mean[neuron_idx, feature_indices])

                                fig_name = '%s_neuron_%s_gray_exp_saccade_neuron_regression_weights' % (exp_id, neuron_idx)
                                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                                plt.close(fig)


            # Plot histogram of saccade repsonse magnitude in all experiments
            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
                fig.set_size_inches(6, 3)
                axs[0].hist(all_exp_sig_saccade_dir_neurons_nasal_postpre_diff, lw=0, bins=30, color='green')
                axs[1].hist(all_exp_sig_saccade_dir_neurons_temporal_postpre_diff, lw=0, bins=30, color='purple')
                axs[0].set_ylabel('Number of neurons', size=11)
                axs[0].set_xlabel('Nasal saccade response', size=11)
                axs[1].set_xlabel('Temporal saccade response', size=11)

                num_nasal_diff = len(all_exp_sig_saccade_dir_neurons_nasal_postpre_diff)
                num_nasal_pos = len(np.where(np.array(all_exp_sig_saccade_dir_neurons_nasal_postpre_diff) > 0)[0])
                num_temporal_diff = len(all_exp_sig_saccade_dir_neurons_temporal_postpre_diff)
                num_temporal_pos = len(np.where(np.array(all_exp_sig_saccade_dir_neurons_temporal_postpre_diff) > 0)[0])

                axs[0].set_title('Prop positive response: %.2f' % (num_nasal_pos / num_nasal_diff), size=10)
                axs[1].set_title('Prop positive response: %.2f' % (num_temporal_pos / num_temporal_diff), size=10)

                axs[0].axvline(0, linestyle='--', color='gray', lw=1, alpha=0.5)
                axs[1].axvline(0, linestyle='--', color='gray', lw=1, alpha=0.5)

                fig_name = 'all_exp_gray_saccade_dir_sig_neuron_response_hist'
                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                plt.close(fig)

                # TEMP: some code here to compare number of saccade on vs. saccade dir neurons
                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(4, 4)

                    all_exp_saccade_on_model_ev = np.array(all_exp_saccade_on_model_ev)
                    all_exp_exp_saccade_dir_model_ev = np.array(all_exp_exp_saccade_dir_model_ev)

                    model_ev_vals = np.concatenate([all_exp_saccade_on_model_ev,
                                                    all_exp_exp_saccade_dir_model_ev])

                    max_val = np.max(model_ev_vals)
                    min_val = -0.01  # np.min(gray_regression_result[explained_var_metric])
                    num_saccade_neurons = len(np.where(
                        (all_exp_saccade_on_model_ev > 0) +
                        (all_exp_exp_saccade_dir_model_ev > 0)
                    )[0])

                    num_saccade_dir_neurons = len(np.where(
                        (all_exp_exp_saccade_dir_model_ev > all_exp_saccade_on_model_ev) &
                        (all_exp_exp_saccade_dir_model_ev > 0)
                    )[0])

                    prop_saccade_dir_neurons = num_saccade_dir_neurons / num_saccade_neurons

                    ax.scatter(all_exp_saccade_on_model_ev,
                               all_exp_exp_saccade_dir_model_ev,
                               s=8, color='black', lw=0)
                    unity_vals = np.linspace(min_val, max_val)
                    ax.set_xlim([min_val, max_val + 0.01])
                    ax.set_ylim([min_val, max_val + 0.01])

                    ax.plot(unity_vals, unity_vals, linestyle='--', color='gray', alpha=0.5, lw=1)
                    ax.axvline(0, linestyle='--', color='gray', lw=1, alpha=0.5)
                    ax.axhline(0, linestyle='--', color='gray', lw=1, alpha=0.5)

                    ax.set_title('Prop saccade dir neurons: %.2f' % prop_saccade_dir_neurons, size=11)

                    ax.set_xlabel('Saccade only model', size=10)
                    ax.set_ylabel('Saccade dir model', size=10)

                    fig_name = 'all_exp_saccade_on_vs_saccade_dir_aligned_var_expalined_in_gray_exp'
                    fig.suptitle('All experiments', size=9)
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


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

            plot_aligned_predictions = True

            fig_folder = process_params[process]['fig_folder']
            text_size = 11


            for fpath in regression_result_files:
                regression_result = np.load(fpath, allow_pickle=True)
                exp_id = '_'.join(os.path.basename(fpath).split('_')[0:2])
                X_sets_names = regression_result['X_sets_names']

                explained_var_per_X_set = regression_result['explained_var_per_X_set']
                vis_aligned_explained_var_per_X_set = regression_result['vis_aligned_var_explained']
                saccade_aligned_explained_var_per_X_set = regression_result['saccade_aligned_var_explained']

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

                    model_a_vis_aligned_explained_var = vis_aligned_explained_var_per_X_set[:, model_a_idx]
                    model_b_vis_aligned_explained_var = vis_aligned_explained_var_per_X_set[:, model_b_idx]

                    model_a_saccade_aligned_explained_var = saccade_aligned_explained_var_per_X_set[:, model_a_idx]
                    model_b_saccade_aligned_explained_var = saccade_aligned_explained_var_per_X_set[:, model_b_idx]

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

                            if plot_aligned_predictions:
                                gs = fig.add_gridspec(4, 4)
                                ax1 = fig.add_subplot(gs[0, 0])
                                ax2 = fig.add_subplot(gs[0, 2])
                                ax3 = fig.add_subplot(gs[0, 3])
                                ax4 = fig.add_subplot(gs[1, :])
                                ax5 = fig.add_subplot(gs[2, 0])
                                ax6 = fig.add_subplot(gs[2, 1])
                                ax7 = fig.add_subplot(gs[2, 2])
                                ax8 = fig.add_subplot(gs[2, 3])
                                fig.set_size_inches(10, 10)

                            else:
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
                        ax4.plot(model_a_Y_hat[:, neuron_idx], label='%s' % model_a, lw=1, color='orange')
                        ax4.plot(model_b_Y_hat[:, neuron_idx], label='%s' % model_b, lw=1, color='purple')
                        ax4.legend()


                        # Aligned explained variance
                        if plot_aligned_predictions:

                            # Saccade aligned explained variance
                            ax5.axvline(0, linestyle='--', color='gray', lw=0.5)
                            ax5.axhline(0, linestyle='--', color='gray', lw=0.5)

                            both_model_explained_var = np.concatenate([model_a_saccade_aligned_explained_var, model_b_saccade_aligned_explained_var])
                            both_min = np.min(both_model_explained_var)
                            both_max = np.max(both_model_explained_var)
                            unity_vals = np.linspace(both_min, both_max, 100)

                            ax5.plot(unity_vals, unity_vals, linestyle='--', color='gray', lw=0.5)

                            ax5.scatter(model_a_saccade_aligned_explained_var, model_b_saccade_aligned_explained_var, color='black', s=10)
                            ax5.scatter(model_a_saccade_aligned_explained_var[neuron_idx], model_b_saccade_aligned_explained_var[neuron_idx],
                                        color='red', s=10)
                            ax5.set_xlabel(model_a, size=11)
                            ax5.set_ylabel(model_b, size=11)
                            ax5.set_xlim([both_min, both_max])
                            ax5.set_ylim([both_min, both_max])
                            ax5.set_title('Saccade aligned EV', size=11)

                            # Vis aligned explained variance
                            ax6.axvline(0, linestyle='--', color='gray', lw=0.5)
                            ax6.axhline(0, linestyle='--', color='gray', lw=0.5)

                            both_model_explained_var = np.concatenate(
                                [model_a_vis_aligned_explained_var, model_b_vis_aligned_explained_var])
                            both_min = np.min(both_model_explained_var)
                            both_max = np.max(both_model_explained_var)
                            unity_vals = np.linspace(both_min, both_max, 100)

                            ax6.plot(unity_vals, unity_vals, linestyle='--', color='gray', lw=0.5)

                            ax6.scatter(model_a_vis_aligned_explained_var, model_b_vis_aligned_explained_var,
                                        color='black', s=10)
                            ax6.scatter(model_a_vis_aligned_explained_var[neuron_idx],
                                        model_b_vis_aligned_explained_var[neuron_idx],
                                        color='red', s=10)
                            ax6.set_xlabel(model_a, size=11)
                            ax6.set_ylabel(model_b, size=11)
                            ax6.set_xlim([both_min, both_max])
                            ax6.set_ylim([both_min, both_max])
                            ax6.set_title('Vis aligned EV', size=11)

                            # Model prediction for vis aligned activity
                            Y_hat_vis_aligned = regression_result['Y_hat_vis_aligned']
                            Y_vis_aligned = regression_result['Y_vis_aligned']

                            neuron_y_vis_aligned = Y_vis_aligned[0, :, :, neuron_idx]  # model x trial x time x neuron
                            neuron_y_vis_aligned_mean = np.mean(neuron_y_vis_aligned, axis=0)

                            model_a_yhat_vis_aligned = Y_hat_vis_aligned[model_a_idx, :, :, neuron_idx]
                            model_a_yhat_vis_aligned_mean = np.mean(model_a_yhat_vis_aligned, axis=0)
                            model_b_yhat_vis_aligned = Y_hat_vis_aligned[model_b_idx, :, :, neuron_idx]
                            model_b_yhat_vis_aligned_mean = np.mean(model_b_yhat_vis_aligned, axis=0)

                            ax7.plot(neuron_y_vis_aligned_mean, color='black', label='Original')
                            ax7.plot(model_a_yhat_vis_aligned_mean, color='orange', label='%s' % model_a)
                            ax7.plot(model_b_yhat_vis_aligned_mean, color='purple', label='%s' % model_b)
                            ax7.set_title('Vis aligned', size=11)

                            # Model prediction for saccade aligned activity
                            Y_hat_saccade_aligned = regression_result['Y_hat_saccade_aligned']
                            Y_saccade_aligned = regression_result['Y_saccade_aligned']

                            neuron_y_saccade_aligned = Y_saccade_aligned[0, :, :, neuron_idx]
                            neuron_y_saccade_aligned_mean = np.mean(neuron_y_saccade_aligned, axis=0)

                            model_a_yhat_saccade_aligned = Y_hat_saccade_aligned[model_a_idx, :, :, neuron_idx]
                            model_a_yhat_saccade_aligned_mean = np.mean(model_a_yhat_saccade_aligned, axis=0)
                            model_b_yhat_saccade_aligned = Y_hat_saccade_aligned[model_b_idx, :, :, neuron_idx]
                            model_b_yhat_saccade_aligned_mean = np.mean(model_b_yhat_saccade_aligned, axis=0)


                            ax8.plot(neuron_y_saccade_aligned_mean, color='black', label='Original')
                            ax8.plot(model_a_yhat_saccade_aligned_mean, color='orange', label='%s' % model_a)
                            ax8.plot(model_b_yhat_saccade_aligned_mean, color='purple', label='%s' % model_b)
                            ax8.set_title('Saccade aligned', size=11)

                        fig.suptitle('%s neuron %s' % (exp_id, neuron_idx), size=11)
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
                explained_var_unity_vals = np.linspace(both_min, both_max, 100)

                # Get visual aligned activity of all neurons
                vis_aligned_activity, vis_ori, saccade_dir_during_vis, saccade_time, time_windows = get_aligned_activity(
                    exp_data, exp_type='grating', aligned_event='vis',
                    alignment_time_window=[-1, 3],
                    exclude_saccade_on_vis_exp=False)

                # Get saccade aligned activity of all neurons
                saccade_aligned_activity, saccade_trial_type, saccade_aligned_time_window, vis_ori_during_saccade = get_aligned_activity(
                    exp_data, exp_type='grating', aligned_event='saccade',
                    alignment_time_window=[-1, 1],
                    exclude_saccade_on_vis_exp=False,
                    return_vis_ori_for_saccade=True)

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

                saccade_dir_colors = {
                    -1: 'green',  # nasal
                    1: 'purple',  # temporal
                }

                for neuron_idx in b_minus_a_sort_idx[0:num_neurons_to_plot]:

                    # Plot the explained variance and individual trials
                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, axs = plt.subplots(1, 3)
                        fig.set_size_inches(9, 6)

                        # Plot explained variance
                        axs[0].axvline(0, linestyle='--', color='gray', lw=1, alpha=0.3)
                        axs[0].axhline(0, linestyle='--', color='gray', lw=1, alpha=0.3)
                        axs[0].plot(explained_var_unity_vals, explained_var_unity_vals, linestyle='--', lw=1, color='gray', alpha=0.3)
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
                            y_vals = y_offset + trial_trace
                            axs[1].plot(time_windows, y_vals, color=color, lw=1)
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

                    # Plot the stimulus and saccade triggered average of the neurons
                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, axs = plt.subplots(2, len(vis_ori_subset), sharey=True, sharex=True)
                        fig.set_size_inches(12, 3)

                        for n_ori, ori_to_plot in enumerate(vis_ori_subset):
                            trial_idx_within_saccade_off = np.where(vis_ori_saccade_off == ori_to_plot)[0]
                            mean_ori_activity = np.mean(vis_aligned_activity_saccade_off[trial_idx_within_saccade_off, :, neuron_idx], axis=0)
                            baseline_time_window_idx = np.where((time_windows >= -0.5) & (time_windows < 0))[0]
                            baseline_activity = mean_ori_activity[baseline_time_window_idx]
                            axs[0, n_ori].plot(time_windows, mean_ori_activity - np.mean(baseline_activity), color=grating_colors[n_ori], lw=1)

                            # Plot saccade triggered average (split by saccade direction)
                            # saccade_trials_w_ori = np.where(vis_ori_during_saccade == ori_to_plot)[0]  # -1 means no visual trial (gray screen) at all, NaN means blank is presented (also gray screen)
                            saccade_trials_w_ori_nasal = np.where(
                                (vis_ori_during_saccade == ori_to_plot) &
                                (saccade_trial_type == -1)
                            )[0]

                            saccade_trials_w_ori_temporal = np.where(
                                (vis_ori_during_saccade == ori_to_plot) &
                                (saccade_trial_type == 1)
                            )[0]

                            if len(saccade_trials_w_ori_nasal) != 0:
                                saccade_activity_for_ori = saccade_aligned_activity[saccade_trials_w_ori_nasal, :, neuron_idx]
                                mean_saccade_activity_for_ori = np.mean(saccade_activity_for_ori, axis=0)
                                saccade_baseline_time_window_idx = np.where((saccade_aligned_time_window >= -0.5) & (saccade_aligned_time_window < 0))[0]
                                baseline_saccade_activity_for_ori = np.mean(mean_saccade_activity_for_ori[saccade_baseline_time_window_idx])
                                axs[1, n_ori].plot(saccade_aligned_time_window, mean_saccade_activity_for_ori - baseline_saccade_activity_for_ori, color=saccade_dir_colors[-1], lw=1)
                            if len(saccade_trials_w_ori_temporal) != 0:
                                saccade_activity_for_ori = saccade_aligned_activity[saccade_trials_w_ori_temporal, :, neuron_idx]
                                mean_saccade_activity_for_ori = np.mean(saccade_activity_for_ori, axis=0)
                                saccade_baseline_time_window_idx = np.where((saccade_aligned_time_window >= -0.5) & (saccade_aligned_time_window < 0))[0]
                                baseline_saccade_activity_for_ori = np.mean(mean_saccade_activity_for_ori[saccade_baseline_time_window_idx])
                                axs[1, n_ori].plot(saccade_aligned_time_window, mean_saccade_activity_for_ori - baseline_saccade_activity_for_ori, color=saccade_dir_colors[1], lw=1)

                            axs[0, n_ori].axvline(0, linestyle='--', lw=0.5, color='gray', alpha=0.3)
                            axs[1, n_ori].axvline(0, linestyle='--', lw=0.5, color='gray', alpha=0.3)
                            axs[0, n_ori].set_title('ori = %.f' % ori_to_plot, size=11)

                        fig.text(0.5, 0.4, 'Visual aligned time (s)', ha='center', va='center', size=11)
                        fig.text(0.5, 0, 'Saccade aligned time (s)', ha='center',  size=11)
                        fig.text(0.0, 0.5, 'Neural activity', ha='center', va='center', rotation=90, size=11)

                        fig.suptitle('%s neuron %.f' % (exp_id, neuron_idx), size=11)
                        fig_name = '%s_neuron_%.f_ori_triggered_average_and_saccade_triggered_average' % (exp_id, neuron_idx)
                        fig.tight_layout()
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                        plt.close(fig)

                    # Plot scatter of vis only vs. vis + saccade activity in particular time windows
                    vis_vs_vis_plus_saccade_window_width = [0, 0.5]
                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, ax = plt.subplots()
                        fig.set_size_inches(4, 4)

                        neuron_vis_aligned_activity_saccade_off = vis_aligned_activity[:, :, neuron_idx]

                        all_trial_vis_only_activity = []
                        all_trials_vis_plus_saccade_activity = []

                        for n_trial, saccade_on_trial_idx in enumerate(saccade_on_trials):
                            trial_vis_ori = vis_ori_saccade_on[n_trial]

                            if np.isnan(trial_vis_ori):
                                continue

                            dot_color = grating_colors[np.where(vis_ori_subset == trial_vis_ori)[0][0]]
                            saccade_off_ori_trials = np.where(vis_ori_saccade_off == trial_vis_ori)[0]  # get trials in saccade off condition with same orientation as this trial
                            neuron_vis_aligned_activity_saccade_on = vis_aligned_activity_saccade_on[n_trial, :, neuron_idx]

                            saccade_dir_during_vis_in_this_trial = saccade_dir_saccade_on_trials[n_trial]

                            for n_trial_saccade, trial_saccade_times in enumerate(saccade_time_saccade_on_trials[n_trial]):

                                if trial_saccade_times > 0:
                                    time_window_idx = np.where(
                                        (time_windows >= trial_saccade_times + vis_vs_vis_plus_saccade_window_width[0]) &
                                        (time_windows <= trial_saccade_times + vis_vs_vis_plus_saccade_window_width[1])
                                    )[0]

                                    neuron_vis_only_activity = np.mean(neuron_vis_aligned_activity_saccade_off[saccade_off_ori_trials, :][:, time_window_idx])
                                    neuron_vis_plus_saccade_activity = np.mean(neuron_vis_aligned_activity_saccade_on[time_window_idx])

                                    saccade_dir_within_trial = saccade_dir_during_vis_in_this_trial[n_trial_saccade]

                                    if saccade_dir_within_trial == -1:
                                        marker = '<'  # nasal
                                    elif saccade_dir_within_trial == 1:
                                        marker = '>'  # temporal

                                    all_trial_vis_only_activity.append(neuron_vis_only_activity)
                                    all_trials_vis_plus_saccade_activity.append(neuron_vis_plus_saccade_activity)

                                    # TOOD: may speed up code a bit if this is outside of a loop (prevent replotting)
                                    ax.scatter(neuron_vis_only_activity,
                                               neuron_vis_plus_saccade_activity,
                                               color=dot_color,
                                               marker=marker)

                        all_vals = np.concatenate([all_trial_vis_only_activity, all_trials_vis_plus_saccade_activity])
                        all_min = np.min(all_vals)
                        all_max = np.max(all_vals)
                        offset = 0.5
                        ax.set_xlim([all_min - offset, all_max + offset])
                        ax.set_ylim([all_min - offset, all_max + offset])
                        unity_vals = np.linspace(all_min, all_max, 100)
                        ax.plot(unity_vals, unity_vals, linestyle='--', color='gray', alpha=0.3)
                        ax.set_xlabel('Vis only activity at time window', size=11)
                        ax.set_ylabel('Vis + saccade activity at time window', size=11)
                        ax.set_title('%s neuron %.f' % (exp_id, neuron_idx), size=11)
                        fig_name = '%s_neuron_%.f_vis_vs_vis_plus_saccade_at_small_time_window' % (exp_id, neuron_idx)
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                        plt.close(fig)

                # Get the prefer orientation of all neurons
                if process_params[process]['plot_prefer_orientation']:
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

        if process == 'fit_saccade_ori_tuning_curves':
            regression_results_folder = process_params[process]['regression_results_folder']
            save_folder = process_params[process]['save_folder']
            neurons_to_fit = process_params[process]['neurons_to_fit']

            # plot_variance_explained_comparison = process_params[process]['plot_variance_explained_comparison']
            grating_exp_model = process_params[process]['grating_exp_model']
            gray_exp_model = process_params[process]['gray_exp_model']
            # metric_to_plot = 'explained_var_per_X_set'
            metric_to_plot = 'saccade_aligned_var_explained'
            do_imputation = process_params[process]['do_imputation']
            run_parallel = process_params[process]['run_parallel']

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

                X_sets_names = regression_result['X_sets_names']
                explained_var_per_X_set = regression_result['explained_var_per_X_set']

                model_a = 'vis_ori'
                model_b = 'vis_and_saccade'

                model_a_idx = np.where(X_sets_names == model_a)[0][0]
                model_b_idx = np.where(X_sets_names == model_b)[0][0]

                model_a_explained_var = explained_var_per_X_set[:, model_a_idx]
                model_b_explained_var = explained_var_per_X_set[:, model_b_idx]

                b_minus_a = model_b_explained_var - model_a_explained_var

                if neurons_to_fit == 'vis_and_saccade':
                    subset_idx = np.where(
                        (model_a_explained_var > 0) &
                        (model_b_explained_var > 0) &
                        (model_b_explained_var > model_a_explained_var)
                    )[0]
                elif neurons_to_fit == 'all':
                    # fit all neurons
                    subset_idx = np.arange(len(model_a_explained_var))
                elif neurons_to_fit == 'vis':
                    subset_idx = np.where(
                        (model_a_explained_var > 0)
                    )[0]

                vis_subset_idx = np.where(
                    (model_a_explained_var > 0)
                )[0]

                b_minus_a_subset_sort_idx = np.argsort(b_minus_a[subset_idx])[::-1]
                b_minus_a_sort_idx = subset_idx[b_minus_a_subset_sort_idx]

                both_model_explained_var = np.concatenate([model_a_explained_var, model_b_explained_var])
                both_min = np.min(both_model_explained_var)
                both_max = np.max(both_model_explained_var)
                explained_var_unity_vals = np.linspace(both_min, both_max, 100)

                # Get visual aligned activity of all neurons
                vis_aligned_activity, vis_ori, saccade_dir_during_vis, saccade_time, time_windows = get_aligned_activity(
                    exp_data, exp_type='grating', aligned_event='vis',
                    alignment_time_window=[-1, 3],
                    exclude_saccade_on_vis_exp=False)

                # Get saccade aligned activity of all neurons
                saccade_aligned_activity, saccade_trial_type, saccade_aligned_time_window, vis_ori_during_saccade = get_aligned_activity(
                    exp_data, exp_type='grating', aligned_event='saccade',
                    alignment_time_window=[-1, 1],
                    exclude_saccade_on_vis_exp=False,
                    return_vis_ori_for_saccade=True)

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

                saccade_dir_colors = {
                    -1: 'green',  # nasal
                    1: 'purple',  # temporal
                }

                # Output is of shape (Trial, Ori, Neurons)
                saccade_off_ori_activity, nasal_saccade_ori_activity, temporal_saccade_ori_activity, ori_groups, all_unique_ori = \
                    get_ori_grouped_vis_and_saccade_activity(vis_aligned_activity, saccade_on_trials,
                                                             saccade_off_trials,
                                                             vis_ori_saccade_on, vis_ori_saccade_off,
                                                             saccade_dir_saccade_on_trials,
                                                             saccade_time_saccade_on_trials, time_windows,
                                                             window_width=[0, 0.5], vis_tuning_window=[0, 1])

                plot_fits = True
                include_explained_variance = True
                include_error_bars = True
                scale_data = False  # whether to scale the visual response (not needed for von-mesis2 function)

                von_mises_explained_variance = np.zeros((len(b_minus_a_sort_idx), 3)) + np.nan
                von_mises_fitted_loc = np.zeros((len(b_minus_a_sort_idx), 3)) + np.nan
                num_params = 5
                von_mises_fitted_params = np.zeros((len(b_minus_a_sort_idx), num_params, 3)) + np.nan
                vis_condition_names = ['vis_only', 'nasal', 'temporal']
                num_vis_conditions = len(vis_condition_names)
                loo_neuron_mse = np.zeros((len(b_minus_a_sort_idx), num_vis_conditions)) + np.nan
                loo_neuron_variance_explained = np.zeros((len(b_minus_a_sort_idx), num_vis_conditions)) + np.nan
                loo_neuron_trial_variance_explained = np.zeros((len(b_minus_a_sort_idx), num_vis_conditions)) + np.nan
                neuron_mean_responses = np.zeros((len(b_minus_a_sort_idx), num_vis_conditions, 12)) + np.nan
                neuron_response_std = np.zeros((len(b_minus_a_sort_idx), num_vis_conditions, 12)) + np.nan

                exp_run_start_time = time.time()

                if run_parallel:
                    print('Running things in parallel')
                    ray.shutdown()
                    ray.init()

                    for n_vis_response, vis_response_name in enumerate(vis_condition_names):
                        if vis_response_name == 'vis_only':
                            vis_response = saccade_off_ori_activity
                        elif vis_response_name == 'nasal':
                            vis_response = nasal_saccade_ori_activity
                        elif vis_response_name == 'temporal':
                            vis_response = temporal_saccade_ori_activity
                            
                        # Get neuron initial guess
                        # kappa guess : RP * (2 - rp^2) / (1 - rp^2)
                        # Peak Bounds: to 5 times
                        vis_response_subset = vis_response[:, :, b_minus_a_sort_idx]
                        vis_response_subset_mean = np.nanmean(vis_response_subset, axis=0)  # mean across trials

                        vis_max_idx = np.argmax(vis_response_subset_mean, axis=0)

                        vis_max_response = np.nanmax(vis_response_subset_mean, axis=0)
                        vis_min_response = np.nanmin(vis_response_subset_mean, axis=0)

                        neuron_init_params = np.zeros((len(b_minus_a_sort_idx), 5))
                        neuron_init_params[:, 0] = all_unique_ori[vis_max_idx]  # preferred orientation from max response
                        neuron_init_params[:, 1] = vis_max_response - vis_min_response  # first peak
                        neuron_init_params[:, 2] = (vis_max_response - vis_min_response) / 2  # second peak
                        neuron_init_params[:, 3] = vis_min_response   # baseline
                        neuron_init_params[:, 4] = 20  # width


                        # For each visual condition, loop through each neuron
                        fit_vis_ori_ids = [fit_vis_ori_response.remote(vis_response[:, :, neuron_idx], neuron_init_params[idx, :])
                                         for neuron_idx, idx in zip(b_minus_a_sort_idx, np.arange(len(b_minus_a_sort_idx)))]


                        fit_vis_ori_results = ray.get(fit_vis_ori_ids)

                        # Probably there is a smarter way to unpack this...
                        for n_neuron in np.arange(len(fit_vis_ori_results)):
                            loo_mse, loo_predictions_mean, vis_response_matrix_mean, loo_variance_explained, trial_by_trial_variance_explained = \
                                fit_vis_ori_results[n_neuron]
                            loo_neuron_mse[n_neuron, n_vis_response] = loo_mse
                            loo_neuron_variance_explained[n_neuron, n_vis_response] = loo_variance_explained
                            loo_neuron_trial_variance_explained[n_neuron, n_vis_response] = trial_by_trial_variance_explained
                            # neuron_mean_responses[n_neuron, n_vis_response, :] = loo_predictions_mean

                    ray.shutdown()

                    # Do the usual of fitting to all the data
                    for n_neuron, neuron_idx in enumerate(b_minus_a_sort_idx):
                        # Do leave one out cross-validation
                        neuron_vis_only_trial_mean = np.nanmean(saccade_off_ori_activity[:, :, neuron_idx], axis=0)
                        neuron_nasal_saccade_trial_mean = np.nanmean(nasal_saccade_ori_activity[:, :, neuron_idx],
                                                                     axis=0)
                        neuron_temporal_saccade_trial_mean = np.nanmean(temporal_saccade_ori_activity[:, :, neuron_idx],
                                                                        axis=0)

                        neuron_vis_only_trial_std = np.nanstd(saccade_off_ori_activity[:, :, neuron_idx], axis=0)
                        neuron_nasal_saccade_trial_std = np.nanstd(nasal_saccade_ori_activity[:, :, neuron_idx], axis=0)
                        neuron_temporal_saccade_trial_std = np.nanstd(temporal_saccade_ori_activity[:, :, neuron_idx],
                                                                      axis=0)

                        vis_response_to_fit = [neuron_vis_only_trial_mean, neuron_nasal_saccade_trial_mean,
                                               neuron_temporal_saccade_trial_mean]
                        vis_response_error = [neuron_vis_only_trial_std, neuron_nasal_saccade_trial_std,
                                              neuron_temporal_saccade_trial_std]
                        try:
                            neuron_mean_responses[n_neuron, :, :] = np.array(vis_response_to_fit)
                            neuron_response_std[n_neuron, :, :] = np.array(vis_response_error)
                        except:
                            pdb.set_trace()
                        
                        for n_vis_response, vis_response in enumerate(vis_response_to_fit):

                            xdata, ydata, ydata_error = transform_ori_data_for_vonmises2(vis_response,
                                                                                         vis_response_error=
                                                                                         vis_response_error[
                                                                                             n_vis_response],
                                                                                         scale_data=scale_data)

                            try:
                                initial_guess = neuron_init_params[n_neuron, :]
                                param_bounds = (
                                (0, 0, 0, -np.inf, 5),  # the '2' here is the spread, roughly about 30 degrees here
                                (360, np.inf, np.inf, np.inf, 100))
                                # fitted_params, _ = spopt.curve_fit(von_mises2, xdata, ydata, p0=initial_guess, maxfev=2000)
                                fitted_params, _ = spopt.curve_fit(von_mises2, xdata, ydata, p0=initial_guess,
                                                                   bounds=param_bounds,
                                                                   maxfev=50000)
                                fit_success = True
                            except:
                                print('Curve fitting failed')
                                fit_success = False
                                pdb.set_trace()

                            if fit_success:
                                xdata_interpolated = np.linspace(xdata[0], xdata[-1], 100)
                                # ydata_predicted = von_mises2(xdata, fitted_params[0], fitted_params[1])
                                # ydata_interpolated = von_mises2(xdata_interpolated, fitted_params[0], fitted_params[1])
                                ydata_predicted = von_mises2(xdata, fitted_params[0], fitted_params[1],
                                                             fitted_params[2],
                                                             fitted_params[3], fitted_params[4])
                                ydata_interpolated = von_mises2(xdata_interpolated, fitted_params[0],
                                                                fitted_params[1], fitted_params[2],
                                                                fitted_params[3],
                                                                fitted_params[4])

                                von_mises_explained_variance[n_neuron, n_vis_response] = \
                                    sklmetrics.explained_variance_score(ydata, ydata_predicted)
                                von_mises_fitted_loc[n_neuron, n_vis_response] = fitted_params[0]
                                von_mises_fitted_params[n_neuron, :, n_vis_response] = fitted_params

                else:

                    for n_neuron, neuron_idx in enumerate(b_minus_a_sort_idx):
    
    
                        # Do leave one out cross-validation
                        neuron_vis_only_trial_mean = np.nanmean(saccade_off_ori_activity[:, :, neuron_idx], axis=0)
                        neuron_nasal_saccade_trial_mean = np.nanmean(nasal_saccade_ori_activity[:, :, neuron_idx],
                                                                     axis=0)
                        neuron_temporal_saccade_trial_mean = np.nanmean(temporal_saccade_ori_activity[:, :, neuron_idx],
                                                                        axis=0)
    
                        neuron_vis_only_trial_std = np.nanstd(saccade_off_ori_activity[:, :, neuron_idx], axis=0)
                        neuron_nasal_saccade_trial_std = np.nanstd(nasal_saccade_ori_activity[:, :, neuron_idx], axis=0)
                        neuron_temporal_saccade_trial_std = np.nanstd(temporal_saccade_ori_activity[:, :, neuron_idx],
                                                                      axis=0)
    
                        vis_response_to_fit = [neuron_vis_only_trial_mean, neuron_nasal_saccade_trial_mean,
                                               neuron_temporal_saccade_trial_mean]
                        vis_response_error = [neuron_vis_only_trial_std, neuron_nasal_saccade_trial_std,
                                               neuron_temporal_saccade_trial_std]
    
                        neuron_mean_responses[n_neuron, :, :] = np.array(vis_response_to_fit)
                        neuron_response_std[n_neuron, :, :] = np.array(vis_response_error)
    
                        for n_vis_response, vis_response_name in enumerate(vis_condition_names):
    
                            if vis_response_name == 'vis_only':
                                vis_response_matrix = saccade_off_ori_activity[:, :, neuron_idx]
                            elif vis_response_name == 'nasal':
                                vis_response_matrix = nasal_saccade_ori_activity[:, :, neuron_idx]
                            elif vis_response_name == 'temporal':
                                vis_response_matrix = temporal_saccade_ori_activity[:, :, neuron_idx]
    
                            # do leave-one-out
                            initial_guess = [180, 1, 1, 1, 5]
                            param_bounds = ([0, 0, 0, -np.inf, 5],  # the '2' here is the spread, roughly about 30 degrees here
                                            [360, np.inf, np.inf, np.inf, 100])
                            num_valid_trials = np.sum(~np.isnan(vis_response_matrix))
                            loo_errors = np.zeros((num_valid_trials, )) + np.nan
                            loo_predictions = np.zeros(np.shape(vis_response_matrix)) + np.nan
                            x_start = 0
                            x_end = 330
                            n_valid_trial = 0
                            start_time = time.time()
                            for ori_idx in np.arange(np.shape(vis_response_matrix)[1]):
                                for trial_idx in np.arange(np.shape(vis_response_matrix)[0]):
                                    if ~np.isnan(vis_response_matrix[trial_idx, ori_idx]):
    
                                        loo_vis_response_matrix = vis_response_matrix.copy()
                                        loo_vis_response_matrix[trial_idx, ori_idx] = np.nan  # set to NaN to exclude when calculating mean
                                        loo_vis_response_mean = np.nanmean(loo_vis_response_matrix, axis=0)  # mean across trials
                                        xdata, ydata = transform_ori_data_for_vonmises2(loo_vis_response_mean,
                                                                                                     vis_response_error=None,
                                                                                                     scale_data=scale_data)
                                        fitted_params, _ = spopt.curve_fit(von_mises2, xdata, ydata, p0=initial_guess,
                                                                           bounds=param_bounds,
                                                                           maxfev=50000)  # method='dogbox'
                                        xdata_interpolated = np.arange(x_start, x_end+1, 30)
                                        # ydata_predicted = von_mises2(xdata, fitted_params[0], fitted_params[1])
                                        # ydata_interpolated = von_mises2(xdata_interpolated, fitted_params[0], fitted_params[1])
                                        ydata_predicted = von_mises2(xdata_interpolated, fitted_params[0], fitted_params[1],
                                                                     fitted_params[2],
                                                                     fitted_params[3], fitted_params[4])
    
                                        loo_value = vis_response_matrix[trial_idx, ori_idx]
                                        loo_ori_prediction = ydata_predicted[ori_idx]
                                        loo_squared_diff = (loo_value - loo_ori_prediction) ** 2
                                        loo_errors[n_valid_trial] = loo_squared_diff
                                        n_valid_trial += 1
    
                                        loo_predictions[trial_idx, ori_idx] = loo_ori_prediction
    
                            end_time = time.time()
                            print('Took %.2f seconds to do loo on %.f trials' % (end_time - start_time, n_valid_trial))
                            loo_neuron_mse[n_neuron, n_vis_response] = np.mean(loo_errors)

                            # Variance explained after averaging over trials with some orientation
                            loo_predictions_mean = np.nanmean(loo_predictions, axis=0)
                            vis_response_matrix_mean = np.nanmean(vis_response_matrix, axis=0)

                            # remove missing orientations
                            loo_predictions_mean = loo_predictions_mean[~np.isnan(loo_predictions_mean)]
                            vis_response_matrix_mean = vis_response_matrix_mean[~np.isnan(vis_response_matrix_mean)]

                            loo_neuron_variance_explained[n_neuron, n_vis_response] = sklmetrics.explained_variance_score(vis_response_matrix_mean, loo_predictions_mean)
    
                        for n_vis_response, vis_response in enumerate(vis_response_to_fit):
    
                            xdata, ydata, ydata_error = transform_ori_data_for_vonmises2(vis_response,
                                                                                         vis_response_error=
                                                                                         vis_response_error[
                                                                                             n_vis_response],
                                                                                         scale_data=scale_data)
    
                            try:
                                initial_guess = [180, 1, 1, 1, 1]
                                # fitted_params, _ = spopt.curve_fit(von_mises2, xdata, ydata, p0=initial_guess, maxfev=2000)
                                fitted_params, _ = spopt.curve_fit(von_mises2, xdata, ydata, p0=initial_guess,
                                                                   bounds=([0, 0, 0, -np.inf, 0.01],
                                                                           [360, np.inf, np.inf, np.inf, np.inf]),
                                                                   maxfev=50000)
                                fit_success = True
                            except:
                                print('Curve fitting failed')
                                fit_success = False
                                pdb.set_trace()
    
    
                            if fit_success:
                                xdata_interpolated = np.linspace(xdata[0], xdata[-1], 100)
                                # ydata_predicted = von_mises2(xdata, fitted_params[0], fitted_params[1])
                                # ydata_interpolated = von_mises2(xdata_interpolated, fitted_params[0], fitted_params[1])
                                ydata_predicted = von_mises2(xdata, fitted_params[0], fitted_params[1],
                                                             fitted_params[2],
                                                             fitted_params[3], fitted_params[4])
                                ydata_interpolated = von_mises2(xdata_interpolated, fitted_params[0],
                                                                fitted_params[1], fitted_params[2],
                                                                fitted_params[3],
                                                                fitted_params[4])
    
                                von_mises_explained_variance[n_neuron, n_vis_response] = \
                                    sklmetrics.explained_variance_score(ydata, ydata_predicted)
                                von_mises_fitted_loc[n_neuron, n_vis_response] = fitted_params[0]
                                von_mises_fitted_params[n_neuron, :, n_vis_response] = fitted_params

                exp_run_end_time = time.time()
                exp_run_elapsed_time = exp_run_end_time - exp_run_start_time
                # save the fits
                save_name = '%s_%s_von_mises_model_fit_results.npz' % (exp_id, neurons_to_fit)
                save_path = os.path.join(save_folder, save_name)
                np.savez(save_path, von_mises_fitted_params=von_mises_fitted_params,
                         loo_neuron_mse=loo_neuron_mse,
                         loo_neuron_variance_explained=loo_neuron_variance_explained,
                         loo_neuron_trial_variance_explained=loo_neuron_trial_variance_explained,
                         neuron_mean_responses=neuron_mean_responses,
                         neuron_response_std=neuron_response_std,
                         conditions=np.array(['vis_only', 'nasal', 'temporal']),
                         exp_run_elapsed_time=exp_run_elapsed_time)


        if process == 'plot_saccade_ori_tuning_curves':

            regression_results_folder = process_params[process]['regression_results_folder']
            ori_fit_results_folder = '/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression/vis_and_saccade_neurons/ori-tuning-curves'
            fig_folder = process_params[process]['fig_folder']

            if not os.path.isdir(fig_folder):
                os.makedirs(fig_folder)

            plot_variance_explained_comparison = process_params[process]['plot_variance_explained_comparison']
            grating_exp_model = process_params[process]['grating_exp_model']
            gray_exp_model = process_params[process]['gray_exp_model']
            # metric_to_plot = 'explained_var_per_X_set'
            metric_to_plot = 'saccade_aligned_var_explained'
            do_imputation = process_params[process]['do_imputation']

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

                ori_fit_results = np.load(
                    os.path.join(ori_fit_results_folder, '%s_von_mises_model_fit_results.npz' % exp_id)
                )

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
                explained_var_unity_vals = np.linspace(both_min, both_max, 100)

                # Get visual aligned activity of all neurons
                vis_aligned_activity, vis_ori, saccade_dir_during_vis, saccade_time, time_windows = get_aligned_activity(
                    exp_data, exp_type='grating', aligned_event='vis',
                    alignment_time_window=[-1, 3],
                    exclude_saccade_on_vis_exp=False)

                # Get saccade aligned activity of all neurons
                saccade_aligned_activity, saccade_trial_type, saccade_aligned_time_window, vis_ori_during_saccade = get_aligned_activity(
                    exp_data, exp_type='grating', aligned_event='saccade',
                    alignment_time_window=[-1, 1],
                    exclude_saccade_on_vis_exp=False,
                    return_vis_ori_for_saccade=True)

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

                saccade_dir_colors = {
                    -1: 'green',  # nasal
                    1: 'purple',  # temporal
                }

                saccade_off_ori_activity, nasal_saccade_ori_activity, temporal_saccade_ori_activity, ori_groups, all_unique_ori = \
                    get_ori_grouped_vis_and_saccade_activity(vis_aligned_activity, saccade_on_trials, saccade_off_trials,
                                         vis_ori_saccade_on, vis_ori_saccade_off, saccade_dir_saccade_on_trials,
                                         saccade_time_saccade_on_trials, time_windows, window_width=[0, 0.5], vis_tuning_window=[0, 1])

                plot_fits = True
                include_explained_variance = True
                include_error_bars = True
                scale_data = False  # whether to scale the visual response (not needed for von-mesis2 function)

                von_mises_explained_variance = np.zeros((len(b_minus_a_sort_idx), 3)) + np.nan
                von_mises_fitted_loc = np.zeros((len(b_minus_a_sort_idx), 3)) + np.nan
                num_params = 5
                von_mises_fitted_params = np.zeros((len(b_minus_a_sort_idx), num_params, 3)) + np.nan

                for n_neuron, neuron_idx in enumerate(b_minus_a_sort_idx):
                    # Plot the explained variance and individual trials
                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, axs = plt.subplots(3, 1, sharex=True)
                        fig.set_size_inches(6, 6)

                        neuron_vis_only_trial_mean = np.nanmean(saccade_off_ori_activity[:, :, neuron_idx], axis=0)
                        neuron_nasal_saccade_trial_mean = np.nanmean(nasal_saccade_ori_activity[:, :, neuron_idx], axis=0)
                        neuron_temporal_saccade_trial_mean = np.nanmean(temporal_saccade_ori_activity[:, :, neuron_idx], axis=0)

                        if include_error_bars:
                            neuron_vis_only_trial_std = np.nanstd(saccade_off_ori_activity[:, :, neuron_idx], axis=0)
                            neuron_nasal_saccade_trial_std = np.nanstd(nasal_saccade_ori_activity[:, :, neuron_idx], axis=0)
                            neuron_temporal_saccade_trial_std = np.nanstd(temporal_saccade_ori_activity[:, :, neuron_idx], axis=0)

                        # Optional : do imputation
                        if do_imputation:
                            neuron_nasal_saccade_trial_mean[np.isnan(neuron_nasal_saccade_trial_mean)] = \
                                np.nanmean(neuron_nasal_saccade_trial_mean)
                            neuron_temporal_saccade_trial_mean[np.isnan(neuron_temporal_saccade_trial_mean)] = \
                                np.nanmean(neuron_temporal_saccade_trial_mean)

                        vis_response_to_fit = [neuron_vis_only_trial_mean, neuron_nasal_saccade_trial_mean, neuron_temporal_saccade_trial_mean]
                        vis_response_error = [neuron_vis_only_trial_std, neuron_nasal_saccade_trial_std, neuron_temporal_saccade_trial_std]

                        vis_titles = ['No saccade', 'Nasal saccade', 'Temporal saccade']
                        line_colors = ['black', 'green', 'purple']

                        for n_vis_response, vis_response in enumerate(vis_response_to_fit):

                            if plot_fits:
                                
                                xdata, ydata, ydata_error = transform_ori_data_for_vonmises2(vis_response,
                                                                                             vis_response_error=
                                                                                             vis_response_error[
                                                                                                 n_vis_response],
                                                                                             scale_data=scale_data)

                                """
                                xdata, ydata, ydata_error = transform_ori_data_for_vonmises2(vis_response,
                                                                        vis_response_error=vis_response_error[n_vis_response],
                                                                        scale_data=scale_data)
                                # initial_guess = np.array([np.pi, 1])
                                try:
                                    initial_guess = [180, 1, 1, 1, 1]
                                    # fitted_params, _ = spopt.curve_fit(von_mises2, xdata, ydata, p0=initial_guess, maxfev=2000)
                                    fitted_params, _ = spopt.curve_fit(von_mises2, xdata, ydata, p0=initial_guess,
                                                                       bounds=([0, 0, 0, -np.inf, 0.01],
                                                                                [360, np.inf, np.inf, np.inf, np.inf]),
                                                                       maxfev=50000)
                                    fit_success = True
                                except:
                                    print('Curve fitting failed')
                                    fit_success = False
                                    pdb.set_trace()
                                """

                                # Replace previous fitting with already fitted parameters
                                fitted_params = ori_fit_results['von_mises_fitted_params'][n_neuron, :, n_vis_response]
                                fit_success = True

                                axs[n_vis_response].scatter(xdata, ydata, color=line_colors[n_vis_response])
                                axs[n_vis_response].plot(xdata, ydata, color=line_colors[n_vis_response])

                                if include_error_bars:
                                    for n_xdata in np.arange(len(xdata)):
                                        axs[n_vis_response].plot([xdata[n_xdata], xdata[n_xdata]],
                                                                 [ydata[n_xdata] - ydata_error[n_xdata],
                                                                  ydata[n_xdata] + ydata_error[n_xdata]],
                                                                 color=line_colors[n_vis_response])

                                if fit_success:
                                    xdata_interpolated = np.linspace(xdata[0], xdata[-1], 100)
                                    # ydata_predicted = von_mises2(xdata, fitted_params[0], fitted_params[1])
                                    # ydata_interpolated = von_mises2(xdata_interpolated, fitted_params[0], fitted_params[1])
                                    ydata_predicted = von_mises2(xdata, fitted_params[0], fitted_params[1], fitted_params[2],
                                                                 fitted_params[3], fitted_params[4])
                                    ydata_interpolated = von_mises2(xdata_interpolated, fitted_params[0],
                                                                    fitted_params[1], fitted_params[2], fitted_params[3],
                                                                    fitted_params[4])
                                    axs[n_vis_response].plot(xdata_interpolated, ydata_interpolated, linestyle='-', color='orange')
                                    """
                                    von_mises_explained_variance[n_neuron, n_vis_response] = \
                                        sklmetrics.explained_variance_score(ydata, ydata_predicted)
                                    """
                                    # replace with the leave-one-out vaariance explained
                                    von_mises_explained_variance[n_neuron, n_vis_response] = \
                                        ori_fit_results['loo_neuron_variance_explained'][n_neuron, n_vis_response]
                                    von_mises_fitted_loc[n_neuron, n_vis_response] = fitted_params[0]
                                    von_mises_fitted_params[n_neuron, :, n_vis_response] = fitted_params

                                # This was in radians
                                # axs[n_vis_response].set_xticks([xdata[0], 0, xdata[-1]])
                                # axs[n_vis_response].set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'], size=12)

                                # This is in degrees
                                axs[n_vis_response].set_xticks([0, 180, 360])

                            else:
                                axs[n_vis_response].scatter(ori_groups, vis_response)

                            # Trial explained varaince
                            tEV = ori_fit_results['loo_neuron_trial_variance_explained'][n_neuron, n_vis_response]

                            if np.isnan(von_mises_explained_variance[n_neuron, n_vis_response]):
                                title_txt = '%s $x_0 = %.2f$, EV = NaN' % (vis_titles[n_vis_response], fitted_params[0])
                            else:
                                title_txt = '%s $x_0 = %.2f$, EV = %.2f, tEV = %.2f' % (vis_titles[n_vis_response],
                                                                            fitted_params[0],
                                                                            von_mises_explained_variance[n_neuron, n_vis_response],
                                                                             tEV)
                            axs[n_vis_response].set_title(title_txt, size=11)

                        # Do von mises fit
                        axs[-1].set_xlabel('Vis orientation', size=11)

                        if plot_fits:
                            if scale_data:
                                axs[1].set_ylabel('(r - min(r)) / max(r)', size=11)
                            else:
                                axs[1].set_ylabel('Activity (z-score)', size=11)

                        fig.suptitle('%s neuron %.f' % (exp_id, neuron_idx), size=11)
                        fig_name = '%s_neuron_%.f_orientation_tuning_saccade_off_and_on' % (exp_id, neuron_idx)
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                        plt.close(fig)

                # TODO: save the fits
                save_folder = fig_folder
                save_name = '%s_von_mises_fit_params.npy' % exp_id
                save_path = os.path.join(save_folder, save_name)
                np.save(save_path, von_mises_fitted_params)

                # Plot summary across all neurons
                with plt.style.context(splstyle.get_style('nature-reviews')):

                    fig, axs = plt.subplots(1, 3)
                    fig.set_size_inches(10, 3)
                    dot_size = 10
                    pdb.set_trace()
                    axs[0].scatter(von_mises_fitted_loc[:, 0], von_mises_fitted_loc[:, 1], color='black', lw=0,
                                   s=dot_size, clip_on=False)
                    axs[0].set_xlabel('No saccade', size=11)
                    axs[0].set_ylabel('Nasal saccade', size=11)

                    axs[1].scatter(von_mises_fitted_loc[:, 0], von_mises_fitted_loc[:, 2], color='black', lw=0,
                                   s=dot_size, clip_on=False)
                    axs[1].set_xlabel('No saccade', size=11)
                    axs[1].set_ylabel('Temporal saccade', size=11)

                    axs[2].scatter(von_mises_fitted_loc[:, 1], von_mises_fitted_loc[:, 2], color='black', lw=0,
                                   s=dot_size, clip_on=False)
                    axs[2].set_xlabel('Nasal saccade', size=11)
                    axs[2].set_ylabel('Temporal saccade', size=11)


                    [ax.set_xlim([0, 2 * np.pi]) for ax in axs]
                    [ax.set_ylim([0, 2 * np.pi]) for ax in axs]
                    [ax.set_xticks([0, np.pi, 2 * np.pi]) for ax in axs]
                    [ax.set_yticks([0, np.pi, 2 * np.pi]) for ax in axs]
                    [ax.set_xticklabels([0, r'$\pi$', r'$2\pi$']) for ax in axs]
                    [ax.set_yticklabels([0, r'$\pi$', r'$2\pi$']) for ax in axs]

                    fig.suptitle('%s' % (exp_id), size=11)
                    fig_name = '%s_von_mises_fitted_loc' % (exp_id)
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                    plt.close(fig)




        if process == 'plot_saccade_ori_preferred_ori':

            fitted_params_folder = process_params[process]['fitted_params_folder']
            fig_folder = process_params[process]['fig_folder']
            regression_results_folder = process_params[process]['regression_results_folder']
            fig_exts = process_params[process]['fig_exts']
            
            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % 'grating'))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in
                       regression_result_files]


            # Store information for plotting across all experiemnts
            all_exp_no_saccade_sig = []
            all_exp_nasal_saccade_sig = []
            all_exp_temporal_saccade_sig = []
            all_exp_von_mises_fitted_loc = []

            # Plot individual experiments
            for exp_id in exp_ids:
                # fitted_params_path = os.path.join(fitted_params_folder, '%s_von_mises_fit_params.npy' % exp_id)
                fitted_params_path = os.path.join(fitted_params_folder, '%s_von_mises_model_fit_results.npz' % exp_id)
                model_results = np.load(fitted_params_path)

                von_mises_fitted_params = model_results['von_mises_fitted_params']
                loo_neuron_trial_variance_explained = model_results['loo_neuron_trial_variance_explained']

                # No flipping
                # von_mises_fitted_loc = von_mises_fitted_params[:, 0, :]

                # Flip the preferred orientation to the one with the greatest value of the scale parameter
                num_stim_cond = 3
                num_neurons = np.shape(von_mises_fitted_params)[0]
                von_mises_fitted_loc = np.zeros((num_neurons, num_stim_cond)) + np.nan

                for neuron_idx in np.arange(num_neurons):
                    for stim_idx in np.arange(num_stim_cond):
                        par1 = von_mises_fitted_params[neuron_idx, 0, stim_idx]
                        par2 = von_mises_fitted_params[neuron_idx, 1, stim_idx]  # scale ori
                        par3 = von_mises_fitted_params[neuron_idx, 2, stim_idx] # scale ori + 180

                        if par2 > par3:
                            preferred_ori = par1
                        else:
                            if par1 < 180:
                                preferred_ori = par1 + 180
                            else:
                                preferred_ori = par1 - 180

                        von_mises_fitted_loc[neuron_idx, stim_idx] = preferred_ori

                # pdb.set_trace()
                # von_mises_fitted_loc[von_mises_fitted_loc < 0] = von_mises_fitted_loc[von_mises_fitted_loc < 0] + 360
                # von_mises_fitted_loc[von_mises_fitted_loc > 360] = von_mises_fitted_loc[von_mises_fitted_loc > 360] - 360

                # Plot summary across all neurons
                fig, axs = plot_von_mises_fitted_loc(von_mises_fitted_loc, fig=None, axs=None)
                fig.suptitle('%s' % (exp_id), x=0.5, y=1.02, size=11)
                fig_name = '%s_von_mises_fitted_loc' % (exp_id)
                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
                plt.close(fig)

                # Plot only the significant ones
                no_saccade_sig = loo_neuron_trial_variance_explained[:, 0] > 0
                nasal_saccade_sig = loo_neuron_trial_variance_explained[:, 1] > 0
                temporal_saccade_sig = loo_neuron_trial_variance_explained[:, 2] > 0

                no_saccade_or_nasal_sig = no_saccade_sig + nasal_saccade_sig
                no_saccade_or_temp_sig = no_saccade_sig + temporal_saccade_sig
                nasal_or_temp_sig = temporal_saccade_sig + nasal_saccade_sig

                no_saccade_and_nasal_sig = no_saccade_sig & nasal_saccade_sig
                no_saccade_and_temp_sig = no_saccade_sig & temporal_saccade_sig
                nasal_and_temp_sig = temporal_saccade_sig & nasal_saccade_sig

                # pdb.set_trace()

                # subset_indices = [no_saccade_or_nasal_sig, no_saccade_or_nasal_sig, nasal_or_temp_sig]
                subset_indices = [no_saccade_and_nasal_sig, no_saccade_and_temp_sig, nasal_and_temp_sig]
                fig, axs = plot_von_mises_fitted_loc(von_mises_fitted_loc, subset_indices=subset_indices, fig=None, axs=None)
                fig.suptitle('%s' % (exp_id), x=0.5, y=1.02, size=11)
                fig_name = '%s_von_mises_fitted_loc_sig' % (exp_id)

                for ext in fig_exts:
                    fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')
                    plt.close(fig)

                all_exp_von_mises_fitted_loc.append(von_mises_fitted_loc)
                all_exp_no_saccade_sig.extend(no_saccade_sig)
                all_exp_nasal_saccade_sig.extend(nasal_saccade_sig)
                all_exp_temporal_saccade_sig.extend(temporal_saccade_sig)

            # Plot all experiments : any neuron
            all_exp_von_mises_fitted_loc = np.vstack(all_exp_von_mises_fitted_loc)
            fig, axs = plot_von_mises_fitted_loc(all_exp_von_mises_fitted_loc,
                                                 subset_indices=None, fig=None,
                                                 axs=None)
            fig.suptitle('All exps', x=0.5, y=1.02, size=11)
            fig_name = 'all_exp_von_mises_fitted_loc'
            for ext in fig_exts:
                fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')
                plt.close(fig)


            # Plot all experiments : sig neurons
            all_exp_no_saccade_sig = np.array(all_exp_no_saccade_sig)
            all_exp_nasal_saccade_sig = np.array(all_exp_nasal_saccade_sig)
            all_exp_temporal_saccade_sig = np.array(all_exp_temporal_saccade_sig)

            no_saccade_and_nasal_sig = all_exp_no_saccade_sig & all_exp_nasal_saccade_sig
            no_saccade_and_temp_sig = all_exp_no_saccade_sig & all_exp_temporal_saccade_sig
            nasal_and_temp_sig = all_exp_temporal_saccade_sig & all_exp_nasal_saccade_sig
            subset_indices = [no_saccade_and_nasal_sig, no_saccade_and_temp_sig, nasal_and_temp_sig]
            
            fig, axs = plot_von_mises_fitted_loc(all_exp_von_mises_fitted_loc, subset_indices=subset_indices, fig=None,
                                                 axs=None)
            fig.suptitle('All exps', x=0.5, y=1.02, size=11)
            fig_name = 'all_exp_von_mises_fitted_loc_sig'
            for ext in fig_exts:
                fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')
                plt.close(fig)




            # Plot venn diagram of significance in each group
            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, ax = plt.subplots()

                set1 = set(np.where(all_exp_no_saccade_sig)[0])
                set2 = set(np.where(all_exp_nasal_saccade_sig)[0])
                set3 = set(np.where(all_exp_temporal_saccade_sig)[0])

                v = mpl_venn.venn3([set1, set2, set3],
                          set_labels=('Vis', 'Nasal',
                                      'Temporal'),
                          ax=ax)
                fig_name = 'all_exp_sig_venn_diagram'
                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

        if process == 'plot_vis_and_saccade_response_sorted_raster':

            fitted_params_folder = process_params[process]['fitted_params_folder']
            fig_folder = process_params[process]['fig_folder']
            regression_results_folder = process_params[process]['regression_results_folder']
            sort_using = process_params[process]['sort_using']
            zscore_ea_neuron_separately = process_params[process]['zscore_ea_neuron_separately']
            divide_by_max = process_params[process]['divide_by_max']
            scale_to_unit_range = process_params[process]['scale_to_unit_range']
            include_histogram = process_params[process]['include_histogram']
            only_plot_sig_neurons = process_params[process]['only_plot_sig_neurons']
            zscore_cmap = process_params[process]['zscore_cmap']
            gray_nans = process_params[process]['gray_nans']
            general_cmap = process_params[process]['general_cmap']
            plot_fitted_curves = process_params[process]['plot_fitted_curves']

            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % 'grating'))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in
                       regression_result_files]

            all_exp_cond_neuron_pref_ori = []
            all_exp_neuron_mean_responses_subset = []
            all_exp_neuron_pref_ori_during_nasal_saccade = []
            all_exp_neuron_pref_ori_during_temporal_saccade = []

            for exp_id in exp_ids:
                # fitted_params_path = os.path.join(fitted_params_folder, '%s_von_mises_fit_params.npy' % exp_id)
                fitted_params_path = os.path.join(fitted_params_folder, '%s_von_mises_model_fit_results.npz' % exp_id)
                model_results = np.load(fitted_params_path)

                n_neuron = np.shape(model_results['loo_neuron_mse'])[0]
                n_vis_response = np.shape(model_results['neuron_mean_responses'])[2]

                neuron_mean_responses = model_results['neuron_mean_responses']
                fitted_params = model_results['von_mises_fitted_params']
                loo_neuron_trial_variance_explained = model_results['loo_neuron_trial_variance_explained']

                if plot_fitted_curves:
                    n_cond = 3
                    n_interpolated_ori = 100
                    neuron_mean_responses = np.zeros((n_neuron, n_cond, n_interpolated_ori))
                    xdata_interpolated = np.linspace(0, 360, n_interpolated_ori)
                    for neuron_idx in np.arange(n_neuron):
                        for cond_idx in [0, 1, 2]:
                            neuron_mean_responses[neuron_idx, cond_idx, :] = von_mises2(xdata_interpolated,
                                                 fitted_params[neuron_idx, 0, cond_idx],
                                                 fitted_params[neuron_idx, 1, cond_idx],
                                                 fitted_params[neuron_idx, 2, cond_idx],
                                                 fitted_params[neuron_idx, 3, cond_idx],
                                                 fitted_params[neuron_idx, 4, cond_idx])



                if sort_using == 'vis':
                    cond_idx = 0
                elif sort_using == 'nasal':
                    cond_idx = 1
                elif sort_using == 'temporal':
                    cond_idx = 2

                cond_VE = loo_neuron_trial_variance_explained[:, cond_idx]


                if only_plot_sig_neurons:
                    neuron_subset_idx = np.where(cond_VE > 0)[0]
                else:
                    neuron_subset_idx = np.arange(0, int(n_neuron))

                cond_neuron_pref_ori = fitted_params[neuron_subset_idx, 0, cond_idx]

                neuron_pref_ori_during_nasal_saccade = fitted_params[neuron_subset_idx, 0, 1]
                neuron_pref_ori_during_temporal_saccade = fitted_params[neuron_subset_idx, 0, 2]

                vis_ori_sort_idx = np.argsort(cond_neuron_pref_ori)
                neuron_mean_responses_subset = neuron_mean_responses[neuron_subset_idx, :, :].copy()

                vis_only_response = neuron_mean_responses_subset[vis_ori_sort_idx, 0, :]
                nasal_response = neuron_mean_responses_subset[vis_ori_sort_idx, 1, :]
                temporal_response = neuron_mean_responses_subset[vis_ori_sort_idx, 2, :]

                # Add information to all experiments store
                all_exp_cond_neuron_pref_ori.extend(cond_neuron_pref_ori)
                all_exp_neuron_mean_responses_subset.append(neuron_mean_responses_subset)
                all_exp_neuron_pref_ori_during_nasal_saccade.extend(neuron_pref_ori_during_nasal_saccade)
                all_exp_neuron_pref_ori_during_temporal_saccade.extend(neuron_pref_ori_during_temporal_saccade)
                
                fig, axs = plot_ori_grouped_raster(vis_only_response, nasal_response, 
                                                   temporal_response, vis_ori_sort_idx,
                                                   vis_only_neuron_pref_ori=cond_neuron_pref_ori,
                                                   neuron_pref_ori_during_nasal_saccade=neuron_pref_ori_during_nasal_saccade,
                                                   neuron_pref_ori_during_temporal_saccade=neuron_pref_ori_during_temporal_saccade,
                                                   zscore_ea_neuron_separately=zscore_ea_neuron_separately,
                                                   include_histogram=include_histogram,
                                                   zscore_cmap=zscore_cmap, general_cmap=general_cmap,
                                                   divide_by_max=divide_by_max, scale_to_unit_range=scale_to_unit_range,
                                                   gray_nans=gray_nans, fig=None, axs=None)

                fig.suptitle(exp_id, size=11, y=1.04)
                fig_name = '%s_vis_and_saccade_raster_sorted_%s' % (exp_id, sort_using)
                if plot_fitted_curves:
                    fig_name += '_model_fit'
                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


            # Do the same for but for all experiments combined
            all_exp_neuron_mean_responses_subset = np.vstack(all_exp_neuron_mean_responses_subset)
            all_exp_vis_ori_sort_idx = np.argsort(all_exp_cond_neuron_pref_ori)
            all_exp_vis_only_response = all_exp_neuron_mean_responses_subset[all_exp_vis_ori_sort_idx, 0, :]
            all_exp_nasal_response = all_exp_neuron_mean_responses_subset[all_exp_vis_ori_sort_idx, 1, :]
            all_exp_temporal_response = all_exp_neuron_mean_responses_subset[all_exp_vis_ori_sort_idx, 2, :]

            fig, axs = plot_ori_grouped_raster(all_exp_vis_only_response, all_exp_nasal_response,
                                               all_exp_temporal_response, all_exp_vis_ori_sort_idx,
                                               vis_only_neuron_pref_ori=all_exp_cond_neuron_pref_ori,
                                               neuron_pref_ori_during_nasal_saccade=all_exp_neuron_pref_ori_during_nasal_saccade,
                                               neuron_pref_ori_during_temporal_saccade=all_exp_neuron_pref_ori_during_temporal_saccade,
                                               zscore_ea_neuron_separately=zscore_ea_neuron_separately,
                                               include_histogram=include_histogram,
                                               zscore_cmap=zscore_cmap, general_cmap=general_cmap,
                                               divide_by_max=divide_by_max, scale_to_unit_range=scale_to_unit_range,
                                               fig=None, axs=None,
                                               gray_nans=gray_nans)
            fig.suptitle('All experiments', size=11, y=1.04)
            fig_name = 'all_exp_vis_and_saccade_raster_sorted_%s' % sort_using
            if plot_fitted_curves:
                fig_name += '_model_fit'
            fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')
        
        if process == 'plot_vis_neurons_vs_vis_saccade_neuron_preferred_ori':

            fitted_params_folder = process_params[process]['fitted_params_folder']
            fig_folder = process_params[process]['fig_folder']
            regression_results_folder = process_params[process]['regression_results_folder']
            fig_exts = process_params[process]['fig_exts']
            plot_idv_exp_traces = False
            do_ks_test = True

            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % 'grating'))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in
                       regression_result_files]

            # Aggregate for all experiments
            all_exp_prop_vis_only_neuron_best_ori = []
            all_exp_prop_vis_and_saccade_neuron_best_ori = []

            # Plot individual experiments
            for exp_id in exp_ids:
                # fitted_params_path = os.path.join(fitted_params_folder, '%s_von_mises_fit_params.npy' % exp_id)

                # This is for all neurons (regardless of varaince explained)
                fitted_params_path = os.path.join(fitted_params_folder, '%s_all_von_mises_model_fit_results.npz' % exp_id)
                model_results = np.load(fitted_params_path)

                von_mises_fitted_params = model_results['von_mises_fitted_params']
                loo_neuron_trial_variance_explained = model_results['loo_neuron_trial_variance_explained']

                # Load regression result and look get variance explained using vis model
                exp_type = 'both'
                regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*%s*.npz' % (exp_id, exp_type)))[0],
                    allow_pickle=True)

                # Plot vis neurons ori distribution, then vis + saccade neuron ori distribution
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

                vis_only_neurons_subset_idx = np.where(
                    (model_a_explained_var > 0) &
                    (model_b_explained_var <= model_a_explained_var)
                )[0]

                vis_and_saccade_neurosn_subset_idx = np.where(
                    (model_a_explained_var > 0) &
                    (model_b_explained_var > 0) &
                    (model_b_explained_var > model_a_explained_var)
                )[0]

                sig_ori_fitting_idx = np.where(
                    loo_neuron_trial_variance_explained > 0
                )[0]

                neuron_best_ori = von_mises_fitted_params[:, 0, 0]  # first 0 is to get ori, second 0 to get vis cond

                vis_only_neuron_and_sig_ori_idx = np.intersect1d(vis_only_neurons_subset_idx, sig_ori_fitting_idx)
                vis_and_saccade_neuron_and_sig_ori_idx = np.intersect1d(vis_and_saccade_neurosn_subset_idx, sig_ori_fitting_idx)

                vis_only_neuron_best_ori = neuron_best_ori[vis_only_neuron_and_sig_ori_idx]
                vis_and_saccade_neuron_best_ori = neuron_best_ori[vis_and_saccade_neuron_and_sig_ori_idx]

                preferred_ori = np.sort(np.unique(neuron_best_ori))


                best_ori_bins = np.linspace(0, 360, 30)
                num_bins = len(best_ori_bins) - 1
                num_vis_only_neuron_best_ori = np.zeros((num_bins, ))
                num_vis_and_saccade_neuron_best_ori = np.zeros((num_bins, ))

                # for n_ori, ori in enumerate(preferred_ori):
                #     num_vis_only_neuron_best_ori[n_ori] = len(np.where(vis_only_neuron_best_ori == ori)[0])
                #     num_vis_and_saccade_neuron_best_ori[n_ori] = len(np.where(vis_and_saccade_neuron_best_ori == ori)[0])

                for n_bin in np.arange(len(best_ori_bins) - 1):
                    left_edge = best_ori_bins[n_bin]
                    right_edge = best_ori_bins[n_bin+1]

                    num_vis_only_neuron_best_ori[n_bin] = len(np.where(
                        (vis_only_neuron_best_ori >= left_edge) &
                        (vis_only_neuron_best_ori < right_edge)
                    )[0])

                    num_vis_and_saccade_neuron_best_ori[n_bin] = len(np.where(
                        (vis_and_saccade_neuron_best_ori >= left_edge) &
                        (vis_and_saccade_neuron_best_ori < right_edge)
                    )[0])

                prop_vis_only_neuron_best_ori = num_vis_only_neuron_best_ori / len(vis_only_neuron_best_ori)
                prop_vis_and_saccade_neuron_best_ori = num_vis_and_saccade_neuron_best_ori / len(vis_and_saccade_neuron_best_ori)

                all_exp_prop_vis_only_neuron_best_ori.append(prop_vis_only_neuron_best_ori)
                all_exp_prop_vis_and_saccade_neuron_best_ori.append(prop_vis_and_saccade_neuron_best_ori)

                with plt.style.context(splstyle.get_style('nature-reviews')):

                    fig, ax = plt.subplots()
                    ax.plot(best_ori_bins[0:-1], prop_vis_only_neuron_best_ori, color='black', label='Vis only')
                    ax.plot(best_ori_bins[0:-1], prop_vis_and_saccade_neuron_best_ori, color='red', label='Vis + saccade')
                    ax.set_xlabel('Preferred visual orientation', size=11)
                    ax.set_ylabel('Proportion of neurons', size=11)
                    ax.set_title(exp_id, size=11)
                    ax.legend()

                    fig_name = '%s_prop_vis_only_vs_vis_and_saccade_neuron_best_ori' % exp_id
                    for ext in fig_exts:
                        fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')


                # Plot scatter of variance explained by each model and preferred orientation
                preferred_ori_per_neuron = neuron_best_ori.copy()
                preferred_ori_per_neuron[loo_neuron_trial_variance_explained[:, 0] <= 0] = np.nan

                # look for neurons with
                subset_idx_weird = np.where(
                    (model_a_explained_var < 0) & # (model_b_explained_var < 0) &
                    (loo_neuron_trial_variance_explained[:, 0] > 0)
                )[0]
                # ^ a bit strange that there are neurons like this? Not explained well by vis ori model, but von mises fit is significant...
                # will be good to plot these neurons and see what's up...

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(4, 4)
                    cmap = mpl.cm.twilight
                    cmap.set_bad(color='white')
                    all_ev = np.concatenate([model_b_explained_var, model_a_explained_var])
                    all_max = np.max(all_ev)
                    all_min = np.min(all_ev)
                    unity_vals = np.linspace(all_min, all_max, 100)
                    ax.scatter(model_b_explained_var, model_a_explained_var,
                               c=preferred_ori_per_neuron, cmap=cmap, lw=0)
                    ax.plot(unity_vals, unity_vals, color='gray', linestyle='--', lw=0.5)
                    ax.axhline(0, color='gray', linestyle='--', lw=0.5)
                    ax.axvline(0, color='gray', linestyle='--', lw=0.5)
                    ax.set_xlabel('Vis + saccade', size=11)
                    ax.set_ylabel('Vis only', size=11)
                    ax.set_xlim([all_min, all_max])
                    ax.set_ylim([all_min, all_max])

                    fig_name = '%s_model_ve_comparison_w_preferred_ori_colored' % exp_id
                    for ext in fig_exts:
                        fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')


                if do_ks_test:
                    # do KS test and plot the results
                    num_sampling_w_replacement = 1000
                    num_vis_and_saccade_neuron = len(vis_and_saccade_neuron_best_ori)

                    if num_vis_and_saccade_neuron > 0:

                        original_ks_stats = np.zeros((num_sampling_w_replacement, ))
                        shuffled_ks_stats = np.zeros((num_sampling_w_replacement, ))
                        original_ks_pval =  np.zeros((num_sampling_w_replacement, ))

                        for n_sample in np.arange(num_sampling_w_replacement):
                            
                            vis_only_neuron_best_ori_subsampled = np.random.choice(vis_only_neuron_best_ori,
                                                                                   size=num_vis_and_saccade_neuron,
                                                                                   replace=True)
                            ks_res, pval = sstats.ks_2samp(vis_only_neuron_best_ori_subsampled,
                                                   vis_and_saccade_neuron_best_ori)

                            # Do the same, but shuffle the vis-only and vis+saccade identity
                            both_ori = np.concatenate([vis_only_neuron_best_ori_subsampled,
                                                       vis_and_saccade_neuron_best_ori])
                            both_ori = np.random.permutation(both_ori)
                            vis_only_neuron_best_ori_subsampled_shuffled = both_ori[0:num_vis_and_saccade_neuron]
                            vis_and_saccade_neuron_best_ori_shuffled = both_ori[num_vis_and_saccade_neuron:]

                            ks_res_shuffled, pval_shuffled = sstats.ks_2samp(vis_only_neuron_best_ori_subsampled_shuffled,
                                                            vis_and_saccade_neuron_best_ori_shuffled)

                            original_ks_stats[n_sample] = ks_res
                            shuffled_ks_stats[n_sample] = ks_res_shuffled
                            original_ks_pval[n_sample] = pval

                        with plt.style.context(splstyle.get_style('nature-reviews')):
                            fig, ax = plt.subplots()
                            fig.set_size_inches(5, 4)
                            ax.hist(original_ks_stats, lw=0, bins=50, color='black', label='Original')
                            ax.hist(shuffled_ks_stats, lw=0, bins=50, color='gray', alpha=0.5, label='Shuffled')
                            ax.set_ylabel('Number of subsampling runs', size=10)
                            ax.legend()
                            fig_name = '%s_ks_test_stat_for_ori_distribution' % exp_id
                            for ext in fig_exts:
                                fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')

                            plt.close(fig)

                        with plt.style.context(splstyle.get_style('nature-reviews')):
                            fig, ax = plt.subplots()
                            fig.set_size_inches(5, 4)
                            ax.hist(original_ks_pval, lw=0, bins=50, color='black')
                            ax.set_ylabel('Number of subsampling runs', size=10)
                            ax.set_xscale('log')
                            ax.axvline(0.05, linestyle='--', color='gray', lw=1)
                            # ax.hist(shuffled_ks_stats, lw=0, bins=50, color='gray', alpha=0.5)
                            fig_name = '%s_ks_test_pval_for_ori_distribution' % exp_id
                            for ext in fig_exts:
                                fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')

                        with plt.style.context(splstyle.get_style('nature-reviews')):
                            fig, ax = plt.subplots()
                            fig.set_size_inches(5, 4)
                            ax.hist(original_ks_pval, density=True, histtype='step', bins=100,
                            cumulative=True, color='black')
                            ax.set_ylabel('Proportion of subsampling runs', size=10)
                            ax.set_xscale('log')

                            ax.axvline(0.05, linestyle='--', color='gray', lw=1)

                            num_sig_pval = len(np.where(original_ks_pval <= 0.05)[0])

                            ax.set_title('Prop p-values below 0.05: %.2f' % (num_sig_pval / len(original_ks_pval)), size=10)
                            # ax.axvline(0.05, linestyle='--', color='gray', lw=1)
                            # ax.hist(shuffled_ks_stats, lw=0, bins=50, color='gray', alpha=0.5)
                            fig_name = '%s_ks_test_pval_for_ori_distribution_cumulative_hist' % exp_id
                            for ext in fig_exts:
                                fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')


            # Plot for all experiments
            all_exp_mean_prop_vis_only_neuron_best_ori = np.mean(np.array(all_exp_prop_vis_only_neuron_best_ori), axis=0)
            all_exp_mean_prop_vis_and_saccade_neuron_best_ori = np.mean(np.array(all_exp_prop_vis_and_saccade_neuron_best_ori), axis=0)

            with plt.style.context(splstyle.get_style('nature-reviews')):

                fig, ax = plt.subplots()
                ax.plot(best_ori_bins[0:-1], all_exp_mean_prop_vis_only_neuron_best_ori, color='black', label='Vis only')
                ax.plot(best_ori_bins[0:-1], all_exp_mean_prop_vis_and_saccade_neuron_best_ori, color='red', label='Vis + saccade')

                if plot_idv_exp_traces:
                    for exp_vis_only_best_ori in all_exp_prop_vis_only_neuron_best_ori:
                        ax.plot(best_ori_bins[0:-1], exp_vis_only_best_ori, color='black', alpha=0.1)
                    for exp_vis_and_saccade_best_ori in all_exp_prop_vis_and_saccade_neuron_best_ori:
                        ax.plot(best_ori_bins[0:-1], exp_vis_and_saccade_best_ori, color='red', alpha=0.1)

                ax.set_xlabel('Preferred visual orientation', size=11)
                ax.set_ylabel('Proportion of neurons', size=11)
                ax.set_title('All experiments mean', size=11)
                ax.legend()

                fig_name = 'all_exp_mean_prop_vis_only_vs_vis_and_saccade_neuron_best_ori'
                for ext in fig_exts:
                    fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')
                    
        if process == 'plot_kernel_fit_raster':

            """
            This process plots the fitted kernels to a particular experiment 'type' using a particular 
            regression model. The plot is a raster for each particular feature fitted by the regresison model.
            
            The experiment type can be: (1) Grating only (2) Gray screen only (3) Both grating and gray screen 
            The regression model can have different features at different time windows.
            
            This process assumes the fit_regression_model process has already been ran.
            """

            model_X_set_to_plot = process_params[process]['model_X_set_to_plot']
            sort_method = process_params[process]['sort_method']
            fig_folder = process_params[process]['fig_folder']
            transform_dir_to_temporal_nasal = process_params[process]['transform_dir_to_temporal_nasal']
            explained_var_threshold = process_params[process]['explained_var_threshold']
            exp_type = process_params[process]['exp_type']
            kernels_to_include = process_params[process]['kernels_to_include']
            excited_threshold = 0.1
            inhibited_threshold = -0.1

            if not os.path.isdir(fig_folder):
                os.makedirs(fig_folder)

            # Load all experiment data
            data = load_data(data_folder=process_params[process]['data_folder'],
                             file_types_to_load=process_params[process]['file_types_to_load'])

            regression_results_folder = process_params[process]['regression_results_folder']
            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % exp_type))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in
                       regression_result_files]


            # Compare the temporal profile of the saccade kernels

            all_X_set_weights_all_mean = []

            if sort_method == 'Anya-identical':

                print('Using sorting method exactly identical to Anya, so will override EV threshold to None')
                explained_var_threshold = -np.inf

                sort_mat_filepath = '/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults/sorting-info/orderMatrix.mat'
                sort_data = loadmat(sort_mat_filepath)['orderMatrix']

                all_neuron_within_exp_sort_id = []
                all_neuron_group_sort_id = []
                for neuron_type, sort_prop in sort_data.items():
                    neuron_id = sort_prop['neuronIDs']
                    group_id = sort_prop['group']

                    if type(neuron_id) is int:
                        neuron_id = [neuron_id]
                        group_id = [group_id]
                    all_neuron_within_exp_sort_id.extend(neuron_id)
                    all_neuron_group_sort_id.extend(group_id)

                # this is info from Anya
                group_id_number_to_exp_name = {
                    1: 'SS041_2015-04-23',
                    2: 'SS044_2015-04-28',
                    3: 'SS045_2015-05-04',
                    4: 'SS045_2015-05-05',
                    5: 'SS047_2015-11-23',
                    6: 'SS047_2015-12-03',
                    7: 'SS048_2015-11-09',
                    8: 'SS048_2015-12-02',
                }
                
                exp_name_to_group_id = {x: y for (y, x) in group_id_number_to_exp_name.items()}

                all_neuron_within_exp_id = []
                all_neuron_group_ids = []
                all_neuron_ev = []
                

            for exp_id in exp_ids:
                # Experiment data
                exp_data = data[exp_id]

                regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*%s*.npz' % (exp_id, exp_type)))[0],
                    allow_pickle=True)

                X_set_idx = np.where(regression_result['X_sets_names'] == model_X_set_to_plot)[0][0]
                model_weights_per_X_set = regression_result['model_weights_per_X_set'].item()
                explained_var_per_X_set = regression_result['explained_var_per_X_set']
                X_set_ev = explained_var_per_X_set[:, X_set_idx]
                X_set_weights = model_weights_per_X_set[model_X_set_to_plot]  # numCV x numNeuron x numFeatures
                feature_name_and_indices = regression_result['feature_indices_per_X_set'].item()[model_X_set_to_plot]

                X_set_weights_all_mean = np.mean(X_set_weights, axis=0)


                if transform_dir_to_temporal_nasal:

                    if 'vis_on' in kernels_to_include:
                        vis_on_idx = feature_name_and_indices['vis_on']
                    if 'vis_dir' in kernels_to_include:
                        vis_dir_idx = feature_name_and_indices['vis_dir']

                    saccade_on_idx = feature_name_and_indices['saccade_on']
                    saccade_dir_idx = feature_name_and_indices['saccade_dir']

                    X_set_weights_all_mean_transformed = X_set_weights_all_mean.copy()

                    if 'vis_dir' in kernels_to_include:
                        X_set_weights_all_mean_transformed[:, vis_on_idx] = \
                            X_set_weights_all_mean[:, vis_on_idx] - X_set_weights_all_mean[:, vis_dir_idx]
                        X_set_weights_all_mean_transformed[:, vis_dir_idx] = \
                            X_set_weights_all_mean[:, vis_on_idx] + X_set_weights_all_mean[:, vis_dir_idx]

                    X_set_weights_all_mean_transformed[:, saccade_on_idx] = \
                        X_set_weights_all_mean[:, saccade_on_idx] - X_set_weights_all_mean[:, saccade_dir_idx]

                    X_set_weights_all_mean_transformed[:, saccade_dir_idx] = \
                        X_set_weights_all_mean[:, saccade_on_idx] + X_set_weights_all_mean[:, saccade_dir_idx]

                    X_set_weights_all_mean = X_set_weights_all_mean_transformed

                    if 'vis_dir' in kernels_to_include:
                        feature_name_and_indices['vis_nasal'] = feature_name_and_indices.pop('vis_on')
                        feature_name_and_indices['vis_temporal'] = feature_name_and_indices.pop('vis_dir')

                    feature_name_and_indices['saccade_nasal'] = feature_name_and_indices.pop('saccade_on')
                    feature_name_and_indices['saccade_temporal'] = feature_name_and_indices.pop('saccade_dir')



                if explained_var_threshold is None:
                    explained_var_threshold = 0
                    subset_neuron_idx = np.arange(np.shape(X_set_weights_all_mean)[0])
                else:
                    subset_neuron_idx = np.where(X_set_ev >= explained_var_threshold)[0]

                X_set_weights_all_mean = X_set_weights_all_mean[subset_neuron_idx, :]
                all_X_set_weights_all_mean.append(X_set_weights_all_mean)

                if sort_method == 'rastermap':
                    model = rastermap.Rastermap(n_components=1, n_X=30, nPC=50, init='pca')
                    model.fit(X_set_weights_all_mean)
                    sort_idx = model.isort
                    X_set_weights_all_mean_sorted = X_set_weights_all_mean[sort_idx, :]
                elif sort_method == 'Anya':

                    temporal_response_per_t_neuron = X_set_weights_all_mean[:, feature_name_and_indices['saccade_temporal']]
                    nasal_response_per_t_neuron = X_set_weights_all_mean[:, feature_name_and_indices['saccade_nasal']]

                    sort_idx = get_Anya_raster_sort_idx(temporal_response_per_t_neuron, nasal_response_per_t_neuron,
                                                        excited_threshold=excited_threshold,
                                                        inhibited_threshold=inhibited_threshold)

                    X_set_weights_all_mean_sorted = X_set_weights_all_mean[sort_idx, :]

                elif sort_method == 'Anya-identical':

                    # Anya-identical methods only apply to "all recordings" sort, not single session sort
                    num_neurons = np.shape(X_set_weights_all_mean)[0]
                    all_neuron_group_ids.extend(np.repeat(exp_name_to_group_id[exp_id], num_neurons))
                    all_neuron_within_exp_id.extend(np.arange(1, num_neurons+1))
                    all_neuron_ev.extend(X_set_ev)
                    continue

                    
                elif sort_method == 'none':
                    X_set_weights_all_mean_sorted = X_set_weights_all_mean

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(5, 4)
                    ax.imshow(X_set_weights_all_mean_sorted, cmap='bwr', vmin=-1, vmax=1,
                              interpolation='none', aspect='auto')

                    ax.set_xlabel('Regressors', size=11)
                    ax.set_ylabel('Cells', size=11)

                    for n_feature, (feature_name, feature_idx) in enumerate(feature_name_and_indices.items()):

                        mid_point = np.mean(feature_idx)
                        if num_separating_columns > 0:
                            mid_point = mid_point + (n_feature * num_separating_columns)

                        ax.text(mid_point, -1, feature_name, ha='center', rotation=45)

                    fig.suptitle('%s min EV: %.2f' % (exp_id, explained_var_threshold), size=11)
                    fig_name = '%s_%s_exp_%s_model_features_sorted_using_%s' % (exp_id, exp_type, model_X_set_to_plot, sort_method)

                    if transform_dir_to_temporal_nasal:
                        fig_name += '_transformed_to_temporal_nasal'

                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                plt.close(fig)

            # Plot all neurons
            all_X_set_weights_all_mean = np.concatenate(all_X_set_weights_all_mean)


            # Duplicate the running kernel, currently assuming it's the last index
            num_running_kernels_to_duplicate = process_params[process]['num_running_kernels_to_duplicate']

            if num_running_kernels_to_duplicate > 0:
                running_kernel = all_X_set_weights_all_mean[:, -1]
                running_kernel_duplicated = np.tile(running_kernel, (num_running_kernels_to_duplicate, 1)).T
                all_X_set_weights_all_mean = np.concatenate([all_X_set_weights_all_mean, running_kernel_duplicated], axis=1)

                feature_name_and_indices['running'] = np.arange(feature_name_and_indices['running'][0],
                                                               feature_name_and_indices['running'][0]+ num_running_kernels_to_duplicate)


            if sort_method == 'rastermap':
                model = rastermap.Rastermap(n_components=1, n_X=30, nPC=50, init='pca')
                model.fit(all_X_set_weights_all_mean)
                sort_idx = model.isort
                all_X_set_weights_all_mean_sorted = all_X_set_weights_all_mean[sort_idx, :]
            elif sort_method == 'Anya':

                temporal_response_per_t_neuron = all_X_set_weights_all_mean[:, feature_name_and_indices['saccade_temporal']]
                nasal_response_per_t_neuron = all_X_set_weights_all_mean[:, feature_name_and_indices['saccade_nasal']]

                sort_idx = get_Anya_raster_sort_idx(temporal_response_per_t_neuron, nasal_response_per_t_neuron,
                                                    excited_threshold=excited_threshold,
                                                    inhibited_threshold=inhibited_threshold)

                all_X_set_weights_all_mean_sorted = all_X_set_weights_all_mean[sort_idx, :]
            
            elif sort_method == 'Anya-identical':

                num_neurons_to_sort = len(all_neuron_within_exp_sort_id)
                num_features = np.shape(all_X_set_weights_all_mean)[1]

                all_neuron_within_exp_id = np.array(all_neuron_within_exp_id)
                all_neuron_group_ids = np.array(all_neuron_group_ids)

                all_X_set_weights_all_mean_sorted = np.zeros((num_neurons_to_sort, num_features))
                all_used_neurons_explained_variance = np.zeros((num_neurons_to_sort, ))

                # this is the slow and naive way but it works
                for n_idx in np.arange(num_neurons_to_sort):

                    neuron_idx_to_get = np.where(
                        (all_neuron_within_exp_id == all_neuron_within_exp_sort_id[n_idx]) &
                        (all_neuron_group_ids == all_neuron_group_sort_id[n_idx])
                    )[0][0]

                    all_X_set_weights_all_mean_sorted[n_idx, :] = all_X_set_weights_all_mean[neuron_idx_to_get, :]
                    all_used_neurons_explained_variance[n_idx] = all_neuron_ev[neuron_idx_to_get]

            elif sort_method == 'none':
                all_X_set_weights_all_mean_sorted = all_X_set_weights_all_mean

            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, ax = plt.subplots()

                if 'vis_on' in kernels_to_include:
                    fig.set_size_inches(5, 4)
                else:
                    fig.set_size_inches(4, 4)

                num_separating_columns = process_params[process]['num_separating_columns']


                if num_separating_columns > 0:
                    num_neurons, num_features_indices = np.shape(all_X_set_weights_all_mean_sorted)
                    num_features = len(feature_name_and_indices)
                    num_features_indices_w_gaps = num_features_indices + (num_features - 1) * num_separating_columns
                    all_X_set_weights_all_mean_sorted_w_gaps = np.zeros((num_neurons, num_features_indices_w_gaps))

                    # temp hack, do use some form of ordered dict based on the indices value in the future
                    kernel_name_order = ['bias', 'vis_nasal', 'vis_temporal', 'saccade_nasal', 'saccade_temporal', 'running']

                    # offset_counter = 0
                    for n_kernel, kernel_name in enumerate(kernel_name_order):
                        print(kernel_name)
                        kernel_indices = feature_name_and_indices[kernel_name] + (n_kernel * num_separating_columns)

                        all_X_set_weights_all_mean_sorted_w_gaps[:, kernel_indices] = all_X_set_weights_all_mean_sorted[:, feature_name_and_indices[kernel_name]]

                        if n_kernel != len(kernel_name_order) - 1:
                            all_X_set_weights_all_mean_sorted_w_gaps[:, (kernel_indices[-1]+1):int(kernel_indices[-1]+num_separating_columns)] = 0


                        # offset_counter = kernel_indices[-1] + num_separating_columns

                if num_separating_columns > 0:
                    ax.imshow(all_X_set_weights_all_mean_sorted_w_gaps, cmap='bwr', vmin=-1, vmax=1,
                              interpolation='none', aspect='auto')
                else:

                    ax.imshow(all_X_set_weights_all_mean_sorted, cmap='bwr', vmin=-1, vmax=1,
                          interpolation='none', aspect='auto')

                ax.set_xlabel('Regressors', size=11)
                ax.set_ylabel('Cells', size=11)

                for n_feature, (feature_name, feature_idx) in enumerate(feature_name_and_indices.items()):
                    mid_point = np.mean(feature_idx)
                    if num_separating_columns > 0:
                        mid_point = mid_point + (n_feature * num_separating_columns)
                    ax.text(mid_point, -1, feature_name, ha='center', rotation=45)

                fig.suptitle('All exp min EV: %.2f' % (explained_var_threshold), size=11)
                fig_name ='all_%s_exp_%s_model_features_sorted_using_%s' % (
                exp_type, model_X_set_to_plot, sort_method)
                if transform_dir_to_temporal_nasal:
                    fig_name += '_transformed_to_temporal_nasal'
                fig_path = os.path.join(fig_folder, fig_name)
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print('Saved figure to %s' % fig_path)

            if sort_method == 'Anya-identical':
                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(5, 4)

                    ax.hist(all_used_neurons_explained_variance, lw=0, color='black', bins=50)

                    fig_name = 'Anja_identical_method_EV_distribution'
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


        if process == 'plot_kernel_scatter':

            """
            This process plots the mean kernel values for any two type of kernels 
            eg. plotting the nasal kernel against the temporal kernel for each neuron 
            """

            model_X_set_to_plot = process_params[process]['model_X_set_to_plot']
            null_X_set = process_params[process]['null_X_set']

            fig_folder = process_params[process]['fig_folder']
            neuron_subset_condition = process_params[process]['neuron_subset_condition']
            explained_var_threshold = process_params[process]['explained_var_threshold']
            exp_type = process_params[process]['exp_type']
            x_axis_kernel = process_params[process]['x_axis_kernel']
            y_axis_kernel = process_params[process]['y_axis_kernel']
            same_x_y_range = process_params[process]['same_x_y_range']
            plot_indv_lobf = process_params[process]['plot_indv_lobf']
            kernel_metric = process_params[process]['kernel_metric']


            if not os.path.isdir(fig_folder):
                os.makedirs(fig_folder)

            regression_results_folder = process_params[process]['regression_results_folder']
            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % exp_type))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in
                       regression_result_files]

            all_exp_x_vals = []
            all_exp_y_vals = []
            all_subject = []
            all_exp = []

            exp_intercepts = []
            exp_slopes = []


            for exp_id in exp_ids:

                regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*%s*.npz' % (exp_id, exp_type)))[0],
                    allow_pickle=True)


                X_sets_names = regression_result['X_sets_names']
                explained_var_per_X_set = regression_result['explained_var_per_X_set']


                # Subset neurons based on some condition

                if neuron_subset_condition == 'sig_saccade_neurons':
                    # Get neurons with significant saccade kernel
                    model_a = 'vis_and_saccade_shuffled_and_running'
                    model_b = 'vis_and_saccade_and_running'

                    model_a_idx = np.where(X_sets_names == model_a)[0][0]
                    model_b_idx = np.where(X_sets_names == model_b)[0][0]

                    model_a_explained_var = explained_var_per_X_set[:, model_a_idx]
                    model_b_explained_var = explained_var_per_X_set[:, model_b_idx]

                    b_minus_a = model_b_explained_var - model_a_explained_var

                    subset_idx = np.where(
                        (model_a_explained_var > explained_var_threshold) &
                        (model_b_explained_var > explained_var_threshold) &
                        (model_b_explained_var > model_a_explained_var)
                    )[0]

                    vis_only_neurons_subset_idx = np.where(
                        (model_a_explained_var > 0) &
                        (model_b_explained_var <= model_a_explained_var)
                    )[0]

                    subset_idx = np.where(
                        (model_a_explained_var > explained_var_threshold) &
                        (model_b_explained_var > explained_var_threshold) &
                        (model_b_explained_var > model_a_explained_var)
                    )[0]

                elif neuron_subset_condition == 'better_than_null':

                    model_a = null_X_set
                    model_b = model_X_set_to_plot

                    model_a_idx = np.where(X_sets_names == model_a)[0][0]
                    model_b_idx = np.where(X_sets_names == model_b)[0][0]

                    model_a_explained_var = explained_var_per_X_set[:, model_a_idx]
                    model_b_explained_var = explained_var_per_X_set[:, model_b_idx]

                    subset_idx = np.where(
                        (model_b_explained_var > model_a_explained_var) &
                        (model_b_explained_var > explained_var_threshold)
                    )[0]
                
                elif neuron_subset_condition == 'both_vis_dir_and_saccade_dir_selective':

                    # Get sig saccade dir neurons
                    model_a_idx = np.where(X_sets_names == 'vis_and_saccade_and_running')[0][0]
                    model_b_idx = np.where(X_sets_names == 'vis_and_saccade_on_and_running')[0][0]
                    model_a_explained_var = explained_var_per_X_set[:, model_a_idx]
                    model_b_explained_var = explained_var_per_X_set[:, model_b_idx]

                    saccade_dir_sig_idx = np.where(
                        (model_a_explained_var > model_b_explained_var) &
                        (model_a_explained_var > explained_var_threshold)
                    )[0]

                    # Get sig vis dir neurons
                    model_a_idx = np.where(X_sets_names == 'vis_and_saccade_and_running')[0][0]
                    model_b_idx = np.where(X_sets_names == 'vis_on_and_saccade_and_running')[0][0]
                    model_a_explained_var = explained_var_per_X_set[:, model_a_idx]
                    model_b_explained_var = explained_var_per_X_set[:, model_b_idx]

                    vis_dir_sig_idx = np.where(
                        (model_a_explained_var > model_b_explained_var) &
                        (model_a_explained_var > explained_var_threshold)
                    )[0]

                    subset_idx = np.intersect1d(saccade_dir_sig_idx, vis_dir_sig_idx)

                else:
                    subset_idx = np.arange(0, np.shape(explained_var_per_X_set)[0])

                # Get the kernels
                model_weights_per_X_set = regression_result['model_weights_per_X_set'].item()
                X_set_weights = model_weights_per_X_set[model_X_set_to_plot]  # numCV x numNeuron x numFeatures
                feature_name_and_indices = regression_result['feature_indices_per_X_set'].item()[model_X_set_to_plot]
                regression_kernel_windows = regression_result['regression_kernel_windows']
                saccade_on_kernel_idx = np.where(regression_result['regression_kernel_names'] == 'saccade_on')[0][0]
                saccade_on_window = regression_kernel_windows[saccade_on_kernel_idx]


                saccade_onset_kernel_indices = feature_name_and_indices['saccade_on']
                saccade_dir_kernel_indices = feature_name_and_indices['saccade_dir']

                saccade_on_kernels = np.mean(X_set_weights[:, subset_idx, :][:, :, saccade_onset_kernel_indices], axis=0)
                saccade_dir_kernels = np.mean(X_set_weights[:, subset_idx, :][:, :, saccade_dir_kernel_indices], axis=0)

                saccade_on_peri_event_window = np.linspace(saccade_on_window[0], saccade_on_window[1],
                                                           len(saccade_onset_kernel_indices))
                post_saccade_idx = np.where(saccade_on_peri_event_window >= 0)[0]

                # Visual kernels
                if exp_type in ['grating', 'both']:
                    vis_onset_kernel_indices = feature_name_and_indices['vis_on']
                    vis_dir_kernel_indices = feature_name_and_indices['vis_dir']

                    vis_on_kernels = np.mean(X_set_weights[:, subset_idx, :][:, :, vis_onset_kernel_indices], axis=0)
                    vis_dir_kernels = np.mean(X_set_weights[:, subset_idx, :][:, :, vis_dir_kernel_indices], axis=0)

                    vis_on_kernel_idx = np.where(regression_result['regression_kernel_names'] == 'vis_on')[0][0]
                    vis_on_window = regression_kernel_windows[vis_on_kernel_idx]
                    vis_on_peri_event_window = np.linspace(vis_on_window[0], vis_on_window[1],
                                                           len(vis_onset_kernel_indices))
                    post_vis_on_idx = np.where(vis_on_peri_event_window >= 0)[0]


                if x_axis_kernel == 'saccade_dir_nasal':

                    # get only post-saccade time window
                    saccade_nasal_kernels = saccade_on_kernels - saccade_dir_kernels
                    if kernel_metric == 'mean-post-saccade':
                        saccade_nasal_kernels_post_saccade_mean = np.mean(saccade_nasal_kernels[:, post_saccade_idx], axis=1)
                        exp_x_vals = saccade_nasal_kernels_post_saccade_mean
                    elif kernel_metric == 'mean':
                        saccade_nasal_kernels_post_saccade_mean = np.mean(saccade_nasal_kernels, axis=1)
                        exp_x_vals = saccade_nasal_kernels_post_saccade_mean
                    elif kernel_metric == 'peak':
                        # include pre-saccade time window (?)
                        saccade_nasal_kernels_peak_loc = np.argmax(np.abs(saccade_nasal_kernels), axis=1)
                        # saccade_nasal_kernels_peak_loc = np.argmax(np.abs(saccade_nasal_kernels[:, post_saccade_idx]), axis=1)
                        exp_x_vals = []
                        for n_neuron in np.arange(np.shape(saccade_nasal_kernels)[0]):
                            exp_x_vals.append(saccade_nasal_kernels[n_neuron, saccade_nasal_kernels_peak_loc[n_neuron]])
                    elif kernel_metric == 'peak-post-saccade':
                        saccade_nasal_kernels_peak_loc = np.argmax(np.abs(saccade_nasal_kernels[:, post_saccade_idx]), axis=1)
                        exp_x_vals = []
                        for n_neuron in np.arange(np.shape(saccade_nasal_kernels)[0]):
                            exp_x_vals.append(saccade_nasal_kernels[n_neuron, saccade_nasal_kernels_peak_loc[n_neuron]])


                elif x_axis_kernel == 'vis_dir_nasal':

                    vis_nasal_kernels = vis_on_kernels - vis_dir_kernels
                    exp_x_vals = np.mean(vis_nasal_kernels[:, post_vis_on_idx], axis=1)

                elif x_axis_kernel == 'vis_dir_diff':

                    vis_nasal_kernels = vis_on_kernels - vis_dir_kernels
                    vis_temporal_kernels = vis_on_kernels + vis_dir_kernels
                    # temporal - nasal = (vis on + vis_dir) - (vis_on - vis_dir) = 2 * vis_dir
                    exp_x_vals = 2 * np.mean(vis_dir_kernels[:, post_vis_on_idx], axis=1)  # take the mean across time

                if y_axis_kernel == 'saccade_dir_temporal':
                    saccade_temporal_kernels = saccade_on_kernels + saccade_dir_kernels
                    # get only post-saccade time window
                    if kernel_metric == 'mean-post-saccade':
                        saccade_temporal_kernels_post_saccade_mean = np.mean(saccade_temporal_kernels[:, post_saccade_idx],
                                                                          axis=1)
                        exp_y_vals = saccade_temporal_kernels_post_saccade_mean
                    elif kernel_metric == 'mean':
                        saccade_temporal_kernels_post_saccade_mean = np.mean(saccade_temporal_kernels, axis=1)
                        exp_y_vals = saccade_temporal_kernels_post_saccade_mean
                    elif kernel_metric == 'peak':
                        # include pre-saccade time window (?)
                        saccade_temporal_kernels_peak_loc = np.argmax(np.abs(saccade_temporal_kernels), axis=1)
                        # saccade_nasal_kernels_peak_loc = np.argmax(np.abs(saccade_nasal_kernels[:, post_saccade_idx]), axis=1)
                        exp_y_vals = []
                        for n_neuron in np.arange(np.shape(saccade_temporal_kernels)[0]):
                            exp_y_vals.append(saccade_temporal_kernels[n_neuron, saccade_temporal_kernels_peak_loc[n_neuron]])
                    elif kernel_metric == 'peak-post-saccade':
                        saccade_temporal_kernels_peak_loc = np.argmax(np.abs(saccade_temporal_kernels[:, post_saccade_idx]), axis=1)
                        exp_y_vals = []
                        for n_neuron in np.arange(np.shape(saccade_temporal_kernels)[0]):
                            exp_y_vals.append(saccade_temporal_kernels[n_neuron, saccade_temporal_kernels_peak_loc[n_neuron]])

                elif y_axis_kernel == 'vis_dir_temporal':

                    vis_temporal_kernels = vis_on_kernels + vis_dir_kernels
                    exp_y_vals = np.mean(vis_temporal_kernels[:, post_vis_on_idx], axis=1)

                elif y_axis_kernel == 'saccade_dir_diff':

                    exp_y_vals = 2 * np.mean(saccade_dir_kernels, axis=1)  # take the mean across time

                # Make plot for individual experiment
                with plt.style.context(splstyle.get_style('nature-reviews')):

                    fig, ax = plt.subplots()
                    fig.set_size_inches(4, 4)

                    ax.scatter(exp_x_vals, exp_y_vals, s=9, color='black')

                    ax.set_xlabel(x_axis_kernel, size=11)
                    ax.set_ylabel(y_axis_kernel, size=11)
                    ax.set_title(exp_id, size=11)
                    fig_name = '%s_%s_%s_vs_%s_kernel_model_%s' % (exp_type, exp_id, x_axis_kernel, y_axis_kernel, model_X_set_to_plot)

                    if neuron_subset_condition is not None:
                        fig_name += '_%s' % neuron_subset_condition
                    else:
                        fig_name += '_all_neurons'

                    if neuron_subset_condition == 'better_than_null':
                        fig_name += '_null_X_set_%s' % null_X_set

                    fig_name += '_%s' % kernel_metric

                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                plt.close(fig)

                # accumulate neurons across all experiments
                all_exp_x_vals.extend(exp_x_vals)
                all_exp_y_vals.extend(exp_y_vals)
                subject = exp_id.split('_')[0]
                exp = exp_id.split('_')[1]
                all_subject.extend(np.repeat(subject, len(exp_x_vals)))
                all_exp.extend(np.repeat(exp, len(exp_x_vals)))

                if plot_indv_lobf:
                    # fit line of best fit
                    lobf_result = sstats.linregress(x=exp_x_vals, y=exp_y_vals)
                    exp_intercepts.append(lobf_result.intercept)
                    exp_slopes.append(lobf_result.slope)


            # Make plot for all experiments
            with plt.style.context(splstyle.get_style('nature-reviews')):

                fig, ax = plt.subplots()
                fig.set_size_inches(4, 4)

                print('Number of neurons plotted: %.f' % len(all_exp_x_vals))
                ax.scatter(all_exp_x_vals, all_exp_y_vals, s=9, color='black')

                if same_x_y_range:
                    both_x_y_vals = np.concatenate([all_exp_x_vals, all_exp_y_vals])
                    both_min = np.min(both_x_y_vals)
                    both_max = np.max(both_x_y_vals)
                    ax.set_xlim([both_min, both_max])
                    ax.set_ylim([both_min, both_max])

                if plot_indv_lobf:
                    for slope, intercept in zip(exp_slopes, exp_intercepts):

                        if same_x_y_range:
                            x_vals = np.linspace(both_min, both_max, 100)
                        else:
                            min_x = np.min(all_exp_x_vals)
                            max_x = np.max(all_exp_x_vals)
                            x_vals = np.linspace(min_x, max_x, 100)

                        ax.plot(x_vals, intercept + x_vals * slope, color='gray', lw=1)

                if process_params[process]['do_stats']:

                    # Pearson correlation
                    pearson_r, pearson_p_val = sstats.pearsonr(all_exp_x_vals, all_exp_y_vals)
                    print('Pearson r : %.3f' % pearson_r)
                    print('Pearson p val: %.5f' % pearson_p_val)

                    # Linear mixed effects models
                    kernel_weights_df = pd.DataFrame.from_dict({
                        'xval': all_exp_x_vals,
                        'yval': all_exp_y_vals,
                        'subject': all_subject,
                        'exp': all_exp,
                    })

                    x_y_random_intercept_md = Lmer("yval ~ xval + (1 | subject)",
                                                       data=kernel_weights_df)
                    x_y_random_intercept_md_fitted = x_y_random_intercept_md.fit()
                    x_p_val = sstats.t.sf(abs(x_y_random_intercept_md_fitted['T-stat']['xval']),
                                                x_y_random_intercept_md_fitted['DF']['xval'])
                    print('Linear mixed effects model p val %.5f' % x_p_val)

                    print('Random intercept + slope model')
                    x_y_random_intercept_and_slope_md = Lmer("yval ~ xval + (1 + xval | subject)",
                                                   data=kernel_weights_df)
                    x_y_random_intercept_and_slope_md_fitted = x_y_random_intercept_and_slope_md.fit()
                    x_rand_int_slope_p_val = sstats.t.sf(abs(x_y_random_intercept_and_slope_md_fitted['T-stat']['xval']),
                                          x_y_random_intercept_and_slope_md_fitted['DF']['xval'])
                    print('Linear mixed effects model p val %.5f' % x_rand_int_slope_p_val)

                ax.set_xlabel(x_axis_kernel, size=11)
                ax.set_ylabel(y_axis_kernel, size=11)

                if process_params[process]['do_stats']:
                    # title_txt = 'All exps, pearson r: %.4f, pearson p-val: %.4f \n LME p val: %.4f' % (pearson_r, pearson_p_val, x_p_val)
                    # use random slope and intercept model instead
                    title_txt = 'All exps, pearson r: %.4f, pearson p-val: %.4f \n LME p val: %.4f' % (pearson_r, pearson_p_val, x_rand_int_slope_p_val)
                else:
                    title_txt = 'All exps'

                ax.set_title(title_txt, size=11)
                fig_name = '%s_all_exp_%s_vs_%s_kernel_model_%s_%s' % (exp_type, x_axis_kernel, y_axis_kernel, model_X_set_to_plot, kernel_metric)

                if neuron_subset_condition is not None:
                    fig_name += '_%s' % neuron_subset_condition
                else:
                    fig_name += '_all_neurons'

                if neuron_subset_condition == 'better_than_null':
                    fig_name += '_null_X_set_%s' % null_X_set

                fig_path = os.path.join(fig_folder, fig_name)
                fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                print('Saved figure to %s' % fig_path)


            plt.close(fig)
        if process == 'plot_kernel_train_test':

            model_X_set_to_plot = process_params[process]['model_X_set_to_plot']
            null_X_set = process_params[process]['null_X_set']
            kernels_to_plot = process_params[process]['kernels_to_plot']
            fig_folder = process_params[process]['fig_folder']
            neuron_subset_condition = process_params[process]['neuron_subset_condition']
            explained_var_threshold = process_params[process]['explained_var_threshold']
            exp_type = process_params[process]['exp_type']
            same_x_y_range = process_params[process]['same_x_y_range']
            plot_indv_lobf = process_params[process]['plot_indv_lobf']
            kernel_metric = process_params[process]['kernel_metric']

            if not os.path.isdir(fig_folder):
                os.makedirs(fig_folder)

            regression_results_folder = process_params[process]['regression_results_folder']
            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % exp_type))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in
                       regression_result_files]

            all_exp_kernel_vals_per_cv = defaultdict(list)
            all_subject = []
            all_exp = []

            exp_intercepts = []
            exp_slopes = []


            for exp_id in exp_ids:

                regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*%s*.npz' % (exp_id, exp_type)))[0],
                    allow_pickle=True)


                X_sets_names = regression_result['X_sets_names']
                explained_var_per_X_set = regression_result['explained_var_per_X_set']

                # Subset neurons based on some condition

                if neuron_subset_condition == 'sig_saccade_neurons':
                    # Get neurons with significant saccade kernel
                    model_a = 'vis_and_saccade_shuffled_and_running'
                    model_b = 'vis_and_saccade_and_running'

                    model_a_idx = np.where(X_sets_names == model_a)[0][0]
                    model_b_idx = np.where(X_sets_names == model_b)[0][0]

                    model_a_explained_var = explained_var_per_X_set[:, model_a_idx]
                    model_b_explained_var = explained_var_per_X_set[:, model_b_idx]

                    b_minus_a = model_b_explained_var - model_a_explained_var

                    subset_idx = np.where(
                        (model_a_explained_var > explained_var_threshold) &
                        (model_b_explained_var > explained_var_threshold) &
                        (model_b_explained_var > model_a_explained_var)
                    )[0]

                    vis_only_neurons_subset_idx = np.where(
                        (model_a_explained_var > 0) &
                        (model_b_explained_var <= model_a_explained_var)
                    )[0]

                    subset_idx = np.where(
                        (model_a_explained_var > explained_var_threshold) &
                        (model_b_explained_var > explained_var_threshold) &
                        (model_b_explained_var > model_a_explained_var)
                    )[0]

                elif neuron_subset_condition == 'better_than_null':

                    model_a = null_X_set
                    model_b = model_X_set_to_plot

                    model_a_idx = np.where(X_sets_names == model_a)[0][0]
                    model_b_idx = np.where(X_sets_names == model_b)[0][0]

                    model_a_explained_var = explained_var_per_X_set[:, model_a_idx]
                    model_b_explained_var = explained_var_per_X_set[:, model_b_idx]

                    subset_idx = np.where(
                        (model_b_explained_var > model_a_explained_var) &
                        (model_b_explained_var > explained_var_threshold)
                    )[0]

                elif neuron_subset_condition == 'both_vis_dir_and_saccade_dir_selective':

                    # Get sig saccade dir neurons
                    model_a_idx = np.where(X_sets_names == 'vis_and_saccade_and_running')[0][0]
                    model_b_idx = np.where(X_sets_names == 'vis_and_saccade_on_and_running')[0][0]
                    model_a_explained_var = explained_var_per_X_set[:, model_a_idx]
                    model_b_explained_var = explained_var_per_X_set[:, model_b_idx]

                    saccade_dir_sig_idx = np.where(
                        (model_a_explained_var > model_b_explained_var) &
                        (model_a_explained_var > explained_var_threshold)
                    )[0]

                    # Get sig vis dir neurons
                    model_a_idx = np.where(X_sets_names == 'vis_and_saccade_and_running')[0][0]
                    model_b_idx = np.where(X_sets_names == 'vis_on_and_saccade_and_running')[0][0]
                    model_a_explained_var = explained_var_per_X_set[:, model_a_idx]
                    model_b_explained_var = explained_var_per_X_set[:, model_b_idx]

                    vis_dir_sig_idx = np.where(
                        (model_a_explained_var > model_b_explained_var) &
                        (model_a_explained_var > explained_var_threshold)
                    )[0]

                    subset_idx = np.intersect1d(saccade_dir_sig_idx, vis_dir_sig_idx)

                else:
                    subset_idx = np.arange(0, np.shape(explained_var_per_X_set)[0])

                # Get the kernels
                model_weights_per_X_set = regression_result['model_weights_per_X_set'].item()
                X_set_weights = model_weights_per_X_set[model_X_set_to_plot]  # numCV x numNeuron x numFeatures
                n_cv_folds = np.shape(X_set_weights)[0]
                feature_name_and_indices = regression_result['feature_indices_per_X_set'].item()[model_X_set_to_plot]
                regression_kernel_windows = regression_result['regression_kernel_windows']
                saccade_on_kernel_idx = np.where(regression_result['regression_kernel_names'] == 'saccade_on')[0][0]
                saccade_on_window = regression_kernel_windows[saccade_on_kernel_idx]

                saccade_onset_kernel_indices = feature_name_and_indices['saccade_on']
                saccade_dir_kernel_indices = feature_name_and_indices['saccade_dir']

                saccade_on_kernels = X_set_weights[:, subset_idx, :][:, :, saccade_onset_kernel_indices]
                saccade_dir_kernels = X_set_weights[:, subset_idx, :][:, :, saccade_dir_kernel_indices]

                saccade_on_peri_event_window = np.linspace(saccade_on_window[0], saccade_on_window[1],
                                                           len(saccade_onset_kernel_indices))
                post_saccade_idx = np.where(saccade_on_peri_event_window >= 0)[0]
                for kernel_name in kernels_to_plot:

                    if kernel_name == 'saccade_temporal':
                        saccade_temporal_kernels = saccade_on_kernels + saccade_dir_kernels
                        # get only post-saccade time window
                        if kernel_metric == 'mean':
                            kernel_val = np.mean(saccade_temporal_kernels[:, :, post_saccade_idx],
                                                                                 axis=2)
                        elif kernel_metric == 'peak':
                            # TODO
                            # include pre-saccade time window (?)
                            saccade_temporal_kernels_peak_loc = np.argmax(np.abs(saccade_temporal_kernels), axis=1)
                            # saccade_nasal_kernels_peak_loc = np.argmax(np.abs(saccade_nasal_kernels[:, post_saccade_idx]), axis=1)
                            exp_y_vals = []
                            for n_neuron in np.arange(np.shape(saccade_temporal_kernels)[0]):
                                exp_y_vals.append(
                                    saccade_temporal_kernels[n_neuron, saccade_temporal_kernels_peak_loc[n_neuron]])


                    elif kernel_name == 'saccade_nasal':

                        # get only post-saccade time window
                        saccade_nasal_kernels = saccade_on_kernels - saccade_dir_kernels
                        if kernel_metric == 'mean':
                            kernel_val = np.mean(saccade_nasal_kernels[:, :, post_saccade_idx], axis=2)
                        elif kernel_metric == 'peak':
                            # include pre-saccade time window (?)
                            saccade_nasal_kernels_peak_loc = np.argmax(np.abs(saccade_nasal_kernels), axis=1)
                            # saccade_nasal_kernels_peak_loc = np.argmax(np.abs(saccade_nasal_kernels[:, post_saccade_idx]), axis=1)
                            exp_x_vals = []
                            for n_neuron in np.arange(np.shape(saccade_nasal_kernels)[0]):
                                exp_x_vals.append(
                                    saccade_nasal_kernels[n_neuron, saccade_nasal_kernels_peak_loc[n_neuron]])

                    all_exp_kernel_vals_per_cv[kernel_name].append(kernel_val)


            for kernel_name in kernels_to_plot:

                kernel_cv_vals = np.concatenate(all_exp_kernel_vals_per_cv[kernel_name], axis=1)

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plt.subplots(n_cv_folds, n_cv_folds, sharex=True, sharey=True)
                    fig.set_size_inches(n_cv_folds * 2, n_cv_folds * 2)

                    all_vals_min = np.min(kernel_cv_vals.flatten())
                    all_vals_max = np.max(kernel_cv_vals.flatten())

                    for cv_j, cv_k in itertools.product(np.arange(n_cv_folds), np.arange(n_cv_folds)):

                        axs[cv_j, cv_k].scatter(kernel_cv_vals[cv_j, :],
                                                kernel_cv_vals[cv_k, :], color='black', s=8, lw=0)

                        axs[cv_j, cv_k].set_xlabel('CV %.f' % (cv_j+1), size=11)
                        axs[cv_j, cv_k].set_ylabel('CV %.f' % (cv_k+1), size=11)
                        axs[cv_j, cv_k].set_xlim([all_vals_min, all_vals_max])
                        axs[cv_j, cv_k].set_ylim([all_vals_min, all_vals_max])

                    fig.suptitle('All %s exp %s %s' % (exp_type, kernel_name, kernel_metric), size=11)
                    fig_name = 'all_%s_exp_train_test_%s_kernel_%s' % (exp_type, kernel_name, kernel_metric)
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                plt.close(fig)



        if process == 'plot_running_weights':

            print('Plotting running weights')

            fig_folder = process_params[process]['fig_folder']
            both_exp_model = process_params[process]['both_exp_model']
            grating_exp_model = process_params[process]['grating_exp_model']
            gray_exp_model = process_params[process]['gray_exp_model']

            if not os.path.isdir(fig_folder):
                os.makedirs(fig_folder)

            regression_results_folder = process_params[process]['regression_results_folder']
            regression_result_files = glob.glob(os.path.join(regression_results_folder,
                                                             '*%s*npz' % 'both'))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in
                       regression_result_files]

            all_both_exp_running_weights = []
            all_gray_exp_running_weights = []
            all_grating_exp_running_weights = []

            for exp_id in exp_ids:

                # Both experiments
                both_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*%s*.npz' % (exp_id, 'both')))[0],
                    allow_pickle=True)

                X_sets_names = both_regression_result['X_sets_names']
                explained_var_per_X_set = both_regression_result['explained_var_per_X_set']
                
                # Get the running weights for both exp
                feature_name_and_indices = both_regression_result['feature_indices_per_X_set'].item()[both_exp_model]
                running_kernel_idx = np.where(both_regression_result['regression_kernel_names'] == 'running')[0][0]
                model_weights_per_X_set = both_regression_result['model_weights_per_X_set'].item()
                both_exp_X_set_weights = model_weights_per_X_set[both_exp_model]  # numCV x numNeuron x numFeatures

                both_exp_running_weights = np.mean(both_exp_X_set_weights[:, :, running_kernel_idx], axis=0)
                all_both_exp_running_weights.append(both_exp_running_weights)

                # Gray experiments
                gray_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*%s*.npz' % (exp_id, 'gray')))[0],
                    allow_pickle=True)
                feature_name_and_indices = gray_regression_result['feature_indices_per_X_set'].item()[gray_exp_model]
                running_kernel_idx = np.where(gray_regression_result['regression_kernel_names'] == 'running')[0][0]
                model_weights_per_X_set = gray_regression_result['model_weights_per_X_set'].item()
                gray_exp_X_set_weights = model_weights_per_X_set[gray_exp_model]  # numCV x numNeuron x numFeatures

                gray_exp_running_weights = np.mean(gray_exp_X_set_weights[:, :, running_kernel_idx], axis=0)
                all_gray_exp_running_weights.append(gray_exp_running_weights)

                # Grating experiments
                grating_regression_result = np.load(
                    glob.glob(os.path.join(regression_results_folder, '*%s*%s*.npz' % (exp_id, 'grating')))[0],
                    allow_pickle=True)
                feature_name_and_indices = grating_regression_result['feature_indices_per_X_set'].item()[grating_exp_model]
                running_kernel_idx = np.where(grating_regression_result['regression_kernel_names'] == 'running')[0][0]
                model_weights_per_X_set = grating_regression_result['model_weights_per_X_set'].item()
                grating_exp_X_set_weights = model_weights_per_X_set[grating_exp_model]  # numCV x numNeuron x numFeatures

                grating_exp_running_weights = np.mean(grating_exp_X_set_weights[:, :, running_kernel_idx], axis=0)
                all_grating_exp_running_weights.append(grating_exp_running_weights)

                # Plot the distribution of the weights
                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plot_running_weight_dist(both_exp_running_weights,
                                                        gray_exp_running_weights,
                                                        grating_exp_running_weights)

                fig_name = '%s_running_weight_distribution.png' % (exp_id)
                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                plt.close(fig)

            # Plot all experiments
            all_both_exp_running_weights = np.concatenate(all_both_exp_running_weights)
            all_gray_exp_running_weights = np.concatenate(all_gray_exp_running_weights)
            all_grating_exp_running_weights = np.concatenate(all_grating_exp_running_weights)

            # Truncate some outliers for visualization
            all_grating_exp_running_weights = all_grating_exp_running_weights[all_grating_exp_running_weights < 5]

            # pdb.set_trace()

            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, axs = plot_running_weight_dist(all_both_exp_running_weights,
                                                    all_gray_exp_running_weights,
                                                    all_grating_exp_running_weights)

            fig_name = 'all_exp_running_weight_distribution.png'
            fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

            plt.close(fig)
        
        if process == 'plot_before_after_saccade_exclusion_ev':

            exp_type = process_params[process]['exp_type']
            fig_folder = process_params[process]['fig_folder']
            before_regression_results_folder = process_params[process]['before_regression_results_folder']
            after_regression_results_folder = process_params[process]['after_regression_results_folder']
            metric_to_plot = process_params[process]['metric_to_plot']
            plot_kernel_traces = process_params[process]['plot_kernel_traces']
            
            before_regression_result_files = glob.glob(os.path.join(before_regression_results_folder,
                                                             '*%s*npz' % exp_type))
            exp_ids = ['_'.join(os.path.basename(fpath).split('.')[0].split('_')[0:2]) for fpath in
                       before_regression_result_files]

            all_exp_x_vals = []
            all_exp_y_vals = []
            all_subject = []
            all_exp = []

            prop_sig_neurons_before = []
            prop_sig_neurons_after = []

            for exp_id in exp_ids:
                
                before_regression_result = np.load(
                    glob.glob(os.path.join(before_regression_results_folder, '*%s*%s*.npz' % (exp_id, exp_type)))[0],
                    allow_pickle=True)

                after_regression_result = np.load(
                    glob.glob(os.path.join(after_regression_results_folder, '*%s*%s*.npz' % (exp_id, exp_type)))[0],
                    allow_pickle=True)
                
                X_sets_names = before_regression_result['X_sets_names']
                before_explained_var_per_X_set = before_regression_result[metric_to_plot]
                after_explained_var_per_X_set = after_regression_result[metric_to_plot]

                model_X_set_to_plot = 'saccade_and_running'
                X_set_idx = np.where(X_sets_names == model_X_set_to_plot)[0][0]

                x_vals = before_explained_var_per_X_set[:, X_set_idx]
                y_vals = after_explained_var_per_X_set[:, X_set_idx]

                prop_sig_neurons_before.append(np.sum(x_vals > 0) / len(x_vals))
                prop_sig_neurons_after.append(np.sum(y_vals > 0) / len(y_vals))

                x_y_vals = np.concatenate([
                    before_explained_var_per_X_set[:, X_set_idx],
                    after_explained_var_per_X_set[:, X_set_idx]
                ])

                x_y_min = -0.1 # np.nanmin(x_y_vals)
                x_y_max = np.nanmax(x_y_vals)
                unity_vals = np.linspace(x_y_min, x_y_max, 100)


                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plt.subplots()
                    fig.set_size_inches(4, 4)

                    ax.plot(unity_vals, unity_vals, linestyle='--', color='gray')
                    ax.axvline(0, linestyle='--', lw=0.5, color='gray', alpha=0.5)
                    ax.axhline(0, linestyle='--', lw=0.5, color='gray', alpha=0.5)
                    ax.scatter(before_explained_var_per_X_set[:, X_set_idx],
                               after_explained_var_per_X_set[:, X_set_idx], lw=0, color='black', s=8)
                    

                    ax.set_xlim([x_y_min, x_y_max])
                    ax.set_ylim([x_y_min, x_y_max])

                    ax.set_xlabel('All saccades', size=11)
                    ax.set_ylabel('In-screen saccades', size=11)
                    ax.set_title('%s %s %s' % (exp_id, exp_type, metric_to_plot), size=11)

                    fig_name = '%s_%s_before_after_saccade_exclusion_%s_%s' % (exp_type, exp_id, metric_to_plot, model_X_set_to_plot)
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                plt.close(fig)

                if plot_kernel_traces:

                    both_sig_indices = np.where((x_vals > 0) & (y_vals > 0))[0]


                    before_model_weights_per_X_set = before_regression_result['model_weights_per_X_set'].item()
                    before_X_set_weights = np.mean(before_model_weights_per_X_set[model_X_set_to_plot], axis=0)  # numCV x numNeuron x numFeatures

                    after_model_weights_per_X_set = after_regression_result['model_weights_per_X_set'].item()
                    after_X_set_weights = np.mean(after_model_weights_per_X_set[model_X_set_to_plot], axis=0)  # numCV x numNeuron x numFeatures

                    feature_name_and_indices = before_regression_result['feature_indices_per_X_set'].item()[
                        model_X_set_to_plot]

                    saccade_on_idx = feature_name_and_indices['saccade_on']
                    saccade_dir_idx = feature_name_and_indices['saccade_dir']


                    regression_kernel_windows = before_regression_result['regression_kernel_windows']
                    saccade_on_kernel_idx = np.where(before_regression_result['regression_kernel_names'] == 'saccade_on')[0][0]
                    saccade_on_window = regression_kernel_windows[saccade_on_kernel_idx]

                    for neuron_idx in both_sig_indices:

                        with plt.style.context(splstyle.get_style('nature-reviews')):

                            fig, axs = plt.subplots(1, 2)
                            fig.set_size_inches(6, 3)

                            axs[0].set_title('Saccade onset', size=11)
                            axs[0].plot(before_X_set_weights[neuron_idx, saccade_on_idx], color='black')
                            axs[0].plot(after_X_set_weights[neuron_idx, saccade_on_idx], color='red')

                            axs[1].set_title('Saccade direction', size=11)
                            axs[1].plot(before_X_set_weights[neuron_idx, saccade_dir_idx], color='black', label='All saccades')
                            axs[1].plot(after_X_set_weights[neuron_idx, saccade_dir_idx], color='red', label='In-screen saccades')

                            axs[1].legend(bbox_to_anchor=(0.8, 0.7))

                        fig_name = '%s_%s_%s_before_after_kernels' % (exp_id, exp_type, neuron_idx)
                        fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                        plt.close(fig)

            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, ax = plt.subplots()
                fig.set_size_inches(3, 3)
                for n_exp, exp_id in enumerate(exp_ids):

                    ax.plot([0, 1], [prop_sig_neurons_before[n_exp], prop_sig_neurons_after[n_exp]],
                            label=exp_id, lw=1)

                ax.legend(bbox_to_anchor=(0.8, 0.8))
                ax.set_xticks([0, 1])
                ax.set_xlim([-0.5, 1.5])
                ax.set_xticklabels(['All saccade', 'In-screen saccade'])
                ax.set_ylabel('Prop significant neurons', size=11)
                ax.set_title('%s %s' % (exp_type, metric_to_plot), size=11)
                fig_name = '%s_%s_each_exp_before_after_exclusion_prop_sig_neurons' % (exp_type, metric_to_plot)
                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

            plt.close(fig)

        if process == 'compare_exp_times':

            data_folders = process_params[process]['data_folders']
            fig_folder = process_params[process]['fig_folder']

            data_list = []
            for data_folder in data_folders:
                data = load_data(data_folder=data_folder,
                             file_types_to_load=process_params[process]['file_types_to_load'])

                data_list.append(data)


            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, axs = plt.subplots(len(data), len(data_list), sharex=False, sharey=False)
                fig.set_size_inches(8, 8)
                for n_data, data in enumerate(data_list):
                    n_exp = 0
                    for exp_id, exp_data in data.items():
                        vis_exp_times = exp_data['_windowVis'].flatten()
                        axs[n_exp, n_data].hist(np.diff(vis_exp_times), lw=0, color='black', bins=50)
                        axs[n_exp, n_data].set_title('Mean %.3f Min: %.3f Max: %.3f' % (np.mean(np.diff(vis_exp_times)),
                                                                                        np.min(np.diff(vis_exp_times)),
                                                                                 np.max(np.diff(vis_exp_times))), size=11)


                        n_exp += 1

            fig_name = 'old_new_data_vis_exp_times'
            fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()