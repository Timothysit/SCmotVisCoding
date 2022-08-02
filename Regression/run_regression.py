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

import matplotlib.pyplot as plt
import matplotlib as mpl
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

def make_X_Y_for_regression(exp_data, feature_set=['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir'],
                            feature_time_windows={'vis_on': [-1.0, 3.0], 'vis_dir': [-1.0, 3.0],
                                                  'saccade_on': [-1.0, 3.0], 'saccade_dir': [-1.0, 3.0]},
                            neural_preprocessing_steps=['zscore']):
    """
    Make feature matrix (or design matrix) X and target matrix Y from experiment data

    Parameters


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

    sec_per_time_samples = np.mean(np.diff(vis_exp_times))
    num_time_samples = int(len(vis_exp_times))


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

                feature_mat[row_idx, col_idx] = 1


        elif feature_name == 'vis_dir':
            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            grating_onset_times = grating_intervals[:, 0]
            # grating_offset_times = grating_intervals[:, 1]

            for n_trial, onset_time in enumerate(grating_onset_times):

                if onset_time + feature_time_window[1] > vis_exp_times[-1]:
                    print('onset time exceeds experiment time, skipping')
                    continue

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

        elif feature_name == 'saccade_on':
            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

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

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                feature_mat[row_idx, col_idx] = 1

        elif feature_name == 'saccade_dir':
            feature_time_window = feature_time_windows[feature_name]
            num_sample_in_window = int((feature_time_window[1] - feature_time_window[0]) / sec_per_time_samples)
            feature_mat = np.zeros((num_time_samples, num_sample_in_window))
            col_idx = np.arange(0, num_sample_in_window)

            for n_saccade_time, onset_time in enumerate(vis_exp_saccade_onset_times):

                if onset_time + feature_time_window[1] > vis_exp_times[-1]:
                    print('onset time exceeds experiment time, skipping')
                    continue

                start_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[0])))
                end_sample = np.argmin(np.abs(vis_exp_times - (onset_time + feature_time_window[1])))
                row_idx = np.arange(start_sample, end_sample)

                if len(row_idx) > len(col_idx):
                    row_idx = row_idx[0:len(col_idx)]

                feature_mat[row_idx, col_idx] = saccade_dirs[n_saccade_time]

        else:
            print('%s is not a valid feature name' % feature_name)

        feature_matrices.append(feature_mat)

    X = np.concatenate(feature_matrices, axis=1)

    # Make target matrix Y
    Y = neural_activity
    if 'zscore' in neural_preprocessing_steps:
        Y  = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    return X, Y


def fit_regression_model(X, Y, model_type='Ridge', train_test_split_method='half'):



    regression_result = dict()

    if train_test_split_method == 'half':
        num_time_points = np.shape(Y)[0]
        first_half_indices = np.arange(0, int(num_time_points/2))
        second_half_indices = np.arange(int(num_time_points/2), num_time_points)
        train_indices = [first_half_indices, second_half_indices]
        test_indices = [second_half_indices, first_half_indices]

    if model_type == 'Ridge':
        model = sklinear.Ridge(alpha=1.0, fit_intercept=False)

    num_cv_set = len(train_indices)
    num_neurons = np.shape(Y)[1]
    explained_variance_per_cv_set = np.zeros((num_neurons, num_cv_set))
    Y_test_hat_per_cv_set = []
    Y_test_per_cv_set = []

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

    regression_result['Y_test_hat_per_cv_set'] = Y_test_hat_per_cv_set
    regression_result['explained_variance_per_cv_set'] = explained_variance_per_cv_set
    regression_result['Y_test_per_cv_set'] = np.array(Y_test_per_cv_set)

    return regression_result

def main():

    available_processes = ['load_data', 'plot_data', 'fit_regression_model', 'plot_regression_model_explained_var']
    processes_to_run = ['plot_regression_model_explained_var']
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
            dataset='grating',
        ),
        'plot_regression_model_explained_var': dict(
            regression_results_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Data/RegressionResults',
            fig_folder='/Volumes/Macintosh HD/Users/timothysit/SCmotVisCoding/Figures/regression',
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

            X_sets_to_compare = {'bias_only': ['bias'],
                                 'vis_on_only': ['bias', 'vis_on'],
                                 'vis': ['bias', 'vis_on', 'vis_dir'],
                                 'saccade_on_only': ['bias', 'saccade_on'],
                                 'saccade': ['bias','saccade_on', 'saccade_dir'],
                                 'vis_and_saccade': ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir']
                                 }
            # feature_set = ['bias', 'vis_on', 'vis_dir', 'saccade_on', 'saccade_dir']

            for exp_id, exp_data in data.items():

                num_X_set = len(X_sets_to_compare.keys())
                num_neurons = np.shape(exp_data['_tracesVis'])[1]
                explained_var_per_X_set = np.zeros((num_neurons, num_X_set))
                exp_regression_result = dict()
                exp_regression_result['X_sets_names'] = list(X_sets_to_compare.keys())
                Y_test_hat_per_X_set = []

                for n_X_set, (X_set_name, feature_set) in enumerate(X_sets_to_compare.items()):
                    X, Y = make_X_Y_for_regression(exp_data, feature_set=feature_set)

                    regression_result = fit_regression_model(X, Y)

                    explained_var_per_X_set[:, n_X_set] = np.mean(regression_result['explained_variance_per_cv_set'], axis=1)
                    Y_test_hat_per_X_set.append(regression_result['Y_test_hat_per_cv_set'])

                exp_regression_result['explained_var_per_X_set'] = explained_var_per_X_set
                exp_regression_result['Y_test'] = regression_result['Y_test_per_cv_set']
                exp_regression_result['Y_test_hat'] = np.array(Y_test_hat_per_X_set)


                regression_result_savename = '%s_regression_results.npz' % exp_id
                regression_result_savepath = os.path.join(regression_results_folder, regression_result_savename)
                np.savez(regression_result_savepath, **exp_regression_result)

        if process == 'plot_regression_model_explained_var':

            regression_result_files = glob.glob(os.path.join(process_params[process]['regression_results_folder'],
                                                             '*npz'))
            text_size = 11

            for fpath in regression_result_files:

                regression_result = np.load(fpath)

                X_sets_names = regression_result['X_sets_names']
                explained_var_per_X_set = regression_result['explained_var_per_X_set']

                X_sets_to_compare = [
                    ['vis_on_only', 'vis'],
                    ['saccade_on_only', 'saccade'],
                    ['vis_on_only', 'saccade_on_only']
                ]

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plt.subplots(1, len(X_sets_to_compare))
                    fig.set_size_inches(len(X_sets_to_compare)*3, 3)
                    for n_comparison in np.arange(0, len(X_sets_to_compare)):

                        model_a = X_sets_to_compare[n_comparison][0]
                        model_b = X_sets_to_compare[n_comparison][1]
                        model_a_idx = np.where(X_sets_names == model_a)
                        model_b_idx = np.where(X_sets_names == model_b)

                        axs[n_comparison].scatter(explained_var_per_X_set[:, model_a_idx],
                                          explained_var_per_X_set[:, model_b_idx], color='black',
                                                  s=10)

                        both_model_explained_var = np.concatenate([explained_var_per_X_set[:, model_a_idx],
                                                                  explained_var_per_X_set[:, model_b_idx]])

                        both_model_min = np.min(both_model_explained_var)
                        both_model_max = np.max(both_model_explained_var)
                        unity_vals = np.linspace(both_model_min, both_model_max, 100)
                        axs[n_comparison].plot(unity_vals, unity_vals, linestyle='--', color='gray', alpha=0.5)
                        axs[n_comparison].set_xlabel(model_a, size=text_size)
                        axs[n_comparison].set_ylabel(model_b, size=text_size)

                    fig_name = '%s_explained_variance_per_X_set_comparison' % exp_id
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbbox_inches='tight')






if __name__ == '__main__':
    main()