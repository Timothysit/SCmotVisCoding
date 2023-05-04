# This is a sample Python script.
import os  # library for basic functionalities like mkdir ecc
import glob  # library containing dir
import numpy as np  # library for maths (to install, open terminal, activate env, and type pip install numpy)
import scipy.stats as sstats  # library for stats
import sklearn  # library for machine learning (to install, open terminal, activate env, and type pip install sklearn)
from sklearn.svm import LinearSVC # help compact coding and interpreter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KernelDensity

import matplotlib.pyplot as plt  # sub-library of matplotlib for plotting
import matplotlib as mpl
import sciplotlib.style as splstyle  # sub-library of sciplotlib for fancypants plots, which is Tim's brainchild
import sciplotlib.text as spltext
import pdb  # python debugger
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from collections import defaultdict
import pandas as pd
from tqdm import tqdm  # loading bar


# Stats
import statsmodels.api as sm
from pymer4.models import Lmer


def load_data(path, data_type='motor'):
    """
    Load data used for decoding analysis.

    Parameters
    ----------
    path : str
        path to folder with the data files
    data_type : str
        'motor' : load motor data
        'visual' : load visual data
        'motor_time' : load motor data with time dimension (aligned to saccade onset)
        'visual_time' : load visual data with time dimension (aligned to visual stimulus onset)

    Returns
    -------
    resp_ls : list of numpy arrays
        length of list represent the total number of recording sessions
        each numpy array is a (num_neuron, num_trial) matrix
        The data is normalised.
        Z-scoring was computed by pooling the baseline activity prior to saccade initiation or
        visual stimulus presentation and using the stdev. from the pooled baseline activity.
        The responses are computed as the average within a window close to where the peak occurs.
    neun_ls : list of numpy arrays
        each numpy array has shape (num_neuron, ), and denotes whether the neuron is significant
        in the case of visual data (data_type == 'visual')
            1 : preference for temporal grating
            0 : preference for nasal grating
            NaN : no significant preference, or preferred grating is not nasal nor temporal
    tr_ls : list of numpy arrays
        each numpy array has shape (num_trial, ), and denotes the trial type
        in the case of visual data (data_type == 'visual')
            1 : temporal grating
            0 : nasal grating
        in the case of motor data (data_type == 'motor')
            1 : temporal direction
            0 : nasal direction
    resp_t_ls : list of numpy arrays, optional
        numpy arrays contained activity aligned to either visual onset or saccade onset
        for visual, ie. data_type = 'visual_w_time' : trial x time points x neuron
        for saccade: time x trial x neuron
    """

    # Get file paths
    if data_type == 'motor':
        file_list_resp = np.sort(glob.glob(os.path.join(path, '*matNeuronTrials*.npy')))
        file_list_neuron = np.sort(glob.glob(os.path.join(path, '*sigNeuronsLogical*.npy')))
        file_list_trial = np.sort(glob.glob(os.path.join(path, '*trial*.npy')))
        rec_name = [x[-21:-4] for x in file_list_resp]
    elif data_type == 'visual':
        file_list_resp = np.sort(glob.glob(os.path.join(path, '*avgActVec*.npy')))
        file_list_neuron = np.sort(glob.glob(os.path.join(path, '*prefVis*.npy')))
        file_list_trial = np.sort(glob.glob(os.path.join(path, '*dirOrder*.npy')))
        rec_name = [x[-21:-4] for x in file_list_resp]
    elif data_type == 'motor_w_time':
        file_list_resp = np.sort(glob.glob(os.path.join(path, '*matNeuronTrials*.npy')))
        file_list_neuron = np.sort(glob.glob(os.path.join(path, '*sigNeuronsLogical*.npy')))
        file_list_trial = np.sort(glob.glob(os.path.join(path, '*trial*.npy')))
        file_list_resp_t = np.sort(glob.glob(os.path.join(path, '_neuronSaccadeActivity*.npy')))
        rec_name = [x[-21:-4] for x in file_list_resp]
        aligned_time_file = os.path.join(path, 'saccadeWindow.npy')
    elif data_type == 'visual_w_time':
        file_list_resp = np.sort(glob.glob(os.path.join(path, '*avgActVec*.npy')))
        file_list_neuron = np.sort(glob.glob(os.path.join(path, '*prefVis*.npy')))
        file_list_trial = np.sort(glob.glob(os.path.join(path, '*dirOrder*.npy')))
        file_list_resp_t = np.sort(glob.glob(os.path.join(path, 'neuronVisActivity*')))
        file_list_aligned_time = np.sort(glob.glob(os.path.join(path, '*window*.npy')))
        # aligned_time_file = os.path.join(path, 'visWindow.npy')
        rec_name = [x[-21:-4] for x in file_list_resp]


    nFiles_r = len(file_list_resp)
    nFiles_n = len(file_list_neuron)
    nFiles_t = len(file_list_trial)

    # Load data
    if (nFiles_r != nFiles_n) or (nFiles_r != nFiles_t):
        print('Number of matNeuronTrial, sigNeuronsLogical, and trial files do not match, returning None')
        return None
    else:
        resp_ls = []
        neun_ls = []
        tr_ls = []

        if 'time' in data_type:
            resp_t_ls = []
            if data_type == 'visual_w_time':
                aligned_time = []
            for n_file in np.arange(nFiles_r):
                resp_ls.append(np.load(file_list_resp[n_file]))
                neun_ls.append(np.load(file_list_neuron[n_file]))
                tr_ls.append(np.load(file_list_trial[n_file]))
                resp_t_ls.append(np.load(file_list_resp_t[n_file]))
                if data_type == 'visual_w_time':
                    aligned_time.append(np.load(file_list_aligned_time[n_file]))
                else:
                    aligned_time = np.load(aligned_time_file)
        else:

            for r_file, n_file, t_file in zip(file_list_resp, file_list_neuron, file_list_trial):
                resp_ls.append(np.load(r_file))
                neun_ls.append(np.load(n_file).flatten())
                tr_ls.append(np.load(t_file).flatten())

    if 'time' in data_type:
        return resp_ls, neun_ls,tr_ls, rec_name, resp_t_ls, aligned_time
    else:
        return resp_ls, neun_ls, tr_ls, rec_name


def motor_decoder(resp, neun, tr, estimator=None):
    """

    :param resp:
    :param neun:
    :param tr:
    :param estimator:
    :return:
    """

    resp = resp[neun, :]
    sorted_resp = resp[:, np.argsort(tr)]
    #with plt.style.use(splstyle.get_style('nature-reviews')):
    '''fig, ax = plt.subplots()
    im = ax.imshow(sorted_resp, cmap='PiYG', vmin= -6, vmax = 6)
    ax.set_ylabel('Neurons')
    ax.set_xlabel('Trials')
    fig.colorbar(im)
    plt.show()
    plt.close(fig)'''

    nNeurons = sum(neun)
    resp_mean = np.nanmean(resp, axis = 1)
    resp = resp[~np.isnan(resp_mean), :]
    resp_mean = resp_mean[~np.isnan(resp_mean)]

    for iN in np.arange(nNeurons):
        resp[iN, np.isnan(resp[iN, :])] = resp_mean[iN]

    X = resp.T
    y = tr

    X = sstats.zscore(X, axis=0)
    classifier = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, penalty='l2', dual=False))
    cv_splitter = StratifiedKFold(n_splits=2)
    cv_results = cross_validate(classifier, X, y, cv=cv_splitter, return_estimator=True)
    score = cv_results['test_score']
    estimator = cv_results['estimator']
    imbalance = np.sum(y)/len(y)
    if imbalance < 0.5:
        imbalance = 1-imbalance

    performance = np.mean((score-imbalance)/(1-imbalance))
    print(score)
    print(imbalance)
    print(performance)

    return estimator, performance, score, imbalance

# Press the green button in the gutter to run the script.

def visual_decoder(resp, neun, tr, estimator=None):
    resp = resp[neun, :]
    sorted_resp = resp[:, np.argsort(tr)]
    # with plt.style.use(splstyle.get_style('nature-reviews')):
    '''fig, ax = plt.subplots()
    im = ax.imshow(sorted_resp, cmap='PiYG', vmin=-6, vmax=6, aspect='auto')
    ax.set_ylabel('Neurons')
    ax.set_xlabel('Trials')
    fig.colorbar(im)
    plt.show()
    plt.close(fig)'''

    nNeurons = sum(neun)
    resp_mean = np.nanmean(resp, axis=1)
    resp = resp[~np.isnan(resp_mean), :]
    resp_mean = resp_mean[~np.isnan(resp_mean)]

    for iN in np.arange(nNeurons):
        resp[iN, np.isnan(resp[iN, :])] = resp_mean[iN]

    X = resp.T
    y = tr

    X = sstats.zscore(X, axis=0)
    score = list()
    for iEst in estimator:

        y_hat = iEst.predict(X)
        score.append(np.sum(y==y_hat)/len(y))
    score = np.array(score)

    imbalance = np.sum(y) / len(y)
    if imbalance < 0.5:
        imbalance = 1 - imbalance
    performance = np.mean((score - imbalance) / (1 - imbalance))

    print(score)
    print(imbalance)
    print(performance)

    return performance, score, imbalance


def decode_trial_type(resp, neun, tr, estimator=None, cv_splitter=None,
                      pre_processing_steps=['exclude_nan_response', 'impute_nan_response', 'zscore_activity',
                                            'remove_zero_variance_neurons', 'only_include_sig_neurons'],
                      resp_other_cond=None, smallest_tolerable_var=0.0001, include_shuffles=False, num_shuffles=500,
                      plot_checks=False, verbose=True, fit_estimator=True, fig_folder=None):
    """
    General decoding function for decoding trial type from neural data
    Parameters
    ----------
    resp : numpy ndarray
        matrix with neural activity, with shape (num_neuron, num_trial)
    neun : numpy ndarray
        vector denoting which neurons are significant, with shape (num_neuron)
    tr : numpy ndarray
        vector denoting the identity of each trial, with shape (num_trial)
    estimator : sklearn estimator object
        sklearn estimator object for doing decoding
        can also be a pipeline object to include pre-processing steps
    cv_splitter : sklearn cv splitter object
        splitter that returns train and test set partitions
    pre_processing_steps : list
        supported pre-processing steps
            'exclude_nan_response' : excludes NaN in resp
            'impute_nan_response' : impute NaN in resp with mean response across all trials
    resp_other_cond : numpy ndarray
        response to the other condition (eg. resp is the response to visual stimulus, resp_other_cond is the
        motor response)
    smallest_tolerable_var : float
        smallest tolerable variance before neuron is excluded
    plot_checks : bool
        whether to plot diagnostic / summary plot
    verbose : bool
        whether to print out what the code is doing
    fig_folder : str
        path to save plots generated by this function

    Returns
    -------
    cv_results : dict
    weights_per_cv : numpy ndarray
    accuracy_per_shuffle : numpy ndarray
    """


    # Create a copy to compare with the pre-processed data
    original_resp = resp.copy()

    if 'only_include_sig_neurons' in pre_processing_steps:
        if verbose:
            print('Only including significant neurons')
        subset_neuron_bool = ~np.isnan(neun)
        resp = resp[subset_neuron_bool, :]

    # check for zero variance neurons
    var_per_neuron = np.nanvar(resp, axis=1)
    zero_var_neuron = np.where(np.abs(var_per_neuron) < smallest_tolerable_var)[0]
    num_zero_var_neuron = len(zero_var_neuron)
    # if num_zero_var_neuron > 0:
    #   pdb.set_trace()

    if 'exclude_nan_response' in pre_processing_steps:
        resp_mean = np.nanmean(resp, axis=1)  # take the mean response of each neuron across trials (excluding nan)
        subset_neuron_bool = ~np.isnan(resp_mean)

        if resp_other_cond is not None:
            resp_mean_other_cond = np.nanmean(resp_other_cond, axis=1)
            subset_neuron_bool_other_cond = ~np.isnan(resp_mean_other_cond)
            subset_neuron_bool = subset_neuron_bool & subset_neuron_bool_other_cond
            resp_other_cond = resp_other_cond[subset_neuron_bool, :]
            resp_mean_other_cond = resp_mean_other_cond[subset_neuron_bool]

        resp = resp[subset_neuron_bool, :]  # throw out neurons with NaNs in all trials
        num_all_nan_neurons = np.sum(np.isnan(resp_mean))
        resp_mean = resp_mean[subset_neuron_bool]
        if verbose:
            print('Original number of neurons: %.f' % np.shape(original_resp)[0])
            print('Found %.f neurons with NaNs in all trials' % num_all_nan_neurons)
            print('Number of neurons after subsetting: %.f' % len(resp_mean))

    if 'impute_nan_response' in pre_processing_steps:
        num_neurons = len(resp_mean)
        for iN in np.arange(num_neurons):
            resp[iN, np.isnan(resp[iN, :])] = resp_mean[iN]

        if resp_other_cond is not None:
            num_neurons_rep_other_cond = len(resp_mean_other_cond)
            for iN in np.arange(num_neurons_rep_other_cond):
                resp_other_cond[iN, np.isnan(resp_other_cond[iN, :])] = resp_mean_other_cond[iN]

    if 'remove_zero_variance_neurons' in pre_processing_steps:
        if verbose:
            print('Removing neurons with zero variance')
        var_per_neuron = np.var(resp, axis=1)
        subset_neuron_bool = np.abs(var_per_neuron) >= smallest_tolerable_var

        if resp_other_cond is not None:
            var_per_neuron_other_cond = np.var(resp_other_cond, axis=1)
            subset_neuron_other_cond_bool = np.abs(var_per_neuron_other_cond) >= smallest_tolerable_var
            subset_neuron_bool = subset_neuron_bool & subset_neuron_other_cond_bool

        resp = resp[subset_neuron_bool, :]

    # if 'remove_zero_variance_neurons_two_conditions' in pre_processing_steps:
    #     pdb.set_trace()

    X = resp.T  # reshape to (num_trial, num_neurons)
    y = tr

    if 'zscore_activity' in pre_processing_steps:
        if plot_checks:
            X_before_z_score = X.copy()

        X = sstats.zscore(X, axis=0)

        if plot_checks:
            with plt.style.context(splstyle.get_style('nature-reviews')):
                mean_response_of_each_neuron_before = np.mean(X_before_z_score, axis=0)
                mean_response_of_each_neuron_after = np.mean(X, axis=0)
                fig, axs = plt.subplots(1, 2, sharex=True)
                axs[0].hist(mean_response_of_each_neuron_before)
                axs[1].hist(mean_response_of_each_neuron_after)
                fig.text(0.5, 0, 'Mean response across trials', size=11, ha='center')
                axs[0].set_ylabel('Number of neurons', size=11)
                axs[0].set_title('Before z-score')
                axs[1].set_title('After z-score')

    if plot_checks:
        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            axs[0].imshow(original_resp.T)
            axs[1].imshow(X)
            axs[0].set_xlabel('Neurons')
            axs[0].set_ylabel('Trials')
            axs[0].set_title('Original resp')
            axs[1].set_title('Pre-processed resp')

    if np.sum(np.isnan(X)) > 0:
        print('NaNs found in X, entering debug mode')
        pdb.set_trace()
    if np.shape(X)[1] == 0:
        print('No neurons after preprocessing, entering debug mode')
        pdb.set_trace()

    if estimator is None:
        estimator = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, penalty='l2', dual=False))

    if cv_splitter is None:
        cv_splitter = StratifiedKFold(n_splits=2)

    if fit_estimator:
        cv_results = cross_validate(estimator, X, y, cv=cv_splitter, return_estimator=True)
    else:
        cv_results = dict()
        y_hat = estimator.predict(X)
        accuracy_a = np.sum(y == y_hat) / len(y)
        accuracy_b = np.sum(y == 1 - y_hat) / len(y)
        cv_results['accuracy'] = np.max([accuracy_a, accuracy_b])
        cv_results['estimator'] = [estimator]

    # Calculate baseline accuracy based on the proportion of most represented class
    baseline_accuracy_a = np.sum(y == np.unique(y)[0]) / len(y)
    baseline_accuracy_b = np.sum(y == np.unique(y)[1]) / len(y)
    cv_results['baseline_accuracy'] = np.max([baseline_accuracy_a, baseline_accuracy_b])


    cv_results['X'] = X
    cv_results['y'] = y

    if include_shuffles:
        if verbose:
            print('Running cross validation on shuffled labels')
        accuracy_per_shuffle = np.zeros(num_shuffles) + np.nan
        for n_shuffle in tqdm(np.arange(num_shuffles), disable=(not verbose)):
            y_shuffled = np.random.permutation(y)
            cv_results_shuffled = cross_validate(estimator, X, y_shuffled, cv=cv_splitter, return_estimator=False)
            accuracy_per_shuffle[n_shuffle] = np.mean(cv_results_shuffled['test_score'])
    else:
        accuracy_per_shuffle = None

    weights_per_cv = []

    for estimator_obj in cv_results['estimator']:
        try:
            decoder_weights = estimator_obj['linearsvc'].coef_
        except:
            pdb.set_trace()

        weights_per_cv.append(decoder_weights.flatten())

    if plot_checks:
        with plt.style.context(splstyle.get_style('nature-reviews')):
            fig, ax = plt.subplots()
            fig.set_size_inches(5, 5)
            ax.scatter(weights_per_cv[0], weights_per_cv[1], color='black', s=8)
            corr_stat, corr_pval = sstats.pearsonr(weights_per_cv[0], weights_per_cv[1])
            ax.text(0.8, 0.8, 'r = %.2f, p = %.3f' % (corr_stat, corr_pval), size=11, transform=ax.transAxes)
            ax.set_xlabel('Weights on CV fold 1', size=11)
            ax.set_ylabel('Weights on CV fold 2', size=11)

    weights_per_cv = np.array(weights_per_cv)

    return cv_results, weights_per_cv, accuracy_per_shuffle


def window_decoding(resp_w_t, neun, tr, aligned_time, decoding_window_width=0.1, fixed_decoding_bins=None,
                    estimator=None, cv_splitter=None,
                    pre_processing_steps=['exclude_nan_response', 'impute_nan_response', 'zscore_activity',
                                            'remove_zero_variance_neurons', 'only_include_sig_neurons'],
                    include_shuffles=True,
                    num_shuffles=100, decode_onset=False, onset_baseline_time_window=[-0.15, -0.05],
                    resp_w_t_2=None, tr_2=None,
                    verbose=False):
    """
    Do decoding for each time window (by taking the mean activity over the time window)

    Parameters
    -----------
    resp_w_t : numpy ndarray
        array with shape (time, trial, num_neuron)
    neun : numpy ndarray
        1D array with shape (num_neuron) indicating whether the neuron has significant selectivity
    tr : numpy ndarray
    aligned_time : numpy ndarray
        1D array with shape (time) indicating the time relative to stimulus or movement onset in seconds
    decoding_window_width : float
        width of decoding window in seconds
    estimator : sklearn estimator object
    cv_splitter : sklearn cross validation object
    pre_processing_steps : list
        list of pre-processing steps to perform before doing classification
    verbose : bool
        whether to print out progress of the programme
    decode_onset : bool
        whether to decode the onset of movement / stimulus rather than decoding between trial types
    Returns
    -------
    windowed_decoding_results : dict

    """

    if (decoding_window_width is not None) and (fixed_decoding_bins is None):
        decoding_window_bins = np.arange(aligned_time[0], aligned_time[-1], decoding_window_width)
    else:
        decoding_window_bins = fixed_decoding_bins


    if cv_splitter is None:
        n_cv_folds = 2

    windowed_decoding_results = dict()
    tot_n_window = len(decoding_window_bins) - 1
    accuracy_per_window = np.zeros((tot_n_window, n_cv_folds)) + np.nan
    shuffled_accuracy_per_window = np.zeros((tot_n_window, num_shuffles))

    if decode_onset:
        onset_baseline_index = np.where(
            (aligned_time >= onset_baseline_time_window[0]) &
            (aligned_time <= onset_baseline_time_window[1])
        )[0]
        onset_baseline_activity = np.mean(resp_w_t[onset_baseline_index, :, :], axis=0).T # neuron x trial

    for n_window in np.arange(len(decoding_window_bins) - 1):

        window_start = decoding_window_bins[n_window]
        window_end = decoding_window_bins[n_window+1]

        if verbose:
            print('Decoding from time %.2f to time %.2f' % (window_start, window_end))

        subset_time_idx = np.where(
            (aligned_time >= window_start) & (aligned_time <= window_end))[0]

        resp_at_time_window = np.mean(resp_w_t[subset_time_idx, :, :], axis=0).T  # neuron x trial

        if decode_onset:
            resp = np.concatenate([resp_at_time_window, onset_baseline_activity], axis=1)  # neuron x (on trial + "off" trials)
            trial_type = np.repeat([1, 0], np.shape(resp_at_time_window)[1])  # 1 : on, 0 : off

        else:
            resp = resp_at_time_window
            trial_type = tr

        cv_results_at_t, weights_at_t, accuracy_per_shuffle_at_t = decode_trial_type(resp=resp, neun=neun, tr=trial_type,
                          estimator=None, cv_splitter=cv_splitter,
                          pre_processing_steps=pre_processing_steps,
                          resp_other_cond=None, smallest_tolerable_var=0.0001,
                          include_shuffles=include_shuffles, num_shuffles=num_shuffles,
                          plot_checks=False, verbose=verbose, fit_estimator=True, fig_folder=None)

        accuracy_per_window[n_window, :] = cv_results_at_t['test_score']
        shuffled_accuracy_per_window[n_window, :] = accuracy_per_shuffle_at_t

    windowed_decoding_results['accuracy_per_window'] = accuracy_per_window
    windowed_decoding_results['shuffled_accuracy_per_window'] = shuffled_accuracy_per_window
    windowed_decoding_results['window_starts'] = decoding_window_bins[0:-1]
    windowed_decoding_results['window_ends'] = decoding_window_bins[1:]

    if decode_onset:
        windowed_decoding_results['decoding_type'] = 'onset'
    else:
        windowed_decoding_results['decoding_type'] = 'direction'

    return windowed_decoding_results


def subset_resp_and_other_resp(resp, resp_other_cond, pre_processing_steps=['exclude_nan_response'],
                               smallest_tolerable_var=0.0001, verbose=True):
    """
    Subset response of neurons based on their response to two conditions,
    one given by resp (eg. visual response) and one given by (resp_other_cond), mainly because
    a neuron may have responses in one condition but (for some reason) NaNs in the other condition, which
    excludes the neuron from being analyzed as we want a neuron's response in both conditions.

    Parameters
    ----------
    resp : numpy ndarray
    resp_other_cond : numpy ndarray
    pre_processing_steps : list of str
    versbose : bool

    Returns
    -------
    resp : numpy ndarray
    resp_other_cond : numpy ndarray

    """

    # Create a copy to compare with the pre-processed data
    original_resp = resp.copy()
    original_resp_other_cond = resp_other_cond.copy()


    if 'exclude_nan_response' in pre_processing_steps:
        resp_mean = np.nanmean(resp, axis=1)  # take the mean response of each neuron across trials (excluding nan)
        subset_neuron_bool = ~np.isnan(resp_mean)

        if resp_other_cond is not None:
            resp_mean_other_cond = np.nanmean(resp_other_cond, axis=1)
            subset_neuron_bool_other_cond = ~np.isnan(resp_mean_other_cond)
            subset_neuron_bool = subset_neuron_bool & subset_neuron_bool_other_cond
            resp_other_cond = resp_other_cond[subset_neuron_bool, :]
            resp_mean_other_cond = resp_mean_other_cond[subset_neuron_bool]

        resp = resp[subset_neuron_bool, :]  # throw out neurons with NaNs in all trials
        num_all_nan_neurons = np.sum(np.isnan(resp_mean))
        resp_mean = resp_mean[subset_neuron_bool]
        if verbose:
            print('Original number of neurons: %.f' % np.shape(original_resp)[0])
            print('Found %.f neurons with NaNs in all trials' % num_all_nan_neurons)
            print('Number of neurons after subsetting: %.f' % len(resp_mean))

    if 'impute_nan_response' in pre_processing_steps:
        num_neurons = len(resp_mean)
        for iN in np.arange(num_neurons):
            resp[iN, np.isnan(resp[iN, :])] = resp_mean[iN]

        if resp_other_cond is not None:
            num_neurons_rep_other_cond = len(resp_mean_other_cond)
            for iN in np.arange(num_neurons_rep_other_cond):
                resp_other_cond[iN, np.isnan(resp_other_cond[iN, :])] = resp_mean_other_cond[iN]

    if 'remove_zero_variance_neurons' in pre_processing_steps:
        print('Removing neurons with zero variance')
        var_per_neuron = np.var(resp, axis=1)
        subset_neuron_bool = np.abs(var_per_neuron) >= smallest_tolerable_var

        if resp_other_cond is not None:
            var_per_neuron_other_cond = np.var(resp_other_cond, axis=1)
            subset_neuron_other_cond_bool = np.abs(var_per_neuron_other_cond) >= smallest_tolerable_var
            subset_neuron_bool = subset_neuron_bool & subset_neuron_other_cond_bool

        resp = resp[subset_neuron_bool, :]
        resp_other_cond = resp_other_cond[subset_neuron_bool, :]

    return resp, resp_other_cond


def plot_data(m_resp, v_resp, neun, m_tr, v_tr):
    """

    :param m_resp:
    :param v_resp:
    :param neun:
    :param m_tr:
    :param v_tr:
    :return:
    """
    m_resp = m_resp[neun, :]
    sorted_m_resp = m_resp[:, np.argsort(m_tr)]

    v_resp = v_resp[neun, :]
    sorted_v_resp = v_resp[:, np.argsort(v_tr)]
    with plt.style.context(splstyle.get_style('nature-reviews')):
        fig, ax = plt.subplots(1, 2)
        im = ax[0].imshow(sorted_m_resp, cmap='PiYG', vmin=-6, vmax=6, aspect='auto')
        im = ax[1].imshow(sorted_v_resp, cmap='PiYG', vmin=-6, vmax=6, aspect='auto')
        ax[0].set_ylabel('Neurons')
        ax[0].set_xlabel('Trials')
        ax[1].set_xlabel('Trials')
        fig.colorbar(im)
        #plt.show()
        plt.close(fig)

    return fig


def plot_raster(m_resp_w_time, v_resp_w_time, m_aligned_time, v_aligned_time,
                m_trial_types=None, v_trial_types=None,
                fig=None, axs=None):
    """
    Plot raster of movement and visual related activity across neurons (averaged across trials)
    # TODO: separate the two trial types!!!

    Parameters
    ----------
    m_resp_w_time : numpy ndarray
        array with shape (time, trial, neuron)
    v_resp_w_time : numpy array
        array with shape (time, trial, neuron)
    m_aligned_time : 1D numpy ndarray

    :param v_aligned_time:
    :param fig:
    :param ax:

    Returns
    -------

    """


    with plt.style.context(splstyle.get_style('nature-reviews')):

        if (fig is None) and (axs is None):
            if m_trial_types is None:
                fig, axs = plt.subplots(1, 2)
            else:
                fig, axs = plt.subplots(2, 2)

        if m_trial_types is None:
            m_resp_w_time_trial_mean = np.nanmean(m_resp_w_time, axis=1)
            v_resp_w_time_trial_mean = np.nanmean(v_resp_w_time, axis=1)
            num_neuron = np.shape(m_resp_w_time)[2]

            axs[0].imshow(m_resp_w_time_trial_mean, aspect='auto', extent=[
                m_aligned_time[0], m_aligned_time[-1], 1, num_neuron
            ])

            axs[1].imshow(v_resp_w_time_trial_mean, aspect='auto', extent=[
                v_aligned_time[0], v_aligned_time[-1], 1, num_neuron
            ])
        else:
            num_neuron = np.shape(m_resp_w_time)[2]

            m_resp_w_time_cond_1 = m_resp_w_time[:, m_trial_types == 0, :]
            m_resp_w_time_cond_2 = m_resp_w_time[:, m_trial_types == 1, :]
            m_resp_w_time_trial_mean_cond_1 = np.nanmean(m_resp_w_time_cond_1, axis=1).T
            m_resp_w_time_trial_mean_cond_2 = np.nanmean(m_resp_w_time_cond_2, axis=1).T

            # check averaged matrix is neron x time
            if not (np.shape(m_resp_w_time_trial_mean_cond_1)[1] == len(m_aligned_time)):
                pdb.set_trace()

            axs[0, 0].imshow(m_resp_w_time_trial_mean_cond_1, aspect='auto', extent=[
                m_aligned_time[0], m_aligned_time[-1], 1, num_neuron
            ])

            axs[0, 1].imshow(m_resp_w_time_trial_mean_cond_2, aspect='auto', extent=[
                m_aligned_time[0], m_aligned_time[-1], 1, num_neuron
            ])


            v_resp_w_time_cond_1 = v_resp_w_time[:, v_trial_types == 0, :]
            v_resp_w_time_cond_2 = v_resp_w_time[:, v_trial_types == 1, :]
            v_resp_w_time_trial_mean_cond_1 = np.nanmean(v_resp_w_time_cond_1, axis=1).T
            v_resp_w_time_trial_mean_cond_2 = np.nanmean(v_resp_w_time_cond_2, axis=1).T

            # check averaged matrix is neuron x time
            if not (np.shape(v_resp_w_time_trial_mean_cond_1)[1] == len(v_aligned_time)):
                pdb.set_trace()
            # assert np.shape(v_resp_w_time_trial_mean_cond_1)[0] == num_neuron

            axs[1, 0].imshow(v_resp_w_time_trial_mean_cond_1, aspect='auto', extent=[
                v_aligned_time[0], v_aligned_time[-1], 1, num_neuron
            ])

            axs[1, 1].imshow(v_resp_w_time_trial_mean_cond_2, aspect='auto', extent=[
                v_aligned_time[0], v_aligned_time[-1], 1, num_neuron
            ])

            text_size = 10
            axs[0, 0].set_title('Motor cond 1', size=text_size)
            axs[0, 1].set_title('Motor cond 2', size=text_size)
            axs[1, 0].set_title('Visual cond 1', size=text_size)
            axs[1, 1].set_title('Visual cond 2', size=text_size)

    return fig, axs


def plot_visual_vs_motor_mean_response(fig=None, ax=None):

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()




    return fig, ax


def plot_motor_and_visual_decoder_weights(visual_weights, motor_weights, dot_size=10, label_size=11,
                                          dot_color='black', annotation_text_size=10, rec_name=None,
                                          p_val_text_threshold=0.001, include_line_of_best_fit=False,
                                          line_of_best_fit_method='linear_regression',
                                          num_plots=4, center_zero=False, plot_motor_first=False,
                                          sharex=False, sharey=False, line_of_best_fit_width=2,
                                          line_color='red', tick_length=4, custom_n_y_ticks=None,
                                          fig=None, axs=None):
    """
    Makes scatter plots of the weights for each neuron used in motor decoding and visual decoding (decoder trained
    separately)

    Parameters
    ----------
    visual_weights : 1D numpy ndarray
        weights obtained using the decoder to predict visual location
    motor_weights: 1D numpy ndarray
        weighted obtained using the decoder to predict motor location
    dot_size : int or float
        size of the scatter points
    label_size : int or float
        size of the text labels
    dot_color : str
        color of the scatter dots
    annotation_text_size : int or float
        size of the annotation text to show the p-values and r-values from pearson correlation
    rec_name : None or str
    p_val_text_threshold : float
        p value threshold before using power notation to denote p-value range
    line_of_best_fit : str
        method for getting the line of best fit
        'linear_regression' : standard linear regression
        'robust_fit' : robust linear regression
    plot_motor_first : bool
        whether to plot motor first, default is False
        if motor first, then the ordering is mot-mot, vis-vis, and mot-vis
    fig : matplotlib figure object
    axs : matplotlib axes object, with multiple axes
    Returns
    --------

    """


    with plt.style.context(splstyle.get_style('nature-reviews')):

        if num_plots == 4:
            if (fig is None) and (axs is None):
                fig, axs = plt.subplots(2, 2)
                fig.set_size_inches(7, 7)

            coords_to_plot = [(0, 0), (0, 1), (1, 0), (1, 1)]
            weight_pairs_to_plot = [
                                    [visual_weights[0], visual_weights[1]],
                                    [visual_weights[0], motor_weights[1]],
                                    [visual_weights[1], motor_weights[1]],
                                    [motor_weights[0], motor_weights[1]],
                                    ]
            xy_labels = [
                ['Vis CV 1', 'Vis CV 2'],
                ['Vis CV 1', 'Mot CV 1'],
                ['Vis CV 2', 'Mot CV 2'],
                ['Mot CV 1', 'Mot CV 2']
            ]
        elif num_plots == 3:
            if (fig is None) and (axs is None):
                fig, axs = plt.subplots(1, 3, sharex=sharex, sharey=sharey)
                fig.set_size_inches(9, 3)

            coords_to_plot = [0, 1, 2]

            if plot_motor_first:
                weight_pairs_to_plot = [
                    [motor_weights[0], motor_weights[1]],
                    [visual_weights[0], visual_weights[1]],
                    [visual_weights[0], motor_weights[1]],
                ]
                xy_labels = [
                    ['Mot train', 'Mot test'],
                    ['Vis train', 'Vis test'],
                    ['Vis train', 'Mot test'],
                ]
            else:
                weight_pairs_to_plot = [
                    [visual_weights[0], visual_weights[1]],
                    [motor_weights[0], motor_weights[1]],
                    [visual_weights[0], motor_weights[1]],
                ]
                xy_labels = [
                    ['Vis CV 1', 'Vis CV 2'],
                    ['Mot CV 1', 'Mot CV 2'],
                    ['Vis CV 1', 'Mot CV 2'],
                ]


        for n_plots in np.arange(len(coords_to_plot)):

            axs[coords_to_plot[n_plots]].scatter(weight_pairs_to_plot[n_plots][0],
                                                 weight_pairs_to_plot[n_plots][1], s=dot_size, color=dot_color)
            axs[coords_to_plot[n_plots]].set_xlabel(xy_labels[n_plots][0], size=label_size)
            axs[coords_to_plot[n_plots]].set_ylabel(xy_labels[n_plots][1], size=label_size)
            corr_r, corr_pval = sstats.pearsonr(weight_pairs_to_plot[n_plots][0], weight_pairs_to_plot[n_plots][1])
            if corr_pval < p_val_text_threshold:
                p_val_text = r'$p < 10^{%.f}$' % np.round(np.log10(corr_pval))
            else:
                p_val_text = 'p = %.3f' % corr_pval

            if center_zero:
                x_val_max_abs = np.max(np.abs(weight_pairs_to_plot[n_plots][0]))
                y_val_max_abs = np.max(np.abs(weight_pairs_to_plot[n_plots][1]))
                pad = 0.005
                axs[coords_to_plot[n_plots]].set_xlim([-x_val_max_abs - pad, x_val_max_abs + pad])
                axs[coords_to_plot[n_plots]].set_ylim([-y_val_max_abs - pad, y_val_max_abs + pad])

            if include_line_of_best_fit:
                x_vals = np.linspace(np.min(weight_pairs_to_plot[n_plots][0]), np.max(weight_pairs_to_plot[n_plots][1]),
                                     100)
                if line_of_best_fit_method == 'linear_regression':
                    fit_params = np.polyfit(weight_pairs_to_plot[n_plots][0], weight_pairs_to_plot[n_plots][1], deg=1)
                    y_vals = fit_params[0] * x_vals  + fit_params[1]
                elif line_of_best_fit_method == 'robust_fit_bisquare':
                    X = np.stack([np.repeat(1, len(weight_pairs_to_plot[n_plots][0])), weight_pairs_to_plot[n_plots][0]]).T
                    y = weight_pairs_to_plot[n_plots][1].reshape(-1, 1)
                    rlm_model = sm.RLM(exog=X, endog=y, M=sm.robust.norms.TukeyBiweight())
                    rlm_results = rlm_model.fit()
                    y_vals = rlm_results.params[1] * x_vals + rlm_results.params[0]

                axs[coords_to_plot[n_plots]].plot(x_vals, y_vals, color=line_color, alpha=1,
                                                  lw=line_of_best_fit_width)

            if num_plots == 4:
                text_coord = [0.7, 0.7]
            elif num_plots == 3:
                text_coord = [0.7, 1.0]

            axs[coords_to_plot[n_plots]].text(text_coord[0], text_coord[1], 'r = %.3f \n %s' % (corr_r, p_val_text), size=annotation_text_size,
                           transform=axs[coords_to_plot[n_plots]].transAxes)

            if tick_length is not None:
                [x.xaxis.set_tick_params(length=tick_length) for x in axs]
                [x.yaxis.set_tick_params(length=tick_length) for x in axs]

            if custom_n_y_ticks is not None:
                [x.yaxis.set_major_locator(mpl.ticker.MaxNLocator(custom_n_y_ticks)) for x in axs]

        if rec_name is not None:
            fig.suptitle(rec_name, size=11)

        fig.tight_layout()

    return fig, axs


def plot_decoding_summary(visual_accuracy, motor_accuracy, visual_shuffled_accuracy, motor_shuffled_accuracy,
                          visual_X, motor_X, visual_y, motor_y, visual_weights=None, motor_weights=None, fig=None, axs=None):
    """

    Parameters
    ----------
    :param visual_accuracy:
    :param motor_accuracy:
    :param visual_shuffled_accuracy:
    :param motor_shuffled_accuracy:
    visual_X : numpy ndarray
        numTrial x numFeature matrix
    :param motor_X:
    :param visual_y:
    :param motor_y:
    :param visual_weights:
    :param motor_weights:
    :param fig:
    :param axs:

    Returns
    -------
    """

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(2, 3)
        fig.set_size_inches(8, 4)

    visual_trial_sort = np.argsort(visual_y)
    visual_weights_mean = np.mean(visual_weights, axis=0)
    visual_neuron_sort = np.argsort(visual_weights_mean)
    visual_X_sorted = visual_X[visual_trial_sort, :][:, visual_neuron_sort]

    motor_trial_sort = np.argsort(motor_y)
    motor_weights_mean = np.mean(motor_weights, axis=0)
    motor_neuron_sort = np.argsort(motor_weights_mean)
    motor_X_sorted = motor_X[motor_trial_sort, :][:, motor_neuron_sort]

    axs[0, 0].imshow(visual_X_sorted, aspect='auto', vmin=-2, vmax=2, cmap='bwr')
    axs[0, 0].set_xlabel('Neurons sorted by visual weight', size=11)
    axs[1, 0].imshow(motor_X_sorted, aspect='auto', vmin=-2, vmax=2, cmap='bwr')
    axs[1, 0].set_xlabel('Neurons sorted by motor weight', size=11)

    axs[0, 1].hist(visual_y)
    axs[1, 1].hist(motor_y)
    axs[0, 1].set_ylabel('Number of trials', size=11)
    axs[1, 1].set_ylabel('Number of trials', size=11)
    axs[1, 1].set_xlabel('Class labels', size=11)

    axs[0, 2].set_title('Decoding accuracy', size=11)
    axs[0, 2].hist(visual_shuffled_accuracy, bins=50, lw=0, color='gray')
    axs[0, 2].axvline(np.mean(visual_accuracy), color='red', lw=1)
    axs[0, 2].set_xlim([0, 1.05])

    axs[1, 2].hist(motor_shuffled_accuracy, bins=50, lw=0, color='gray', label='Shuffled')
    axs[1, 2].axvline(np.mean(motor_accuracy), color='red', lw=1, label='Observed')
    axs[1, 2].set_xlim([0, 1.05])

    fig.legend(bbox_to_anchor=(1.04, 0.7))

    return fig, axs


def plot_accuracy_of_decoders(accuracy_matrix, legend_labels=None, xlabels=None, include_legend_labels=True,
                              ylabel='Accuracy', ylim=[0, 1.5], yticks=[0, 0.5, 1], ybounds=[0, 1], highlight_index=None,
                              include_stat_text=False, decoder_ordering_idx=np.array([0, 1, 2, 3]),
                              fig=None, ax=None):
    """
    Plot accuracy of decoders in 4 conditions:
        (1) Decoder trained on visual trials, evaluated on visual trials
        (2) Decoder trained on motor trials, evaluated on motor trials
        (3) Decoder trained on visual trials, evaluated on motor trials
        (4) Decoder trained on motor trials, evaluated on visual trials
    Parameters
    ----------
    accuracy_matrix : numpy ndarray
        array with shape (num_experiment, num_decoders)
    :param legend_labels:
    :param xlabels:
    :param fig:
    :param ax:

    Returns
    -------
    fig : matplotlib figure object
    ax : matplotlib figure object
    """


    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(5, 6)

    num_experiment = np.shape(accuracy_matrix)[0]
    num_decoders = np.shape(accuracy_matrix)[1]

    for n_experiment in np.arange(num_experiment):

        if highlight_index is not None:
            if n_experiment == highlight_index:
                color = 'red'
                zorder = 1
            else:
                color = 'gray'
                zorder = 0
        else:
            color = 'gray'
            zorder = 0

        if include_legend_labels:
            ax.plot(accuracy_matrix[n_experiment, decoder_ordering_idx], label=legend_labels[n_experiment][1:])
            ax.scatter(np.arange(num_decoders), accuracy_matrix[n_experiment, decoder_ordering_idx])
        else:
            ax.plot(accuracy_matrix[n_experiment, decoder_ordering_idx], label=legend_labels[n_experiment][1:], color=color, zorder=zorder)
            ax.scatter(np.arange(num_decoders), accuracy_matrix[n_experiment, decoder_ordering_idx], color=color, zorder=zorder)

    _, vis_vis_vs_mot_vis_pval = sstats.wilcoxon(accuracy_matrix[:, 0], accuracy_matrix[:, 2])
    vis_vis_vs_mot_vis_pval_str = '$p = %.4f$' % vis_vis_vs_mot_vis_pval

    _, vis_vis_vs_vis_mot_pval = sstats.wilcoxon(accuracy_matrix[:, 0], accuracy_matrix[:, 3])
    vis_vis_vs_vis_mot_pval_str = '$p = %.4f$' % vis_vis_vs_vis_mot_pval

    _, mot_mot_vs_mot_vis_pval = sstats.wilcoxon(accuracy_matrix[:, 1], accuracy_matrix[:, 2])
    mot_mot_vs_mot_vis_pval_str = '$p = %.4f$' % mot_mot_vs_mot_vis_pval

    _, mot_mot_vs_vis_mot_pval = sstats.wilcoxon(accuracy_matrix[:, 1], accuracy_matrix[:, 3])
    mot_mot_vs_vis_mot_pval_str = '$p = %.4f$' % mot_mot_vs_vis_mot_pval


    x_start_list = [0, 0,
                    1, 1]
    x_end_list = [2, 3,
                  2, 3]
    y_start_list = [1.25, 1.35,
                    1.0, 1.1]
    y_end_list = [1.25, 1.25,
                  0.8, 0.85]
    stat_list = [vis_vis_vs_mot_vis_pval_str, vis_vis_vs_vis_mot_pval_str,
                 mot_mot_vs_mot_vis_pval_str, mot_mot_vs_vis_mot_pval_str]

    stat_list =  [mot_mot_vs_mot_vis_pval_str, mot_mot_vs_vis_mot_pval_str,
                 vis_vis_vs_mot_vis_pval_str, vis_vis_vs_vis_mot_pval_str]
    # stat_list = np.array(stat_list)[decoder_ordering_idx]

    line_height = 0.05
    text_y_offset = 0.025
    fig, ax = spltext.add_stat_annot(fig, ax,  x_start_list=x_start_list, x_end_list=x_end_list,
                                     y_start_list=y_start_list, y_end_list=y_end_list, line_height=line_height,
                                     stat_list=stat_list, text_y_offset=text_y_offset, text_x_offset=-0.01)

    if include_stat_text:
        ax.text(0.55, 0.05, 'Statistic: Wilcoxon signed-rank test', size=9, color='gray', alpha=0.5,
                transform=ax.transAxes)

    ax.set_xticks(np.arange(num_decoders))
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(ylabel, size=11)
    ax.yaxis.set_label_coords(-.1, 0.3)

    ax.set_ylim(ylim)
    ax.spines['left'].set_bounds(ybounds)
    ax.set_yticks(yticks)

    ax.set_xlabel(r'Training set $\rightarrow$ testing set', size=11)

    if include_legend_labels:
        ax.legend()

    return fig, ax


def cal_dprime(response, trial_id):
    """

    Parameters
    ----------
    response : numpy ndarray
        numNeuron x numTrial matrix
    trial_id :
        vector with shape numTrial x 1
        denoting the identity of each trial, which can either be 0 or 1
    Returns
    -------
    dprime :  numpy ndarray
        vector with the dprime of each neuron, with shape numNeuron x 1
    """

    response_trial_cond_a = response[:, trial_id == 0]
    response_trial_cond_b = response[:, trial_id == 1]
    mean_diff = np.nanmean(response_trial_cond_a, axis=1) - np.nanmean(response_trial_cond_b, axis=1)
    ave_std = (np.nanstd(response_trial_cond_a, axis=1) + np.nanstd(response_trial_cond_b, axis=1)) / 2
    dprime = mean_diff / ave_std

    return dprime


def get_dprime(samples_1, samples_2, return_abs=True):

    mean_diff = np.mean(samples_1) - np.mean(samples_2)
    ave_std = (np.std(samples_1) + np.std(samples_2)) / 2
    dprime = mean_diff / ave_std

    if return_abs:
        dprime = np.abs(dprime)

    return dprime


def plot_vis_vs_motor_dprime(visual_dprime, motor_dprime, center_zero=False, same_x_y_range=False,
                             fig=None, axs=None, labelsize=11):
    """

    :param visual_dprime:
    :param motor_dprime:
    :param fig:
    :param ax:
    :return:

    """

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(9, 4)


    axs[0].scatter(visual_dprime, motor_dprime, color='black', s=10, lw=0)

    if center_zero:
        pad = 0.25
        max_abs_visual_dprime = np.max(np.abs(visual_dprime))
        max_abs_motor_dprime = np.max(np.abs(motor_dprime))
        axs[0].set_xlim([-max_abs_visual_dprime - pad, max_abs_visual_dprime + pad])
        axs[0].set_ylim([-max_abs_motor_dprime - pad, max_abs_motor_dprime + pad])
        axs[0].spines.left.set_position('zero')
        axs[0].spines.bottom.set_position('zero')
        axs[0].spines.left.set_alpha(1)
        axs[0].spines.bottom.set_alpha(1)

        axs[0].set_xlabel('Visual dprime', size=labelsize)
        axs[0].set_ylabel('Motor dprime', size=labelsize)
        axs[0].set_xticks([-np.floor(max_abs_visual_dprime), np.floor(max_abs_visual_dprime)])
        axs[0].set_yticks([-np.floor(max_abs_motor_dprime), np.floor(max_abs_motor_dprime)])
        axs[0].xaxis.set_label_coords(0.5, 0)
        axs[0].yaxis.set_label_coords(0, 0.5)

    else:
        axs[0].set_xlabel('Visual dprime', size=labelsize)
        axs[0].set_ylabel('Motor dprime', size=labelsize)

    if same_x_y_range:
        max_abs_visual_dprime = np.max(np.abs(visual_dprime))
        max_abs_motor_dprime = np.max(np.abs(motor_dprime))
        max_abs_all_dprime = np.max([max_abs_visual_dprime, max_abs_motor_dprime])
        axs[0].set_xlim([-max_abs_all_dprime - pad, max_abs_all_dprime + pad])
        axs[0].set_ylim([-max_abs_all_dprime - pad, max_abs_all_dprime + pad])
        axs[0].set_xticks([-np.floor(max_abs_all_dprime), np.floor(max_abs_all_dprime)])
        axs[0].set_yticks([-np.floor(max_abs_all_dprime), np.floor(max_abs_all_dprime)])


    axs[1].scatter(np.abs(visual_dprime), np.abs(motor_dprime), color='black', s=10, lw=0)
    axs[1].set_xlabel('Absolute visual dprime', size=labelsize)
    axs[1].set_ylabel('Absolute motor dprime', size=labelsize)

    nan_locs = np.logical_or(np.isnan(visual_dprime), np.isnan(motor_dprime))

    signed_corr_stat, signed_corr_pval = sstats.pearsonr(visual_dprime[~nan_locs], motor_dprime[~nan_locs])
    axs[0].text(0.8, 0.8, '$r = %.2f$ \n $p = %.3f$' % (signed_corr_stat, signed_corr_pval), size=11, transform=axs[0].transAxes)

    abs_corr_stat, abs_corr_pval = sstats.pearsonr(np.abs(visual_dprime[~nan_locs]), np.abs(motor_dprime[~nan_locs]))
    axs[1].text(0.8, 0.8, '$r = %.2f$ \n $p = %.3f$' % (abs_corr_stat, abs_corr_pval), size=11, transform=axs[1].transAxes)

    return fig, axs


def plot_dprime_vs_weights(visual_dprime, visual_weights, motor_dprime, motor_weights, fig=None, axs=None, labelsize=11):
    """

    :param visual_dprime:
    :param visual_weights:
    :param motor_dprime:
    :param motor_weights:
    :param fig:
    :param axs:
    :param labelsize:
    :return:
    """

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(8, 4)

    # Visual
    axs[0].scatter(visual_dprime, visual_weights, color='black', s=10, lw=0)
    axs[0].set_xlabel('Visual dprime', size=labelsize)
    axs[0].set_ylabel('Visual weight', size=labelsize)

    vis_nan_locs = np.logical_or(np.isnan(visual_dprime), np.isnan(visual_weights))
    vis_corr_stat, vis_corr_pval = sstats.pearsonr(visual_dprime[~vis_nan_locs], visual_weights[~vis_nan_locs])

    if vis_corr_pval > 0.001:
        vis_p_val_str = '$p = %.3f$' % vis_corr_pval
    else:
        vis_p_val_str = '$p < 10^{-%.f}$' % -np.floor(np.log10(vis_corr_pval))

    axs[0].text(0.8, 0.8, '$r = %.2f$ \n %s' % (vis_corr_stat, vis_p_val_str), size=11, transform=axs[0].transAxes)

    # Motor
    axs[1].scatter(motor_dprime, motor_weights, color='black', s=10, lw=0)
    axs[1].set_xlabel('Motor dprime', size=labelsize)
    axs[1].set_ylabel('Motor weight', size=labelsize)

    mot_nan_locs = np.logical_or(np.isnan(motor_dprime), np.isnan(motor_weights))
    mot_corr_stat, mot_corr_pval = sstats.pearsonr(motor_dprime[~mot_nan_locs], motor_weights[~mot_nan_locs])

    if mot_corr_pval > 0.001:
        mot_p_val_str = '$p = %.3f$' % mot_corr_pval
    else:
        mot_p_val_str = '$p < 10^{-%.f}$' % -np.floor(np.log10(mot_corr_pval))

    axs[1].text(0.8, 0.8, '$r = %.2f$ \n %s' % (mot_corr_stat, mot_p_val_str), size=11,
                transform=axs[1].transAxes)

    return fig, axs


def plot_windowed_decoding(v_windowed_decoding_results, m_windowed_decoding_results, custom_xlim=None,
                           plot_motor_first=False, custom_ylim=None,
                           fig=None, axs=None):

    if (fig is None) and (axs is None):
        fig, axs = plt.subplots(1, 2, sharex=False, sharey=True)
        fig.set_size_inches(8, 4)


    v_mean_accuracy_per_window = np.mean(v_windowed_decoding_results['accuracy_per_window'], axis=1)
    m_mean_accuracy_per_window = np.mean(m_windowed_decoding_results['accuracy_per_window'], axis=1)

    # v_mean_shuffled_accuracy_per_window = np.mean(v_windowed_decoding_results['shuffled_accuracy_per_window'], axis=1)
    # m_mean_shuffled_accuracy_per_window = np.mean(m_windowed_decoding_results['shuffled_accuracy_per_window'], axis=1)

    v_shuffled_upper_percentile_per_window = np.percentile(v_windowed_decoding_results['shuffled_accuracy_per_window'], 95, axis=1)
    v_shuffled_lower_percentile_per_window = np.percentile(v_windowed_decoding_results['shuffled_accuracy_per_window'], 5, axis=1)

    m_shuffled_upper_percentile_per_window = np.percentile(m_windowed_decoding_results['shuffled_accuracy_per_window'], 95, axis=1)
    m_shuffled_lower_percentile_per_window = np.percentile(m_windowed_decoding_results['shuffled_accuracy_per_window'], 5, axis=1)


    if plot_motor_first:
        mot_plot_idx = 0
        vis_plot_idx = 1
    else:
        mot_plot_idx = 1
        vis_plot_idx = 0


    axs[vis_plot_idx].plot(v_windowed_decoding_results['window_ends'],
                v_mean_accuracy_per_window, color='black')
    axs[mot_plot_idx].plot(m_windowed_decoding_results['window_ends'],
                m_mean_accuracy_per_window, color='black')

    axs[vis_plot_idx].fill_between(v_windowed_decoding_results['window_ends'],
                        v_shuffled_upper_percentile_per_window, v_shuffled_lower_percentile_per_window, lw=0,
                        color='gray', alpha=0.5)
    axs[mot_plot_idx].fill_between(m_windowed_decoding_results['window_ends'],
                        m_shuffled_upper_percentile_per_window, m_shuffled_lower_percentile_per_window, lw=0,
                        color='gray', alpha=0.5)

    if custom_xlim is not None:
        [x.set_xlim(custom_xlim) for x in axs]
    if custom_ylim is not None:
        [x.set_ylim(custom_ylim) for x in axs]

    axs[0].set_xticks([-2, 0, 2])
    axs[1].set_xticks([-2, 0, 2])
    axs[0].set_yticks([0, 0.5, 1])
    axs[1].set_yticks([0, 0.5, 1])

    axs[0].set_ylabel('Decoding accuracy', size=11)
    axs[vis_plot_idx].set_xlabel('Peri-stimulus time (s)', size=11)
    axs[mot_plot_idx].set_xlabel('Peri-movement time (s)', size=11)
    axs[vis_plot_idx].set_title('Visual', size=11)
    axs[mot_plot_idx].set_title('Motor', size=11)

    return fig, axs


def plot_all_exp_weight_relationship(vis_vis_line_params_per_subject,
                                    mot_mot_line_params_per_subject,
                                    vis_mot_line_params_per_subject,
                                    vis_train_weights_per_subject,
                                    vis_test_weights_per_subject,
                                    mot_train_weights_per_subject,
                                    mot_test_weights_per_subject,
                                    center_zero=False,
                                    range_to_interpolate=[-0.02, 0.02],
                                    tick_length=4, custom_n_y_ticks=None,
                                    custom_y_ticks=[-0.01, 0, 0.01],
                                    ylabelpad=6, include_stats=True,
                                    highlight_index=None, square_axis=False,
                                    fig=None, axs=None):
    """
    Plots the line of best fit between two sets of weights for each recording session (mouse)

    Parameters
    ----------
    vis_vis_line_params_per_subject : numpy ndarray
    :param mot_mot_line_params_per_subject:
    :param vis_mot_line_params_per_subject:
    :param range_to_interpolate:
    :param tick_length:
    :param ylabelpad:
    :param fig:
    :param axs:
    :return:
    """

    if (fig is None) and (axs is None):

        if square_axis:
            sharey = False
            sharex = False
        else:
            sharex = True
            sharey = True

        fig, axs = plt.subplots(1, 3, sharey=sharey, sharex=sharex)
        fig.set_size_inches(9, 3)


    x_vals = np.linspace(range_to_interpolate[0], range_to_interpolate[1], 100)

    vis_vis_plot_idx = 1
    mot_mot_plot_idx = 0
    vis_mot_plot_idx = 2

    for exp_idx, vis_vis_params in enumerate(vis_vis_line_params_per_subject):

        if highlight_index is not None:
            if exp_idx == highlight_index:
                color = 'red'
                zorder = 1
            else:
                color = 'gray'
                zorder = 0
        else:
            color = 'gray'
            zorder = 0

        axs[vis_vis_plot_idx].plot(x_vals, vis_vis_params[1] + x_vals * vis_vis_params[0], color=color, zorder=zorder)
        axs[vis_vis_plot_idx].set_xlabel('Vis weights train', size=11)
        axs[vis_vis_plot_idx].set_ylabel('Vis weights test', size=11, labelpad=ylabelpad)
    for exp_idx, mot_mot_params in enumerate(mot_mot_line_params_per_subject):
        if highlight_index is not None:
            if exp_idx == highlight_index:
                color = 'red'
                zorder = 1
            else:
                color = 'gray'
                zorder = 0
        else:
            color = 'gray'
            zorder = 0

        axs[mot_mot_plot_idx].plot(x_vals, mot_mot_params[1] + x_vals * mot_mot_params[0], color=color, zorder=zorder)
        axs[mot_mot_plot_idx].set_xlabel('Mot weights train', size=11)
        axs[mot_mot_plot_idx].set_ylabel('Mot weights test', size=11, labelpad=ylabelpad)

    for exp_idx, vis_mot_params in enumerate(vis_mot_line_params_per_subject):

        if highlight_index is not None:
            if exp_idx == highlight_index:
                color = 'red'
                zorder = 1
            else:
                color = 'gray'
                zorder = 0
        else:
            color = 'gray'
            zorder = 0

        axs[vis_mot_plot_idx].plot(x_vals, vis_mot_params[1] + x_vals * vis_mot_params[0], color=color, zorder=zorder)
        axs[vis_mot_plot_idx].set_xlabel('Vis weights train', size=11)
        axs[vis_mot_plot_idx].set_ylabel('Mot weights test', size=11, labelpad=ylabelpad)


    # Adjust axis limits so that center is at zero
    if center_zero:

        y_vals = np.concatenate([
            vis_vis_params[1] + x_vals * vis_vis_params[0],
            mot_mot_params[1] + x_vals * mot_mot_params[0],
            vis_mot_params[1] + x_vals * vis_mot_params[0]
        ])

        y_vals_abs_max = np.max(np.abs(y_vals))

        for ax in axs:
            ax.set_xlim(range_to_interpolate)
            ax.set_ylim([-y_vals_abs_max, y_vals_abs_max])

    if include_stats:
        # Fit a linear mixed effects model here
        # see this: https://stats.stackexchange.com/questions/13166/rs-lmer-cheat-sheet

        subject = []

        for n_subject, weights in enumerate(vis_test_weights_per_subject):
            subject.extend(np.repeat(str(n_subject), len(weights)))

        neuron_weights_df = pd.DataFrame.from_dict({
            'vis_train_weights': np.concatenate(vis_train_weights_per_subject),
            'vis_test_weights': np.concatenate(vis_test_weights_per_subject),
            'mot_train_weights': np.concatenate(mot_train_weights_per_subject),
            'mot_test_weights': np.concatenate(mot_test_weights_per_subject),
            'subject': subject
        })

        vis_vis_random_intercept_md = Lmer("vis_test_weights ~ vis_train_weights + (1 | subject)",
                                   data=neuron_weights_df)
        vis_vis_random_intercept_md_fitted = vis_vis_random_intercept_md.fit()
        vis_vis_p_val = sstats.t.sf(abs(vis_vis_random_intercept_md_fitted['T-stat']['vis_train_weights']),
                                    vis_vis_random_intercept_md_fitted['DF']['vis_train_weights'])

        print(vis_vis_random_intercept_md_fitted)

        mot_mot_random_intercept_md = Lmer("mot_test_weights ~ mot_train_weights + (1 | subject)",
                                           data=neuron_weights_df)
        mot_mot_random_intercept_md_fitted = mot_mot_random_intercept_md.fit()
        mot_mot_p_val = sstats.t.sf(abs(mot_mot_random_intercept_md_fitted['T-stat']['mot_train_weights']),
                                    mot_mot_random_intercept_md_fitted['DF']['mot_train_weights'])

        print(mot_mot_random_intercept_md_fitted)

        vis_mot_random_intercept_md = Lmer("mot_test_weights ~ vis_train_weights + (1 | subject)",
                                           data=neuron_weights_df)
        vis_mot_random_intercept_md_fitted = vis_mot_random_intercept_md.fit()

        vis_mot_p_val = sstats.t.sf(abs(vis_mot_random_intercept_md_fitted['T-stat']['vis_train_weights']),
                                    vis_mot_random_intercept_md_fitted['DF']['vis_train_weights'])

        print(vis_mot_random_intercept_md_fitted)

        for n_test, p_val in enumerate([vis_vis_p_val, mot_mot_p_val, vis_mot_p_val]):
            if p_val < 0.01:
                p_val_str = 'LME model: p < 0.01'
            else:
                p_val_str = 'LME model: p = %.2f' % p_val

            axs[n_test].set_title(p_val_str, size=11)

    [x.xaxis.set_tick_params(length=tick_length) for x in axs]
    [x.yaxis.set_tick_params(length=tick_length) for x in axs]

    if custom_n_y_ticks is not None:
        [x.yaxis.set_major_locator(mpl.ticker.MaxNLocator(custom_n_y_ticks)) for x in axs]

    if custom_y_ticks is not None:
        axs[0].set_yticks(custom_y_ticks)

    # Square axis
    # [x.axis('equal') for x in axs]


    return fig, axs


def cal_amp_difference(resp_w_time, aligned_time, baseline_window, window):
    """

    Parameters
    ----------
    resp_w_time : numpy ndarray
        array with shape (time x trial x neuron)
    aligned_time : numpy ndarray
    baseline_window : list
        list with 2 elements, corresponding to the start and end time point (s) of the window for baseline period
    window : list
        list with 2 elements, corresponding to the start and end time point (s) of the window for activity period
        of interest; eg. visual stimulus time, or movement saccade time

    Returns
    -------
    amp_difference : numpy ndarray
        array with shape (neuron, )

    """

    baseline_subset_time_idx = np.where(
        (aligned_time >= baseline_window[0]) &
        (aligned_time <= baseline_window[1])
    )[0]

    activity_subset_time_idx = np.where(
        (aligned_time >= window[0]) &
        (aligned_time <= window[1])
    )[0]

    baseline_activity = np.nanmean(resp_w_time[baseline_subset_time_idx, :, :], axis=0)  # Trial x Neuron
    activity = np.nanmean(resp_w_time[activity_subset_time_idx, :, :], axis=0)

    amp_difference = np.nanmean(
        activity - baseline_activity, axis=0
    )  # Mean across trials

    return amp_difference


def plot_amp_difference(v_on_off_diff, m_on_off_diff, include_stats=True, fig=None, ax=None):
    """
    Parameters
    -----------
    :param v_on_off_diff:
    :param m_on_off_diff:
    :param include_stats:
    :param fig:
    :param ax:
    :return:
    """

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

    subset_bool = np.logical_and(~np.isnan(v_on_off_diff), ~np.isnan(m_on_off_diff))
    v_on_off_diff_subset = v_on_off_diff[subset_bool]
    m_on_off_diff_subset = m_on_off_diff[subset_bool]

    ax.scatter(v_on_off_diff_subset, m_on_off_diff_subset, color='black', s=10)

    if include_stats:
        corr_stat, corr_pval = sstats.pearsonr(v_on_off_diff_subset, m_on_off_diff_subset)
        ax.text(0.8, 0.8, 'r = %.2f, p = %.3f' % (corr_stat, corr_pval), size=11, transform=ax.transAxes)


    ax.set_xlabel('Vis - baseline', size=11)
    ax.set_ylabel('Motor - baseline', size=11)

    return fig, ax

def plot_amp_difference_summary(neuron_on_off_diff_df, fig=None, ax=None, include_stats=True,
                                text_size=11):

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)


    all_x_val_min = np.min(neuron_on_off_diff_df['v_on_off_diff'])
    all_x_val_max = np.max(neuron_on_off_diff_df['v_on_off_diff'])
    x_vals = np.linspace(all_x_val_min, all_x_val_max, 100)

    for rec_name in np.unique(neuron_on_off_diff_df['rec_name']):

        recording_df = neuron_on_off_diff_df.loc[
            neuron_on_off_diff_df['rec_name'] == rec_name
        ]

        recording_df = recording_df.dropna()

        vis_mot_line_params = np.polyfit(recording_df['v_on_off_diff'], recording_df['m_on_off_diff'], deg=1)
        ax.plot(x_vals, vis_mot_line_params[0] * x_vals + vis_mot_line_params[1], color='gray')


    ax.set_xlabel('Post-stimulus - pre-stimulus', size=text_size)
    ax.set_ylabel('Post-saccade - pre-saccade', size=text_size)

    if include_stats:
        vis_mot_random_intercept_md = Lmer("m_on_off_diff ~ v_on_off_diff + (1 | rec_name)",
                                           data=neuron_on_off_diff_df.dropna())
        vis_mot_random_intercept_md_fitted = vis_mot_random_intercept_md.fit()

        vis_mot_p_val = sstats.t.sf(abs(vis_mot_random_intercept_md_fitted['T-stat']['v_on_off_diff']),
                                    vis_mot_random_intercept_md_fitted['DF']['v_on_off_diff'])

        if vis_mot_p_val < 10 ** -4:
            p_val_str = 'LME: $p < 10^{%.f}$' % np.log10(vis_mot_p_val)
        else:
            p_val_str = 'LME: p = %.4f' % vis_mot_p_val

        ax.set_title(p_val_str, size=11)


    return fig, ax


def plot_all_exp_dprime_relationship(dprime_df, metric='signed_dprime', include_stats=True, text_size=11, fig=None, ax=None):
    """

    :param dprime_df:
    :param include_stats:
    :param text_size:
    :param fig:
    :param ax:
    :return:
    """

    if (fig is None) and (ax is None):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 4)

    # all_x_val_min = np.min(dprime_df['vis_dprime'])
    # all_x_val_max = np.max(dprime_df['vis_dprime'])

    x_abs_max = np.max(np.abs(dprime_df['vis_dprime']))

    if metric == 'abs_dprime':
        x_vals = np.linspace(0, x_abs_max, 100)
    else:
        x_vals = np.linspace(-x_abs_max, x_abs_max, 100)

    y_max_abs_list = []
    # x_vals = np.linspace(all_x_val_min, all_x_val_max, 100)

    for rec_name in np.unique(dprime_df['rec_name']):

        recording_df = dprime_df.loc[
            dprime_df['rec_name'] == rec_name
        ]

        recording_df = recording_df.dropna()

        vis_mot_line_params = np.polyfit(recording_df['vis_dprime'], recording_df['mot_dprime'], deg=1)
        ax.plot(x_vals, vis_mot_line_params[0] * x_vals + vis_mot_line_params[1], color='gray')

        y_max_abs_list.append(np.max(np.abs(vis_mot_line_params[0] * x_vals + vis_mot_line_params[1])))

    y_max_abs_all_exp = np.max(y_max_abs_list)

    if metric == 'abs_dprime':
        ax.set_ylim([0, y_max_abs_all_exp])
        ax.set_xlim([0, x_abs_max])
    else:
        ax.set_ylim([-y_max_abs_all_exp, y_max_abs_all_exp])
        ax.set_xlim([-x_abs_max, x_abs_max])

    if include_stats:
        vis_mot_random_intercept_md = Lmer("mot_dprime ~ vis_dprime + (1 | rec_name)",
                                           data=dprime_df.dropna())
        vis_mot_random_intercept_md_fitted = vis_mot_random_intercept_md.fit()

        vis_mot_p_val = sstats.t.sf(abs(vis_mot_random_intercept_md_fitted['T-stat']['vis_dprime']),
                                    vis_mot_random_intercept_md_fitted['DF']['vis_dprime'])

        if vis_mot_p_val < 10 ** -4:
            p_val_str = 'LME: $p < 10^{%.f}$' % np.log10(vis_mot_p_val)
        else:
            p_val_str = 'LME: p = %.4f' % vis_mot_p_val

        ax.set_title(p_val_str, size=11)

    if metric == 'abs_dprime':
        ax.set_xlabel('Abs Visual dprime', size=text_size)
        ax.set_ylabel('Abs Motor dprime', size=text_size)
    else:
        ax.set_xlabel('Visual dprime', size=text_size)
        ax.set_ylabel('Motor dprime', size=text_size)

    return fig, ax


def get_windowed_mean_activity(m_resp_t_ls, v_resp_t_ls, m_resp_ls, v_resp_ls, m_tr_ls, v_tr_ls,
                               m_aligned_time, v_aligned_time_ls,
                               num_experiments, on_time_window, off_time_window):
    """

    :param m_resp_t_ls:
    :param v_resp_t_ls:
    :param v_aligned_time_ls:
    :param num_experiments:
    :param on_time_window:
    :param off_time_window:
    :return:
    """

    # Re-compute windowed response to get onset and offset "trials"
    for n_recording in np.arange(num_experiments):
        m_resp_w_time = m_resp_t_ls[n_recording]  # time x trial x neuron
        v_resp_w_time = v_resp_t_ls[n_recording]  # trial x time x neuron
        v_resp_w_time = np.swapaxes(v_resp_w_time, 0, 1)  # make into time x trial x neuron

        # Update motor response
        m_on_subset_index = np.where(
            (m_aligned_time >= on_time_window[0]) &
            (m_aligned_time <= on_time_window[1])
        )[0]

        m_off_subset_index = np.where(
            (m_aligned_time >= off_time_window[0]) &
            (m_aligned_time <= off_time_window[1])
        )[0]

        m_on_resp = np.nanmean(m_resp_w_time[m_on_subset_index, :, :], axis=0)  # trial x neuron
        m_off_resp = np.nanmean(m_resp_w_time[m_off_subset_index, :, :], axis=0)  # trial x neuron

        m_resp_new = np.concatenate([m_on_resp, m_off_resp], axis=0)
        m_resp_ls[n_recording] = m_resp_new.T  # neuron x trial (now doubled)
        m_tr_ls[n_recording] = np.repeat([0, 1], np.shape(m_on_resp)[0])  # trial (doubled) x 1

        # Update visual response
        v_aligned_t = v_aligned_time_ls[n_recording].flatten()
        v_on_subset_index = np.where(
            (v_aligned_t >= on_time_window[0]) &
            (v_aligned_t <= on_time_window[1])
        )[0]

        v_off_subset_index = np.where(
            (v_aligned_t >= off_time_window[0]) &
            (v_aligned_t <= off_time_window[1])
        )[0]

        v_on_resp = np.nanmean(v_resp_w_time[v_on_subset_index, :, :], axis=0)  # trial x neuron
        v_off_resp = np.nanmean(v_resp_w_time[v_off_subset_index, :, :], axis=0)  # trial x neuron

        v_resp_new = np.concatenate([v_on_resp, v_off_resp], axis=0)
        v_resp_ls[n_recording] = v_resp_new.T  # neuron x trial (now doubled)
        v_tr_ls[n_recording] = np.repeat([0, 1], np.shape(v_on_resp)[0])  # trial (doubled) x 1


    return m_resp_ls, v_resp_ls, m_tr_ls, v_tr_ls


def main():
    available_processes = ['test_load_data', 'plot_data', 'plot_data_over_time',
                           'fit_motor_evaluate_on_motor_and_visual',
                           'fit_separate_decoders', 'plot_motor_and_visual_decoder_weights', 'plot_decoding_summary',
                           'fit_a_evaluate_on_a_and_b', 'plot_a_evaluate_on_a_and_b_results',
                           'cal_d_prime', 'plot_dprime', 'do_windowed_decoding', 'plot_windowed_decoding',
                           'cal_amp_difference', 'plot_amp_difference', 'cal_trial_angles',
                           'cal_trial_angles_train_test']

    processes_to_run = ['do_windowed_decoding']

    process_params = {
        'test_load_data': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/MatsNewNew'
        ),
        'plot_data': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/MatsNewNew',
            fig_ext='.svg'
        ),
        'plot_data_over_time': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            fig_folder='/Users/timothysit/neurCorrEyeMove/figures',
            eyeMovementSubFolder='eyeMovementsLong',
            fig_ext='.svg'
        ),
        'fit_motor_evaluate_on_motor_and_visual': dict(
            data_repo='/Users/lfedros/OneDrive - University College London/Documents - Neural correlates of eye movements/MatsNewNew',
        ),
        'fit_separate_decoders': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            eyeMovementSubFolder='eyeMovementsLong',
            include_shuffles=True,
            num_shuffles=500,
            plot_checks=True,
            decode_onset=True,
            on_time_window=[0.05, 0.15],
            off_time_window = [-0.15, -0.05],
            pre_processing_steps=['exclude_nan_response', 'impute_nan_response', 'zscore_activity',
                                  'remove_zero_variance_neurons'],
        ),
        'plot_motor_and_visual_decoder_weights': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            fig_folder='/Users/timothysit/neurCorrEyeMove/figures',
            decode_onset=False,
            fig_ext=['.png', '.svg'],
            transparent=False,
            sharex=False, sharey=False,
            include_line_of_best_fit=True,
            plot_motor_first=True,
            line_of_best_fit_method='linear_regression',  # options are : 'linear_regression', 'robust_fit_bisquare'
            center_zero=True,
            include_stats=True,
            num_plots=3,
            highlight_rec_name='_SS044_2015-04-28',
            range_to_interpolate=[-0.015, 0.015],
            custom_n_y_ticks=3,
            svg_text_as_path=False,
        ),
        'plot_visual_vs_motor_mean_response': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/MatsNewNew',
        ),
        'plot_decoding_summary': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/MatsNewNew',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            fig_folder='/Users/timothysit/neurCorrEyeMove/figures',
            highlight_example_name=None,
        ),
        'fit_a_evaluate_on_a_and_b': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            eyeMovementSubFolder='eyeMovementsLong',
            modality_a='motor',  # visual or motor
            modality_b='visual',  # visual or motor
            decode_onset=False,
            on_time_window=[0.05, 0.15],
            off_time_window=[-0.15, -0.05],
            pre_processing_steps=['exclude_nan_response', 'impute_nan_response', 'zscore_activity',
                                  'remove_zero_variance_neurons'],
        ),
        'plot_a_evaluate_on_a_and_b_results': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            fig_folder='/Users/timothysit/neurCorrEyeMove/figures',
            eyeMovementSubFolder='eyeMovementsLong',
            decode_onset=False,
            include_stats=True,
            include_legend_labels=False,
            metric='accuracy_rel_baseline',  # accuracy or accuracy_rel_baseline (use accuracy if decode_onset=True)
            highlight_rec_name='_SS044_2015-04-28',
            decoder_ordering_idx=np.array([1, 0, 2, 3]),  # show motor first, then visual
            custom_y_ticks=[-0.5, 0, 1.0],
            fig_ext=['.png', '.svg'],
            svg_text_as_path=False,
        ),
        'cal_amp_difference': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            pre_processing_steps=None,
            eyeMovementSubFolder='eyeMovementsLong',
            baseline_time_window=[-1.5, -0.5],
            window=[0, 1],
        ),
        'plot_amp_difference': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            eyeMovementSubFolder='eyeMovementsLong',
            fig_folder='/Users/timothysit/neurCorrEyeMove/figures',
            fig_ext='.png'
        ),
        'cal_d_prime': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            eyeMovementSubFolder='eyeMovementsLong',
            decode_onset=False,
            on_time_window=[0.05, 0.15],
            off_time_window=[-0.15, -0.05],
            pre_processing_steps=['exclude_nan_response', 'impute_nan_response',
                                  'remove_zero_variance_neurons'],
        ),
        'plot_dprime': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            fig_folder='/Users/timothysit/neurCorrEyeMove/figures',
            decode_onset=False,
            center_zero=True,
            same_x_y_range=True,
            fig_ext=['.png', '.svg']
        ),
        'do_windowed_decoding': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            eyeMovementSubFolder='eyeMovementsLong',  # either eyeMovementsLong or eyeMovements
            fig_folder='/Users/timothysit/neurCorrEyeMove/figures',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            pre_processing_steps=['exclude_nan_response', 'impute_nan_response', 'zscore_activity',
                                  'remove_zero_variance_neurons'],
            decoding_window_width=0.1,
            decode_onset=False,
        ),
        'plot_windowed_decoding': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            fig_folder='/Users/timothysit/neurCorrEyeMove/figures',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            decode_onset=False,
            plot_motor_first=True,
            custom_xlim=[-2, 2],
            custom_ylim=[0, 1.04],
            fig_ext=['.png', '.svg'],
            svg_text_as_path=False,
        ),
        'do_onset_windowed_decoding': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            eyeMovementSubFolder='eyeMovementsLong',  # either eyeMovementsLong or eyeMovements
            fig_folder='/Users/timothysit/neurCorrEyeMove/figures',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
        ),
        'cal_trial_angles': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            fig_folder='/Users/timothysit/neurCorrEyeMove/figures',
            eyeMovementSubFolder='eyeMovementsLong',
            decode_onset=False,
            on_time_window=[0.05, 0.15],
            off_time_window=[-0.15, -0.05],
            pre_processing_steps=['exclude_nan_response', 'impute_nan_response',
                                  'remove_zero_variance_neurons'],
        ),
        'cal_trial_angles_train_test': dict(
            data_repo='/Users/timothysit/neurCorrEyeMove/decodingData/',
            save_folder='/Users/timothysit/neurCorrEyeMove/decodingResults',
            fig_folder='/Users/timothysit/neurCorrEyeMove/figures',
            eyeMovementSubFolder='eyeMovementsLong',
            decode_onset=False,
            on_time_window=[0.05, 0.15],
            off_time_window=[-0.15, -0.05],
            pre_processing_steps=['exclude_nan_response', 'impute_nan_response',
                                  'remove_zero_variance_neurons'],
            fig_exts=['.png', '.svg'],
        )
    }

    for process in processes_to_run:
        assert process in available_processes
        if process == 'test_load_data':
            data_repo = process_params[process]['data_repo']
            m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'Motor'), 'motor')
            v_resp_ls, v_neun_ls, v_tr_ls, _ = load_data(os.path.join(data_repo, 'Visual'), 'visual')
            print('Data loaded without error')

        if process == 'plot_data':

            data_repo = process_params[process]['data_repo']
            m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'Motor'), 'motor')
            v_resp_ls, v_neun_ls, v_tr_ls, _ = load_data(os.path.join(data_repo, 'Visual'), 'visual')

            print('Plotting data')

        if process == 'plot_data_over_time':

            data_repo = process_params[process]['data_repo']
            fig_folder = process_params[process]['fig_folder']
            m_resp_ls, m_neun_ls, m_tr_ls, m_rec_name, m_resp_t_ls, m_aligned_time = load_data(os.path.join(data_repo, process_params[process]['eyeMovementSubFolder']), 'motor_w_time')
            v_resp_ls, v_neun_ls, v_tr_ls, v_rec_name, v_resp_t_ls, v_aligned_time_ls = load_data(os.path.join(data_repo, 'visualStim'), 'visual_w_time')

            # Check time dimensions
            # time_dim_per_m_recording = [np.shape(x)[0] for x in m_resp_t_ls]
            # time_dim_per_v_recording = [np.shape(x)[1] for x in v_resp_t_ls]

            for n_recording in np.arange(len(m_rec_name)):

                rec_name = m_rec_name[n_recording]

                m_resp_w_time = m_resp_t_ls[n_recording]  # time x trial x neuron
                v_resp_w_time = v_resp_t_ls[n_recording]  # trial x time x neuron
                v_resp_w_time = np.swapaxes(v_resp_w_time, 0, 1)
                m_trial_types = m_tr_ls[n_recording].flatten()
                v_trial_types = v_tr_ls[n_recording].flatten()
                v_aligned_time = v_aligned_time_ls[n_recording].flatten()

                fig, axs= plot_raster(m_resp_w_time, v_resp_w_time, m_aligned_time.flatten(), v_aligned_time,
                                      m_trial_types=m_trial_types, v_trial_types=v_trial_types,
                                      fig=None, axs=None)

                fig.suptitle(rec_name)
                fig.tight_layout()
                fig_name = '%s_raster' % rec_name
                fig_path = os.path.join(fig_folder, fig_name)

                print('Saving plot to %s' % fig_path)

                fig.savefig(fig_path, dpi=300, bbox_inches='tight')



        if process == 'fit_separate_decoders':
            data_repo = process_params[process]['data_repo']
            pre_processing_steps = process_params[process]['pre_processing_steps']
            decode_onset = process_params[process]['decode_onset']
            on_time_window = process_params[process]['on_time_window']
            off_time_window = process_params[process]['off_time_window']


            if decode_onset:
                m_resp_ls, m_neun_ls, m_tr_ls, rec_name, m_resp_t_ls, m_aligned_time = load_data(
                    os.path.join(data_repo, process_params[process]['eyeMovementSubFolder']), 'motor_w_time')
                v_resp_ls, v_neun_ls, v_tr_ls, _, v_resp_t_ls, v_aligned_time_ls = load_data(
                    os.path.join(data_repo, 'visualStim'), 'visual_w_time')

                m_aligned_time = m_aligned_time.flatten()

                num_experiments = len(m_resp_ls)
                # Re-compute windowed response to get onset and offset "trials"
                for n_recording in np.arange(num_experiments):

                    m_resp_w_time = m_resp_t_ls[n_recording]  # time x trial x neuron
                    v_resp_w_time = v_resp_t_ls[n_recording]  # trial x time x neuron
                    v_resp_w_time = np.swapaxes(v_resp_w_time, 0, 1)  # make into time x trial x neuron

                    # Update motor response
                    m_on_subset_index = np.where(
                        (m_aligned_time >= on_time_window[0]) &
                        (m_aligned_time <= on_time_window[1])
                    )[0]

                    m_off_subset_index = np.where(
                        (m_aligned_time >= off_time_window[0]) &
                        (m_aligned_time <= off_time_window[1])
                    )[0]

                    m_on_resp = np.nanmean(m_resp_w_time[m_on_subset_index, :, :], axis=0)  # trial x neuron
                    m_off_resp = np.nanmean(m_resp_w_time[m_off_subset_index, :, :], axis=0) # trial x neuron

                    m_resp_new = np.concatenate([m_on_resp, m_off_resp], axis=0)
                    m_resp_ls[n_recording] = m_resp_new.T  # neuron x trial (now doubled)
                    m_tr_ls[n_recording] = np.repeat([0, 1], np.shape(m_on_resp)[0])  # trial (doubled) x 1

                    # Update visual response
                    v_aligned_t = v_aligned_time_ls[n_recording].flatten()
                    v_on_subset_index = np.where(
                        (v_aligned_t >= on_time_window[0]) &
                        (v_aligned_t <= on_time_window[1])
                    )[0]

                    v_off_subset_index = np.where(
                        (v_aligned_t >= off_time_window[0]) &
                        (v_aligned_t <= off_time_window[1])
                    )[0]

                    v_on_resp = np.nanmean(v_resp_w_time[v_on_subset_index, :, :], axis=0)  # trial x neuron
                    v_off_resp = np.nanmean(v_resp_w_time[v_off_subset_index, :, :], axis=0)  # trial x neuron

                    v_resp_new = np.concatenate([v_on_resp, v_off_resp], axis=0)
                    v_resp_ls[n_recording] = v_resp_new.T  # neuron x trial (now doubled)
                    v_tr_ls[n_recording] = np.repeat([0, 1], np.shape(v_on_resp)[0])  # trial (doubled) x 1

            else:
                m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'Motor'), 'motor')
                v_resp_ls, v_neun_ls, v_tr_ls, _ = load_data(os.path.join(data_repo, 'Visual'), 'visual')

            num_experiments = len(m_resp_ls)

            for exp_idx in np.arange(num_experiments):

                motor_cv_results, motor_decoder_weights_per_cv, motor_accuracy_per_shuffle = decode_trial_type(
                                  resp=m_resp_ls[exp_idx], neun=m_neun_ls[exp_idx],
                                  tr=m_tr_ls[exp_idx], resp_other_cond=v_resp_ls[exp_idx], estimator=None,
                                  plot_checks=process_params[process]['plot_checks'],
                                  include_shuffles=process_params[process]['include_shuffles'],
                                  num_shuffles=process_params[process]['num_shuffles'],
                                  pre_processing_steps=pre_processing_steps)

                visual_cv_results, visual_decoder_weights_per_cv, visual_accuracy_per_shuffle = decode_trial_type(
                                  resp=v_resp_ls[exp_idx], neun=v_neun_ls[exp_idx],
                                  tr=v_tr_ls[exp_idx], resp_other_cond=m_resp_ls[exp_idx], estimator=None,
                                  plot_checks=process_params[process]['plot_checks'],
                                  include_shuffles=process_params[process]['include_shuffles'],
                                  num_shuffles=process_params[process]['num_shuffles'],
                                 pre_processing_steps=pre_processing_steps)

                # if np.shape(motor_decoder_weights_per_cv)[1] != np.shape(visual_decoder_weights_per_cv)

                # TODO: re-write all of these to savez

                # Save decoder accuracy
                if decode_onset:
                    motor_decoder_accuracy_save_name = '%s_decode_onset_motor_accuracy.npy' % rec_name[exp_idx]
                    visual_decoder_accuracy_save_name = '%s_decode_onset_visual_accuracy.npy' % rec_name[exp_idx]
                else:
                    motor_decoder_accuracy_save_name = '%s_motor_accuracy.npy' % rec_name[exp_idx]
                    visual_decoder_accuracy_save_name = '%s_visual_accuracy.npy' % rec_name[exp_idx]

                np.save(os.path.join(process_params[process]['save_folder'], motor_decoder_accuracy_save_name),
                        motor_cv_results['test_score'])
                np.save(os.path.join(process_params[process]['save_folder'], visual_decoder_accuracy_save_name),
                        visual_cv_results['test_score'])

                if decode_onset:
                    motor_decoder_baseline_accuracy_save_name = '%s_decode_onset_motor_baseline_accuracy.npy' % rec_name[exp_idx]
                    visual_decoder_baseline_accuracy_save_name = '%s_decode_onset_visual_baseline_accuracy.npy' % rec_name[exp_idx]
                else:
                    motor_decoder_baseline_accuracy_save_name = '%s_motor_baseline_accuracy.npy' % rec_name[exp_idx]
                    visual_decoder_baseline_accuracy_save_name = '%s_visual_baseline_accuracy.npy' % rec_name[exp_idx]

                np.save(os.path.join(process_params[process]['save_folder'], motor_decoder_baseline_accuracy_save_name),
                        motor_cv_results['baseline_accuracy'])
                np.save(os.path.join(process_params[process]['save_folder'], visual_decoder_baseline_accuracy_save_name),
                        visual_cv_results['baseline_accuracy'])

                # Save shuffled accuracy (if available)
                if (motor_accuracy_per_shuffle is not None) and (visual_accuracy_per_shuffle is not None):
                    if decode_onset:
                        motor_decoder_shuffled_accuracy_save_name = '%s_decode_onset_motor_shuffled_accuracy.npy' % rec_name[exp_idx]
                        visual_decoder_shuffled_accuracy_save_name = '%s_decode_onset_visual_shuffled_accuracy.npy' % rec_name[
                            exp_idx]
                    else:
                        motor_decoder_shuffled_accuracy_save_name = '%s_motor_shuffled_accuracy.npy' % rec_name[exp_idx]
                        visual_decoder_shuffled_accuracy_save_name = '%s_visual_shuffled_accuracy.npy' % rec_name[exp_idx]

                    np.save(os.path.join(process_params[process]['save_folder'], motor_decoder_shuffled_accuracy_save_name),
                            motor_accuracy_per_shuffle)
                    np.save(os.path.join(process_params[process]['save_folder'], visual_decoder_shuffled_accuracy_save_name),
                            visual_accuracy_per_shuffle)

                # Save Weights
                if decode_onset:
                    motor_weight_save_name = '%s_decode_onset_motor_weights.npy' % rec_name[exp_idx]
                    visual_weight_save_name = '%s_decode_onset_visual_weights.npy' % rec_name[exp_idx]
                else:
                    motor_weight_save_name = '%s_motor_weights.npy' % rec_name[exp_idx]
                    visual_weight_save_name = '%s_visual_weights.npy' % rec_name[exp_idx]

                np.save(os.path.join(process_params[process]['save_folder'], motor_weight_save_name),
                        motor_decoder_weights_per_cv)
                np.save(os.path.join(process_params[process]['save_folder'], visual_weight_save_name),
                        visual_decoder_weights_per_cv)

                # Save X and y
                if decode_onset:
                    motor_X_save_name = '%s_decode_onset_motor_X.npy' % rec_name[exp_idx]
                    visual_X_save_name = '%s_decode_onset_visual_X.npy' % rec_name[exp_idx]
                else:
                    motor_X_save_name = '%s_motor_X.npy' % rec_name[exp_idx]
                    visual_X_save_name = '%s_visual_X.npy' % rec_name[exp_idx]

                np.save(os.path.join(process_params[process]['save_folder'], motor_X_save_name),
                        motor_cv_results['X'])
                np.save(os.path.join(process_params[process]['save_folder'], visual_X_save_name),
                        visual_cv_results['X'])

                if decode_onset:
                    motor_y_save_name = '%s_decode_onset_motor_y.npy' % rec_name[exp_idx]
                    visual_y_save_name = '%s_decode_onset_visual_y.npy' % rec_name[exp_idx]
                else:
                    motor_y_save_name = '%s_motor_y.npy' % rec_name[exp_idx]
                    visual_y_save_name = '%s_visual_y.npy' % rec_name[exp_idx]

                np.save(os.path.join(process_params[process]['save_folder'], motor_y_save_name),
                        motor_cv_results['y'])
                np.save(os.path.join(process_params[process]['save_folder'], visual_y_save_name),
                        visual_cv_results['y'])

            print('Finished getting decoder weights')

        if process == 'plot_motor_and_visual_decoder_weights':
            data_repo = process_params[process]['data_repo']
            save_folder = process_params[process]['save_folder']
            fig_folder = process_params[process]['fig_folder']
            fig_ext = process_params[process]['fig_ext']
            include_line_of_best_fit = process_params[process]['include_line_of_best_fit']
            line_of_best_fit_method = process_params[process]['line_of_best_fit_method']
            decode_onset = process_params[process]['decode_onset']
            plot_motor_first = process_params[process]['plot_motor_first']
            highlight_rec_name = process_params[process]['highlight_rec_name']
            range_to_interpolate = process_params[process]['range_to_interpolate']
            custom_n_y_ticks = process_params[process]['custom_n_y_ticks']
            svg_text_as_path = process_params[process]['svg_text_as_path']

            m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'eyeMovements'), 'motor')

            num_experiments = len(m_resp_ls)

            vis_vis_line_params_per_subject = []
            mot_mot_line_params_per_subject = []
            vis_mot_line_params_per_subject = []

            vis_train_weights_per_subject = []
            vis_test_weights_per_subject = []
            mot_train_weights_per_subject = []
            mot_test_weights_per_subject = []

            for exp_idx in np.arange(num_experiments):

                if decode_onset:
                    visual_weight_fpath = os.path.join(save_folder, '%s_decode_onset_visual_weights.npy' % rec_name[exp_idx])
                    motor_weight_fpath = os.path.join(save_folder, '%s_decode_onset_motor_weights.npy' % rec_name[exp_idx])
                else:
                    visual_weight_fpath = os.path.join(save_folder, '%s_visual_weights.npy' % rec_name[exp_idx])
                    motor_weight_fpath = os.path.join(save_folder, '%s_motor_weights.npy' % rec_name[exp_idx])

                if highlight_rec_name is not None:
                    if rec_name[exp_idx] == highlight_rec_name:
                        highlight_index = exp_idx
                else:
                    highlight_index = None

                visual_weights = np.load(visual_weight_fpath)
                motor_weights = np.load(motor_weight_fpath)

                if np.shape(visual_weights)[1] != np.shape(motor_weights)[1]:
                    print('%s decoding results has different number of neurons, skipping for now' % rec_name[exp_idx])
                    print('Number of visual weights: %.f' % np.shape(visual_weights)[1])
                    print('Number of motor weights: %.f' % np.shape(motor_weights)[1])
                    continue

                fig, axs = plot_motor_and_visual_decoder_weights(visual_weights, motor_weights, rec_name=rec_name[exp_idx],
                                                                 include_line_of_best_fit=include_line_of_best_fit,
                                                                 num_plots=process_params[process]['num_plots'],
                                                                 center_zero=process_params[process]['center_zero'],
                                                                 line_of_best_fit_method=line_of_best_fit_method,
                                                                 plot_motor_first=plot_motor_first,
                                                                 sharex=process_params[process]['sharex'],
                                                                 sharey=process_params[process]['sharey'],
                                                                 tick_length=4,
                                                                 custom_n_y_ticks=custom_n_y_ticks)
                if decode_onset:
                    fig_name = '%s_decode_onset_vis_motor_weights_%s' % (rec_name[exp_idx], line_of_best_fit_method)
                else:
                    fig_name = '%s_vis_motor_weights_%s' % (rec_name[exp_idx], line_of_best_fit_method)

                for ext in fig_ext:
                    fig.savefig(os.path.join(fig_folder, fig_name + ext),
                            bbox_inches='tight', transparent=process_params[process]['transparent'], dpi=300)


                # Do some line of best fit for summary plot for subject
                vis_vis_line_params = np.polyfit(visual_weights[0], visual_weights[1], deg=1)
                mot_mot_line_params = np.polyfit(motor_weights[0], motor_weights[1], deg=1)
                vis_mot_line_params = np.polyfit(visual_weights[0], motor_weights[1], deg=1)
                vis_vis_line_params_per_subject.append(vis_vis_line_params)
                mot_mot_line_params_per_subject.append(mot_mot_line_params)
                vis_mot_line_params_per_subject.append(vis_mot_line_params)

                vis_train_weights_per_subject.append(visual_weights[0])
                vis_test_weights_per_subject.append(visual_weights[1])
                mot_train_weights_per_subject.append(motor_weights[0])
                mot_test_weights_per_subject.append(motor_weights[1])

            # Make one plot to summarise all of the above
            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, axs = plt.subplots(1, 3, sharey=True, sharex=True)
                fig.set_size_inches(9, 3)


                fig, axs = plot_all_exp_weight_relationship(vis_vis_line_params_per_subject,
                                                            mot_mot_line_params_per_subject,
                                                            vis_mot_line_params_per_subject,
                                                            vis_train_weights_per_subject,
                                                            vis_test_weights_per_subject,
                                                            mot_train_weights_per_subject,
                                                            mot_test_weights_per_subject,
                                                            range_to_interpolate=range_to_interpolate,
                                                            include_stats=process_params[process]['include_stats'],
                                                            highlight_index=highlight_index,
                                                            tick_length=4, custom_n_y_ticks=custom_n_y_ticks,
                                                            fig=fig, axs=axs)
                if decode_onset:
                    fig_name = 'all_exp_decode_onset_weight_relationship'
                else:
                    fig_name = 'all_exp_weight_relationship'

                fig.tight_layout()
                for ext in fig_ext:

                    if not svg_text_as_path:
                        # save font as text rather than paths
                        plt.rcParams['svg.fonttype'] = 'none'

                    fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')


            print('Finished plotting visual and motor weights')

        if process == 'plot_visual_vs_motor_mean_response':

            data_repo = process_params[process]['data_repo']
            m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'Motor'), 'motor')
            v_resp_ls, v_neun_ls, v_tr_ls, _ = load_data(os.path.join(data_repo, 'Visual'), 'visual')


        if process == 'fit_motor_evaluate_on_motor_and_visual':

            data_repo = process_params[process]['data_repo']
            m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'Motor'), 'motor')
            v_resp_ls, v_neun_ls, v_tr_ls, _ = load_data(os.path.join(data_repo, 'Visual'), 'visual')

            decoder_dict = defaultdict(list)
            for m_r, m_n, m_t, v_r, v_n, v_t, r_n in zip(m_resp_ls, m_neun_ls, m_tr_ls, v_resp_ls, v_neun_ls, v_tr_ls,
                                                         rec_name):
                estimator, m_performance, m_score, m_imbalance = motor_decoder(m_r, m_n, m_t)
                v_performance, v_score, v_imbalance = visual_decoder(v_r, m_n, v_t, estimator)
                fig = plot_data(m_r, v_r, m_n, m_t, v_t)
                fig.suptitle(
                    'm_decod = %.2f v_decod = %.2f \n m_score = %.2f v_score = %.2f \n m_imb = %.2f v_imb = %.2f' %
                    (m_performance, v_performance, np.mean(m_score), np.mean(v_score), m_imbalance, v_imbalance),
                    size=8)
                fig.savefig(os.path.join(data_repo, '%s_plot.png' % r_n), bbox_inches='tight', dpi=300)

                decoder_dict['m_performance'].append(m_performance)
                decoder_dict['m_score'].append(m_score)
                decoder_dict['m_imbalance'].append(m_imbalance)
                decoder_dict['v_performance'].append(v_performance)
                decoder_dict['v_score'].append(v_score)
                decoder_dict['v_imbalance'].append(v_imbalance)
            decoder_dataframe = pd.DataFrame.from_dict(decoder_dict)
            decoder_dataframe.to_csv(os.path.join(data_repo, 'decoder_results.csv'))

        if process == 'fit_a_evaluate_on_a_and_b':

            data_repo = process_params[process]['data_repo']
            modality_a = process_params[process]['modality_a']
            modality_b = process_params[process]['modality_b']
            pre_processing_steps = process_params[process]['pre_processing_steps']
            save_folder = process_params[process]['save_folder']
            decode_onset = process_params[process]['decode_onset']
            on_time_window = process_params[process]['on_time_window']
            off_time_window = process_params[process]['off_time_window']


            if decode_onset:

                m_resp_ls, m_neun_ls, m_tr_ls, rec_name, m_resp_t_ls, m_aligned_time = load_data(
                    os.path.join(data_repo, process_params[process]['eyeMovementSubFolder']), 'motor_w_time')
                v_resp_ls, v_neun_ls, v_tr_ls, _, v_resp_t_ls, v_aligned_time_ls = load_data(
                    os.path.join(data_repo, 'visualStim'), 'visual_w_time')

                m_aligned_time = m_aligned_time.flatten()
                # TODO: make this into a function
                num_experiments = len(m_resp_ls)
                # Re-compute windowed response to get onset and offset "trials"
                for n_recording in np.arange(num_experiments):

                    m_resp_w_time = m_resp_t_ls[n_recording]  # time x trial x neuron
                    v_resp_w_time = v_resp_t_ls[n_recording]  # trial x time x neuron
                    v_resp_w_time = np.swapaxes(v_resp_w_time, 0, 1)  # make into time x trial x neuron

                    # Update motor response
                    m_on_subset_index = np.where(
                        (m_aligned_time >= on_time_window[0]) &
                        (m_aligned_time <= on_time_window[1])
                    )[0]

                    m_off_subset_index = np.where(
                        (m_aligned_time >= off_time_window[0]) &
                        (m_aligned_time <= off_time_window[1])
                    )[0]

                    m_on_resp = np.nanmean(m_resp_w_time[m_on_subset_index, :, :], axis=0)  # trial x neuron
                    m_off_resp = np.nanmean(m_resp_w_time[m_off_subset_index, :, :], axis=0) # trial x neuron

                    m_resp_new = np.concatenate([m_on_resp, m_off_resp], axis=0)
                    m_resp_ls[n_recording] = m_resp_new.T  # neuron x trial (now doubled)
                    m_tr_ls[n_recording] = np.repeat([0, 1], np.shape(m_on_resp)[0])  # trial (doubled) x 1

                    # Update visual response
                    v_aligned_t = v_aligned_time_ls[n_recording].flatten()
                    v_on_subset_index = np.where(
                        (v_aligned_t >= on_time_window[0]) &
                        (v_aligned_t <= on_time_window[1])
                    )[0]

                    v_off_subset_index = np.where(
                        (v_aligned_t >= off_time_window[0]) &
                        (v_aligned_t <= off_time_window[1])
                    )[0]

                    v_on_resp = np.nanmean(v_resp_w_time[v_on_subset_index, :, :], axis=0)  # trial x neuron
                    v_off_resp = np.nanmean(v_resp_w_time[v_off_subset_index, :, :], axis=0)  # trial x neuron

                    v_resp_new = np.concatenate([v_on_resp, v_off_resp], axis=0)
                    v_resp_ls[n_recording] = v_resp_new.T  # neuron x trial (now doubled)
                    v_tr_ls[n_recording] = np.repeat([0, 1], np.shape(v_on_resp)[0])  # trial (doubled) x 1

                decode_onset_str = 'decode_onset_'

            else:
                # I think these two lines are outdated??? (2023-01-15)
                # m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'Motor'), 'motor')
                # v_resp_ls, v_neun_ls, v_tr_ls, _ = load_data(os.path.join(data_repo, 'Visual'), 'visual')

                m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'eyeMovements'), 'motor')
                v_resp_ls, v_neun_ls, v_tr_ls, _ = load_data(os.path.join(data_repo, 'visualStim'), 'visual')
                decode_onset_str = ''

            decoder_dict = defaultdict(list)

            for recording_n in np.arange(0, len(rec_name)):
                m_r = m_resp_ls[recording_n]
                m_n = m_neun_ls[recording_n]
                m_t = m_tr_ls[recording_n]
                v_r = v_resp_ls[recording_n]
                v_n = v_neun_ls[recording_n]
                v_t = v_tr_ls[recording_n]
                r_n = rec_name[recording_n]

                if modality_a == 'motor':

                    response_a = m_r
                    neurons_a = m_n
                    trials_a = m_t
                    response_b = v_r
                    neurons_b = v_n
                    trials_b = v_t

                    results_save_name = '%s_%strain_motor_test_visual_accuracy.npy' % (r_n, decode_onset_str)
                    baseline_accuracy_save_name = '%s_%strain_motor_test_visual_baseline_accuracy.npy' % (r_n, decode_onset_str)

                elif modality_a == 'visual':

                    response_a = v_r
                    neurons_a = v_n
                    trials_a = v_t
                    response_b = m_r
                    neurons_b = m_n
                    trials_b = m_t

                    results_save_name = '%s_%strain_visual_test_motor_accuracy.npy' % (r_n, decode_onset_str)
                    baseline_accuracy_save_name = '%s_%strain_visual_test_motor_baseline_accuracy.npy' % (r_n, decode_onset_str)

                a_cv_results, a_weights_per_cv, _ = decode_trial_type(resp=response_a, neun=neurons_a, tr=trials_a,
                                                                      resp_other_cond=response_b,
                                                                      pre_processing_steps=pre_processing_steps)
                train_a_test_b_accuracy = []
                for a_estimator in a_cv_results['estimator']:
                    b_cv_results, b_weights_per_cv, _ = decode_trial_type(resp=response_b, neun=neurons_b, tr=trials_b,
                                                                          estimator=a_estimator, fit_estimator=False,
                                                                          resp_other_cond=response_a,
                                                                          pre_processing_steps=pre_processing_steps)
                    train_a_test_b_accuracy.append(np.mean(b_cv_results['accuracy']))

                train_a_test_b_accuracy = np.array(train_a_test_b_accuracy)

                np.save(os.path.join(save_folder, results_save_name), train_a_test_b_accuracy)

                np.save(os.path.join(save_folder, baseline_accuracy_save_name), b_cv_results['baseline_accuracy'])

        if process == 'plot_decoding_summary':
            data_repo = process_params[process]['data_repo']
            save_folder = process_params[process]['save_folder']
            fig_folder = process_params[process]['fig_folder']
            m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'Motor'), 'motor')

            num_experiments = len(m_resp_ls)
            for exp_idx in np.arange(num_experiments):
                visual_accuracy_fpath = glob.glob(os.path.join(save_folder, '%s*visual_accuracy.npy' % rec_name[exp_idx]))
                motor_accuracy_fpath = glob.glob(os.path.join(save_folder, '%s*motor_accuracy.npy' % rec_name[exp_idx]))
                assert len(visual_accuracy_fpath) == 1
                assert len(motor_accuracy_fpath) == 1
                visual_accuracy = np.load(visual_accuracy_fpath[0])
                motor_accuracy = np.load(motor_accuracy_fpath[0])

                visual_shuffled_accuracy_fpath = glob.glob(os.path.join(save_folder, '%s*visual_shuffled_accuracy.npy' % rec_name[exp_idx]))
                motor_shuffled_accuracy_fpath = glob.glob(os.path.join(save_folder, '%s*motor_shuffled_accuracy.npy' % rec_name[exp_idx]))
                assert len(visual_shuffled_accuracy_fpath) == 1
                assert len(motor_shuffled_accuracy_fpath) == 1

                visual_shuffled_accuracy = np.load(visual_shuffled_accuracy_fpath[0])
                motor_shuffled_accuracy = np.load(motor_shuffled_accuracy_fpath[0])

                visual_X_fpath = glob.glob(os.path.join(save_folder, '%s*visual_X.npy' % rec_name[exp_idx]))
                motor_X_fpath = glob.glob(os.path.join(save_folder, '%s*motor_X.npy' % rec_name[exp_idx]))
                visual_y_fpath = glob.glob(os.path.join(save_folder, '%s*visual_y.npy' % rec_name[exp_idx]))
                motor_y_fpath = glob.glob(os.path.join(save_folder, '%s*motor_y.npy' % rec_name[exp_idx]))

                assert len(visual_X_fpath) == 1
                assert len(motor_X_fpath) == 1
                assert len(visual_y_fpath) == 1
                assert len(motor_y_fpath) == 1

                visual_X = np.load(visual_X_fpath[0])
                motor_X = np.load(motor_X_fpath[0])
                visual_y = np.load(visual_y_fpath[0])
                motor_y = np.load(motor_y_fpath[0])

                # Load weights
                visual_weight_fpath = glob.glob(os.path.join(save_folder, '%s*visual_weights.npy' % rec_name[exp_idx]))
                motor_weight_fpath = glob.glob(os.path.join(save_folder, '%s*motor_weights.npy' % rec_name[exp_idx]))
                assert len(visual_weight_fpath) == 1
                assert len(motor_weight_fpath) == 1
                visual_weights = np.load(visual_weight_fpath[0])
                motor_weights = np.load(motor_weight_fpath[0])

                # Make plot
                pdb.set_trace()

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plot_decoding_summary(visual_accuracy, motor_accuracy, visual_shuffled_accuracy,
                                                     motor_shuffled_accuracy,
                                                      visual_X, motor_X, visual_y, motor_y, visual_weights,
                                                     motor_weights, fig=None, axs=None)

                    fig_name = '%s_decoding_summary.png' % rec_name[exp_idx]
                    fig.suptitle(rec_name[exp_idx], size=11)
                    fig.tight_layout()
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, transparent=False, bbox_inches='tight')
                    plt.close(fig)

        if process == 'plot_a_evaluate_on_a_and_b_results':
            data_repo = process_params[process]['data_repo']
            save_folder = process_params[process]['save_folder']
            fig_folder = process_params[process]['fig_folder']
            fig_ext = process_params[process]['fig_ext']
            metric = process_params[process]['metric']
            decode_onset = process_params[process]['decode_onset']
            highlight_rec_name = process_params[process]['highlight_rec_name']
            decoder_ordering_idx =  process_params[process]['decoder_ordering_idx']
            custom_y_ticks = process_params[process]['custom_y_ticks']

            xlabels = [r'Vis $\rightarrow$ Vis', r'Mot $\rightarrow$ Mot',
                                                            r'Mot $\rightarrow$ Vis', r'Vis $\rightarrow$ Mot']
            xlabels = np.array(xlabels)[decoder_ordering_idx]
            svg_text_as_path = process_params[process]['svg_text_as_path']

            if decode_onset:
                decode_onset_str = '_decode_onset'
            else:
                decode_onset_str = ''

            m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'eyeMovements'), 'motor')

            num_experiments = len(m_resp_ls)
            accuracy_matrix = np.zeros((num_experiments, 4)) + np.nan

            print('Number of experiments found %.f' % num_experiments)

            for exp_idx in np.arange(num_experiments):

                # visual_accuracy_fpath = glob.glob(os.path.join(save_folder, '%s*visual_accuracy.npy' % rec_name[exp_idx]))
                # motor_accuracy_fpath = glob.glob(os.path.join(save_folder, '%s*motor_accuracy.npy' % rec_name[exp_idx]))
                # visual_accuracy_fpath = [x for x in visual_accuracy_fpath if 'motor' not in x]
                # motor_accuracy_fpath = [x for x in motor_accuracy_fpath if 'visual' not in x]
                # assert len(visual_accuracy_fpath) == 1
                # assert len(motor_accuracy_fpath) == 1

                visual_accuracy_fpath = os.path.join(save_folder, '%s%s_visual_accuracy.npy' % (rec_name[exp_idx], decode_onset_str))
                motor_accuracy_fpath = os.path.join(save_folder, '%s%s_motor_accuracy.npy' % (rec_name[exp_idx], decode_onset_str))

                visual_accuracy = np.load(visual_accuracy_fpath)
                motor_accuracy = np.load(motor_accuracy_fpath)


                # train_visual_test_motor_fpath = glob.glob(os.path.join(save_folder, '%s*train_visual_test_motor_accuracy.npy' % rec_name[exp_idx]))
                # train_motor_test_visual_fpath = glob.glob(os.path.join(save_folder, '%s*train_motor_test_visual_accuracy.npy' % rec_name[exp_idx]))

                # assert len(train_visual_test_motor_fpath) == 1
                # assert len(train_motor_test_visual_fpath) == 1

                train_visual_test_motor_fpath = os.path.join(save_folder, '%s%s_train_visual_test_motor_accuracy.npy' % (rec_name[exp_idx], decode_onset_str))
                train_motor_test_visual_fpath = os.path.join(save_folder, '%s%s_train_motor_test_visual_accuracy.npy' % (rec_name[exp_idx], decode_onset_str))

                train_visual_test_motor_accuracy = np.load(train_visual_test_motor_fpath)
                train_motor_test_visual_accuracy = np.load(train_motor_test_visual_fpath)

                if metric == 'accuracy':
                    print('Using accuracy as metric')
                    accuracy_matrix[exp_idx, 0] = np.mean(visual_accuracy)
                    accuracy_matrix[exp_idx, 1] = np.mean(motor_accuracy)
                    accuracy_matrix[exp_idx, 2] = np.mean(train_visual_test_motor_accuracy)
                    accuracy_matrix[exp_idx, 3] = np.mean(train_motor_test_visual_accuracy)
                    ylabel = 'Accuracy'
                    ylim = [0, 1.5]
                    yticks = [0, 0.5, 1]
                    ybounds = [0, 1]

                elif metric == 'accuracy_rel_baseline':
                    motor_decoder_baseline_accuracy_save_name = os.path.join(save_folder, '%s_motor_baseline_accuracy.npy' % rec_name[exp_idx])
                    visual_decoder_baseline_accuracy_save_name = os.path.join(save_folder, '%s_visual_baseline_accuracy.npy' % rec_name[exp_idx])
                    train_visual_test_motor_baseline_accuracy_save_name = os.path.join(save_folder, '%s_train_visual_test_motor_baseline_accuracy.npy' % rec_name[exp_idx])
                    train_motor_test_visual_baseline_accuracy_save_name = os.path.join(save_folder, '%s_train_motor_test_visual_baseline_accuracy.npy' % rec_name[exp_idx])

                    motor_decoder_baseline_accuracy = np.load(motor_decoder_baseline_accuracy_save_name)
                    visual_decoder_baseline_accuracy = np.load(visual_decoder_baseline_accuracy_save_name)
                    train_visual_test_motor_baseline_accuracy = np.load(train_visual_test_motor_baseline_accuracy_save_name)
                    train_motor_test_visual_baseline_accuracy = np.load(train_motor_test_visual_baseline_accuracy_save_name)

                    accuracy_matrix[exp_idx, 0] = (np.mean(visual_accuracy) - visual_decoder_baseline_accuracy) / (1 - visual_decoder_baseline_accuracy)
                    accuracy_matrix[exp_idx, 1] = (np.mean(motor_accuracy) - motor_decoder_baseline_accuracy) / (1 - motor_decoder_baseline_accuracy)
                    accuracy_matrix[exp_idx, 2] = (np.mean(train_visual_test_motor_accuracy) - train_visual_test_motor_baseline_accuracy) / (1 - train_visual_test_motor_baseline_accuracy)
                    accuracy_matrix[exp_idx, 3] = (np.mean(train_motor_test_visual_accuracy) - train_motor_test_visual_baseline_accuracy) / (1 - train_motor_test_visual_baseline_accuracy)

                    ylim = [-0.5, 1.5]
                    ylabel = '(Accuracy - baseline) / (1 - baseline)'
                    yticks = [-0.5, 0, 0.5, 1]
                    ybounds = [-0.5, 1]

            if custom_y_ticks is not None:
                yticks = custom_y_ticks

            if highlight_rec_name is not None:
                highlight_index = np.where(np.array(rec_name) == highlight_rec_name)[0][0]

            pdb.set_trace()
            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, ax = plot_accuracy_of_decoders(accuracy_matrix=accuracy_matrix,
                                                    legend_labels=rec_name,
                                                    xlabels=xlabels,
                                                    include_legend_labels=process_params[process]['include_legend_labels'],
                                                    ylabel=ylabel, ylim=ylim, yticks=yticks, ybounds=ybounds,
                                                    highlight_index=highlight_index,
                                                    decoder_ordering_idx=decoder_ordering_idx,
                                                    )

                fig_name = '%strain_vis_test_mot_and_vv_%s' % (decode_onset_str, metric)

                for ext in fig_ext:

                    if not svg_text_as_path:
                        # save font as text rather than paths
                        plt.rcParams['svg.fonttype'] = 'none'

                    fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')

        if process == 'cal_d_prime':

            data_repo = process_params[process]['data_repo']
            save_folder = process_params[process]['save_folder']
            pre_processing_steps = process_params[process]['pre_processing_steps']
            decode_onset = process_params[process]['decode_onset']
            on_time_window = process_params[process]['on_time_window']
            off_time_window = process_params[process]['off_time_window']

            if decode_onset:

                m_resp_ls, m_neun_ls, m_tr_ls, rec_name, m_resp_t_ls, m_aligned_time = load_data(
                    os.path.join(data_repo, process_params[process]['eyeMovementSubFolder']), 'motor_w_time')
                v_resp_ls, v_neun_ls, v_tr_ls, _, v_resp_t_ls, v_aligned_time_ls = load_data(
                    os.path.join(data_repo, 'visualStim'), 'visual_w_time')

                m_aligned_time = m_aligned_time.flatten()
                # TODO: make this into a function
                num_experiments = len(m_resp_ls)

                m_resp_ls, v_resp_ls, m_tr_ls, v_tr_ls = \
                    get_windowed_mean_activity(m_resp_t_ls, v_resp_t_ls, m_resp_ls, v_resp_ls,  m_tr_ls, v_tr_ls,
                                               m_aligned_time, v_aligned_time_ls, num_experiments, on_time_window,
                                           off_time_window)

                decode_onset_str = 'decode_onset_'

            else:
                m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'Motor'), 'motor')
                v_resp_ls, v_neun_ls, v_tr_ls, _ = load_data(os.path.join(data_repo, 'Visual'), 'visual')

                decode_onset_str = ''

            num_experiments = len(m_resp_ls)
            print('Number of experiments found %.f' % num_experiments)

            for recording_n in np.arange(0, len(rec_name)):
                m_r = m_resp_ls[recording_n]
                m_n = m_neun_ls[recording_n]
                m_t = m_tr_ls[recording_n]
                v_r = v_resp_ls[recording_n]
                v_n = v_neun_ls[recording_n]
                v_t = v_tr_ls[recording_n]
                r_n = rec_name[recording_n]

                motor_response, visual_response = subset_resp_and_other_resp(resp=m_r, resp_other_cond=v_r,
                                                                        pre_processing_steps=pre_processing_steps,
                                                                        verbose=True)
                motor_dprime = cal_dprime(motor_response, trial_id=m_t)
                visual_dprime = cal_dprime(visual_response, trial_id=v_t)

                motor_dprime_save_name = '%s_%smotor_dprime.npy' % (r_n, decode_onset_str)
                visual_dprime_save_name = '%s_%svisual_dprime.npy' % (r_n, decode_onset_str)

                np.save(os.path.join(save_folder, motor_dprime_save_name), motor_dprime)
                np.save(os.path.join(save_folder, visual_dprime_save_name), visual_dprime)

        if process == 'plot_dprime':

            data_repo = process_params[process]['data_repo']
            save_folder = process_params[process]['save_folder']
            fig_folder = process_params[process]['fig_folder']
            center_zero = process_params[process]['center_zero']
            fig_ext = process_params[process]['fig_ext']
            decode_onset = process_params[process]['decode_onset']
            same_x_y_range = process_params[process]['same_x_y_range']

            m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'eyeMovements'), 'motor')
            num_experiments = len(rec_name)

            if decode_onset:
                decode_onset_str = 'decode_onset_'
                file_decode_onset_str = '_decode_onset'
            else:
                decode_onset_str = ''
                file_decode_onset_str = ''

            rec_name_list = []
            visual_dprime_list = []
            motor_dprime_list = []

            for exp_idx in np.arange(num_experiments):

                # visual_dprime_fpath = glob.glob(os.path.join(save_folder, '%s*visual_dprime.npy' % rec_name[exp_idx]))
                # motor_dprime_fpath = glob.glob(os.path.join(save_folder, '%s*motor_dprime.npy' % rec_name[exp_idx]))
                # assert len(visual_dprime_fpath) == 1
                # assert len(motor_dprime_fpath) == 1

                visual_dprime_fpath = os.path.join(save_folder, '%s%s_visual_dprime.npy' % (rec_name[exp_idx], file_decode_onset_str))
                motor_dprime_fpath = os.path.join(save_folder, '%s%s_motor_dprime.npy' % (rec_name[exp_idx], file_decode_onset_str))

                visual_dprime = np.load(visual_dprime_fpath)
                motor_dprime = np.load(motor_dprime_fpath)


                # visual_weight_fpath = glob.glob(os.path.join(save_folder, '%s*visual_weights.npy' % rec_name[exp_idx]))
                # motor_weight_fpath = glob.glob(os.path.join(save_folder, '%s*motor_weights.npy' % rec_name[exp_idx]))
                # assert len(visual_weight_fpath) == 1
                # assert len(motor_weight_fpath) == 1

                visual_weight_fpath = os.path.join(save_folder, '%s%s_visual_weights.npy' % (rec_name[exp_idx], file_decode_onset_str))
                motor_weight_fpath = os.path.join(save_folder, '%s%s_motor_weights.npy' % (rec_name[exp_idx], file_decode_onset_str))

                visual_weights = np.load(visual_weight_fpath)
                motor_weights = np.load(motor_weight_fpath)

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plot_vis_vs_motor_dprime(visual_dprime, motor_dprime, center_zero=center_zero,
                                                        same_x_y_range=same_x_y_range,
                                                        fig=None, axs=None)
                    fig_name = '%s_%svisual_vs_motor_dprime' % (rec_name[exp_idx], decode_onset_str)
                    fig.suptitle('%s' % rec_name[exp_idx], size=11)

                    for ext in fig_ext:
                        fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')

                if (len(visual_dprime) != len(visual_weights[0])) or (len(motor_dprime) != len(motor_weights[0])):
                    pdb.set_trace()

                with plt.style.context(splstyle.get_style('nature-reviews')):

                    fig, axs = plot_dprime_vs_weights(visual_dprime, visual_weights[0], motor_dprime, motor_weights[0],
                                                        fig=None, axs=None)
                    fig.tight_layout()
                    fig_name = '%s_%sdprime_vs_weights' % (rec_name[exp_idx], decode_onset_str)
                    fig.suptitle('%s' % rec_name[exp_idx], size=11)
                    for ext in fig_ext:
                        fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')


                # Add things to dataframe
                rec_name_list.extend(np.repeat(rec_name[exp_idx], len(visual_dprime)))
                visual_dprime_list.extend(visual_dprime)
                motor_dprime_list.extend(motor_dprime)

            # Plot summary : signed dprime
            dprime_df = pd.DataFrame.from_dict({
                'rec_name': rec_name_list,
                'vis_dprime': visual_dprime_list,
                'mot_dprime': motor_dprime_list,
            })
            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, ax = plot_all_exp_dprime_relationship(dprime_df)
                fig_name = '%sall_exp_dprime_relationship' % (decode_onset_str)
                for ext in fig_ext:
                    fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')


            # Plot summary : absolute dprime
            abs_dprime_df = pd.DataFrame.from_dict({
                'rec_name': rec_name_list,
                'vis_dprime': np.abs(visual_dprime_list),
                'mot_dprime': np.abs(motor_dprime_list),
            })
            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, ax = plot_all_exp_dprime_relationship(dprime_df=abs_dprime_df, metric='abs_dprime')
                fig_name = '%sall_exp_abs_dprime_relationship' % (decode_onset_str)
                for ext in fig_ext:
                    fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')

        if process == 'cal_amp_difference':

            print('Running process %s' % process)

            data_repo = process_params[process]['data_repo']
            save_folder = process_params[process]['save_folder']
            pre_processing_steps = process_params[process]['pre_processing_steps']
            baseline_time_window = process_params[process]['baseline_time_window']
            window = process_params[process]['window']

            m_resp_ls, m_neun_ls, m_tr_ls, m_rec_name, m_resp_t_ls, m_aligned_time = load_data(
                os.path.join(data_repo, process_params[process]['eyeMovementSubFolder']), 'motor_w_time')
            v_resp_ls, v_neun_ls, v_tr_ls, v_rec_name, v_resp_t_ls, v_aligned_time_ls = load_data(
                os.path.join(data_repo, 'visualStim'), 'visual_w_time')

            num_experiments = len(m_resp_ls)

            rec_name = m_rec_name
            print('Found %.f experiment files' % num_experiments)

            for n_recording in np.arange(0, len(rec_name)):

                m_resp_w_time = m_resp_t_ls[n_recording]  # time x trial x neuron
                v_resp_w_time = v_resp_t_ls[n_recording]  # trial x time x neuron
                v_resp_w_time = np.swapaxes(v_resp_w_time, 0, 1) # make into time x trial x neuron
                # m_trial_types = m_tr_ls[n_recording].flatten()
                # v_trial_types = v_tr_ls[n_recording].flatten()
                v_aligned_time = v_aligned_time_ls[n_recording].flatten()
                m_aligned_time = m_aligned_time.flatten()

                # v_neun = v_neun_ls[n_recording].flatten()
                # m_neun = m_neun_ls[n_recording].flatten()

                v_on_off_diff = cal_amp_difference(resp_w_time=v_resp_w_time, aligned_time=v_aligned_time, baseline_window=baseline_time_window, window=window)
                m_on_off_diff = cal_amp_difference(resp_w_time=m_resp_w_time, aligned_time=m_aligned_time, baseline_window=baseline_time_window, window=window)

                if len(v_on_off_diff) != len(m_on_off_diff):
                    pdb.set_trace()

                v_save_path = os.path.join(save_folder, '%s_v_vis_onset_baseline_subtracted' % rec_name[n_recording])
                np.save(v_save_path, v_on_off_diff)

                m_save_path = os.path.join(save_folder, '%s_m_saccade_onset_baseline_subtracted' % rec_name[n_recording])
                np.save(m_save_path, m_on_off_diff)

            print('Finished running %s' % process)

        if process == 'plot_amp_difference':

            print('Running process %s' % process)

            data_repo = process_params[process]['data_repo']
            save_folder = process_params[process]['save_folder']
            fig_folder = process_params[process]['fig_folder']
            fig_ext = process_params[process]['fig_ext']

            m_resp_ls, m_neun_ls, m_tr_ls, m_rec_name, m_resp_t_ls, m_aligned_time = load_data(
                os.path.join(data_repo, process_params[process]['eyeMovementSubFolder']), 'motor_w_time')
            v_resp_ls, v_neun_ls, v_tr_ls, v_rec_name, v_resp_t_ls, v_aligned_time_ls = load_data(
                os.path.join(data_repo, 'visualStim'), 'visual_w_time')

            num_experiments = len(m_resp_ls)

            rec_name = m_rec_name
            print('Found %.f experiment files' % num_experiments)

            rec_name_list = []
            v_on_off_diff_all = []
            m_on_off_diff_all = []

            for n_recording in np.arange(0, len(rec_name)):

                v_save_path = os.path.join(save_folder,
                                           '%s_v_vis_onset_baseline_subtracted.npy' % rec_name[n_recording])
                v_on_off_diff = np.load(v_save_path)

                m_save_path = os.path.join(save_folder,
                                           '%s_m_saccade_onset_baseline_subtracted.npy' % rec_name[n_recording])
                m_on_off_diff = np.load(m_save_path)


                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, ax = plot_amp_difference(v_on_off_diff, m_on_off_diff, include_stats=True)
                    fig_name = '%s_vis_vs_saccade_onset_baseline_subtracted%s' % (rec_name[n_recording], fig_ext)
                    fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')

                # Aggregate data to run LME
                rec_name_list.extend(np.repeat(rec_name[n_recording], len(v_on_off_diff)))
                v_on_off_diff_all.extend(v_on_off_diff)
                m_on_off_diff_all.extend(m_on_off_diff)

            neuron_on_off_diff_df = pd.DataFrame.from_dict({
                'v_on_off_diff': v_on_off_diff_all,
                'm_on_off_diff': m_on_off_diff_all,
                'rec_name': rec_name_list
            })

            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, ax = plot_amp_difference_summary(neuron_on_off_diff_df, include_stats=True)
                fig_name = 'vis_vs_saccade_onset_baseline_subtracted_LME%s' % (fig_ext)
                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


            print('Finished running %s' % process)


        if process == 'do_windowed_decoding':
            print('Running %s' % process)
            data_repo = process_params[process]['data_repo']
            save_folder =  process_params[process]['save_folder']
            decode_onset = process_params[process]['decode_onset']
            pre_processing_steps = process_params[process]['pre_processing_steps']

            m_resp_ls, m_neun_ls, m_tr_ls, m_rec_name, m_resp_t_ls, m_aligned_time = load_data(
                os.path.join(data_repo, process_params[process]['eyeMovementSubFolder']), 'motor_w_time')
            v_resp_ls, v_neun_ls, v_tr_ls, v_rec_name, v_resp_t_ls, v_aligned_time_ls = load_data(
                os.path.join(data_repo, 'visualStim'), 'visual_w_time')

            verbose = True

            for n_recording in np.arange(len(m_rec_name)):
                rec_name = m_rec_name[n_recording]

                m_resp_w_time = m_resp_t_ls[n_recording]  # time x trial x neuron
                v_resp_w_time = v_resp_t_ls[n_recording]  # trial x time x neuron
                v_resp_w_time = np.swapaxes(v_resp_w_time, 0, 1) # make into time x trial x neuron
                m_trial_types = m_tr_ls[n_recording].flatten()
                v_trial_types = v_tr_ls[n_recording].flatten()
                v_aligned_time = v_aligned_time_ls[n_recording].flatten()
                m_aligned_time = m_aligned_time.flatten()

                v_neun = v_neun_ls[n_recording].flatten()
                m_neun = m_neun_ls[n_recording].flatten()

                v_windowed_decoding_results = window_decoding(resp_w_t=v_resp_w_time,
                                                   neun=v_neun, tr=v_trial_types, aligned_time=v_aligned_time,
                                                   verbose=verbose, decode_onset=decode_onset,
                                                   pre_processing_steps=pre_processing_steps)
                m_windowed_decoding_results = window_decoding(resp_w_t=m_resp_w_time,
                                                   neun=m_neun, tr=m_trial_types, aligned_time=m_aligned_time,
                                                   verbose=verbose, decode_onset=decode_onset,
                                                   pre_processing_steps=pre_processing_steps)

                if decode_onset:
                    v_save_path = os.path.join(save_folder, '%s_v_windowed_decoding_onset_results.npz' % rec_name)
                    m_save_path = os.path.join(save_folder, '%s_m_windowed_decoding_onset_results.npz' % rec_name)
                else:
                    v_save_path = os.path.join(save_folder, '%s_v_windowed_decoding_results.npz' % rec_name)
                    m_save_path = os.path.join(save_folder, '%s_m_windowed_decoding_results.npz' % rec_name)

                np.savez(v_save_path, **v_windowed_decoding_results)
                np.savez(m_save_path, **m_windowed_decoding_results)
                print('Saved windowed decoding results for %s in %s' % (rec_name, save_folder))


        if process == 'plot_windowed_decoding':

            data_repo = process_params[process]['data_repo']
            save_folder = process_params[process]['save_folder']
            fig_folder = process_params[process]['fig_folder']
            fig_ext = process_params[process]['fig_ext']
            decode_onset = process_params[process]['decode_onset']
            plot_motor_first = process_params[process]['plot_motor_first']

            m_resp_ls, m_neun_ls, m_tr_ls, m_rec_name, m_resp_t_ls, m_aligned_time = load_data(
                os.path.join(data_repo, 'eyeMovements'), 'motor_w_time')
            v_resp_ls, v_neun_ls, v_tr_ls, v_rec_name, v_resp_t_ls, v_aligned_time_ls = load_data(
                os.path.join(data_repo, 'visualStim'), 'visual_w_time')


            for n_recording in np.arange(len(m_rec_name)):
                rec_name = m_rec_name[n_recording]
                print("Plotting windowed decoding performance for %s" % rec_name)

                if decode_onset:
                    v_save_path = os.path.join(save_folder, '%s_v_windowed_decoding_onset_results.npz' % rec_name)
                    m_save_path = os.path.join(save_folder, '%s_m_windowed_decoding_onset_results.npz' % rec_name)
                    fig_name = '%s_onset_windowed_decoding_accuracy' % rec_name
                else:
                    v_save_path = os.path.join(save_folder, '%s_v_windowed_decoding_results.npz' % rec_name)
                    m_save_path = os.path.join(save_folder, '%s_m_windowed_decoding_results.npz' % rec_name)
                    fig_name = '%s_windowed_decoding_accuracy' % rec_name

                v_windowed_decoding_results = np.load(v_save_path)
                m_windowed_decoding_results = np.load(m_save_path)



                for ext in fig_ext:

                    with plt.style.context(splstyle.get_style('nature-reviews')):
                        fig, axs = plot_windowed_decoding(v_windowed_decoding_results, m_windowed_decoding_results,
                                                          custom_xlim=process_params[process]['custom_xlim'],
                                                          custom_ylim=process_params[process]['custom_ylim'],
                                                          plot_motor_first=plot_motor_first,
                                                          fig=None, axs=None)
                        fig.suptitle('%s' % rec_name)
                        fig.tight_layout()

                        # save font as text rather than paths
                        plt.rcParams['svg.fonttype'] = 'none'

                        fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')

        if process == 'cal_trial_angles':

            data_repo = process_params[process]['data_repo']
            save_folder = process_params[process]['save_folder']
            fig_folder = process_params[process]['fig_folder']
            pre_processing_steps = process_params[process]['pre_processing_steps']
            decode_onset = process_params[process]['decode_onset']
            on_time_window = process_params[process]['on_time_window']
            off_time_window = process_params[process]['off_time_window']

            if decode_onset:

                m_resp_ls, m_neun_ls, m_tr_ls, rec_name, m_resp_t_ls, m_aligned_time = load_data(
                    os.path.join(data_repo, process_params[process]['eyeMovementSubFolder']), 'motor_w_time')
                v_resp_ls, v_neun_ls, v_tr_ls, _, v_resp_t_ls, v_aligned_time_ls = load_data(
                    os.path.join(data_repo, 'visualStim'), 'visual_w_time')

                m_aligned_time = m_aligned_time.flatten()
                # TODO: make this into a function
                num_experiments = len(m_resp_ls)

                m_resp_ls, v_resp_ls, m_tr_ls, v_tr_ls = \
                    get_windowed_mean_activity(m_resp_t_ls, v_resp_t_ls, m_resp_ls, v_resp_ls, m_tr_ls, v_tr_ls,
                                               m_aligned_time, v_aligned_time_ls, num_experiments, on_time_window,
                                               off_time_window)

                decode_onset_str = 'decode_onset_'

            else:
                m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'eyeMovements'), 'motor')
                v_resp_ls, v_neun_ls, v_tr_ls, _ = load_data(os.path.join(data_repo, 'visualStim'), 'visual')


                decode_onset_str = ''

            num_experiments = len(m_resp_ls)
            print('Found %.f experiments' % num_experiments)

            for recording_n in np.arange(0, len(rec_name)):
                m_r = m_resp_ls[recording_n]
                m_n = m_neun_ls[recording_n]
                m_t = m_tr_ls[recording_n]
                v_r = v_resp_ls[recording_n]
                v_n = v_neun_ls[recording_n]
                v_t = v_tr_ls[recording_n]
                r_n = rec_name[recording_n]

                motor_response, visual_response = subset_resp_and_other_resp(resp=m_r, resp_other_cond=v_r,
                                                                             pre_processing_steps=pre_processing_steps,
                                                                             verbose=True)

                mot_response_trial_cond_a = motor_response[:, m_t == 0]
                mot_response_trial_cond_b = motor_response[:, m_t == 1]

                vis_response_trial_cond_a = visual_response[:, v_t == 0]
                vis_response_trial_cond_b = visual_response[:, v_t == 1]

                mot_response_trial_cond_a_mean = np.mean(mot_response_trial_cond_a, axis=1)
                mot_response_trial_cond_b_mean = np.mean(mot_response_trial_cond_b, axis=1)

                # Do LDA
                vis_lda = LinearDiscriminantAnalysis(n_components=1)
                mot_lda = LinearDiscriminantAnalysis(n_components=1)

                vis_lda.fit(visual_response.T, v_t)
                mot_lda.fit(motor_response.T, m_t)

                vis_lda_on_vis_a = vis_lda.transform(vis_response_trial_cond_a.T)
                vis_lda_on_vis_b = vis_lda.transform(vis_response_trial_cond_b.T)
                mot_lda_on_vis_a = mot_lda.transform(vis_response_trial_cond_a.T)
                mot_lda_on_vis_b = mot_lda.transform(vis_response_trial_cond_b.T)

                vis_lda_on_mot_a = vis_lda.transform(mot_response_trial_cond_a.T)
                vis_lda_on_mot_b = vis_lda.transform(mot_response_trial_cond_b.T)
                mot_lda_on_mot_a = mot_lda.transform(mot_response_trial_cond_a.T)
                mot_lda_on_mot_b = mot_lda.transform(mot_response_trial_cond_b.T)

                with plt.style.context(splstyle.get_style('nature-reviews')):

                    fig = plt.figure(figsize=(8, 8))
                    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                                          wspace=0.05, hspace=0.05)

                    ax = fig.add_subplot(gs[1, 0])
                    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
                    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

                    nasal_color = np.array([120, 173, 52]) / 255
                    temporal_color = np.array([179, 30, 129]) / 255

                    vis_nasal = ax.scatter(vis_lda_on_vis_a, mot_lda_on_vis_a, color=nasal_color, label='Vis nasal')
                    vis_temporal = ax.scatter(vis_lda_on_vis_b, mot_lda_on_vis_b, color=temporal_color, label='Vis temporal')

                    # ax.scatter(vis_lda_on_mot_a, mot_lda_on_mot_a, color='blue', marker='x')
                    # ax.scatter(vis_lda_on_mot_b, mot_lda_on_mot_b, color='red', marker='x')

                    mot_nasal = ax.scatter(vis_lda_on_mot_a, mot_lda_on_mot_a, facecolor='none', edgecolor=nasal_color, linestyle=':', label='Mot nasal')
                    mot_temporal = ax.scatter(vis_lda_on_mot_b, mot_lda_on_mot_b, facecolor='none', edgecolor=temporal_color, linestyle=':', label='Mot temporal')

                    ax.set_xlabel('Vis LDA axis', size=11)
                    ax.set_ylabel('Mot LDA axis', size=11)

                    ax.set_xticks([-4, 0, 4])
                    ax.set_yticks([-4, 0, 4])

                    # 1 : temporal grating
                    # 0 : nasal grating

                    # Legend
                    # legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color='blue', label='Vis nasal', lw=0,
                    #                           markerfacecolor='blue', markersize=15),
                    #                    mpl.lines.Line2D([0], [0], marker='o', color='red', label='Vis temporal', lw=0,
                    #                           markerfacecolor='red', markersize=15),
                    #                    mpl.lines.Line2D([0], [0], marker=r'$\u25CC$', color='blue', label='Mot nasal', lw=0,
                    #                                     markerfacecolor='none', linestyle=':',
                    #                                     markersize=15)
                   #                    mpl.lines.Line2D([0], [0], marker=r'$\u25CC$',  color='red', label='Mot temporal', lw=0,
                    #                           markerfacecolor='none', markeredgecolor='blue', markersize=15)]

                    ax.legend(bbox_to_anchor=(0.7, 0.2))

                    mot_linestyle = ':'

                    # Kernel Density Estimate of the marginals : visual dimension
                    vis_lda_points_to_sample = np.linspace(-4, 4, 100)

                    vis_lda_on_vis_a_kde = sstats.gaussian_kde(vis_lda_on_vis_a.flatten())
                    vis_lda_on_vis_a_curve = vis_lda_on_vis_a_kde(vis_lda_points_to_sample)
                    ax_histx.plot(vis_lda_points_to_sample, vis_lda_on_vis_a_curve, color=nasal_color)

                    vis_lda_on_vis_b_kde = sstats.gaussian_kde(vis_lda_on_vis_b.flatten())
                    vis_lda_on_vis_b_curve = vis_lda_on_vis_b_kde(vis_lda_points_to_sample)
                    ax_histx.plot(vis_lda_points_to_sample, vis_lda_on_vis_b_curve, color=temporal_color)

                    vis_lda_on_mot_a_kde = sstats.gaussian_kde(vis_lda_on_mot_a.flatten())
                    vis_lda_on_mot_a_curve = vis_lda_on_mot_a_kde(vis_lda_points_to_sample)
                    ax_histx.plot(vis_lda_points_to_sample, vis_lda_on_mot_a_curve, color=nasal_color, linestyle=mot_linestyle)

                    vis_lda_on_mot_b_kde = sstats.gaussian_kde(vis_lda_on_mot_b.flatten())
                    vis_lda_on_mot_b_curve = vis_lda_on_mot_b_kde(vis_lda_points_to_sample)
                    ax_histx.plot(vis_lda_points_to_sample, vis_lda_on_mot_b_curve, color='red', linestyle=mot_linestyle)

                    margin_pos_vis = -0.1
                    margin_pos_mot = -0.2
                    marker_alpha = 1
                    ax_histx.scatter(vis_lda_on_vis_a, np.repeat(margin_pos_vis, len(vis_lda_on_vis_a)), color=nasal_color, alpha=marker_alpha)
                    ax_histx.scatter(vis_lda_on_vis_b, np.repeat(margin_pos_vis, len(vis_lda_on_vis_b)), color=temporal_color, alpha=marker_alpha)
                    # ax_histx.scatter(vis_lda_on_mot_a, np.repeat(margin_pos_mot, len(vis_lda_on_mot_a)), color='blue', marker='x', alpha=marker_alpha)
                    # ax_histx.scatter(vis_lda_on_mot_b, np.repeat(margin_pos_mot, len(vis_lda_on_mot_b)), color='red', marker='x', alpha=marker_alpha)

                    ax_histx.scatter(vis_lda_on_mot_a, np.repeat(margin_pos_mot, len(vis_lda_on_mot_a)), facecolor='none', edgecolor=nasal_color, linestyle=':', alpha=marker_alpha)
                    ax_histx.scatter(vis_lda_on_mot_b, np.repeat(margin_pos_mot, len(vis_lda_on_mot_b)), facecolor='none', edgecolor=temporal_color, linestyle=':',  alpha=marker_alpha)



                    ax_histx.tick_params(labelbottom=False, length=0)
                    ax_histx.spines['bottom'].set_visible(False)

                    ax_histx.set_yticks([])
                    ax_histx.spines['left'].set_visible(False)

                    # Kernel Density Estimate of the marginals : motor dimension
                    mot_lda_points_to_sample = np.linspace(-4, 4, 100)

                    mot_lda_on_vis_a_kde = sstats.gaussian_kde(mot_lda_on_vis_a.flatten())
                    mot_lda_on_vis_a_curve = mot_lda_on_vis_a_kde(mot_lda_points_to_sample)
                    ax_histy.plot(mot_lda_on_vis_a_curve, mot_lda_points_to_sample, color=nasal_color)

                    mot_lda_on_vis_b_kde = sstats.gaussian_kde(mot_lda_on_vis_b.flatten())
                    mot_lda_on_vis_b_curve = mot_lda_on_vis_b_kde(mot_lda_points_to_sample)
                    ax_histy.plot(mot_lda_on_vis_b_curve, mot_lda_points_to_sample, color=temporal_color)

                    mot_lda_on_mot_a_kde = sstats.gaussian_kde(mot_lda_on_mot_a.flatten())
                    mot_lda_on_mot_a_curve = mot_lda_on_mot_a_kde(mot_lda_points_to_sample)
                    ax_histy.plot(mot_lda_on_mot_a_curve, mot_lda_points_to_sample, color=nasal_color, linestyle=mot_linestyle)

                    mot_lda_on_mot_b_kde = sstats.gaussian_kde(mot_lda_on_mot_b.flatten())
                    mot_lda_on_mot_b_curve = mot_lda_on_mot_b_kde(mot_lda_points_to_sample)
                    ax_histy.plot(mot_lda_on_mot_b_curve, mot_lda_points_to_sample, color=temporal_color, linestyle=mot_linestyle)

                    ax_histy.scatter(np.repeat(margin_pos_vis, len(mot_lda_on_vis_a)), mot_lda_on_vis_a, color=nasal_color, alpha=marker_alpha)
                    ax_histy.scatter(np.repeat(margin_pos_vis, len(mot_lda_on_vis_b)), mot_lda_on_vis_b,  color=temporal_color, alpha=marker_alpha)

                    # ax_histy.scatter(np.repeat(margin_pos_mot, len(mot_lda_on_mot_a)), mot_lda_on_mot_a, color='blue', marker='x', alpha=marker_alpha)
                    # ax_histy.scatter(np.repeat(margin_pos_mot, len(mot_lda_on_mot_b)), mot_lda_on_mot_b, color='red', marker='x', alpha=marker_alpha)
                    ax_histy.scatter(np.repeat(margin_pos_mot, len(mot_lda_on_mot_a)), mot_lda_on_mot_a, facecolor='none', edgecolor=nasal_color, linestyle=':', alpha=marker_alpha)
                    ax_histy.scatter(np.repeat(margin_pos_mot, len(mot_lda_on_mot_b)), mot_lda_on_mot_b, facecolor='none', edgecolor=temporal_color, linestyle=':', alpha=marker_alpha)

                    ax_histy.tick_params(labelleft=False, length=0)
                    ax_histy.spines['left'].set_visible(False)

                    ax_histy.set_xticks([])
                    ax_histy.spines['bottom'].set_visible(False)




                fig_name = '%s_LDA' % rec_name[recording_n]
                fig_ext = '.svg'

                # save font as text rather than paths
                plt.rcParams['svg.fonttype'] = 'none'

                fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')

                """
                cosine_angle_to_mot_a = []
                cosine_angle_to_mot_b = []

                trial_cond_matrices = [mot_response_trial_cond_a, mot_response_trial_cond_b,
                                       vis_response_trial_cond_a, vis_response_trial_cond_b]

                for n_trial_cond, trial_cond_matrix in enumerate(trial_cond_matrices):

                    cosine_angle_to_mot_a_trial_cond = []
                    cosine_angle_to_mot_b_trial_cond = []

                    for n_trial, trial_vec in enumerate(trial_cond_matrix.T):

                        cosine_angle_to_mot_a_trial_cond.append(np.dot(mot_response_trial_cond_a_mean, trial_vec) \
                                                                       / (np.linalg.norm(mot_response_trial_cond_a_mean) * np.linalg.norm(trial_vec)))
                        cosine_angle_to_mot_b_trial_cond.append(np.dot(mot_response_trial_cond_b_mean, trial_vec) \
                                                                / (np.linalg.norm(
                            mot_response_trial_cond_b_mean) * np.linalg.norm(trial_vec)))


                    cosine_angle_to_mot_a.append(cosine_angle_to_mot_a_trial_cond)
                    cosine_angle_to_mot_b.append(cosine_angle_to_mot_b_trial_cond)

                fig, ax = plt.subplots()
                ax.scatter(cosine_angle_to_mot_a[0], cosine_angle_to_mot_b[0], color='blue')
                ax.scatter(cosine_angle_to_mot_a[1], cosine_angle_to_mot_b[1], color='red')
                ax.scatter(cosine_angle_to_mot_a[2], cosine_angle_to_mot_b[2], color='blue', marker='x')
                ax.scatter(cosine_angle_to_mot_a[3], cosine_angle_to_mot_b[3], color='red', marker='x')
                """

        if process == 'cal_trial_angles_train_test':

            data_repo = process_params[process]['data_repo']
            save_folder = process_params[process]['save_folder']
            fig_folder = process_params[process]['fig_folder']
            pre_processing_steps = process_params[process]['pre_processing_steps']
            decode_onset = process_params[process]['decode_onset']
            on_time_window = process_params[process]['on_time_window']
            off_time_window = process_params[process]['off_time_window']
            nasal_color = np.array([120, 173, 52]) / 255
            temporal_color = np.array([179, 30, 129]) / 255

            if decode_onset:

                m_resp_ls, m_neun_ls, m_tr_ls, rec_name, m_resp_t_ls, m_aligned_time = load_data(
                    os.path.join(data_repo, process_params[process]['eyeMovementSubFolder']), 'motor_w_time')
                v_resp_ls, v_neun_ls, v_tr_ls, _, v_resp_t_ls, v_aligned_time_ls = load_data(
                    os.path.join(data_repo, 'visualStim'), 'visual_w_time')

                m_aligned_time = m_aligned_time.flatten()
                # TODO: make this into a function
                num_experiments = len(m_resp_ls)

                m_resp_ls, v_resp_ls, m_tr_ls, v_tr_ls = \
                    get_windowed_mean_activity(m_resp_t_ls, v_resp_t_ls, m_resp_ls, v_resp_ls, m_tr_ls, v_tr_ls,
                                               m_aligned_time, v_aligned_time_ls, num_experiments, on_time_window,
                                               off_time_window)

                decode_onset_str = 'decode_onset_'

            else:
                m_resp_ls, m_neun_ls, m_tr_ls, rec_name = load_data(os.path.join(data_repo, 'eyeMovements'), 'motor')
                v_resp_ls, v_neun_ls, v_tr_ls, _ = load_data(os.path.join(data_repo, 'visualStim'), 'visual')

                decode_onset_str = ''

            num_experiments = len(m_resp_ls)
            print('Found %.f experiments' % num_experiments)


            vis_vis_dprime_per_exp = []
            mot_mot_dprime_per_exp = []
            vis_mot_dprime_per_exp = []
            mot_vis_dprime_per_exp = []

            vis_vis_weight_cosine_sim_per_exp = []
            mot_mot_weight_cosine_sim_per_exp = []
            vis_mot_weight_cosine_sim_per_exp = []
            mot_vis_weight_cosine_sim_per_exp = []

            for recording_n in np.arange(0, len(rec_name)):
                m_r = m_resp_ls[recording_n]
                m_n = m_neun_ls[recording_n]
                m_t = m_tr_ls[recording_n]
                v_r = v_resp_ls[recording_n]
                v_n = v_neun_ls[recording_n]
                v_t = v_tr_ls[recording_n]
                r_n = rec_name[recording_n]

                motor_response, visual_response = subset_resp_and_other_resp(resp=m_r, resp_other_cond=v_r,
                                                                             pre_processing_steps=pre_processing_steps,
                                                                             verbose=True)

                mot_response_trial_cond_a = motor_response[:, m_t == 0]

                mot_response_trial_cond_b = motor_response[:, m_t == 1]

                vis_response_trial_cond_a = visual_response[:, v_t == 0]
                vis_response_trial_cond_b = visual_response[:, v_t == 1]

                mot_response_trial_cond_a_mean = np.mean(mot_response_trial_cond_a, axis=1)
                mot_response_trial_cond_b_mean = np.mean(mot_response_trial_cond_b, axis=1)

                # mot_response_trial_cond_a_split_1 = mot_response_trial_cond_a[:, 0::2]
                # mot_response_trial_cond_a_split_2 = mot_response_trial_cond_a[:, 1::2]

                mot_cond_a_split_1_indices = np.where(m_t == 0)[0][0::2]
                mot_cond_a_split_2_indices = np.where(m_t == 0)[0][1::2]
                mot_cond_b_split_1_indices = np.where(m_t == 1)[0][0::2]
                mot_cond_b_split_2_indices = np.where(m_t == 1)[0][1::2]

                vis_cond_a_split_1_indices = np.where(v_t == 0)[0][0::2]
                vis_cond_a_split_2_indices = np.where(v_t == 0)[0][1::2]
                vis_cond_b_split_1_indices = np.where(v_t == 1)[0][0::2]
                vis_cond_b_split_2_indices = np.where(v_t == 1)[0][1::2]

                vis_response_trial_cond_a_split_1 = visual_response[:, vis_cond_a_split_1_indices]
                vis_response_trial_cond_b_split_1 = visual_response[:, vis_cond_b_split_1_indices]
                mot_response_trial_cond_a_split_1 = motor_response[:, mot_cond_a_split_1_indices]
                mot_response_trial_cond_b_split_1 = motor_response[:, mot_cond_b_split_1_indices]

                vis_response_trial_cond_a_split_2 = visual_response[:, vis_cond_a_split_2_indices]
                vis_response_trial_cond_b_split_2 = visual_response[:, vis_cond_b_split_2_indices]
                mot_response_trial_cond_a_split_2 = motor_response[:, mot_cond_a_split_2_indices]
                mot_response_trial_cond_b_split_2 = motor_response[:, mot_cond_b_split_2_indices]


                vis_split_1_indices = np.concatenate([vis_cond_a_split_1_indices, vis_cond_b_split_1_indices])
                visual_response_split_1 = visual_response[:, vis_split_1_indices]
                v_t_split_1 = v_t[vis_split_1_indices]

                mot_split_1_indices = np.concatenate([mot_cond_a_split_1_indices, mot_cond_b_split_1_indices])
                motor_response_split_1 = motor_response[:, mot_split_1_indices]
                m_t_split_1 = m_t[mot_split_1_indices]

                vis_split_2_indices = np.concatenate([vis_cond_a_split_2_indices, vis_cond_b_split_2_indices])
                visual_response_split_2 = visual_response[:, vis_split_2_indices]
                v_t_split_2 = v_t[vis_split_2_indices]

                mot_split_2_indices = np.concatenate([mot_cond_a_split_2_indices, mot_cond_b_split_2_indices])
                motor_response_split_2 = motor_response[:, mot_split_2_indices]
                m_t_split_2 = m_t[mot_split_2_indices]


                # Do LDA on first split
                vis_lda_split_1 = LinearDiscriminantAnalysis(n_components=1)
                mot_lda_split_1 = LinearDiscriminantAnalysis(n_components=1)
                vis_lda_split_1.fit(visual_response_split_1.T, v_t_split_1)
                mot_lda_split_1.fit(motor_response_split_1.T, m_t_split_1)

                # Do LDA on second split
                vis_lda_split_2 = LinearDiscriminantAnalysis(n_components=1)
                mot_lda_split_2 = LinearDiscriminantAnalysis(n_components=1)
                vis_lda_split_2.fit(visual_response_split_2.T, v_t_split_2)
                mot_lda_split_2.fit(motor_response_split_2.T, m_t_split_2)


                vis_lda_on_vis_a = vis_lda_split_1.transform(vis_response_trial_cond_a_split_1.T)
                vis_lda_on_vis_b = vis_lda_split_1.transform(vis_response_trial_cond_b_split_1.T)
                mot_lda_on_vis_a = mot_lda_split_1.transform(vis_response_trial_cond_a_split_1.T)
                mot_lda_on_vis_b = mot_lda_split_1.transform(vis_response_trial_cond_b_split_1.T)

                vis_lda_on_mot_a = vis_lda_split_1.transform(mot_response_trial_cond_a_split_1.T)
                vis_lda_on_mot_b = vis_lda_split_1.transform(mot_response_trial_cond_b_split_1.T)
                mot_lda_on_mot_a = mot_lda_split_1.transform(mot_response_trial_cond_a_split_1.T)
                mot_lda_on_mot_b = mot_lda_split_1.transform(mot_response_trial_cond_b_split_1.T)


                # Get the cross-validated LDA splits and dprime : VIS -> VIS
                vis_lda_on_vis_a_train_1_test_2 = vis_lda_split_1.transform(vis_response_trial_cond_a_split_2.T)
                vis_lda_on_vis_b_train_1_test_2 = vis_lda_split_1.transform(vis_response_trial_cond_b_split_2.T)
                vis_lda_on_vis_a_train_2_test_1 = vis_lda_split_2.transform(vis_response_trial_cond_a_split_1.T)
                vis_lda_on_vis_b_train_2_test_1 = vis_lda_split_2.transform(vis_response_trial_cond_b_split_1.T)

                vis_train_1_test_2_dprime = get_dprime(samples_1=vis_lda_on_vis_a_train_1_test_2, samples_2=vis_lda_on_vis_b_train_1_test_2)
                vis_train_2_test_1_dprime = get_dprime(samples_1=vis_lda_on_vis_a_train_2_test_1, samples_2=vis_lda_on_vis_b_train_2_test_1)

                # Get the cross-validated LDA splits and dprime : MOT -> MOT
                mot_lda_on_mot_a_train_1_test_2 = mot_lda_split_1.transform(mot_response_trial_cond_a_split_2.T)
                mot_lda_on_mot_b_train_1_test_2 = mot_lda_split_1.transform(mot_response_trial_cond_b_split_2.T)
                mot_lda_on_mot_a_train_2_test_1 = mot_lda_split_2.transform(mot_response_trial_cond_a_split_1.T)
                mot_lda_on_mot_b_train_2_test_1 = mot_lda_split_2.transform(mot_response_trial_cond_b_split_1.T)

                mot_train_1_test_2_dprime = get_dprime(samples_1=mot_lda_on_mot_a_train_1_test_2,
                                                       samples_2=mot_lda_on_mot_b_train_1_test_2)
                mot_train_2_test_1_dprime = get_dprime(samples_1=mot_lda_on_mot_a_train_2_test_1,
                                                       samples_2=mot_lda_on_mot_b_train_2_test_1)

                # Get the cross-validated LDA splits and dprime : VIS -> MOT
                vis_lda_on_mot_a_train_1_test_2 = vis_lda_split_1.transform(mot_response_trial_cond_a_split_2.T)
                vis_lda_on_mot_b_train_1_test_2 = vis_lda_split_1.transform(mot_response_trial_cond_b_split_2.T)
                vis_lda_on_mot_a_train_2_test_1 = vis_lda_split_2.transform(mot_response_trial_cond_a_split_1.T)
                vis_lda_on_mot_b_train_2_test_1 = vis_lda_split_2.transform(mot_response_trial_cond_b_split_1.T)

                vis_lda_mot_train_1_test_2_dprime = get_dprime(samples_1=vis_lda_on_mot_a_train_1_test_2,
                                                       samples_2=vis_lda_on_mot_b_train_1_test_2)
                vis_lda_mot_train_2_test_1_dprime = get_dprime(samples_1=vis_lda_on_mot_a_train_2_test_1,
                                                       samples_2=vis_lda_on_mot_b_train_2_test_1)

                # Get the cross-validated LDA splits and dprime : MOT -> VIS
                mot_lda_on_vis_a_train_1_test_2 = mot_lda_split_1.transform(vis_response_trial_cond_a_split_2.T)
                mot_lda_on_vis_b_train_1_test_2 = mot_lda_split_1.transform(vis_response_trial_cond_b_split_2.T)
                mot_lda_on_vis_a_train_2_test_1 = mot_lda_split_2.transform(vis_response_trial_cond_a_split_1.T)
                mot_lda_on_vis_b_train_2_test_1 = mot_lda_split_2.transform(vis_response_trial_cond_b_split_1.T)

                mot_lda_vis_train_1_test_2_dprime = get_dprime(samples_1=mot_lda_on_vis_a_train_1_test_2,
                                                               samples_2=mot_lda_on_vis_b_train_1_test_2)
                mot_lda_vis_train_2_test_1_dprime = get_dprime(samples_1=mot_lda_on_vis_a_train_2_test_1,
                                                               samples_2=mot_lda_on_vis_b_train_2_test_1)

                vis_vis_dprime_per_exp.append(np.mean([vis_train_1_test_2_dprime, vis_train_2_test_1_dprime]))
                mot_mot_dprime_per_exp.append(np.mean([mot_train_1_test_2_dprime, mot_train_2_test_1_dprime]))
                vis_mot_dprime_per_exp.append(np.mean([vis_lda_mot_train_1_test_2_dprime, vis_lda_mot_train_2_test_1_dprime]))
                mot_vis_dprime_per_exp.append(np.mean([mot_lda_vis_train_1_test_2_dprime, mot_lda_vis_train_2_test_1_dprime]))


                # Calculate the cosine angle of the LDA weights
                vis_vis_weight_cosine_sim = np.dot(vis_lda_split_1.coef_.flatten(), vis_lda_split_2.coef_.flatten()) / \
                                            (np.linalg.norm(vis_lda_split_1.coef_) * np.linalg.norm(vis_lda_split_2.coef_))

                mot_mot_weight_cosine_sim = np.dot(mot_lda_split_1.coef_.flatten(), mot_lda_split_2.coef_.flatten()) / \
                                            (np.linalg.norm(mot_lda_split_1.coef_) * np.linalg.norm(
                                                mot_lda_split_2.coef_))

                vis_mot_weight_cosine_sim = np.dot(vis_lda_split_1.coef_.flatten(), mot_lda_split_2.coef_.flatten()) / \
                                            (np.linalg.norm(vis_lda_split_1.coef_) * np.linalg.norm(
                                                mot_lda_split_2.coef_))

                mot_vis_weight_cosine_sim = np.dot(mot_lda_split_1.coef_.flatten(), vis_lda_split_2.coef_.flatten()) / \
                                            (np.linalg.norm(mot_lda_split_1.coef_) * np.linalg.norm(
                                                vis_lda_split_2.coef_))

                vis_vis_weight_cosine_sim_per_exp.append(vis_vis_weight_cosine_sim)
                mot_mot_weight_cosine_sim_per_exp.append(mot_mot_weight_cosine_sim)
                vis_mot_weight_cosine_sim_per_exp.append(vis_mot_weight_cosine_sim)
                mot_vis_weight_cosine_sim_per_exp.append(mot_vis_weight_cosine_sim)

                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig, axs = plt.subplots(2, 4, sharex=True, sharey=True)
                    fig.set_size_inches(8, 4)

                    title_txt_size = 11
                    margin_y_val = 0.1

                    # Vis classifcation
                    axs[0, 0].scatter(vis_lda_on_vis_a_train_1_test_2, np.repeat(margin_y_val, len(vis_lda_on_vis_a_train_1_test_2)),
                                      color=nasal_color)
                    axs[0, 0].scatter(vis_lda_on_vis_b_train_1_test_2, np.repeat(margin_y_val, len(vis_lda_on_vis_b_train_1_test_2)),
                                      color=temporal_color)

                    axs[0, 0].set_title('dprime = %.2f' % vis_train_1_test_2_dprime, size=title_txt_size)

                    axs[1, 0].scatter(vis_lda_on_vis_a_train_2_test_1,
                                      np.repeat(margin_y_val, len(vis_lda_on_vis_a_train_2_test_1)),
                                      color=nasal_color)
                    axs[1, 0].scatter(vis_lda_on_vis_b_train_2_test_1,
                                      np.repeat(margin_y_val, len(vis_lda_on_vis_b_train_2_test_1)),
                                      color=temporal_color)

                    axs[1, 0].set_title('dprime = %.2f' % vis_train_2_test_1_dprime, size=title_txt_size)


                    # Mot classification
                    axs[0, 1].scatter(mot_lda_on_mot_a_train_1_test_2,
                                      np.repeat(margin_y_val, len(mot_lda_on_mot_a_train_1_test_2)),
                                      color=nasal_color)
                    axs[0, 1].scatter(mot_lda_on_mot_b_train_1_test_2,
                                      np.repeat(margin_y_val, len(mot_lda_on_mot_b_train_1_test_2)),
                                      color=temporal_color)

                    axs[0, 1].set_title('dprime = %.2f' % mot_train_1_test_2_dprime, size=title_txt_size)

                    axs[1, 1].scatter(mot_lda_on_mot_a_train_2_test_1,
                                      np.repeat(margin_y_val, len(mot_lda_on_mot_a_train_2_test_1)),
                                      color=nasal_color)
                    axs[1, 1].scatter(mot_lda_on_mot_b_train_2_test_1,
                                      np.repeat(margin_y_val, len(mot_lda_on_mot_b_train_2_test_1)),
                                      color=temporal_color)

                    axs[1, 1].set_title('dprime = %.2f' % mot_train_2_test_1_dprime, size=title_txt_size)

                    # VIS -> MOT classification
                    axs[0, 2].scatter(vis_lda_on_mot_a_train_1_test_2,
                                      np.repeat(margin_y_val, len(vis_lda_on_mot_a_train_1_test_2)),
                                      color=nasal_color)
                    axs[0, 2].scatter(vis_lda_on_mot_b_train_1_test_2,
                                      np.repeat(margin_y_val, len(vis_lda_on_mot_b_train_1_test_2)),
                                      color=temporal_color)

                    axs[0, 2].set_title('dprime = %.2f' % vis_lda_mot_train_1_test_2_dprime, size=title_txt_size)

                    axs[1, 2].scatter(vis_lda_on_mot_a_train_2_test_1,
                                      np.repeat(margin_y_val, len(vis_lda_on_mot_a_train_2_test_1)),
                                      color=nasal_color)
                    axs[1, 2].scatter(vis_lda_on_mot_b_train_2_test_1,
                                      np.repeat(margin_y_val, len(vis_lda_on_mot_b_train_2_test_1)),
                                      color=temporal_color)

                    axs[1, 2].set_title('dprime = %.2f' % vis_lda_mot_train_2_test_1_dprime, size=title_txt_size)


                    # MOT -> VIS classification
                    axs[0, 3].scatter(mot_lda_on_vis_a_train_1_test_2,
                                      np.repeat(margin_y_val, len(mot_lda_on_vis_a_train_1_test_2)),
                                      color=nasal_color)
                    axs[0, 3].scatter(mot_lda_on_vis_b_train_1_test_2,
                                      np.repeat(margin_y_val, len(mot_lda_on_vis_b_train_1_test_2)),
                                      color=temporal_color)

                    axs[0, 3].set_title('dprime = %.2f' % mot_lda_vis_train_1_test_2_dprime, size=title_txt_size)

                    axs[1, 3].scatter(mot_lda_on_vis_a_train_2_test_1,
                                      np.repeat(margin_y_val, len(mot_lda_on_vis_a_train_2_test_1)),
                                      color=nasal_color)
                    axs[1, 3].scatter(mot_lda_on_vis_b_train_2_test_1,
                                      np.repeat(margin_y_val, len(mot_lda_on_vis_b_train_2_test_1)),
                                      color=temporal_color)

                    axs[1, 3].set_title('dprime = %.2f' % mot_lda_vis_train_2_test_1_dprime, size=title_txt_size)

                fig_name = '%s_cvLDA' % rec_name[recording_n]
                fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')



                with plt.style.context(splstyle.get_style('nature-reviews')):
                    fig = plt.figure(figsize=(8, 8))
                    gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                                          wspace=0.05, hspace=0.05)

                    ax = fig.add_subplot(gs[1, 0])
                    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
                    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

                    nasal_color = np.array([120, 173, 52]) / 255
                    temporal_color = np.array([179, 30, 129]) / 255

                    vis_nasal = ax.scatter(vis_lda_on_vis_a, mot_lda_on_vis_a, color=nasal_color, label='Vis nasal')
                    vis_temporal = ax.scatter(vis_lda_on_vis_b, mot_lda_on_vis_b, color=temporal_color,
                                              label='Vis temporal')
                    # Draw the separating line


                    # ax.scatter(vis_lda_on_mot_a, mot_lda_on_mot_a, color='blue', marker='x')
                    # ax.scatter(vis_lda_on_mot_b, mot_lda_on_mot_b, color='red', marker='x')

                    mot_nasal = ax.scatter(vis_lda_on_mot_a, mot_lda_on_mot_a, facecolor='none', edgecolor=nasal_color,
                                           linestyle=':', label='Mot nasal')
                    mot_temporal = ax.scatter(vis_lda_on_mot_b, mot_lda_on_mot_b, facecolor='none',
                                              edgecolor=temporal_color, linestyle=':', label='Mot temporal')

                    ax.set_xlabel('Vis LDA axis', size=11)
                    ax.set_ylabel('Mot LDA axis', size=11)

                    ax.set_xticks([-4, 0, 4])
                    ax.set_yticks([-4, 0, 4])

                    # 1 : temporal grating
                    # 0 : nasal grating

                    # Legend
                    # legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color='blue', label='Vis nasal', lw=0,
                    #                           markerfacecolor='blue', markersize=15),
                    #                    mpl.lines.Line2D([0], [0], marker='o', color='red', label='Vis temporal', lw=0,
                    #                           markerfacecolor='red', markersize=15),
                    #                    mpl.lines.Line2D([0], [0], marker=r'$\u25CC$', color='blue', label='Mot nasal', lw=0,
                    #                                     markerfacecolor='none', linestyle=':',
                    #                                     markersize=15)
                    #                    mpl.lines.Line2D([0], [0], marker=r'$\u25CC$',  color='red', label='Mot temporal', lw=0,
                    #                           markerfacecolor='none', markeredgecolor='blue', markersize=15)]

                    ax.legend(bbox_to_anchor=(0.7, 0.2))

                    mot_linestyle = ':'

                    # Kernel Density Estimate of the marginals : visual dimension
                    vis_lda_points_to_sample = np.linspace(-4, 4, 100)

                    vis_lda_on_vis_a_kde = sstats.gaussian_kde(vis_lda_on_vis_a.flatten())
                    vis_lda_on_vis_a_curve = vis_lda_on_vis_a_kde(vis_lda_points_to_sample)
                    ax_histx.plot(vis_lda_points_to_sample, vis_lda_on_vis_a_curve, color=nasal_color)

                    vis_lda_on_vis_b_kde = sstats.gaussian_kde(vis_lda_on_vis_b.flatten())
                    vis_lda_on_vis_b_curve = vis_lda_on_vis_b_kde(vis_lda_points_to_sample)
                    ax_histx.plot(vis_lda_points_to_sample, vis_lda_on_vis_b_curve, color=temporal_color)

                    vis_lda_on_mot_a_kde = sstats.gaussian_kde(vis_lda_on_mot_a.flatten())
                    vis_lda_on_mot_a_curve = vis_lda_on_mot_a_kde(vis_lda_points_to_sample)
                    ax_histx.plot(vis_lda_points_to_sample, vis_lda_on_mot_a_curve, color=nasal_color,
                                  linestyle=mot_linestyle)

                    vis_lda_on_mot_b_kde = sstats.gaussian_kde(vis_lda_on_mot_b.flatten())
                    vis_lda_on_mot_b_curve = vis_lda_on_mot_b_kde(vis_lda_points_to_sample)
                    ax_histx.plot(vis_lda_points_to_sample, vis_lda_on_mot_b_curve, color='red',
                                  linestyle=mot_linestyle)

                    margin_pos_vis = -0.1
                    margin_pos_mot = -0.2
                    marker_alpha = 1
                    ax_histx.scatter(vis_lda_on_vis_a, np.repeat(margin_pos_vis, len(vis_lda_on_vis_a)),
                                     color=nasal_color, alpha=marker_alpha)
                    ax_histx.scatter(vis_lda_on_vis_b, np.repeat(margin_pos_vis, len(vis_lda_on_vis_b)),
                                     color=temporal_color, alpha=marker_alpha)
                    # ax_histx.scatter(vis_lda_on_mot_a, np.repeat(margin_pos_mot, len(vis_lda_on_mot_a)), color='blue', marker='x', alpha=marker_alpha)
                    # ax_histx.scatter(vis_lda_on_mot_b, np.repeat(margin_pos_mot, len(vis_lda_on_mot_b)), color='red', marker='x', alpha=marker_alpha)

                    ax_histx.scatter(vis_lda_on_mot_a, np.repeat(margin_pos_mot, len(vis_lda_on_mot_a)),
                                     facecolor='none', edgecolor=nasal_color, linestyle=':', alpha=marker_alpha)
                    ax_histx.scatter(vis_lda_on_mot_b, np.repeat(margin_pos_mot, len(vis_lda_on_mot_b)),
                                     facecolor='none', edgecolor=temporal_color, linestyle=':', alpha=marker_alpha)

                    ax_histx.tick_params(labelbottom=False, length=0)
                    ax_histx.spines['bottom'].set_visible(False)

                    ax_histx.set_yticks([])
                    ax_histx.spines['left'].set_visible(False)

                    # Kernel Density Estimate of the marginals : motor dimension
                    mot_lda_points_to_sample = np.linspace(-4, 4, 100)

                    mot_lda_on_vis_a_kde = sstats.gaussian_kde(mot_lda_on_vis_a.flatten())
                    mot_lda_on_vis_a_curve = mot_lda_on_vis_a_kde(mot_lda_points_to_sample)
                    ax_histy.plot(mot_lda_on_vis_a_curve, mot_lda_points_to_sample, color=nasal_color)

                    mot_lda_on_vis_b_kde = sstats.gaussian_kde(mot_lda_on_vis_b.flatten())
                    mot_lda_on_vis_b_curve = mot_lda_on_vis_b_kde(mot_lda_points_to_sample)
                    ax_histy.plot(mot_lda_on_vis_b_curve, mot_lda_points_to_sample, color=temporal_color)

                    mot_lda_on_mot_a_kde = sstats.gaussian_kde(mot_lda_on_mot_a.flatten())
                    mot_lda_on_mot_a_curve = mot_lda_on_mot_a_kde(mot_lda_points_to_sample)
                    ax_histy.plot(mot_lda_on_mot_a_curve, mot_lda_points_to_sample, color=nasal_color,
                                  linestyle=mot_linestyle)

                    mot_lda_on_mot_b_kde = sstats.gaussian_kde(mot_lda_on_mot_b.flatten())
                    mot_lda_on_mot_b_curve = mot_lda_on_mot_b_kde(mot_lda_points_to_sample)
                    ax_histy.plot(mot_lda_on_mot_b_curve, mot_lda_points_to_sample, color=temporal_color,
                                  linestyle=mot_linestyle)

                    ax_histy.scatter(np.repeat(margin_pos_vis, len(mot_lda_on_vis_a)), mot_lda_on_vis_a,
                                     color=nasal_color, alpha=marker_alpha)
                    ax_histy.scatter(np.repeat(margin_pos_vis, len(mot_lda_on_vis_b)), mot_lda_on_vis_b,
                                     color=temporal_color, alpha=marker_alpha)

                    # ax_histy.scatter(np.repeat(margin_pos_mot, len(mot_lda_on_mot_a)), mot_lda_on_mot_a, color='blue', marker='x', alpha=marker_alpha)
                    # ax_histy.scatter(np.repeat(margin_pos_mot, len(mot_lda_on_mot_b)), mot_lda_on_mot_b, color='red', marker='x', alpha=marker_alpha)
                    ax_histy.scatter(np.repeat(margin_pos_mot, len(mot_lda_on_mot_a)), mot_lda_on_mot_a,
                                     facecolor='none', edgecolor=nasal_color, linestyle=':', alpha=marker_alpha)
                    ax_histy.scatter(np.repeat(margin_pos_mot, len(mot_lda_on_mot_b)), mot_lda_on_mot_b,
                                     facecolor='none', edgecolor=temporal_color, linestyle=':', alpha=marker_alpha)

                    ax_histy.tick_params(labelleft=False, length=0)
                    ax_histy.spines['left'].set_visible(False)

                    ax_histy.set_xticks([])
                    ax_histy.spines['bottom'].set_visible(False)

                fig_name = '%s_LDA' % rec_name[recording_n]
                fig_ext = '.svg'

                # save font as text rather than paths
                plt.rcParams['svg.fonttype'] = 'none'

                fig.savefig(os.path.join(fig_folder, fig_name + fig_ext), dpi=300, bbox_inches='tight')



            # Summarise the dprime for each experiment

            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, ax = plt.subplots()
                fig.set_size_inches(5, 4)
                all_dprime = np.stack([mot_mot_dprime_per_exp, vis_vis_dprime_per_exp,
                                       mot_vis_dprime_per_exp, vis_mot_dprime_per_exp])

                for n_exp in np.arange(0, len(rec_name)):
                    ax.plot(all_dprime[:, n_exp])


                ax.set_ylabel('Absolute dprime value', size=11)
                ax.set_xticks([0, 1, 2, 3])
                ax.set_xticklabels([r'Mot $\rightarrow$ Mot', r'Vis $\rightarrow$ Vis',
                                    r'Mot $\rightarrow$ Vis', r'Vis $\rightarrow$ Mot'])

            fig_name = 'all_exp_dprime_values'
            fig.savefig(os.path.join(fig_folder, fig_name), dpi=300, bbox_inches='tight')


            # Summarise the vis mot weights
            with plt.style.context(splstyle.get_style('nature-reviews')):
                fig, ax = plt.subplots()
                fig.set_size_inches(4, 4)
                all_weight_cosine_sim = np.stack([mot_mot_weight_cosine_sim_per_exp, vis_vis_weight_cosine_sim_per_exp,
                                       mot_vis_weight_cosine_sim_per_exp, vis_mot_weight_cosine_sim_per_exp])

                for n_exp in np.arange(0, len(rec_name)):
                    ax.plot(all_weight_cosine_sim[:, n_exp], color='gray')


                ax.set_ylabel('Weight cosine similarity', size=11)
                ax.set_xticks([0, 1, 2, 3])
                ax.set_xticklabels([r'Mot $\rightarrow$ Mot', r'Vis $\rightarrow$ Vis',
                                    r'Mot $\rightarrow$ Vis', r'Vis $\rightarrow$ Mot'])

            fig_name = 'all_exp_LDA_weight_cosine_similarity_values'

            for ext in process_params[process]['fig_exts']:
                fig.savefig(os.path.join(fig_folder, fig_name + ext), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
   main()


