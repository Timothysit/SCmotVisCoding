# [Title of Paper] 


# How to generate each figure in the paper

## Figure 4 : Decoding

Code for generating panels in figure 4 are in SCmotVisCoding/Decoding/run_decoding.py

To generate figure 4a and figure 4b (windowed decoding)

 - set `processes_to_run = ['do_windowed_decoding', 'plot_windowed_decoding']` in the main function 
 - this will perform both the saccade direction and visual grating direction decoding 

To generate figure 4c 

 - set `processes_to_run = ['fit_a_evaluate_on_a_and_b']` in the main function 
 - this will fit a model on either saccade direction or visual direction ("a" or "b") and evaluate the decoding peformance on noth 
 - once you have ran this process twice (once setting modality_a to motor and once settign modality_b to visual, you can run the script with `processes_to_run = ['plot_a_evaluate_on_a_and_b_results']` to plot the results

To generate figure 4d, 4e, 4f, 4g, 4h, 4i (the weights plots)

 - first set `processes_to_run = ['fit_separate_decoders']`
 - then set `processes_to_run = ['plot_motor_and_visual_decoder_weights']`

To generate the LDA figures

 - first set `processes_to_run = ['cal_d_prime']`
 - then set `processes_to_run = ['plot_d_prime']`
 - the first figure is one example of these plots
 - then set `processes_to_run = ['cal_trial_angles_train_test']` to get the rest of the LDA-related figures (the cosine similarity figure etc.)



## Figure 5 : Regression

The regresison figures (other than figure 5a) first requires the regression analysis to be done. You do this by

 - going to SCmotVisCoding/Regression/run_regression.py
 - set `processes_to_run = ['fit_regression_model']`
 - you need to run this three times, once setting exp_type to 'both', once setting exp_type to 'grating', and once setting exp_type to 'gray', also adjust the X_sets_to_compare parameter accordingly: only fit saccade models in gray sreen, and fit saccade and visual models in "both" or "grating" experiments

To generate figure 5a

 - follow the notebook in SCmotVisCoding/Regression/plot-regression-model-schematic.ipynb

To generate figure 5b, 5c, 5d

 - follow the notebook in SCmotVisCoding/Regression/example-neuron-fit
 - sorry it is a bit messy at the moment, but if you run through each cell then in the final few cells you should get those figures

To generate figure 5e

 - in SCmotVisCoding/Regression/run_regression.py, set `processes_to_run = ['compare_ev_with_shuffled']` in the main function 

To generate figure 5f

 - in SCmotVisCoding/Regression/run_regression.py, set `processes_to_run = ['compare_saccade_kernels']` in the main function

To generate figure 5g

 - in SCmotVisCoding/Regression/run_regression.py, set `processes_to_run = ['plot_kernel_scatter']` in the main function
 - set  x_axis_kernel='saccade_dir_nasal', y_axis_kernel='saccade_dir_temporal'

To generate figure 5h

 - in SCmotVisCoding/Regression/run_regression.py, set `processes_to_run = ['plot_kernel_scatter']` in the main function
 - set  x_axis_kernel='saccade_dir_diff', y_axis_kernel='saccade_dir_diff'
 - this will produce the version of the plot without the truncated axis and saved the corresponding data for this scatter
 - to make the version with the truncated axis, follow the code in SCmotVisCoding/Regression/compare-saccade-kernel-in-vis-and-gray-exp.ipynb


