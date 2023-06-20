# [Title of Paper] 


## Figure 4 : Decoding 

Code for generating panels in figure 4 are in SCmotVisCoding/Decoding/run_decoding.py

To generate figure 4a and figure 4b (windowed decoding)

 - set `processes_to_run = ['do_windowed_decoding', 'plot_windowed_decoding']` in the main function 
 - this will perform both the saccade direction and visual grating direction decoding 

To generate figure 4c 

 - set `processes_to_run = ['fit_a_evaluate_on_a_and_b']` in the main function 
 - this will fit a model on either saccade direction or visual direction ("a" or "b") and evaluate the decoding peformance on noth 
 - once you have ran this process twice (once setting modality_a to motor and once settign modality_b to visual, you can run the script with `processes_to_run = ['plot_a_evaluate_on_a_and_b_results']` to plot the results



## Figure 5 : Regression 


