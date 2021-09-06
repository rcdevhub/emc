# -*- coding: utf-8 -*-
"""
Ethical Model Calibration

Main file for running experiments

Created on Fri Aug  6 13:46:14 2021

@author: rcpc4
"""

'''
#============================================
# Setup
#============================================
'''

import os
os.chdir("WORKING DIRECTORY")
import sys

import numpy as np
import pandas as pd
import time
import copy

from Preprocessing import *
from EDA import *
from Functions import *
from Parameters import *

'''
#============================================
# Initialise parameters
#============================================
'''

params = call_params()

'''
#============================================
# Experiment administration
#============================================
'''

# Get timestamp for experiment
timestamp = time.strftime('-%Y-%m-%d-%H-%M',time.localtime())
print(timestamp)

# Choose dataset ('diabetes','neuroticism')
data_type = 'neuroticism'
# Choose task type ('cls','reg')
task_type = 'cls'
# Choose model type ('nn') only
model_type = 'nn'
# Choose target variable
# Choose from {'diabetes': ('diab', 'hba1c'), 'neuroticism': ('neur','sex','Hb')}
# Not all combinations are available
target_var = 'sex'

# Switches for optional actions
perform_eda = True
fit_rf = True
reg_plot = True
auto_plot = True
gmm_plot = True
worst_grp_plot = True
gmm_plot_rebal = True
save_results = True
save_data = True

# Make plots directory
plots_main_dir = 'Plots/'+data_type+'_'+task_type+'_'+target_var+'_'+timestamp
os.mkdir(plots_main_dir)

# Make output directory
outputdir = 'Output/'+data_type+'_'+task_type+'_'+target_var+'_'+timestamp
os.mkdir(outputdir)

'''
#============================================
# Import and preprocess data
#============================================
'''

filepaths = params['filepaths']

# Import and preprocess data
desc_train,desc_val,data_train,data_train_norm,data_val,data_val_norm\
    = data_preprocess(data_type,filepaths)

# Make datafiles for modelling
data_train_mod,target_train_mod,data_val_mod,target_val_mod,\
    data_train_auto,data_val_auto\
        = prep_datasets(data_train,data_train_norm,data_val,data_val_norm,data_type,
                  task_type,target_var)

'''
#============================================
# Exploratory Data Analysis
#============================================
'''

if perform_eda:
    # Save EDA graphs
    eda_graphs(data_train,plots_main_dir,data_type)
    
'''
#============================================
# Experiment loop
#============================================
'''

# Make dict to store results of n trials
trials = {}
trials['timestamp'] = timestamp
trials['data_type'] = data_type
trials['task_type'] = task_type
trials['model_type'] = model_type
trials['target_var'] = target_var

# Make dict to store data of n trials
trial_data = {}

num_trials = params['num_trials']

for trial in range(num_trials):
    
    # Make plots sub-directory
    plotsdir = plots_main_dir+'/'+str(trial)
    os.mkdir(plotsdir)
        
    '''
    #============================================
    # Fit supervised models
    #============================================
    '''
    
    # Fit random forest
    if fit_rf:
    
        rf_max_depth = params[data_type][task_type]['rf']['max_depth']    
        colnames = params[data_type][task_type]['rf'][target_var]['feat_colnames']
        
        if task_type == 'cls':
            pred_train_rf,pred_val_rf,results_rf = fit_cls_rf(data_train_mod,target_train_mod,data_val_mod,target_val_mod,
                                                              plotsdir,rf_max_depth,colnames)
        elif task_type == 'reg':
            pred_train_rf,pred_val_rf,results_rf = fit_reg_rf(data_train_mod,target_train_mod,data_val_mod,target_val_mod,
                                                              plotsdir,rf_max_depth,colnames)
        else:
            raise ValueError("Unknown task type")
    
    # Fit neural network
    nums = params[data_type][task_type]['nn']['nums']
    num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out = nums
    learn_rate = params['learn_rate']
    batch_size = params['batch_size']
    num_epochs = params[data_type][task_type]['nn'][target_var]['epochs']
    
    if task_type == 'cls':
        model_nn,pred_train_nn,pred_val_nn,results_nn = fit_cls_nn(num_in, num_hidden_1,
                                                      num_hidden_2, num_hidden_3, num_out,
                                                      learn_rate, batch_size, num_epochs,
                                                       data_train_mod, target_train_mod,
                                                       data_val_mod, target_val_mod,               
                                                       overweight=False, metrics=True)
    elif task_type == 'reg':
        model_nn,pred_train_nn,pred_val_nn,results_nn = fit_reg_nn(num_in, num_hidden_1,
                                                          num_hidden_2, num_hidden_3, num_out,
                                                          learn_rate, batch_size, num_epochs,
                                                          data_train_mod, target_train_mod,
                                                          data_val_mod, target_val_mod,
                                                          metrics=True)
    else:
        raise ValueError("Unknown task type")
    
    # Save regression diagnostic plots
    if (task_type=='reg') & (reg_plot):
        
        box_var = params[data_type][task_type]['nn'][target_var]['box_var']
        p_a_limits = params[data_type][task_type]['nn'][target_var]['p_a_limits']
        r_a_limits = params[data_type][task_type]['nn'][target_var]['r_a_limits']
        bins = params[data_type][task_type]['nn'][target_var]['bins']
        
        # Save regression diagnostic plots
        plot_reg_diag(pred_train_nn,target_train_mod,target_var,
                      plotsdir,p_a_limits,r_a_limits,bins,
                      data_train,box_var=box_var)
        
    '''
    #============================================
    # Demonstrate bias
    #============================================
    '''
    
    split_vars = params[data_type][task_type]['nn'][target_var]['bias_split_vars']
    
    results_bias = demo_bias(data_type,task_type,target_var,
                             data_train,data_val,
                             target_train_mod,target_val_mod,
                             pred_train_nn,pred_val_nn,
                             split_vars)
    
    '''
    #============================================
    # Fit autoencoder
    #============================================
    '''
    
    # While loop to catch cases where no validation data in GMM groups
    # or not enough groups made
    attempts = 1
    while True:
        
        print("auto-grouping attempts",attempts)
    
        nums = params[data_type]['auto']['nums']
        num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out = nums
        learn_rate = params['learn_rate']
        batch_size = params['batch_size']
        num_epochs = params[data_type]['auto']['epochs']
        
        model_auto,pred_train_auto,latent_train,latent_val = fit_auto(num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out,
                                                           learn_rate, batch_size, num_epochs,
                                                           data_train_auto, data_val_auto)
        
        # Save autoencoder plots
        plot_vars = params[data_type]['auto']['plot_vars']
        
        if auto_plot:
            plot_auto(pred_train_auto,latent_train,data_train,
                      data_train_auto,plotsdir,plot_vars,
                      task_type,target_var,
                      pred_train_nn,target_train_mod)
            
        '''
        #============================================
        # Fit GMM to autoencoder latent space
        #============================================
        '''
        
        num_components = params[data_type]['gmm']['num_components']
       
        pred_gmm_train,pred_gmm_val = fit_gmm(num_components,latent_train,latent_val)
    
        # Calculate counts by GMM component group
        num_gmm_train_component,count_gmm_train_component = np.unique(pred_gmm_train,return_counts=True)
        num_gmm_val_component,count_gmm_val_component = np.unique(pred_gmm_val,return_counts=True)
        
        if (num_gmm_train_component.shape == num_gmm_val_component.shape)\
            & (num_gmm_train_component.shape[0] == num_components):
            break
        
        attempts += 1
        
        if attempts > 10:
            raise ValueError("GMM group with no validation data or not enough GMM groups")
    
    # Calculate performance by GMM component group
    metric = params[data_type][task_type]['metric']
    
    perf_train_component = perf_by_component(target_train_mod,pred_train_nn,pred_gmm_train,
                                             task_type,num_components,metric)
    
    perf_val_component = perf_by_component(target_val_mod,pred_val_nn,pred_gmm_val,
                                             task_type,num_components,metric)
    
    # Calculate Gini coefficient
    gini_train = calc_gini(perf_train_component,count_gmm_train_component)
    gini_val = calc_gini(perf_val_component,count_gmm_val_component)
    
    # Save model performance by GMM group plots
    if gmm_plot:
        plot_gmm(pred_gmm_train,latent_train,
                 data_type,task_type,target_var,
                 plotsdir,count_gmm_train_component,
                 perf_train_component,metric)
    
    '''
    #============================================
    # Identify underperforming groups
    #============================================
    '''
    
    # There are two approaches here:
    # (i)  Identify the n largest groups in the worst x-th %ile performance
    #      These "worst large" groups are later analysed to see their composition
    # (ii) Identify the lower half of groups in terms of performance (below median)
    #      These "underperforming" groups are later the subject of debiasing (rebalancing)
    
    percentile_threshold = params[data_type][task_type]['percentile_threshold']
    num_worst_lrg_groups = params[data_type][task_type]['num_worst_lrg_groups']
    
    # Find GMM groups in worst x-th %ile, ordered by size descending
    percentile_value = np.percentile(perf_train_component,percentile_threshold)
    sort_idx = np.flip(np.argsort(count_gmm_train_component))
    
    # Select top n of these groups by size ("worst large") 
    if task_type == 'reg': # Lower RMSE better
        worst_group_nums = num_gmm_train_component[sort_idx][perf_train_component[sort_idx] > percentile_value]
    elif task_type == 'cls': # Higher F1ma better
        worst_group_nums = num_gmm_train_component[sort_idx][perf_train_component[sort_idx] < percentile_value]
    else:
        raise ValueError("Unknown task type")
    worst_lrg_group_nums = worst_group_nums[0:num_worst_lrg_groups]
    
    # Indicate observations in these groups
    worst_lrg_indic_train = np.zeros(pred_gmm_train.shape[0])
    for i in range(num_worst_lrg_groups):
        worst_lrg_indic_train += 1*(pred_gmm_train == worst_lrg_group_nums[i])
        
    worst_lrg_indic_val = np.zeros(pred_gmm_val.shape[0])
    for i in range(num_worst_lrg_groups):
        worst_lrg_indic_val += 1*(pred_gmm_val == worst_lrg_group_nums[i])    
    
    # Find GMM groups in lower half of performance, sorted by size descending
    sort_idx = np.flip(np.argsort(count_gmm_train_component))
    if task_type == 'reg': # Lower RMSE better
        underperf_group_nums =  num_gmm_train_component[sort_idx][perf_train_component[sort_idx] > np.median(perf_train_component)]
    elif task_type == 'cls': # Higher F1ma better
        underperf_group_nums =  num_gmm_train_component[sort_idx][perf_train_component[sort_idx] < np.median(perf_train_component)]
    else:
        raise ValueError("Unknown task type")
        
    # Indicate groups in lower half of performance ("underperforming")
    num_underperf_groups = underperf_group_nums.shape[0]
    underperf_grp_indic_train = np.zeros(pred_gmm_train.shape[0])
    for i in range(num_underperf_groups):
        underperf_grp_indic_train += 1*(pred_gmm_train == underperf_group_nums[i])
        
    underperf_grp_indic_val = np.zeros(pred_gmm_val.shape[0])
    for i in range(num_underperf_groups):
        underperf_grp_indic_val += 1*(pred_gmm_val == underperf_group_nums[i])
    
    # Calculate performance
    results_worst_lrg = perf_by_group(target_train_mod,pred_train_nn,worst_lrg_indic_train,
                                      target_val_mod,pred_val_nn,worst_lrg_indic_val,
                                      task_type)
    
    results_underperf = perf_by_group(target_train_mod,pred_train_nn,underperf_grp_indic_train,
                                      target_val_mod,pred_val_nn,underperf_grp_indic_val,
                                      task_type)
    
    '''
    #============================================
    # Find characteristics of worst large groups
    #============================================
    '''
    
    # Conduct permutation tests on worst large groups
    permut_vars = params[data_type]['gmm']['permut_vars']
    
    results_permut = conduct_permutation_tests(permut_vars,num_worst_lrg_groups,
                                              worst_lrg_group_nums,data_train,
                                              pred_gmm_train)
    
    # Visualise characteristics of worst large groups
    if worst_grp_plot:
        hist_vars = params[data_type]['gmm']['hist_vars']
        barplot_vars = params[data_type]['gmm']['barplot_vars']
        
        # Save graphs
        plt_grp_latent(latent_train,worst_lrg_indic_train,
                       data_type,task_type,target_var,plotsdir)
        plot_grp_hist(data_train,pred_gmm_train,worst_lrg_group_nums,hist_vars,
                       data_type,task_type,target_var,plotsdir)
        plot_grp_bar(data_train,pred_gmm_train,worst_lrg_group_nums,barplot_vars,
                       data_type,task_type,target_var,plotsdir)
    
    '''
    #============================================
    # Debias by rebalancing on the underperforming group
    #============================================
    '''
    
    upsample_mult = params[data_type][task_type][model_type][target_var]['upsample_mult']
    downsample_mult = params[data_type][task_type][model_type][target_var]['downsample_mult']
    
    # Sometimes it is necessary to slightly adjust the multiplier to prevent a PyTorch error
    mod_attempts = 1
    while True:
    
        print("mod_attempts",mod_attempts)
    
        # Make rebalanced datasets
        data_train_mod_rebal,target_train_mod_rebal = get_rebal_data(data_train_mod,
                                                                   target_train_mod,
                                                                   underperf_grp_indic_train,
                                                                   upsample_mult,
                                                                   downsample_mult)
        
        # Make sure we don't send a batch of 1 into PyTorch, which causes an error
        if data_train_mod_rebal.shape[0] % batch_size != 1:
            break
            
        # Slightly adjust the mult to stop the error
        upsample_mult += 0.1
        mod_attempts += 1
            
        if mod_attempts > 10:
            raise ValueError("Training data results in batch size of one")
    
    
    # Retrain neural network model
    nums = params[data_type][task_type]['nn']['nums']
    num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out = nums
    learn_rate = params['learn_rate']
    batch_size = params['batch_size']
    num_epochs = params[data_type][task_type]['nn'][target_var]['epochs_rebalanced']
    
    # Notes:
    # (a) Validation set is unchanged
    # (b) Initial results not stored as include oversampled points
    if task_type == 'cls':
        model_nn_rebal,_,_,_ = fit_cls_nn(num_in, num_hidden_1,
                                        num_hidden_2, num_hidden_3, num_out,
                                        learn_rate, batch_size, num_epochs,
                                         data_train_mod_rebal, target_train_mod_rebal,
                                         data_val_mod, target_val_mod,               
                                         overweight=False, metrics=True)
    elif task_type == 'reg':
        model_nn_rebal,_,_,_ = fit_reg_nn(num_in, num_hidden_1,
                                        num_hidden_2, num_hidden_3, num_out,
                                        learn_rate, batch_size, num_epochs,
                                        data_train_mod_rebal, target_train_mod_rebal,
                                        data_val_mod, target_val_mod,
                                        metrics=True)
    
    else:
        raise ValueError("Unknown task type")
        
    # Get results on original training data
    pred_train_nn_rebal,pred_val_nn_rebal,results_nn_rebal = pred_perf(task_type,model_type,
                                                                              model_nn_rebal,
                                                                              data_train_mod,target_train_mod,
                                                                              data_val_mod,target_val_mod)
    
    '''
    #============================================
    # Analyse effect of rebalancing with metrics and plots
    #============================================
    '''
    
    # Calculate performance of rebalanced group
    results_underperf_rebal = perf_by_group(target_train_mod,pred_train_nn_rebal,
                                            underperf_grp_indic_train,
                                            target_val_mod,pred_val_nn_rebal,
                                            underperf_grp_indic_val,
                                            task_type)
    
    # Calculate performance on worst large group (for analysis)
    results_worst_lrg_rebal = perf_by_group(target_train_mod,pred_train_nn_rebal,
                                            worst_lrg_indic_train,
                                            target_val_mod,pred_val_nn_rebal,
                                            worst_lrg_indic_val,
                                            task_type)
    
    # Calculate rebalanced performance by GMM component
    perf_train_component_rebal = perf_by_component(target_train_mod,pred_train_nn_rebal,
                                                   pred_gmm_train,task_type,
                                                   num_components,metric)
    
    perf_val_component_rebal = perf_by_component(target_val_mod,pred_val_nn_rebal,
                                                 pred_gmm_val,task_type,
                                                 num_components,metric)
    
    
    # Calculate rebalanced gini coefficient
    gini_train_rebal = calc_gini(perf_train_component_rebal,count_gmm_train_component)
    gini_val_rebal = calc_gini(perf_val_component_rebal,count_gmm_val_component)
    
    # Calculate results by one-way bias splits
    results_bias_rebal = demo_bias(data_type,task_type,target_var,
                                 data_train,data_val,
                                 target_train_mod,target_val_mod,
                                 pred_train_nn_rebal,pred_val_nn_rebal,
                                 split_vars)
    
    # Add results to dataframe for plotting
    results_rebal_component = pd.DataFrame(columns=['grp_num','count','orig','rebal','change'])
    results_rebal_component['grp_num'] = num_gmm_train_component
    results_rebal_component['count'] = count_gmm_train_component
    results_rebal_component['orig'] = perf_train_component
    results_rebal_component['rebal'] = perf_train_component_rebal
    results_rebal_component['change'] = perf_train_component-perf_train_component_rebal
    
    # Save model performance by GMM group plots
    if gmm_plot_rebal:
        plot_gmm(pred_gmm_train,latent_train,
                 data_type,task_type,target_var,
                 plotsdir,count_gmm_train_component,
                 perf_train_component_rebal,metric,
                 desc=' rebalanced')
    
        plot_gmm_comp(results_rebal_component,
                      data_type,task_type,target_var,
                      metric,plotsdir)
        
    '''
    #============================================
    # Store results
    #============================================
    '''
        
    trial_ind = str(trial)
    
    trials[trial_ind] = {}
    trials[trial_ind]['results'] = {}
    trials[trial_ind]['results']['results_mod'] = results_nn
    trials[trial_ind]['results']['results_bias'] = results_bias
    trials[trial_ind]['results']['results_worst_lrg'] = results_worst_lrg
    trials[trial_ind]['results']['results_underperf'] = results_underperf
    trials[trial_ind]['results']['results_permut'] = results_permut
    trials[trial_ind]['results']['results_nn_rebal'] = results_nn_rebal
    trials[trial_ind]['results']['results_bias_rebal'] = results_bias_rebal
    trials[trial_ind]['results']['results_worst_lrg_rebal'] = results_worst_lrg_rebal
    trials[trial_ind]['results']['results_underperf_rebal'] = results_underperf_rebal
    trials[trial_ind]['results']['results_rebal_component'] = results_rebal_component
    trials[trial_ind]['results']['perf_train_component'] = perf_train_component
    trials[trial_ind]['results']['perf_val_component'] = perf_val_component
    trials[trial_ind]['results']['perf_train_component_rebal'] = perf_train_component_rebal
    trials[trial_ind]['results']['perf_val_component_rebal'] = perf_val_component_rebal
    trials[trial_ind]['results']['gini_train'] = gini_train
    trials[trial_ind]['results']['gini_val'] = gini_val
    trials[trial_ind]['results']['gini_train_rebal'] = gini_train_rebal
    trials[trial_ind]['results']['gini_val_rebal'] = gini_val_rebal
    
    if save_results:
        np.save(outputdir+'/trials.npy',trials)
    
    '''
    #============================================
    # Store models and data
    #============================================
    '''
    
    trial_data[trial_ind] = {}
    trial_data[trial_ind]['timestamp'] = timestamp
    trial_data[trial_ind]['models'] = {}
    trial_data[trial_ind]['models']['mod'] = model_nn
    trial_data[trial_ind]['models']['mod_params'] = model_nn.state_dict()
    trial_data[trial_ind]['models']['auto'] = model_auto
    trial_data[trial_ind]['models']['auto_params'] = model_auto.state_dict()
    trial_data[trial_ind]['models']['mod_rebal'] = model_nn_rebal
    trial_data[trial_ind]['models']['mod_rebal_params'] = model_nn_rebal.state_dict()
    trial_data[trial_ind]['output'] = {}
    trial_data[trial_ind]['output']['pred_train_nn'] = pred_train_nn
    trial_data[trial_ind]['output']['pred_val_nn'] = pred_val_nn
    trial_data[trial_ind]['output']['latent_train'] = latent_train
    trial_data[trial_ind]['output']['latent_val'] = latent_val
    trial_data[trial_ind]['output']['pred_gmm_train'] = pred_gmm_train
    trial_data[trial_ind]['output']['pred_gmm_val'] = pred_gmm_val
    trial_data[trial_ind]['output']['worst_lrg_group_nums'] = worst_lrg_group_nums
    trial_data[trial_ind]['output']['underperf_group_nums'] = underperf_group_nums
    trial_data[trial_ind]['output']['pred_train_nn_rebal'] = pred_train_nn_rebal
    trial_data[trial_ind]['output']['pred_val_nn_rebal'] = pred_val_nn_rebal

    if save_data:
        np.save(outputdir+'/trial_data.npy',trial_data)
    
# Load saved items
# trials = np.load(outputdir+'/trials.npy',allow_pickle=True).item()
# trial_data = np.load(outputdir+'/trial_data.npy',allow_pickle=True).item()

'''
#============================================
# Plot results
#============================================
'''

# Calculate and plot results
results_box,results_summary,gini_box,results_gini = plot_results_box(num_trials,metric,trials,plots_main_dir,
                                                            data_type,task_type,model_type,target_var)