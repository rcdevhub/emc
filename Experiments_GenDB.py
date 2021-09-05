# -*- coding: utf-8 -*-
"""
Ethical Model Calibration

Additional Experiments - Generative Debiasing (GenDB)

Created on Thu Sep  2 16:11:13 2021

@author: rcpc4
"""

# Requires preprocessed data from main
# Dataset: neuroticism
# Task type: reg
# Target variable: neur

# Imports
from Functions_BVAE import *

'''
#============================================
# Experiment loop
#============================================
'''

# Make dict to store results of n trials
trials_gendb = {}

num_trials = 10

for trial in range(num_trials):

    # Make dict to store results
    
    results_gendb = {}
    results_gendb['training'] = {}
    results_gendb['training']['Original'] = {}
    results_gendb['training']['Original']['Underperf'] = {}
    results_gendb['training']['GenDB'] = {}
    results_gendb['training']['GenDB']['Underperf'] = {}
    results_gendb['validation'] = {}
    results_gendb['validation']['Original'] = {}
    results_gendb['validation']['Original']['Underperf'] = {}
    results_gendb['validation']['GenDB'] = {}
    results_gendb['validation']['GenDB']['Underperf'] = {}
    
    '''
    #============================================
    # Neural network model - Regression
    #============================================
    '''
    
    # Regression - Neural Network - Neuroticism
    
    nums = params[data_type][task_type]['nn']['nums']
    num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out = nums
    learn_rate = params['learn_rate']
    batch_size = params['batch_size']
    num_epochs = params[data_type][task_type]['nn'][target_var]['epochs']

    model_nn,pred_train_nn,pred_val_nn,results_nn = fit_reg_nn(num_in, num_hidden_1,
                                                          num_hidden_2, num_hidden_3, num_out,
                                                          learn_rate, batch_size, num_epochs,
                                                          data_train_mod, target_train_mod,
                                                          data_val_mod, target_val_mod,
                                                          metrics=True)
    
    mse_train = mean_squared_error(target_train_mod, pred_train_nn)
    rmse_train = np.sqrt(mse_train)
    nrmse_train = rmse_train/np.mean(target_train_mod)
    
    mse_val = mean_squared_error(target_val_mod, pred_val_nn)
    rmse_val = np.sqrt(mse_val)
    nrmse_val = rmse_val/np.mean(target_val_mod)
    
    print("Training RMSE",rmse_train,"Validation RMSE",rmse_val)
    print("Training NRMSE",nrmse_train,"Validation RMSE",nrmse_val)
    
    # Store
    results_gendb['training']['Original']['NRMSE'] = nrmse_train
    results_gendb['validation']['Original']['NRMSE'] = nrmse_val
    
    '''
    #============================================
    # Autoencoder
    #============================================
    '''
    
    # Set layer sizes
    nums = params[data_type]['auto']['nums']
    num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out = nums
    learn_rate = params['learn_rate']
    batch_size = params['batch_size']
    num_epochs = params[data_type]['auto']['epochs']
    
    # While loop to catch cases where no validation data in GMM groups
    # or not enough groups made
    attempts = 1
    while True:
        
        print("auto-grouping attempts",attempts)
    
        # Train model
        model_auto,pred_train_auto,latent_train,latent_val = fit_auto(num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out,
                                                           learn_rate, batch_size, num_epochs,
                                                           data_train_auto, data_val_auto)
            
        '''
        #============================================
        # Mixture fit to autoencoder latent space
        #============================================
        '''
        
        num_components = params[data_type]['gmm']['num_components']
        
        # Fit model
        gm = GaussianMixture(n_components=num_components, random_state=0).fit(latent_train)
        
        # Output means
        gm.means_
        
        # Classify points
        pred_gmm_train = gm.predict(latent_train)
        pred_gmm_val = gm.predict(latent_val)
        
        num_gmm_train_component,count_gmm_train_component = np.unique(pred_gmm_train,return_counts=True)
        num_gmm_val_component,count_gmm_val_component = np.unique(pred_gmm_val,return_counts=True)
        
        if (num_gmm_train_component.shape == num_gmm_val_component.shape)\
            & (num_gmm_train_component.shape[0] == num_components):
            break
            
        attempts += 1
            
        if attempts > 10:
            raise ValueError("GMM group with no validation data or not enough GMM groups")
    
    rmse_train_component = np.zeros(num_components)
    nrmse_train_component = np.zeros(num_components)
    for i in range(num_components):
        rmse_comp = np.sqrt(mean_squared_error(target_train_mod[pred_gmm_train==i],
                                               pred_train_nn[pred_gmm_train==i]))
        nrmse_comp = rmse_comp/np.mean(target_train_mod[pred_gmm_train==i])
        rmse_train_component[i] = rmse_comp
        nrmse_train_component[i] = nrmse_comp
    
    rmse_val_component = np.zeros(num_components)
    nrmse_val_component = np.zeros(num_components)
    for i in range(num_components):
        rmse_comp = np.sqrt(mean_squared_error(target_val_mod[pred_gmm_val==i],
                                               pred_val_nn[pred_gmm_val==i]))
        nrmse_comp = rmse_comp/np.mean(target_val_mod[pred_gmm_val==i])
        rmse_val_component[i] = rmse_comp
        nrmse_val_component[i] = rmse_comp
    
    '''
    #============================================
    # Identify underperforming groups
    #============================================
    '''
    
    # Find groups in lower half of performance, with RMSE above that of overall model
    # Sorted by size descending (for consistency with "worst large" approach)
    sort_idx = np.flip(np.argsort(count_gmm_train_component))
    underperf_group_nums =  num_gmm_train_component[sort_idx]\
        [nrmse_train_component[sort_idx] > np.median(nrmse_train_component)]
    
    # Indicate groups in lower half of RMSE performance
    num_underperf_groups = underperf_group_nums.shape[0]
    underperf_grp_indic_train = np.zeros(pred_gmm_train.shape[0])
    for i in range(num_underperf_groups):
        underperf_grp_indic_train += 1*(pred_gmm_train == underperf_group_nums[i])
        
    underperf_grp_indic_val = np.zeros(pred_gmm_val.shape[0])
    for i in range(num_underperf_groups):
        underperf_grp_indic_val += 1*(pred_gmm_val == underperf_group_nums[i])
    
    # Calculate performance of this "underperforming" group (might need to rename)
    for j in np.unique(underperf_grp_indic_train):
        train_pct = 100*np.sum(underperf_grp_indic_train)/underperf_grp_indic_train.shape[0]
        val_pct = 100*np.sum(underperf_grp_indic_val)/underperf_grp_indic_val.shape[0]
        rmse_train = np.sqrt(mean_squared_error(target_train_mod[underperf_grp_indic_train==j],
                                                pred_train_nn[underperf_grp_indic_train==j]))
        nrmse_train = rmse_train/np.mean(target_train_mod[underperf_grp_indic_train==j])
    
        rmse_val = np.sqrt(mean_squared_error(target_val_mod[underperf_grp_indic_val==j],
                                                pred_val_nn[underperf_grp_indic_val==j]))
        nrmse_val = rmse_val/np.mean(target_val_mod[underperf_grp_indic_val==j])
        
        # Store
        results_gendb['training']['Original']['Underperf'][str(j)] = {}
        results_gendb['training']['Original']['Underperf'][str(j)]['NRMSE'] = nrmse_train
        results_gendb['validation']['Original']['Underperf'][str(j)] = {}
        results_gendb['validation']['Original']['Underperf'][str(j)]['NRMSE'] = nrmse_val
        
        print("Neuroticism prediction by underperf grp",j,"RMSE","Training",rmse_train,
              "Validation",rmse_val)
        print("Neuroticism prediction by underperf grp",j,"NRMSE","Training",nrmse_train,
              "Validation",nrmse_val)
    print("Underperf grp pct of data","Training",train_pct,"%","Validation",val_pct,"%")
    
    '''
    #============================================
    # Make BVAE training set
    #============================================
    '''
    
    # Use fully normalised data to match the earlier models
    data_train_bvae_gendb = data_train_norm.drop(['Unnamed: 0',
                                             'biobank_id',
                                             'haematocrit'],axis=1).values
    
    data_val_bvae_gendb = data_val_norm.drop(['Unnamed: 0',
                                             'biobank_id',
                                             'haematocrit'],axis=1).values
    
    '''
    #============================================
    # Make and train BVAE on points to be synthesized
    #============================================
    '''
    
    # Train the BVAE
    
    d_train = data_train_bvae_gendb[underperf_grp_indic_train==1]
    d_val = data_val_bvae_gendb[underperf_grp_indic_val==1]
    
    num_input = 29
    num_hidden_1 = 16
    num_hidden_2 = 16
    num_latent = 8
    num_output = 29
    
    batch_size = 10
    learning_rate = 0.001
    beta = 0.005 
    
    num_epochs = 20
    
    # Initialise model
    bva1 = bvae(num_input,num_hidden_1,num_hidden_2,num_latent,num_output)
    
    optimiser = torch.optim.Adam(bva1.parameters(),lr=learning_rate)
    
    # Define training set and dataloader
    X_tensor = torch.tensor(d_train, requires_grad=False)
    trainset = torch.utils.data.TensorDataset(X_tensor)
    dloader = get_dataloader(trainset,batch_size)
    
    # Validation tensor
    X_tensor_val = torch.tensor(d_val, requires_grad=False)
    
    # Train model
    
    train_loss_by_epoch = np.zeros(num_epochs)
    val_loss_by_epoch = np.zeros(num_epochs)
    
    start = time.perf_counter()
    for i in range(num_epochs):
        epoch_loss = []
        
        # Get a batch and do the training step
        for idx, batch in enumerate(dloader):
            
            inputs = batch
    
            # Perform training step
            loss = training_step(inputs[0],bva1,optimiser,beta)
            epoch_loss.append(loss.item())
        total_epoch_loss = np.mean(epoch_loss)
        train_loss_by_epoch[i] = total_epoch_loss
        # Calc validation loss
        z_m_val,z_l_val,rec_val = bva1(X_tensor_val.float())
        loss_val = bvae_loss_fn(X_tensor_val, rec_val, z_m_val, z_l_val).detach().numpy()
        val_loss_by_epoch[i] = loss_val
        print("Epoch",i,"Training loss",total_epoch_loss,"Validation loss",loss_val)
    end = time.perf_counter()
    print("Training time",(end-start)/60,"minutes")
    
    # Plot losses
    # plt.figure()
    # plt.plot(train_loss_by_epoch)
    # plt.figure()
    # plt.plot(val_loss_by_epoch)
    # print("argminval",np.argmin(val_loss_by_epoch))
    
    # Encoded means, logsigmas, recon
    bvae_mu_train,bvae_logsigma_train,bvae_recon_train = bva1(X_tensor.float())
    bvae_mu_train = bvae_mu_train.detach().numpy()
    bvae_logsigma_train = bvae_logsigma_train.detach().numpy()
    bvae_recon_train = bvae_recon_train.detach().numpy()
    
    bvae_mu_val,bvae_logsigma_val,bvae_recon_val = bva1(X_tensor_val.float())
    bvae_mu_val = bvae_mu_val.detach().numpy()
    bvae_logsigma_val = bvae_logsigma_val.detach().numpy()
    bvae_recon_val = bvae_recon_val.detach().numpy()
    
    # Calculate reconstruction error
    # print("Reconstruction MSE","Training",mean_squared_error(d_train,bvae_recon_train))
    # print("Reconstruction MSE","Validation",mean_squared_error(d_val,bvae_recon_val))
    
    '''
    #============================================
    # Generate synthetic points and add to training data
    #============================================
    '''
    
    # Sample synthetic data points from VAE
    
    sample_size = 100000
    
    sample_ind = np.random.choice(np.array(range(bvae_mu_train.shape[0])),
                                  size=sample_size,
                                  replace=True)
    
    # Get encoded mu and logsigma
    sample_mu = bvae_mu_train[sample_ind]
    sample_logsigma = bvae_logsigma_train[sample_ind]
    
    # Sample from the corresponding distributions
    sample_z = bva1.reparameterise(torch.tensor(sample_mu,requires_grad=False),
                                   torch.tensor(sample_logsigma,requires_grad=False))
    
    # Decode the sample values to give synthetic points
    sample_recon = bva1.decode(sample_z)
    sample_recon = sample_recon.detach().numpy()
    
    # Get labels for transparency
    sample_recon = pd.DataFrame(sample_recon,
                                columns=data_train_norm.drop(['Unnamed: 0',
                                                             'haematocrit',
                                                             'biobank_id'],axis=1).columns)
    
    # Diagnostic check
    desc_train_gendb = pd.DataFrame(d_train).describe()
    desc_sample_gendb = sample_recon.describe()
    
    # Add to training data
    # Regression - Neuroticism
    
    # Separate into data and labels
    data_train_mod_synth = sample_recon.drop(['neuroticism'],axis=1).values
    target_train_mod_synth = sample_recon['neuroticism'].values
    
    # Denormalise labels
    # Denormalised targets were used in the original models
    target_train_mod_synth = target_train_mod_synth*desc_train.loc['std','neuroticism']\
        + desc_train.loc['mean','neuroticism']
    
    # Diagnostic comparison of labels - doesn't look too bad
    print(pd.DataFrame(target_train_mod_synth).describe())
    print(data_train.loc[underperf_grp_indic_train==1,'neuroticism'].describe())
    
    # Augment training data with synthetic data
    data_train_mod_augment = np.concatenate((data_train_mod,
                                                  data_train_mod_synth),
                                                 axis=0)
    
    target_train_mod_augment = np.concatenate((target_train_mod,
                                                     target_train_mod_synth),
                                                    axis=0)
    
    '''
    #============================================
    # Train new model on augmented data
    #============================================
    '''
    
    # Train new model  
    nums = params[data_type][task_type]['nn']['nums']
    num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out = nums
    learn_rate = params['learn_rate']
    batch_size = params['batch_size']
    num_epochs = params[data_type][task_type]['nn'][target_var]['epochs']

    model_nn_gendb,pred_train_nn_gendb_augment,pred_val_nn_gendb_augment,results_nn_gendb = fit_reg_nn(num_in, num_hidden_1,
                                                          num_hidden_2, num_hidden_3, num_out,
                                                          learn_rate, batch_size, num_epochs,
                                                          data_train_mod_augment, target_train_mod_augment,
                                                          data_val_mod, target_val_mod,
                                                          metrics=True)
    
    # Predictions and RMSE (use non-augmented data)
    pred_train_nn_gendb = model_nn_gendb(torch.tensor(data_train_mod).float())\
        .detach().numpy()
    pred_val_nn_gendb = model_nn_gendb(torch.tensor(data_val_mod).float())\
        .detach().numpy()
      
    mse_train_gendb = mean_squared_error(target_train_mod, pred_train_nn_gendb)
    rmse_train_gendb = np.sqrt(mse_train_gendb)
    nrmse_train_gendb = rmse_train_gendb/np.mean(target_train_mod)
    
    mse_val_gendb = mean_squared_error(target_val_mod, pred_val_nn_gendb)
    rmse_val_gendb = np.sqrt(mse_val_gendb)
    nrmse_val_gendb = rmse_val_gendb/np.mean(target_val_mod)
    
    print("Gen-DB Training RMSE",rmse_train_gendb,"Validation RMSE",rmse_val_gendb)
    print("Gen-DB Training NRMSE",nrmse_train_gendb,"Validation NRMSE",nrmse_val_gendb)
    
    # Store
    results_gendb['training']['GenDB']['NRMSE'] = nrmse_train_gendb
    results_gendb['validation']['GenDB']['NRMSE'] = nrmse_val_gendb
    
    # Performance by underperforming group
    for j in np.unique(underperf_grp_indic_train):
        train_pct = 100*np.sum(underperf_grp_indic_train)/underperf_grp_indic_train.shape[0]
        val_pct = 100*np.sum(underperf_grp_indic_val)/underperf_grp_indic_val.shape[0]
        rmse_train = np.sqrt(mean_squared_error(target_train_mod[underperf_grp_indic_train==j],
                                                pred_train_nn_gendb[underperf_grp_indic_train==j]))
        nrmse_train = rmse_train/np.mean(target_train_mod[underperf_grp_indic_train==j])
        rmse_val = np.sqrt(mean_squared_error(target_val_mod[underperf_grp_indic_val==j],
                                                pred_val_nn_gendb[underperf_grp_indic_val==j]))
        nrmse_val = rmse_val/np.mean(target_val_mod[underperf_grp_indic_val==j])
        
        # Store
        results_gendb['training']['GenDB']['Underperf'][str(j)] = {}
        results_gendb['training']['GenDB']['Underperf'][str(j)]['NRMSE'] = nrmse_train
        results_gendb['validation']['GenDB']['Underperf'][str(j)] = {}
        results_gendb['validation']['GenDB']['Underperf'][str(j)]['NRMSE'] = nrmse_val
        
        print("Gen-DB Neuroticism prediction by underperf grp",j,"RMSE","Training",rmse_train,
              "Validation",rmse_val)
        print("Gen-DB Neuroticism prediction by underperf grp",j,"NRMSE","Training",nrmse_train,
              "Validation",nrmse_val)
    print("Underperf grp pct of data","Training",train_pct,"%","Validation",val_pct,"%")
    
    # Calculate rebalanced Gini coefficient
    rmse_train_component_gendb = np.zeros(num_components)
    for i in range(num_components):
        rmse_comp = np.sqrt(mean_squared_error(target_train_mod[pred_gmm_train==i],
                                               pred_train_nn_gendb[pred_gmm_train==i]))
        rmse_train_component_gendb[i] = rmse_comp
        
    rmse_val_component_gendb = np.zeros(num_components)
    for i in range(num_components):
        rmse_comp = np.sqrt(mean_squared_error(target_val_mod[pred_gmm_val==i],
                                               pred_val_nn_gendb[pred_gmm_val==i]))
        rmse_val_component_gendb[i] = rmse_comp
       
    gini_train = calc_gini(rmse_train_component,count_gmm_train_component)
    gini_val = calc_gini(rmse_val_component,count_gmm_val_component)
    gini_train_gendb = calc_gini(rmse_train_component_gendb,count_gmm_train_component)
    gini_val_gendb = calc_gini(rmse_val_component_gendb,count_gmm_val_component)
    
    print("Training Gini Gen-DB Before",gini_train,\
          "After",gini_train_gendb)
    
    print("Validation Gini Gen-DB Before",gini_val,\
          "After",gini_val_gendb)
    
    # Store results        
    results_gendb['training']['Original']['gini'] = gini_train
    results_gendb['training']['GenDB']['gini'] = gini_train_gendb
    results_gendb['validation']['Original']['gini'] = gini_val
    results_gendb['validation']['GenDB']['gini'] = gini_val_gendb
        
    '''
    #============================================
    # Store results
    #============================================
    '''
    
    trial_ind = str(trial)
    
    trials_gendb[trial_ind] = results_gendb
    
# Save results
np.save(outputdir+'/trials_gendb.npy',trials_gendb)

# Load results
# res2 = np.load(outputdir+'/trials_gendb.npy',allow_pickle=True).item()

'''
#============================================
# Present results
#============================================
'''

# Make dataframes to hold output
results_box_gendb = pd.DataFrame(np.zeros(12*num_trials),
                                 index=pd.MultiIndex.from_product([['Training','Validation'],
                                                                   ['All','Base','Underpriv'],
                                                                   ['Original','GenDB'],
                                                                   [*range(num_trials)]],
                                                                  names=['Data','Group',
                                                                         'Debiasing','Trial']),
                                 columns=['nrmse'])

results_summary_gendb = pd.DataFrame(np.zeros(12),
                                   index=pd.MultiIndex.from_product([['Training','Validation'],
                                                                     ['Mean','Stdev'],
                                                                     ['All','Base','Underpriv']],
                                                                    names=['Data','Metric','Group']),
                                   columns=['value'])

gini_box_gendb = pd.DataFrame(np.zeros(4*num_trials),
                             index=pd.MultiIndex.from_product([['Training','Validation'],
                                                               ['Original','GenDB'],
                                                               [*range(num_trials)]],
                                                              names=['Data','Debiasing','Trial']),
                             columns=['gini'])
    
results_gini_gendb = pd.DataFrame(np.zeros(12),
                                index = pd.MultiIndex.from_product([['Training','Validation'],
                                                                    ['Mean','Stdev'],
                                                                    ['Original','GenDB','Change']],
                                                                   names=['Data','Metric','Debiasing']),
                                columns=['value'])

# Populate boxplot dataframe
for i in range(num_trials):
        results_box_gendb.loc[('Training','All','Original',i),'nrmse'] = trials_gendb[str(i)]['training']['Original']['NRMSE']
        results_box_gendb.loc[('Training','All','GenDB',i),'nrmse'] = trials_gendb[str(i)]['training']['GenDB']['NRMSE']
        results_box_gendb.loc[('Training','Base','Original',i),'nrmse'] = trials_gendb[str(i)]['training']['Original']['Underperf']['0.0']['NRMSE']
        results_box_gendb.loc[('Training','Base','GenDB',i),'nrmse'] = trials_gendb[str(i)]['training']['GenDB']['Underperf']['0.0']['NRMSE']
        results_box_gendb.loc[('Training','Underpriv','Original',i),'nrmse'] = trials_gendb[str(i)]['training']['Original']['Underperf']['1.0']['NRMSE']
        results_box_gendb.loc[('Training','Underpriv','GenDB',i),'nrmse'] = trials_gendb[str(i)]['training']['GenDB']['Underperf']['1.0']['NRMSE']
        results_box_gendb.loc[('Validation','All','Original',i),'nrmse'] = trials_gendb[str(i)]['validation']['Original']['NRMSE']
        results_box_gendb.loc[('Validation','All','GenDB',i),'nrmse'] = trials_gendb[str(i)]['validation']['GenDB']['NRMSE']
        results_box_gendb.loc[('Validation','Base','Original',i),'nrmse'] = trials_gendb[str(i)]['validation']['Original']['Underperf']['0.0']['NRMSE']
        results_box_gendb.loc[('Validation','Base','GenDB',i),'nrmse'] = trials_gendb[str(i)]['validation']['GenDB']['Underperf']['0.0']['NRMSE']
        results_box_gendb.loc[('Validation','Underpriv','Original',i),'nrmse'] = trials_gendb[str(i)]['validation']['Original']['Underperf']['1.0']['NRMSE']
        results_box_gendb.loc[('Validation','Underpriv','GenDB',i),'nrmse'] = trials_gendb[str(i)]['validation']['GenDB']['Underperf']['1.0']['NRMSE']
        
# Populate summary dataframe
results_summary_gendb.loc[('Training','Mean','All'),'value'] = np.mean(results_box_gendb.loc[('Training','All','Original')]-results_box_gendb.loc[('Training','All','GenDB')]).values
results_summary_gendb.loc[('Training','Mean','Base'),'value'] = np.mean(results_box_gendb.loc[('Training','Base','Original')]-results_box_gendb.loc[('Training','Base','GenDB')]).values
results_summary_gendb.loc[('Training','Mean','Underpriv'),'value'] = np.mean(results_box_gendb.loc[('Training','Underpriv','Original')]-results_box_gendb.loc[('Training','Underpriv','GenDB')]).values
results_summary_gendb.loc[('Training','Stdev','All'),'value'] = np.std(results_box_gendb.loc[('Training','All','Original')]-results_box_gendb.loc[('Training','All','GenDB')]).values
results_summary_gendb.loc[('Training','Stdev','Base'),'value'] = np.std(results_box_gendb.loc[('Training','Base','Original')]-results_box_gendb.loc[('Training','Base','GenDB')]).values
results_summary_gendb.loc[('Training','Stdev','Underpriv'),'value'] = np.std(results_box_gendb.loc[('Training','Underpriv','Original')]-results_box_gendb.loc[('Training','Underpriv','GenDB')]).values
results_summary_gendb.loc[('Validation','Mean','All'),'value'] = np.mean(results_box_gendb.loc[('Validation','All','Original')]-results_box_gendb.loc[('Validation','All','GenDB')]).values
results_summary_gendb.loc[('Validation','Mean','Base'),'value'] = np.mean(results_box_gendb.loc[('Validation','Base','Original')]-results_box_gendb.loc[('Validation','Base','GenDB')]).values
results_summary_gendb.loc[('Validation','Mean','Underpriv'),'value'] = np.mean(results_box_gendb.loc[('Validation','Underpriv','Original')]-results_box_gendb.loc[('Validation','Underpriv','GenDB')]).values
results_summary_gendb.loc[('Validation','Stdev','All'),'value'] = np.std(results_box_gendb.loc[('Validation','All','Original')]-results_box_gendb.loc[('Validation','All','GenDB')]).values
results_summary_gendb.loc[('Validation','Stdev','Base'),'value'] = np.std(results_box_gendb.loc[('Validation','Base','Original')]-results_box_gendb.loc[('Validation','Base','GenDB')]).values
results_summary_gendb.loc[('Validation','Stdev','Underpriv'),'value'] = np.std(results_box_gendb.loc[('Validation','Underpriv','Original')]-results_box_gendb.loc[('Validation','Underpriv','GenDB')]).values

# Populate Gini dataframes
for i in range(num_trials):
    gini_box_gendb.loc[('Training','Original',i),'gini'] = trials_gendb[str(i)]['training']['Original']['gini']
    gini_box_gendb.loc[('Training','GenDB',i),'gini'] = trials_gendb[str(i)]['training']['GenDB']['gini']
    gini_box_gendb.loc[('Validation','Original',i),'gini'] = trials_gendb[str(i)]['validation']['Original']['gini']
    gini_box_gendb.loc[('Validation','GenDB',i),'gini'] = trials_gendb[str(i)]['validation']['GenDB']['gini']


results_gini_gendb.loc[('Training','Mean','Original'),'value'] = np.mean(gini_box_gendb.loc[('Training','Original')]).values
results_gini_gendb.loc[('Training','Mean','GenDB'),'value'] = np.mean(gini_box_gendb.loc[('Training','GenDB')]).values
results_gini_gendb.loc[('Training','Mean','Change'),'value'] = np.mean(gini_box_gendb.loc[('Training','Original')]-gini_box_gendb.loc[('Training','GenDB')]).values
results_gini_gendb.loc[('Training','Stdev','Original'),'value'] = np.std(gini_box_gendb.loc[('Training','Original')]).values
results_gini_gendb.loc[('Training','Stdev','GenDB'),'value'] = np.std(gini_box_gendb.loc[('Training','GenDB')]).values
results_gini_gendb.loc[('Training','Stdev','Change'),'value'] = np.std(gini_box_gendb.loc[('Training','Original')]-gini_box_gendb.loc[('Training','GenDB')]).values
results_gini_gendb.loc[('Validation','Mean','Original'),'value'] = np.mean(gini_box_gendb.loc[('Validation','Original')]).values
results_gini_gendb.loc[('Validation','Mean','GenDB'),'value'] = np.mean(gini_box_gendb.loc[('Validation','GenDB')]).values
results_gini_gendb.loc[('Validation','Mean','Change'),'value'] = np.mean(gini_box_gendb.loc[('Validation','Original')]-gini_box_gendb.loc[('Validation','GenDB')]).values
results_gini_gendb.loc[('Validation','Stdev','Original'),'value'] = np.std(gini_box_gendb.loc[('Validation','Original')]).values
results_gini_gendb.loc[('Validation','Stdev','GenDB'),'value'] = np.std(gini_box_gendb.loc[('Validation','GenDB')]).values
results_gini_gendb.loc[('Validation','Stdev','Change'),'value'] = np.std(gini_box_gendb.loc[('Validation','Original')]-gini_box_gendb.loc[('Validation','GenDB')]).values
    
# Plot variation in experiments
results_box_gendb_data = results_box_gendb.reset_index()
plt.figure()
sns.factorplot(data=results_box_gendb_data,col='Data',x='Group',y='nrmse',kind='box',hue='Debiasing',palette='seismic')
plt.savefig(plotsdir+'/GenDB variation box.png',
            format='png',dpi=1200,bbox_inches="tight")

# Combine plot with rebalancing (requires data saved from main experiments)

# Import data from rebalancing
curr_dir = 'Output/neuroticism_reg_neur_-2021-08-21-11-10/'
plots_main_dir  = 'Plots/test/overflow'
trials = np.load(curr_dir+'/trials.npy',allow_pickle=True).item()

# Reset experiment type to match loaded trials
data_type = trials['data_type']
model_type = trials['model_type']
task_type = trials['task_type']
target_var = trials['target_var']

metric = 'nrmse'

# Calculate and plot results
results_box,results_summary,gini_box,results_gini = plot_results_box(num_trials,metric,trials,plots_main_dir,
                                                            data_type,task_type,model_type,target_var)

# Concatenate frames for three-way plot
# Filter out unnecessary set of original results
filter1 = results_box_gendb.index.get_level_values('Debiasing') == 'GenDB'
tmp = results_box_gendb.loc[filter1]
results_comb = pd.concat([results_box,tmp])
results_comb_data = results_comb.reset_index()
plt.figure()
sns.factorplot(data=results_comb_data,col='Data',x='Group',y='nrmse',kind='box',hue='Debiasing',palette='seismic')
plt.savefig(plotsdir+'/threeway GenDB variation box.png',
            format='png',dpi=1200,bbox_inches="tight")
