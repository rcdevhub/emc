# -*- coding: utf-8 -*-
"""
Ethical Model Calibration

Additional Experiments - Debiasing Variational Autoencoder (DB-VAE)

Created on Thu Sep  2 14:52:29 2021

@author: rcpc4
"""

# Based on work by Amini and Soleimany https://github.com/aamini/

# Requires preprocessed data from main
# Dataset: neuroticism
# Task type: cls
# Target variable: sex

# Imports
from Functions_BVAE import *

# Train the DBVAE

# Datasets
db_train = data_train_mod
lb_train = target_train_mod

db_val = data_val_mod
lb_val = target_val_mod

# Hyperparameters

num_input = 28        
num_hidden_1 = 8
num_hidden_2 = 4
num_latent = 2
num_output = 28

kl_weight = 0.5 # regularisation of VAE loss (orig value = 0.005)
smoothing_fac = 0.001 # Inversely proportional to resampling debiasing (orig value = 0.001)

batch_size = 10
learning_rate = 5e-4

num_epochs = 50

# Initialise model
dbv1 = dbvae(num_input,num_hidden_1,num_hidden_2,num_latent,num_output) 

optimiser = torch.optim.Adam(dbv1.parameters(),lr=learning_rate) 

# Define training set
X_tensor = torch.tensor(db_train, requires_grad=False)
Y_tensor = torch.tensor(lb_train, requires_grad=False)
trainset = torch.utils.data.TensorDataset(X_tensor,Y_tensor)

# Train model

train_loss_by_epoch = np.zeros(num_epochs)
val_loss_by_epoch = np.zeros(num_epochs)

for i in range(num_epochs):
    epoch_loss = []
    
    # Recompute sampling probabilities at start of each epoch (adaptive reweighting)
    samp_prob = get_training_sample_probabilities(X_tensor.float(), dbv1, num_latent,
                                                  smoothing_fac=smoothing_fac)
    dloader = get_dataloader(trainset, batch_size, wgt_sampling=True, sampling_weights=samp_prob)
    
    # Get a batch and do the training step
    for idx, batch in enumerate(dloader):
        
        inputs,labels = batch
        
        # Perform training step
        loss = debiasing_train_step(inputs,labels,kl_weight,dbv1,optimiser)
        epoch_loss.append(loss.item())
    total_epoch_loss = np.mean(epoch_loss)
    print("Epoch",i,"Training loss",total_epoch_loss)

# Predictions
# Training
dbvae_train_logits = dbv1.predict(X_tensor.float())
dbvae_train_probs = torch.sigmoid(dbvae_train_logits)
dbvae_train_pred = np.round(dbvae_train_probs.detach().numpy(),0)
# Validation
dbvae_val_logits = dbv1.predict(torch.tensor(db_val,requires_grad=False).float())
dbvae_val_probs = torch.sigmoid(dbvae_val_logits)
dbvae_val_pred = np.round(dbvae_val_probs.detach().numpy(),0)

# Classification report
print("Training - DB-VAE\n",confusion_matrix(lb_train,dbvae_train_pred))
print("Training",classification_report(lb_train,dbvae_train_pred))
print("Validation - DB-VAE\n",confusion_matrix(lb_val,dbvae_val_pred))
print("Validation",classification_report(lb_val,dbvae_val_pred))

# Save dbv autoencoder
# Model
torch.save(dbv1,outputdir+"/model_dbv1"+timestamp)
# Parameters (recommended method)
torch.save(dbv1.state_dict(),outputdir+"/params_dbv1"+timestamp)
# Latent space
np.save(outputdir+"/dbvae_mu_train"+timestamp+".npy",dbvae_mu_train)
np.save(outputdir+"/dbvae_mu_val"+timestamp+".npy",dbvae_mu_val)
# Logits
np.save(outputdir+"/dbvae_train_logits_cls_sex"+timestamp+".npy",dbvae_train_logits.detach().numpy())
np.save(outputdir+"/dbvae_val_logits_cls_sex"+timestamp+".npy",dbvae_val_logits.detach().numpy())

# Bias check
for j in np.unique(data_train['Hb_bin']):
    print('Hb',j,
          '\nConfusion matrix for classification of sex - Training - DB-VAE',
          '\n',
          confusion_matrix(lb_train[data_train['Hb_bin']==j],
                           dbvae_train_pred[data_train['Hb_bin']==j]),
          '\n',
          classification_report(lb_train[data_train['Hb_bin']==j],
                      dbvae_train_pred[data_train['Hb_bin']==j]))
    
for j in np.unique(data_val['Hb_bin']):
    print('Hb',j,
          '\nConfusion matrix for classification of sex - Validation - DB-VAE',
          '\n',
          confusion_matrix(lb_val[data_val['Hb_bin']==j],
                           dbvae_val_pred[data_val['Hb_bin']==j]),
          '\n',
          classification_report(lb_val[data_val['Hb_bin']==j],
                      dbvae_val_pred[data_val['Hb_bin']==j]))

# Get latent space (means)
dbvae_mu_train = get_latent_mu(X_tensor.float(), dbv1).detach().numpy()
dbvae_mu_val = get_latent_mu(torch.tensor(db_val,requires_grad=False).float(), dbv1).detach().numpy()

# Plot latent space
plt.scatter(dbvae_mu_train[:,0],dbvae_mu_train[:,1])
plt.scatter(dbvae_mu_train[:,0],dbvae_mu_train[:,1],c=data_train["age"])
plt.scatter(dbvae_mu_train[:,0],dbvae_mu_train[:,1],c=data_train["sex"])
plt.scatter(dbvae_mu_train[:,0],dbvae_mu_train[:,1],c=data_train["smoking"])
plt.scatter(dbvae_mu_train[:,0],dbvae_mu_train[:,1],c=data_train["reaction_time"])
plt.scatter(dbvae_mu_train[:,0],dbvae_mu_train[:,1],c=data_train["neuroticism"])
plt.scatter(dbvae_mu_train[:,0],dbvae_mu_train[:,1],c=data_train["heart_attack"])
plt.scatter(dbvae_mu_train[:,0],dbvae_mu_train[:,1],c=data_train["Hb"])

# Plot latent space by point density
plt.hist2d(dbvae_mu_train[:,0],dbvae_mu_train[:,1],bins=150,cmap=plt.cm.jet)
plt.colorbar()
plt.title("DB-VAE latent density")

# Plot latent space by sampling probability
plt.scatter(dbvae_mu_train[:,0],dbvae_mu_train[:,1],c=samp_prob,marker='+',s=20)
plt.colorbar()

# Diagnostic checks
# Sampling probabilities
# Made table because doesn't histogram well due to extreme values
samp_prob_uniq,samp_prob_uniq_cnt = np.unique(samp_prob,return_counts=True)
samp_prob_uniq_cnt_pct = samp_prob_uniq_cnt/np.sum(samp_prob_uniq_cnt)
samp_prob_uniq_mult = 1/samp_prob_uniq
samp_prob_uniq_rel = samp_prob_uniq/samp_prob_uniq[0]
samp_prob_exp_cnt_pct = samp_prob_uniq*samp_prob_uniq_cnt
samp_prob_exp_cnt = np.sum(samp_prob_uniq_cnt)*samp_prob_exp_cnt_pct
samp_prob_tab = np.concatenate((np.expand_dims(samp_prob_uniq,axis=1),
                               np.expand_dims(samp_prob_uniq_cnt,axis=1),
                               np.expand_dims(samp_prob_uniq_cnt_pct,axis=1),
                               np.expand_dims(samp_prob_uniq_mult,axis=1),
                               np.expand_dims(samp_prob_uniq_rel,axis=1),
                               np.expand_dims(samp_prob_exp_cnt_pct,axis=1),
                               np.expand_dims(samp_prob_exp_cnt,axis=1)),
                               axis=1)
samp_prob_tab = pd.DataFrame(samp_prob_tab)
samp_prob_tab.columns = ['Samp prob','count','count pct',
                         'Equiv 1 in','Relative prob',
                         'Expected pct','Expected count']
