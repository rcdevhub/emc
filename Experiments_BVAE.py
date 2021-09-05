# -*- coding: utf-8 -*-
"""
Ethical Model Calibration

Additional Experiments - Beta Variational Autoencoder (BVAE)

Created on Thu Sep  2 15:49:53 2021

@author: rcpc4
"""

# Requires preprocessed data from main
# Dataset: diabetes
# Task type: reg
# Target variable: hba1c

# Imports
from Functions_BVAE import *

# Set plotsdir
plotsdir = plots_main_dir

# Train the BVAE

d_train = data_train_auto
d_val = data_val_auto

# Hyperparameters

num_input = 22
num_hidden_1 = 8
num_hidden_2 = 4
num_latent = 2
num_output = 22

batch_size = 10
learning_rate = 0.001
beta = 2 

num_epochs = 5

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
    loss_val = bvae_loss_fn(X_tensor_val, rec_val, z_m_val, z_l_val, beta).detach().numpy()
    val_loss_by_epoch[i] = loss_val
    print("Epoch",i,"Training loss",total_epoch_loss,"Validation loss",loss_val)
end = time.perf_counter()
print("Training time",(end-start)/60,"minutes")

# Plot losses
plt.figure()
plt.plot(train_loss_by_epoch)
plt.figure()
plt.plot(val_loss_by_epoch)

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
print("Reconstruction MSE","Training",mean_squared_error(d_train,bvae_recon_train))
print("Reconstruction MSE","Validation",mean_squared_error(d_val,bvae_recon_val))

# Save BVAE
# Model
torch.save(bva1,outputdir+"/model_bva1"+timestamp)
# Parameters
torch.save(bva1.state_dict(),outputdir+"/params_bva11"+timestamp)
# Latent space
np.save(outputdir+"/bva1_latent_mu_train"+timestamp+".npy",bvae_mu_train)
np.save(outputdir+"/bva1_latent_logsigma_train"+timestamp+".npy",bvae_logsigma_train)
np.save(outputdir+"/bva1_latent_mu_val"+timestamp+".npy",bvae_mu_val)
np.save(outputdir+"/bva1_latent_logsigma_val"+timestamp+".npy",bvae_logsigma_val)

# Plot example reconstructions

plt.figure()
plt.imshow(d_train[20500:20600])
plt.figure()
plt.imshow(bvae_recon_train[20500:20600])

plt.figure()
sns.heatmap(d_train[30000:30100])
plt.xlabel('Variable')
plt.ylabel('Sample')
plt.savefig(plotsdir+'/bvae'+timestamp+' train data orig.png',format='png',dpi=1200,bbox_inches='tight')
plt.figure()
sns.heatmap(bvae_recon_train[30000:30100])
plt.xlabel('Variable')
plt.ylabel('Sample')
plt.savefig(plotsdir+'/bvae'+timestamp+' train data recon.png',format='png',dpi=1200,bbox_inches='tight')

for i in range(10):
    j = np.random.randint(20000)
    print(j)
    plt.figure()
    plt.plot(d_train[j])
    plt.plot(bvae_recon_train[j])
    plt.xlabel('Variable')
    plt.ylabel('Value (normalised)')
    plt.legend(handles=[mpatches.Patch(color='#1f77b4',label="Original"),
                    mpatches.Patch(color='#ff7f0e',label="Reconstructed")])
    plt.savefig(plotsdir+'/bvae'+timestamp+' recon ex'+str(i)+'.png',format='png',dpi=1200,bbox_inches='tight')

# Scatterplot of means
# Not saved as use mid versions below
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1])
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1],c=data_train['age'])
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1],c=data_train['sex'])
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1],c=data_train['smoking'])
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1],c=data_train['diabetes'])
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1],c=data_train['high_blood_pressure'])
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1],c=data_train['townsend_deprivation_index'])
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1],c=data_train['weight'])
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1],c=data_train['bmi'])
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1],c=data_train['body_fat'])
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1],c=data_train['Hb'])
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1],c=data_train['hba1c'])
plt.scatter(bvae_mu_train[:,0],bvae_mu_train[:,1],c=data_train['ethnic_white'])

# Density plot of means
plt.hist2d(x=bvae_mu_train[:,0],y=bvae_mu_train[:,1],bins=150,cmap=plt.cm.jet)
plt.colorbar()
plt.savefig(plotsdir+'/bvae'+timestamp+' lat dens overall.png',format='png',dpi=1200,bbox_inches='tight')

# Below plots the densest part of the data, to avoid being distracted by
# non-dense parts above.
# Plot middle %iles of latent range
low_pcl = 5
high_pcl = 95

x_low = np.percentile(bvae_mu_train[:,0],low_pcl)
x_high = np.percentile(bvae_mu_train[:,0],high_pcl)
y_low = np.percentile(bvae_mu_train[:,1],low_pcl)
y_high = np.percentile(bvae_mu_train[:,1],high_pcl)

# Find points in this range on both axes
x_mid = (bvae_mu_train[:,0] > x_low) & (bvae_mu_train[:,0] < x_high)
y_mid = (bvae_mu_train[:,1] > y_low) & (bvae_mu_train[:,1] < y_high)
x_y_mid = (x_mid) & (y_mid)
print("Coverage",100*np.sum(x_y_mid)/bvae_mu_train.shape[0],"% of data")

# Density plot of middle of latent space
plt.figure()
plt.hist2d(x=bvae_mu_train[x_y_mid,0],y=bvae_mu_train[x_y_mid,1],bins=25,cmap=plt.cm.jet)
plt.colorbar()
plt.savefig(plotsdir+'/bvae'+timestamp+' lat dens mid overall.png',format='png',dpi=1200,bbox_inches='tight')

# Plot in that region only
# All
plt.scatter(bvae_mu_train[x_y_mid,0],bvae_mu_train[x_y_mid,1],marker='+',s=10)
plt.savefig(plotsdir+'/bvae'+timestamp+' mid vars all.png',format='png',dpi=1200,bbox_inches="tight")
# Individal variables
# plt.scatter(bvae_mu_train[x_y_mid,0],bvae_mu_train[x_y_mid,1],marker='+',s=10,c=data_train.loc[x_y_mid,'sex'])

plot_vars = ['sex', 'age', 'smoking', 'diabetes', 
              'high_blood_pressure', 'townsend_deprivation_index',
              'weight', 'bmi', 'body_fat',
              'Hb', 'hba1c', 'ethnic_white']

for i,var in enumerate(plot_vars):
    fig = plt.figure()
    plt.scatter(bvae_mu_train[x_y_mid,0],bvae_mu_train[x_y_mid,1],marker='+',s=10,c=data_train.loc[x_y_mid,var])
    plt.xlabel(var)
    plt.colorbar()
    plt.savefig(plotsdir+'/bvae'+timestamp+' mid vars '+var+'.png',format='png',dpi=1200,bbox_inches="tight")


