# -*- coding: utf-8 -*-
"""
Ethical Model Calibration

Functions for Beta Variational Autoencoder (BVAE)
and for Debiasing Variational Autoencoder (DB-VAE)

Created on Thu Sep  2 12:26:04 2021

@author: rcpc4
"""

'''
#============================================
# Setup
#============================================
'''

import numpy as np
import torch
from torch import nn

'''
#============================================
# Shared Functions
#============================================
'''

def sampling(z_mean,z_logsigma):
    '''
    Sample from the latent Gaussians.

    Parameters
    ----------
    z_mean : tensor, mean values of latent distributions
    z_logsigma : tensor, log covariance matrix of latent distributions

    Returns
    -------
    z : tensor, value sampled from latent Gaussian

    '''
    
    batch, latent_dim = z_mean.shape
    
    # Use standard normal trick
    epsilon = torch.normal(mean=torch.tensor(np.zeros((batch,latent_dim)),requires_grad=False),
                           std=torch.tensor(np.ones((batch,latent_dim)),requires_grad=False))
    # Compute reparameterisation
    z = z_mean + torch.exp(0.5 * z_logsigma) * epsilon
    
    return z

def get_dataloader(pytorch_dataset,batch_size,wgt_sampling=False,sampling_weights=None):
    '''
    Define weighted data loader.

    Parameters
    ----------
    pytorch_dataset : PyTorch dataset, PyTorch training dataset containing data and labels
    batch_size : scalar, batch size for training
    wgt_sampling : boolean, optional, default False, use weights when sampling training points
    sampling_weights : array,optional, default None, weights for sampling training points

    Returns
    -------
    weighted_loader : PyTorch dataloader, optionally weighted

    '''
    
    # Optionally use weighted random sampler
    if wgt_sampling:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            sampling_weights,
            num_samples = len(sampling_weights),
            replacement=True)
        
    else:
        sampler = None
        
    weighted_loader = torch.utils.data.DataLoader(
        pytorch_dataset,
        batch_size = batch_size,
        sampler = sampler)
    
    return weighted_loader

'''
#============================================
# BVAE Functions
#============================================
'''

def bvae_loss_fn(x, x_recon, mu, logsigma, beta):
    '''
    Calculate BVAE loss.
    
    Parameters
    ----------
    x : tensor, training batch input
    x_recon : tensor, training batch model output (reconstruction)
    mu : tensor, mean values of (encoded) latent distributions for input
    logsigma : tensor, log covariance of latent distributions (diagonal covar matrix, no covars between dists)
    beta : scalar, coefficient determining degree of regularisation

    Returns
    -------
    bvae_loss : tensor, total loss of the batch data points
    
    '''
    # Define latent loss
    latent_loss = 0.5 * torch.sum(torch.exp(logsigma)+torch.square(mu) - 1 - logsigma, axis=1)
    # Define reconstruction loss
    reconstruction_loss = torch.mean(torch.abs(x - x_recon), axis=1)
    # Define BVAE loss
    bvae_loss = beta * latent_loss + reconstruction_loss
    
    bvae_loss = torch.sum(bvae_loss)
        
    return bvae_loss

class bvae(nn.Module):
    '''Define the Beta Variational Autoencoder (BVAE) module.'''
    def __init__(self,num_in,num_hidden_1,num_hidden_2,num_latent,num_out):
        '''
        Beta Variational Autoencoder Architecture
        
        Allowing specification of the number of neurons in each layer.
        '''
        
        super(bvae,self).__init__()
        
        self.latent_dim = num_latent
        
        num_encoder_dims = 2*self.latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(num_in,num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1,num_hidden_2),
            nn.ReLU(),
            nn.Linear(num_hidden_2,num_encoder_dims))
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim,num_hidden_2),
            nn.ReLU(),
            nn.Linear(num_hidden_2,num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1,num_out))
        
    def encode(self,x):
        ''' Encode input to latent distributions.'''
        encoder_output = self.encoder(x)
        
        # Latent variable distribution parameters
        z_mean = encoder_output[:,:self.latent_dim]
        z_logsigma = encoder_output[:,self.latent_dim:]
        
        return z_mean, z_logsigma
    
    def reparameterise(self,z_mean,z_logsigma):
        '''
        Sample from latent distributions (uses separate function).
        
        Parameters
        ----------
        z_mean : tensor, mean values of latent distributions
        z_logsigma : tensor, log covariance matrix of latent distributions
    
        Returns
        -------
        z : tensor, values sampled from latent Gaussians
            
        '''
    
        z = sampling(z_mean,z_logsigma)
        
        return z
    
    def decode(self,z):
        ''' Decode from latent samples to output. '''        
        reconstruction = self.decoder(z.float())
        
        return reconstruction
    
    def forward(self,x):
        ''' Encode, sample from latent distributions and decode. '''
        # Encode input to latent distributions
        z_mean, z_logsigma = self.encode(x)
        
        z = self.reparameterise(z_mean, z_logsigma)
        
        recon = self.decode(z)
        
        return z_mean, z_logsigma, recon

def training_step(x,model,optimiser,beta):
    '''
    Perform a gradient descent step.
    
    Used in BVAE training
    
    Parameters
    ----------
    x : tensor, training batch input
    model : PyTorch BVAE module, model
    optimiser : PyTorch optimiser
    beta : scalar, coefficient determining degree of regularisation

    Returns
    -------
    bvae_loss : tensor, total loss of the batch data points
    
    '''
    
    # Forward pass
    z_mean, z_logsigma,x_recon = model(x.float())
    # Calculate loss
    loss = bvae_loss_fn(x, x_recon, z_mean, z_logsigma, beta)
    # Reset gradients
    optimiser.zero_grad()
    # Calculate gradients
    loss.backward()
    # Take gradient descent step
    optimiser.step()
    
    return loss

'''
#============================================
# DB-VAE Functions
#============================================
'''

def vae_loss_fn(x, x_recon, mu, logsigma, kl_weight):
    '''
    Calculate VAE loss.
    
    Forms part of DB-VAE loss
    
    Parameters
    ---------
    x : tensor, training batch input
    x_recon : tensor, training batch model output (reconstruction)
    mu : tensor, mean values of latent distributions
    logsigma : tensor, log covariance of latent distributions (diagonal covar matrix, no covars between dists)
    kl_weight : scalar, weights importance of regularisation (latent_loss)
    
    Returns
    -------
    vae_loss, tensor, total VAE loss for the batch
    '''
    
    # Define latent loss
    latent_loss = 0.5 * torch.sum(torch.exp(logsigma) + torch.square(mu) - 1 - logsigma, axis=1)
    # Define reconstruction loss
    reconstruction_loss = torch.mean(torch.abs(x-x_recon), axis=1)
    # Define VAE loss
    vae_loss = kl_weight * latent_loss + reconstruction_loss
        
    return vae_loss

def debiasing_loss_fn(x, x_pred, y, y_logit, mu, logsigma, kl_weight):
    '''
    Calculate total loss for the DB-VAE.
    
    Loss can vary depending on whether input is in the positive class or not, 
    if yes then total loss is sum of losses, if not then total loss is just 
    classification loss

    Parameters
    ----------
    x : tensor, training batch input
    x_pred : tensor, training batch model output (reconstruction)
    y : tensor, training batch input labels
    y_logit : tensor, model output logits for classification of training batch
    mu : tensor, mean values of (encoded) latent distributions for input
    logsigma : tensor, log covariance of latent distributions (diagonal covar matrix, no covars between dists)

    Returns
    -------
    total_loss : tensor, (scalar) total DB-VAE loss (average across batch)
    classification_loss :  tensor, per-datapoint classification loss (unused)

    '''
    # Call VAE loss
    vae_loss = vae_loss_fn(x, x_pred, mu, logsigma, kl_weight)

    # Define classification loss
    cls_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    classification_loss = cls_loss_fn(y_logit,y.float())

    # Define positive class indicator
    pos_indic = torch.eq(y,torch.tensor(1)).float()

    # Define total DB-VAE loss
    total_loss = torch.mean(classification_loss + pos_indic*vae_loss)
    
    return total_loss, classification_loss

class dbvae(nn.Module):
    '''Define the Debiasing Variational Autoencoder (DB-VAE) module.'''
    def __init__(self,num_in,num_hidden_1,num_hidden_2,num_latent,num_out):
        '''
        Beta Variational Autoencoder Architecture
        
        Allowing specification of the number of neurons in each layer.
        '''
        
        super(dbvae,self).__init__()
        
        self.latent_dim = num_latent
        
        num_encoder_dims = 2*self.latent_dim + 1
                
        self.encoder = nn.Sequential(
            nn.Linear(num_in,num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1,num_hidden_2),
            nn.ReLU(),
            nn.Linear(num_hidden_2,num_encoder_dims))
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim,num_hidden_2),
            nn.ReLU(),
            nn.Linear(num_hidden_2,num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1,num_out))
    
    def encode(self,x):
        ''' Encode input to latent distributions and also classify points.'''
        encoder_output = self.encoder(x)
        
        # Classification prediction
        y_logit = encoder_output[:,0]
        # Latent variable distribution parameters
        z_mean = encoder_output[:,1:self.latent_dim+1]
        z_logsigma = encoder_output[:,self.latent_dim+1:]
        
        return y_logit, z_mean, z_logsigma
    
    def reparameterise(self,z_mean,z_logsigma):
        '''
        Sample from latent distributions (uses separate function)
        
        Parameters
        ----------
        z_mean : tensor, mean values of latent distributions
        z_logsigma : tensor, log covariance matrix of latent distributions
    
        Returns
        -------
        z : tensor, values sampled from latent Gaussians
        
        '''
        z = sampling(z_mean,z_logsigma)
        
        return z
    
    def decode(self,z):
        ''' Decode from latent sample to output. '''
        reconstruction = self.decoder(z.float())
        
        return reconstruction
    
    def forward(self,x):
        ''' Encode, classify, sample from latent distributions and decode. '''
        # Encode input and latent distributions and classify
        y_logit, z_mean, z_logsigma = self.encode(x)
        
        z = self.reparameterise(z_mean,z_logsigma)

        recon = self.decode(z)

        return y_logit, z_mean, z_logsigma, recon
    
    def predict(self, x):
        ''' Encode but return classification logits only. '''
        y_logit, z_mean, z_logsigma = self.encode(x)
        
        return y_logit
        

def get_latent_mu(inputs, dbvae):
    ''' Encodes a set of inputs and returns the latent means only.

    Used for reweighting step

    Parameters
    ----------
    inputs : tensor, training input data
    dbvae : PyTorch dbvae module, model

    Returns
    -------
    mu : tensor, mean values of latent distributions
    
    '''
        
    _, mu, _ = dbvae.encode(inputs)
        
    return mu

def get_training_sample_probabilities(inputs, dbvae, latent_dim, smoothing_fac, bins=10):    
    '''
    Compute sampling weights for adaptive weighted sampling (debiasing).
    
    Goal is to progressively samples rarer inputs more often after each epoch
    
    Parameters
    ----------
    inputs : tensor, training input data
    dbvae : PyTorch dbvae module, model
    latent_dim : number of latent distributions in the DB-VAE
    smoothing_fac : scalar, must be between zero and one, a hyperparameter that controls the amount 
    of debiasing, inversely proportional to debiasing, a value of one leads to no debiasing
    bins : scalar, default=10, number of bins in the histograms used to calculate probabilities
    
    Returns
    -------
    training_sample_p : array, weights for sampling training points, as a probability distribution
    
    '''
    
    print("Recomputing sampling probabilities")
    
    # Check smoothing factor is between 0 and 1
    assert (smoothing_fac > 0) & (smoothing_fac < 1), "Smoothing factor not between zero and one"
    
    # Calculate latent means for all inputs
    mu = get_latent_mu(inputs,dbvae)
    
    # Define array for sampling probabilities
    training_sample_p = np.zeros(mu.shape[0])
    
    # For each latent dimension, make a histogram
    for i in range(latent_dim):
        
        latent_distribution = mu[:,i].detach().numpy()
        
        hist_density, bin_edges = np.histogram(latent_distribution, density=True, bins=bins)
        
        bin_edges[0] = -float('inf')
        bin_edges[-1] = float('inf')
        
        # Find bins for each record
        bin_idx = np.digitize(latent_distribution, bin_edges)
        
        # Introduce smoothing factor
        # Low smoothing factor leads to highest debiasing, as distribution
        # gets flipped
        # High smoothing factor makes sampling uniform (as it was before)
        hist_smoothed_density = hist_density + smoothing_fac
        hist_smoothed_density = hist_smoothed_density/np.sum(hist_smoothed_density)
        
        # Invert density function
        p = 1/(hist_smoothed_density[bin_idx-1])
        
        # Normalise
        p = p/np.sum(p)
        
        # Update sampling probabilities
        training_sample_p = np.maximum(p, training_sample_p)
        
    # Final normalisation
    training_sample_p = training_sample_p/np.sum(training_sample_p)
        
    return training_sample_p

def debiasing_train_step(x,y,kl_weight,model,optimiser):
    '''
    Perform a gradient descent step.
    
    Used when training the DB-VAE
    
    Parameters
    ----------
    x : tensor, training batch input data
    y : tensor, training batch input labels
    kl_weight : scalar, weights importance of regularisation (latent_loss)
    model : PyTorch dbvae module, model
    optimiser : PyTorch optimiser

    Returns
    -------
    loss : tensor, (scalar) total DB-VAE loss of the batch data points
    
    '''
    # Forward pass
    y_logit,z_mean,z_logsigma,x_recon = model(x.float())
    # Calculate loss
    loss, class_loss = debiasing_loss_fn(x,x_recon,y,y_logit,z_mean,z_logsigma,kl_weight)
    # Reset gradients
    optimiser.zero_grad()
    # Calculate gradients
    loss.backward()
    # Take gradient descent step
    optimiser.step()
    
    return loss