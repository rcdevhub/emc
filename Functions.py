# -*- coding: utf-8 -*-
"""
Ethical Model Calibration

Functions

Created on Fri Aug  6 14:27:56 2021

@author: rcpc4
"""

'''
#============================================
# Setup
#============================================
'''

import numpy as np
import pandas as pd
import time
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Custom legends
from matplotlib.cm import ScalarMappable # Colour barplots by value
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000 # Prevent overflow on certain plots

import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score,\
    classification_report, f1_score, confusion_matrix, silhouette_score
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import statsmodels.api as sm

from mlxtend.evaluate import permutation_test

import torch
from torch import nn
import torch.optim as optim

'''
#============================================
# Functions
#============================================
'''

def fit_cls_rf(data_train,labels_train,data_val,labels_val,
               plotsdir,max_depth,colnames):
    '''
     Fit random forest classification model.
     
     Parameters
     ----------
     data_train : array, training data
     labels_train : array, training labels
     data_val : array, validation data
     labels_val : array, validation labels
     plotsdir : string, directory in which to save plots
     max_depth : maximum depth (splits) of tree in the forest
     colnames : list, variable names for feature importance graph
     
     Returns
     -------
     output : tuple, prediction arrays for training and validation data, dict of performance metrics
     
    '''
         
    cls_rf = RandomForestClassifier(max_depth=max_depth,
                                         random_state=0).fit(data_train,
                                                             labels_train)
                                                             
    pred_train_rf = cls_rf.predict(data_train)
    pred_val_rf = cls_rf.predict(data_val)
    
    results = {'cls':
               {'rf':
                {'training':
                 {'confusion':confusion_matrix(labels_train,pred_train_rf),
                 'class_rep':classification_report(labels_train,pred_train_rf),
                 'f1ma':f1_score(labels_train,pred_train_rf,average='macro')},
                 'validation':
                 {'confusion':confusion_matrix(labels_val,pred_val_rf),
                 'class_rep':classification_report(labels_val,pred_val_rf),
                 'f1ma':f1_score(labels_val,pred_val_rf,average='macro')}
                 }
                    }
                     }          
                               
    print("Training - RF\n",confusion_matrix(labels_train,pred_train_rf))
    print("Training - RF\n",classification_report(labels_train,pred_train_rf))
    
    print("Validation - RF\n",confusion_matrix(labels_val,pred_val_rf))
    print("Validation - RF\n",classification_report(labels_val,pred_val_rf))
                             
    # Plot feature importances
    
    importances = cls_rf.feature_importances_
    importances_std = np.std([
        tree.feature_importances_ for tree in cls_rf.estimators_],axis=0)
        
    cls_rf_importances = pd.Series(importances, index=colnames)
    
    plt.figure()
    fig, ax = plt.subplots()
    cls_rf_importances.sort_values(ascending=False).plot.bar(yerr=importances_std[np.flip(np.argsort(importances))],ax=ax)
    plt.savefig(plotsdir+'/cls_rf_feat_imp.png',format='png',dpi=1200,bbox_inches='tight')
    
    output = (pred_train_rf,pred_val_rf,results)
    
    return output

def fit_reg_rf(data_train,target_train,data_val,target_val,
               plotsdir,max_depth,colnames):
    '''
    Fit random forest regression model.

     Parameters
     ----------
     data_train : array, training data
     target_train : array, training target
     data_val : array, validation data
     target_val : array, validation target
     plotsdir : string, directory in which to save plots
     max_depth : maximum depth (splits) of tree in the forest
     colnames : list, variable names for feature importance graph
     
     Returns
     -------
     output : tuple, prediction arrays for training and validation data, dict of performance metrics
     
    '''
        
    reg_rf = RandomForestRegressor(max_depth=max_depth,
                                         random_state=0).fit(data_train,
                                                             target_train)
                                                             
    pred_train_rf = reg_rf.predict(data_train)
    pred_val_rf = reg_rf.predict(data_val)
    
    mse_train_reg_rf = mean_squared_error(target_train, pred_train_rf)
    rmse_train_reg_rf = np.sqrt(mse_train_reg_rf)
    nrmse_train_reg_rf = rmse_train_reg_rf/np.mean(target_train)
    
    mse_val_reg_rf = mean_squared_error(target_val, pred_val_rf)
    rmse_val_reg_rf = np.sqrt(mse_val_reg_rf)
    nrmse_val_reg_rf = rmse_val_reg_rf/np.mean(target_val)
    
    results = {'reg':
               {'rf':
                {'training':
                 {'mse': mse_train_reg_rf,
                 'rmse':rmse_train_reg_rf,
                 'nrmse':nrmse_train_reg_rf},
                 'validation':
                 {'mse': mse_val_reg_rf,
                 'rmse':rmse_val_reg_rf,
                 'nrmse':nrmse_val_reg_rf},
                 }
                    }
                     }          
                               
    print("Training RMSE",rmse_train_reg_rf,"Validation RMSE",rmse_val_reg_rf)
    print("Training NRMSE",nrmse_train_reg_rf,"Validation NRMSE",nrmse_val_reg_rf)
                        
    # Plot feature importances
    
    importances = reg_rf.feature_importances_
    importances_std = np.std([
        tree.feature_importances_ for tree in reg_rf.estimators_],axis=0)
        
    reg_rf_importances = pd.Series(importances, index=colnames)
    
    plt.figure()
    fig, ax = plt.subplots()
    reg_rf_importances.sort_values(ascending=False).plot.bar(yerr=importances_std[np.flip(np.argsort(importances))],ax=ax)
    plt.savefig(plotsdir+'/reg_rf_feat_imp.png',format='png',dpi=1200,bbox_inches='tight')
    
    output = (pred_train_rf,pred_val_rf,results)
    
    return output

class nn_cls_model1(nn.Module):
    ''' Neural network for classification module. '''

    def __init__(self,num_in,num_hidden_1,num_hidden_2,num_hidden_3,num_out):
        '''
        Neural network architecture.
        
        Allowing specification of the number of neurons in each layer.

        '''
        
        super().__init__()
        self.fc1 = nn.Linear(num_in,num_hidden_1)
        self.fc2 = nn.Linear(num_hidden_1,num_hidden_2)
        self.fc3 = nn.Linear(num_hidden_2,num_hidden_3)
        self.fc4 = nn.Linear(num_hidden_3,num_out)
    def forward(self,x):
        ''' Neural network forward pass. '''
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

class nn_reg_model1(nn.Module):
    '''Neural network for regression module.'''
    
    def __init__(self,num_in,num_hidden_1,num_hidden_2,num_hidden_3,num_out):
        '''
        Neural network architecture.
        
        Allowing specification of the number of neurons in each layer.

        '''
        
        super().__init__()
        self.fc1 = nn.Linear(num_in,num_hidden_1)
        self.fc2 = nn.Linear(num_hidden_1,num_hidden_2)
        self.fc3 = nn.Linear(num_hidden_2,num_hidden_3)
        self.fc4 = nn.Linear(num_hidden_3,num_out)
        
    def forward(self,x):
        ''' Neural network forward pass '''
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        return x

def train_model(model,loss_fn,optimiser,batch_size,num_epochs,
                data_train,labels_train,data_val,labels_val,
                metrics=False):
    '''
    Train neural network model.

    Parameters
    ----------
    model : PyTorch neural network model
    loss_fn : Pytorch Loss function
    optimiser : PyTorch optimiser
    batch_size : int, batch size for backpropagation
    num_epochs : int, number of epochs to run
    data_train : Array, training data
    labels_train : Array, training labels/target
    data_val : Array, validation data
    labels_val : Array, validation labels/target
    metrics: Boolean, whether to calculate metrics

    Returns
    -------
    Output: list: arrays, training and validation losses for each epoch

    '''
    # Define training set and dataloader
    X_tensor = torch.tensor(data_train, requires_grad=False)
    Y_tensor = torch.tensor(labels_train, requires_grad=False)
    trainset = torch.utils.data.TensorDataset(X_tensor,Y_tensor)
    dloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
    
    X_val_tensor = torch.tensor(data_val, requires_grad=False)
    Y_val_tensor = torch.tensor(labels_val, requires_grad=False)
    
    train_loss_by_epoch = np.zeros(num_epochs)
    val_loss_by_epoch = np.zeros(num_epochs)
    
    # Train model
    for epoch in range(num_epochs):
        epoch_loss = []
        for i,data in enumerate(dloader):
            inputs, labels = data
            # Reset gradients
            optimiser.zero_grad()
            # Forward pass
            batch_outputs = model(inputs.float())
            # Calculate loss
            loss = loss_fn(torch.squeeze(batch_outputs), labels.float())
            epoch_loss.append(loss.item())
            # Calculate gradients
            loss.backward()
            # Take gradient descent step
            optimiser.step()
        total_epoch_loss = np.mean(epoch_loss)
        train_loss_by_epoch[epoch] = total_epoch_loss
        if metrics:           
            loss_val = loss_fn(torch.squeeze(model(torch.squeeze(X_val_tensor.float())))
                               ,Y_val_tensor.float())\
                .detach().numpy()
            val_loss_by_epoch[epoch] = loss_val        
        print("Epoch",epoch,"Training loss",total_epoch_loss,"Validation loss",loss_val)
    output = [train_loss_by_epoch,val_loss_by_epoch]
        
    return output

def get_loss_fn_weight_BCE(labels):
    '''
    Get loss function weight for an unbalanced binary class.

    Parameters
    ----------
    labels : numpy array, Vector of unbalanced labels

    Returns
    -------
    loss_fn_weight : scalar, weight for the positive class
    
    '''
    
    # Take unique
    uniq_label, uniq_label_cnt = np.unique(labels, return_counts=True)
    
    # Calculate actual proportions
    uniq_prop = uniq_label_cnt / np.sum(uniq_label_cnt)
    # Note that for BCEWithLogitsLoss, we require only a single weight
    # for the positive class
    loss_fn_weight = np.amax(np.amax(uniq_prop)/uniq_prop)
    
    return loss_fn_weight

def pred_perf(task_type,model_type,model,
              data_train,target_train,
              data_val,target_val):
    '''
    Calculate model predictions and performance.
    
    Parameters
    ----------   
    
    task_type : string, ('cls','reg') for classification or regression
    model_type : string, type of model ('nn')
    model : PyTorch neural network model
    data_train : Array, training data
    target_train : Array, training labels/target
    data_val : Array, validation data
    target_val : Array, validation labels/target
    
    Returns
    -------
    
    output : tuple, prediction arrays for training and validation, and dict of performance metrics
    
    '''
    
    if (task_type == 'cls') & (model_type == 'nn'):
        
        # Make predictions
        pred_train = np.round(torch.sigmoid(model(torch.tensor(data_train).float()))\
                                      .detach().numpy(),0)
        pred_val = np.round(torch.sigmoid(model(torch.tensor(data_val).float()))\
                                    .detach().numpy(),0)
     
        # Store results
        results = {'cls':
               {'nn':
                {'training':
                 {'confusion':confusion_matrix(target_train,pred_train),
                 'class_rep':classification_report(target_train,pred_train),
                 'f1ma':f1_score(target_train,pred_train,average='macro')},
                 'validation':
                 {'confusion':confusion_matrix(target_val,pred_val),
                 'class_rep':classification_report(target_val,pred_val),
                 'f1ma':f1_score(target_val,pred_val,average='macro')},
                 }
                    }
                     }     
            
        print("Confusion Matrix - Training")
        print(results['cls']['nn']['training']['confusion'])
        print("Classification Report - Training")
        print(results['cls']['nn']['training']['class_rep'])
        print("Confusion Matrix - Validation")
        print(results['cls']['nn']['validation']['confusion'])
        print("Classification Report - Validation")
        print(results['cls']['nn']['validation']['class_rep'])
    
        output = (pred_train,pred_val,results)
        
    elif (task_type == 'reg') & (model_type == 'nn'):
        
        # Store predictions
        pred_train = model(torch.tensor(data_train).float())\
                                          .detach().numpy()
        pred_val = model(torch.tensor(data_val).float())\
                                        .detach().numpy()
        
        mse_train = mean_squared_error(target_train, pred_train)
        rmse_train = np.sqrt(mse_train)
        nrmse_train = rmse_train/np.mean(target_train)
        
        mse_val = mean_squared_error(target_val, pred_val)
        rmse_val = np.sqrt(mse_val)
        nrmse_val = rmse_val/np.mean(target_val)
         
        # Store results
        results = {'reg':
                   {'nn':
                    {'training':
                     {'mse': mse_train,
                     'rmse':rmse_train,
                     'nrmse':nrmse_train},
                     'validation':
                     {'mse': mse_val,
                     'rmse':rmse_val,
                     'nrmse':nrmse_val},
                     }
                        }
                         }          
                                   
        print("Training RMSE",rmse_train,"Validation RMSE",rmse_val)
        print("Training NRMSE",nrmse_train,"Validation NRMSE",nrmse_val)
                
        output = (pred_train,pred_val,results)
        
    else:
        raise ValueError("Unsupported model or task type")
    
    return output

def fit_cls_nn(num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out,
               learn_rate, batch_size, epochs,
               data_train, labels_train,
               data_val, labels_val,
               overweight=False, metrics=True):

    '''
    Define and fit neural network model for classification.
   
    Parameters
    ----------
   
    num_in : scalar, number of input variables
    num_hidden_1 : scalar, number of neurons in layer
    num_hidden_2 : scalar, number of neurons in layer
    num_hidden_3 : scalar, number of neurons in layer
    num_out : scalar, number of output neurons
    learn_rate : scalar, learning rate for optimiser
    batch_size : scalar, batch size for training
    epochs : scalar, number of epochs for training
    data_train : Array, training data
    labels_train : Array, training labels/target
    data_val : Array, validation data
    labels_val : Array, validation labels/target
    overweight : bool, overweight positive class (for unbalanced classes)
    metrics : bool, calculate metrics
    
    Returns
    -------
    output : tuple: model object, predictions for training and validation data, dict of performance metrics
    
    '''
    
    model_cls = nn_cls_model1(num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out)
    
    # Optionally overweight the positive class to increase recall at expense of precision
    if overweight:
        loss_fn_weights = torch.tensor(get_loss_fn_weight_BCE(labels_train))
    else:
        loss_fn_weights = torch.ones(1)
    
    # Define loss function and optimiser
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=loss_fn_weights)
    optimiser = torch.optim.Adam(model_cls.parameters(),lr=learn_rate)

    output = train_model(model_cls, loss_fn, optimiser, batch_size, epochs,
                     data_train, labels_train,
                     data_val, labels_val, metrics)
    
    train_loss, val_loss = output
    
    plt.figure()
    plt.plot(train_loss)
    plt.figure()
    plt.plot(val_loss)
    print("argminloss",np.argmin(val_loss))
    
    # Store predictions
    # Note I'm using the torch.sigmoid wrapper to turn the logits into probabilities,
    # because I'm using the BCEWithLogitsLoss function and no sigmoid in the network,
    # which is best practice and also allows me to use class weights easily
    pred_train_cls_nn = np.round(torch.sigmoid(model_cls(torch.tensor(data_train).float()))\
                                      .detach().numpy(),0)
    pred_val_cls_nn = np.round(torch.sigmoid(model_cls(torch.tensor(data_val).float()))\
                                    .detach().numpy(),0)
        
    # Store results
    results = {'cls':
           {'nn':
            {'training':
             {'confusion':confusion_matrix(labels_train,pred_train_cls_nn),
             'class_rep':classification_report(labels_train,pred_train_cls_nn),
             'f1ma':f1_score(labels_train,pred_train_cls_nn,average='macro')},
             'validation':
             {'confusion':confusion_matrix(labels_val,pred_val_cls_nn),
             'class_rep':classification_report(labels_val,pred_val_cls_nn),
             'f1ma':f1_score(labels_val,pred_val_cls_nn,average='macro')},
             }
                }
                 }     
        
    print("Confusion Matrix - Training")
    print(results['cls']['nn']['training']['confusion'])
    print("Classification Report - Training")
    print(results['cls']['nn']['training']['class_rep'])
    print("Confusion Matrix - Validation")
    print(results['cls']['nn']['validation']['confusion'])
    print("Classification Report - Validation")
    print(results['cls']['nn']['validation']['class_rep'])
    
    output = (model_cls,pred_train_cls_nn,pred_val_cls_nn,results)
    
    return output

def fit_reg_nn(num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out,
               learn_rate, batch_size, epochs,
               data_train, target_train,
               data_val, target_val,               
               metrics=True):
    '''
    Define and fit neural network model for regression.
   
    Parameters
    ----------
   
    num_in : scalar, number of input variables
    num_hidden_1 : scalar, number of neurons in layer
    num_hidden_2 : scalar, number of neurons in layer
    num_hidden_3 : scalar, number of neurons in layer
    num_out : scalar, number of output neurons
    learn_rate : scalar, learning rate for optimiser
    batch_size : scalar, batch size for training
    epochs : scalar, number of epochs for training
    data_train : Array, training data
    target_train : Array, training labels/target
    data_val : Array, validation data
    target_val : Array, validation labels/target
    metrics : bool, calculate metrics
    
    Returns
    -------
    output : tuple: model object, predictions for training and validation data, dict of performance metrics
    
    '''
    
    model_reg = nn_reg_model1(num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out)
        
    # Define loss function and optimiser
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model_reg.parameters(),lr=learn_rate)

    output = train_model(model_reg, loss_fn, optimiser, batch_size, epochs,
                     data_train, target_train,
                     data_val, target_val, metrics)
    
    train_loss, val_loss = output
    
    plt.figure()
    plt.plot(train_loss)
    plt.figure()
    plt.plot(val_loss)
    print("argminloss",np.argmin(val_loss))
    
    # Store predictions
    pred_train_reg_nn = model_reg(torch.tensor(data_train).float())\
                                      .detach().numpy()
    pred_val_reg_nn = model_reg(torch.tensor(data_val).float())\
                                    .detach().numpy()
    
    mse_train_reg_nn = mean_squared_error(target_train, pred_train_reg_nn)
    rmse_train_reg_nn = np.sqrt(mse_train_reg_nn)
    nrmse_train_reg_nn = rmse_train_reg_nn/np.mean(target_train)
    
    mse_val_reg_nn = mean_squared_error(target_val, pred_val_reg_nn)
    rmse_val_reg_nn = np.sqrt(mse_val_reg_nn)
    nrmse_val_reg_nn = rmse_val_reg_nn/np.mean(target_val)
     
    # Store results
    results = {'reg':
               {'nn':
                {'training':
                 {'mse': mse_train_reg_nn,
                 'rmse':rmse_train_reg_nn,
                 'nrmse':nrmse_train_reg_nn},
                 'validation':
                 {'mse': mse_val_reg_nn,
                 'rmse':rmse_val_reg_nn,
                 'nrmse':nrmse_val_reg_nn},
                 }
                    }
                     }          
                               
    print("Training RMSE",rmse_train_reg_nn,"Validation RMSE",rmse_val_reg_nn)
    print("Training NRMSE",nrmse_train_reg_nn,"Validation NRMSE",nrmse_val_reg_nn)
            
    output = (model_reg,pred_train_reg_nn,pred_val_reg_nn,results)
    
    return output

def plot_reg_diag(pred_train_reg_nn,target_train_reg,target_var,
                  plotsdir,p_a_limits,r_a_limits,bins,
                  data_train,box_var=None):
    '''
    Display and save regression diagnostic plots.
    
    Parameters
    ----------
    pred_train_reg_nn : array, model predictions
    target_train_reg : array, training target
    target_var : string, target variable {'diabetes': ('diab', 'hba1c'), 'neuroticism': ('neur','sex','Hb')}
    plotsdir : string, directory in which to save plots
    p_a_limits : tuple, axes range limits for predicted vs actual plot
    r_a_limits : tuple, axes range limits for residual vs actual plot
    bins : number of histogram bins for density plots
    data_train : dataframe, training data
    box_var : string, analysis variable for residual box plot x-axis
    
    Returns
    -------
    None

    '''
    
    # Predicted vs actual
    plt.figure()
    sns.regplot(x=target_train_reg,y=pred_train_reg_nn)
    plt.plot(target_train_reg,target_train_reg,linestyle="dotted",color="gray")
    plt.xlabel('Actual '+target_var)
    plt.ylabel('Predicted '+target_var)
    plt.savefig(plotsdir+'/reg_nn_pred_act.png',format='png',dpi=1200,bbox_inches='tight')
    
    # Predicted vs actual, with density, over a defined range
    fig = plt.figure()
    ax1 = fig.add_subplot()
    x_low, x_high, y_low, y_high = p_a_limits
    plt.hist2d(x=target_train_reg,y=np.squeeze(pred_train_reg_nn),bins=bins,cmap=plt.cm.jet)
    plt.plot(target_train_reg,target_train_reg,linestyle="-",linewidth=0.25,color="white")
    plt.colorbar()
    ax1.set_xlim([x_low,x_high])
    ax1.set_ylim([y_low,y_high])
    plt.xlabel('Actual '+target_var)
    plt.ylabel('Predicted '+target_var)
    plt.savefig(plotsdir+'/reg_nn_pred_act_density.png',format='png',dpi=1200,bbox_inches='tight')
    
    # Calculate training residuals
    plt.figure()
    resid_train_reg_nn = target_train_reg-np.squeeze(pred_train_reg_nn)
    
    # Plot residuals
    plt.figure()
    sns.distplot(resid_train_reg_nn)
    plt.axvline(x=0,color="black",linewidth=1)
    plt.xlabel('Residual '+target_var)
    plt.savefig(plotsdir+'/reg_nn_resid.png',format='png',dpi=1200,bbox_inches='tight')
    
    # Residual vs actual
    plt.figure()
    sns.regplot(x=target_train_reg,y=resid_train_reg_nn)
    
    # Residual vs actual, with density, over a defined range
    fig = plt.figure()
    ax1 = fig.add_subplot()
    x_low, x_high, y_low, y_high = r_a_limits
    plt.hist2d(x=target_train_reg,y=resid_train_reg_nn,bins=bins,cmap=plt.cm.jet)
    plt.axhline(y=0,linewidth=1,linestyle='-',color="white")
    plt.colorbar()
    ax1.set_xlim([x_low,x_high])
    ax1.set_ylim([y_low,y_high])
    plt.xlabel("Actual "+target_var)
    plt.ylabel("Residual "+target_var)
    plt.savefig(plotsdir+'/reg_nn_resid_act_density.png',format='png',dpi=1200,bbox_inches='tight')
    
    # Residual vs actual - boxplot
    if box_var:
        plt.figure()
        sns.boxplot(data_train[box_var],resid_train_reg_nn)
        plt.axhline(y=0,color="black",linewidth=1)
        plt.xlabel(box_var)
        plt.ylabel('Residual')
        plt.xticks(rotation=90)
        plt.savefig(plotsdir+'/reg_nn_resid_box.png',format='png',dpi=1200,bbox_inches='tight')
    
    # QQ plot of residuals
    plt.figure()
    sm.qqplot(resid_train_reg_nn,line='s')
    
    print("Regression plots saved")
    
    return None

def demo_bias(data_type,task_type,target_var,
              data_train,data_val,
              target_train_mod,target_val_mod,
              pred_train,pred_val,
              split_vars):
    '''
    Test the model for bias by variable
    
    Parameters
    ----------
    data_type : string, dataset ('diabetes','neuroticism')
    task_type : string, ('cls','reg') for classification or regression
    target_var : string, target variable {'diabetes': ('diab', 'hba1c'), 'neuroticism': ('neur','sex','Hb')}
    data_train : dataframe, training data
    data_val : dataframe, validation data
    target_train_mod : array, training target/labels
    target_val_mod : array, validation target/labels
    pred_train : array, model predictions on training data
    pred_val : array, model predictions on validation data
    split_vars : list, list of variables by which to conduct bias analysis
    
    Returns
    -------
    results: dict of model performance metrics split by the given variables
    
    '''
    
    # Set up results dict
    results = {}
    results[data_type] = {}
    results[data_type][task_type] = {}
    results[data_type][task_type][target_var] = {}
    
    for i, var in enumerate(split_vars):
        results[data_type][task_type][target_var][var] = {}
        for j in np.unique(data_train[var]):
            results[data_type][task_type][target_var][var][str(j)] = {}
    
    if task_type == 'reg':
        for i,var in enumerate(split_vars):
            for j in np.unique(data_train[var]):
                
                    rmse_train = np.sqrt(mean_squared_error(target_train_mod[data_train[var]==j],
                                            pred_train[data_train[var]==j]))
                    nrmse_train = rmse_train/np.mean(target_train_mod[data_train[var]==j])
                    
                    rmse_val = np.sqrt(mean_squared_error(target_val_mod[data_val[var]==j],
                                                            pred_val[data_val[var]==j]))
                    nrmse_val = rmse_val/np.mean(target_val_mod[data_val[var]==j])
                    
                    results[data_type][task_type][target_var][var][str(j)]['training'] = {}
                    results[data_type][task_type][target_var][var][str(j)]['training']['rmse'] = rmse_train
                    results[data_type][task_type][target_var][var][str(j)]['training']['nrmse'] = nrmse_train
                    
                    results[data_type][task_type][target_var][var][str(j)]['validation'] = {}
                    results[data_type][task_type][target_var][var][str(j)]['validation']['rmse'] = rmse_val 
                    results[data_type][task_type][target_var][var][str(j)]['validation']['nrmse'] = nrmse_val
                    
                    print(target_var+" prediction by ",var," ",j,"\n",
                          "RMSE","Training",rmse_train,"Validation",rmse_val,'\n',
                          "NRMSE","Training",nrmse_train,"Validation",nrmse_val)
                    
    elif task_type == 'cls':
        for i,var in enumerate(split_vars):
            for j in np.unique(data_train[var]):
                
                conf_train = confusion_matrix(target_train_mod[data_train[var]==j],
                                              pred_train[data_train[var]==j])
                class_rep_train = classification_report(target_train_mod[data_train[var]==j],
                                              pred_train[data_train[var]==j])
                f1_train = f1_score(target_train_mod[data_train[var]==j],
                                              pred_train[data_train[var]==j],
                                              average='macro')
                
                conf_val = confusion_matrix(target_val_mod[data_val[var]==j],
                                              pred_val[data_val[var]==j])
                class_rep_val = classification_report(target_val_mod[data_val[var]==j],
                                              pred_val[data_val[var]==j])
                f1_val = f1_score(target_val_mod[data_val[var]==j],
                                              pred_val[data_val[var]==j],
                                              average='macro')
                
                results[data_type][task_type][target_var][var][str(j)]['training'] = {}
                results[data_type][task_type][target_var][var][str(j)]['training']['confusion'] = conf_train
                results[data_type][task_type][target_var][var][str(j)]['training']['class_rep'] = class_rep_train
                results[data_type][task_type][target_var][var][str(j)]['training']['f1ma'] = f1_train              
                
                results[data_type][task_type][target_var][var][str(j)]['validation'] = {}
                results[data_type][task_type][target_var][var][str(j)]['validation']['confusion'] = conf_val
                results[data_type][task_type][target_var][var][str(j)]['validation']['class_rep'] = class_rep_val
                results[data_type][task_type][target_var][var][str(j)]['validation']['f1ma'] = f1_val                
                
                print(target_var+" classification by ",var," ",j,"\n",
                      'Confusion matrix - training',
                      '\n',
                      conf_train,
                      '\n',
                      'Classification report - training',
                      '\n',
                      class_rep_train,
                      'Confusion matrix - validation',
                      '\n',
                      conf_val,
                      '\n',
                      'Classification report - validation',
                      '\n',
                      class_rep_val)
                
    else:
        raise ValueError("Unknown task type")
    
    return results
    
class auto_enc1(nn.Module):
    '''Neural network for autoencoder module.'''
    
    def __init__(self,num_in,num_hidden_1,num_hidden_2,num_hidden_3,num_out):
        '''
        Autoencoder architecture.
        
        Allowing specification of the number of neurons in each layer.

        '''
        
        super(auto_enc1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_in,num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1,num_hidden_2),
            nn.ReLU(),
            nn.Linear(num_hidden_2,num_hidden_3))
        self.decoder = nn.Sequential(
            nn.Linear(num_hidden_3,num_hidden_2),
            nn.ReLU(),
            nn.Linear(num_hidden_2,num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1,num_out))
        
    def forward(self,x):
        ''' Autoencoder forward pass. '''
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self,x):
        ''' Autoencoder encode only. '''
        x = self.encoder(x)
        return x

def train_auto_enc(model,loss_fn,optimiser,batch_size,num_epochs,
                   data_train,data_val):
    '''
    Train autoencoder.
    
    Parameters
    ----------
    model : PyTorch Neural network model
    loss_fn : Pytorch Loss function
    optimiser : PyTorch optimiser
    batch_size : int, batch size for backpropagation
    num_epochs : int, number of epochs to run
    data_train : Array, training data
    data_val : Array, validation data

    Returns
    -------
    Output: list, arrays: training and validation losses for each epoch
    
    '''
    
    # Define training set and dataloader
    X_tensor = torch.tensor(data_train, requires_grad=False)
    trainset = torch.utils.data.TensorDataset(X_tensor)
    dloader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
    
    X_val_tensor = torch.tensor(data_val, requires_grad=False)
    
    train_loss_by_epoch = np.zeros(num_epochs)
    val_loss_by_epoch = np.zeros(num_epochs)
    
    # Train model
    for epoch in range(num_epochs):
        epoch_loss=[]
        for i,data in enumerate(dloader):
            inputs = data[0]
            # Reset gradients
            optimiser.zero_grad()
            # Forward pass
            batch_outputs = model(inputs.float())
            # Calculate loss
            loss = loss_fn(torch.squeeze(batch_outputs),inputs.float())
            epoch_loss.append(loss.item())
            # Calculate gradients
            loss.backward()
            # Take gradient descent step
            optimiser.step()
        total_epoch_loss = np.mean(epoch_loss)
        train_loss_by_epoch[epoch] = total_epoch_loss
        loss_val = loss_fn(torch.squeeze(model(torch.squeeze(X_val_tensor).float())),
                           torch.squeeze(X_val_tensor).float()).detach().numpy()
        val_loss_by_epoch[epoch] = loss_val
        print("Epoch",epoch,"Training loss",total_epoch_loss,
              "Validation loss",loss_val)
    output = [train_loss_by_epoch,val_loss_by_epoch]
    
    return output

def fit_auto(num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out,
               learn_rate, batch_size, num_epochs,
               data_train, data_val):
    '''
    Define and fit autoencoder.
    
    Parameters
    ----------
   
    num_in : scalar, number of input variables
    num_hidden_1 : scalar, number of neurons in layer
    num_hidden_2 : scalar, number of neurons in layer
    num_hidden_3 : scalar, number of neurons in layer
    num_out : scalar, number of output neurons
    learn_rate : scalar, learning rate for optimiser
    batch_size : scalar, batch size for training
    num_epochs : scalar, number of epochs for training
    data_train : Array, training data
    data_val : Array, validation data
    
    Returns
    -------
    output : tuple: model object, training predictions (reconstructions) array, training and validation encoding arrays
    
    '''
    
    model_auto = auto_enc1(num_in, num_hidden_1, num_hidden_2, num_hidden_3, num_out)
        
    # Define loss function and optimiser
    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(model_auto.parameters(),lr=learn_rate)

    output = train_auto_enc(model_auto,loss_fn,optimiser,batch_size,num_epochs,
                   data_train,data_val)
    
    train_loss, val_loss = output
    
    plt.figure()
    plt.plot(train_loss)
    plt.figure()
    plt.plot(val_loss)
    print("argminval",np.argmin(val_loss))
    
    # Reconstruct data - training
    pred_train_auto = model_auto(torch.tensor(data_train).float())\
                              .detach().numpy()
                              
    # Extract latent space
    latent_train = model_auto.encode(torch.tensor(data_train).float())\
        .detach().numpy()
    latent_val = model_auto.encode(torch.tensor(data_val).float())\
        .detach().numpy()
                              
    output = (model_auto,pred_train_auto,latent_train,latent_val)
    
    return output

def plot_auto(pred_train_auto,latent_train,data_train,
              data_train_auto,plotsdir,plot_vars,
              task_type,target_var,
              pred_train=None,target_train=None):
    '''
    Display and save autoencoder plots.
    
    Parameters
    ----------
    pred_train_auto : array, model predictions (reconstructions)
    latent_train : array, training encoding
    data_train: dataframe, training data (original)
    data_train_auto : array, autoencoder training data
    plotsdir : string, directory in which to save plots
    plot_vars : list, variables by which to colour latent space plots
    task_type : string, ('cls','reg') for classification or regression
    target_var : string, target variable {'diabetes': ('diab', 'hba1c'), 'neuroticism': ('neur','sex','Hb')}
    pred_train : array, supervised model predictions on training data
    target_train : array, supervised model target, training data
    
    Returns
    -------
    None
    
    '''
    
    # Latent space
    plt.figure()
    plt.scatter(latent_train[:,0],latent_train[:,1])
    plt.title('Latent space')
    plt.savefig(plotsdir+'/auto_latent.png',format='png',dpi=1200,bbox_inches='tight')
    
    # Latent space by variable
    for i,var in enumerate(plot_vars):
        plt.figure()
        plt.scatter(latent_train[:,0],latent_train[:,1],c=data_train[var])
        plt.title('Latent space by '+str(var))
        plt.xticks(rotation=90)
        plt.colorbar()
        plt.savefig(plotsdir+'/auto_latent vars '+str(var)+'.png',format='png',dpi=1200,bbox_inches='tight')
    
    # Latent density
    plt.figure()
    plt.hist2d(latent_train[:,0],latent_train[:,1],bins=150,cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('Latent density')
    plt.savefig(plotsdir+'/auto_latent density.png',format='png',dpi=1200,bbox_inches='tight')
    
    # Reconstructions
    sample_ind = np.random.randint(data_train_auto.shape[0],size=100)
    # Heatmap - original
    plt.figure()
    sns.heatmap(data_train_auto[sample_ind])
    plt.xlabel('Variable')
    plt.ylabel('Sample')
    plt.savefig(plotsdir+'/auto_heat_orig.png',format='png',dpi=1200,bbox_inches='tight')
    # Heatmap - reconstructed
    plt.figure()
    sns.heatmap(pred_train_auto[sample_ind])
    plt.xlabel('Variable')
    plt.ylabel('Sample')
    plt.savefig(plotsdir+'/auto_heat_recon.png',format='png',dpi=1200,bbox_inches='tight')
    # Individual examples
    for i,j in enumerate(sample_ind[:10]):
        plt.figure()
        plt.plot(data_train_auto[j])
        plt.plot(pred_train_auto[j])
        plt.xlabel('Variable')
        plt.ylabel('Value (normalised)')
        plt.legend(handles=[mpatches.Patch(color='#1f77b4',label="Original"),
                    mpatches.Patch(color='#ff7f0e',label="Reconstructed")])
        plt.savefig(plotsdir+'/auto_recon_sample '+str(i)+'.png',format='png',dpi=1200,bbox_inches='tight')
    
    # Plot model performance within latent space
    if pred_train is not None:
        if target_train is None:
            raise ValueError("Requires target")
        if task_type == 'cls':
            # Calculate correct predictions
            correct_pred_train = (np.squeeze(pred_train) == np.squeeze(target_train))
            # Plot in latent space
            plt.figure()
            plt.scatter(latent_train[:,0],latent_train[:,1],c=correct_pred_train,marker="+",s=1,cmap="nipy_spectral")
            plt.legend(handles=[mpatches.Patch(color='#99A3A4',label="Correct"),mpatches.Patch(color='#212F3D',label="Incorrect")])
            plt.title('Classification performance in latent space - '+target_var)
            plt.savefig(plotsdir+'/auto_lat err '+task_type+' '+target_var+'.png',format='png',dpi=1200,bbox_inches='tight')
        elif task_type == 'reg':
            # Calc absolute error for each observation
            abserr_train = np.abs(np.squeeze(target_train)-np.squeeze(pred_train))
            # Plot in latent space
            plt.figure()
            plt.scatter(latent_train[:,0],latent_train[:,1],c=abserr_train,marker='+',s=5,cmap='jet')
            plt.colorbar()
            plt.title('Regression abs error in latent space - '+target_var)
            plt.savefig(plotsdir+'/auto_lat err '+task_type+' '+target_var+'.png',format='png',dpi=1200,bbox_inches="tight")
        else:
            raise ValueError("Unknown task type")

    print("Autoencoder plots saved")
    
    return None

def fit_gmm(num_components,latent_train,latent_val):
    '''
    Fit Gaussian Mixture Model to latent space.
    
    Parameters
    ----------
    num_components : scalar, number of componenets in mixture model
    latent_train : array, training encoding
    latent_val : array, validation encoding
    
    Returns
    -------
    output : tuple, GMM prediction arrays on training and validation data
    
    '''
    
    # Fit model
    gm = GaussianMixture(n_components=num_components,random_state=0).fit(latent_train)
    
    # Classify points
    pred_gmm_train = gm.predict(latent_train)
    pred_gmm_val = gm.predict(latent_val)
    
    output = (pred_gmm_train,pred_gmm_val)
    
    return output

def perf_by_component(target_train,pred_train,pred_gmm_train,
                      task_type,num_components,metric=None):
    '''
    Calculate model results by GMM component for the given task.

    Parameters
    ----------
    target_train : array, supervised model target, training data
    pred_train : array, supervised model predictions on training data
    pred_gmm_train : array, GMM prediction array on training data
    task_type : string, ('cls','reg') for classification or regression
    num_components : scalar, number of componenets in mixture model
    metric : string, performance metric to use

    Returns
    -------
    output: array, for regression: RMSE or Normalised RMSE by GMM component group;
    for classification: F1 Macro Average by GMM component group

    '''
    
    if task_type == 'reg':
        
        # Calculate RMSE by GMM group
        rmse_train_reg_component = np.zeros(num_components)
        nrmse_train_reg_component = np.zeros(num_components)
        for i in range(num_components):
            rmse_comp = np.sqrt(mean_squared_error(target_train[pred_gmm_train==i],
                                                   pred_train[pred_gmm_train==i]))
            nrmse_comp = rmse_comp/np.mean(target_train[pred_gmm_train==i])
            rmse_train_reg_component[i] = rmse_comp
            nrmse_train_reg_component[i] = nrmse_comp
        if metric == 'rmse':
            output = rmse_train_reg_component
        else:
            output = nrmse_train_reg_component
            
    elif task_type == 'cls':
        
        # Calculate F1ma by GMM group
        f1ma_train_cls_component = np.zeros(num_components)
        for i in range(num_components):
            f1ma_comp = f1_score(target_train[pred_gmm_train==i],
                                 pred_train[pred_gmm_train==i],
                                 average='macro')
            f1ma_train_cls_component[i] = f1ma_comp
        output = f1ma_train_cls_component
            
    else:
        raise ValueError("Unknown task type")
        
    return output

def plot_gmm(pred_gmm_train,latent_train,
             data_type,task_type,target_var,
             plotsdir,count_gmm_train_component,
             perf_train_component,metric,desc=''):
    '''
    Plot counts and model performance by GMM group in latent space.
    
    Parameters
    ----------
    pred_gmm_train : array, GMM prediction array on training data
    latent_train : array, training encoding
    data_type : string, dataset ('diabetes','neuroticism')
    task_type : string, ('cls','reg') for classification or regression
    target_var : string, target variable {'diabetes': ('diab', 'hba1c'), 'neuroticism': ('neur','sex','Hb')}    
    plotsdir : string, directory in which to save plots
    count_gmm_train_component : array, observation counts in each GMM component
    perf_train_component : array, performance metric by GMM component group
    metric : string, performance metric to use for plot title
    desc : string, optional description for plot title

    Returns
    -------
    None
    
    '''

    # Plot latent space by count in GMM region
    plt.figure()
    plt.scatter(latent_train[:,0],latent_train[:,1],
                c=count_gmm_train_component[pred_gmm_train],
                marker='+',s=30,cmap='jet')
    plt.colorbar()
    plt.title("Latent space by observation count in GMM region")
    plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var+' gmm count'+desc+'.png',format='png',dpi=1200,bbox_inches="tight")
    
    if task_type == 'reg':
                
        # Plot latent space by metric in GMM region
        plt.figure()
        plt.scatter(latent_train[:,0],latent_train[:,1],
                    c=perf_train_component[pred_gmm_train],
                    marker='+',s=30,cmap='jet')
        plt.colorbar()
        plt.title('Latent space by '+metric.upper()+' in GMM region - '+target_var+desc)
        plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var+' gmm '+metric+desc+'.png',format='png',dpi=1200,bbox_inches="tight")
                
        # Bar plot of gmm counts with colour showing metric
        sort_idx = np.flip(np.argsort(count_gmm_train_component))
        sort_idx_str = [str(x) for x in sort_idx] # for sorting x-axis correctly
        normin = np.amin(perf_train_component)
        normax = np.amax(perf_train_component)
        data_color = perf_train_component[sort_idx]
        data_color = [(x-normin)/(normax-normin) for x in data_color]
        my_cmap = plt.cm.get_cmap('jet')
        colors = my_cmap(data_color)
        plt.figure()
        plt.bar(x=sort_idx_str,
                height=count_gmm_train_component[sort_idx],
                color=colors)
        plt.xticks(ticks=[])
        plt.colorbar(ScalarMappable(cmap=my_cmap,
                                    norm=plt.Normalize(normin,normax)))
        plt.title('Count by GMM region, colour by '+metric.upper()+' - '+target_var+desc)
        plt.xlabel('GMM region')
        plt.ylabel('Observation count')
        plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var+' gmm count bar '+metric+desc+'.png',format='png',dpi=1200,bbox_inches="tight")
                
    elif task_type == 'cls':
                
        # Plot latent space by F1 macro average in GMM region
        plt.figure()
        plt.scatter(latent_train[:,0],latent_train[:,1],
                    c=perf_train_component[pred_gmm_train],
                    marker='+',s=30,cmap='jet')
        plt.colorbar()
        plt.title('Latent space by F1 macro average in GMM region - '+target_var+desc)
        plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var+' gmm f1ma'+desc+'.png',format='png',dpi=1200,bbox_inches="tight")
        
        # Bar plot of gmm counts with colour showing F1ma
        sort_idx = np.flip(np.argsort(count_gmm_train_component))
        sort_idx_str = [str(x) for x in sort_idx] # for sorting x-axis correctly
        normin = np.amin(perf_train_component)
        normax = np.amax(perf_train_component)
        data_color = perf_train_component[sort_idx]
        data_color = [(x-normin)/(normax-normin) for x in data_color]
        my_cmap = plt.cm.get_cmap('jet')
        colors = my_cmap(data_color)
        plt.figure()
        plt.bar(x=sort_idx_str,
                height=count_gmm_train_component[sort_idx],
                color=colors)
        plt.xticks(ticks=[])
        plt.colorbar(ScalarMappable(cmap=my_cmap,
                                    norm=plt.Normalize(normin,normax)))
        plt.title('Count by GMM region, colour by F1 macro avg - '+target_var+desc)
        plt.xlabel('GMM region')
        plt.ylabel('Observation count')
        plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var+desc+' gmm count bar f1ma.png',format='png',dpi=1200,bbox_inches="tight")
    
    else:
        raise ValueError("Unknown task type")

    print("GMM performance plots saved")
    
    return None

def perf_by_group(target_train,pred_train,group_indic_train,
                  target_val,pred_val,group_indic_val,
                  task_type):
    '''
    Calculate model results by the given split.
    
    Parameters
    ----------
    target_train : array, training target
    pred_train : array, model predictions on training data
    group_indic_train: array, binary scalar indicator of group membership, training data
    target_val : array, validation target
    pred_val : array, model predictions on validation data
    group_indic_val: array, binary scalar indicator of group membership, validation data
    task_type : string, ('cls','reg') for classification or regression
    
    Returns
    -------
    results : dict, performance metrics by group for training and validation data
    
    '''
    
    train_pct = 100*np.sum(group_indic_train)/group_indic_train.shape[0]
    val_pct = 100*np.sum(group_indic_val)/group_indic_val.shape[0]
    
    if task_type == 'reg':
        
        results = {}
        results['reg'] = {}
        results['reg']['training'] = {}
        results['reg']['validation'] = {}
        
        results['reg']['training']['grp pct'] = train_pct
        results['reg']['validation']['grp pct'] = val_pct
        
        # Calculate RMSE by GMM group
        for j in np.unique(group_indic_train):
            
            results['reg']['training'][str(j)] = {}
            results['reg']['validation'][str(j)] = {}
            
            # Training
            rmse_train = np.sqrt(mean_squared_error(target_train[group_indic_train==j],
                                                    pred_train[group_indic_train==j]))
            nrmse_train = rmse_train/np.mean(target_train[group_indic_train==j])
            
            # Validation
            rmse_val = np.sqrt(mean_squared_error(target_val[group_indic_val==j],
                                                    pred_val[group_indic_val==j]))
            nrmse_val = rmse_val/np.mean(target_val[group_indic_val==j])

            # Store results
            results['reg']['training'][str(j)]['rmse'] = rmse_train
            results['reg']['training'][str(j)]['nrmse'] = nrmse_train
            
            results['reg']['validation'][str(j)]['rmse'] = rmse_val
            results['reg']['validation'][str(j)]['nrmse'] = nrmse_val
        
        print("Prediction by group",
              "\n",results)
                   
    elif task_type == 'cls':
        
        results = {}
        results['cls'] = {}
        results['cls']['training'] = {}
        results['cls']['validation'] = {}
        
        results['cls']['training']['grp pct'] = train_pct
        results['cls']['validation']['grp pct'] = val_pct
        
        # Calculate RMSE by GMM group
        for j in np.unique(group_indic_train):
            
            results['cls']['training'][str(j)] = {}
            results['cls']['validation'][str(j)] = {}
            
            # Training
            conf_train = confusion_matrix(target_train[group_indic_train==j],
                                          pred_train[group_indic_train==j])
            class_rep_train = classification_report(target_train[group_indic_train==j],
                                          pred_train[group_indic_train==j])
            
            f1_train = f1_score(target_train[group_indic_train==j],
                                pred_train[group_indic_train==j],
                                average='macro')
            
            # Validation
            conf_val = confusion_matrix(target_val[group_indic_val==j],
                                          pred_val[group_indic_val==j])
            class_rep_val = classification_report(target_val[group_indic_val==j],
                                          pred_val[group_indic_val==j])
            
            f1_val = f1_score(target_val[group_indic_val==j],
                                pred_val[group_indic_val==j],
                                average='macro')

            results['cls']['training'][str(j)]['confusion'] = conf_train
            results['cls']['training'][str(j)]['class_rep'] = class_rep_train
            results['cls']['training'][str(j)]['f1ma'] = f1_train
            
            results['cls']['validation'][str(j)]['confusion'] = conf_val
            results['cls']['validation'][str(j)]['class_rep'] = class_rep_val
            results['cls']['validation'][str(j)]['f1ma'] = f1_val
        
        print("Prediction by group",
              "\n",results)
            
    else:
        raise ValueError("Unknown task type")
        
    return results

def calc_gini(x, weights=None):
    '''
    Calculate Gini coefficient.
    
    Credit to HYRY (stack overflow)

    Parameters
    ----------
    x : Array, Income for each group, MUST BE SORTED ASCENDING
    weights : Array, Population Weights, associated with each income.
        If None, then equal weight, like individual people.

    Returns
    -------
    gini: scalar, Gini coefficient

    '''
    
    if weights is None:
        weights = np.ones_like(x)
    # Sort incomes ascending
    sort_order = np.argsort(x)
    x = x[sort_order]
    weights = weights[sort_order]
    # Calculate mean absolute deviation in two steps
    count = np.multiply.outer(weights,weights)
    mad = np.abs(np.subtract.outer(x,x) * count).sum() / count.sum()
    rmad = mad / np.average(x, weights=weights)
    # Gini equals half the relative mean absolute deviation
    gini = 0.5 * rmad
    
    return gini

def conduct_permutation_tests(permut_vars,num_groups,
                              group_nums,data_train,
                              pred_gmm_train):
    '''
    Conduct permutation tests on given variables for each group.
    
    Parameters
    ----------
    permut_vars : list, variables to be tested
    num_groups : scalar, number of groups to run tests on
    group_nums : array, specific GMM group component numbers to be tested
    data_train : dataframe, training data
    pred_gmm_train : array, GMM prediction array on training data
    
    Returns
    -------
    permut_res2 : dataframe, permutation test p-values
    
    '''
    
    start = time.perf_counter()
    permut_res = np.zeros((len(permut_vars),num_groups))
    for i,var in enumerate(permut_vars):
        for j,grp in enumerate(group_nums):
            print(var,grp)
            test_a = data_train.loc[pred_gmm_train != grp,var]
            test_b = data_train.loc[pred_gmm_train == grp,var]
            # Use approximation as computationally unfeasible otherwise
            p_val = permutation_test(test_a,test_b,method='approximate',num_rounds=1000)
            permut_res[i,j] = p_val
    end = time.perf_counter()
    print("Runtime",(end-start)/60,"minutes")
    
    permut_res2 = pd.DataFrame(permut_res,
                              index=permut_vars,
                              columns=group_nums)
    
    return permut_res2

def plt_grp_latent(latent_train,worst_lrg_indic_train,
                   data_type,task_type,target_var,plotsdir):
    '''
    Plot worst large groups in latent space.
    
    Note: Groups are not differentiated by colour and are treated as one group
    
    Parameters
    ----------
    latent_train : array, training encoding
    worst_lrg_indic_train : array, binary scalar indicator of group membership, training data
    data_type : string, dataset ('diabetes','neuroticism')
    task_type : string, ('cls','reg') for classification or regression
    target_var : string, target variable {'diabetes': ('diab', 'hba1c'), 'neuroticism': ('neur','sex','Hb')}    
    plotsdir : string, directory in which to save plots
    
    Returns
    -------
    None
    
    '''
    plt.figure()
    plt.scatter(latent_train[:,0],latent_train[:,1],c=worst_lrg_indic_train,marker="+",s=1,cmap="nipy_spectral_r")
    plt.legend(handles=[mpatches.Patch(color='#99A3A4',label="Other"),mpatches.Patch(color='#212F3D',label="Large and underperforming")])
    plt.title("Worst large performing regions in latent space - "+target_var)
    plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var+' gmm worst_lrg.png',format='png',dpi=1200,bbox_inches='tight')
    
    return None
    
def plot_grp_hist(data_train,pred_gmm_train,group_nums,hist_vars,
                  data_type,task_type,target_var,plotsdir):
    '''
    Save histograms of groups by given non-binary variables.
    
    Parameters
    ----------
    data_train : dataframe, training data
    pred_gmm_train : array, GMM prediction array on training data
    group_nums : array, specific GMM group component numbers to be plotted
    hist_vars : list, variables by which to analyse group composition
    data_type : string, dataset ('diabetes','neuroticism')
    task_type : string, ('cls','reg') for classification or regression
    target_var : string, target variable {'diabetes': ('diab', 'hba1c'), 'neuroticism': ('neur','sex','Hb')}    
    plotsdir : string, directory in which to save plots
    
    Returns
    -------
    None
    
    '''
    for i,var in enumerate(hist_vars):
        for j,grp in enumerate(group_nums):
            plt.figure()
            sns.distplot(data_train.loc[pred_gmm_train != group_nums[j],var],hist=False,kde=True)
            sns.distplot(data_train.loc[pred_gmm_train == group_nums[j],var],hist=False,kde=True)
            plt.legend(handles=[mpatches.Patch(color='#1f77b4',label="Remainder"),
                                mpatches.Patch(color='#ff7f0e',label="Group "+str(grp))])
            plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var
                        +' gmm worst_lrg hist '+str(j)+' '+str(grp)+' '+var+'.png',
                        format='png',dpi=1200,bbox_inches="tight")
            
    return None
    
    
def plot_grp_bar(data_train,pred_gmm_train,group_nums,barplot_vars,
                 data_type,task_type,target_var,plotsdir):
    '''
    Save stacked barplots of groups by given binary variables.
    
    Parameters
    ----------
    data_train : dataframe, training data
    pred_gmm_train : array, GMM prediction array on training data
    group_nums : array, specific GMM group component numbers to be plotted
    barplot_vars : list, variables by which to analyse group composition
    data_type : string, dataset ('diabetes','neuroticism')
    task_type : string, ('cls','reg') for classification or regression
    target_var : string, target variable {'diabetes': ('diab', 'hba1c'), 'neuroticism': ('neur','sex','Hb')}    
    plotsdir : string, directory in which to save plots
    
    Returns
    -------
    None    

    '''
    for i,var in enumerate(barplot_vars):
        for j,grp in enumerate(group_nums):
            tmp_df = pd.DataFrame([data_train.loc[pred_gmm_train != group_nums[j],var].sum()\
                /data_train.loc[pred_gmm_train != group_nums[j],var].count(),
                data_train.loc[pred_gmm_train == group_nums[j],var].sum()\
                /data_train.loc[pred_gmm_train == group_nums[j],var].count()])
            plt.figure()
            plt.bar(x=["Remainder","Group "+str(grp)],height=[1,1])
            plt.bar(x=["Remainder","Group "+str(grp)],height=np.squeeze(tmp_df.values))
            plt.axhline(y=0.5,color="black",linewidth=1)
            plt.ylabel('Proportion')
            plt.legend(handles=[mpatches.Patch(color='#1f77b4',label=0),
                                mpatches.Patch(color='#ff7f0e',label=1)])
            plt.title(var)
            plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var
                        +' gmm worst_lrg bar '+str(j)+' '+str(grp)+' '+var+'.png',
                        format='png',dpi=1200,bbox_inches="tight")
    
    return None

def get_rebal_data(data_train,target_train,group_indic,
                   upsample_mult,downsample_mult):
    '''
    Make rebalanced training datasets.
    
    The underprivileged group is upsampled WITH replacement so that its rebalanced number of points is
    (the number of points in the base group * the upsampling multiplier)
    The base group is optionally downsampled WITHOUT replacement so that its rebalanced number of points is
    (the number of points in the base group * the downsampling multiplier)
    The resulting underprivileged and base groups are then joined to make a rebalanced dataset
    
    Parameters
    ----------
    data_train : array, training data
    target_train : array, training target/labels
    group_indic : array, binary scalar indicator of group membership, training data
    upsample_mult : scalar, multiplier to indicate size of rebalanced underprivileged group compared to original base group
    downsample_mult : scalar, multiplier to indicate size of rebalanced base group compared to original base group, must be < 1.
    
    Returns
    -------
    data_train_rebalanced : array, training data rebalanced after up/downsampling
    target_train_rebalanced : array, training targets/labels rebalanced after up/downsampling
    
    '''
    
    # Separate into base and underpriv sets
    tmp_data_base = data_train[group_indic.astype(int)==0]
    tmp_data_underpriv = data_train[group_indic.astype(int)==1]
    
    tmp_target_base = target_train[group_indic.astype(int)==0]
    tmp_target_underpriv = target_train[group_indic.astype(int)==1]
    
    # Set size of balanced sets
    base_downsample_num = int(tmp_data_base.shape[0]*downsample_mult)
    underpriv_upsample_num = int(tmp_data_base.shape[0]*upsample_mult)
    
    # Perform up and (optionally) downsampling
    base_downsample_ind = np.random.choice(np.array(range(tmp_data_base.shape[0])),
                                           size=base_downsample_num,
                                           replace=False)
    underpriv_upsample_ind = np.random.choice(np.array(range(tmp_data_underpriv.shape[0])),
                                             size=underpriv_upsample_num,
                                             replace=True)
    
    tmp_data_base_downsampled = tmp_data_base[base_downsample_ind]
    tmp_data_underpriv_upsampled = tmp_data_underpriv[underpriv_upsample_ind]
    
    tmp_target_base_downsampled = tmp_target_base[base_downsample_ind]
    tmp_target_underpriv_upsampled = tmp_target_underpriv[underpriv_upsample_ind]
    
    data_train_rebalanced = np.concatenate((tmp_data_base_downsampled,
                                           tmp_data_underpriv_upsampled),
                                           axis=0)
    
    target_train_rebalanced = np.concatenate((tmp_target_base_downsampled,
                                              tmp_target_underpriv_upsampled),
                                             axis=0)
    
    return data_train_rebalanced,target_train_rebalanced

def plot_gmm_comp(results_rebal_component,
                  data_type,task_type,target_var,
                  metric,plotsdir):
    '''
    Save graphs comparing original and rebalanced metrics by group
    
    Parameters
    ----------
    results_rebal_component : dataframe, contains group sizes and performance metrics
    data_type : string, dataset ('diabetes','neuroticism')
    task_type : string, ('cls','reg') for classification or regression
    target_var : string, target variable {'diabetes': ('diab', 'hba1c'), 'neuroticism': ('neur','sex','Hb')}    
    metric : string, performance metric to use for labelling
    plotsdir : string, directory in which to save plots
    
    Returns
    -------
    None    
    
    '''
    
    # Make ordered dataframes for graphs
    results_by_count = results_rebal_component.sort_values('count',ascending=False)
    results_by_metric = results_rebal_component.sort_values('orig',ascending=False)
    
    # Plot metric before and after, ordered by group size descending
    # Absolute
    plt.figure()
    plt.plot(np.array(results_by_count['orig']),marker='.')
    plt.plot(np.array(results_by_count['rebal']),marker='.')
    plt.title(metric.upper()+' by GMM group, ordered by grp size desc')
    plt.xlabel('GMM group')
    plt.ylabel(metric.upper())
    plt.legend(handles=[mpatches.Patch(color='#1f77b4',label='original'),
                        mpatches.Patch(color='#ff7f0e',label='rebalanced')])
    plt.xticks(ticks=[])
    plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var
                        +' gmm '+metric+' rebal comparison by size.png',
                        format='png',dpi=1200,bbox_inches="tight")
    
    # Plot change (orig-rebal), ordered by group size descending
    plt.figure()
    sns.barplot(x=results_by_count['grp_num'].map(str), y=results_by_count['change'],color='blue')
    plt.axhline(y=0,color='black',linewidth=1)
    plt.xlabel('GMM group')
    plt.ylabel('Change in '+metric.upper())
    plt.xticks(ticks=[])
    plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var
                +' gmm '+metric+' rebal change by size.png',
                format='png',dpi=1200,bbox_inches="tight")
    
    # Plot size (for reference), ordered by group size descending
    plt.figure()
    sns.barplot(x=results_by_count['grp_num'].map(str), y=results_by_count['count'],color='grey')
    plt.xlabel('GMM group')
    plt.xticks(ticks=[])
    plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var
            +' gmm '+metric+' rebal size by size.png',
            format='png',dpi=1200,bbox_inches="tight")
    
    # Plot metric before and after, ordered by metric descending
    # Absolute
    plt.figure()
    plt.plot(np.array(results_by_metric['orig']),marker='.')
    plt.plot(np.array(results_by_metric['rebal']),marker='.')
    plt.title(metric.upper()+' by GMM group, ordered by '+metric.upper()+' desc')
    plt.xlabel('GMM group')
    plt.ylabel(metric.upper())
    plt.legend(handles=[mpatches.Patch(color='#1f77b4',label='original'),
                        mpatches.Patch(color='#ff7f0e',label='rebalanced')])
    plt.xticks(ticks=[])
    plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var
                        +' gmm '+metric+' rebal comparison by metric.png',
                        format='png',dpi=1200,bbox_inches="tight")
    
    # Plot change (orig-rebal), ordered by group size descending
    plt.figure()
    sns.barplot(x=results_by_metric['grp_num'].map(str), y=results_by_metric['change'],color='blue')
    plt.axhline(y=0,color='black',linewidth=1)
    plt.xlabel('GMM group')
    plt.ylabel('Change in '+metric.upper())
    plt.xticks(ticks=[])
    plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var
                +' gmm '+metric+' rebal change by metric.png',
                format='png',dpi=1200,bbox_inches="tight")
    
    # Plot size (for reference), ordered by group size descending
    plt.figure()
    sns.barplot(x=results_by_metric['grp_num'].map(str), y=results_by_metric['count'],color='grey')
    plt.xlabel('GMM group')
    plt.xticks(ticks=[])
    plt.savefig(plotsdir+'/auto '+data_type+' '+task_type+' '+target_var
            +' gmm '+metric+' rebal size by metric.png',
            format='png',dpi=1200,bbox_inches="tight")
    
    return None

def plot_results_box(num_trials,metric,trials,plots_main_dir,
                     data_type,task_type,model_type,target_var):
    '''
    Make dataframe of debiasing results and boxplots.
    
    Parameters
    ----------
    num_trials : scalar, the number of iterations of the experiment
    metric : string, performance metric to use for labelling
    trials: dict, model results for all trials
    plots_main_dir : string, directory in which to save plots
    data_type : string, dataset ('diabetes','neuroticism')
    task_type : string, ('cls','reg') for classification or regression
    model_type : string, type of model ('nn')
    target_var : string, target variable {'diabetes': ('diab', 'hba1c'), 'neuroticism': ('neur','sex','Hb')}    
    
    Returns
    -------
    results_box : dataframe, results boxplot data
    results_summary : dataframe, results overall summary data
    gini_box : dataframe, gini boxplot data
    results_gini : dataframe, gini overall summary data
    
    '''
    # Make dataframes to hold output
    results_box = pd.DataFrame(np.zeros(12*num_trials),
                               index=pd.MultiIndex.from_product([['Training','Validation'],
                                                                 ['All','Base','Underpriv'],
                                                                 ['Original','Rebalanced'],
                                                                 [*range(num_trials)]],
                                                                names=['Data','Group',
                                                                       'Debiasing','Trial']),
                               columns=[metric])
    
    results_summary = pd.DataFrame(np.zeros(12),
                                   index=pd.MultiIndex.from_product([['Training','Validation'],
                                                                     ['Mean','Stdev'],
                                                                     ['All','Base','Underpriv']],
                                                                    names=['Data','Metric','Group']),
                                   columns=['value'])
    
    gini_box = pd.DataFrame(np.zeros(4*num_trials),
                             index=pd.MultiIndex.from_product([['Training','Validation'],
                                                               ['Original','Rebalanced'],
                                                               [*range(num_trials)]],
                                                              names=['Data','Debiasing','Trial']),
                             columns=['gini'])
    
    results_gini = pd.DataFrame(np.zeros(12),
                                index = pd.MultiIndex.from_product([['Training','Validation'],
                                                                    ['Mean','Stdev'],
                                                                    ['Original','Rebalanced','Change']],
                                                                   names=['Data','Metric','Debiasing']),
                                columns=['value'])
    
    
    # Populate boxplot dataframe
    for i in range(num_trials):
        results_box.loc[('Training','All','Original',i),metric] = trials[str(i)]['results']['results_mod'][task_type][model_type]['training'][metric]
        results_box.loc[('Training','All','Rebalanced',i),metric] = trials[str(i)]['results']['results_nn_rebal'][task_type][model_type]['training'][metric]
        results_box.loc[('Training','Base','Original',i),metric] = trials[str(i)]['results']['results_underperf'][task_type]['training']['0.0'][metric]
        results_box.loc[('Training','Base','Rebalanced',i),metric] = trials[str(i)]['results']['results_underperf_rebal'][task_type]['training']['0.0'][metric]
        results_box.loc[('Training','Underpriv','Original',i),metric] = trials[str(i)]['results']['results_underperf'][task_type]['training']['1.0'][metric]
        results_box.loc[('Training','Underpriv','Rebalanced',i),metric] = trials[str(i)]['results']['results_underperf_rebal'][task_type]['training']['1.0'][metric]
        results_box.loc[('Validation','All','Original',i),metric] = trials[str(i)]['results']['results_mod'][task_type][model_type]['validation'][metric]
        results_box.loc[('Validation','All','Rebalanced',i),metric] = trials[str(i)]['results']['results_nn_rebal'][task_type][model_type]['validation'][metric]
        results_box.loc[('Validation','Base','Original',i),metric] = trials[str(i)]['results']['results_underperf'][task_type]['validation']['0.0'][metric]
        results_box.loc[('Validation','Base','Rebalanced',i),metric] = trials[str(i)]['results']['results_underperf_rebal'][task_type]['validation']['0.0'][metric]
        results_box.loc[('Validation','Underpriv','Original',i),metric] = trials[str(i)]['results']['results_underperf'][task_type]['validation']['1.0'][metric]
        results_box.loc[('Validation','Underpriv','Rebalanced',i),metric] = trials[str(i)]['results']['results_underperf_rebal'][task_type]['validation']['1.0'][metric]
               
    # Populate summary dataframe
    results_summary.loc[('Training','Mean','All'),'value'] = np.mean(results_box.loc[('Training','All','Original')]-results_box.loc[('Training','All','Rebalanced')]).values
    results_summary.loc[('Training','Mean','Base'),'value'] = np.mean(results_box.loc[('Training','Base','Original')]-results_box.loc[('Training','Base','Rebalanced')]).values
    results_summary.loc[('Training','Mean','Underpriv'),'value'] = np.mean(results_box.loc[('Training','Underpriv','Original')]-results_box.loc[('Training','Underpriv','Rebalanced')]).values
    results_summary.loc[('Training','Stdev','All'),'value'] = np.std(results_box.loc[('Training','All','Original')]-results_box.loc[('Training','All','Rebalanced')]).values
    results_summary.loc[('Training','Stdev','Base'),'value'] = np.std(results_box.loc[('Training','Base','Original')]-results_box.loc[('Training','Base','Rebalanced')]).values
    results_summary.loc[('Training','Stdev','Underpriv'),'value'] = np.std(results_box.loc[('Training','Underpriv','Original')]-results_box.loc[('Training','Underpriv','Rebalanced')]).values
    results_summary.loc[('Validation','Mean','All'),'value'] = np.mean(results_box.loc[('Validation','All','Original')]-results_box.loc[('Validation','All','Rebalanced')]).values
    results_summary.loc[('Validation','Mean','Base'),'value'] = np.mean(results_box.loc[('Validation','Base','Original')]-results_box.loc[('Validation','Base','Rebalanced')]).values
    results_summary.loc[('Validation','Mean','Underpriv'),'value'] = np.mean(results_box.loc[('Validation','Underpriv','Original')]-results_box.loc[('Validation','Underpriv','Rebalanced')]).values
    results_summary.loc[('Validation','Stdev','All'),'value'] = np.std(results_box.loc[('Validation','All','Original')]-results_box.loc[('Validation','All','Rebalanced')]).values
    results_summary.loc[('Validation','Stdev','Base'),'value'] = np.std(results_box.loc[('Validation','Base','Original')]-results_box.loc[('Validation','Base','Rebalanced')]).values
    results_summary.loc[('Validation','Stdev','Underpriv'),'value'] = np.std(results_box.loc[('Validation','Underpriv','Original')]-results_box.loc[('Validation','Underpriv','Rebalanced')]).values
    
    # Populate Gini dataframes
    for i in range(num_trials):
        gini_box.loc[('Training','Original',i),'gini'] = trials[str(i)]['results']['gini_train']
        gini_box.loc[('Training','Rebalanced',i),'gini'] = trials[str(i)]['results']['gini_train_rebal']
        gini_box.loc[('Validation','Original',i),'gini'] = trials[str(i)]['results']['gini_val']
        gini_box.loc[('Validation','Rebalanced',i),'gini'] = trials[str(i)]['results']['gini_val_rebal']
    
    
    results_gini.loc[('Training','Mean','Original'),'value'] = np.mean(gini_box.loc[('Training','Original')]).values
    results_gini.loc[('Training','Mean','Rebalanced'),'value'] = np.mean(gini_box.loc[('Training','Rebalanced')]).values
    results_gini.loc[('Training','Mean','Change'),'value'] = np.mean(gini_box.loc[('Training','Original')]-gini_box.loc[('Training','Rebalanced')]).values
    results_gini.loc[('Training','Stdev','Original'),'value'] = np.std(gini_box.loc[('Training','Original')]).values
    results_gini.loc[('Training','Stdev','Rebalanced'),'value'] = np.std(gini_box.loc[('Training','Rebalanced')]).values
    results_gini.loc[('Training','Stdev','Change'),'value'] = np.std(gini_box.loc[('Training','Original')]-gini_box.loc[('Training','Rebalanced')]).values
    results_gini.loc[('Validation','Mean','Original'),'value'] = np.mean(gini_box.loc[('Validation','Original')]).values
    results_gini.loc[('Validation','Mean','Rebalanced'),'value'] = np.mean(gini_box.loc[('Validation','Rebalanced')]).values
    results_gini.loc[('Validation','Mean','Change'),'value'] = np.mean(gini_box.loc[('Validation','Original')]-gini_box.loc[('Validation','Rebalanced')]).values
    results_gini.loc[('Validation','Stdev','Original'),'value'] = np.std(gini_box.loc[('Validation','Original')]).values
    results_gini.loc[('Validation','Stdev','Rebalanced'),'value'] = np.std(gini_box.loc[('Validation','Rebalanced')]).values
    results_gini.loc[('Validation','Stdev','Change'),'value'] = np.std(gini_box.loc[('Validation','Original')]-gini_box.loc[('Validation','Rebalanced')]).values
     
    # Plot variation in experiments
    results_box_data = results_box.reset_index()
    # Box plot
    plt.figure()
    sns.factorplot(data=results_box_data,col='Data', x='Group',y=metric,kind='box',hue='Debiasing',palette='seismic')
    plt.savefig(plots_main_dir+'/'+data_type+' '+task_type+' '+target_var
                +' '+metric+' rebal variation box.png',
                format='png',dpi=1200,bbox_inches="tight")
    # Violin plot
    plt.figure()
    sns.factorplot(data=results_box_data,col='Data',x='Group',y=metric,kind='violin',hue='Debiasing',inner='quartiles',palette='seismic')
    plt.savefig(plots_main_dir+'/'+data_type+' '+task_type+' '+target_var
                +' '+metric+' rebal variation violin.png',
                format='png',dpi=1200,bbox_inches="tight")
    
    return results_box, results_summary, gini_box, results_gini