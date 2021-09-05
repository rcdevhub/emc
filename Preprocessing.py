# -*- coding: utf-8 -*-
"""
Ethical Model Calibration

Data Preprocessing

Created on Fri Aug  6 14:27:24 2021

@author: rcpc4
"""

import numpy as np
import pandas as pd
import copy

from sklearn.preprocessing import StandardScaler

def data_preprocess(data_type,filepaths):
    '''
    Import and preprocess data.

    Parameters
    ----------
    data_type : string, choice of dataset {'neuroticism','diabetes'}
    filepaths : dict, contains filepaths

    Returns
    -------
    output : tuple, processed dataframes: training and validation descriptions (pandas),
            training and validation data, training and validation data normalised

    '''    
    
    # Neuroticism
    if data_type == 'neuroticism':
        
        data_train_raw = pd.read_csv(filepaths['neur']['train'])
        data_val_raw = pd.read_csv(filepaths['neur']['val'])
                
        # Used in pipeline for denormalising
        desc_train = data_train_raw.describe()
        desc_val = data_val_raw.describe()
        
        # Training data
        
        # Add categorical variables for classification modelling
        # Based on the mean value
        data_train2 = copy.deepcopy(data_train_raw)
        
        data_train2.loc[data_train2['neuroticism']>data_train2['neuroticism'].mean(),
                        'neuroticism_bin']=1
        data_train2.loc[data_train2['neuroticism']<=data_train2['neuroticism'].mean(),
                        'neuroticism_bin']=0
        data_train2.loc[data_train2['Hb']>data_train2['Hb'].mean(),
                        'Hb_bin']=1
        data_train2.loc[data_train2['Hb']<=data_train2['Hb'].mean(),
                        'Hb_bin']=0
        data_train2.loc[data_train2['haematocrit']>data_train2['haematocrit'].mean(),
                        'haematocrit_bin']=1
        data_train2.loc[data_train2['haematocrit']<=data_train2['haematocrit'].mean(),
                        'haematocrit_bin']=0
        data_train2.loc[data_train2['age']>data_train2['age'].mean(),
                        'age_bin']=1
        data_train2.loc[data_train2['age']<=data_train2['age'].mean(),
                        'age_bin']=0
        data_train2.loc[data_train2['reaction_time']>data_train2['reaction_time'].mean(),
                        'reaction_time_bin']=1
        data_train2.loc[data_train2['reaction_time']<=data_train2['reaction_time'].mean(),
                        'reaction_time_bin']=0
        # Add age buckets
        data_train2.loc[:,'age_bucket'] = pd.cut(data_train2['age'],bins=[39,45,50,55,60,65,70])
                
        # Normalisation
        scaler = StandardScaler().fit(data_train_raw)
        train_std = scaler.transform(data_train_raw)
        
        data_train_norm = pd.DataFrame(train_std)
        data_train_norm.columns = list(data_train2.columns)[0:32]
        
        # Validation data
        
        # Add categorical variables for classification modelling
        data_val2 = copy.deepcopy(data_val_raw)
        
        data_val2.loc[data_val2['neuroticism']>data_val2['neuroticism'].mean(),
                        'neuroticism_bin']=1
        data_val2.loc[data_val2['neuroticism']<=data_val2['neuroticism'].mean(),
                        'neuroticism_bin']=0
        data_val2.loc[data_val2['Hb']>data_val2['Hb'].mean(),
                        'Hb_bin']=1
        data_val2.loc[data_val2['Hb']<=data_val2['Hb'].mean(),
                        'Hb_bin']=0
        data_val2.loc[data_val2['haematocrit']>data_val2['haematocrit'].mean(),
                        'haematocrit_bin']=1
        data_val2.loc[data_val2['haematocrit']<=data_val2['haematocrit'].mean(),
                        'haematocrit_bin']=0
        data_val2.loc[data_val2['age']>data_val2['age'].mean(),
                        'age_bin']=1
        data_val2.loc[data_val2['age']<=data_val2['age'].mean(),
                        'age_bin']=0
        data_val2.loc[data_val2['reaction_time']>data_val2['reaction_time'].mean(),
                        'reaction_time_bin']=1
        data_val2.loc[data_val2['reaction_time']<=data_val2['reaction_time'].mean(),
                        'reaction_time_bin']=0
        # Add age buckets
        data_val2.loc[:,'age_bucket'] = pd.cut(data_val2['age'],bins=[39,45,50,55,60,65,70])
                
        # Normalisation
        scaler = StandardScaler().fit(data_val_raw)
        val_std = scaler.transform(data_val_raw)
        
        data_val_norm = pd.DataFrame(val_std)
        data_val_norm.columns = list(data_val2.columns)[0:32]
        
        output = (desc_train,
                  desc_val,
                  data_train2,
                  data_train_norm,
                  data_val2,
                  data_val_norm)
           
    # Diabetes    
    elif data_type == 'diabetes':
        
        data_all_raw = pd.read_csv(filepaths['diab'])
        
        all_vars = ['biobank_id', 'sex', 'age', 'neuroticism', 'mood_swings',
               'miserableness', 'irritability', 'sensitivity', 'fed_up', 'nervous',
               'anxious', 'tense', 'worry', 'nerves', 'lonely', 'guilty',
               'gp_visits_mentalhealth', 'psychiatrist_visits_mentalhealth',
               'handedness', 'smoking', 'diabetes', 'hypertension', 'angina', 'atopy',
               'asthma', 'heart_attack', 'COPD', 'stroke', 'Hb', 'haematocrit',
               'reaction_time', 'townsend_deprivation_index', 'income_before_tax',
               'FEV1', 'bmi', 'weight', 'body_fat', 'ethnic_background', 'diabetes.1',
               'vascular_heart_problems', 'clot_dvt_bronchitis_emphysema_asthma_etc',
               'otherseriouscondition', 'hba1c', 'HDL', 'LDL', 'date_of_death',
               'shift_work']
        
        # Make list of unique value arrays to examine for preprocessing
        uniq_res = []
        for i, var in enumerate(all_vars):
            # DOD needs string conversion (which messes its order up)
            if var == "date_of_death":
                vals,counts = np.unique(data_all_raw[str(var)].astype(str),return_counts=True)
            else:
                vals,counts = np.unique(data_all_raw[str(var)].values,return_counts=True)
            table = np.concatenate((np.expand_dims(vals,axis=1),
                                    np.expand_dims(counts,axis=1)),axis=1)
            uniq_res.append(table)
        
        # Shuffle the imported data just in case the order is meaningful
        # Use a specific random seed (456) for reproducibility
        # Reset the dataframe index and retain the original
        data_all_2 = data_all_raw.sample(frac=1,random_state=456,replace=False).reset_index()
        data_all_2 = data_all_2.rename(columns={"index":"orig_index"})
        
        # Notes
        # ID (0) has unique values
        # Sex (1) has one nan
        # Age (2) has one nan
        # Neuroticism (3) has 100k nans and can be removed
        # Psychological (4)-(17) variables can be removed
        # Handedness (18) can be removed
        # Smoking (19) needs conversion to binary
        # Diabetes (20) needs conversion to binary
        # Hypertension (21), age diagnosed, replace with vascular q
        # Angina (22), age diagnosed, replace with vascular q
        # Atopy (23) age diagnosed, replace with clot q
        # Asthma (24) age diagnosed, replace with clot q
        # Heart attack (25), age diagnosed, replace with vascular q
        # COPD (26) I think this is pulmonary embolism, replace with clot q
        # stroke (27), age diagnosed, replace with vascular q
        # Hb (28) has 5% nan
        # haematocrit (29) has same records as Hb nan - dropping as 83% correlation
        # reaction_time (30) 1% nans
        # townsend_deprivation_index (31) 0.1% nans
        # income_before_tax (32) 15% unusable, categorical, but ordered
        # FEV1 (33) - 30% nans, unfortunately have to remove as gets rid of non-white
        # ethnicities. Also seems to be mainly for COPD and lungs, and we have diagnoses for those.
        # bmi (34) - 0.6% nans
        # weight (35) - 0.5% nans, looks to have a few outliers
        # body_fat (36) - 2% nans, important though
        # ethnic_background (37) - categorical, will have to be one-hotted and grouped
        # diabetes.1 (38), duplicate, remove
        # vascular_heart_problems (39), nctb and possible splitting
        # clot_dvt_bronchitis_emphysema_asthma_etc (40), nctb and possible splitting
        # otherseriouscondition (41), nctb 
        # hba1c (42), has 7% nan unfortunately, has outliers, removed above 160 (6 records)
        # HDL (43) - 14% nan, remove
        # LDL (44) - 7% outliers, possibly remove
        # date_of_death (45) - about 4% prevalence, less than diabetes - remove
        # shift_work (46) - 76% missing, remove
        
        # Drop variables
        data_all_2 = data_all_2.drop(['neuroticism', 'mood_swings',
               'miserableness', 'irritability', 'sensitivity', 'fed_up', 'nervous',
               'anxious', 'tense', 'worry', 'nerves', 'lonely', 'guilty',
               'gp_visits_mentalhealth', 'psychiatrist_visits_mentalhealth',
               'handedness','diabetes.1', 'hypertension', 'atopy', 'asthma','COPD',
               'angina', 'heart_attack', 'stroke', 'haematocrit','HDL','LDL','date_of_death',
               'shift_work'],axis=1)
                
        # Mapping variables for use
        # Smoking
        data_all_2.loc[data_all_2['smoking'] == -3,'smoking'] = np.nan
        data_all_2.loc[data_all_2['smoking'] == 2,'smoking'] = 1
        # Diabetes
        data_all_2.loc[data_all_2['diabetes'] == -3,'diabetes'] = np.nan
        data_all_2.loc[data_all_2['diabetes'] == -1,'diabetes'] = np.nan
        # Income before tax - replace with average income in the band, extremes estimated
        data_all_2.loc[data_all_2['income_before_tax'] == -3,'income_before_tax'] = np.nan
        data_all_2.loc[data_all_2['income_before_tax'] == -1,'income_before_tax'] = np.nan
        data_all_2.loc[data_all_2['income_before_tax'] == 1,'income_before_tax'] = np.average([10000,17999])
        data_all_2.loc[data_all_2['income_before_tax'] == 2,'income_before_tax'] = np.average([18000,30999])
        data_all_2.loc[data_all_2['income_before_tax'] == 3,'income_before_tax'] = np.average([31000,51999])
        data_all_2.loc[data_all_2['income_before_tax'] == 4,'income_before_tax'] = np.average([52000,100000])
        data_all_2.loc[data_all_2['income_before_tax'] == 5,'income_before_tax'] = np.average([100000,150000])
        # Ethnic background
        data_all_2.loc[data_all_2['ethnic_background'] == -3,'ethnic_background'] = np.nan
        data_all_2.loc[data_all_2['ethnic_background'] == -1,'ethnic_background'] = np.nan
        # Ethnic background - grouping into new variables
        data_all_2['ethnic_white'] = 0
        data_all_2.loc[data_all_2['ethnic_background'] == 1,'ethnic_white'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 1001,'ethnic_white'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 1002,'ethnic_white'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 1003,'ethnic_white'] = 1
        data_all_2['ethnic_mixed'] = 0
        data_all_2.loc[data_all_2['ethnic_background'] == 2,'ethnic_mixed'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 2001,'ethnic_mixed'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 2002,'ethnic_mixed'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 2003,'ethnic_mixed'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 2004,'ethnic_mixed'] = 1
        data_all_2['ethnic_asian'] = 0
        data_all_2.loc[data_all_2['ethnic_background'] == 3,'ethnic_asian'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 3001,'ethnic_asian'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 3002,'ethnic_asian'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 3003,'ethnic_asian'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 3004,'ethnic_asian'] = 1
        data_all_2['ethnic_black'] = 0
        data_all_2.loc[data_all_2['ethnic_background'] == 4,'ethnic_black'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 4001,'ethnic_black'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 4002,'ethnic_black'] = 1
        data_all_2.loc[data_all_2['ethnic_background'] == 4003,'ethnic_black'] = 1
        data_all_2['ethnic_chinese'] = 0
        data_all_2.loc[data_all_2['ethnic_background'] == 5,'ethnic_chinese'] = 1
        data_all_2['ethnic_other'] = 0
        data_all_2.loc[data_all_2['ethnic_background'] == 6,'ethnic_other'] = 1
        # Add ethnic category for graphs
        data_all_2['ethnic_category'] = "Unknown"
        data_all_2.loc[data_all_2['ethnic_white'] == 1,'ethnic_category'] = 'white'
        data_all_2.loc[data_all_2['ethnic_mixed'] == 1,'ethnic_category'] = 'mixed'
        data_all_2.loc[data_all_2['ethnic_asian'] == 1,'ethnic_category'] = 'asian'
        data_all_2.loc[data_all_2['ethnic_black'] == 1,'ethnic_category'] = 'black'
        data_all_2.loc[data_all_2['ethnic_chinese'] == 1,'ethnic_category'] = 'chinese'
        data_all_2.loc[data_all_2['ethnic_other'] == 1,'ethnic_category'] = 'other'
        # Vascular etc question
        # High blood pressure is about 24% prevalence and so is given its own variable
        # Heart attack, angina and stroke combined are about 6% overall prevalence
        data_all_2['high_blood_pressure'] = 0
        data_all_2.loc[data_all_2['vascular_heart_problems'] == 4,'high_blood_pressure'] = 1
        data_all_2['heart_attack_angina_stroke'] = 0
        data_all_2.loc[data_all_2['vascular_heart_problems'] == 1,'heart_attack_angina_stroke'] = 1
        data_all_2.loc[data_all_2['vascular_heart_problems'] == 2,'heart_attack_angina_stroke'] = 1
        data_all_2.loc[data_all_2['vascular_heart_problems'] == 3,'heart_attack_angina_stroke'] = 1
        data_all_2.loc[data_all_2['vascular_heart_problems'] == -3,'vascular_heart_problems'] = np.nan
        # Clot dvt etc question
        # Blood clots, emphysema and lung clots are combined
        # Asthma (11% prevalence) and hayfever/rhinitis/eczema (17% prevalence) are split out
        data_all_2['bloodclot_emphysema'] = 0
        data_all_2.loc[data_all_2['clot_dvt_bronchitis_emphysema_asthma_etc'] == 5, 'bloodclot_emphysema'] = 1
        data_all_2.loc[data_all_2['clot_dvt_bronchitis_emphysema_asthma_etc'] == 6, 'bloodclot_emphysema'] = 1
        data_all_2.loc[data_all_2['clot_dvt_bronchitis_emphysema_asthma_etc'] == 7, 'bloodclot_emphysema'] = 1
        data_all_2['asthma'] = 0
        data_all_2.loc[data_all_2['clot_dvt_bronchitis_emphysema_asthma_etc'] == 8, 'asthma'] = 1
        data_all_2['rhinitis_eczema'] = 0
        data_all_2.loc[data_all_2['clot_dvt_bronchitis_emphysema_asthma_etc'] == 9, 'rhinitis_eczema'] = 1
        data_all_2.loc[data_all_2['clot_dvt_bronchitis_emphysema_asthma_etc'] == -3,\
                       'clot_dvt_bronchitis_emphysema_asthma_etc'] = np.nan
        # Other serious condition
        data_all_2.loc[data_all_2['otherseriouscondition'] == -3,'otherseriouscondition'] = np.nan
        data_all_2.loc[data_all_2['otherseriouscondition'] == -1,'otherseriouscondition'] = np.nan    
        # Deprivation bucket
        data_all_2['deprivation_bucket'] = pd.cut(data_all_2['townsend_deprivation_index'],bins=10)
        # Body fat bucket
        data_all_2['body_fat_bucket'] = pd.cut(data_all_2['body_fat'],bins=10)
        # Hba1c outlier removal and bucket
        data_all_2.loc[data_all_2['hba1c'] > 160,'hba1c'] = np.nan
        data_all_2['hba1c_bucket'] = pd.cut(data_all_2['hba1c'],bins=10)
        # Age bucket
        data_all_2['age_bucket'] = pd.cut(data_all_2['age'],bins=[39,45,50,55,60,65,70])
        
        # Drop less important columns
        # This is done to maximise data coverage, by dropping columns that
        # either contain a lot of nans or are probably less important
        # than the potential data lost by keeping them, not to
        # mention the additional noise they could introduce.
        data_all_3 = data_all_2.drop(['reaction_time',
                                      'income_before_tax',
                                      'FEV1'],axis=1)
        # Remove nans and reset index
        data_all_3 = data_all_3.dropna().reset_index(drop=True)
        # Drop redundant columns (note nans need to be removed first due to new columns)
        # This is now the full prepared dataset, before splitting
        data_all_3 = data_all_3.drop(['ethnic_background',
                                      'vascular_heart_problems',
                                      'clot_dvt_bronchitis_emphysema_asthma_etc'],
                                     axis=1)
        
        # Split data into train, validation and test sets of 150k, 50k, 50k
        # Reset index as can cause errors later when editing dfs
        data_diab_train = data_all_3.loc[0:149999,:].reset_index(drop=True)
        data_diab_val = data_all_3.loc[150000:199999].reset_index(drop=True)
        data_diab_test = data_all_3.loc[200000:249999].reset_index(drop=True)
        
        # Save files
        # data_diab_train.to_csv("Data/Prep/data_diab_train"+timestamp+".csv")
        # data_diab_val.to_csv("Data/Prep/data_diab_val"+timestamp+".csv")
        # data_diab_test.to_csv("Data/Prep/data_diab_test"+timestamp+".csv")
        
        # Normalisation - Training
        data_diab_train_norm = data_diab_train.drop(['orig_index',
                                                     'biobank_id',
                                                     'ethnic_category',
                                                     'deprivation_bucket',
                                                     'body_fat_bucket',
                                                     'hba1c_bucket',
                                                     'age_bucket'],
                                                    axis=1)
        col_list = data_diab_train_norm.columns
        scaler = StandardScaler().fit(data_diab_train_norm)
        train_std = scaler.transform(data_diab_train_norm)
        data_diab_train_norm = pd.DataFrame(train_std)
        data_diab_train_norm.columns = col_list
        
        # Normalisation - Validation
        data_diab_val_norm = data_diab_val.drop(['orig_index',
                                                     'biobank_id',
                                                     'ethnic_category',
                                                     'deprivation_bucket',
                                                     'body_fat_bucket',
                                                     'hba1c_bucket',
                                                     'age_bucket'],
                                                    axis=1)
        col_list = data_diab_val_norm.columns
        scaler = StandardScaler().fit(data_diab_val_norm)
        train_std = scaler.transform(data_diab_val_norm)
        data_diab_val_norm = pd.DataFrame(train_std)
        data_diab_val_norm.columns = col_list
        
        desc_train = data_diab_train.describe()
        desc_val = data_diab_val.describe()

        output = (desc_train,
                  desc_val,
                  data_diab_train,
                  data_diab_train_norm,
                  data_diab_val,
                  data_diab_val_norm)
    
    else:
        raise ValueError('Unknown data type')
        
    return output

def prep_datasets(data_train,data_train_norm,data_val,data_val_norm,data_type,
                  task_type,target_var):
    '''
    Take prepped datasets and get ready for modelling.

    Parameters
    ----------
    data_train : dataframe, training data
    data_train_norm : dataframe, normalised training data
    data_val : dataframe, validation data
    data_val_norm : dataframe, normalised validation data
    data_type : string, dataset ('diabetes','neuroticism')
    task_type : string, ('cls','reg') for classification or regression
    target_var : string, target variable {'diabetes': ('diab', 'hba1c'), 'neuroticism': ('neur','sex','Hb')}

    Returns
    -------
    output : tuple, arrays for modelling: training data and labels/targets,
            validation data and labels/targets, autoencoder training data,
            autoencoder validation data

    '''   
    
    # Neuroticism
    # Drop haematocrit as correlated with Hb
    if data_type == 'neuroticism':

        # Prep datasets - Classification
        data_train_class_neur = data_train_norm.drop(['Unnamed: 0',
                                                 'biobank_id',
                                                 'haematocrit',
                                                 'neuroticism'],axis=1).values
        
        labels_train_class_neur = data_train['neuroticism_bin'].values
        
        data_val_class_neur = data_val_norm.drop(['Unnamed: 0',
                                              'biobank_id',
                                              'haematocrit',
                                              'neuroticism'],axis=1).values
        
        labels_val_class_neur = data_val['neuroticism_bin'].values
        
        # Sex
        data_train_class_sex = data_train_norm.drop(['Unnamed: 0',
                                                 'biobank_id',
                                                 'haematocrit',
                                                 'sex'],axis=1).values
        
        labels_train_class_sex = data_train['sex'].values
        
        data_val_class_sex = data_val_norm.drop(['Unnamed: 0',
                                              'biobank_id',
                                              'haematocrit',
                                              'sex'],axis=1).values
        
        labels_val_class_sex = data_val['sex'].values
        
        # Hb
        data_train_class_Hb = data_train_norm.drop(['Unnamed: 0',
                                                 'biobank_id',
                                                 'haematocrit',
                                                 'Hb'],axis=1).values
        
        labels_train_class_Hb = data_train['Hb_bin'].values
        
        data_val_class_Hb = data_val_norm.drop(['Unnamed: 0',
                                              'biobank_id',
                                              'haematocrit',
                                              'Hb'],axis=1).values
        
        labels_val_class_Hb = data_val['Hb_bin'].values
        
        # Prep datasets - Regression
        
        # Neuroticism
        data_train_reg_neur = copy.deepcopy(data_train_class_neur)
        target_train_reg_neur = data_train['neuroticism'].values
        
        data_val_reg_neur = copy.deepcopy(data_val_class_neur)
        target_val_reg_neur = data_val['neuroticism'].values
        
        # Hb
        data_train_reg_Hb = copy.deepcopy(data_train_class_Hb)
        target_train_reg_Hb = data_train['Hb'].values
        
        data_val_reg_Hb = copy.deepcopy(data_val_class_Hb)
        target_val_reg_Hb = data_val['Hb'].values
        
        # Prep datasets - Autoencoder
        # Only normalise the non-binary variables
        
        data_train_auto = data_train.drop(['Unnamed: 0',
                                               'haematocrit',
                                               'biobank_id',
                                               'neuroticism_bin',
                                               'Hb_bin',
                                               'haematocrit_bin',
                                               'age_bin',
                                               'reaction_time_bin',
                                               'age_bucket'],axis=1)
        data_train_auto['age'] = data_train_norm['age']
        data_train_auto['neuroticism'] = data_train_norm['neuroticism']
        data_train_auto['Hb'] = data_train_norm['Hb']
        data_train_auto['reaction_time'] = data_train_norm['reaction_time']
        data_train_auto = data_train_auto.values
        
        data_val_auto = data_val.drop(['Unnamed: 0',
                                               'haematocrit',
                                               'biobank_id',
                                               'neuroticism_bin',
                                               'Hb_bin',
                                               'haematocrit_bin',
                                               'age_bin',
                                               'reaction_time_bin',
                                               'age_bucket'],axis=1)
        data_val_auto['age'] = data_val_norm['age']
        data_val_auto['neuroticism'] = data_val_norm['neuroticism']
        data_val_auto['Hb'] = data_val_norm['Hb']
        data_val_auto['reaction_time'] = data_val_norm['reaction_time']
        data_val_auto = data_val_auto.values
        
        # Choose which sets to output
        if task_type == 'cls':
            if target_var == 'neur':
                
                data_train_mod = data_train_class_neur
                target_train_mod = labels_train_class_neur
                data_val_mod = data_val_class_neur
                target_val_mod = labels_val_class_neur
                
            elif target_var == 'sex':
            
                data_train_mod = data_train_class_sex
                target_train_mod = labels_train_class_sex
                data_val_mod = data_val_class_sex
                target_val_mod = labels_val_class_sex
              
            elif target_var == 'Hb':
                
                data_train_mod = data_train_class_Hb
                target_train_mod = labels_train_class_Hb
                data_val_mod = data_val_class_Hb
                target_val_mod = labels_val_class_Hb
            
            else:
                raise ValueError("Unknown target variable")
            
        elif task_type == 'reg':
            if target_var == 'neur':
                
                data_train_mod = data_train_reg_neur
                target_train_mod = target_train_reg_neur
                data_val_mod = data_val_reg_neur
                target_val_mod = target_val_reg_neur
            
            elif target_var == 'Hb':
                
                data_train_mod = data_train_reg_Hb
                target_train_mod = target_train_reg_Hb
                data_val_mod = data_val_reg_Hb
                target_val_mod = target_val_reg_Hb
            
            else:
                raise ValueError("Unknown target variable")        
 
        else:
            raise ValueError("Unknown model type")
            
        output = (data_train_mod,target_train_mod,
                  data_val_mod,target_val_mod,
                  data_train_auto,
                  data_val_auto)
        
    # Diabetes    
    elif data_type == 'diabetes':
        
        # Prep datasets for models - Classification
        data_train_cls = data_train_norm.drop(['diabetes'],
                                                   axis=1).values
        
        data_val_cls = data_val_norm.drop(['diabetes'],
                                                   axis=1).values
        
        labels_diab_train_cls = data_train['diabetes'].values
        labels_diab_val_cls = data_val['diabetes'].values
             
        # Prep datasets for models - Regression
        data_train_reg = data_train_norm.drop(['hba1c',
                                                        'diabetes'],
                                                        axis=1).values
        
        data_val_reg = data_val_norm.drop(['hba1c',
                                                     'diabetes'],
                                                    axis=1).values
        
        target_diab_train_reg = data_train['hba1c'].values
        target_diab_val_reg = data_val['hba1c'].values
        
        # Prep datasets for models - Autoencoder
        # Only normalise the non-binary variables
        nonbin_vars = ['age','Hb',
               'townsend_deprivation_index', 'bmi', 'weight', 'body_fat',
               'hba1c']
        data_train_auto = data_train.drop(['orig_index',
                                                     'biobank_id',
                                                     'ethnic_category',
                                                     'deprivation_bucket',
                                                     'body_fat_bucket',
                                                     'hba1c_bucket',
                                                     'age_bucket'],
                                                    axis=1)
        for i in nonbin_vars:
            data_train_auto[i] = data_train_norm[i]
        data_train_auto = data_train_auto.values.astype(float)
        
        data_val_auto = data_val.drop(['orig_index',
                                                 'biobank_id',
                                                 'ethnic_category',
                                                 'deprivation_bucket',
                                                 'body_fat_bucket',
                                                 'hba1c_bucket',
                                                 'age_bucket'],
                                                axis=1)
        data_val_auto['age'] = data_val_norm['age']
        for i in nonbin_vars:
            data_val_auto[i] = data_val_norm[i]
        data_val_auto = data_val_auto.values.astype(float)
        
        # Choose which sets to output
        if task_type == 'cls':
            if target_var == 'diab':
                
                data_train_mod = data_train_cls
                target_train_mod = labels_diab_train_cls
                data_val_mod = data_val_cls
                target_val_mod = labels_diab_val_cls
                        
            else:
                raise ValueError("Unknown target variable")
            
        elif task_type == 'reg':
            if target_var == 'hba1c':
                
                data_train_mod = data_train_reg
                target_train_mod = target_diab_train_reg
                data_val_mod = data_val_reg
                target_val_mod = target_diab_val_reg
                
            else:
                raise ValueError("Unknown target variable")        
 
        else:
            raise ValueError("Unknown model type")
            
        output = (data_train_mod,target_train_mod,
                  data_val_mod,target_val_mod,
                  data_train_auto,
                  data_val_auto)

    else:
        raise ValueError('Unknown data type')

    return output