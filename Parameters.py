# -*- coding: utf-8 -*-
"""
Ethical Model Calibration

Parameters

Created on Fri Aug  6 16:12:43 2021

@author: rcpc4
"""

def call_params():    
    ''' Get model parameters dictionary. '''
    params = {}

    params['filepaths'] = {'neur': {'train':"TRAINING FILE",
                                    'val':"VALIDATION FILE"},
                           'diab': "DATA FILE"}
    
    params['num_trials'] = 10

    params['learn_rate'] = 0.001
    params['batch_size'] = 10
    
    # Structure
    params['diabetes'] = {}
    params['neuroticism'] = {}

    params['diabetes']['cls'] = {}
    params['diabetes']['reg'] = {}
    params['diabetes']['auto'] = {}
    params['diabetes']['gmm'] = {}
    params['diabetes']['rebalance'] = {}
    
    params['neuroticism']['cls'] = {}
    params['neuroticism']['reg'] = {}
    params['neuroticism']['auto'] = {}
    params['neuroticism']['gmm'] = {}
    params['neuroticism']['rebalance'] = {}
    
    params['diabetes']['cls']['rf'] = {}
    params['diabetes']['reg']['rf'] = {}
    
    params['diabetes']['cls']['rf']['diab'] = {}
    params['diabetes']['reg']['rf']['hba1c'] = {}
    
    params['neuroticism']['cls']['rf'] = {}    
    params['neuroticism']['reg']['rf'] = {}
    
    params['neuroticism']['cls']['rf']['neur'] = {}    
    params['neuroticism']['cls']['rf']['sex'] = {}
    params['neuroticism']['reg']['rf']['neur'] = {}
    params['neuroticism']['reg']['rf']['Hb'] = {}
    
    params['diabetes']['cls']['nn'] = {}
    params['diabetes']['reg']['nn'] = {}    
    
    params['neuroticism']['cls']['nn'] = {}
    params['neuroticism']['reg']['nn'] = {}

    params['diabetes']['cls']['nn']['diab'] = {}
    params['diabetes']['reg']['nn']['hba1c'] = {}
    
    params['neuroticism']['cls']['nn']['neur'] = {}
    params['neuroticism']['cls']['nn']['sex'] = {}
    params['neuroticism']['reg']['nn']['neur'] = {}
    params['neuroticism']['reg']['nn']['Hb'] = {}
    
    # Random forests
    params['diabetes']['cls']['rf']['max_depth'] = 20
    params['diabetes']['reg']['rf']['max_depth'] = 20

    params['neuroticism']['cls']['rf']['max_depth'] = 20    
    params['neuroticism']['reg']['rf']['max_depth'] = 20
    
    # Feature importance
    params['diabetes']['cls']['rf']['diab']['feat_colnames'] = ['sex', 'age', 'smoking', 'Hb', 'townsend_deprivation_index',
                                                                'bmi', 'weight', 'body_fat', 'otherseriouscondition', 'hba1c',
                                                                'ethnic_white', 'ethnic_mixed', 'ethnic_asian', 'ethnic_black',
                                                                'ethnic_chinese', 'ethnic_other', 'high_blood_pressure',
                                                                'heart_attack_angina_stroke', 'bloodclot_emphysema', 'asthma',
                                                                'rhinitis_eczema']
    
    params['diabetes']['reg']['rf']['hba1c']['feat_colnames'] = ['sex', 'age', 'smoking', 'Hb', 'townsend_deprivation_index',
                                                               'bmi', 'weight', 'body_fat', 'otherseriouscondition', 
                                                               'ethnic_white', 'ethnic_mixed', 'ethnic_asian', 'ethnic_black',
                                                               'ethnic_chinese', 'ethnic_other', 'high_blood_pressure',
                                                               'heart_attack_angina_stroke', 'bloodclot_emphysema', 'asthma',
                                                               'rhinitis_eczema']

    params['neuroticism']['cls']['rf']['neur']['feat_colnames'] = ['sex', 'age', 'mood_swings',
                                                                   'miserableness', 'irritability', 'sensitivity', 'fed_up', 'nervous',
                                                                   'anxious', 'tense', 'worry', 'nerves', 'lonely', 'guilty',
                                                                   'gp_visits_mentalhealth', 'psychiatrist_visits_mentalhealth',
                                                                   'handedness', 'smoking', 'diabetes', 'hypertension', 'angina', 'atopy',
                                                                   'asthma', 'heart_attack', 'COPD', 'stroke', 'Hb', 'reaction_time' ]    
    
    params['neuroticism']['cls']['rf']['sex']['feat_colnames'] = ['age', 'neuroticism', 'mood_swings',
                                                                   'miserableness', 'irritability', 'sensitivity', 'fed_up', 'nervous',
                                                                   'anxious', 'tense', 'worry', 'nerves', 'lonely', 'guilty',
                                                                   'gp_visits_mentalhealth', 'psychiatrist_visits_mentalhealth',
                                                                   'handedness', 'smoking', 'diabetes', 'hypertension', 'angina', 'atopy',
                                                                   'asthma', 'heart_attack', 'COPD', 'stroke', 'Hb', 'reaction_time' ]    

    params['neuroticism']['reg']['rf']['neur']['feat_colnames'] = ['sex', 'age', 'mood_swings',
                                                                   'miserableness', 'irritability', 'sensitivity', 'fed_up', 'nervous',
                                                                   'anxious', 'tense', 'worry', 'nerves', 'lonely', 'guilty',
                                                                   'gp_visits_mentalhealth', 'psychiatrist_visits_mentalhealth',
                                                                   'handedness', 'smoking', 'diabetes', 'hypertension', 'angina', 'atopy',
                                                                   'asthma', 'heart_attack', 'COPD', 'stroke', 'Hb', 'reaction_time']
    
    params['neuroticism']['reg']['rf']['Hb']['feat_colnames'] = ['sex', 'age', 'neuroticism','mood_swings',
                                                                   'miserableness', 'irritability', 'sensitivity', 'fed_up', 'nervous',
                                                                   'anxious', 'tense', 'worry', 'nerves', 'lonely', 'guilty',
                                                                   'gp_visits_mentalhealth', 'psychiatrist_visits_mentalhealth',
                                                                   'handedness', 'smoking', 'diabetes', 'hypertension', 'angina', 'atopy',
                                                                   'asthma', 'heart_attack', 'COPD', 'stroke', 'reaction_time']
    
    # Neural networks
    params['diabetes']['cls']['nn']['nums'] = (21,16,16,8,1)
    params['diabetes']['reg']['nn']['nums'] = (20,16,16,8,1)
    params['diabetes']['auto']['nums'] = (22,8,4,2,22)
    
    params['neuroticism']['cls']['nn']['nums'] = (28,16,16,8,1)
    params['neuroticism']['reg']['nn']['nums'] = (28,16,16,8,1)
    params['neuroticism']['auto']['nums'] = (29,8,4,2,29)
    
    params['diabetes']['cls']['nn']['diab']['epochs'] = 12
    params['diabetes']['reg']['nn']['hba1c']['epochs'] = 20
    params['diabetes']['auto']['epochs'] = 45
    
    params['neuroticism']['cls']['nn']['neur']['epochs'] = 3
    params['neuroticism']['cls']['nn']['sex']['epochs'] = 5
    params['neuroticism']['reg']['nn']['neur']['epochs'] = 15     
    params['neuroticism']['reg']['nn']['Hb']['epochs'] = 45    
    params['neuroticism']['auto']['epochs'] = 45
    
    params['diabetes']['cls']['nn']['diab']['epochs_rebalanced'] = 4
    params['diabetes']['reg']['nn']['hba1c']['epochs_rebalanced'] = 10
    
    params['neuroticism']['cls']['nn']['neur']['epochs_rebalanced'] = 4
    params['neuroticism']['cls']['nn']['sex']['epochs_rebalanced'] = 4
    params['neuroticism']['reg']['nn']['neur']['epochs_rebalanced'] = 7     
    params['neuroticism']['reg']['nn']['Hb']['epochs_rebalanced'] = 18   
    
    # Regression diagnostic plots
    params['diabetes']['reg']['nn']['hba1c']['box_var'] = 'hba1c_bucket'
    params['diabetes']['reg']['nn']['hba1c']['p_a_limits'] = (30,50,31,50)
    params['diabetes']['reg']['nn']['hba1c']['r_a_limits'] = (30,50,-7,7)
    params['diabetes']['reg']['nn']['hba1c']['bins'] = 250
    
    params['neuroticism']['reg']['nn']['neur']['box_var'] = 'neuroticism'
    params['neuroticism']['reg']['nn']['neur']['p_a_limits'] = (0,12,0,12)
    params['neuroticism']['reg']['nn']['neur']['r_a_limits'] = (0,12,-1,1)
    params['neuroticism']['reg']['nn']['neur']['bins'] = 50
    
    params['neuroticism']['reg']['nn']['Hb']['box_var'] = 'Hb'
    params['neuroticism']['reg']['nn']['Hb']['p_a_limits'] = (9,17,9,17)
    params['neuroticism']['reg']['nn']['Hb']['r_a_limits'] = (9,17,-1,1)
    params['neuroticism']['reg']['nn']['Hb']['bins'] = 50
    
    # Bias demo variables
    params['diabetes']['cls']['nn']['diab']['bias_split_vars'] = ['sex','age_bucket','smoking',
                                                                   'deprivation_bucket','ethnic_category']
    params['diabetes']['reg']['nn']['hba1c']['bias_split_vars'] = ['sex','age_bucket','smoking',
                                                                   'deprivation_bucket','ethnic_category',
                                                                   'diabetes']
    
    params['neuroticism']['cls']['nn']['neur']['bias_split_vars'] = ['sex','mood_swings',
                                                                       'miserableness', 'irritability', 'sensitivity', 'fed_up', 'nervous',
                                                                       'anxious', 'tense', 'worry', 'nerves', 'lonely', 'guilty',
                                                                       'gp_visits_mentalhealth', 'psychiatrist_visits_mentalhealth',
                                                                       'handedness', 'smoking', 'diabetes', 'hypertension', 'angina', 'atopy',
                                                                       'asthma', 'heart_attack', 'COPD', 'stroke', 'Hb_bin',
                                                                       'age_bin', 'reaction_time_bin']
    params['neuroticism']['cls']['nn']['sex']['bias_split_vars'] = ['mood_swings',
                                                                       'miserableness', 'irritability', 'sensitivity', 'fed_up', 'nervous',
                                                                       'anxious', 'tense', 'worry', 'nerves', 'lonely', 'guilty',
                                                                       'gp_visits_mentalhealth', 'psychiatrist_visits_mentalhealth',
                                                                       'handedness', 'smoking', 'diabetes', 'hypertension', 'angina', 'atopy',
                                                                       'asthma', 'heart_attack', 'COPD', 'stroke', 'neuroticism_bin', 'Hb_bin', 
                                                                       'age_bin', 'reaction_time_bin']
    params['neuroticism']['reg']['nn']['neur']['bias_split_vars'] = ['sex','mood_swings',
                                                                       'miserableness', 'irritability', 'sensitivity', 'fed_up', 'nervous',
                                                                       'anxious', 'tense', 'worry', 'nerves', 'lonely', 'guilty',
                                                                       'gp_visits_mentalhealth', 'psychiatrist_visits_mentalhealth',
                                                                       'handedness', 'smoking', 'diabetes', 'hypertension', 'angina', 'atopy',
                                                                       'asthma', 'heart_attack', 'COPD', 'stroke', 'Hb_bin',
                                                                       'age_bin', 'reaction_time_bin']
    params['neuroticism']['reg']['nn']['Hb']['bias_split_vars'] = ['sex','mood_swings',
                                                                   'miserableness', 'irritability', 'sensitivity', 'fed_up', 'nervous',
                                                                   'anxious', 'tense', 'worry', 'nerves', 'lonely', 'guilty',
                                                                   'gp_visits_mentalhealth', 'psychiatrist_visits_mentalhealth',
                                                                   'handedness', 'smoking', 'diabetes', 'hypertension', 'angina', 'atopy',
                                                                   'asthma', 'heart_attack', 'COPD', 'stroke', 'neuroticism_bin', 
                                                                   'age_bin', 'reaction_time_bin']
    
    # Autoencoder plot variables    
    params['diabetes']['auto']['plot_vars'] = ['age','sex','smoking',
                                               'diabetes','high_blood_pressure',
                                               'townsend_deprivation_index',
                                               'bmi','body_fat','hba1c']
    params['neuroticism']['auto']['plot_vars'] = ['age', 'sex', 'neuroticism', 'smoking', 'reaction_time', 
                                                  'worry', 'mood_swings', 'lonely', 'heart_attack',
                                                  'COPD', 'stroke', 'Hb', 'Hb_bin']
    
    # Gaussian Mixture Model
    params['diabetes']['gmm']['num_components'] = 50
    params['neuroticism']['gmm']['num_components'] = 50
    
    # Metrics for latent performance plots
    params['diabetes']['cls']['metric'] = 'f1ma'
    params['diabetes']['reg']['metric'] = 'nrmse'
    params['neuroticism']['cls']['metric'] = 'f1ma'
    params['neuroticism']['reg']['metric'] = 'nrmse'

    # Parameters for worst large groups
    params['diabetes']['cls']['percentile_threshold'] = 25
    params['diabetes']['cls']['num_worst_lrg_groups'] = 5
    params['diabetes']['reg']['percentile_threshold'] = 75
    params['diabetes']['reg']['num_worst_lrg_groups'] = 5

    params['neuroticism']['cls']['percentile_threshold'] = 25
    params['neuroticism']['cls']['num_worst_lrg_groups'] = 5
    params['neuroticism']['reg']['percentile_threshold'] = 75
    params['neuroticism']['reg']['num_worst_lrg_groups'] = 5
    
    # Variables for permutation tests
    params['diabetes']['gmm']['permut_vars'] = ['sex', 'age', 'smoking', 'diabetes', 'Hb',
                                               'townsend_deprivation_index', 'bmi', 'weight', 'body_fat',
                                               'otherseriouscondition', 'hba1c', 'ethnic_white', 'ethnic_mixed',
                                               'ethnic_asian', 'ethnic_black', 'ethnic_chinese', 'ethnic_other',
                                               'high_blood_pressure', 'heart_attack_angina_stroke',
                                               'bloodclot_emphysema', 'asthma', 'rhinitis_eczema']
    
    params['neuroticism']['gmm']['permut_vars'] = ['sex', 'age', 'neuroticism', 'mood_swings',
                                                   'miserableness', 'irritability', 'sensitivity', 'fed_up', 'nervous',
                                                   'anxious', 'tense', 'worry', 'nerves', 'lonely', 'guilty',
                                                   'gp_visits_mentalhealth', 'psychiatrist_visits_mentalhealth',
                                                   'handedness', 'smoking', 'diabetes', 'hypertension', 'angina', 'atopy',
                                                   'asthma', 'heart_attack', 'COPD', 'stroke', 'Hb','reaction_time']
    
    # Group composition plot variables
    params['diabetes']['gmm']['hist_vars'] = ['age','Hb', 'townsend_deprivation_index',
                                              'bmi', 'weight', 'body_fat','hba1c']
    params['diabetes']['gmm']['barplot_vars'] = ['sex', 'smoking', 'diabetes', 
                                                 'otherseriouscondition', 'ethnic_white', 'ethnic_mixed',
                                                 'ethnic_asian', 'ethnic_black', 'ethnic_chinese', 'ethnic_other',
                                                 'high_blood_pressure', 'heart_attack_angina_stroke',
                                                 'bloodclot_emphysema', 'asthma', 'rhinitis_eczema']
    
    params['neuroticism']['gmm']['hist_vars'] = ['age','neuroticism','Hb','reaction_time']
    params['neuroticism']['gmm']['barplot_vars'] = ['sex','gp_visits_mentalhealth','smoking','hypertension']
    
    # Rebalancing
    params['diabetes']['cls']['nn']['diab']['upsample_mult'] = 2.5
    params['diabetes']['reg']['nn']['hba1c']['upsample_mult'] = 5
    
    params['neuroticism']['cls']['nn']['neur']['upsample_mult'] = 5
    params['neuroticism']['cls']['nn']['sex']['upsample_mult'] = 2.5
    params['neuroticism']['reg']['nn']['neur']['upsample_mult'] = 6
    params['neuroticism']['reg']['nn']['Hb']['upsample_mult'] = 6
    
    params['diabetes']['cls']['nn']['diab']['downsample_mult'] = 1
    params['diabetes']['reg']['nn']['hba1c']['downsample_mult'] = 1
    
    params['neuroticism']['cls']['nn']['neur']['downsample_mult'] = 1
    params['neuroticism']['cls']['nn']['sex']['downsample_mult'] = 1
    params['neuroticism']['reg']['nn']['neur']['downsample_mult'] = 1
    params['neuroticism']['reg']['nn']['Hb']['downsample_mult'] = 1
        
    return params    