# -*- coding: utf-8 -*-
"""
Ethical Model Calibration

Exploratory Data Analysis (EDA)

Created on Fri Aug  6 14:26:52 2021

@author: rcpc4
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # Custom legends
import seaborn as sns

def eda_graphs(data_train,plotsdir,data_type):
    '''
    Plot and save EDA graphs.

    Parameters
    ----------
    data_train : dataframe, training data
    plotsdir : string, directory in which to save plots
    data_type : string, choice of dataset {'neuroticism','diabetes'}

    Returns
    -------
    None.

    '''
    
    plotsdir = plotsdir+'/EDA'
    os.mkdir(plotsdir)
    
    if data_type == "neuroticism":
        bin_vars_short = ['sex','gp_visits_mentalhealth',
                    'psychiatrist_visits_mentalhealth',
               'handedness', 'smoking', 'diabetes', 'hypertension', 'angina', 'atopy',
               'asthma', 'heart_attack', 'COPD', 'stroke']  
                
        # Stacked barplot of binary variables
        plt.figure()
        (data_train[bin_vars_short].sum()/data_train[bin_vars_short].sum()).plot(kind='bar')
        (data_train[bin_vars_short].sum()/data_train[bin_vars_short].count()).plot(kind='bar',color='#ff7f0e')
        plt.axhline(y=0.5,color="black",linewidth=1)
        plt.ylabel('Proportion')
        plt.legend(handles=[mpatches.Patch(color='#1f77b4',label=0),mpatches.Patch(color='#ff7f0e',label=1)])
        plt.savefig(plotsdir+'/bin_stackedbar.png',format='png',dpi=1200,bbox_inches="tight")
            
        # sns.pairplot(data_train) # Very slow
        # Pairplot of continuous variables
        plt.figure()
        sns.pairplot(data_train[['neuroticism','age','reaction_time','Hb']])
        plt.savefig(plotsdir+'/pairplot_cont.png',format='png',dpi=1200,bbox_inches="tight")
        # Pairplot of continuous variables by sex
        plt.figure()
        sns.pairplot(data_train[['neuroticism','age','reaction_time','Hb','sex']], hue='sex')
        plt.savefig(plotsdir+'/pairplot_cont_sex.png',format='png',dpi=600,bbox_inches="tight")
        # Pairplot of continuous variables by smoking
        plt.figure()
        sns.pairplot(data_train[['neuroticism','age','reaction_time','Hb','smoking']], hue='smoking')
        plt.savefig(plotsdir+'/pairplot_cont_smoking.png',format='png',dpi=1200,bbox_inches="tight")
        # Pairplot of continuous variables by handedness
        plt.figure()
        sns.pairplot(data_train[['neuroticism','age','reaction_time','Hb','handedness']], hue='handedness')
        plt.savefig(plotsdir+'/pairplot_cont_hand.png',format='png',dpi=1200,bbox_inches="tight")
        
        # Correlation of all features
        plt.figure()
        sns.heatmap(data_train.corr())
        # Correlation of Psychological variables
        plt.figure()
        sns.heatmap(data_train[['neuroticism', 'mood_swings',
               'miserableness', 'irritability', 'sensitivity', 'fed_up', 'nervous',
               'anxious', 'tense', 'worry', 'nerves', 'lonely', 'guilty']].corr())
        plt.savefig(plotsdir+'/corr_pysch.png',format='png',dpi=1200,bbox_inches="tight")
        # Correlation of variables excluding detailed psychological
        plt.figure()
        sns.heatmap(data_train[['sex', 'age', 'neuroticism',
               'gp_visits_mentalhealth', 'psychiatrist_visits_mentalhealth',
               'handedness', 'smoking', 'diabetes', 'hypertension', 'angina', 'atopy',
               'asthma', 'heart_attack', 'COPD', 'stroke', 'Hb', 'haematocrit',
               'reaction_time']].corr())
        plt.savefig(plotsdir+'/corr_excldetpsych.png',format='png',dpi=1200,bbox_inches="tight")
        # Correlation of continuous variables
        plt.figure()
        sns.heatmap(data_train[['neuroticism','age','reaction_time','haematocrit','Hb']].corr())
        plt.savefig(plotsdir+'/corr_cont.png',format='png',dpi=1200,bbox_inches="tight")
        
    elif data_type == "diabetes":
        
        bin_vars = ['sex', 'smoking', 'diabetes',
        'otherseriouscondition', 'ethnic_white', 'ethnic_mixed',
        'ethnic_asian', 'ethnic_black', 'ethnic_chinese', 'ethnic_other',
        'high_blood_pressure', 'heart_attack_angina_stroke',
        'bloodclot_emphysema', 'asthma', 'rhinitis_eczema']
        
        nonbin_vars = ['age','Hb',
                       'townsend_deprivation_index', 'bmi', 'weight', 'body_fat',
                       'hba1c']
        
        # Stacked barplot of binary variables
        plt.figure()
        (data_train[bin_vars].sum()/data_train[bin_vars].sum()).plot(kind='bar')
        (data_train[bin_vars].sum()/data_train[bin_vars].count()).plot(kind='bar',color='#ff7f0e')
        plt.axhline(y=0.5,color="black",linewidth=1)
        plt.ylabel('Proportion')
        plt.legend(handles=[mpatches.Patch(color='#1f77b4',label=0),mpatches.Patch(color='#ff7f0e',label=1)])
        plt.savefig(plotsdir+'/bin_stackedbar.png',format='png',dpi=1200,bbox_inches="tight")
        
        # Pairplot of continuous variables
        # Slow to run
        plt.figure()
        sns.pairplot(data_train[nonbin_vars])
        # plt.savefig(plotsdir+'/pairplot_cont.png',format='png',dpi=1200,bbox_inches="tight")
        
        # Pairplot by sex (by diabetes one wasn't great due to low prevalence)
        # Slow to run
        plt.figure()
        sns.pairplot(data_train[['age',
         'Hb',
         'townsend_deprivation_index',
         'bmi',
         'weight',
         'body_fat',
         'hba1c',
         'sex']], hue='sex')
        
        # Pairplot of selected variables
        # Slow to run
        plt.figure()
        sns.pairplot(data_train[['age',
         'weight',
         'bmi',
         'body_fat',
         'sex']], hue='sex')
        plt.savefig(plotsdir+'/pairplot_selected_sex.png',format='png',dpi=1200,bbox_inches="tight")
        
        # Correlation of all features
        plt.figure()
        sns.heatmap(data_train.corr(),xticklabels=True,yticklabels=True)
        plt.savefig(plotsdir+'/diab_corr.png',format='png',dpi=1200,bbox_inches="tight")
        
        # Exposure plots
        # Age bucket
        plt.figure()
        sns.countplot(x=data_train['age_bucket'],color='grey')
        plt.savefig(plotsdir+'/reg_nn_diab_count_age_bucket.png',format='png',dpi=1200,bbox_inches='tight')
        # Ethnic category
        plt.figure()
        sns.countplot(x=data_train['ethnic_category'],color='grey')
        plt.savefig(plotsdir+'/reg_nn_diab_count_ethnic_category.png',format='png',dpi=1200,bbox_inches='tight')
        # Deprivation bucket
        plt.figure()
        sns.countplot(x=data_train['deprivation_bucket'],color='grey')
        plt.xticks(rotation=90)
        plt.savefig(plotsdir+'/reg_nn_diab_count_deprivation_bucket.png',format='png',dpi=1200,bbox_inches='tight')
        # Body fat bucket
        plt.figure()
        sns.countplot(x=data_train['body_fat_bucket'],color='grey')
        plt.xticks(rotation=90)
        plt.savefig(plotsdir+'/reg_nn_diab_count_body_fat_bucket.png',format='png',dpi=1200,bbox_inches='tight')
        # hba1c bucket
        plt.figure()
        sns.countplot(x=data_train['hba1c_bucket'],color='grey')
        plt.xticks(rotation=90)
        plt.savefig(plotsdir+'/reg_nn_diab_count_hba1c_bucket.png',format='png',dpi=1200,bbox_inches='tight')
        
        # Check relationship of diabetes diagnosis and hba1c
        # Not as simple as hba1c >= 48 --> diabetes
        # Most people with hba1c >= 48 have diabetes
        # But many people with diabetes do not have hba1c >= 48
        plt.figure()
        sns.distplot(data_train.loc[data_train['diabetes']==0,'hba1c'],hist=True,kde=True)
        sns.distplot(data_train.loc[data_train['diabetes']==1,'hba1c'],hist=True,kde=True)
        plt.axvline(x=48,color="black",linewidth=1)
        plt.legend(handles=[mpatches.Patch(color='#1f77b4',label="Not Diabetes"),mpatches.Patch(color='#ff7f0e',label="Diabetes")])
        plt.text(50,0.08,"<-- Hba1c diabetes diagnosis threshold")
        plt.savefig(plotsdir+'/diab_hba1c_relationship.png',format='png',dpi=1200,bbox_inches="tight")
        
        # One-way diabetes prevalence by different variables
        
        # Pivots
        pvt_train_diab_by_sex = data_train.pivot_table('diabetes',index='sex',aggfunc='mean')
        pvt_train_diab_by_smoking = data_train.pivot_table('diabetes',index='smoking',aggfunc='mean')
        pvt_train_diab_by_highbp = data_train.pivot_table('diabetes',index='high_blood_pressure',aggfunc='mean')
        pvt_train_diab_by_age = data_train.pivot_table('diabetes',index='age',aggfunc='mean')
        pvt_train_diab_by_ethnic = data_train.pivot_table('diabetes',index='ethnic_category',aggfunc='mean')
        pvt_train_diab_by_depriv = data_train.pivot_table('diabetes',index='deprivation_bucket',aggfunc='mean')
        pvt_train_diab_by_bodyfat = data_train.pivot_table('diabetes',index='body_fat_bucket',aggfunc='mean')
        pvt_train_diab_by_hba1c = data_train.pivot_table('diabetes',index='hba1c_bucket',aggfunc='mean')
        
        # Plots
        # Sex
        plt.figure()
        sns.barplot(data=pvt_train_diab_by_sex,x=pvt_train_diab_by_sex.index,y='diabetes',color='navy')
        plt.ylabel('diabetes prevalence')
        plt.savefig(plotsdir+'/diab_prev_sex.png',format='png',dpi=1200,bbox_inches="tight")
        # Smoking
        plt.figure()
        sns.barplot(data=pvt_train_diab_by_smoking,x=pvt_train_diab_by_smoking.index,y='diabetes',color='navy')
        plt.ylabel('diabetes prevalence')
        plt.savefig(plotsdir+'/diab_prev_smoking.png',format='png',dpi=1200,bbox_inches="tight")
        # High blood pressure
        plt.figure()
        sns.barplot(data=pvt_train_diab_by_highbp,x=pvt_train_diab_by_highbp.index,y='diabetes',color='navy')
        plt.ylabel('diabetes prevalence')
        plt.savefig(plotsdir+'/diab_prev_high_blood_pressure.png',format='png',dpi=1200,bbox_inches="tight")
        # Age plot requires some messing due to xticklabel density
        plt.figure()
        ageplot = sns.barplot(data=pvt_train_diab_by_age,x=pvt_train_diab_by_age.index.astype(int),y='diabetes',color='navy')
        for i, label in enumerate(ageplot.get_xticklabels()):
            if i % 2 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)
        plt.xlabel('age')
        plt.ylabel('diabetes prevalence')
        plt.savefig(plotsdir+'/diab_prev_age.png',format='png',dpi=1200,bbox_inches="tight")
        # Ethnic
        plt.figure()
        sns.barplot(data=pvt_train_diab_by_ethnic,x=pvt_train_diab_by_ethnic.index,y='diabetes',color='navy')
        plt.ylabel('diabetes prevalence')
        plt.savefig(plotsdir+'/diab_prev_ethnic.png',format='png',dpi=1200,bbox_inches="tight")
        # Deprivation bucket
        plt.figure()
        sns.barplot(data=pvt_train_diab_by_depriv,x=pvt_train_diab_by_depriv.index,y='diabetes',color='navy')
        plt.xticks(rotation=90)
        plt.ylabel('diabetes prevalence')
        plt.savefig(plotsdir+'/diab_prev_deprivation.png',format='png',dpi=1200,bbox_inches="tight")
        # Body fat bucket
        plt.figure()
        sns.barplot(data=pvt_train_diab_by_bodyfat,x=pvt_train_diab_by_bodyfat.index,y='diabetes',color='navy')
        plt.xticks(rotation=90)
        plt.ylabel('diabetes prevalence')
        plt.savefig(plotsdir+'/diab_prev_body_fat.png',format='png',dpi=1200,bbox_inches="tight")
        # Hba1c bucket
        plt.figure()
        sns.barplot(data=pvt_train_diab_by_hba1c,x=pvt_train_diab_by_hba1c.index,y='diabetes',color='navy')
        plt.xticks(rotation=90)
        plt.ylabel('diabetes prevalence')
        plt.savefig(plotsdir+'/diab_prev_hba1c_bucket.png',format='png',dpi=1200,bbox_inches="tight")
        
        # Deprivation by ethnicity
        # Histplot doesn't work, something to do with the variance in height across the groups
        # sns.histplot(data=data_train,
        #              x='townsend_deprivation_index',
        #              stat ='density',
        #              hue='ethnic_category',kde=True)
        plt.figure()
        sns.distplot(data_train.loc[data_train['ethnic_category']=='white',
                                         'townsend_deprivation_index'],hist=False,kde=True)
        sns.distplot(data_train.loc[data_train['ethnic_category']=='mixed',
                                         'townsend_deprivation_index'],hist=False,kde=True)
        sns.distplot(data_train.loc[data_train['ethnic_category']=='asian',
                                         'townsend_deprivation_index'],hist=False,kde=True)
        sns.distplot(data_train.loc[data_train['ethnic_category']=='black',
                                         'townsend_deprivation_index'],hist=False,kde=True)
        sns.distplot(data_train.loc[data_train['ethnic_category']=='chinese',
                                         'townsend_deprivation_index'],hist=False,kde=True)
        sns.distplot(data_train.loc[data_train['ethnic_category']=='other',
                                         'townsend_deprivation_index'],hist=False,kde=True)
        plt.legend(labels=['white','mixed','asian','black','chinese','other'])
        plt.savefig(plotsdir+'/deprivation by ethnicity.png',format='png',dpi=1200,bbox_inches="tight")
        
        # Body fat by ethnicity
        plt.figure()
        sns.distplot(data_train.loc[data_train['ethnic_category']=='white',
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[data_train['ethnic_category']=='mixed',
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[data_train['ethnic_category']=='asian',
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[data_train['ethnic_category']=='black',
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[data_train['ethnic_category']=='chinese',
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[data_train['ethnic_category']=='other',
                                         'body_fat'],hist=False,kde=True)
        plt.legend(labels=['white','mixed','asian','black','chinese','other'])
        plt.savefig(plotsdir+'/body_fat by ethnicity.png',format='png',dpi=1200,bbox_inches="tight")
        
        # Body fat by ethnicity and sex - female
        plt.figure()
        sns.distplot(data_train.loc[(data_train['ethnic_category']=='white') &\
                                         (data_train['sex']==0),
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[(data_train['ethnic_category']=='mixed') &\
                                         (data_train['sex']==0),
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[(data_train['ethnic_category']=='asian') &\
                                         (data_train['sex']==0),
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[(data_train['ethnic_category']=='black') &\
                                         (data_train['sex']==0),
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[(data_train['ethnic_category']=='chinese') &\
                                         (data_train['sex']==0),
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[(data_train['ethnic_category']=='other') &\
                                         (data_train['sex']==0),
                                         'body_fat'],hist=False,kde=True)
        plt.legend(labels=['white','mixed','asian','black','chinese','other'])
        plt.xlabel("Body fat by ethnicity - female")
        plt.savefig(plotsdir+'/body_fat by ethnicity sex female.png',format='png',dpi=1200,bbox_inches="tight")
        
        # Body fat by ethnicity and sex - male
        plt.figure()
        sns.distplot(data_train.loc[(data_train['ethnic_category']=='white') &\
                                         (data_train['sex']==1),
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[(data_train['ethnic_category']=='mixed') &\
                                         (data_train['sex']==1),
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[(data_train['ethnic_category']=='asian') &\
                                         (data_train['sex']==1),
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[(data_train['ethnic_category']=='black') &\
                                         (data_train['sex']==1),
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[(data_train['ethnic_category']=='chinese') &\
                                         (data_train['sex']==1),
                                         'body_fat'],hist=False,kde=True)
        sns.distplot(data_train.loc[(data_train['ethnic_category']=='other') &\
                                         (data_train['sex']==1),
                                         'body_fat'],hist=False,kde=True)
        plt.legend(labels=['white','mixed','asian','black','chinese','other'])
        plt.xlabel("Body fat by ethnicity - male")
        plt.savefig(plotsdir+'/body_fat by ethnicity sex male.png',format='png',dpi=1200,bbox_inches="tight")
        
        # Graphs of HbA1c by variable
        # HbA1c by sex
        plt.figure()
        sns.distplot(data_train.loc[(data_train['sex']==0),'hba1c'],
                                         hist=False,kde=True)
        sns.distplot(data_train.loc[(data_train['sex']==1),'hba1c'],
                                         hist=False,kde=True)
        plt.legend(labels=['male','female'])
        plt.xlabel("HbA1c")
        plt.savefig(plotsdir+'/hba1c by sex.png',format='png',dpi=1200,bbox_inches="tight")
        
        # HbA1c by smoking
        plt.figure()
        sns.distplot(data_train.loc[(data_train['smoking']==0),'hba1c'],
                                         hist=False,kde=True)
        sns.distplot(data_train.loc[(data_train['smoking']==1),'hba1c'],
                                         hist=False,kde=True)
        plt.legend(labels=['smoking=0','smoking=1'])
        plt.xlabel("HbA1c")
        plt.savefig(plotsdir+'/hba1c by smoking.png',format='png',dpi=1200,bbox_inches="tight")
    
    else:
        raise ValueError("Unknown data type")
    
    print("EDA graphs saved")
    
    return None