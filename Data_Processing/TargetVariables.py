#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy.stats import zscore
# jupyter nbconvert --to script TargetVariables.ipynb


# In[2]:


# Y = Standardized T1 (HDRS and CDI)
def standardized_T1(df_model, MAX_HDRS, MAX_CDI):
    df_model.insert(len(df_model.columns), 'Y_Standardized_T1', None)
    for line_index in df_model.index:
        if pd.notna(df_model.loc[line_index, 'Questionnary1_HDRS_T1_TOT21_Score']):
            df_model.loc[line_index, 'Y_Standardized_T1'] = df_model.loc[line_index, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        elif pd.notna(df_model.loc[line_index, 'Questionnary2_CDI_T1_Score']):
            df_model.loc[line_index, 'Y_Standardized_T1'] = (df_model.loc[line_index, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
    return df_model


# In[3]:


# Y = T1 - HDRS
def standardized_T1_HDRS(df_model, MAX_HDRS):
    df_model.insert(len(df_model.columns), 'Y_Standardized_T1_HDRS', None)
    for line_index in df_model.index:
        if pd.notna(df_model.loc[line_index, 'Questionnary1_HDRS_T1_TOT21_Score']):
            df_model.loc[line_index, 'Y_Standardized_T1_HDRS'] = df_model.loc[line_index, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
    return df_model


# In[4]:


# Y = T1 - CDI
def standardized_T1_CDI(df_model, MAX_CDI):
    df_model.insert(len(df_model.columns), 'Y_Standardized_T1_CDI', None)
    for line_index in df_model.index:
        if pd.notna(df_model.loc[line_index, 'Questionnary2_CDI_T1_Score']):
            df_model.loc[line_index, 'Y_Standardized_T1_CDI'] = (df_model.loc[line_index, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
    return df_model


# In[5]:


# Alexithymia: Z-score (TAS and AQC)
def zscore_TO_T1_TOT_Alexithymia(df_model):
    df_model.insert(5, 'Y_Zscore_TO_TOT_Alexithymia', np.nan)
    df_model.insert(6, 'Y_Zscore_T1_TOT_Alexithymia', np.nan)
    
    # Young Adults
    mean_TAS_TO = df_model['Questionnary3_TAS_20_T0_TOT_Score'].mean()
    std_dev_TAS_TO = df_model['Questionnary3_TAS_20_T0_TOT_Score'].std()
    mean_TAS_T1 = df_model['Questionnary3_TAS_20_T1_TOT_Score'].mean()
    std_dev_TAS_T1 = df_model['Questionnary3_TAS_20_T1_TOT_Score'].std()
    df_model['Questionnary3_TAS_20_T0_TOT_Score'] = (df_model['Questionnary3_TAS_20_T0_TOT_Score'] - mean_TAS_TO) / std_dev_TAS_TO
    df_model['Questionnary3_TAS_20_T1_TOT_Score'] = (df_model['Questionnary3_TAS_20_T1_TOT_Score'] - mean_TAS_T1) / std_dev_TAS_T1
    # Adolescents
    mean_AQC_TO = df_model['Questionnary4_AQC_T0_TOT_Score'].mean()
    std_dev_AQC_TO = df_model['Questionnary4_AQC_T0_TOT_Score'].std()
    mean_AQC_T1 = df_model['Questionnary4_AQC_T1_TOT_Score'].mean()
    std_dev_AQC_T1 = df_model['Questionnary4_AQC_T1_TOT_Score'].std()
    df_model['Questionnary4_AQC_T0_TOT_Score'] = (df_model['Questionnary4_AQC_T0_TOT_Score'] - mean_AQC_TO) / std_dev_AQC_TO
    df_model['Questionnary4_AQC_T1_TOT_Score'] = (df_model['Questionnary4_AQC_T1_TOT_Score'] - mean_AQC_T1) / std_dev_AQC_T1

    for line_index in df_model.index:
        if pd.notna(df_model.loc[line_index, 'Questionnary3_TAS_20_T0_TOT_Score']) or pd.notna(df_model.loc[line_index, 'Questionnary3_TAS_20_T1_TOT_Score']):
            df_model.loc[line_index, 'Y_Zscore_TO_TOT_Alexithymia'] = df_model.loc[line_index, 'Questionnary3_TAS_20_T0_TOT_Score']
            df_model.loc[line_index, 'Y_Zscore_T1_TOT_Alexithymia'] = df_model.loc[line_index, 'Questionnary3_TAS_20_T1_TOT_Score']
        elif pd.notna(df_model.loc[line_index, 'Questionnary4_AQC_T0_TOT_Score']) or pd.notna(df_model.loc[line_index, 'Questionnary4_AQC_T1_TOT_Score']):
            df_model.loc[line_index, 'Y_Zscore_TO_TOT_Alexithymia'] = df_model.loc[line_index, 'Questionnary4_AQC_T0_TOT_Score']
            df_model.loc[line_index, 'Y_Zscore_T1_TOT_Alexithymia'] = df_model.loc[line_index, 'Questionnary4_AQC_T1_TOT_Score']
        else:
            df_model.loc[line_index, 'Y_Zscore_TO_TOT_Alexithymia'] = np.nan
            df_model.loc[line_index, 'Y_Zscore_T1_TOT_Alexithymia'] = np.nan       
    return df_model


# In[6]:


# Alexithymia: Z-score (TAS)
def zscore_TO_T1_TOT_TAS(df_model):
    df_model.insert(5, 'Y_Zscore_TO_TOT_Alexithymia', np.nan)
    df_model.insert(6, 'Y_Zscore_T1_TOT_Alexithymia', np.nan)
    # Young Adults
    mean_TAS_TO = df_model['Questionnary3_TAS_20_T0_TOT_Score'].mean()
    std_dev_TAS_TO = df_model['Questionnary3_TAS_20_T0_TOT_Score'].std()
    mean_TAS_T1 = df_model['Questionnary3_TAS_20_T1_TOT_Score'].mean()
    std_dev_TAS_T1 = df_model['Questionnary3_TAS_20_T1_TOT_Score'].std()
    df_model['Questionnary3_TAS_20_T0_TOT_Score'] = (df_model['Questionnary3_TAS_20_T0_TOT_Score'] - mean_TAS_TO) / std_dev_TAS_TO
    df_model['Questionnary3_TAS_20_T1_TOT_Score'] = (df_model['Questionnary3_TAS_20_T1_TOT_Score'] - mean_TAS_T1) / std_dev_TAS_T1

    for line_index in df_model.index:
        if pd.notna(df_model.loc[line_index, 'Questionnary3_TAS_20_T0_TOT_Score']) or pd.notna(df_model.loc[line_index, 'Questionnary3_TAS_20_T1_TOT_Score']):
            df_model.loc[line_index, 'Y_Zscore_TO_TOT_Alexithymia'] = df_model.loc[line_index, 'Questionnary3_TAS_20_T0_TOT_Score']
            df_model.loc[line_index, 'Y_Zscore_T1_TOT_Alexithymia'] = df_model.loc[line_index, 'Questionnary3_TAS_20_T1_TOT_Score']
        else:
            df_model.loc[line_index, 'Y_Zscore_TO_TOT_Alexithymia'] = np.nan
            df_model.loc[line_index, 'Y_Zscore_T1_TOT_Alexithymia'] = np.nan        
    return df_model


# In[7]:


# Alexithymia: Z-score (AQC)
def zscore_TO_T1_TOT_AQC(df_model):
    df_model.insert(5, 'Y_Zscore_TO_TOT_Alexithymia', np.nan)
    df_model.insert(6, 'Y_Zscore_T1_TOT_Alexithymia', np.nan)
    # Adolescents
    mean_AQC_TO = df_model['Questionnary4_AQC_T0_TOT_Score'].mean()
    std_dev_AQC_TO = df_model['Questionnary4_AQC_T0_TOT_Score'].std()
    mean_AQC_T1 = df_model['Questionnary4_AQC_T1_TOT_Score'].mean()
    std_dev_AQC_T1 = df_model['Questionnary4_AQC_T1_TOT_Score'].std()
    df_model['Questionnary4_AQC_T0_TOT_Score'] = (df_model['Questionnary4_AQC_T0_TOT_Score'] - mean_AQC_TO) / std_dev_AQC_TO
    df_model['Questionnary4_AQC_T1_TOT_Score'] = (df_model['Questionnary4_AQC_T1_TOT_Score'] - mean_AQC_T1) / std_dev_AQC_T1

    for line_index in df_model.index:
        if pd.notna(df_model.loc[line_index, 'Questionnary4_AQC_T0_TOT_Score']) or pd.notna(df_model.loc[line_index, 'Questionnary4_AQC_T1_TOT_Score']):
            df_model.loc[line_index, 'Y_Zscore_TO_TOT_Alexithymia'] = df_model.loc[line_index, 'Questionnary4_AQC_T0_TOT_Score']
            df_model.loc[line_index, 'Y_Zscore_T1_TOT_Alexithymia'] = df_model.loc[line_index, 'Questionnary4_AQC_T1_TOT_Score']
        else:
            df_model.loc[line_index, 'Y_Zscore_TO_TOT_Alexithymia'] = np.nan
            df_model.loc[line_index, 'Y_Zscore_T1_TOT_Alexithymia'] = np.nan        
    return df_model


# In[8]:


# Y = Delta_y (HDRS and CDI) - Regression
# T0 > T1 → improvement → positive delta
# T1 > T0 → worsening → negative delta
def standardized_delta_y(df_model, MAX_HDRS, MAX_CDI):
    col = "Y_Standardized_Delta_Y"
    if col not in df_model.columns:
        df_model.insert(len(df_model.columns), col, np.nan)
    else:
        df_model[col] = np.nan
    for line_index in df_model.index:
        t0 = np.nan
        t1 = np.nan
        if pd.notna(df_model.loc[line_index, "Questionnary1_HDRS_T0_TOT21_Score"]) and pd.notna(df_model.loc[line_index, "Questionnary1_HDRS_T1_TOT21_Score"]):
            t0 = df_model.loc[line_index, "Questionnary1_HDRS_T0_TOT21_Score"] / MAX_HDRS
            t1 = df_model.loc[line_index, "Questionnary1_HDRS_T1_TOT21_Score"] / MAX_HDRS
        elif pd.notna(df_model.loc[line_index, "Questionnary2_CDI_T0_Score"]) and pd.notna(df_model.loc[line_index, "Questionnary2_CDI_T1_Score"]):
            t0 = (df_model.loc[line_index, "Questionnary2_CDI_T0_Score"] - 40) / (MAX_CDI - 40)
            t1 = (df_model.loc[line_index, "Questionnary2_CDI_T1_Score"] - 40) / (MAX_CDI - 40)
        if pd.notna(t0) and pd.notna(t1):
            delta_y = t0 - t1
        else:
            delta_y = np.nan
        df_model.loc[line_index, col] = delta_y
    return df_model


# In[9]:


# Deltas
def get_deltas(df_model, MAX_HDRS, MAX_CDI):
    deltas = []
    for idx in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score']):
            t0 = df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score'] / MAX_HDRS
            t1 = df_model.loc[idx, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        elif pd.notna(df_model.loc[idx, 'Questionnary2_CDI_T0_Score']):
            t0 = (df_model.loc[idx, 'Questionnary2_CDI_T0_Score'] - 40) / (MAX_CDI - 40)
            t1 = (df_model.loc[idx, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
        if pd.notna(t1) and pd.notna(t0):
            deltas.append(t1 - t0)
    return deltas


# In[10]:


# Calculate lower_threshold and upper_threshold
def calculate_threshold(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct):
    deltas = []
    for idx in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score']):
            t0 = df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score'] / MAX_HDRS
            t1 = df_model.loc[idx, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        elif pd.notna(df_model.loc[idx, 'Questionnary2_CDI_T0_Score']):
            t0 = (df_model.loc[idx, 'Questionnary2_CDI_T0_Score'] - 40) / (MAX_CDI - 40)
            t1 = (df_model.loc[idx, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
        if pd.notna(t1) and pd.notna(t0):
            deltas.append(t1 - t0)
    lower_threshold = np.percentile(deltas, lower_pct)
    upper_threshold = np.percentile(deltas, upper_pct)
    return lower_threshold, upper_threshold


# In[11]:


# Y = Delta_y (HDRS and CDI) - Standardized Binary: worse and better
def standardized_binary_evolution(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct):
    # Strong relative improvement -> Better = delta_y <= -0.04348
    # vs. little improvement / stability / worsening -> Worse  = delta_y >= -0.01449
    df_model.insert(len(df_model.columns), 'Y_Binary_Classe_Delta_Y', None)
    lower_threshold, upper_threshold = calculate_threshold(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct)
    for idx in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score']):
            t0 = df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score'] / MAX_HDRS
            t1 = df_model.loc[idx, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        elif pd.notna(df_model.loc[idx, 'Questionnary2_CDI_T0_Score']):
            t0 = (df_model.loc[idx, 'Questionnary2_CDI_T0_Score'] - 40) / (MAX_CDI - 40)
            t1 = (df_model.loc[idx, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
        if pd.notna(t1):
            delta_y = t1 - t0
            if delta_y <= lower_threshold:
                df_model.loc[idx, 'Y_Binary_Classe_Delta_Y'] = "Better"
            elif delta_y >= upper_threshold:
                df_model.loc[idx, 'Y_Binary_Classe_Delta_Y'] = "Worse"
        else:
            df_model.loc[idx, 'Y_Binary_Classe_Delta_Y'] = np.nan 
    return df_model


# In[12]:


# Meaningful improvement = a reduction of at least 5% from the standardized scale.
# delta_y < 0 = any improvement
# delta_y <= -0.05 = minimal but consistent improvement
# delta_y >= 0 = stable or worsening
def standardized_binary_improvement(df_model, MAX_HDRS, MAX_CDI, min_improvement=0.05):
    df_model = df_model.copy()
    df_model["Delta_Y_raw"] = np.nan
    df_model["Y_Binary_Improvement"] = np.nan
    for idx in df_model.index:
        t0 = t1 = np.nan
        # Young Adults / HDRS
        if (pd.notna(df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score']) and pd.notna(df_model.loc[idx, 'Questionnary1_HDRS_T1_TOT21_Score'])):
            t0 = df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score'] / MAX_HDRS
            t1 = df_model.loc[idx, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        # Adolescents / CDI
        elif (pd.notna(df_model.loc[idx, 'Questionnary2_CDI_T0_Score']) and pd.notna(df_model.loc[idx, 'Questionnary2_CDI_T1_Score'])):
            t0 = (df_model.loc[idx, 'Questionnary2_CDI_T0_Score'] - 40) / (MAX_CDI - 40)
            t1 = (df_model.loc[idx, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
        if pd.notna(t0) and pd.notna(t1):
            delta_y = t1 - t0
            df_model.loc[idx, "Delta_Y_raw"] = delta_y
            # 1 = meaningful clinical improvement
            # 0 = low improvement, stable, or worsened
            df_model.loc[idx, "Y_Binary_Improvement"] = int(delta_y <= -min_improvement)
    return df_model


# In[13]:


# Y = Delta_y values (HDRS and CDI) - Standardized Binary: worse and better
def standardized_binary_evolution_delta_value(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct):
    df_model.insert(len(df_model.columns), 'Y_Binary_Classe_Delta_Y_Value', None)
    df_model.insert(len(df_model.columns), 'Y_Binary_Classe_Delta_Y', None)
    lower_threshold, upper_threshold = calculate_threshold(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct)
    for idx in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score']):
            t0 = df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score'] / MAX_HDRS
            t1 = df_model.loc[idx, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        elif pd.notna(df_model.loc[idx, 'Questionnary2_CDI_T0_Score']):
            t0 = (df_model.loc[idx, 'Questionnary2_CDI_T0_Score'] - 40) / (MAX_CDI - 40)
            t1 = (df_model.loc[idx, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
        if pd.notna(t1):
            delta_y = t1 - t0
            if delta_y <= lower_threshold:
                df_model.loc[idx, 'Y_Binary_Classe_Delta_Y_Value'] = delta_y
                df_model.loc[idx, 'Y_Binary_Classe_Delta_Y'] = "Better"
            elif delta_y >= upper_threshold:
                df_model.loc[idx, 'Y_Binary_Classe_Delta_Y_Value'] = delta_y
                df_model.loc[idx, 'Y_Binary_Classe_Delta_Y'] = "Worse"
        else:
            df_model.loc[idx, 'Y_Binary_Classe_Delta_Y_Value'] = np.nan 
            df_model.loc[idx, 'Y_Binary_Classe_Delta_Y'] = np.nan 
    return df_model


# In[14]:


# Y = Delta_y (HDRS and CDI) - Standardized Multiclass: worse, stable, better
def standardized_multiclass_evolution(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct):
    df_model.insert(len(df_model.columns), 'Y_Classe_Evolution_Delta_Y', None)
    lower_threshold, upper_threshold = calculate_threshold(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct)
    for idx in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score']):
            t0 = df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score'] / MAX_HDRS
            t1 = df_model.loc[idx, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        elif pd.notna(df_model.loc[idx, 'Questionnary2_CDI_T0_Score']):
            t0 = (df_model.loc[idx, 'Questionnary2_CDI_T0_Score'] - 40) / (MAX_CDI - 40)
            t1 = (df_model.loc[idx, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
        if pd.notna(t1):
            delta_y = t1 - t0
            if delta_y <= lower_threshold:
                df_model.loc[idx, 'Y_Classe_Evolution_Delta_Y'] = "Better"
            elif delta_y >= upper_threshold:
                df_model.loc[idx, 'Y_Classe_Evolution_Delta_Y'] = "Worse"
            else:
                df_model.loc[idx, 'Y_Classe_Evolution_Delta_Y'] = "Stable"
        else:
            df_model.loc[idx, 'Y_Classe_Evolution_Delta_Y'] = np.nan 
    return df_model


# In[15]:


# Y = Delta_HDRS - Regression
def standardized_delta_HDRS(df_model, MAX_HDRS):
    df_model.insert(len(df_model.columns), 'Y_Standardized_Delta_HDRS', None)
    for line_index in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[line_index, 'Questionnary1_HDRS_T0_TOT21_Score']):
            t0 = df_model.loc[line_index, 'Questionnary1_HDRS_T0_TOT21_Score'] / MAX_HDRS
            t1 = df_model.loc[line_index, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        if pd.notna(t1):
            delta_y = abs(t1 - t0)  
        else:
            delta_y = np.nan  
        df_model.loc[line_index, 'Y_Standardized_Delta_HDRS'] = delta_y
    return df_model


# In[16]:


# Y = Delta_HDRS - Standardized Binary: worse and better
def standardized_binary_evolution_HDRS(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct):
    df_model.insert(len(df_model.columns), 'Y_Binary_Classe_HDRS', None)
    lower_threshold, upper_threshold = calculate_threshold(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct)
    for idx in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score']):
            t0 = df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score'] / MAX_HDRS
            t1 = df_model.loc[idx, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        if pd.notna(t1):
            delta_y = t1 - t0
            if delta_y <= lower_threshold:
                df_model.loc[idx, 'Y_Binary_Classe_HDRS'] = "Better"
            elif delta_y >= upper_threshold:
                df_model.loc[idx, 'Y_Binary_Classe_HDRS'] = "Worse"
        else:
            df_model.loc[idx, 'Y_Binary_Classe_HDRS'] = np.nan 
    return df_model


# In[17]:


# Y = Delta_HDRS - Standardized Multiclass: worse, stable, better
def standardized_multiclass_evolution_HDRS(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct):
    df_model.insert(len(df_model.columns), 'Y_Classe_Evolution_HDRS', None)
    lower_threshold, upper_threshold = calculate_threshold(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct)
    for idx in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score']):
            t0 = df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score'] / MAX_HDRS
            t1 = df_model.loc[idx, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        if pd.notna(t1):
            delta_y = t1 - t0
            if delta_y <= lower_threshold:
                df_model.loc[idx, 'Y_Classe_Evolution_HDRS'] = "Better"
            elif delta_y >= upper_threshold:
                df_model.loc[idx, 'Y_Classe_Evolution_HDRS'] = "Worse"
            else:
                df_model.loc[idx, 'Y_Classe_Evolution_HDRS'] = "Stable"
        else:
            df_model.loc[idx, 'Y_Classe_Evolution_HDRS'] = np.nan 
    return df_model


# In[18]:


# Y = Delta_CDI - Regression
def standardized_delta_CDI(df_model, MAX_CDI):
    df_model.insert(len(df_model.columns), 'Y_Standardized_Delta_CDI', None)
    for line_index in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[line_index, 'Questionnary2_CDI_T0_Score']):
            t0 = (df_model.loc[line_index, 'Questionnary2_CDI_T0_Score'] - 40) / (MAX_CDI - 40)
            t1 = (df_model.loc[line_index, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
        if pd.notna(t1):
            delta_y = abs(t1 - t0)  
        else:
            delta_y = np.nan  
        df_model.loc[line_index, 'Y_Standardized_Delta_CDI'] = delta_y
    return df_model


# In[19]:


# Y = Delta_CDI - Standardized Binary: worse and better
def standardized_binary_evolution_CDI(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct):
    df_model.insert(len(df_model.columns), 'Y_Binary_Classe_CDI', None)
    lower_threshold, upper_threshold = calculate_threshold(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct)
    for idx in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[idx, 'Questionnary2_CDI_T0_Score']):
            t0 = (df_model.loc[idx, 'Questionnary2_CDI_T0_Score'] - 40) / (MAX_CDI - 40)
            t1 = (df_model.loc[idx, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
        if pd.notna(t1):
            delta_y = t1 - t0 
            if delta_y <= lower_threshold:
                df_model.loc[idx, 'Y_Binary_Classe_CDI'] = "Better"
            elif delta_y >= upper_threshold:
                df_model.loc[idx, 'Y_Binary_Classe_CDI'] = "Worse"
        else:
            df_model.loc[idx, 'Y_Binary_Classe_CDI'] = np.nan 
    return df_model


# In[20]:


# Y = Delta_CDI - Standardized Multiclass: worse, stable, better
def standardized_multiclass_evolution_CDI(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct):
    df_model.insert(len(df_model.columns), 'Y_Classe_Evolution_CDI', None)
    lower_threshold, upper_threshold = calculate_threshold(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct)
    for idx in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[idx, 'Questionnary2_CDI_T0_Score']):
            t0 = (df_model.loc[idx, 'Questionnary2_CDI_T0_Score'] - 40) / (MAX_CDI - 40)
            t1 = (df_model.loc[idx, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
        if pd.notna(t1):
            delta_y = t1 - t0
            if delta_y <= lower_threshold:
                df_model.loc[idx, 'Y_Classe_Evolution_CDI'] = "Better"
            elif delta_y >= upper_threshold:
                df_model.loc[idx, 'Y_Classe_Evolution_CDI'] = "Worse"
            else:
                df_model.loc[idx, 'Y_Classe_Evolution_CDI'] = "Stable"
        else:
            df_model.loc[idx, 'Y_Classe_Evolution_CDI'] = np.nan 
    return df_model


# In[21]:


# Raw Score Multiclass: worse, stable, better
def multiclass_raw_evolution(df_model, MAX_HDRS, MAX_CDI):
    df_model.insert(len(df_model.columns), 'Y_Classe_Raw_Evolution', None)
    for idx in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score']):
            t0 = df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score'] / MAX_HDRS
            t1 = df_model.loc[idx, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        elif pd.notna(df_model.loc[idx, 'Questionnary2_CDI_T0_Score']):
            t0 = (df_model.loc[idx, 'Questionnary2_CDI_T0_Score'] - 40) / (MAX_CDI - 40)
            t1 = (df_model.loc[idx, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
        if pd.notna(t0) and pd.notna(t1):
            if t1 < t0:
                df_model.loc[idx, 'Y_Classe_Raw_Evolution'] = "Better"
            elif t1 > t0:
                df_model.loc[idx, 'Y_Classe_Raw_Evolution'] = "Worse"
            else:
                df_model.loc[idx, 'Y_Classe_Raw_Evolution'] = "Stable"
    return df_model


# In[22]:


# Get t0 and t1 standardized 
# Y = Delta_y (HDRS and CDI) - Regression
def standardized_t0_t1(df_model, MAX_HDRS, MAX_CDI):
    for line_index in df_model.index:
        if pd.notna(df_model.loc[line_index, 'Questionnary1_HDRS_T0_TOT21_Score']):
            df_model.loc[line_index, 'Questionnary1_HDRS_T0_TOT21_Score'] = df_model.loc[line_index, 'Questionnary1_HDRS_T0_TOT21_Score'] / MAX_HDRS
            df_model.loc[line_index, 'Questionnary1_HDRS_T1_TOT21_Score'] = df_model.loc[line_index, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        elif pd.notna(df_model.loc[line_index, 'Questionnary2_CDI_T0_Score']):
            df_model.loc[line_index, 'Questionnary2_CDI_T0_Score'] = (df_model.loc[line_index, 'Questionnary2_CDI_T0_Score'] - 40) / (MAX_CDI - 40)
            df_model.loc[line_index, 'Questionnary2_CDI_T1_Score'] = (df_model.loc[line_index, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40) 
    return df_model


# In[23]:


# Get standardized delta HDRS/CDI after calculating delta
def standardized_binary_evolution_delta_value(df_model, MAX_HDRS, MAX_CDI, lower_pct, upper_pct, deltas):
    df_model.insert(len(df_model.columns), 'Y_Binary_Classe_Delta_Y_Value', np.nan)
    df_model.insert(len(df_model.columns), 'Y_Binary_Classe_Delta_Y', np.nan)
    lower_threshold = float(np.percentile(deltas, lower_pct))
    upper_threshold = float(np.percentile(deltas, upper_pct))
    for idx in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score']):
            t0 = df_model.loc[idx, 'Questionnary1_HDRS_T0_TOT21_Score'] / MAX_HDRS
            t1 = df_model.loc[idx, 'Questionnary1_HDRS_T1_TOT21_Score'] / MAX_HDRS
        elif pd.notna(df_model.loc[idx, 'Questionnary2_CDI_T0_Score']):
            t0 = (df_model.loc[idx, 'Questionnary2_CDI_T0_Score'] - 40) / (MAX_CDI - 40)
            t1 = (df_model.loc[idx, 'Questionnary2_CDI_T1_Score'] - 40) / (MAX_CDI - 40)
        if pd.notna(t0) and pd.notna(t1):
            delta_y = t1 - t0
            df_model.loc[idx, 'Y_Binary_Classe_Delta_Y_Value'] = float(delta_y)
            if delta_y <= lower_threshold:
                df_model.loc[idx, 'Y_Binary_Classe_Delta_Y'] = "Better"
            elif delta_y >= upper_threshold:
                df_model.loc[idx, 'Y_Binary_Classe_Delta_Y'] = "Worse"
            else:
                df_model.loc[idx, 'Y_Binary_Classe_Delta_Y'] = np.nan
        else:
            df_model.loc[idx, 'Y_Binary_Classe_Delta_Y_Value'] = np.nan 
            df_model.loc[idx, 'Y_Binary_Classe_Delta_Y'] = np.nan 
    return df_model


# In[24]:


# HQ25: Z-score 
def zscore_TO_T1_TOT_HQ25(df_model):
    if 'Zscore_Delta_HQ25' not in df_model.columns:
        df_model['Zscore_Delta_HQ25'] = np.nan
    
    mean_HQ_T0 = df_model['Questionnary5_HQ_25_T0_TOT_Score'].mean()
    std_HQ_T0 = df_model['Questionnary5_HQ_25_T0_TOT_Score'].std()
    mean_HQ_T1 = df_model['Questionnary5_HQ_25_T1_TOT_Score'].mean()
    std_HQ_T1 = df_model['Questionnary5_HQ_25_T1_TOT_Score'].std()

    df_model['Questionnary5_HQ_25_T0_TOT_Score'] = (df_model['Questionnary5_HQ_25_T0_TOT_Score'] - mean_HQ_T0) / std_HQ_T0
    df_model['Questionnary5_HQ_25_T1_TOT_Score'] = (df_model['Questionnary5_HQ_25_T1_TOT_Score'] - mean_HQ_T1) / std_HQ_T1

    for line_index in df_model.index:
        t0 = t1 = np.nan
        if pd.notna(df_model.loc[line_index, 'Questionnary5_HQ_25_T0_TOT_Score']) or pd.notna(df_model.loc[line_index, 'Questionnary5_HQ_25_T1_TOT_Score']):
            t0 = df_model.loc[line_index, 'Questionnary5_HQ_25_T0_TOT_Score']
            t1 = df_model.loc[line_index, 'Questionnary5_HQ_25_T1_TOT_Score']
            if pd.notna(t1):
                delta_hq = abs(t0 - t1) # improvement
            else:
                delta_hq = np.nan  
        df_model.loc[line_index, 'Zscore_Delta_HQ25'] = delta_hq      
    return df_model


# In[25]:


# Alexithymia: Z-score for T0 (TAS and AQC)
def zscore_TO_TOT_Alexithymia(df_model):
    if 'Alexithymia_T0' not in df_model.columns:
        df_model['Alexithymia_T0'] = np.nan
    # Young Adults
    mean_TAS_TO = df_model['Questionnary3_TAS_20_T0_TOT_Score'].mean()
    std_dev_TAS_TO = df_model['Questionnary3_TAS_20_T0_TOT_Score'].std()
    df_model['Questionnary3_TAS_20_T0_TOT_Score'] = (df_model['Questionnary3_TAS_20_T0_TOT_Score'] - mean_TAS_TO) / std_dev_TAS_TO
    # Adolescents
    mean_AQC_TO = df_model['Questionnary4_AQC_T0_TOT_Score'].mean()
    std_dev_AQC_TO = df_model['Questionnary4_AQC_T0_TOT_Score'].std()
    df_model['Questionnary4_AQC_T0_TOT_Score'] = (df_model['Questionnary4_AQC_T0_TOT_Score'] - mean_AQC_TO) / std_dev_AQC_TO
    for line_index in df_model.index:
        if pd.notna(df_model.loc[line_index, 'Questionnary3_TAS_20_T0_TOT_Score']):
            df_model.loc[line_index, 'Alexithymia_T0'] = df_model.loc[line_index, 'Questionnary3_TAS_20_T0_TOT_Score']
        elif pd.notna(df_model.loc[line_index, 'Questionnary4_AQC_T0_TOT_Score']):
            df_model.loc[line_index, 'Alexithymia_T0'] = df_model.loc[line_index, 'Questionnary4_AQC_T0_TOT_Score']
        else:
            df_model.loc[line_index, 'Alexithymia_T0'] = np.nan      
    return df_model


# In[26]:


# Alexithymia: Z-score subscales F1, F2 and F3
def zscore_T0_Alexithymia_F123(df_model):
    df_model = df_model.copy()
    tas_cols = [
        "Questionnary3_TAS_20_T0_F1_Score",
        "Questionnary3_TAS_20_T0_F2_Score",
        "Questionnary3_TAS_20_T0_F3_Score"]
    aqc_cols = [
        "Questionnary4_AQC_T0_F1_Score",
        "Questionnary4_AQC_T0_F2_Score",
        "Questionnary4_AQC_T0_F3_Score"]
    out_cols = [
        "Alex_F1_T0",
        "Alex_F2_T0",
        "Alex_F3_T0"]
    for tas_col, aqc_col, out_col in zip(tas_cols, aqc_cols, out_cols):
        tas_mean = df_model[tas_col].mean()
        tas_std = df_model[tas_col].std()
        aqc_mean = df_model[aqc_col].mean()
        aqc_std = df_model[aqc_col].std()
        tas_z = (df_model[tas_col] - tas_mean) / tas_std
        aqc_z = (df_model[aqc_col] - aqc_mean) / aqc_std
        df_model[out_col] = np.nan
        df_model.loc[tas_z.notna(), out_col] = tas_z[tas_z.notna()]
        df_model.loc[df_model[out_col].isna() & aqc_z.notna(), out_col] = aqc_z[df_model[out_col].isna() & aqc_z.notna()]
    return df_model

