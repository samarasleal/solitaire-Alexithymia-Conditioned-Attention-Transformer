#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
# jupyter nbconvert --to script DataPreprocessing_melt.ipynb


# ___

# Melt the dfs - MFCCs

# In[2]:


def get_mfccs_per_segment_mean(df, meta_cols):
    mfcc_cols = [col for col in df.columns if col.startswith('MFCC_')]
    df_melt = pd.melt(
        df, id_vars=meta_cols, value_vars=mfcc_cols,
        var_name='MFCC_feature', value_name='value'
    )
    df_melt[['Session', 'Segment', 'Coefficient']] = df_melt['MFCC_feature'] \
        .str.extract(r'MFCC_Session(\d+)_Segment(\d+)_Coefficient(\d+)', expand=True)
    df_melt = df_melt.dropna(subset=['Session', 'Segment', 'Coefficient'])
    df_melt['Session'] = df_melt['Session'].astype(int)
    df_melt['Segment'] = df_melt['Segment'].astype(int)
    df_melt['Coefficient'] = df_melt['Coefficient'].astype(int)
    df_melt['MFCC_label'] = 'coefficient' + df_melt['Coefficient'].astype(str)
    if len(meta_cols) <= 2:
        index_cols = ['Patient_ID', 'Session', 'Segment', meta_cols[1]]
    else:
        index_cols = ['Patient_ID', 'Session', 'Segment'] + meta_cols[1:]
    df_long = df_melt.pivot_table(
        index=index_cols, columns='MFCC_label', values='value', dropna=True
    ).reset_index()
    df_long.columns.name = None
    coef_cols = sorted(
        [col for col in df_long.columns if 'coefficient' in col],
        key=lambda x: int(x.split('coefficient')[1])
    )
    new_order = ['Patient_ID', 'Session', 'Segment']
    if len(meta_cols) > 2:
        new_order += meta_cols[1:-1]
    new_order += coef_cols
    if len(meta_cols) >= 2:
        new_order.append(meta_cols[-1])
    return df_long[new_order]


# In[3]:


def get_mfccs_per_segment_mean_std(df, meta_cols):
    mfcc_cols = [col for col in df.columns if col.startswith('MFCC_')]
    df_melt = pd.melt(df, id_vars=meta_cols, value_vars=mfcc_cols,
                      var_name='MFCC_feature', value_name='value')
    df_melt[['Session', 'Segment', 'StatType', 'Coefficient']] = df_melt['MFCC_feature']\
        .str.extract(r'MFCC_Session(\d+)_Segment(\d+)_(Mean|Std)Coefficient(\d+)', expand=True)
    df_melt['Session'] = df_melt['Session'].astype(int)
    df_melt['Segment'] = df_melt['Segment'].astype(int)
    df_melt['Coefficient'] = df_melt['Coefficient'].astype(int)

    df_melt['MFCC_label'] = df_melt['StatType'].str.lower() + '_coefficient' + df_melt['Coefficient'].astype(str)

    if len(meta_cols) <= 2:
        index_cols = ['Patient_ID', 'Session', 'Segment', meta_cols[1]]
    else:
        index_cols = ['Patient_ID', 'Session', 'Segment'] + meta_cols[1:]

    # Pivot
    df_long = df_melt.pivot_table(index=index_cols, columns='MFCC_label', values='value', dropna=True).reset_index()
    df_long.columns.name = None

    coef_cols = sorted([col for col in df_long.columns if 'coefficient' in col], key=lambda x: (x.split('_')[0], int(x.split('coefficient')[1])))
    
    new_order = ['Patient_ID', 'Session', 'Segment']
    if len(meta_cols) > 2:
        new_order += meta_cols[1:-1]
    new_order += coef_cols
    if len(meta_cols) >= 2:
        new_order.append(meta_cols[-1])
    df_long = df_long[new_order]
    return df_long


# ___

# Melt the dfs - Embeddings

# In[4]:


def get_embeddings_per_segment_mean_2D(df, meta_cols):
    embedding_cols = [col for col in df.columns if col.startswith('Embeddings_Session')] 
    df_melt = pd.melt(df, id_vars=meta_cols, value_vars=embedding_cols, var_name='Embedding_feature', value_name='value')
    df_melt[['Session', 'Segment']] = df_melt['Embedding_feature']\
        .str.extract(r'Embeddings_Session(\d+)_Segment(\d+)', expand=True)   
    df_melt['Session'] = df_melt['Session'].astype(int)
    df_melt['Segment'] = df_melt['Segment'].astype(int)
    df_melt['Embedding_label'] = 'embedding_mean'
    if len(meta_cols) <= 2:
        index_cols = ['Patient_ID', 'Session', 'Segment', meta_cols[1]]
    else:
        index_cols = ['Patient_ID', 'Session', 'Segment'] + meta_cols[1:]
    df_long = df_melt.pivot_table(index=index_cols, columns='Embedding_label', values='value', dropna=True).reset_index()
    df_long.columns.name = None  
    emb_cols = [col for col in df_long.columns if col.startswith('embedding_')]
    new_order = ['Patient_ID', 'Session', 'Segment']
    if len(meta_cols) > 2:
        new_order += meta_cols[1:-1]
    new_order += emb_cols
    if len(meta_cols) >= 2:
        new_order.append(meta_cols[-1])
    df_long = df_long[new_order]
    return df_long


# In[5]:


def get_embeddings_per_segment_mean_std_2D(df, meta_cols):
    embedding_cols = [col for col in df.columns if col.startswith('Embeddings_Session')]
    df_melt = pd.melt(df, id_vars=meta_cols, value_vars=embedding_cols,
                      var_name='Embedding_feature', value_name='value')    
    df_melt[['Session', 'StatType', 'Segment']] = df_melt['Embedding_feature']\
        .str.extract(r'Embeddings_Session(\d+)_(Mean|Std)Segment(\d+)', expand=True)   
    df_melt['Session'] = df_melt['Session'].astype(int)
    df_melt['Segment'] = df_melt['Segment'].astype(int)    
    df_melt['Embedding_label'] = 'embedding_' + df_melt['StatType'].str.lower()    
    if len(meta_cols) <= 2:
        index_cols = ['Patient_ID', 'Session', 'Segment', meta_cols[1]]
    else:
        index_cols = ['Patient_ID', 'Session', 'Segment'] + meta_cols[1:]
    df_long = df_melt.pivot_table(index=index_cols, columns='Embedding_label', values='value', dropna=True).reset_index()
    df_long.columns.name = None
    emb_cols = [col for col in df_long.columns if col.startswith('embedding_')]
    new_order = ['Patient_ID', 'Session', 'Segment']
    if len(meta_cols) > 2:
        new_order += meta_cols[1:-1]
    new_order += emb_cols
    if len(meta_cols) >= 2:
        new_order.append(meta_cols[-1])
    df_long = df_long[new_order]
    return df_long


# In[6]:


def get_embeddings_per_segment_mean_3D(df, meta_cols):
    embedding_cols = [col for col in df.columns if col.startswith('Embeddings_Session')] 
    df_melt = pd.melt(df, id_vars=meta_cols, value_vars=embedding_cols, var_name='Embedding_feature', value_name='value')
    df_melt[['Session', 'Segment']] = df_melt['Embedding_feature']\
        .str.extract(r'Embeddings_Session(\d+)_Segment(\d+)', expand=True)   
    df_melt['Session'] = df_melt['Session'].astype(int)
    df_melt['Segment'] = df_melt['Segment'].astype(int)
    df_melt['Embedding_label'] = 'Embedding_npy'
    if len(meta_cols) <= 2:
        index_cols = ['Patient_ID', 'Session', 'Segment', meta_cols[1]]
    else:
        index_cols = ['Patient_ID', 'Session', 'Segment'] + meta_cols[1:]
    df_long = df_melt.pivot_table(index=index_cols, columns='Embedding_label', values='value', aggfunc='first', dropna=True).reset_index()
    df_long.columns.name = None  
    emb_cols = [col for col in df_long.columns if col.startswith('Embedding_')]
    new_order = ['Patient_ID', 'Session', 'Segment']
    if len(meta_cols) > 2:
        new_order += meta_cols[1:-1]
    new_order += emb_cols
    if len(meta_cols) >= 2:
        new_order.append(meta_cols[-1])
    df_long = df_long[new_order]
    return df_long


# ___

# Melt the dfs - F0

# In[ ]:


def get_f0_per_segment_mean_std(df, meta_cols):
    f0_cols = [col for col in df.columns if col.startswith('F0_Session')]
    df_melt = pd.melt(df, id_vars=meta_cols, value_vars=f0_cols, var_name='F0_feature', value_name='value')
    df_melt[['Session', 'Segment', 'StatType']] = df_melt['F0_feature']\
        .str.extract(r'F0_Session(\d+)_Segment(\d+)_(mean|std)', expand=True)
    df_melt['Session'] = df_melt['Session'].astype(int)
    df_melt['Segment'] = df_melt['Segment'].astype(int)
    # labels: mean / std
    df_melt['F0_label'] = df_melt['StatType'].str.lower()
    # --- index ---
    if len(meta_cols) <= 2:
        index_cols = ['Patient_ID', 'Session', 'Segment', meta_cols[1]]
    else:
        index_cols = ['Patient_ID', 'Session', 'Segment'] + meta_cols[1:]
    # --- pivot: generate columns mean, std ---
    df_long = df_melt.pivot_table(
        index=index_cols,
        columns='F0_label',
        values='value',
        dropna=True
    ).reset_index()
    df_long.columns.name = None
    new_order = ['Patient_ID', 'Session', 'Segment']
    if len(meta_cols) > 2:
        new_order += meta_cols[1:-1] 
    if 'mean' in df_long.columns:
        new_order.append('mean')
    if 'std' in df_long.columns:
        new_order.append('std')
    if len(meta_cols) >= 2:
        new_order.append(meta_cols[-1])
    return df_long[new_order]


# ___

# Melt the dfs - Combination of features 

# In[7]:


def get_mfccs_embeddings_per_segment_mean(df, meta_cols):
    key = 'Patient_ID'
    meta_extra = [c for c in meta_cols if c != key]
    # ---------- MFCCs ----------
    mfcc_cols = [c for c in df.columns if c.startswith('MFCC_')]
    dfm = pd.melt(df, id_vars=[key], value_vars=mfcc_cols, var_name='MFCC_feature', value_name='value')
    rex = r'MFCC_Session(\d+)_Segment(\d+)_(Mean|Std)Coefficient(\d+)'
    dfm[['Session', 'Segment', 'StatType', 'Coefficient']] = (dfm['MFCC_feature'].str.extract(rex, expand=True))
    dfm['Session'] = dfm['Session'].astype(int)
    dfm['Segment'] = dfm['Segment'].astype(int)
    dfm['Coefficient'] = dfm['Coefficient'].astype(int)
    # Only mean
    dfm = dfm[dfm['StatType'] == 'Mean'].copy()
    dfm['MFCC_label'] = ('mean_coefficient' + dfm['Coefficient'].astype(str))
    mfcc_wide = (
        dfm.pivot_table(
            index=[key, 'Session', 'Segment'],
            columns='MFCC_label',
            values='value',
            dropna=True
        )
        .reset_index())
    mfcc_wide.columns.name = None
    coef_cols = sorted([c for c in mfcc_wide.columns if 'coefficient' in c], 
                       key=lambda x: int(x.split('coefficient')[1]))  
    # ---------- Embeddings ----------
    emb_cols = [c for c in df.columns if c.startswith('Embeddings_Session')]
    dfe = pd.melt(df, id_vars=[key], value_vars=emb_cols,
                  var_name='Feature', value_name='Embedding_npy')
    dfe[['Session', 'Segment']] = (dfe['Feature'].str.extract(r'Embeddings_Session(\d+)_Segment(\d+)').astype(int))
    dfe = (dfe[[key, 'Session', 'Segment', 'Embedding_npy']].drop_duplicates([key, 'Session', 'Segment']))
    # ---------- Merge MFCC + Embedding ----------
    out = pd.merge(mfcc_wide, dfe, on=[key, 'Session', 'Segment'], how='outer')
    if meta_extra:
        meta_df = df[[key] + meta_extra].drop_duplicates(key)
        out = out.merge(meta_df, on=key, how='left')
    base = [key, 'Session', 'Segment']
    final_order = base + meta_extra + [c for c in coef_cols if c in out.columns] + ['Embedding_npy']
    final_order += [c for c in out.columns if c not in final_order]
    return out[final_order]


# In[8]:


def get_mfccs_embeddings_per_segment_mean_std(df, meta_cols):
    key = 'Patient_ID'
    meta_extra = [c for c in meta_cols if c != key]
    # ---------- MFCCs ----------
    mfcc_cols = [c for c in df.columns if c.startswith('MFCC_')]
    dfm = pd.melt(df, id_vars=[key], value_vars=mfcc_cols, var_name='MFCC_feature', value_name='value')
    rex = r'MFCC_Session(\d+)_Segment(\d+)_(Mean|Std)Coefficient(\d+)'
    dfm[['Session', 'Segment', 'StatType', 'Coefficient']] = (dfm['MFCC_feature'].str.extract(rex, expand=True))
    dfm['Session'] = dfm['Session'].astype(int)
    dfm['Segment'] = dfm['Segment'].astype(int)
    dfm['Coefficient'] = dfm['Coefficient'].astype(int)
    dfm['MFCC_label'] = (dfm['StatType'].str.lower() + '_coefficient' + dfm['Coefficient'].astype(str))
    mfcc_wide = (
        dfm.pivot_table(
            index=[key, 'Session', 'Segment'],
            columns='MFCC_label',
            values='value',
            dropna=True
        )
        .reset_index())
    mfcc_wide.columns.name = None
    coef_cols = sorted([c for c in mfcc_wide.columns if 'coefficient' in c], key=lambda x: (x.split('_')[0], int(x.split('coefficient')[1]))) 
    # ---------- Embeddings ----------
    emb_cols = [c for c in df.columns if c.startswith('Embeddings_Session')]
    dfe = pd.melt(df, id_vars=[key], value_vars=emb_cols,
                  var_name='Feature', value_name='Embedding_npy')
    dfe[['Session', 'Segment']] = (dfe['Feature'].str.extract(r'Embeddings_Session(\d+)_Segment(\d+)').astype(int))
    dfe = (dfe[[key, 'Session', 'Segment', 'Embedding_npy']].drop_duplicates([key, 'Session', 'Segment']))
    # ---------- Merge MFCC + Embedding ----------
    out = pd.merge(mfcc_wide, dfe, on=[key, 'Session', 'Segment'], how='outer')
    if meta_extra:
        meta_df = df[[key] + meta_extra].drop_duplicates(key)
        out = out.merge(meta_df, on=key, how='left')
    base = [key, 'Session', 'Segment']
    final_order = base + meta_extra + [c for c in coef_cols if c in out.columns] + ['Embedding_npy']
    final_order += [c for c in out.columns if c not in final_order]
    return out[final_order]


# In[ ]:


def get_f0_mfcc_per_segment_mean(df, meta_cols):
    key = 'Patient_ID'
    meta_extra = [c for c in meta_cols if c != key]
    # ---------- MFCCs ----------
    mfcc_cols = [c for c in df.columns if c.startswith('MFCC_')]
    dfm = pd.melt(df, id_vars=[key], value_vars=mfcc_cols, var_name='MFCC_feature', value_name='value')
    rex = r'MFCC_Session(\d+)_Segment(\d+)_Coefficient(\d+)'
    dfm[['Session', 'Segment', 'Coefficient']] = dfm['MFCC_feature'].str.extract(rex, expand=True)
    dfm = dfm.dropna(subset=['Session', 'Segment', 'Coefficient'])
    dfm['Session'] = dfm['Session'].astype(int)
    dfm['Segment'] = dfm['Segment'].astype(int)
    dfm['Coefficient'] = dfm['Coefficient'].astype(int)
    dfm['MFCC_label'] = 'mean_coefficient' + dfm['Coefficient'].astype(str)
    mfcc_wide = (dfm.pivot_table(index=[key, 'Session', 'Segment'], columns='MFCC_label', values='value', dropna=True).reset_index())
    mfcc_wide.columns.name = None
    coef_cols = sorted([c for c in mfcc_wide.columns if 'coefficient' in c], key=lambda x: int(x.split('coefficient')[1]))
    
    # ---------- F0 ----------
    f0_cols = [c for c in df.columns if c.startswith('F0_')]
    if f0_cols:
        dff = pd.melt(df, id_vars=[key], value_vars=f0_cols, var_name='F0_feature', value_name='value')
        rexf0 = r'F0_Session(\d+)_Segment(\d+)_(mean|std)'
        dff[['Session', 'Segment', 'StatType']] = dff['F0_feature'].str.extract(rexf0, expand=True)
        dff = dff.dropna(subset=['Session', 'Segment', 'StatType'])
        dff['Session'] = dff['Session'].astype(int)
        dff['Segment'] = dff['Segment'].astype(int)
        dff['StatType'] = dff['StatType'].str.lower()
        dff['F0_label'] = 'F0_' + dff['StatType']
        f0_wide = (
            dff.pivot_table(
                index=[key, 'Session', 'Segment'],
                columns='F0_label',
                values='value',
                aggfunc='mean'
            ).reset_index())
        f0_wide.columns.name = None
    else:
        f0_wide = None
    # ---------- Merge MFCC + F0 ----------
    out = mfcc_wide.copy()
    if f0_wide is not None:
        out = out.merge(f0_wide, on=[key, 'Session', 'Segment'], how='left')
    # ---------- Meta ----------
    if meta_extra:
        meta_df = df[[key] + meta_extra].drop_duplicates(key)
        out = out.merge(meta_df, on=key, how='left')
    base = [key, 'Session', 'Segment']
    f0_out_cols = [c for c in ['F0_mean', 'F0_std'] if c in out.columns]
    final_order = base + meta_extra + [c for c in coef_cols if c in out.columns] + f0_out_cols
    final_order += [c for c in out.columns if c not in final_order]
    return out[final_order]


# In[ ]:


def get_f0_mfccs_embeddings_per_segment_mean(df, meta_cols):
    key = 'Patient_ID'
    meta_extra = [c for c in meta_cols if c != key]
    # ---------- MFCCs ----------
    mfcc_cols = [c for c in df.columns if c.startswith('MFCC_')]
    dfm = pd.melt(df, id_vars=[key], value_vars=mfcc_cols, var_name='MFCC_feature', value_name='value')
    rex = r'MFCC_Session(\d+)_Segment(\d+)_Coefficient(\d+)'
    dfm[['Session', 'Segment', 'Coefficient']] = dfm['MFCC_feature'].str.extract(rex, expand=True)
    dfm = dfm.dropna(subset=['Session', 'Segment', 'Coefficient'])
    dfm['Session'] = dfm['Session'].astype(int)
    dfm['Segment'] = dfm['Segment'].astype(int)
    dfm['Coefficient'] = dfm['Coefficient'].astype(int)
    dfm['MFCC_label'] = 'mean_coefficient' + dfm['Coefficient'].astype(str)
    mfcc_wide = (dfm.pivot_table(index=[key, 'Session', 'Segment'], columns='MFCC_label', values='value', dropna=True).reset_index())
    mfcc_wide.columns.name = None
    coef_cols = sorted([c for c in mfcc_wide.columns if 'coefficient' in c], key=lambda x: int(x.split('coefficient')[1]))   
    
    # ---------- Embeddings ----------
    emb_cols = [c for c in df.columns if c.startswith('Embeddings_Session')]
    dfe = pd.melt(
        df,
        id_vars=[key],
        value_vars=emb_cols,
        var_name='Feature',
        value_name='Embedding_npy'
    )
    dfe[['Session', 'Segment']] = (dfe['Feature'].str.extract(r'Embeddings_Session(\d+)_Segment(\d+)', expand=True).astype(int))
    dfe = dfe[[key, 'Session', 'Segment', 'Embedding_npy']].drop_duplicates([key, 'Session', 'Segment'])
    
    # ---------- F0 (mean + std) ----------
    f0_cols = [c for c in df.columns if c.startswith('F0_Session')]
    if f0_cols:
        dff0 = pd.melt(df, id_vars=[key], value_vars=f0_cols, var_name='F0_feature', value_name='value')
        re_f0 = r'F0_Session(\d+)_Segment(\d+)_(mean|std)'
        dff0[['Session', 'Segment', 'StatType']] = (dff0['F0_feature'].str.extract(re_f0, expand=True))
        dff0['Session'] = dff0['Session'].astype(int)
        dff0['Segment'] = dff0['Segment'].astype(int)
        dff0['F0_label'] = 'F0_' + dff0['StatType'].str.lower()  # F0_mean / F0_std
        f0_wide = (
            dff0.pivot_table(
                index=[key, 'Session', 'Segment'],
                columns='F0_label',
                values='value',
                dropna=True
            ).reset_index()
        )
        f0_wide.columns.name = None
        f0_feat_cols = [c for c in f0_wide.columns if c.startswith('F0_')]
    else:
        f0_wide = None
        f0_feat_cols = []
        
    # ---------- Merge MFCC + Embedding + F0 ----------
    out = mfcc_wide.copy()
    # add embeddings
    out = pd.merge(out, dfe, on=[key, 'Session', 'Segment'], how='outer')
    # add F0
    if f0_wide is not None:
        out = pd.merge(out, f0_wide, on=[key, 'Session', 'Segment'], how='outer')
    # meta
    if meta_extra:
        meta_df = df[[key] + meta_extra].drop_duplicates(key)
        out = out.merge(meta_df, on=key, how='left')
    base = [key, 'Session', 'Segment']
    final_order = base + meta_extra
    # MFCCs 
    final_order += [c for c in coef_cols if c in out.columns]
    # F0 (mean/std)
    final_order += [c for c in f0_feat_cols if c in out.columns]
    # Embedding
    if 'Embedding_npy' in out.columns:
        final_order.append('Embedding_npy')
    # Other columns
    final_order += [c for c in out.columns if c not in final_order]
    return out[final_order]


# In[ ]:


def get_f0_mfccs_embeddings_per_session_mean_std(df: pd.DataFrame, meta_cols):
    key = "Patient_ID"
    meta_extra = [c for c in meta_cols if c != key]

    # 1) MFCCs (segment-level -> session-level)
    mfcc_cols = [c for c in df.columns if c.startswith("MFCC_")]
    dfm = pd.melt(df, id_vars=[key], value_vars=mfcc_cols, var_name="MFCC_feature", value_name="value")

    rex = r"MFCC_Session(\d+)_Segment(\d+)_Coefficient(\d+)"
    dfm[["Session", "Segment", "Coefficient"]] = dfm["MFCC_feature"].str.extract(rex, expand=True)
    dfm = dfm.dropna(subset=["Session", "Segment", "Coefficient"])
    dfm["Session"] = dfm["Session"].astype(int)
    dfm["Segment"] = dfm["Segment"].astype(int)
    dfm["Coefficient"] = dfm["Coefficient"].astype(int)

    # Create a stable label per coefficient at segment level
    dfm["MFCC_label"] = "MFCC" + dfm["Coefficient"].astype(str)

    # Segment-level wide: (Patient, Session, Segment) x MFCC1..MFCC13
    mfcc_seg = (
        dfm.pivot_table(
            index=[key, "Session", "Segment"],
            columns="MFCC_label",
            values="value",
            dropna=True,
            aggfunc="mean",
        )
        .reset_index()
    )
    mfcc_seg.columns.name = None
    mfcc_feat_cols = sorted(
        [c for c in mfcc_seg.columns if c.startswith("MFCC")],
        key=lambda x: int(x.replace("MFCC", "")),
    )

    # Session-level mean/std across segments
    mfcc_session = mfcc_seg.groupby([key, "Session"], as_index=False)[mfcc_feat_cols].agg(["mean", "std"])
    # Flatten columns: MFCC5_mean / MFCC5_std
    mfcc_session.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in mfcc_session.columns.to_flat_index()
    ]
    mfcc_session = mfcc_session.rename(columns={"Patient_ID_": key, "Session_": "Session"})

    # 2) Embeddings (keep .npy per segment; aggregate list per session)
    emb_cols = [c for c in df.columns if c.startswith("Embeddings_Session")]
    if emb_cols:
        dfe = pd.melt(df, id_vars=[key], value_vars=emb_cols, var_name="Feature", value_name="Embedding_npy")

        dfe[["Session", "Segment"]] = (dfe["Feature"].str.extract(r"Embeddings_Session(\d+)_Segment(\d+)", expand=True).astype(int))
        dfe = dfe[[key, "Session", "Segment", "Embedding_npy"]].drop_duplicates([key, "Session", "Segment"])

        # Keep list of npy paths/handles per session. Do NOT average here.
        emb_session = (
            dfe.sort_values([key, "Session", "Segment"])
               .groupby([key, "Session"], as_index=False)["Embedding_npy"]
               .apply(list)
               .rename(columns={"Embedding_npy": "Embedding_npy_list"})
        )
    else:
        emb_session = None

    # 3) F0 (segment-level -> session-level)
    f0_cols = [c for c in df.columns if c.startswith("F0_Session")]
    if f0_cols:
        dff0 = pd.melt(df, id_vars=[key], value_vars=f0_cols, var_name="F0_feature", value_name="value")

        re_f0 = r"F0_Session(\d+)_Segment(\d+)_(mean|std)"
        dff0[["Session", "Segment", "StatType"]] = dff0["F0_feature"].str.extract(re_f0, expand=True)
        dff0 = dff0.dropna(subset=["Session", "Segment", "StatType"])
        dff0["Session"] = dff0["Session"].astype(int)
        dff0["Segment"] = dff0["Segment"].astype(int)

        dff0["F0_label"] = "F0_" + dff0["StatType"].str.lower()  # F0_mean / F0_std

        # Segment-level wide: (Patient, Session, Segment) x (F0_mean, F0_std)
        f0_seg = (
            dff0.pivot_table(
                index=[key, "Session", "Segment"],
                columns="F0_label",
                values="value",
                dropna=True,
                aggfunc="mean",
            )
            .reset_index()
        )
        f0_seg.columns.name = None

        f0_feat_cols = [c for c in f0_seg.columns if c.startswith("F0_")]

        # Session-level MEAN across segments (keep only F0_mean and F0_std)
        f0_session = (
            f0_seg
            .groupby([key, "Session"], as_index=False)[f0_feat_cols]
            .mean()
        )
    else:
        f0_session = None

    # 4) Merge session-level pieces + metadata
    out = mfcc_session.copy()

    if f0_session is not None:
        out = pd.merge(out, f0_session, on=[key, "Session"], how="left")

    if emb_session is not None:
        out = pd.merge(out, emb_session, on=[key, "Session"], how="left")

    if meta_extra:
        meta_df = df[[key] + meta_extra].drop_duplicates(subset=[key])
        out = out.merge(meta_df, on=key, how="left")

    # 5) Column ordering: Patient_ID | Session | meta | MFCC means/std | F0 | embeddings
    base = [key, "Session"]
    final_order = base + meta_extra

    # MFCC order
    mfcc_mean_cols = [f"{c}_mean" for c in mfcc_feat_cols if f"{c}_mean" in out.columns]
    mfcc_std_cols  = [f"{c}_std"  for c in mfcc_feat_cols if f"{c}_std"  in out.columns]
    final_order += mfcc_mean_cols + mfcc_std_cols

    # F0 order (keep deterministic)
    f0_cols_out = sorted([c for c in out.columns if c.startswith("F0_")])
    final_order += [c for c in f0_cols_out if c in out.columns]

    # Embeddings
    if "Embedding_npy_list" in out.columns:
        final_order.append("Embedding_npy_list")

    # Any remaining columns
    final_order += [c for c in out.columns if c not in final_order]
    return out[final_order]


# ___

# Melt the df - logMel

# In[ ]:


def get_logMel_per_segment_mean_3D(df, meta_cols):
    lolMel_cols = [col for col in df.columns if col.startswith('logMel_Session')]
    df_melt = pd.melt(df, id_vars=meta_cols, value_vars=lolMel_cols, var_name='logMel_feature', value_name='value')
    df_melt[['Session', 'Segment']] = df_melt['logMel_feature']\
        .str.extract(r'logMel_Session(\d+)_Segment(\d+)', expand=True)   
    df_melt['Session'] = df_melt['Session'].astype(int)
    df_melt['Segment'] = df_melt['Segment'].astype(int)
    df_melt['logMel_label'] = 'logMel_npy'
    if len(meta_cols) <= 2:
        index_cols = ['Patient_ID', 'Session', 'Segment', meta_cols[1]]
    else:
        index_cols = ['Patient_ID', 'Session', 'Segment'] + meta_cols[1:]
    df_long = df_melt.pivot_table(index=index_cols, columns='logMel_label', values='value', aggfunc='first', dropna=True).reset_index()
    df_long.columns.name = None  
    emb_cols = [col for col in df_long.columns if col.startswith('logMel_')]
    new_order = ['Patient_ID', 'Session', 'Segment']
    if len(meta_cols) > 2:
        new_order += meta_cols[1:-1]
    new_order += emb_cols
    if len(meta_cols) >= 2:
        new_order.append(meta_cols[-1])
    df_long = df_long[new_order]
    return df_long


# ___

# Rebuild CNN Dataframe

# In[ ]:


def rebuild_cnn_dataframe(metadata_df, fold=0, root="/workspace/app/cnn_segment_embeddings"):
    rows = []
    fold_root = os.path.join(root, f"fold_{fold}")

    for split in ["train", "test"]:
        split_root = os.path.join(fold_root, split)

        for patient_name in os.listdir(split_root):
            patient_dir = os.path.join(split_root, patient_name)
            if not os.path.isdir(patient_dir):
                continue

            for session_name in os.listdir(patient_dir):
                session_dir = os.path.join(patient_dir, session_name)
                if not os.path.isdir(session_dir) or not session_name.startswith("Session"):
                    continue

                patient_id = int(float(patient_name))
                session_id = int(session_name.replace("Session", ""))

                for filename in os.listdir(session_dir):
                    if not filename.startswith("seg_") or not filename.endswith(".npy"):
                        continue

                    segment_index = int(filename[4:-4])
                    rows.append({
                        "Original_Index": segment_index,
                        "Patient_ID": patient_id,
                        "Session": session_id
                    })

    cnn_df = pd.DataFrame(rows).drop_duplicates("Original_Index")
    cnn_df = cnn_df.sort_values("Original_Index").set_index("Original_Index")

    meta_cols = [
        "Patient_ID",
        "Y_Standardized_T1",
        "Y_Binary_Classe_Delta_Y",
        "Questionnary3_TAS_20_T0_TOT_Score",
        "Questionnary4_AQC_T0_TOT_Score"
    ]
    meta_cols = [col for col in meta_cols if col in metadata_df.columns]

    patient_meta = metadata_df[meta_cols].copy()
    patient_meta["Patient_ID"] = patient_meta["Patient_ID"].astype(int)
    patient_meta = patient_meta.drop_duplicates("Patient_ID")

    cnn_df = cnn_df.reset_index().merge(
        patient_meta, on="Patient_ID", how="left", validate="many_to_one"
    ).set_index("Original_Index")

    if cnn_df.index.duplicated().any():
        raise ValueError("Duplicated CNN segment indices were found.")

    return cnn_df

