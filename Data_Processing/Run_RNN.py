#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
import math
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# jupyter nbconvert --to script Run_RNN.ipynb


# ___

# Dataset: patient/session-level

# In[2]:


def get_io_loaders(train_ds, test_ds, BS, seed=42):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, collate_fn=collate_patient_embeddings, generator=generator)
    test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False, collate_fn=collate_patient_embeddings)
    return train_dl, test_dl


# In[3]:


_ARR_CACHE = {}
class SeqNPYDataset_patient(Dataset):
    """
    One item = one patient.
    Each session contains all available segment embeddings.
    Dataset output before collation:
        sessions: list with max_sessions elements
                  each valid element has shape [N_session, D]
        y: [1]
        session_mask: [S]
        alex_y: [1], optional
    DataLoader output after collation:
        x: [B, S, N_max, D]
        y: [B, 1]
        session_mask: [B, S]
        segment_mask: [B, S, N_max]
        alex_y: [B, 1], optional
    """
    def __init__(self, df, npy_col, target_col, alex_col=None, preload_to_ram=False, pool_within_file=True, verbose=True):
        self.df = df.copy()
        self.patient_col = "Patient_ID"
        self.session_col = "Session"
        self.npy_col = npy_col
        self.target_col = target_col
        self.alex_col = alex_col
        self.max_sessions = 8
        self.preload = bool(preload_to_ram)
        self.pool_within_file = bool(pool_within_file)
        self.verbose = verbose
        self.pad_value = 0.0 # Padding for embeddings is zero
        self.patients = sorted(self.df[self.patient_col].dropna().unique())
        if self.preload:
            paths = (self.df[self.npy_col].dropna().astype(str).unique())
            total_mb = 0.0
            for path in paths:
                if path not in _ARR_CACHE:
                    arr = self._load_array_from_disk(path)
                    _ARR_CACHE[path] = arr
                    total_mb += arr.nbytes / (1024 ** 2)
            if self.verbose:
                print(f"Preloaded {len(paths)} embedding files "f"(~{total_mb:.1f} MB new).")

    def __len__(self):
        return len(self.patients)

    def _load_array_from_disk(self, path):
        arr = np.load(path, mmap_mode=None)
        arr = np.asarray(arr, dtype=np.float32)
        # Accept [D] or [T, D]
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected [D] or [T, D], got {arr.shape} in {path}")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # Convert each wav2vec segment from [T, 768] to one segment vector [1, 768].
        # CNN segment embeddings already have shape [1, 128] and remain unchanged.
        # All segment vectors are preserved and later grouped by patient and session.   
        if self.pool_within_file and arr.shape[0] > 1:
            arr = arr.mean(axis=0, keepdims=True)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _load_array(self, path):
        if self.preload and path in _ARR_CACHE:
            return _ARR_CACHE[path]
        arr = self._load_array_from_disk(path)
        if self.preload:
            _ARR_CACHE[path] = arr
        return arr

    def __getitem__(self, idx):
        patient = self.patients[idx]
        df_p = self.df[self.df[self.patient_col] == patient].copy()
        sessions = [None] * self.max_sessions
        session_mask = np.zeros(self.max_sessions, dtype=np.float32)
        for session_number in range(1, self.max_sessions + 1):
            df_s = df_p[df_p[self.session_col] == session_number].copy()
            if df_s.empty:
                continue
            if "Segment" in df_s.columns:
                df_s = df_s.sort_values("Segment")
            paths = (df_s[self.npy_col].dropna().astype(str).tolist())
            if len(paths) == 0:
                continue
            segment_embeddings = []
            for path in paths:
                arr = self._load_array(path)
                segment_embeddings.append(arr)
            # All segment embeddings from this session.
            # CNN:     [N_segments, 128]
            # wav2vec: [N_segments, 768]
            session_arr = np.concatenate(segment_embeddings, axis=0)
            session_arr = np.nan_to_num(session_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            sessions[session_number - 1] = session_arr
            session_mask[session_number - 1] = 1.0
        y_value = float(df_p[self.target_col].iloc[0])
        if not np.isfinite(y_value):
            raise ValueError(f"Non-finite target for patient {patient}")
        y = torch.tensor([y_value], dtype=torch.float32)
        session_mask = torch.from_numpy(session_mask).float()
        if self.alex_col is not None:
            alex_values = (df_p[self.alex_col].dropna().astype(float).unique())
            if len(alex_values) == 0:
                raise ValueError(f"Missing alexithymia value for patient {patient}")
            alex_y = torch.tensor([alex_values[0]], dtype=torch.float32)
            return sessions, y, session_mask, alex_y
        return sessions, y, session_mask


# In[4]:


def collate_patient_embeddings(batch):
    """
    Dynamically pad segment embeddings to the largest number of segments present in the current batch.
    """
    has_alex = len(batch[0]) == 4
    if has_alex:
        sessions_list, y_list, session_mask_list, alex_list = zip(*batch)
    else:
        sessions_list, y_list, session_mask_list = zip(*batch)
    batch_size = len(sessions_list)
    max_sessions = len(sessions_list[0])
    embedding_dim = None
    max_segments = 0
    # Discover embedding dimension and the largest session
    # only inside the current batch.
    for patient_sessions in sessions_list:
        for session_arr in patient_sessions:
            if session_arr is not None:
                if embedding_dim is None:
                    embedding_dim = session_arr.shape[1]
                elif session_arr.shape[1] != embedding_dim:
                    raise ValueError(
                        "Embedding dimension mismatch: "
                        f"expected {embedding_dim}, "
                        f"got {session_arr.shape[1]}")
                max_segments = max(max_segments, session_arr.shape[0])
    if embedding_dim is None:
        raise ValueError("No valid session embeddings found in the batch.")

    x = torch.zeros((batch_size, max_sessions, max_segments, embedding_dim), dtype=torch.float32)
    segment_mask = torch.zeros((batch_size, max_sessions, max_segments), dtype=torch.float32)

    for patient_idx, patient_sessions in enumerate(sessions_list):
        for session_idx, session_arr in enumerate(patient_sessions):
            if session_arr is None:
                continue
            n_segments = session_arr.shape[0]
            x[patient_idx, session_idx, :n_segments, :] = torch.from_numpy(session_arr)
            segment_mask[patient_idx, session_idx, :n_segments] = 1.0
    y = torch.stack(y_list)
    session_mask = torch.stack(session_mask_list)

    if has_alex:
        alex_y = torch.stack(alex_list)
        return (x, y, session_mask, segment_mask, alex_y)
        
    return (x, y, session_mask, segment_mask)


# In[5]:


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ____

# GRU

# In[6]:


class GRU(nn.Module):
    def __init__(self, input_dim=768, emb_dim=128, hidden=128, layers=1, bidirectional=False, 
                dropout=0.2, segment_pooling="attention"):
        super().__init__()

        # Segment pooling converts all segment embeddings from one session
        # into a single session representation.
        # "mean":      equal contribution from every valid segment.
        # "attention": learned contribution for every valid segment.
        self.segment_pooling = segment_pooling

        # Project different acoustic representations into a shared feature space.
        # CNN input:     [B, S, N, 128] -> [B, S, N, emb_dim]
        # wav2vec input: [B, S, N, 768] -> [B, S, N, emb_dim]
        # This allows the same longitudinal architecture to process both representations.   
        self.input_projection = nn.Sequential(nn.LayerNorm(input_dim),
                                              nn.Linear(input_dim, emb_dim), nn.ReLU())

        # Assign one learnable relevance score to each segment.
        # These scores are used only when segment_pooling="attention".
        # This is not Transformer self-attention and not CoLA trait-query attention.
        self.segment_score = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, 64),
                                           nn.Tanh(), nn.Linear(64, 1))

        self.rnn = nn.GRU(emb_dim, hidden, layers, batch_first=True,
                          bidirectional=bidirectional,
                          dropout=dropout if layers > 1 else 0.0)

        out_dim = hidden * (2 if bidirectional else 1)

        self.head = nn.Sequential(nn.Linear(out_dim, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def pool_segments(self, x, segment_mask):
        """
        Convert all segment embeddings from each session into one session vector.
        Input:
            x:            [B, S, N, input_dim]
            segment_mask: [B, S, N]
        Output:
            session_emb:  [B, S, emb_dim]
        Pooling modes:
            mean:      all valid segments receive equal weight.
            attention: the model learns one weight per valid segment.
        """
        # Map CNN or wav2vec segment embeddings to the shared emb_dim.
        x = self.input_projection(x)
        # Add one dimension so the mask can multiply the embedding vectors.
        mask = segment_mask.unsqueeze(-1)  # [B, S, N, 1]
        if self.segment_pooling == "mean":
            # Sum valid segment embeddings and divide by the number
            # of valid segments. Padded segments do not contribute.
            session_sum = (x * mask).sum(dim=2)
            valid_count = mask.sum(dim=2).clamp(min=1.0)
            return session_sum / valid_count
        # Produce one learnable relevance score for each segment.
        scores = self.segment_score(x).squeeze(-1)  # [B, S, N]
        # Prevent padded segments from receiving attention weight.
        scores = scores.masked_fill(segment_mask <= 0, -1e4)
        # Convert scores into positive weights.
        weights = torch.softmax(scores, dim=2) * segment_mask
        # Normalize valid weights to sum to one inside each session.
        weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
        # Weighted sum of all valid segment embeddings.
        return torch.sum(weights.unsqueeze(-1) * x, dim=2)

    def forward(self, x, session_mask, segment_mask):
        # x: [B, S, N, input_dim]
        session_emb = self.pool_segments(x, segment_mask)
        session_emb = session_emb * session_mask.unsqueeze(-1)

        # Remove missing sessions while preserving temporal order.
        sequences = [session_emb[i][session_mask[i].bool()] for i in range(x.size(0))]
        lengths = torch.tensor([seq.size(0) for seq in sequences], device=x.device)
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, h = self.rnn(packed)
        h_last = torch.cat([h[-2], h[-1]], dim=1) if self.rnn.bidirectional else h[-1]
        return self.head(h_last)


# Tranformer Light Encoder Version

# In[7]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                                      # create a position matrix
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)        # [[0], [1], [2], [3]... max_len-1 ] | [max_len, 1] -> [16,1]
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# In[8]:


class LightTransformer(nn.Module):
    def __init__(self, input_dim=128, emb_dim=128, nhead=4, layers=2,
                 dim_ff=256, dropout=0.2, pooling="last", segment_pooling="attention"):
        super().__init__()
        self.pooling = pooling
        self.segment_pooling = segment_pooling
        self.input_projection = nn.Sequential(nn.LayerNorm(input_dim),
                                              nn.Linear(input_dim, emb_dim), nn.ReLU())
        self.segment_score = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, 64),
                                           nn.Tanh(), nn.Linear(64, 1))
        self.input_norm = nn.LayerNorm(emb_dim)
        self.pos = PositionalEncoding(emb_dim, max_len=16)

        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead,
                                           dim_feedforward=dim_ff, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, 64),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def pool_segments(self, x, segment_mask):
        x = self.input_projection(x)
        mask = segment_mask.unsqueeze(-1)

        if self.segment_pooling == "mean":
            return (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)

        scores = self.segment_score(x).squeeze(-1)
        scores = scores.masked_fill(segment_mask <= 0, -1e4)
        weights = torch.softmax(scores, dim=2) * segment_mask
        weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
        return torch.sum(weights.unsqueeze(-1) * x, dim=2)

    def forward(self, x, session_mask, segment_mask):
        # x: [B, S, N, input_dim]
        session_emb = self.pool_segments(x, segment_mask)
        session_emb = session_emb * session_mask.unsqueeze(-1)

        z = self.pos(self.input_norm(session_emb))
        z = self.encoder(z, src_key_padding_mask=session_mask == 0)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        if self.pooling == "mean":
            mask = session_mask.unsqueeze(-1)
            pooled = (z * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        elif self.pooling == "last":
            valid_any = session_mask.sum(dim=1) > 0
            idx = session_mask.size(1) - 1 - torch.flip(session_mask, dims=[1]).argmax(dim=1)
            idx = torch.where(valid_any, idx.long(), torch.zeros_like(idx).long())
            pooled = z[torch.arange(x.size(0), device=x.device), idx]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        return self.head(pooled)


# Transformer Trait Attention

# In[9]:


class TraitQueryCoLAAttention(nn.Module):
    def __init__(self, input_dim=128, emb_dim=128, trait_dim=128, att_dim=128, alex_dim=1,
                 dropout=0.2, mode="full", trait_source="alex", segment_pooling="attention"):
        super().__init__()
        if trait_dim != emb_dim:
            raise ValueError("trait_dim must equal emb_dim for trait-state interactions.")
        if mode not in ["trait", "interaction", "delta_h1", "full"]:
            raise ValueError(f"Unknown CoLA mode: {mode}")
        if trait_source not in ["alex", "speech"]:
            raise ValueError("trait_source must be 'alex' or 'speech'.")
        if segment_pooling not in ["mean", "attention"]:
            raise ValueError("segment_pooling must be 'mean' or 'attention'.")

        self.mode = mode
        self.emb_dim = emb_dim
        self.att_dim = att_dim
        self.trait_source = trait_source
        self.segment_pooling = segment_pooling

        # CNN-logMel [D=128] or wav2vec [D=768] -> transform both into [B,S,N,128]
        self.input_projection = nn.Sequential(
                                nn.LayerNorm(input_dim), 
                                nn.Linear(input_dim, emb_dim), 
                                nn.ReLU())
        
        # Learns which segments represent each session.
        self.segment_score = nn.Sequential(
                            nn.LayerNorm(emb_dim), 
                            nn.Linear(emb_dim, 64), 
                            nn.Tanh(), 
                            nn.Linear(64, 1))

        # Two different trait definitions.
        self.speech_trait_encoder = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, trait_dim), nn.Tanh())
        self.alex_trait_encoder = nn.Sequential(nn.Linear(alex_dim, trait_dim), nn.Tanh())

        # Used only when trait_source="speech".
        self.alexithymia_head = nn.Sequential(nn.LayerNorm(trait_dim), 
                                            nn.Linear(trait_dim, 32),
                                            nn.ReLU(), 
                                            nn.Dropout(dropout), 
                                            nn.Linear(32, alex_dim))

        self.temporal_norm = nn.LayerNorm(emb_dim)
        self.temporal_pos = PositionalEncoding(emb_dim, max_len=16)
        temporal_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4, dim_feedforward=256,
                                                     dropout=dropout, batch_first=True, norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=2)

        self.trait_query = nn.Sequential(nn.LayerNorm(trait_dim), 
                                        nn.Linear(trait_dim, att_dim), 
                                        nn.Tanh())
        self.temporal_key = nn.Sequential(nn.LayerNorm(emb_dim * 2),
                                          nn.Linear(emb_dim * 2, att_dim), 
                                          nn.Tanh())
        if mode == "trait":
            cola_dim = emb_dim + trait_dim
        elif mode in ["interaction", "delta_h1"]:
            cola_dim = emb_dim + trait_dim + emb_dim
        else:
            cola_dim = emb_dim + trait_dim + emb_dim + emb_dim

        self.head = nn.Sequential(nn.LayerNorm(cola_dim), 
                                nn.Linear(cola_dim, 64),
                                nn.ReLU(), 
                                nn.Dropout(dropout), 
                                nn.Linear(64, 1))

    def pool_segments(self, x, segment_mask):
        mask = segment_mask.unsqueeze(-1)    # x: [B, S, N, D]
        if self.segment_pooling == "mean":
            return (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0), None
        scores = self.segment_score(x).squeeze(-1)
        scores = scores.masked_fill(segment_mask <= 0, -1e4)
        weights = torch.softmax(scores, dim=2) * segment_mask
        weights = weights / weights.sum(dim=2, keepdim=True).clamp(min=1e-8)
        session_emb = torch.sum(weights.unsqueeze(-1) * x, dim=2)
        return session_emb, weights

    def build_cola_input(self, h, z_trait):
        z = z_trait.unsqueeze(1).expand(-1, h.size(1), -1)
        h1 = h[:, 0, :].unsqueeze(1).expand_as(h)
        delta_h1 = h - h1
        interaction = h * z
        if self.mode == "trait":
            cola_input = torch.cat([h, z], dim=-1)
        elif self.mode == "interaction":
            cola_input = torch.cat([h, z, interaction], dim=-1)
        elif self.mode == "delta_h1":
            cola_input = torch.cat([h, z, delta_h1], dim=-1)
        else:
            cola_input = torch.cat([h, z, interaction, delta_h1], dim=-1)
        return cola_input, delta_h1

    def trait_query_attention(self, h, z_trait, delta_h1, session_mask):
        q = self.trait_query(z_trait)
        k = self.temporal_key(torch.cat([h, delta_h1], dim=-1))
        scores = (k * q.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.att_dim)
        scores = scores.masked_fill(session_mask <= 0, -1e4)
        weights = torch.softmax(scores, dim=1) * session_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return weights, scores

    def forward(self, x, session_mask, segment_mask, alex_input=None):
        """
        x:            [B, S, N, input_dim]
        session_mask: [B, S]
        segment_mask: [B, S, N]
        alex_input:   [B, alex_dim], standardized within the training fold
        """
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        session_mask = session_mask.float()
        segment_mask = segment_mask.float()

        # Project CNN or wav2vec embeddings to the same dimension.
        x = self.input_projection(x)

        # Use all segments to construct each session.
        session_emb, segment_attention = self.pool_segments(x, segment_mask)
        session_emb = session_emb * session_mask.unsqueeze(-1)

        # Baseline speech representation from Session 1.
        if torch.any(session_mask[:, 0] == 0):
            raise ValueError("CoLA requires Session 1 for every patient.")
        s1_emb = session_emb[:, 0, :]

        # Trait source:
        if self.trait_source == "alex":
            if alex_input is None:
                raise ValueError("alex_input is required when trait_source='alex'.")
            z_trait = self.alex_trait_encoder(alex_input.float())
            alexithymia_pred = None
        else:
            z_trait = self.speech_trait_encoder(s1_emb)
            alexithymia_pred = self.alexithymia_head(z_trait)

        h = self.temporal_pos(self.temporal_norm(session_emb))
        h = self.temporal_encoder(h, src_key_padding_mask=session_mask == 0)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        h = h * session_mask.unsqueeze(-1)

        cola_input, delta_h1 = self.build_cola_input(h, z_trait)
        session_attention, scores = self.trait_query_attention(h, z_trait, delta_h1, session_mask)
        pooled = torch.sum(session_attention.unsqueeze(-1) * cola_input, dim=1)
        out = self.head(pooled)

        return {"out": out, 
                "alexithymia_pred": alexithymia_pred, 
                "z": z_trait,
                "segment_attention": segment_attention,
                "attention": session_attention.unsqueeze(-1),
                "attention_scores": scores.unsqueeze(-1)}


# ___

# Train/Evaluate

# In[10]:


def unpack_batch(batch):
    if len(batch) == 4:
        xb, yb, session_mask, segment_mask = batch
        alex_y = None
    elif len(batch) == 5:
        xb, yb, session_mask, segment_mask, alex_y = batch
    else:
        raise ValueError(f"Unexpected batch format with {len(batch)} elements.")
    return xb, yb, session_mask, segment_mask, alex_y


# In[11]:


def forward_longitudinal(model, xb, session_mask, segment_mask, alex_y, model_name, trait_source):
    if model_name in ["TRAIT", "COLA"]:
        alex_input = alex_y if trait_source == "alex" else None
        return model(xb, session_mask, segment_mask, alex_input=alex_input)
    return model(xb, session_mask, segment_mask)


# In[12]:


def train_one_epoch(model, dataloader, optimizer, criterion, device, model_name,
                    trait_source="alex", lambda_alex=0.05, aux_criterion=None):
    model.train()
    aux_criterion = aux_criterion or nn.SmoothL1Loss()
    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    total_loss, total_main, total_alex, n_samples = 0.0, 0.0, 0.0, 0

    for batch_idx, batch in enumerate(dataloader):
        xb, yb, session_mask, segment_mask, alex_y = unpack_batch(batch)
        xb = xb.to(device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device, dtype=torch.float32, non_blocking=True)
        session_mask = session_mask.to(device, dtype=torch.float32, non_blocking=True)
        segment_mask = segment_mask.to(device, dtype=torch.float32, non_blocking=True)

        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        if alex_y is not None:
            alex_y = alex_y.to(device, dtype=torch.float32, non_blocking=True)

        if not torch.isfinite(yb).all():
            raise FloatingPointError(f"Non-finite target at batch {batch_idx}")
        if alex_y is not None and not torch.isfinite(alex_y).all():
            raise FloatingPointError(f"Non-finite alexithymia at batch {batch_idx}")

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = forward_longitudinal(model, xb, session_mask, segment_mask,
                                           alex_y, model_name, trait_source)
            preds = outputs["out"] if isinstance(outputs, dict) else outputs
            preds = preds.view_as(yb)
            loss_main = criterion(preds, yb)
            loss_alex = torch.zeros((), device=device)
            loss = loss_main

            if model_name in ["TRAIT", "COLA"] and trait_source == "speech":
                alex_pred = outputs["alexithymia_pred"].view_as(alex_y)
                loss_alex = aux_criterion(alex_pred, alex_y)
                loss = loss_main + lambda_alex * loss_alex

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at batch {batch_idx}: "
                f"main={loss_main.item()}, alex={loss_alex.item()}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        total_main += loss_main.item() * batch_size
        total_alex += loss_alex.item() * batch_size
        n_samples += batch_size

    n_samples = max(1, n_samples)
    return {"loss": total_loss / n_samples, 
            "main": total_main / n_samples,
            "alex": total_alex / n_samples}


# In[13]:


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device, model_name, trait_source="alex", task="reg", return_attention=False):
    model.eval()
    total_loss, n_samples = 0.0, 0
    y_true, y_pred = [], []
    attention_all = []

    for batch_idx, batch in enumerate(dataloader):
        xb, yb, session_mask, segment_mask, alex_y = unpack_batch(batch)
        xb = xb.to(device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device, dtype=torch.float32, non_blocking=True)
        session_mask = session_mask.to(device, dtype=torch.float32, non_blocking=True)
        segment_mask = segment_mask.to(device, dtype=torch.float32, non_blocking=True)

        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
        if alex_y is not None:
            alex_y = alex_y.to(device, dtype=torch.float32, non_blocking=True)

        outputs = forward_longitudinal(model, xb, session_mask, segment_mask, alex_y, model_name, trait_source)
        logits = outputs["out"] if isinstance(outputs, dict) else outputs

        if return_attention:
            if not isinstance(outputs, dict) or "attention" not in outputs:
                raise KeyError("The model output does not contain temporal attention weights.")
            att_batch = outputs["attention"].detach().cpu()
            mask_batch = session_mask.detach().cpu().bool()
            if att_batch.ndim == 3 and att_batch.shape[1] == 1:
                att_batch = att_batch[:, 0, :]
            elif att_batch.ndim == 3 and att_batch.shape[-1] == 1:
                att_batch = att_batch[:, :, 0]
            if att_batch.ndim != 2:
                raise ValueError(f"Unexpected attention shape: {tuple(att_batch.shape)}")
            for i in range(att_batch.shape[0]):
                valid_attention = att_batch[i][mask_batch[i]]
                valid_attention = valid_attention / valid_attention.sum().clamp(min=1e-8)
                attention_all.append(valid_attention.numpy())

        logits = logits.view_as(yb)
        loss = criterion(logits, yb)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite evaluation loss at batch {batch_idx}")

        batch_size = xb.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size
        y_true.append(yb.detach().cpu())

        if task == "cls":
            y_pred.append(torch.sigmoid(logits).detach().cpu())
        else:
            y_pred.append(logits.detach().cpu())

    if not y_true:
        empty_result = float("nan"), np.array([]), np.array([])
        if return_attention:
            return *empty_result, []
        return empty_result

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    result = total_loss / max(1, n_samples), y_true, y_pred
    if return_attention:
        return *result, attention_all
    return result


# ___

# Running the experiments using LOO-CV patient-independent

# Regression

# In[14]:


def attach_cnn_fold_paths(df, test_patient, split, out_col="CNN_SegEmb_npy", root="/workspace/app/cnn_segment_embeddings"):
    d = df.copy()
    fold_id = int(test_patient)
    d[out_col] = [
        os.path.join(
            root, f"fold_{fold_id}", split, str(int(row["Patient_ID"])),
            f"Session{int(row['Session'])}", f"seg_{int(idx)}.npy")
        for idx, row in d.iterrows()]
    missing = [path for path in d[out_col] if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} CNN embeddings are missing. First: {missing[0]}")
    return d


# In[15]:


def build_fold_alexithymia_total(train_df, test_df, patient_col="Patient_ID",
                                 tas_col="Questionnary3_TAS_20_T0_TOT_Score",
                                 aqc_col="Questionnary4_AQC_T0_TOT_Score",
                                 out_col="Alexithymia_T0"):
    train_df, test_df = train_df.copy(), test_df.copy()
    train_pat = train_df[[patient_col, tas_col, aqc_col]].drop_duplicates(patient_col)

    tas_mean, tas_std = train_pat[tas_col].mean(), train_pat[tas_col].std()
    aqc_mean, aqc_std = train_pat[aqc_col].mean(), train_pat[aqc_col].std()
    tas_std = 1.0 if not np.isfinite(tas_std) or tas_std == 0 else tas_std
    aqc_std = 1.0 if not np.isfinite(aqc_std) or aqc_std == 0 else aqc_std

    for d in [train_df, test_df]:
        tas_z = (d[tas_col] - tas_mean) / tas_std
        aqc_z = (d[aqc_col] - aqc_mean) / aqc_std
        d[out_col] = tas_z.combine_first(aqc_z)
    return train_df, test_df


# In[16]:


def reg_lopoRNN(patient, df, patient_col, npy_col, target_col, model_name, alex_col=None, representation="cnn", modeCola="full", pooling_trans="last", num_epochs=20, BS=2, device="cuda", lambda_alex=0.05, trait_source="alex", seed=42):
    """     
        GRU:         segment mean → GRU → last hidden state
        Transformer: segment mean → Transformer → last valid session or mean
        CoLA-trait:  segment mean → temporal Transformer → trait-query attention 
    """
    global _ARR_CACHE
    _ARR_CACHE = {}

    model_name = model_name.upper()
    representation = representation.lower()

    fold_seed = int(seed) + int(patient)
    set_seed(fold_seed)

    use_alex = model_name in ["TRAIT", "COLA"]

    if representation not in ["cnn", "wav2vec"]:
        raise ValueError("representation must be 'cnn' or 'wav2vec'.")
    if use_alex and alex_col is None:
        raise ValueError("alex_col is required for TRAIT/COLA.")

    train_df = df[df[patient_col] != patient].dropna(subset=[target_col]).copy()
    test_df = df[df[patient_col] == patient].dropna(subset=[target_col]).copy()

    if use_alex:
        train_df, test_df = build_fold_alexithymia_total(
            train_df, test_df, patient_col=patient_col, out_col="Alexithymia_T0")
        alex_col = "Alexithymia_T0"
        train_df = train_df.dropna(subset=[alex_col]).copy()
        test_df = test_df.dropna(subset=[alex_col]).copy()

    if representation == "cnn":
        npy_col = "CNN_SegEmb_npy"
        train_df = attach_cnn_fold_paths(train_df, patient, "train", out_col=npy_col)
        test_df = attach_cnn_fold_paths(test_df, patient, "test", out_col=npy_col)
        input_dim = 128
    else:
        train_df = train_df.dropna(subset=[npy_col]).copy()
        test_df = test_df.dropna(subset=[npy_col]).copy()
        input_dim = 768

    aux_dim = 1
    alex_model_col = alex_col if use_alex else None

    print(f"LOPO - {model_name} | representation={representation}")
    if use_alex:
        print(f"Trait source: {trait_source} | Alex col: {alex_col}")

    # pool_within_file=True : wav2vec [T,768] → temporal mean → [1,768]
    print(f"[Patient {patient}] Building datasets")
    train_ds = SeqNPYDataset_patient(train_df, npy_col, target_col,
                                     alex_col=alex_model_col if use_alex else None,
                                     preload_to_ram=True, pool_within_file=True, verbose=False)
    print(f"[Patient {patient}] Train dataset ready")
    test_ds = SeqNPYDataset_patient(test_df, npy_col, target_col,
                                    alex_col=alex_model_col if use_alex else None,
                                    preload_to_ram=True, pool_within_file=True, verbose=True)
    train_dl, test_dl = get_io_loaders(train_ds, test_ds, BS, seed=fold_seed)
    print(f"[Patient {patient}] Test dataset ready")

    if model_name == "GRU":
        model = GRU(input_dim=input_dim)
    elif model_name == "TRANS":
        model = LightTransformer(input_dim=input_dim, pooling=pooling_trans)
    elif model_name == "TRAIT":
        model = TraitQueryCoLAAttention(input_dim=input_dim, alex_dim=aux_dim,
                                        mode="trait", trait_source=trait_source)
    elif model_name == "COLA":
        model = TraitQueryCoLAAttention(input_dim=input_dim, alex_dim=aux_dim,
                                        mode=modeCola, trait_source=trait_source)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    model = model.to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Auxiliary loss applied only to standardized alexithymia.
    aux_criterion = nn.SmoothL1Loss()
    best_monitor, best_state, best_epoch = float("inf"), None, None

    for epoch in range(num_epochs):
        train_stats = train_one_epoch(model, train_dl, opt, criterion, device, model_name=model_name,
                                    trait_source=trait_source, lambda_alex=lambda_alex, aux_criterion=aux_criterion)
        monitor_loss, _, _ = eval_epoch(model, train_dl, criterion, device, model_name=model_name,
                                    trait_source=trait_source, task="reg")

        print(f"[{model_name}][Patient {patient}] Epoch {epoch + 1}/{num_epochs} - "
                f"loss={train_stats['loss']:.4f} main={train_stats['main']:.4f} "
                f"alex={train_stats['alex']:.4f} monitor={monitor_loss:.4f}")

        if monitor_loss < best_monitor and np.isfinite(monitor_loss):
            best_monitor, best_epoch = monitor_loss, epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    loss, y_true, y_pred = eval_epoch(model, test_dl, criterion, device, model_name=model_name, trait_source=trait_source)

    if y_true.size == 0 or y_pred.size == 0:
        rmse = mse = mae = float("nan")
    else:
        y_true_flat, y_pred_flat = y_true.reshape(-1), y_pred.reshape(-1)
        mse = float(mean_squared_error(y_true_flat, y_pred_flat))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true_flat, y_pred_flat))

    return {"Model": model_name, 
            "Representation": representation, 
            "Seed": int(seed),
            "Fold_Seed": fold_seed,
            "Trait_Source": trait_source if use_alex else None,
            "CoLA_Mode": modeCola if model_name == "COLA" else None, "Patient_ID": patient,
            "Best_Epoch": best_epoch, 
            "Best_Train_Monitor_Loss": best_monitor,
            "Test_Loss": float(loss), 
            "RMSE": rmse, 
            "MSE": mse, 
            "MAE": mae,
            "y_true": y_true.reshape(-1).tolist(), 
            "y_pred": y_pred.reshape(-1).tolist()}


# Classification (Embeddings)

# In[17]:


def class_lopoRNN(patient, df, patient_col, target_col, npy_col, model_name, alex_col=None, representation="cnn", modeCola="full", pooling_trans="last", BS=2, device="cuda", epochs=20, pos_weight=None, trait_source="alex", lambda_alex=0.05, tau=0.5, seed=42):
    global _ARR_CACHE
    _ARR_CACHE = {}
    model_name = model_name.upper()
    representation = representation.lower()

    fold_seed = int(seed) + int(patient)
    set_seed(fold_seed)

    use_alex = model_name in ["TRAIT", "COLA"]

    if representation not in ["cnn", "wav2vec"]:
        raise ValueError("representation must be 'cnn' or 'wav2vec'.")
    if use_alex and alex_col is None:
        raise ValueError("alex_col is required for TRAIT/COLA.")

    train_df = df[df[patient_col] != patient].dropna(subset=[target_col]).copy()
    test_df = df[df[patient_col] == patient].dropna(subset=[target_col]).copy()

    if use_alex:
        train_df, test_df = build_fold_alexithymia_total(train_df, test_df, patient_col=patient_col, out_col="Alexithymia_T0")
        alex_col = "Alexithymia_T0"
        train_df = train_df.dropna(subset=[alex_col]).copy()
        test_df = test_df.dropna(subset=[alex_col]).copy()

    if representation == "cnn":
        npy_col = "CNN_SegEmb_npy"
        train_df = attach_cnn_fold_paths(train_df, patient, "train", out_col=npy_col)
        test_df = attach_cnn_fold_paths(test_df, patient, "test", out_col=npy_col)
        input_dim = 128
    else:
        train_df = train_df.dropna(subset=[npy_col]).copy()
        test_df = test_df.dropna(subset=[npy_col]).copy()
        input_dim = 768

    aux_dim = 1
    alex_model_col = alex_col if use_alex else None

    train_ds = SeqNPYDataset_patient(train_df, npy_col, target_col, alex_col=alex_model_col, preload_to_ram=True, pool_within_file=True, verbose=False)
    test_ds = SeqNPYDataset_patient(test_df, npy_col, target_col, alex_col=alex_model_col, preload_to_ram=True, pool_within_file=True, verbose=False)
    train_dl, test_dl = get_io_loaders(train_ds, test_ds, BS, seed=fold_seed)

    if model_name == "GRU":
        model = GRU(input_dim=input_dim)
    elif model_name == "TRANS":
        model = LightTransformer(input_dim=input_dim, pooling=pooling_trans)
    elif model_name == "TRAIT":
        model = TraitQueryCoLAAttention(
            input_dim=input_dim, alex_dim=aux_dim, mode="trait", trait_source=trait_source)
    elif model_name == "COLA":
        model = TraitQueryCoLAAttention(
            input_dim=input_dim, alex_dim=aux_dim, mode=modeCola, trait_source=trait_source)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model = model.to(device=device, dtype=torch.float32)

    if pos_weight is None:
        train_pat = train_df[[patient_col, target_col]].drop_duplicates(patient_col)
        pos = int((train_pat[target_col] == 1).sum())
        neg = int((train_pat[target_col] == 0).sum())
        pw_value = neg / max(1, pos)
    else:
        pw_value = float(pos_weight)

    print(f"LOPO - {model_name} | representation={representation} | " f"pos_weight={pw_value:.4f} | tau={tau:.2f}")

    pw_tensor = torch.tensor([pw_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    aux_criterion = nn.SmoothL1Loss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_monitor, best_state, best_epoch = float("inf"), None, None

    for epoch in range(epochs):
        train_stats = train_one_epoch(model, train_dl, opt, criterion, device, model_name=model_name, trait_source=trait_source, lambda_alex=lambda_alex, aux_criterion=aux_criterion)
        monitor_loss, _, _ = eval_epoch(model, train_dl, criterion, device, model_name=model_name, trait_source=trait_source, task="cls")

        print(f"[{model_name}][Patient {patient}] Epoch {epoch + 1}/{epochs} - "
              f"loss={train_stats['loss']:.4f} main={train_stats['main']:.4f} "
              f"alex={train_stats['alex']:.4f} monitor={monitor_loss:.4f}")

        if monitor_loss < best_monitor and np.isfinite(monitor_loss):
            best_monitor, best_epoch = monitor_loss, epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    save_attention = use_alex and trait_source == "alex"
    if save_attention:
        loss, y_true, y_prob, attention_all = eval_epoch(
            model, test_dl, criterion, device, model_name=model_name,
            trait_source=trait_source, task="cls", return_attention=True)
        attention = attention_all[0] if attention_all else np.array([])
    else:
        loss, y_true, y_prob = eval_epoch(
            model, test_dl, criterion, device, model_name=model_name,
            trait_source=trait_source, task="cls")
        attention = np.array([])
    
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)

    y_patient_true = int(round(float(y_true[0]))) if y_true.size else 0
    p_patient = float(y_prob[0]) if y_prob.size else float("nan")
    y_patient_pred = int(p_patient >= tau) if np.isfinite(p_patient) else 0


    alex_test = float(test_df[alex_col].iloc[0]) if save_attention and not test_df.empty else np.nan
    sessions = sorted(test_df["Session"].dropna().astype(int).unique().tolist())
    if attention.size:
        temporal_positions = np.linspace(0.0, 1.0, attention.size)
        attention_center = float(np.sum(attention * temporal_positions))
    else:
        attention_center = np.nan

    return {"Model": model_name, 
            "Representation": representation,
            "Seed": int(seed),
            "Fold_Seed": fold_seed,
            "Trait_Source": trait_source if use_alex else None,
            "CoLA_Mode": modeCola if model_name == "COLA" else None,
            "Patient_ID": patient, 
            "Best_Epoch": best_epoch,
            "Best_Train_Monitor_Loss": best_monitor, 
            "Test_Loss": float(loss),
            "p_patient": p_patient, 
            "y_patient_true": y_patient_true,
            "y_patient_pred": y_patient_pred, 
            "Session_Attention": attention.tolist(),
            "Sessions": sessions,
            "Temporal_Attention_Center": attention_center,
            "Alexithymia_Standardized_Fold": alex_test}

