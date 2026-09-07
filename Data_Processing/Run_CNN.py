#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# jupyter nbconvert --to script Run_CNN.ipynb


# ___

# ### Dataset

# In[2]:


def get_io_loaders(train_ds, test_ds, BS):
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available(), persistent_workers=False)
    test_dl = DataLoader(test_ds, batch_size=max(1, BS // 4), shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(), persistent_workers=False)
    return train_dl, test_dl


# Dataset: patient/session-level
# * Using all segments

# In[ ]:


_ARR_CACHE = {}   
class SeqNPYDataset_all_segments(Dataset):
    def __init__(self, df, npy_col, target_col, max_T=256, preload_to_ram=False, verbose=True, model_name="CNN"):
        self.paths   = df[npy_col].tolist()         # npy_col = Log-Mel
        self.targets = df[target_col].values.astype("float32")
        self.max_T   = int(max_T)
        self.preload = bool(preload_to_ram)
        self.verbose = verbose
        self.arrs = None
        self.model_name = model_name.upper()
        if self.preload:
            self.arrs = []
            total_mb = 0.0
            new_count = 0
            for p in self.paths:
                key = (p, self.max_T)  # key: (path, max_T) -> np.ndarray float32 (T x D)
                a = _ARR_CACHE.get(key)
                if a is None:
                    if self.verbose:
                        print(f"[load] {p}")
                    a = np.load(p, mmap_mode=None)   
                    if a.shape[0] > self.max_T:
                        a = a[:self.max_T]
                    elif a.shape[0] < self.max_T:
                        pad = np.full((self.max_T - a.shape[0], a.shape[1]), -80.0, dtype=np.float32)
                        a = np.concatenate([a, pad], axis=0)
                    a = np.ascontiguousarray(a, dtype=np.float32)
                    if not np.isfinite(a).all():
                        print(f"[WARN][DATASET] Non-finite in preloaded array: {p}. Applying nan_to_num.")
                        a = np.nan_to_num(a, nan=-80.0, posinf=0.0, neginf=-80.0)
                    _ARR_CACHE[key] = a
                    total_mb += a.nbytes / (1024**2)
                    new_count += 1
                self.arrs.append(_ARR_CACHE[key])
            if self.verbose:
                print(f"Preloaded {len(self.arrs)} segments (~{total_mb:.1f} MB new,"
                      f" reused {len(self.arrs)-new_count} from cache).")
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        if self.preload and self.arrs is not None:
            arr = self.arrs[idx]
        else:
            p = self.paths[idx]
            key = (p, self.max_T)
            a = _ARR_CACHE.get(key)
            if a is None:
                a = np.load(p, mmap_mode=None).astype(np.float32)  
                if a.shape[0] > self.max_T:
                    a = a[:self.max_T]
                elif a.shape[0] < self.max_T:
                    pad = np.full((self.max_T - a.shape[0], a.shape[1]), -80.0, dtype=np.float32)
                    a = np.concatenate([a, pad], axis=0)
                a = np.ascontiguousarray(a, dtype=np.float32)
                if not np.isfinite(a).all():
                    print(f"[WARN][DATASET] Non-finite values in loaded array: {p}. Applying nan_to_num.")
                    a = np.nan_to_num(a, nan=-80.0, posinf=0.0, neginf=-80.0)
                _ARR_CACHE[key] = a
            arr = _ARR_CACHE[key]
        if not np.isfinite(arr).all():
            p = self.paths[idx]
            print(f"[WARN][DATASET] Non-finite in cached array: {p}. Fixing in-place.")
            arr = np.nan_to_num(arr, nan=-80.0, posinf=0.0, neginf=-80.0)
            _ARR_CACHE[(p, self.max_T)] = arr
        x = torch.from_numpy(arr)
        x = torch.nan_to_num(x, nan=-80.0, posinf=0.0, neginf=-80.0)
        
        x = x.clamp(-80.0, 0.0)
        x = x.unsqueeze(0) # [1, T, n_mels]
        y = torch.tensor([self.targets[idx]], dtype=torch.float32)   
        return x, y


# ____

# CNN Extractor and Prediction

# In[ ]:


class CNN_Extractor(nn.Module):
    # Extract log-Mel using all segments to predict Y - depression outcome
    # Return embedding [B_cnn, 128] - one embedding per log-Mel segment to be aggregated by session
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 128)
        self.emb_act = nn.ReLU()

        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        x = torch.nan_to_num(x, nan=-80.0, posinf=0.0, neginf=-80.0)
        # log-Mel
        x = x.clamp(-80.0, 0.0)
        x = (x + 80.0) / 80.0

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        embedding = self.emb_act(self.fc(x))
        embedding = torch.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

        out = self.head(embedding)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return {"embedding": embedding, "out": out}


# ____

# Train/Evaluate

# In[ ]:


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_valid_batches = 0
    use_amp = False # use_amp = str(device).startswith("cuda") 
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    for batch_idx, batch in enumerate(dataloader):
        xb, yb = batch
        if not torch.isfinite(xb).all():
            print(f"[FATAL][train] Non-finite xb RIGHT AFTER DATALOADER at batch {batch_idx}")
        if not torch.isfinite(yb).all():
            print(f"[FATAL][train] Non-finite yb RIGHT AFTER DATALOADER at batch {batch_idx}")
        xb = xb.to(device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device, dtype=torch.float32, non_blocking=True)
        xb = torch.nan_to_num(xb, nan=-80.0, posinf=-0.0, neginf=-80.0)
        xb = xb.clamp(-80.0, 0.0)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(xb)         
            if isinstance(outputs, dict):
                preds = outputs["out"]
            else:
                preds = outputs
            preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
            preds = preds.view_as(yb)
            loss_main = criterion(preds, yb) 
            loss = loss_main

        if not torch.isfinite(loss):
            print(f"[WARN][train] loss non-finite at batch {batch_idx}")
            print("  xb stats:", xb.min().item(), xb.max().item(), xb.mean().item(), xb.std().item())
            print("  yb:", yb.detach().cpu().view(-1))
            print("  yb finite:", torch.isfinite(yb).detach().cpu().view(-1))
            print("  preds:", preds.detach().cpu().view(-1))
            print("  preds finite:", torch.isfinite(preds).detach().cpu().view(-1))
            print("  loss_main:", loss_main.detach().cpu())
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_valid_batches += 1

        if (batch_idx + 1) % 200 == 0:
            print(f"[train] batch {batch_idx + 1}/{len(dataloader)} loss={loss.item():.4f}")
    return total_loss / max(1, n_valid_batches)


# In[ ]:


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_valid_batches = 0
    y_true = []
    y_pred = []
    # use_amp = str(device).startswith("cuda") 
    use_amp = False
    for batch_idx, batch in enumerate(dataloader):
        xb, yb, = batch
        xb = xb.to(device, dtype=torch.float32, non_blocking=True)
        yb = yb.to(device, dtype=torch.float32, non_blocking=True)
        xb = torch.nan_to_num(xb, nan=-80.0, posinf=0.0, neginf=-80.0)
        xb = xb.clamp(-80.0, 0.0)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(xb)
            if isinstance(outputs, dict):
                logits = outputs["out"]
            else:
                logits = outputs
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            logits = logits.view_as(yb)
            loss = criterion(logits, yb)
        if not torch.isfinite(loss):
            print("[WARN][eval] loss non-finite at batch", batch_idx)
            print("  xb stats:", xb.min().item(), xb.max().item(), xb.mean().item(), xb.std().item())
            print("  logits stats:", logits.min().item(), logits.max().item(), logits.mean().item(), logits.std().item())
            continue

        total_loss += loss.item()
        n_valid_batches += 1
        y_true.append(yb.detach().cpu())
        y_pred.append(logits.detach().cpu())

    if len(y_true) == 0:
        return float("nan"), np.array([]), np.array([])
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return total_loss / max(1, n_valid_batches), y_true, y_pred


# In[ ]:


@torch.no_grad()
def extract_cnn_segment_embeddings(model, df, npy_col, out_root, patient_col="Patient_ID", session_col="Session", 
                                   out_col="CNN_SegEmb_npy", device="cuda", batch_size=2048, max_T=256, overwrite=False):
    os.makedirs(out_root, exist_ok=True)
    model.eval()
    d = df.copy()
    d[out_col] = None

    paths = d[npy_col].tolist()
    idxs = d.index.tolist()

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start+batch_size]
        batch_idxs = idxs[start:start+batch_size]
        xs = []

        for p in batch_paths:
            a = np.load(p, mmap_mode=None).astype(np.float32)
            if a.shape[0] > max_T:
                a = a[:max_T]
            elif a.shape[0] < max_T:
                pad = np.full((max_T - a.shape[0], a.shape[1]), -80.0, dtype=np.float32)
                a = np.concatenate([a, pad], axis=0)
            a = np.nan_to_num(a, nan=-80.0, posinf=0.0, neginf=-80.0)
            a = np.clip(a, -80.0, 0.0).astype(np.float32)
            xs.append(a)

        xb = torch.from_numpy(np.stack(xs)).unsqueeze(1).to(device, dtype=torch.float32)
        outputs = model(xb)
        emb = outputs["embedding"].detach().cpu().numpy().astype(np.float32)

        for j, idx in enumerate(batch_idxs):
            pid = d.loc[idx, patient_col]
            ses = int(d.loc[idx, session_col])
            out_dir = os.path.join(out_root, str(pid), f"Session{ses}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"seg_{idx}.npy")
            if overwrite or not os.path.exists(out_path):
                np.save(out_path, emb[j:j+1])  # [1, 128]
            d.at[idx, out_col] = out_path
    return d


# ___

# Running the experiments using LOO-CV patient-independent

# Regression

# In[ ]:


def reg_lopoCNN(patient, df, patient_col, npy_col, target_col, model_name, num_epochs=20, BS=512, device="cuda"):
    # logMel_npy per segment
    # CNN_Extractor trained on the fold
    # embedding [128] per segment
    # save segment-level embedding to .npy
    # create new column CNN_SegEmb_npy
    global _ARR_CACHE
    _ARR_CACHE = {}
    train_df = df[df[patient_col] != patient].copy()
    test_df = df[df[patient_col] == patient].copy()

    train_df = train_df.dropna(subset=[npy_col, target_col]).copy()
    test_df = test_df.dropna(subset=[npy_col, target_col]).copy()

    model_name = model_name.upper() # CNN
    train_ds = SeqNPYDataset_all_segments(train_df, npy_col, target_col, max_T=256, preload_to_ram=True, verbose=False, model_name=model_name)
    test_ds = SeqNPYDataset_all_segments(test_df, npy_col, target_col, max_T=256, preload_to_ram=True, verbose=False, model_name=model_name)
    train_dl, test_dl = get_io_loaders(train_ds, test_ds, BS)
    model = CNN_Extractor()

    model = model.to(device=device, dtype=torch.float32)
    print("LOPO -", model_name)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_monitor = float("inf")
    best_state = None
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_dl, opt, criterion, device)
        monitor_loss, _, _ = eval_epoch(model, train_dl, criterion, device)
        print(
            f"[{model_name}][Patient {patient}] "
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"train_loss={train_loss:.4f} "
            f"train_monitor_loss={monitor_loss:.4f}")
        if monitor_loss < best_monitor and np.isfinite(monitor_loss):
            best_monitor = monitor_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    fold_root = f"/workspace/app/cnn_segment_embeddings/fold_{patient}"
    train_emb_df = extract_cnn_segment_embeddings(model, train_df, npy_col, os.path.join(fold_root, "train"), patient_col=patient_col, session_col="Session", out_col="CNN_SegEmb_npy", device=device, batch_size=2048, max_T=256, overwrite=True)
    test_emb_df = extract_cnn_segment_embeddings(model, test_df, npy_col, os.path.join(fold_root, "test"), patient_col=patient_col, session_col="Session", out_col="CNN_SegEmb_npy", device=device, batch_size=2048, max_T=256, overwrite=True)

    loss, y_true, y_pred = eval_epoch(model, test_dl, criterion, device)

    if y_true.size == 0 or y_pred.size == 0:
        print(f"[WARN][LOO] No test samples for patient {patient}. Skipping RMSE.")
        rmse = float("nan")
        mse = float("nan")
        mae = float("nan")
    else:
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
        mse = float(mean_squared_error(y_true_flat, y_pred_flat))
        mae = float(mean_absolute_error(y_true_flat, y_pred_flat))

    return {
        "Model": model_name,
        "Patient_ID": patient,
        "RMSE": rmse,
        "MSE": mse,
        "MAE": mae,
        "y_true": y_true.reshape(-1).tolist(),
        "y_pred": y_pred.reshape(-1).tolist(),
        "train_emb_df": train_emb_df,
        "test_emb_df": test_emb_df}

