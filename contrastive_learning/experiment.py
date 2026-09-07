"""Experiment runner for the contrastive model (K-fold, full-batch bounded).

Extracted verbatim from the contrastive notebooks.
"""

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from .model import ContrastiveGeneticModel
from .evaluation import extract_embeddings, compute_metrics


def run_experiment(name: str, dm, config: dict,
                   max_epochs: int = 5000, k_folds: int = 3) -> dict:

    print(f"\n{'='*80}")
    print(f"🧪 ESPERIMENTO: {name}")
    print(f"   K-Fold: {k_folds} | Max epochs: {max_epochs} | FULL-BATCH (with size limit)")
    print(f"{'='*80}\n")

    X, y = dm.X_processed, dm.y_processed
    skf  = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_metrics_list = []
    best_acc, best_model, best_Z, best_y = -1, None, None, None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n🔹 FOLD {fold+1}/{k_folds}  "
              f"(train={len(tr_idx)}, val={len(val_idx)})")

        X_tr,  X_val  = X[tr_idx],  X[val_idx]
        y_tr,  y_val  = y[tr_idx],  y[val_idx]

        # ── Limit batch size to prevent OOM ──────────────
        # Even though the paper does "full-batch", in practice on consumer GPUs
        # doing N*(N-1) pairs causes OutOfMemory. Bounding to max 512 or 1024.
        eff_batch_size = min(len(X_tr), 512)

        tr_dl  = DataLoader(
            TensorDataset(torch.FloatTensor(X_tr),  torch.LongTensor(y_tr)),
            batch_size=eff_batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

        val_dl = DataLoader(
            TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
            batch_size=eff_batch_size,
            num_workers=4, pin_memory=True, drop_last=False)

        print(f"   Batch size effettivo: {eff_batch_size}")

        model = ContrastiveGeneticModel(
            n_markers         = dm.num_snps,
            embedding_dim     = config['embedding_dim'],
            flip_max          = config['flip_max'],
            mask_max          = config['mask_max'],
            learning_rate     = config['learning_rate'],
            lr_decay_factor   = config['lr_decay_factor'],
            lr_decay_interval = config['lr_decay_interval'],
        )

        ckpt_dir = f"checkpoints/{name.replace(' ','_')}/fold_{fold+1}"
        callbacks = [
            ModelCheckpoint(dirpath=ckpt_dir, filename='{epoch:04d}-{val_loss:.4f}',
                            save_top_k=1, monitor='val_loss', mode='min'),
            # Paper non menziona early stopping — patience=200 come salvaguardia pratica
            EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=False),
            LearningRateMonitor(logging_interval='epoch'),
        ]

        trainer = pl.Trainer(
            max_epochs           = max_epochs,
            accelerator          = config['accelerator'],
            devices              = config['devices'],
            callbacks            = callbacks,
            enable_progress_bar  = True,
            log_every_n_steps    = 1,
            enable_model_summary = (fold == 0),
            precision            = config.get('precision', '32-true'), # Added mixed precision
        )
        trainer.fit(model, tr_dl, val_dl)

        # ── Estrai embedding dal best checkpoint ───────────────
        best_ckpt  = callbacks[0].best_model_path
        best_fold  = ContrastiveGeneticModel.load_from_checkpoint(best_ckpt)
        Z_val = extract_embeddings(best_fold, X_val)
        m     = compute_metrics(Z_val, y_val, k=3)
        print(f"   ✅ Fold {fold+1}: knn_acc@3={m['knn_acc_k3']} | sil={m['silhouette']}")
        fold_metrics_list.append(m)

        if m['knn_acc_k3'] > best_acc:
            best_acc, best_model, best_Z, best_y = m['knn_acc_k3'], best_fold, Z_val, y_val

    # ── Media sui fold ─────────────────────────────────────────
    avg    = {k: np.mean([fm[k] for fm in fold_metrics_list]) for k in fold_metrics_list[0]}
    result = {'Experiment': name, **{f'{k} (mean)': round(v, 4) for k, v in avg.items()}}

    print(f"\n📊 Risultati medi ({k_folds} fold):")
    for k, v in result.items():
        if k != 'Experiment': print(f"   {k}: {v}")

    result['_best_model'] = best_model
    result['_best_Z']     = best_Z
    result['_best_y']     = best_y
    return result
