"""Experiment runner for the contrastive model (protocol-v2 shared layer).

The K-fold loop is driven exclusively by the frozen protocol-v2 fold data from
``genomic.experiment_data``. No independent splitting, cohort filtering or
preprocessing happens here; each fold's model uses that fold's retained SNP
count as ``n_markers`` (folds may legitimately differ).
"""

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from .data import build_fold_dataloaders
from .model import ContrastiveGeneticModel
from .evaluation import extract_embeddings, compute_metrics, evaluate_knn_classification
from genomic.classification_metrics import aggregate_folds
from genomic.selection import best_epoch_from_checkpoint_path


def run_experiment(name: str, experiment_data, config: dict,
                   max_epochs: int = 5000) -> dict:
    """Run the 3-fold contrastive CV over the protocol-v2 split.

    Parameters
    ----------
    name:
        Experiment label (used for checkpoint directories).
    experiment_data:
        A ``genomic.experiment_data.GenomicExperimentData`` providing the frozen
        development folds. The locked test set is never accessed.
    config:
        Model/training configuration (``embedding_dim``, ``flip_max``,
        ``mask_max``, ``learning_rate``, ``lr_decay_factor``,
        ``lr_decay_interval``, ``accelerator``, ``devices``, optional
        ``precision``).
    max_epochs:
        Maximum training epochs per fold.
    """
    n_folds = experiment_data.split.n_folds
    n_classes = experiment_data.n_classes
    labels = np.arange(n_classes)

    print(f"\n{'='*80}")
    print(f"🧪 ESPERIMENTO: {name}")
    print(f"   K-Fold: {n_folds} | Max epochs: {max_epochs} | FULL-BATCH (with size limit)")
    print(f"{'='*80}\n")

    fold_class_metrics = []
    fold_metrics_list = []
    fold_best_epochs = []
    best_acc, best_model, best_Z, best_y = -1, None, None, None

    for fold in range(n_folds):
        fd = experiment_data.fold_data(fold)
        n_markers = fd.X_train.shape[1]

        print(f"\n🔹 FOLD {fold+1}/{n_folds}  "
              f"(train={len(fd.X_train)}, val={len(fd.X_val)}, n_markers={n_markers})")

        # ── Limit batch size to prevent OOM ──────────────
        # Even though the paper does "full-batch", in practice on consumer GPUs
        # doing N*(N-1) pairs causes OutOfMemory. Bounding to max 512 or 1024.
        eff_batch_size = min(len(fd.X_train), 512)
        tr_dl, val_dl = build_fold_dataloaders(fd, eff_batch_size)
        print(f"   Batch size effettivo: {eff_batch_size}")

        model = ContrastiveGeneticModel(
            n_markers         = n_markers,
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
        fold_best_epochs.append(best_epoch_from_checkpoint_path(best_ckpt))
        best_fold  = ContrastiveGeneticModel.load_from_checkpoint(best_ckpt)

        # Deterministic eval embeddings (model.eval(), augment=False):
        # no allele flipping, no random masking.
        Z_train = extract_embeddings(best_fold, fd.X_train)
        Z_val   = extract_embeddings(best_fold, fd.X_val)

        # Correct train -> validation KNN breed classification.
        cm, _ = evaluate_knn_classification(
            Z_train, fd.y_train, Z_val, fd.y_val, k=3, labels=labels
        )
        fold_class_metrics.append(cm)

        # Legacy self-consistency diagnostic (fit + eval on val embeddings),
        # kept only for compatibility; NOT a breed-classification result.
        sc = compute_metrics(Z_val, fd.y_val, k=3)
        fold_metrics_list.append(sc)

        print(f"   ✅ Fold {fold+1}: Macro_F1={cm['macro_f1']:.4f} | "
              f"BalAcc={cm['balanced_accuracy']:.4f} | "
              f"self_consistency_knn_acc_k3={sc['knn_acc_k3']} | sil={sc['silhouette']}")

        # Legacy presentation-only representative fold (self-consistency KNN).
        # Has ZERO role in checkpoint/config selection or scientific comparison.
        if sc['knn_acc_k3'] > best_acc:
            best_acc, best_model, best_Z, best_y = sc['knn_acc_k3'], best_fold, Z_val, fd.y_val

    # ── Scientific 3-fold aggregate (common classification metrics) ──
    result = {'Experiment': name}

    if fold_class_metrics:
        agg = aggregate_folds(fold_class_metrics)
        result['Accuracy (mean)'] = agg['accuracy_mean']
        result['Accuracy (std)'] = agg['accuracy_std']
        result['Macro_F1 (mean)'] = agg['macro_f1_mean']
        result['Macro_F1 (std)'] = agg['macro_f1_std']
        result['Balanced_Accuracy (mean)'] = agg['balanced_accuracy_mean']
        result['Balanced_Accuracy (std)'] = agg['balanced_accuracy_std']
        result['Per_Class_Recall (mean)'] = agg['per_class_recall_mean']
        result['Per_Class_Recall (std)'] = agg['per_class_recall_std']
        result['Confusion_Matrix'] = agg['confusion_matrix_sum']

    result['Best_Epochs'] = fold_best_epochs
    result['Median_Best_Epoch'] = float(np.median(fold_best_epochs)) if fold_best_epochs else None

    # ── Legacy diagnostic fields (self-consistency KNN + clustering) ──
    avg = {k: np.mean([fm[k] for fm in fold_metrics_list]) for k in fold_metrics_list[0]}
    for k, v in avg.items():
        if k.startswith('knn_'):
            result[f'Self_Consistency_{k} (mean)'] = round(v, 4)
        else:
            result[f'{k} (mean)'] = round(v, 4)

    print(f"\n📊 Risultati medi ({n_folds} fold):")
    for k, v in result.items():
        if k != 'Experiment': print(f"   {k}: {v}")

    result['_best_model'] = best_model
    result['_best_Z']     = best_Z
    result['_best_y']     = best_y
    return result
