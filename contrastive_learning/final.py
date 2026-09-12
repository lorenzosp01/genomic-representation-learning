"""PRIMARY RQ1 final-development training + locked-test evaluation (contrastive).

Final training is self-supervised on the FULL development set (no validation
split, no early stopping, no checkpoint selection) for exactly ``FINAL_EPOCHS``
completed epochs. The locked test is obtained only through the explicit
``GenomicExperimentData.transform_locked_test`` call, after training.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from genomic.experiment_data import FinalDevelopmentData, LockedTestData

from .evaluation import evaluate_knn_classification, extract_embeddings
from .model import ContrastiveGeneticModel

FINAL_EMBEDDING_DIM = 3
FINAL_EPOCHS = 1216

DEFAULT_CHECKPOINT = "checkpoints/rq1_final/contrastive_emb3_ep1216.ckpt"
DEFAULT_OUTPUT_DIR = "results/rq1_final"


def build_contrastive_final_model(dev_data: FinalDevelopmentData, config: dict,
                                  *, embedding_dim: int = FINAL_EMBEDDING_DIM) -> ContrastiveGeneticModel:
    """Construct the frozen contrastive model (paper-faithful) for the final fit."""
    return ContrastiveGeneticModel(
        n_markers=dev_data.n_features,
        embedding_dim=embedding_dim,
        flip_max=config["flip_max"],
        mask_max=config["mask_max"],
        learning_rate=config["learning_rate"],
        lr_decay_factor=config["lr_decay_factor"],
        lr_decay_interval=config["lr_decay_interval"],
    )


def train_contrastive_final(dev_data: FinalDevelopmentData, config: dict,
                            *, epochs: int = FINAL_EPOCHS,
                            embedding_dim: int = FINAL_EMBEDDING_DIM,
                            checkpoint_path: str = DEFAULT_CHECKPOINT,
                            seed: int = 42) -> Dict:
    """Self-supervised final training on ALL development rows for exactly ``epochs``.

    NO validation loader, NO EarlyStopping, NO ModelCheckpoint selection. Batch
    size and DataLoader semantics match the production CV runner
    (``min(N_train, 512)``, shuffle, ``drop_last=False``).
    """
    pl.seed_everything(seed)
    model = build_contrastive_final_model(dev_data, config, embedding_dim=embedding_dim)

    eff_batch_size = min(len(dev_data.X_dev), 512)
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(dev_data.X_dev), torch.LongTensor(dev_data.y_dev)),
        batch_size=eff_batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        min_epochs=epochs,
        accelerator=config["accelerator"],
        devices=config["devices"],
        callbacks=[],
        enable_progress_bar=True,
        log_every_n_steps=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
        precision=config.get("precision", "32-true"),
    )
    t0 = time.time()
    trainer.fit(model, train_loader)  # no validation loader
    runtime = time.time() - t0

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    trainer.save_checkpoint(checkpoint_path)

    return {
        "model": model,
        "checkpoint_path": checkpoint_path,
        "embedding_dim": embedding_dim,
        "epochs": epochs,
        "batch_size": eff_batch_size,
        "runtime_s": runtime,
        "n_dev": dev_data.n_samples,
        "n_features": dev_data.n_features,
        "seed": seed,
    }


def evaluate_contrastive_final(trained: Dict, dev_data: FinalDevelopmentData,
                               test_data: LockedTestData, *,
                               output_dir: str = DEFAULT_OUTPUT_DIR, k: int = 3) -> Dict:
    """KNN(dev embeddings + dev labels) -> locked-test embeddings, evaluated once."""
    model = trained["model"]
    Z_dev = extract_embeddings(model, dev_data.X_dev)
    Z_test = extract_embeddings(model, test_data.X_test)

    labels = np.arange(test_data.n_classes, dtype=np.int64)
    metrics, y_pred = evaluate_knn_classification(
        Z_dev, dev_data.y_dev, Z_test, test_data.y_test, k=k, labels=labels
    )

    result = {
        "model": "contrastive",
        "embedding_dim": trained["embedding_dim"],
        "epochs": trained["epochs"],
        "knn_k": k,
        "n_train_embeddings": int(len(Z_dev)),
        "n_test": test_data.n_samples,
        "n_features": test_data.n_features,
        "n_classes": test_data.n_classes,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "per_class_recall": metrics["per_class_recall"].tolist(),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "class_names": list(test_data.class_names),
        "predictions": y_pred.tolist(),
        "checkpoint_path": trained["checkpoint_path"],
        "runtime_s": trained["runtime_s"],
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "contrastive_final_summary.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "contrastive_final_full.pkl"), "wb") as f:
        pickle.dump(result, f)

    return result
