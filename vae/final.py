"""PRIMARY RQ1 final-development training + locked-test evaluation for VMGP.

Final training uses the FULL development set (no validation split, no early
stopping, no checkpoint selection) for exactly ``FINAL_EPOCHS`` completed
epochs. The locked test is obtained only through the explicit
``GenomicExperimentData.transform_locked_test`` call, after training, never
inside ``train_vae_final``.
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

from genomic.classification_metrics import compute_classification_metrics
from genomic.experiment_data import FinalDevelopmentData, LockedTestData

from .lightning_module import VMGP_LightningSystem

FINAL_LATENT_DIM = 96
FINAL_EPOCHS = 168
FINAL_PRECISION = "16-mixed"

DEFAULT_CHECKPOINT = "checkpoints/rq1_final/vae_latent96_ep168.ckpt"
DEFAULT_OUTPUT_DIR = "results/rq1_final"


def build_vae_final_model(dev_data: FinalDevelopmentData, config: dict,
                          *, latent_dim: int = FINAL_LATENT_DIM) -> VMGP_LightningSystem:
    """Construct the frozen VMGP (breed-only) for the final development fit."""
    return VMGP_LightningSystem(
        num_snps=dev_data.n_features,
        num_classes_breed=dev_data.n_classes,
        num_classes_continent=0,
        num_classes_caseina=0,
        num_classes_attitudine=0,
        alpha=config["alpha"],
        lr=config["lr"],
        use_breed=True,
        use_continent=False,
        use_caseina=False,
        use_attitudine=False,
        weight_breed=config["weight_breed"],
        weight_continent=config["weight_continent"],
        weight_caseina=config["weight_caseina"],
        weight_attitudine=config.get("weight_attitudine", 1.0),
        latent_dim=latent_dim,
    )


def _dummy_labels(n: int) -> torch.Tensor:
    return torch.LongTensor(np.full(n, -1, dtype=np.int64))


def train_vae_final(dev_data: FinalDevelopmentData, config: dict,
                    *, epochs: int = FINAL_EPOCHS, latent_dim: int = FINAL_LATENT_DIM,
                    checkpoint_path: str = DEFAULT_CHECKPOINT, seed: int = 42) -> Dict:
    """Train the selected VMGP on ALL development rows for exactly ``epochs`` epochs.

    NO validation loader, NO EarlyStopping, NO ModelCheckpoint selection. The
    locked test is never accessed here. DataLoader semantics match the CV runner
    (batch size, shuffle, ``drop_last=True``, ``num_workers=4``, ``pin_memory``).
    """
    pl.seed_everything(seed)
    model = build_vae_final_model(dev_data, config, latent_dim=latent_dim)

    dummy = _dummy_labels(dev_data.n_samples)
    train_ds = TensorDataset(
        torch.FloatTensor(dev_data.X_dev),
        torch.LongTensor(dev_data.y_dev),
        dummy, dummy, dummy,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    trainer = pl.Trainer(
        precision=FINAL_PRECISION,
        max_epochs=epochs,
        min_epochs=epochs,
        accelerator=config["accelerator"],
        devices=config["devices"],
        callbacks=[],
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
    )
    t0 = time.time()
    trainer.fit(model, train_loader)  # no validation loader
    runtime = time.time() - t0

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    trainer.save_checkpoint(checkpoint_path)

    return {
        "model": model,
        "checkpoint_path": checkpoint_path,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "batch_size": config["batch_size"],
        "runtime_s": runtime,
        "n_dev": dev_data.n_samples,
        "n_features": dev_data.n_features,
        "seed": seed,
    }


def evaluate_vae_final(trained: Dict, test_data: LockedTestData, *,
                       output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict:
    """Evaluate the trained VMGP breed head on the locked test exactly once."""
    model = trained["model"]
    model.eval()
    model.freeze()

    dummy = _dummy_labels(test_data.n_samples)
    test_ds = TensorDataset(
        torch.FloatTensor(test_data.X_test),
        torch.LongTensor(test_data.y_test),
        dummy, dummy, dummy,
    )
    test_loader = DataLoader(
        test_ds, batch_size=trained["batch_size"], num_workers=4, pin_memory=True,
    )

    logits = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(model.device)
            outputs = model(x)
            logits.append(outputs["logits_breed"].detach().cpu().numpy())
    logits = np.concatenate(logits)
    y_pred = logits.argmax(axis=1)

    labels = np.arange(test_data.n_classes, dtype=np.int64)
    metrics = compute_classification_metrics(test_data.y_test, y_pred, labels=labels)

    result = {
        "model": "vae",
        "latent_dim": trained["latent_dim"],
        "epochs": trained["epochs"],
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
    with open(os.path.join(output_dir, "vae_final_summary.json"), "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, "vae_final_full.pkl"), "wb") as f:
        pickle.dump(result, f)

    return result
