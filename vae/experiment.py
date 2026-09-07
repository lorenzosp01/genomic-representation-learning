"""
Experiment Runner
=================

Esegue esperimenti VMGP con Cross Validation sui fold congelati del
protocollo v2 (layer condiviso ``genomic.experiment_data``).

Non esegue alcuno split/preprocessing locale: la coorte, i fold, il
preprocessing (fit sui soli dati di training) e le label canoniche provengono
dal layer condiviso.
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    silhouette_score,
    davies_bouldin_score,
    mean_squared_error,
    confusion_matrix,
    cohen_kappa_score,
)
from sklearn.decomposition import PCA
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from .data_module import build_vae_fold, VAEFoldData
from .lightning_module import VMGP_LightningSystem
from .utils import (
    balance_dataset_multilabel,
    cap_dataset_multilabel,
    compute_local_structure,
    compute_generalization,
    compute_neighbor_overlap,
)


# --- CONFIGURAZIONI ESPERIMENTI ---
CLASSIFIER_CONFIGS = {
    'breed_only': {
        'use_breed': True,
        'use_continent': False,
        'use_caseina': False,
        'use_attitudine': False,
        'description': 'Solo classificatore Breed (razza)'
    },
    'continent_only': {
        'use_breed': False,
        'use_continent': True,
        'use_caseina': False,
        'use_attitudine': False,
        'description': 'Solo classificatore Continent'
    },
    'caseina_only': {
        'use_breed': False,
        'use_continent': False,
        'use_caseina': True,
        'use_attitudine': False,
        'description': 'Solo classificatore Caseina'
    },
    'attitudine_only': {
        'use_breed': False,
        'use_continent': False,
        'use_caseina': False,
        'use_attitudine': True,
        'description': 'Solo classificatore Attitudine (LATTE/FIBRA/CARNE/ALTRO)'
    },
    'breed_continent': {
        'use_breed': True,
        'use_continent': True,
        'use_caseina': False,
        'use_attitudine': False,
        'description': 'Classificatori Breed + Continent'
    },
    'breed_attitudine': {
        'use_breed': True,
        'use_continent': False,
        'use_caseina': False,
        'use_attitudine': True,
        'description': 'Classificatori Breed + Attitudine (LATTE/FIBRA/CARNE/ALTRO)'
    },
    'all_classifiers': {
        'use_breed': True,
        'use_continent': True,
        'use_caseina': True,
        'use_attitudine': False,
        'description': 'Tutti i classificatori (Breed + Continent + Caseina)'
    },
    'all_with_attitudine': {
        'use_breed': True,
        'use_continent': True,
        'use_caseina': True,
        'use_attitudine': True,
        'description': 'Tutti i classificatori inclusa Attitudine'
    },
    'vae_only': {
        'use_breed': False,
        'use_continent': False,
        'use_caseina': False,
        'use_attitudine': False,
        'description': 'Solo VAE (nessun classificatore)'
    }
}


def _build_folds(experiment_data, config, use_breed, use_continent, use_caseina,
                 use_attitudine, attitudine_mapping):
    """Precompute the VAE fold data once (preprocessing is model-independent)."""
    class_names_breed = experiment_data.class_names
    folds = []
    for k in range(experiment_data.split.n_folds):
        fd = experiment_data.fold_data(k)
        folds.append(build_vae_fold(
            fd,
            class_names_breed=class_names_breed,
            metadata_path=config.get('metadata_path'),
            attitudine_mapping=attitudine_mapping,
            use_continent=use_continent,
            use_caseina=use_caseina,
            use_attitudine=use_attitudine,
        ))
    return folds


def run_experiment(name: str, experiment_data, config: dict,
                   max_epochs: int = 50,
                   use_breed: bool = True,
                   use_continent: bool = False,
                   use_caseina: bool = False,
                   use_attitudine: bool = False,
                   attitudine_mapping: dict = None,
                   classifier_config: str = None,
                   balanced: bool = False,
                   cap_samples: bool = False,
                   coarse_mapping: dict = None,
                   folds: list = None,
                   save_checkpoint_path: str = None) -> dict:
    """Esegue un esperimento VMGP sui fold congelati del protocollo v2.

    Args:
        name: Nome dell'esperimento.
        experiment_data: ``GenomicExperimentData`` (coorte + fold condivisi).
        config: Configurazione (alpha, lr, latent_dim, target_samples, ...).
        max_epochs: Numero massimo di epoch per fold.
        use_breed/use_continent/use_caseina/use_attitudine: task attivi.
        classifier_config: Nome di una config predefinita (override dei flag).
        balanced/cap_samples: politica di balancing (solo sul TRAIN trasformato).
        folds: lista precomputata di ``VAEFoldData`` (riusata tra hyperparametri).
        save_checkpoint_path: percorso per il checkpoint del best fold.

    Returns:
        Dizionario con i risultati dell'esperimento (nessuna valutazione sul
        locked test).
    """
    if classifier_config and classifier_config in CLASSIFIER_CONFIGS:
        cfg = CLASSIFIER_CONFIGS[classifier_config]
        use_breed = cfg['use_breed']
        use_continent = cfg['use_continent']
        use_caseina = cfg['use_caseina']
        use_attitudine = cfg['use_attitudine']

    print(f"\n{'='*80}")
    print(f"🧪 EXPERIMENT: {name}")
    print(f"   Balanced: {balanced}")
    print(f"   Cap Samples: {cap_samples}")
    print(f"   Coarse Grained: {coarse_mapping is not None}")
    print(f"   Classificatori: Breed={use_breed}, Continent={use_continent}, "
          f"Caseina={use_caseina}, Attitudine={use_attitudine}")
    print(f"   CV: fold congelati (protocollo v2, nessuno split locale)")
    print(f"{'='*80}\n")

    if folds is None:
        folds = _build_folds(experiment_data, config, use_breed, use_continent,
                             use_caseina, use_attitudine, attitudine_mapping)

    n_folds = len(folds)
    class_names_breed = experiment_data.class_names

    fold_metrics = {
        'Local Structure (L)': [],
        'Generalization (GE)': [],
        'Silhouette': [],
        'Davies-Bouldin': [],
        'Reconstruction MSE': [],
        'NO_k3': [],
        'NO_k10': [],
        'NO_k30': []
    }
    if use_breed:
        fold_metrics['Acc_Breed'] = []
        fold_metrics['Kappa_Breed'] = []
    if use_continent:
        fold_metrics['Acc_Continent'] = []
    if use_caseina:
        fold_metrics['Acc_Caseina'] = []
    if use_attitudine:
        fold_metrics['Acc_Attitudine'] = []

    best_fold_idx = None
    best_fold_acc = -1
    best_fold_emb_val = None
    best_fold_lbl_val = None
    best_fold_emb_train = None
    best_fold_lbl_train = None
    best_model = None
    cm_breed = None

    for vfd in folds:
        fold = vfd.fold
        print(f"\n🔹 Fold {fold+1}/{n_folds} (num_snps={vfd.num_snps})")

        X_train = vfd.X_train
        y_breed_train = vfd.y_breed_train
        y_continent_train = vfd.y_continent_train
        y_caseina_train = vfd.y_caseina_train
        y_attitudine_train = vfd.y_attitudine_train
        X_val = vfd.X_val
        y_breed_val = vfd.y_breed_val
        y_continent_val = vfd.y_continent_val
        y_caseina_val = vfd.y_caseina_val
        y_attitudine_val = vfd.y_attitudine_val

        # ⚠️ BALANCING SOLO SUI DATI DI TRAINING (dopo il preprocessing)
        if cap_samples:
            print(f"   ✂️ Capping SOLO training set a {config['target_samples']} samples/classe...")
            X_train, y_breed_train, y_continent_train, y_caseina_train, y_attitudine_train = cap_dataset_multilabel(
                X_train, y_breed_train, y_continent_train, y_caseina_train,
                config['target_samples'], y_attitudine_train
            )
        elif balanced:
            print(f"   ⚖️ Balancing SOLO training set (SMOTE multilabel)...")
            X_train, y_breed_train, y_continent_train, y_caseina_train, y_attitudine_train = balance_dataset_multilabel(
                X_train, y_breed_train, y_continent_train, y_caseina_train,
                config['target_samples'], y_attitudine_train
            )

        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_breed_train),
            torch.LongTensor(y_continent_train),
            torch.LongTensor(y_caseina_train),
            torch.LongTensor(y_attitudine_train),
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_breed_val),
            torch.LongTensor(y_continent_val),
            torch.LongTensor(y_caseina_val),
            torch.LongTensor(y_attitudine_val),
        )

        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'],
                                num_workers=4, pin_memory=True)

        model_fold = VMGP_LightningSystem(
            num_snps=vfd.num_snps,
            num_classes_breed=vfd.num_classes_breed if use_breed else 0,
            num_classes_continent=vfd.num_classes_continent if use_continent else 0,
            num_classes_caseina=vfd.num_classes_caseina if use_caseina else 0,
            num_classes_attitudine=vfd.num_classes_attitudine if use_attitudine else 0,
            alpha=config['alpha'],
            lr=config['lr'],
            use_breed=use_breed,
            use_continent=use_continent,
            use_caseina=use_caseina,
            use_attitudine=use_attitudine,
            weight_breed=config['weight_breed'],
            weight_continent=config['weight_continent'],
            weight_caseina=config['weight_caseina'],
            weight_attitudine=config.get('weight_attitudine', 1.0),
            latent_dim=config.get('latent_dim', 96)
        )

        early_stop = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=False)
        trainer = pl.Trainer(
            precision="16-mixed",
            max_epochs=max_epochs,
            accelerator=config['accelerator'],
            devices=config['devices'],
            callbacks=[early_stop],
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            check_val_every_n_epoch=1
        )
        trainer.fit(model_fold, train_loader, val_loader)

        model_fold.eval()
        model_fold.freeze()

        # --- Evaluation ---
        emb_train, lbl_train = [], []
        emb_val, lbl_val, snps_val, recon_val = [], [], [], []
        preds_breed_val, preds_continent_val, preds_caseina_val, preds_attitudine_val = [], [], [], []
        true_breed_val, true_continent_val, true_caseina_val, true_attitudine_val = [], [], [], []

        with torch.no_grad():
            for batch in train_loader:
                x, yb, yc, ycas, yatt = batch
                x = x.to(model_fold.device)
                outputs = model_fold(x)
                emb_train.append(outputs['mu'].cpu().numpy())
                lbl_train.append(yb.cpu().numpy())

            for batch in val_loader:
                x, yb, yc, ycas, yatt = batch
                x = x.to(model_fold.device)
                outputs = model_fold(x)

                emb_val.append(outputs['mu'].cpu().numpy())
                lbl_val.append(yb.cpu().numpy())
                snps_val.append(x.cpu().numpy())
                recon_val.append(outputs['x_recon'].cpu().numpy())

                true_breed_val.append(yb.cpu().numpy())
                true_continent_val.append(yc.cpu().numpy())
                true_caseina_val.append(ycas.cpu().numpy())
                true_attitudine_val.append(yatt.cpu().numpy())

                if use_breed and 'logits_breed' in outputs:
                    preds_breed_val.append(torch.argmax(outputs['logits_breed'], dim=1).cpu().numpy())
                if use_continent and 'logits_continent' in outputs:
                    preds_continent_val.append(torch.argmax(outputs['logits_continent'], dim=1).cpu().numpy())
                if use_caseina and 'logits_caseina' in outputs:
                    preds_caseina_val.append(torch.argmax(outputs['logits_caseina'], dim=1).cpu().numpy())
                if use_attitudine and 'logits_attitudine' in outputs:
                    preds_attitudine_val.append(torch.argmax(outputs['logits_attitudine'], dim=1).cpu().numpy())

        emb_train = np.concatenate(emb_train)
        lbl_train = np.concatenate(lbl_train)
        emb_val = np.concatenate(emb_val)
        lbl_val = np.concatenate(lbl_val)
        snps_val = np.concatenate(snps_val)
        recon_val = np.concatenate(recon_val)

        L = compute_local_structure(emb_val, lbl_val)
        GE = compute_generalization(emb_train, lbl_train, emb_val, lbl_val)
        no_metrics = compute_neighbor_overlap(emb_val, snps_val, k_values=[3, 10, 30], max_samples=1000)
        mse_val = mean_squared_error(snps_val, recon_val)
        sil = silhouette_score(emb_val, lbl_val) if len(np.unique(lbl_val)) > 1 else 0
        db = davies_bouldin_score(emb_val, lbl_val) if len(np.unique(lbl_val)) > 1 else 0

        if use_breed and preds_breed_val:
            preds_b = np.concatenate(preds_breed_val)
            true_b = np.concatenate(true_breed_val)
            acc_breed = accuracy_score(true_b, preds_b)
            fold_metrics['Acc_Breed'].append(acc_breed)
            fold_metrics['Kappa_Breed'].append(cohen_kappa_score(true_b, preds_b))
            cm_breed = confusion_matrix(true_b, preds_b)

            cm_dir = 'results/confusion_matrices'
            os.makedirs(cm_dir, exist_ok=True)
            cm_df = pd.DataFrame(cm_breed, index=class_names_breed, columns=class_names_breed)
            cm_df.to_csv(f'{cm_dir}/cm_{name.replace(" ", "_")}.csv')

        if use_continent and preds_continent_val:
            preds_c = np.concatenate(preds_continent_val)
            true_c = np.concatenate(true_continent_val)
            valid_mask = true_c >= 0
            if valid_mask.sum() > 0:
                fold_metrics['Acc_Continent'].append(accuracy_score(true_c[valid_mask], preds_c[valid_mask]))

        if use_caseina and preds_caseina_val:
            preds_cas = np.concatenate(preds_caseina_val)
            true_cas = np.concatenate(true_caseina_val)
            valid_mask = true_cas >= 0
            if valid_mask.sum() > 0:
                fold_metrics['Acc_Caseina'].append(accuracy_score(true_cas[valid_mask], preds_cas[valid_mask]))

        if use_attitudine and preds_attitudine_val:
            preds_att = np.concatenate(preds_attitudine_val)
            true_att = np.concatenate(true_attitudine_val)
            valid_mask = true_att >= 0
            if valid_mask.sum() > 0:
                fold_metrics['Acc_Attitudine'].append(accuracy_score(true_att[valid_mask], preds_att[valid_mask]))

        fold_metrics['Local Structure (L)'].append(L)
        fold_metrics['Generalization (GE)'].append(GE)
        fold_metrics['Silhouette'].append(sil)
        fold_metrics['Davies-Bouldin'].append(db)
        fold_metrics['Reconstruction MSE'].append(mse_val)
        fold_metrics['NO_k3'].append(no_metrics['NO_k3'])
        fold_metrics['NO_k10'].append(no_metrics['NO_k10'])
        fold_metrics['NO_k30'].append(no_metrics['NO_k30'])

        result_str = f"   ✅ Fold {fold+1} | GE: {GE:.4f} | Sil: {sil:.4f} | MSE: {mse_val:.4f}"
        if use_breed and fold_metrics['Acc_Breed']:
            result_str += f" | Acc_Breed: {fold_metrics['Acc_Breed'][-1]:.4f}"
        print(result_str)

        if GE > best_fold_acc:
            best_fold_acc = GE
            best_fold_idx = fold
            best_fold_emb_val = emb_val.copy()
            best_fold_lbl_val = lbl_val.copy()
            best_fold_emb_train = emb_train.copy()
            best_fold_lbl_train = lbl_train.copy()
            best_model = model_fold
            if save_checkpoint_path:
                os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True)
                trainer.save_checkpoint(save_checkpoint_path)
                print(f"   💾 Checkpoint best fold salvato: {save_checkpoint_path}")

        del trainer, train_loader, val_loader, train_ds, val_ds
        del snps_val, recon_val, emb_train, lbl_train, emb_val, lbl_val
        if model_fold is not best_model:
            del model_fold
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*80}")
    print(f"📊 K-FOLD RESULTS (Mean ± Std):")
    for metric, values in fold_metrics.items():
        if values:
            print(f"   {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    print(f"   Best Fold: {best_fold_idx + 1} (GE: {best_fold_acc:.4f})")
    print(f"{'='*80}\n")

    results = {
        'Experiment': name,
        'Best Fold': best_fold_idx + 1,
        'Best Fold GE': best_fold_acc,
        'Num Classes Breed': folds[0].num_classes_breed if use_breed else 0,
        'Num Classes Continent': folds[0].num_classes_continent,
        'Num Classes Caseina': folds[0].num_classes_caseina,
        'Num SNPs': folds[0].num_snps,
        'K Folds': n_folds,
        'Use Breed': use_breed,
        'Use Continent': use_continent,
        'Use Caseina': use_caseina,
        'Class Names Breed': class_names_breed,
        'Confusion Matrix Breed': cm_breed,
    }
    for metric, values in fold_metrics.items():
        if values:
            results[f'{metric} (mean)'] = np.mean(values)
            results[f'{metric} (std)'] = np.std(values)

    # Visualizzazione PCA del best fold
    if best_fold_emb_val is not None:
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(best_fold_emb_val)
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(best_fold_lbl_val)
        for i, cls_idx in enumerate(unique_labels):
            mask = best_fold_lbl_val == cls_idx
            label_name = class_names_breed[cls_idx] if cls_idx < len(class_names_breed) else str(cls_idx)
            plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=label_name, alpha=0.6, s=20)
        plt.title(f'{name} - Best Fold {best_fold_idx+1} (GE: {best_fold_acc:.3f})')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
        plt.tight_layout()
        plt.savefig(f'plot_vae_{name.replace(" ", "_")}_kfold.png')
        plt.show()

    return results


def run_grid(experiment_data, config: dict, latent_dim_range: list,
             all_results: list = None, cache_path: str = "results_cache.json") -> tuple:
    """Run the latent_dim grid over the frozen protocol-v2 cohort.

    ``min_samples`` is no longer a grid dimension (cohort membership is frozen).
    Fold preprocessing is computed once and reused across ``latent_dim`` values
    (preprocessing is independent of model hyperparameters).
    """
    from .results_io import save_results_incrementally

    if all_results is None:
        all_results = []

    print("📦 Pre-caricamento fold condivisi (preprocessing una sola volta)...")
    folds = _build_folds(experiment_data, config, True, False, False, False, None)
    print(f"📦 Pre-caricamento completato ({len(folds)} fold).\n")

    grid_results = []
    total_experiments = len(latent_dim_range)
    exp_counter = 0

    for latent_dim in latent_dim_range:
        exp_counter += 1
        exp_config = config.copy()
        exp_config['latent_dim'] = latent_dim
        exp_name = f"LatDim_{latent_dim}"

        print(f"\n{'='*60}")
        print(f"🔬 [{exp_counter}/{total_experiments}] Esperimento: {exp_name}")
        print(f"{'='*60}")

        try:
            res = run_experiment(
                name=exp_name,
                experiment_data=experiment_data,
                config=exp_config,
                max_epochs=config['max_epochs'],
                classifier_config='breed_only',
                balanced=False,
                cap_samples=True,
                folds=folds,
            )
            res['Latent_Dim'] = latent_dim
            grid_results.append(res)
            all_results.append(res)
            save_results_incrementally(all_results, path=cache_path)
        except Exception as e:
            print(f"⚠️ Errore nell'esperimento {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✅ Griglia esperimenti completata! ({len(grid_results)}/{total_experiments} riusciti)")
    return grid_results, all_results


def run_single(experiment_data, config: dict, latent_dim: int,
               save_checkpoint_path: str = None) -> tuple:
    """Run a single experiment over the frozen protocol-v2 cohort.

    Returns ``(single_result, exp_name)``.
    """
    folds = _build_folds(experiment_data, config, True, False, False, False, None)

    single_config = config.copy()
    single_config['latent_dim'] = latent_dim

    exp_name = f"LatDim_{latent_dim}"
    print(f"🔬 Training: {exp_name}")
    print(f"   K-Fold: {len(folds)} | Max Epochs: {config['max_epochs']}")

    if save_checkpoint_path is None:
        save_checkpoint_path = f"checkpoints/{exp_name}.ckpt"

    single_result = run_experiment(
        name=exp_name,
        experiment_data=experiment_data,
        config=single_config,
        max_epochs=config['max_epochs'],
        classifier_config='breed_only',
        balanced=False,
        cap_samples=True,
        folds=folds,
        save_checkpoint_path=save_checkpoint_path,
    )

    single_result['Latent_Dim'] = latent_dim
    print(f"\n✅ Training completato!")
    return single_result, exp_name
