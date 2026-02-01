"""
Experiment Runner
=================

Gestisce l'esecuzione di esperimenti con K-Fold Cross Validation.
"""

import os
import gc
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score, mean_squared_error
from sklearn.decomposition import PCA
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .data_module import GenomicDataModule
from .lightning_module import VMGP_LightningSystem
from .utils import balance_dataset_multilabel, compute_local_structure, compute_generalization, compute_neighbor_overlap


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


def run_experiment(name: str, config: dict, balanced: bool = False, 
                   coarse_mapping: dict = None, max_epochs: int = 50, 
                   k_folds: int = 2, use_breed: bool = True, 
                   use_continent: bool = False, use_caseina: bool = False,
                   use_attitudine: bool = False, attitudine_mapping: dict = None,
                   classifier_config: str = None) -> dict:
    """
    Esegue un esperimento con configurazione flessibile dei classificatori.
    
    Args:
        name: Nome dell'esperimento
        config: Dizionario di configurazione con hyperparameters
        balanced: Se True, applica SMOTE balancing
        coarse_mapping: Mapping per coarse-grained classification
        max_epochs: Numero massimo di epoch
        k_folds: Numero di fold per cross-validation
        use_breed: Usare classificatore breed
        use_continent: Usare classificatore continent
        use_caseina: Usare classificatore caseina
        use_attitudine: Usare classificatore attitudine
        attitudine_mapping: Mapping breed_prefix -> attitudine (LATTE/FIBRA/CARNE)
        classifier_config: Nome di una configurazione predefinita da CLASSIFIER_CONFIGS
    
    Returns:
        Dizionario con risultati dell'esperimento
    """
    
    # Se specificata una config predefinita, usala
    if classifier_config and classifier_config in CLASSIFIER_CONFIGS:
        cfg = CLASSIFIER_CONFIGS[classifier_config]
        use_breed = cfg['use_breed']
        use_continent = cfg['use_continent']
        use_caseina = cfg['use_caseina']
        use_attitudine = cfg['use_attitudine']
    
    print(f"\n{'='*80}")
    print(f"🧪 EXPERIMENT: {name}")
    print(f"   Balanced: {balanced}")
    print(f"   Coarse Grained: {coarse_mapping is not None}")
    print(f"   Classificatori: Breed={use_breed}, Continent={use_continent}, Caseina={use_caseina}, Attitudine={use_attitudine}")
    print(f"   K-Fold CV: DISABILITATO (Single Split)")
    print(f"{'='*80}\n")
    
    # 1. Setup DataModule
    dm = GenomicDataModule(
        bed_path=config['bed_path'],
        batch_size=config['batch_size'],
        maf_thresh=config['maf_thresh'],
        geno_thresh=config['geno_thresh'],
        min_samples_per_class=config['min_samples_per_class'],
        ld_pruning=config['ld_pruning'],
        ld_threshold=config['ld_threshold'],
        ld_window=config['ld_window'],
        val_split=config['val_split'],
        test_split=config['test_split'],
        coarse_mapping=coarse_mapping,
        metadata_path=config['metadata_path'],
        use_breed=use_breed,
        use_continent=use_continent,
        use_caseina=use_caseina,
        use_attitudine=use_attitudine,
        attitudine_mapping=attitudine_mapping
    )
    
    dm.setup()
    
    # 2. Get processed data for K-Fold
    X = dm.X_processed
    y_breed = dm.y_breed_processed
    y_continent = dm.y_continent_processed
    y_caseina = dm.y_caseina_processed
    y_attitudine = dm.y_attitudine_processed
    
    # 3. Balancing viene fatto DOPO lo split per evitare data leakage!
    # (spostato dentro il loop del fold)
    
    # 4. K-Fold Cross Validation (TEMPORANEAMENTE DISABILITATO - Single Split)
    # skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
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
    best_fold_path = None
    
    # --- SINGLE SPLIT (K-Fold disabilitato) ---
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(y_breed)), test_size=0.2, stratify=y_breed, random_state=42
    )
    k_folds = 1  # Override per report
    
    for fold in range(1):  # Single iteration
        print(f"\n🔹 SINGLE SPLIT (K-Fold disabilitato)")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_breed_train, y_breed_val = y_breed[train_idx], y_breed[val_idx]
        y_continent_train, y_continent_val = y_continent[train_idx], y_continent[val_idx]
        y_caseina_train, y_caseina_val = y_caseina[train_idx], y_caseina[val_idx]
        y_attitudine_train, y_attitudine_val = y_attitudine[train_idx], y_attitudine[val_idx]
        
        # ⚠️ BALANCING SOLO SUI DATI DI TRAINING (per evitare data leakage)
        if balanced:
            print(f"   ⚖️ Balancing SOLO training set (SMOTE multilabel)...")
            X_train, y_breed_train, y_continent_train, y_caseina_train, y_attitudine_train = balance_dataset_multilabel(
                X_train, y_breed_train, y_continent_train, y_caseina_train, 
                config['target_samples'], y_attitudine_train
            )
            print(f"   Training bilanciato: {len(y_breed_train)} samples | Validation (non bilanciato): {len(y_breed_val)} samples")
        
        train_ds = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_breed_train),
            torch.LongTensor(y_continent_train),
            torch.LongTensor(y_caseina_train),
            torch.LongTensor(y_attitudine_train)
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.LongTensor(y_breed_val),
            torch.LongTensor(y_continent_val),
            torch.LongTensor(y_caseina_val),
            torch.LongTensor(y_attitudine_val)
        )
        
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, 
                                  num_workers=4, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], 
                                num_workers=4, pin_memory=True)
        
        model_fold = VMGP_LightningSystem(
            num_snps=dm.num_snps,
            num_classes_breed=dm.num_classes_breed if use_breed else 0,
            num_classes_continent=dm.num_classes_continent if use_continent else 0,
            num_classes_caseina=dm.num_classes_caseina if use_caseina else 0,
            num_classes_attitudine=dm.num_classes_attitudine if use_attitudine else 0,
            alpha=config['alpha'],
            lr=config['lr'],
            use_breed=use_breed,
            use_continent=use_continent,
            use_caseina=use_caseina,
            use_attitudine=use_attitudine,
            weight_breed=config['weight_breed'],
            weight_continent=config['weight_continent'],
            weight_caseina=config['weight_caseina'],
            weight_attitudine=config.get('weight_attitudine', 1.0)
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'checkpoints_vae/{name.replace(" ", "_")}/fold_{fold+1}',
            filename='{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        )
        
        early_stop = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=False)
        
        trainer = pl.Trainer(
            precision="16-mixed",
            max_epochs=max_epochs,
            accelerator=config['accelerator'],
            devices=config['devices'],
            callbacks=[checkpoint_callback, early_stop],
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            check_val_every_n_epoch=1
        )
        
        trainer.fit(model_fold, train_loader, val_loader)
        
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            try:
                model_fold = VMGP_LightningSystem.load_from_checkpoint(best_model_path)
            except Exception as e:
                print(f"   ⚠️ Failed loading best model: {e}")
       
        model_fold.eval()
        model_fold.freeze()
        
        # --- Evaluation ---
        emb_train, lbl_train, snps_train = [], [], []
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
                snps_train.append(x.cpu().numpy())
            
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
        snps_train = np.concatenate(snps_train)
        emb_val = np.concatenate(emb_val)
        lbl_val = np.concatenate(lbl_val)
        snps_val = np.concatenate(snps_val)
        recon_val = np.concatenate(recon_val)
        
        L = compute_local_structure(emb_train, lbl_train)
        GE = compute_generalization(emb_train, lbl_train, emb_val, lbl_val)
        no_metrics = compute_neighbor_overlap(emb_train, snps_train, k_values=[3, 10, 30], max_samples=1000)
        mse_val = mean_squared_error(snps_val, recon_val)
        sil = silhouette_score(emb_train, lbl_train) if len(np.unique(lbl_train)) > 1 else 0
        db = davies_bouldin_score(emb_train, lbl_train) if len(np.unique(lbl_train)) > 1 else 0
        
        if use_breed and preds_breed_val:
            preds_b = np.concatenate(preds_breed_val)
            true_b = np.concatenate(true_breed_val)
            acc_breed = accuracy_score(true_b, preds_b)
            fold_metrics['Acc_Breed'].append(acc_breed)
        
        if use_continent and preds_continent_val:
            preds_c = np.concatenate(preds_continent_val)
            true_c = np.concatenate(true_continent_val)
            valid_mask = true_c >= 0
            if valid_mask.sum() > 0:
                acc_continent = accuracy_score(true_c[valid_mask], preds_c[valid_mask])
                fold_metrics['Acc_Continent'].append(acc_continent)
        
        if use_caseina and preds_caseina_val:
            preds_cas = np.concatenate(preds_caseina_val)
            true_cas = np.concatenate(true_caseina_val)
            valid_mask = true_cas >= 0
            if valid_mask.sum() > 0:
                acc_caseina = accuracy_score(true_cas[valid_mask], preds_cas[valid_mask])
                fold_metrics['Acc_Caseina'].append(acc_caseina)
        
        if use_attitudine and preds_attitudine_val:
            preds_att = np.concatenate(preds_attitudine_val)
            true_att = np.concatenate(true_attitudine_val)
            valid_mask = true_att >= 0
            if valid_mask.sum() > 0:
                acc_attitudine = accuracy_score(true_att[valid_mask], preds_att[valid_mask])
                fold_metrics['Acc_Attitudine'].append(acc_attitudine)
        
        fold_metrics['Local Structure (L)'].append(L)
        fold_metrics['Generalization (GE)'].append(GE)
        fold_metrics['Silhouette'].append(sil)
        fold_metrics['Davies-Bouldin'].append(db)
        fold_metrics['Reconstruction MSE'].append(mse_val)
        fold_metrics['NO_k3'].append(no_metrics['NO_k3'])
        fold_metrics['NO_k10'].append(no_metrics['NO_k10'])
        fold_metrics['NO_k30'].append(no_metrics['NO_k30'])
        
        result_str = f"   ✅ Fold {fold+1} | GE: {GE:.4f} | Sil: {sil:.4f} | MSE: {mse_val:.4f}"
        if use_breed and 'Acc_Breed' in fold_metrics and fold_metrics['Acc_Breed']:
            result_str += f" | Acc_Breed: {fold_metrics['Acc_Breed'][-1]:.4f}"
        if use_continent and 'Acc_Continent' in fold_metrics and fold_metrics['Acc_Continent']:
            result_str += f" | Acc_Cont: {fold_metrics['Acc_Continent'][-1]:.4f}"
        if use_caseina and 'Acc_Caseina' in fold_metrics and fold_metrics['Acc_Caseina']:
            result_str += f" | Acc_Cas: {fold_metrics['Acc_Caseina'][-1]:.4f}"
        print(result_str)
        
        if GE > best_fold_acc:
            best_fold_acc = GE
            best_fold_idx = fold
            best_fold_emb_val = emb_val.copy()
            best_fold_lbl_val = lbl_val.copy()
            best_fold_path = best_model_path
        
        del model_fold, trainer, train_loader, val_loader, train_ds, val_ds
        del snps_train, recon_val, snps_val, emb_train, lbl_train, emb_val, lbl_val
        torch.cuda.empty_cache()
        gc.collect()
    
    exp_checkpoint_dir = f'checkpoints_vae/{name.replace(" ", "_")}'
    os.makedirs(exp_checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(exp_checkpoint_dir, 'best_model.ckpt')
    
    if best_fold_path and os.path.exists(best_fold_path):
        shutil.copy(best_fold_path, best_checkpoint_path)
        print(f"\n💾 Saved best checkpoint from Fold {best_fold_idx+1}: {best_checkpoint_path}")
    
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
        'Num Classes Breed': dm.num_classes_breed,
        'Num Classes Continent': dm.num_classes_continent,
        'Num Classes Caseina': dm.num_classes_caseina,
        'Num SNPs': dm.num_snps,
        'K Folds': k_folds,
        'Use Breed': use_breed,
        'Use Continent': use_continent,
        'Use Caseina': use_caseina
    }
    
    for metric, values in fold_metrics.items():
        if values:
            results[f'{metric} (mean)'] = np.mean(values)
            results[f'{metric} (std)'] = np.std(values)
    
    # Visualizzazione PCA
    print(f"📈 Creating PCA visualization...")
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(best_fold_emb_val)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(best_fold_lbl_val)
    
    for i, cls_idx in enumerate(unique_labels):
        mask = best_fold_lbl_val == cls_idx
        label_name = dm.class_names_breed[cls_idx] if cls_idx < len(dm.class_names_breed) else str(cls_idx)
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], label=label_name, alpha=0.6, s=20)
    
    plt.title(f'{name} - Best Fold {best_fold_idx+1} (GE: {best_fold_acc:.3f})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig(f'plot_vae_{name.replace(" ", "_")}_kfold.png')
    plt.show()
    
    return results
