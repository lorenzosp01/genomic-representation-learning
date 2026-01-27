"""
Model Analysis
==============

Analisi approfondita dei modelli trainati:
- Matrici di confusione
- Metriche dettagliate
- Visualizzazioni
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, silhouette_score, 
    davies_bouldin_score, mean_squared_error
)

from .data_module import GenomicDataModule
from .lightning_module import VMGP_LightningSystem
from .utils import compute_local_structure, compute_generalization, compute_neighbor_overlap


def analyze_model_performance(experiment_name: str, config: dict, 
                               coarse_mapping: dict = None) -> dict:
    """
    Analisi approfondita delle performance di un modello trainato.
    
    Args:
        experiment_name: Nome dell'esperimento
        config: Dizionario di configurazione
        coarse_mapping: Mapping per coarse-grained (se usato)
    
    Returns:
        Dizionario con tutte le metriche calcolate
    """
    print(f"\n{'='*80}")
    print(f"🔬 ANALISI PERFORMANCE APPROFONDITA: {experiment_name}")
    print(f"{'='*80}\n")
    
    # 1. Caricamento Modello
    checkpoint_path = f"checkpoints_vae/{experiment_name.replace(' ', '_')}/best_model.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint non trovato: {checkpoint_path}")
        return None

    print(f"   Caricamento modello da: {checkpoint_path}")
    try:
        model = VMGP_LightningSystem.load_from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Errore caricamento modello: {e}")
        return None
        
    model.eval()
    model.freeze()
    
    # Estrai configurazione classificatori dal modello caricato
    use_breed = getattr(model, 'use_breed', True)
    use_continent = getattr(model, 'use_continent', False)
    use_caseina = getattr(model, 'use_caseina', False)
    use_attitudine = getattr(model, 'use_attitudine', False)
    
    print(f"   Configurazione classificatori dal checkpoint:")
    print(f"      use_breed={use_breed}, use_continent={use_continent}, use_caseina={use_caseina}, use_attitudine={use_attitudine}")
    
    # 2. Caricamento Dati
    use_coarse = "Coarse" in experiment_name
    mapping = coarse_mapping if use_coarse else None
    
    dm_viz = GenomicDataModule(
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
        coarse_mapping=mapping,
        metadata_path=config.get('metadata_path'),
        use_breed=use_breed,
        use_continent=use_continent,
        use_caseina=use_caseina,
        use_attitudine=use_attitudine
    )
    dm_viz.setup()
    train_loader = dm_viz.train_dataloader()
    val_loader = dm_viz.val_dataloader()
    
    # 3. Estrazione Embeddings e Predizioni
    
    # --- TRAIN SET ---
    print("   Estrazione feature Training Set...")
    emb_train = []
    lbl_train = []
    snps_train = []
    
    with torch.no_grad():
        for batch in train_loader:
            x = batch[0]
            y = batch[1]
            x = x.to(model.device)
            outputs = model(x)
            
            emb_train.append(outputs['mu'].cpu().numpy())
            lbl_train.append(y.cpu().numpy())
            snps_train.append(x.cpu().numpy())
            
    emb_train = np.concatenate(emb_train)
    lbl_train = np.concatenate(lbl_train)
    snps_train = np.concatenate(snps_train)
    
    # --- VAL SET ---
    print("   Estrazione feature Validation Set...")
    emb_val = []
    lbl_val = []
    snps_val = []
    recon_val = []
    preds_breed_val = []
    preds_continent_val = []
    preds_caseina_val = []
    true_continent_val = []
    true_caseina_val = []
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0]
            y_breed = batch[1]
            x = x.to(model.device)
            outputs = model(x)
            
            emb_val.append(outputs['mu'].cpu().numpy())
            lbl_val.append(y_breed.cpu().numpy())
            snps_val.append(x.cpu().numpy())
            recon_val.append(outputs['x_recon'].cpu().numpy())
            
            # Estrai predizioni per ogni classificatore attivo
            if use_breed and 'logits_breed' in outputs:
                preds_breed_val.append(torch.argmax(outputs['logits_breed'], dim=1).cpu().numpy())
            
            if use_continent and 'logits_continent' in outputs:
                preds_continent_val.append(torch.argmax(outputs['logits_continent'], dim=1).cpu().numpy())
                if len(batch) > 2:
                    true_continent_val.append(batch[2].cpu().numpy())
            
            if use_caseina and 'logits_caseina' in outputs:
                preds_caseina_val.append(torch.argmax(outputs['logits_caseina'], dim=1).cpu().numpy())
                if len(batch) > 3:
                    true_caseina_val.append(batch[3].cpu().numpy())
            
    emb_val = np.concatenate(emb_val)
    lbl_val = np.concatenate(lbl_val)
    snps_val = np.concatenate(snps_val)
    recon_val = np.concatenate(recon_val)
    
    # 4. Calcolo Metriche
    print("\n📊 CALCOLO METRICHE...")
    
    # A. Metriche Latent Space
    L = compute_local_structure(emb_train, lbl_train)
    GE = compute_generalization(emb_train, lbl_train, emb_val, lbl_val)
    sil = silhouette_score(emb_train, lbl_train) if len(np.unique(lbl_train)) > 1 else 0
    db = davies_bouldin_score(emb_train, lbl_train) if len(np.unique(lbl_train)) > 1 else 0
    
    # B. Neighbor Overlap
    no_metrics = compute_neighbor_overlap(emb_train, snps_train, k_values=[3, 10, 30], max_samples=1000)
    
    # C. Metriche Ricostruzione
    mse_val = mean_squared_error(snps_val, recon_val)
    
    # Genotype Accuracy
    x_true_int = np.clip(np.round(snps_val), 0, 2).astype(int).flatten()
    x_recon_int = np.clip(np.round(recon_val), 0, 2).astype(int).flatten()
    genotype_acc = np.mean(x_true_int == x_recon_int)
    
    # D. Metriche Predittore (per ogni classificatore attivo)
    acc_breed = 0
    acc_continent = 0
    acc_caseina = 0
    
    if use_breed and preds_breed_val:
        preds_b = np.concatenate(preds_breed_val)
        acc_breed = accuracy_score(lbl_val, preds_b)
        print(f"   Acc Breed: {acc_breed:.4f}")
    
    if use_continent and preds_continent_val and true_continent_val:
        preds_c = np.concatenate(preds_continent_val)
        true_c = np.concatenate(true_continent_val)
        valid_mask = true_c >= 0
        if valid_mask.sum() > 0:
            acc_continent = accuracy_score(true_c[valid_mask], preds_c[valid_mask])
            print(f"   Acc Continent: {acc_continent:.4f}")
    
    if use_caseina and preds_caseina_val and true_caseina_val:
        preds_cas = np.concatenate(preds_caseina_val)
        true_cas = np.concatenate(true_caseina_val)
        valid_mask = true_cas >= 0
        if valid_mask.sum() > 0:
            acc_caseina = accuracy_score(true_cas[valid_mask], preds_cas[valid_mask])
            print(f"   Acc Caseina: {acc_caseina:.4f}")
    
    # Predictor Accuracy: usa breed se attivo, altrimenti la migliore tra continent/caseina
    if use_breed:
        acc_predictor = acc_breed
    elif use_continent:
        acc_predictor = acc_continent
    elif use_caseina:
        acc_predictor = acc_caseina
    else:
        acc_predictor = 0  # VAE only
    
    results_dict = {
        'Experiment': experiment_name,
        'Local Structure (L)': L,
        'Generalization (GE)': GE,
        'Predictor Accuracy': acc_predictor,
        'Genotype Accuracy': genotype_acc,
        'Reconstruction MSE': mse_val,
        'Silhouette': sil,
        'Davies-Bouldin': db,
        'NO_k3': no_metrics['NO_k3'],
        'NO_k10': no_metrics['NO_k10'],
        'NO_k30': no_metrics['NO_k30']
    }
    
    # 5. Visualizzazioni
    
    # --- A. MATRICE DI CONFUSIONE CLASSIFICAZIONE ---
    if use_breed and preds_breed_val:
        preds_b = np.concatenate(preds_breed_val)
        n_classes = len(dm_viz.class_names_breed)
        n_show = min(30, n_classes)
        
        print(f"\n📉 Visualizzazione Matrice di Confusione (Prime {n_show}/{n_classes} classi)...")
        
        cm = confusion_matrix(lbl_val, preds_b)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        cm_subset = cm_norm[:n_show, :n_show]
        subset_classes = dm_viz.class_names_breed[:n_show]
        
        figsize_dim = n_show * 0.8
        plt.figure(figsize=(figsize_dim, figsize_dim * 0.8))
        
        annot_labels = np.where(cm_subset > 0.01, np.char.mod('%.2f', cm_subset), "")
        
        sns.heatmap(cm_subset, annot=annot_labels, fmt='', cmap='Blues', 
                    xticklabels=subset_classes, yticklabels=subset_classes,
                    linewidths=1.0, linecolor='lightgray', square=True,
                    cbar_kws={"shrink": .8, "label": "Recall (Normalized by Row)"}, 
                    annot_kws={"size": 12, "weight": "bold"})
                    
        plt.title(f'Confusion Matrix (First {n_show} Classes) - {experiment_name}', fontsize=18, pad=20)
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()
        
        filename_cm = f"confusion_matrix_{experiment_name.replace(' ', '_')}.png"
        plt.savefig(filename_cm, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved confusion matrix to {filename_cm}")
        plt.show()
    else:
        print(f"\n⚠️ Nessuna confusion matrix breed (use_breed={use_breed})")
    
    # --- B. MATRICE DI CONFUSIONE RICOSTRUZIONE ---
    if len(x_true_int) > 1_000_000:
        idx = np.random.choice(len(x_true_int), 1_000_000, replace=False)
        x_true_sample = x_true_int[idx]
        x_recon_sample = x_recon_int[idx]
    else:
        x_true_sample = x_true_int
        x_recon_sample = x_recon_int
        
    cm_geno = confusion_matrix(x_true_sample, x_recon_sample, labels=[0, 1, 2])
    cm_geno_norm = cm_geno.astype('float') / (cm_geno.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_geno_norm, annot=True, fmt='.2%', cmap='Greens', 
                xticklabels=['0 (AA)', '1 (AB)', '2 (BB)'], 
                yticklabels=['0 (AA)', '1 (AB)', '2 (BB)'])
    plt.title(f'Genotype Reconstruction Confusion Matrix (Normalized)\n(Sampled {len(x_true_sample)} SNPs)')
    plt.ylabel('True Genotype')
    plt.xlabel('Reconstructed Genotype (Rounded)')
    plt.tight_layout()
    plt.show()
    
    print(f"\n🧬 METRICA INTUITIVA RICOSTRUZIONE:")
    print(f"   Genotype Accuracy (Accuratezza Genotipica): {genotype_acc*100:.2f}%")
    
    return results_dict


def run_full_analysis(experiments: list, config: dict, 
                       coarse_mapping: dict = None,
                       save_csv: bool = True) -> pd.DataFrame:
    """
    Esegue l'analisi completa su una lista di esperimenti.
    
    Args:
        experiments: Lista di nomi degli esperimenti
        config: Configurazione
        coarse_mapping: Mapping coarse-grained
        save_csv: Se salvare i risultati in CSV
    
    Returns:
        DataFrame con tutti i risultati
    """
    all_results = []
    
    for exp in experiments:
        res = analyze_model_performance(exp, config, coarse_mapping)
        if res:
            all_results.append(res)
    
    if all_results:
        final_df = pd.DataFrame(all_results)
        
        cols = ['Experiment', 'Local Structure (L)', 'Generalization (GE)', 'Predictor Accuracy', 
                'Genotype Accuracy', 'Reconstruction MSE', 'Silhouette', 'Davies-Bouldin', 
                'NO_k3', 'NO_k10', 'NO_k30']
        cols = [c for c in cols if c in final_df.columns]
        final_df = final_df[cols]
        
        if save_csv:
            # Usa path relativo alla directory corrente
            output_path = "final_results_vae_detailed.csv"
            final_df.to_csv(output_path, index=False)
            print(f"\n\n✅ Tutti i risultati salvati in {output_path}")
        
        print(final_df)
        return final_df
    
    return pd.DataFrame()
