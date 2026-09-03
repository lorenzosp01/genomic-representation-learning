"""
Genomic Data Module
===================

Caricamento e preprocessing dati genomici PLINK:
1. Rimozione SNP con troppi missing values (GENO)
2. Rimozione SNP con Minor Allele Frequency bassa (MAF)
3. LD Pruning
4. Imputazione valori mancanti
5. Multi-label support (Breed, Continent, Caseina)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from bed_reader import open_bed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


class GenomicDataModule(pl.LightningDataModule):
    """
    DataModule per dataset genomici PLINK con supporto multi-task.
    
    Args:
        bed_path: Path al file .bed PLINK
        metadata_path: Path al CSV con metadati (Continent, Caseina)
        batch_size: Dimensione batch
        maf_thresh: Soglia Minor Allele Frequency
        geno_thresh: Soglia missingness SNP
        min_samples_per_class: Minimo campioni per classe
        ld_pruning: Abilitare LD pruning
        ld_threshold: Soglia r² per LD pruning
        ld_window: Finestra SNP per LD pruning
        val_split: Frazione validation set
        test_split: Frazione test set
        coarse_mapping: Mapping per coarse-grained classification
        use_breed: Usare classificatore breed
        use_continent: Usare classificatore continent
        use_caseina: Usare classificatore caseina
        use_attitudine: Usare classificatore attitudine (LATTE/FIBRA/CARNE/ALTRO)
        attitudine_mapping: Mapping breed_prefix -> attitudine
    """
    def __init__(self, bed_path, batch_size=32, maf_thresh=0.01, geno_thresh=0.1,
                 min_samples_per_class=10, ld_pruning=True, ld_threshold=0.2,
                 ld_window=50, val_split=0.15, test_split=0.15, coarse_mapping=None,
                 metadata_path=None, use_breed=True, use_continent=False, use_caseina=False,
                 use_attitudine=False, attitudine_mapping=None):
        super().__init__()
        self.bed_path = bed_path
        self.batch_size = batch_size
        self.maf_thresh = maf_thresh
        self.geno_thresh = geno_thresh
        self.min_samples = min_samples_per_class
        self.ld_pruning = ld_pruning
        self.ld_threshold = ld_threshold
        self.ld_window = ld_window
        self.val_split = val_split
        self.test_split = test_split
        self.coarse_mapping = coarse_mapping
        self.metadata_path = metadata_path
        self.use_breed = use_breed
        self.use_continent = use_continent
        self.use_caseina = use_caseina
        self.use_attitudine = use_attitudine
        self.attitudine_mapping = attitudine_mapping
        
        # Label Encoders
        self.label_encoder_breed = LabelEncoder()
        self.label_encoder_continent = LabelEncoder()
        self.label_encoder_caseina = LabelEncoder()
        self.label_encoder_attitudine = LabelEncoder()
        
        self.imputer = SimpleImputer(strategy='mean')
        
        # Stored data
        self.X_processed = None
        self.y_breed_processed = None
        self.y_continent_processed = None
        self.y_caseina_processed = None
        self.y_attitudine_processed = None
        
        self.num_snps = None
        self.num_classes_breed = 0
        self.num_classes_continent = 0
        self.num_classes_caseina = 0
        self.num_classes_attitudine = 0
        self.class_names_breed = []
        self.class_names_continent = []
        self.class_names_caseina = []
        self.class_names_attitudine = []

    def _load_metadata(self):
        """Carica metadati da CSV."""
        if self.metadata_path is None:
            return None
        
        try:
            df = pd.read_csv(self.metadata_path)
            print(f"📋 Metadata caricato: {len(df)} razze")
            return df
        except Exception as e:
            print(f"⚠️ Errore caricamento metadata: {e}")
            return None

    def _match_breed_to_metadata(self, breed_code, metadata_df):
        """Cerca corrispondenza breed -> metadata."""
        if metadata_df is None:
            return {'continent': None, 'caseina': None}
        
        # Cerca per codice breed (può essere esatto o con prefisso)
        # Prima prova match esatto
        match = metadata_df[metadata_df['Breed'] == breed_code]
        
        # Se non trova, prova match con prefisso (es. ALP matches ALP_IT, ALP_FR, ALP_CH)
        if len(match) == 0:
            match = metadata_df[metadata_df['Breed'].str.startswith(breed_code + '_')]
        
        # Se ancora non trova, prova il contrario (es. ALP_IT matches ALP)
        if len(match) == 0:
            prefix = breed_code.split('_')[0] if '_' in breed_code else breed_code
            match = metadata_df[metadata_df['Breed'] == prefix]
        
        if len(match) > 0:
            row = match.iloc[0]
            return {
                'continent': row.get('Continent', None),
                'caseina': row.get('Caseina', None) if pd.notna(row.get('Caseina', None)) and row.get('Caseina', '') != '' else None
            }
        return {'continent': None, 'caseina': None}

    def _get_attitudine(self, breed_code):
        """Mappa breed -> attitudine (LATTE/FIBRA/CARNE/ALTRO)."""
        if self.attitudine_mapping is None:
            return 'ALTRO'
        
        # Estrai prefisso (es. ALP_IT -> ALP)
        prefix = breed_code.split('_')[0] if '_' in breed_code else breed_code
        
        for attitudine, breeds in self.attitudine_mapping.items():
            if prefix in breeds:
                return attitudine
        
        return 'ALTRO'

    def _prune_ld(self, X):
        """LD Pruning con PyTorch per velocità."""
        print(f"4️⃣  LD Pruning (Window={self.ld_window}, r²>{self.ld_threshold})...")
        n_samples, n_snps = X.shape
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"    Device: {device}")
        
        X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
        X_tensor = torch.tensor(X_std, device=device, dtype=torch.float32)
        
        to_remove = np.zeros(n_snps, dtype=bool)
        
        try:
            for i in range(n_snps):
                if to_remove[i]:
                    continue
                
                end = min(i + self.ld_window, n_snps)
                
                # Calcola direttamente senza filtrare gli indici
                r2 = (X_tensor[:, i] @ X_tensor[:, i+1:end] / n_samples) ** 2
                
                high = torch.where(r2 > self.ld_threshold)[0].cpu().numpy() + i + 1
                to_remove[high] = True
                
                if i % 10000 == 0 and i > 0:
                    print(f"    ...{i}/{n_snps} SNP scansionati")
                    
        finally:
            del X_tensor
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        n_removed = to_remove.sum()
        print(f"    ✓ Rimossi {n_removed} SNP ridondanti (LD)")
        
        return X[:, ~to_remove]

    def setup(self, stage=None):
        """Carica e preprocessa i dati."""
        print(f"\n{'='*60}")
        print(f"CARICAMENTO DATASET PLINK: {self.bed_path}")
        print(f"{'='*60}")
        
        metadata = self._load_metadata()

        try:
            # Lettura file PLINK
            G = open_bed(self.bed_path, count_A1=True)
            
            # Lettura FAM per label popolazioni
            fam_path = self.bed_path.replace('.bed', '.fam')
            fam_data = np.loadtxt(fam_path, dtype=str, usecols=0)
            y_raw = fam_data
            y_breeds = y_raw.copy()
            
            # Caricamento genotipi
            X = G.read()
            
            # Gestione orientamento matrice
            if X.shape[1] == len(y_raw):
                print("⚠️  Trasposizione matrice (SNP×Sample → Sample×SNP)...")
                X = X.T
            
            print(f"📊 Shape iniziale: {X.shape} ({X.shape[0]} campioni, {X.shape[1]} SNP)")

            # --- ESTRAZIONE LABEL MULTIPLE ---
            print(f"\n🏷️  Estrazione label multiple...")
            y_continent_raw = []
            y_caseina_raw = []
            y_attitudine_raw = []
            
            if metadata is not None:
                for breed in y_breeds:
                    meta = self._match_breed_to_metadata(breed, metadata)
                    y_continent_raw.append(meta['continent'])
                    y_caseina_raw.append(meta['caseina'])
            else:
                y_continent_raw = [None] * len(y_breeds)
                y_caseina_raw = [None] * len(y_breeds)
            
            # Attitudine (sempre calcolata dal mapping)
            for breed in y_breeds:
                y_attitudine_raw.append(self._get_attitudine(breed))
            
            y_continent_raw = np.array(y_continent_raw, dtype=object)
            y_caseina_raw = np.array(y_caseina_raw, dtype=object)
            y_attitudine_raw = np.array(y_attitudine_raw, dtype=object)
            
            # Statistiche label
            n_continent_valid = np.sum([c is not None for c in y_continent_raw])
            n_caseina_valid = np.sum([c is not None for c in y_caseina_raw])
            print(f"   Continent: {n_continent_valid}/{len(y_breeds)} campioni con label")
            print(f"   Caseina: {n_caseina_valid}/{len(y_breeds)} campioni con label")
            
            if self.use_attitudine:
                unique_att = sorted(set(y_attitudine_raw))
                att_counts = {a: sum(y_attitudine_raw == a) for a in unique_att}
                print(f"   Attitudine: {att_counts}")
            
            if self.use_continent:
                unique_cont = sorted([c for c in set(y_continent_raw) if c is not None])
                print(f"   Continenti: {unique_cont}")
            if self.use_caseina:
                unique_cas = sorted([c for c in set(y_caseina_raw) if c is not None])
                print(f"   Caseina: {unique_cas}")

            # --- FILTRO CLASSI RARE (basato su breed) ---
            unique_classes, counts = np.unique(y_breeds, return_counts=True)
            valid_classes = unique_classes[counts >= self.min_samples]
            mask = np.isin(y_breeds, valid_classes)
            
            n_removed = len(y_breeds) - mask.sum()
            print(f"\n🔍 Filtro Razze Rare (min {self.min_samples} campioni):")
            print(f"   Razze valide: {len(valid_classes)}/{len(unique_classes)}")
            print(f"   Campioni rimossi: {n_removed}")
            
            X = X[mask]
            y_raw = y_raw[mask]
            y_breeds = y_breeds[mask]
            y_continent_raw = y_continent_raw[mask]
            y_caseina_raw = y_caseina_raw[mask]
            y_attitudine_raw = y_attitudine_raw[mask]

            # --- COARSE GRAINED MAPPING ---
            if self.coarse_mapping:
                print(f"\n🌍 Applicazione Coarse-Grained Mapping...")
                new_y = []
                mask_keep = []
                
                breed_to_category = {}
                for cat, breeds in self.coarse_mapping.items():
                    for breed in breeds:
                        breed_to_category[breed] = cat
                
                for label in y_raw:
                    if label in breed_to_category:
                        new_y.append(breed_to_category[label])
                        mask_keep.append(True)
                    else:
                        mask_keep.append(False)
                
                mask_keep = np.array(mask_keep)
                n_dropped = len(y_raw) - mask_keep.sum()
                
                if n_dropped > 0:
                    print(f"   ⚠️ Rimossi {n_dropped} campioni non presenti nel mapping coarse-grained.")
                    X = X[mask_keep]
                    y_raw = np.array(new_y)
                    y_breeds = y_breeds[mask_keep]
                    y_continent_raw = y_continent_raw[mask_keep]
                    y_caseina_raw = y_caseina_raw[mask_keep]
                    y_attitudine_raw = y_attitudine_raw[mask_keep]
                else:
                    y_raw = np.array(new_y)
                    
                print(f"   Nuove classi: {np.unique(y_raw)}")

            # --- QC PIPELINE ---
            print(f"\n🧬 QUALITY CONTROL PIPELINE")
            print(f"{'─'*60}")
            
            # 1. Filtro Missingness SNP
            print("1️⃣  Filtro Missingness SNP...")
            missing_rate_snps = np.isnan(X).mean(axis=0)
            snp_mask = missing_rate_snps < self.geno_thresh
            n_snps_removed = (~snp_mask).sum()
            X = X[:, snp_mask]
            print(f"    ✓ Rimossi {n_snps_removed} SNP con missingness >{self.geno_thresh*100}%")
            
            # 2. Imputazione
            print("2️⃣  Imputazione valori mancanti (media)...")
            X = self.imputer.fit_transform(X)
            print(f"    ✓ Imputazione completata")
            
            # 3. Filtro MAF
            print("3️⃣  Filtro MAF (Minor Allele Frequency)...")
            freq = X.mean(axis=0) / 2
            maf = np.minimum(freq, 1 - freq)
            maf_mask = maf >= self.maf_thresh
            X = X[:, maf_mask]
            print(f"    ✓ Rimossi {(~maf_mask).sum()} SNP con MAF <{self.maf_thresh}")
            
            # 4. LD Pruning
            if self.ld_pruning:
                X = self._prune_ld(X)
            
            print(f"\n{'='*60}")
            print(f"📈 DIMENSIONI FINALI: {X.shape}")
            print(f"{'='*60}")
            
            # --- ENCODING LABELS ---
            # Breed
            y_breed = self.label_encoder_breed.fit_transform(y_raw)
            self.class_names_breed = list(self.label_encoder_breed.classes_)
            self.num_classes_breed = len(self.class_names_breed)
            
            # Continent
            if self.use_continent:
                valid_continent = [c for c in y_continent_raw if c is not None]
                if valid_continent:
                    self.label_encoder_continent.fit(valid_continent)
                    self.class_names_continent = list(self.label_encoder_continent.classes_)
                    self.num_classes_continent = len(self.class_names_continent)
                    
                    y_continent = np.array([
                        self.label_encoder_continent.transform([c])[0] if c is not None else -1
                        for c in y_continent_raw
                    ])
                else:
                    y_continent = np.full(len(y_continent_raw), -1)
            else:
                y_continent = np.full(len(y_continent_raw), -1)
            
            # Caseina
            if self.use_caseina:
                valid_caseina = [c for c in y_caseina_raw if c is not None]
                if valid_caseina:
                    self.label_encoder_caseina.fit(valid_caseina)
                    self.class_names_caseina = list(self.label_encoder_caseina.classes_)
                    self.num_classes_caseina = len(self.class_names_caseina)
                    
                    y_caseina = np.array([
                        self.label_encoder_caseina.transform([c])[0] if c is not None else -1
                        for c in y_caseina_raw
                    ])
                else:
                    y_caseina = np.full(len(y_caseina_raw), -1)
            else:
                y_caseina = np.full(len(y_caseina_raw), -1)
            
            # Attitudine (LATTE/FIBRA/CARNE/ALTRO)
            if self.use_attitudine:
                self.label_encoder_attitudine.fit(y_attitudine_raw)
                self.class_names_attitudine = list(self.label_encoder_attitudine.classes_)
                self.num_classes_attitudine = len(self.class_names_attitudine)
                y_attitudine = self.label_encoder_attitudine.transform(y_attitudine_raw)
            else:
                y_attitudine = np.full(len(y_attitudine_raw), -1)
            
            print(f"\n🏷️  Classi per task:")
            print(f"   Breed: {self.num_classes_breed}")
            print(f"   Continent: {self.num_classes_continent} ({self.class_names_continent})")
            print(f"   Caseina: {self.num_classes_caseina} ({self.class_names_caseina})")
            print(f"   Attitudine: {self.num_classes_attitudine} ({self.class_names_attitudine})")
            
            # Store processed data
            self.X_processed = X.astype(np.float32)
            self.y_breed_processed = y_breed
            self.y_continent_processed = y_continent
            self.y_caseina_processed = y_caseina
            self.y_attitudine_processed = y_attitudine
            self.num_snps = X.shape[1]
            
            print(f"📍 SNP finali: {self.num_snps}")

            # --- TRAIN/VAL/TEST SPLIT ---
            print(f"\n📂 Creazione Split Dataset...")
            
            try:
                test_val_size = self.val_split + self.test_split
                indices = np.arange(len(X))
                
                train_idx, temp_idx = train_test_split(
                    indices, test_size=test_val_size, stratify=y_breed, random_state=42
                )
                
                relative_test_size = self.test_split / test_val_size
                val_idx, test_idx = train_test_split(
                    temp_idx, test_size=relative_test_size, 
                    stratify=y_breed[temp_idx], random_state=42
                )
                
                print(f"   Train: {len(train_idx)} campioni")
                print(f"   Val:   {len(val_idx)} campioni")
                print(f"   Test:  {len(test_idx)} campioni")
                
            except ValueError as e:
                raise ValueError(
                    f"Errore nello split stratificato: {e}\n"
                    f"SOLUZIONE: Aumenta 'min_samples_per_class' (attuale: {self.min_samples})"
                )

            # Creazione TensorDataset con tutte le label
            self.train_dataset = TensorDataset(
                torch.FloatTensor(X[train_idx]), 
                torch.LongTensor(y_breed[train_idx]),
                torch.LongTensor(y_continent[train_idx]),
                torch.LongTensor(y_caseina[train_idx]),
                torch.LongTensor(y_attitudine[train_idx])
            )
            self.val_dataset = TensorDataset(
                torch.FloatTensor(X[val_idx]), 
                torch.LongTensor(y_breed[val_idx]),
                torch.LongTensor(y_continent[val_idx]),
                torch.LongTensor(y_caseina[val_idx]),
                torch.LongTensor(y_attitudine[val_idx])
            )
            self.test_dataset = TensorDataset(
                torch.FloatTensor(X[test_idx]), 
                torch.LongTensor(y_breed[test_idx]),
                torch.LongTensor(y_continent[test_idx]),
                torch.LongTensor(y_caseina[test_idx]),
                torch.LongTensor(y_attitudine[test_idx])
            )
            
            print(f"✅ Setup completato!\n")

        except Exception as e:
            raise RuntimeError(f"Errore nel setup DataModule: {e}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )
