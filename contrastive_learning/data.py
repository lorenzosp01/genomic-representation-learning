"""PLINK data module for the contrastive model (Thor & Nettelblad 2025).

Extracted verbatim from the contrastive notebooks. QC, MAF filter, LD pruning,
downsampling and the 70/15/15 split are preserved exactly as executed.
"""

import os

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from bed_reader import open_bed


class GenomicDataModule(pl.LightningDataModule):
    """DataModule PLINK con QC, MAF filter, LD pruning, downsampling."""

    def __init__(self, bed_path, batch_size=None,
                 maf_thresh=0.01, geno_thresh=0.1, min_samples_per_class=30,
                 ld_pruning=True, ld_threshold=0.2, ld_window=50,
                 val_split=0.15, test_split=0.15, coarse_mapping=None,
                 downsample_per_class=None):
        super().__init__()
        # batch_size=None → full-batch (paper faithful)
        self.bed_path    = bed_path
        self.batch_size  = batch_size
        self.maf_thresh  = maf_thresh
        self.geno_thresh = geno_thresh
        self.min_samples = min_samples_per_class
        self.ld_pruning  = ld_pruning
        self.ld_threshold= ld_threshold
        self.ld_window   = ld_window
        self.val_split   = val_split
        self.test_split  = test_split
        self.coarse_mapping       = coarse_mapping
        self.downsample_per_class = downsample_per_class
        self.imputer = SimpleImputer(strategy='mean')
        self.le      = LabelEncoder()

    def setup(self, stage=None):
        if hasattr(self, 'train_dataset'):
            return

        print(f"\n{'='*60}")
        print(f"CARICAMENTO: {self.bed_path}")
        print(f"{'='*60}")

        if not os.path.exists(self.bed_path):
            raise FileNotFoundError(f"File non trovato: {self.bed_path}")

        G     = open_bed(self.bed_path, count_A1=True)
        y_raw = np.loadtxt(self.bed_path.replace('.bed', '.fam'), dtype=str, usecols=0)
        y_breeds = y_raw.copy()
        X = G.read()
        if X.shape[1] == len(y_raw):
            print("⚠️  Trasposizione SNP×Sample → Sample×SNP")
            X = X.T

        if self.coarse_mapping:
            b2c = {b: c for c, bs in self.coarse_mapping.items() for b in bs}
            mask = np.array([l in b2c for l in y_raw])
            y_raw = np.array([b2c.get(l, '') for l in y_raw])
            X, y_raw, y_breeds = X[mask], y_raw[mask], y_breeds[mask]

        print(f"📊 Shape iniziale: {X.shape}")

        unique, counts = np.unique(y_raw, return_counts=True)
        valid = unique[counts >= self.min_samples]
        mask  = np.isin(y_raw, valid)
        print(f"🔍 Razze ≥{self.min_samples} campioni: {len(valid)}/{len(unique)}")
        X, y_raw, y_breeds = X[mask], y_raw[mask], y_breeds[mask]

        if self.downsample_per_class is not None:
            target = int(self.downsample_per_class)
            rng = np.random.default_rng(42)
            idx = np.concatenate([
                rng.choice(np.where(y_raw == c)[0], size=min(target, (y_raw==c).sum()), replace=False)
                for c in np.unique(y_raw)
            ])
            rng.shuffle(idx)
            X, y_raw, y_breeds = X[idx], y_raw[idx], y_breeds[idx]
            print(f"⚖️  Downsampling → {target}/razza, totale {len(y_raw)}")

        print("\n🧬 QC PIPELINE")
        missing = np.isnan(X).mean(axis=0)
        X = X[:, missing < self.geno_thresh]
        print(f"   1. Missingness: {X.shape[1]} SNP rimasti")

        X = self.imputer.fit_transform(X)
        print(f"   2. Imputazione: OK")

        af  = np.mean(X, axis=0) / 2.0
        maf = np.minimum(af, 1 - af)
        X   = X[:, maf > self.maf_thresh]
        print(f"   3. MAF: {X.shape[1]} SNP rimasti")

        if self.ld_pruning:
            X = self._prune_ld(X)

        print(f"\n📈 FINALI: {X.shape}")

        y = self.le.fit_transform(y_raw)
        self.num_classes  = len(self.le.classes_)
        self.num_snps     = X.shape[1]
        self.class_names  = self.le.classes_
        self.X_processed  = X.astype(np.float32)
        self.y_processed  = y
        self.y_breeds     = y_breeds

        print(f"🏷️  Razze: {self.num_classes} | SNP: {self.num_snps} | Campioni: {len(y)}")

        tv = self.val_split + self.test_split
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=tv, stratify=y, random_state=42)
        X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp,
                                                      test_size=self.test_split/tv,
                                                      stratify=y_tmp, random_state=42)
        print(f"   Train {X_tr.shape[0]} | Val {X_val.shape[0]} | Test {X_te.shape[0]}")

        mk = lambda Xa, ya: TensorDataset(torch.FloatTensor(Xa), torch.LongTensor(ya))
        self.train_dataset = mk(X_tr,  y_tr)
        self.val_dataset   = mk(X_val, y_val)
        self.test_dataset  = mk(X_te,  y_te)
        print("✅ Setup completato!")

    def _prune_ld(self, X):
        print(f"   4. LD Pruning (window={self.ld_window}, r²>{self.ld_threshold})...")
        n, p = X.shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
        Xt = torch.tensor(Xs, device=device, dtype=torch.float32)
        remove = np.zeros(p, dtype=bool)
        for i in range(p):
            if remove[i]: continue
            end = min(i + self.ld_window, p)
            r2  = (Xt[:, i] @ Xt[:, i+1:end] / n) ** 2
            high = torch.where(r2 > self.ld_threshold)[0].cpu().numpy() + i + 1
            remove[high] = True
        keep = ~remove
        print(f"      Rimossi {remove.sum()} SNP → {keep.sum()} rimasti")
        return X[:, keep]

    def train_dataloader(self):
        bs = self.batch_size or len(self.train_dataset)
        return DataLoader(self.train_dataset, batch_size=bs, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=False)

    def val_dataloader(self):
        bs = self.batch_size or len(self.val_dataset)
        return DataLoader(self.val_dataset, batch_size=bs,
                          num_workers=4, pin_memory=True, drop_last=False)

    def test_dataloader(self):
        bs = self.batch_size or len(self.test_dataset)
        return DataLoader(self.test_dataset, batch_size=bs,
                          num_workers=4, pin_memory=True, drop_last=False)
