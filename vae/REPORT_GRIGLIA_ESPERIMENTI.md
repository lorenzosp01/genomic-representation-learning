# VMGP Grid Search - Report Esperimenti
**Data:** 23 Febbraio 2026

## 1. Obiettivo
Esplorazione sistematica dello spazio degli iperparametri del modello VMGP (Variational Auto-Encoder Multi-task Genomic Prediction) tramite una griglia completa di **20 combinazioni**:
- **Min Samples per Razza:** 30, 40, 50, 60, 70 (solo campioni reali, NO SMOTE)
- **Dimensione Latent Space:** 32, 64, 96, 128
- **Task:** Solo classificazione breed
- **Validazione:** 3-Fold Cross Validation per ogni esperimento

### Glossario metriche
| Metrica | Descrizione | Obiettivo |
|---|---|---|
| **Breed Accuracy** | Accuratezza raw del classificatore KNN sui dati compressi. **Non direttamente comparabile** tra soglie diverse (classificare 10 razze è più facile che classificarne 34). | 📈 Alto |
| **Cohen's Kappa** | Accuratezza **chance-adjusted**: tiene conto del numero di classi e del bilanciamento. **Metrica preferita** per confronti equi tra soglie diverse. | 📈 Alto |
| **Generalization (GE)** | Accuratezza su animali mai visti (Validation Set). Misura la capacità di generalizzazione. | 📈 Alto |
| **Silhouette Score** | Quanto sono ben separati i cluster (razze) nello spazio latente. >0.7: Ottimo. <0.5: Scarso. | 📈 Alto |
| **Davies-Bouldin** | Rapporto tra dispersione intra-cluster e distanza inter-cluster. Più basso = cluster più compatti e separati. | 📉 Basso |
| **Reconstruction MSE** | Errore medio di ricostruzione del genotipo dopo compressione. | 📉 Basso |

---

## 2. Miglior Modello

### **`Samples_30_LatDim_96`** (per Cohen's Kappa — confronto equo)

**Performance (media ± std su 3 fold):**
- **Cohen's Kappa:** `0.9714` — Il più alto di tutta la griglia
- **Breed Accuracy:** `0.9731` (97.31%)
- **Generalization (GE):** `0.9724` (97.24%)
- **Silhouette Score:** `0.7210`
- **Davies-Bouldin:** `0.5737`
- **Reconstruction MSE:** `0.4018`

**Configurazione:**
- **Latent Dimension:** 96
- **Min Samples per Class:** 30
- **Numero Razze:** 34 (il massimo testato)
- **Numero SNPs:** 49,586
- **Balancing:** Solo downsampling (cap a 30), NO SMOTE
- **Task:** Solo classificazione breed
- **K-Fold CV:** 3 fold

> **Perché questo modello?** Pur non avendo l'Accuracy raw più alta (che spetta a soglie con meno razze, quindi task più facili), ha il **Kappa più alto**, dimostrando che riesce a distinguere **34 razze** con precisione superiore rispetto a modelli con meno classi — una volta corretta per il caso.

---

## 3. Classifica Top 5 Modelli (per Cohen's Kappa)

| Rank | Experiment | Razze | Kappa | Accuracy | GE | Silhouette |
|---|---|---|---|---|---|---|
| 🥇 | **Samples_30_LatDim_96** | **34** | **0.9714** | 0.9731 | 0.9724 | 0.7210 |
| 🥈 | Samples_40_LatDim_128 | 26 | 0.9710 | 0.9731 | 0.9742 | 0.7053 |
| 🥉 | Samples_40_LatDim_96 | 26 | 0.9710 | 0.9731 | 0.9742 | 0.7083 |
| 4 | Samples_40_LatDim_32 | 26 | 0.9710 | 0.9731 | 0.9742 | 0.7388 |
| 5 | Samples_50_LatDim_128 | 16 | 0.9708 | 0.9737 | 0.9741 | 0.6990 |

> **Nota:** La soglia **min_samples=40** domina le posizioni 2-4 con tre diverse dimensioni latenti che raggiungono tutte Kappa=0.9710. Questo suggerisce che a 40 campioni il modello è **robusto rispetto alla dimensione latente**.

---

## 4. Tabella Completa: Tutti i 20 Esperimenti

### Min Samples = 30 (34 razze, 49,586 SNPs)

| Latent Dim | Accuracy | Kappa | GE | Silhouette | Davies-Bouldin | MSE |
|---|---|---|---|---|---|---|
| **96** | **0.9731** | **0.9714** | **0.9724** | 0.7210 | 0.5737 | 0.4018 |
| 32 | 0.9711 | 0.9692 | 0.9711 | **0.7357** | **0.5625** | 0.4029 |
| 64 | 0.9708 | 0.9689 | 0.9721 | 0.7225 | 0.5853 | 0.4019 |
| 128 | 0.9688 | 0.9667 | 0.9681 | 0.7157 | 0.6099 | 0.4025 |

### Min Samples = 40 (26 razze, 49,640 SNPs)

| Latent Dim | Accuracy | Kappa | GE | Silhouette | Davies-Bouldin | MSE |
|---|---|---|---|---|---|---|
| **32** | 0.9731 | **0.9710** | **0.9742** | **0.7388** | **0.5004** | 0.4012 |
| **96** | 0.9731 | **0.9710** | **0.9742** | 0.7083 | 0.5397 | 0.3992 |
| **128** | 0.9731 | **0.9710** | **0.9742** | 0.7053 | 0.5380 | **0.3991** |
| 64 | 0.9716 | 0.9694 | 0.9734 | 0.7199 | 0.5345 | 0.3994 |

### Min Samples = 50 (16 razze, 49,359 SNPs)

| Latent Dim | Accuracy | Kappa | GE | Silhouette | Davies-Bouldin | MSE |
|---|---|---|---|---|---|---|
| **128** | **0.9737** | **0.9708** | **0.9741** | 0.6990 | 0.5192 | 0.4013 |
| 32 | 0.9732 | 0.9703 | 0.9728 | **0.7405** | **0.4569** | 0.4019 |
| 96 | 0.9715 | 0.9684 | 0.9723 | 0.7055 | 0.5087 | 0.4009 |
| 64 | 0.9706 | 0.9674 | 0.9706 | 0.7131 | 0.5053 | 0.4010 |

### Min Samples = 60 (12 razze, 49,250 SNPs)

| Latent Dim | Accuracy | Kappa | GE | Silhouette | Davies-Bouldin | MSE |
|---|---|---|---|---|---|---|
| **64** | **0.9738** | **0.9704** | **0.9743** | 0.7062 | 0.5221 | 0.4006 |
| 96 | 0.9734 | 0.9698 | 0.9738 | 0.6939 | 0.5436 | 0.3996 |
| 128 | 0.9729 | 0.9693 | 0.9738 | 0.6843 | 0.5433 | 0.4003 |
| 32 | 0.9714 | 0.9676 | 0.9743 | **0.7309** | **0.4857** | **0.3999** |

### Min Samples = 70 (10 razze, 49,023 SNPs)

| Latent Dim | Accuracy | Kappa | GE | Silhouette | Davies-Bouldin | MSE |
|---|---|---|---|---|---|---|
| **32** | **0.9742** | **0.9703** | **0.9737** | **0.7398** | **0.4807** | 0.3939 |
| **128** | **0.9742** | **0.9703** | **0.9737** | 0.6846 | 0.5691 | 0.3942 |
| 64 | 0.9721 | 0.9679 | 0.9721 | 0.7055 | 0.5316 | **0.3936** |
| 96 | 0.9711 | 0.9668 | 0.9711 | 0.6974 | 0.5574 | 0.3945 |

---

## 5. Analisi: Effetto del Numero Minimo di Sample

Statistiche medie su tutte e 4 le dimensioni latenti per ogni soglia:

| Min Samples | Razze | SNPs | Kappa (media) | Accuracy (media) | GE (media) | Silhouette (media) |
|---|---|---|---|---|---|---|
| **30** | **34** | 49,586 | **0.9691** | 0.9709 | 0.9709 | **0.7237** |
| **40** | 26 | 49,640 | **0.9706** | 0.9727 | **0.9740** | 0.7181 |
| 50 | 16 | 49,359 | 0.9692 | 0.9722 | 0.9725 | 0.7145 |
| 60 | 12 | 49,250 | 0.9693 | 0.9729 | **0.9741** | 0.7038 |
| 70 | 10 | 49,023 | 0.9688 | **0.9729** | 0.9727 | 0.7068 |

**Osservazioni:**
1. **Il Kappa è straordinariamente stabile** lungo tutte le soglie (range: 0.9688–0.9706). Questo significa che il modello VMGP mantiene una performance *chance-adjusted* quasi identica indipendentemente dal numero di razze incluse.
2. **L'Accuracy raw cresce con meno razze** (0.9709 → 0.9729 da 30 a 70 campioni), ma questo è un artefatto: classificare 10 classi è intrinsecamente più facile che classificarne 34. Il Kappa corregge per questo.
3. **La soglia 40 ha il miglior Kappa medio** (0.9706), ma la differenza con 30 (0.9691) è minima.
4. **Il Silhouette decresce** al crescere dei campioni: con 30 campioni le "isole" sono più nette (0.7237), con 70 si sovrappongono leggermente di più (0.7068). Probabile effetto del ridotto numero di razze su come i cluster occupano lo spazio.

> **Conclusione:** La soglia **min_samples=30** offre il miglior trade-off: include il **massimo numero di razze (34)** con Kappa competitivo e il miglior clustering (Silhouette). Se si preferisce massimizzare GE e stabilità, **min_samples=40** è l'alternativa.

---

## 6. Analisi: Effetto della Dimensione Latente

Statistiche medie su tutte e 5 le soglie per ogni dimensione latente:

| Latent Dim | Kappa (media) | Accuracy (media) | GE (media) | Silhouette (media) |
|---|---|---|---|---|
| **32** | **0.9697** | **0.9726** | **0.9732** | **0.7371** |
| 64 | 0.9688 | 0.9718 | 0.9725 | 0.7134 |
| 96 | 0.9695 | 0.9724 | 0.9728 | 0.7052 |
| **128** | **0.9696** | **0.9725** | **0.9728** | 0.6978 |

**Osservazioni:**
1. **Le dimensioni latenti hanno un effetto trascurabile** sul Kappa (range: 0.9688–0.9697). Il modello performa quasi identicamente con 32 o 128 dimensioni.
2. **Silhouette favorisce dimensioni basse** (32: 0.7371 vs 128: 0.6978). Con meno dimensioni, i cluster sono geometricamente più compatti.
3. **La dimensione 32 è leggermente migliore su tutte le metriche** in media, suggerendo che il segnale genomico utile per la classificazione breed è comprimibile in uno spazio molto ridotto.
4. **La dimensione 64 è la peggiore** in media per Kappa (0.9688), forse in una zona intermedia non ottimale.

> **Conclusione:** **Latent Dim = 32** è sufficiente e persino preferibile. Dimensioni maggiori non aggiungono informazione utile e peggiorano leggermente il clustering. Questo è un risultato importante per la riduzione dimensionale: il genoma discriminante per le razze caprine si comprime efficacemente in **32 dimensioni** (da ~49,500 SNPs — compressione ~1,550×).

---

## 7. Miglior Configurazione per Soglia (Best Latent Dim per ogni Min Samples)

| Min Samples | Razze | Best Latent Dim | Kappa | Accuracy | GE | Silhouette | MSE |
|---|---|---|---|---|---|---|---|
| **30** | 34 | **96** | **0.9714** | 0.9731 | 0.9724 | 0.7210 | 0.4018 |
| **40** | 26 | 128* | 0.9710 | 0.9731 | 0.9742 | 0.7053 | 0.3991 |
| 50 | 16 | 128 | 0.9708 | 0.9737 | 0.9741 | 0.6990 | 0.4013 |
| 60 | 12 | 64 | 0.9704 | 0.9738 | 0.9743 | 0.7062 | 0.4006 |
| 70 | 10 | 32 | 0.9703 | 0.9742 | 0.9737 | 0.7398 | 0.3939 |

*\* Per min_samples=40, le dimensioni 32, 96 e 128 hanno lo stesso Kappa (0.9710). La 32 ha il miglior Silhouette (0.7388).*

> **Pattern emergente:** Con più razze (30-50), dimensioni latenti più alte (96-128) sono leggermente favorite. Con poche razze (60-70), dimensioni basse (32-64) bastano. Questo è coerente: più classi richiedono più "spazio" per essere rappresentate.

---

## 9. Grafici

### A. Heatmap Griglia Esperimenti
- **Righe:** Min Samples per Breed (30→70)
- **Colonne:** Latent Dimension (32→128)
- **Colore:** Verde = buono, Rosso = scarso
- **Interpretazione:** 4 heatmap separate per Accuracy, Kappa, Silhouette e GE. Cercare le celle più verdi per la configurazione ottimale.

![Heatmap Griglia Esperimenti](grid_experiment_heatmaps.png)

### B. Andamento Metriche in 3D
- **Asse X:** Dimensione Latente (32, 64, 96, 128)
- **Asse Y:** Min Samples per Breed (30, 40, 50, 60, 70)
- **Asse Z (altezza + colore):** Valore della metrica
- **Superficie:** Interpolazione tra i 20 punti della griglia (punti rossi)
- **Interpretazione:** Superfici piatte indicano robustezza agli iperparametri; pendenze indicano sensibilità.

![Andamento Metriche per Dimensione Latente](metrics_by_latent_dim.png)

### C. Confusion Matrix
- **1 figura per ogni soglia** min_samples, con 4 subplot (1 per latent_dim)
- **Normalizzate per riga:** ogni cella mostra la recall per classe
- **Diagonale scura:** alta recall (classificazione corretta)

#### Min Samples = 30 (34 razze)
![Confusion Matrix — Min Samples 30](results/confusion_matrices/cm_grid_samples_30.png)

#### Min Samples = 40 (26 razze)
![Confusion Matrix — Min Samples 40](results/confusion_matrices/cm_grid_samples_40.png)

#### Min Samples = 50 (16 razze)
![Confusion Matrix — Min Samples 50](results/confusion_matrices/cm_grid_samples_50.png)

#### Min Samples = 60 (12 razze)
![Confusion Matrix — Min Samples 60](results/confusion_matrices/cm_grid_samples_60.png)

#### Min Samples = 70 (10 razze)
![Confusion Matrix — Min Samples 70](results/confusion_matrices/cm_grid_samples_70.png)

---

## 10. Analisi della Confusione tra Razze

Aggregando le 20 matrici di confusione (5 soglie min_samples × 4 dimensioni latenti), è possibile identificare le razze più problematiche e le coppie che il modello confonde sistematicamente.

### 10.1 Razze più problematiche

| Razza | #Esperimenti con errori | Recall medio | Confusa più spesso con |
|-------|:-----------------------:|:------------:|------------------------|
| **SEA** | 12/12 | 75.2% | BUR (42), WYG (13) |
| **NBN** | 20/20 | 85.3% | CRE (59), SAA (38), ALP (4) |
| **SAA** | 20/20 | 88.9% | CRE (60), SEA (20), ALP (16) |
| **CRE** | 20/20 | 94.1% | SAA (39), NBN (23), ANG (6) |
| **BUR** | 20/20 | 95.8% | BOE (20) |
| **OSS** | 12/20 | 95.1% | SID (14) |
| **ABR** | 8/12 | 94.1% | GUM (8) |
| **BRK** | 18/20 | 96.1% | NBN (21), SID (12), CRE (2) |
| **ANG** | 17/20 | 98.7% | CRE (10), NBN (7), LND (4) |
| **BOE** | 8/20 | 99.1% | BUR (8) |

**Osservazioni:** Le razze SEA (South East Anatolian), NBN (Nubian) e SAA (Saanen) sono le più critiche, con recall medio rispettivamente del 75.2%, 85.3% e 88.9%. In particolare, il cluster CRE-SAA-NBN mostra forte confusione reciproca: queste tre razze concentrano **200 errori** complessivi tra loro (CRE↔SAA: 99, CRE↔NBN: 82, NBN↔SAA: 50 errori su 20 esperimenti).

### 10.2 Coppie di razze più confuse

| Razza A | Razza B | Errori totali |
|---------|---------|:-------------:|
| CRE | SAA | 99 |
| CRE | NBN | 82 |
| NBN | SAA | 50 |
| BUR | SEA | 42 |
| BOE | BUR | 28 |
| BRK | NBN | 21 |
| SAA | SEA | 20 |
| OSS | SID | 18 |
| ANG | CRE | 16 |
| ALP | SAA | 16 |
| BUR | SAA | 16 |
| SEA | WYG | 13 |
| BRK | SID | 12 |
| ABR | GUM | 8 |
| ANG | NBN | 7 |

### 10.3 Modello migliore: Samples_30_LatDim_96

Nel modello migliore per Kappa (0.9714, 34 razze), 25 razze su 34 raggiungono recall del 100%. Le 9 razze con errori:

| Razza | Recall | Campioni | Confusa con |
|-------|:------:|:--------:|-------------|
| SEA | 61.1% | 18 | BUR (4), WYG (3) |
| SAA | 86.0% | 57 | CRE (6), SEA (1), ALP (1) |
| NBN | 88.2% | 34 | CRE (3), ALP (1) |
| CRE | 91.9% | 62 | SAA (4), NBN (1) |
| BRK | 92.2% | 51 | SID (3), CRE (1) |
| ABR | 94.1% | 17 | GUM (1) |
| TED | 94.1% | 17 | CRE (1) |
| BUR | 95.8% | 24 | BOE (1) |
| OSS | 95.8% | 24 | SID (1) |

### 10.4 Visualizzazioni

![Analisi errori per razza e matrice confusione aggregata](breed_confusion_analysis.png)

A sinistra, il grafico a barre mostra gli errori totali per razza (colore = recall medio: verde = alto, rosso = basso). A destra, la heatmap mostra la matrice di confusione bidirezionale aggregata su tutti i 20 esperimenti.

![Errori per coppia al variare di min_samples](breed_confusion_by_samples.png)

L'andamento delle principali coppie di confusione al variare della soglia min_samples. Le coppie CRE↔SAA e CRE↔NBN dominano costantemente, ma non si riducono con l'aumento della soglia (poiché queste razze restano sempre nel dataset).

### 10.5 Interpretazione

1. **Cluster CRE-SAA-NBN:** Creole, Saanen e Nubian formano un triangolo di confusione reciproca con 231 errori combinati. Questo suggerisce somiglianza genetica o sovrapposizione fenotipica tra queste razze.
2. **BUR↔SEA:** Burma e South East Anatolian (42 errori) sono il secondo fatto critico; SEA ha il recall più basso in assoluto (75.2%).
3. **Confusione non simmetrica:** CRE→SAA (39 errori) è più frequente di SAA→CRE (60 errori), indicando che SAA viene confusa con CRE più spesso del viceversa.
4. **Razze con pochi campioni:** SEA (18 campioni), ABR (17), TED (17) sono le razze con meno campioni e mostrano errori — ma NBN (34 campioni) e SAA (57 campioni) hanno errori significativi anche con campioni sufficienti, indicando vera similarità genomica.
