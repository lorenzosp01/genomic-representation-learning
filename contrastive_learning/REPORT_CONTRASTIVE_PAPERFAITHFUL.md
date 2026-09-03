# Contrastive Learning — Report Esperimento Paper-Faithful
**Data:** 24 Febbraio 2026

## 1. Obiettivo

Implementazione **completamente fedele** al paper:

> Thor & Nettelblad (2025) — *"Dimensionality Reduction of Genetic Data using Contrastive Learning"*

L'obiettivo è apprendere una rappresentazione compressa del genoma caprina su una **sfera unitaria 3D** tramite contrastive learning, mantenendo alta la discriminabilità tra razze. A differenza di un VAE, questo approccio non ricostruisce il genotipo ma impara direttamente uno spazio metrico dove campioni della stessa razza sono vicini e campioni di razze diverse sono lontani.

---

## 2. Architettura del Modello

### Encoder (GeneticEncoder)
L'encoder mappa il genotipo SNP su una sfera unitaria tridimensionale:

```
Input (B, M) → One-Hot (B, M, 4) → Conv1D(4→5, k=3) → SiLU
  → Conv1D(5→5, k=3) → SiLU → Flatten → Dense(256) → BN → SiLU
  → Dense(256) → BN → SiLU → Dense(256) → BN → SiLU
  → Dense(3) → L2-normalize → Sfera 3D
```

- **Parametri totali:** 59.7M (trainable)
- **Output:** Vettore 3D normalizzato L2 (sulla sfera unitaria)
- **Attivazione:** SiLU/Swish (x · σ(x))

### Augmentation (GeneticAugmentation)
Data augmentation specifica per dati genetici, applicata ad ogni epoch:

1. **Flip allelico:** Con probabilità p_flip ~ U(0.01, 0.99) per campione, gli alleli vengono "flippati" (0↔1↔2)
2. **Masking:** Con probabilità p_mask ~ U(0.01, 0.99), gli SNP vengono mascherati (impostati a -1 = "sconosciuto")
3. **One-Hot Encoding:** 0→[1,0,0,0] | 1→[0,1,0,0] | 2→[0,0,1,0] | -1→[0,0,0,1]

### Loss Function — Centroid N-Pair Loss (Eq. 6 del paper)

![Centroid N-Pair Loss](results/centroid_loss_eq.png)

La somma sui negativi è **dentro il logaritmo** — differenza cruciale rispetto all'implementazione naïve.

Passi del calcolo:
1. **Centroide:** C_i = (z + 2z⁻ + z⁺) / 4
2. **Centratura:** z_c = z − C, z⁺_c = z⁺ − C, z⁻_c = z⁻ − C
3. **Scaling:** μ_i = max(‖z_c‖², ‖z⁺_c‖², ‖z⁻_c‖²), poi z̃ = z_c / √μ
4. **Loss:** Somma delle differenze di similarità (positiva vs negativa), dentro un singolo logaritmo

---

## 3. Configurazione Sperimentale

### Dataset
| Parametro | Valore |
|---|---|
| **Dataset** | ADAPTmap (capre) |
| **Campioni totale (raw)** | 4,653 |
| **Razze totali (raw)** | 144 |
| **Soglia minima per razza** | 30 campioni |
| **Razze incluse** | 34 |
| **Downsampling** | 30 campioni/razza |
| **Campioni usati** | 1,020 |
| **SNP dopo QC + LD Pruning** | 46,565 |

### QC Pipeline
1. **Missingness:** SNP con >10% dati mancanti rimossi → 51,255 SNP
2. **Imputazione:** Media per SNP mancanti
3. **MAF filter:** SNP con MAF < 0.01 rimossi → 51,244 SNP
4. **LD Pruning:** Window=50, r² > 0.2 → rimossi 4,679 SNP → **46,565 SNP finali**

### Iperparametri (Paper-Faithful)
| Parametro | Valore |
|---|---|
| **Embedding dim** | 3 (sfera 3D) |
| **Batch size** | Full-batch (680 campioni = N_train) |
| **Negativi per anchor** | 679 (N_train − 1) |
| **Optimizer** | Adam (β₁=0.9, β₂=0.999) |
| **Learning rate** | 0.001 |
| **LR decay** | ×0.99 ogni 10 epoche (StepLR) |
| **Max epochs** | 5,000 |
| **Early stopping patience** | 200 epoche |
| **Augmentation flip/mask** | p ~ U(0.01, 0.99) |
| **K-Fold CV** | 3 fold |
| **GPU** | NVIDIA GeForce RTX 5070 Ti |

### Differenze chiave rispetto all'implementazione naïve

| Componente | Versione naïve | Versione paper-faithful |
|---|---|---|
| **Loss** | `log(1+clamp(term))` per-pair | `log(1 + Σ term)` — somma **dentro** il log |
| **Batch size** | 128 (mini-batch) | **Full-batch** (N_train) |
| **Negativi** | 127 per anchor | **N_train − 1** per anchor |
| **Optimizer** | AdamW | **Adam** (nessun weight decay) |
| **LR decay** | ExponentialLR ogni epoch | **StepLR** γ=0.99 ogni **10 epoche** |
| **Augmentation** | flip=mask=0.90 | flip=mask=0.99, **U(0.01, 0.99)** |
| **Epochs** | 100 | **5,000** |

---

## 4. Risultati

### Metriche

| Metrica | Descrizione | Obiettivo |
|---|---|---|
| **KNN Accuracy @3** | Accuratezza di un classificatore KNN (k=3) sugli embedding 3D. Misura quanto bene lo spazio appreso separa le razze. | ↑ Alto |
| **KNN F1-Score @3** | F1-score macro del classificatore KNN. Tiene conto del bilanciamento tra precision e recall per tutte le classi. | ↑ Alto |
| **KNN Recall @3** | Recall macro del classificatore KNN. Misura la capacità del modello di identificare correttamente tutti i campioni di ogni razza (sensibilità media tra le classi). | ↑ Alto |
| **Silhouette Score** | Coesione intra-cluster e separazione inter-cluster nello spazio 3D. Range: [-1, 1]. | ↑ Alto |
| **Davies-Bouldin Index** | Rapporto tra dispersione intra-cluster e distanza inter-cluster. | ↓ Basso |

### Risultati per Fold

| Fold | KNN Accuracy @3 | Silhouette | Epoche (best ckpt) | Val Loss |
|---|---|---|---|---|
| 1 | 0.9382 | 0.5249 | 1,078 | 2.1516 |
| 2 | **0.9706** | 0.5261 | 1,734 | 2.0142 |
| 3 | 0.9529 | **0.5768** | 1,027 | 2.1644 |

### Risultati Medi (3-Fold CV)

| Metrica | Valore |
|---|---|
| **KNN Accuracy @3** | **0.9539** (95.39%) |
| **KNN F1-Score @3** | **0.9519** (95.19%) |
| **Silhouette Score** | **0.5426** |
| **Davies-Bouldin Index** | **0.9442** |

### Analisi per Razza (dal best fold — Fold 2)

Dalla confusion matrix del best fold (KNN @3):

**Razze con 100% di recall (classificazione perfetta):**
ABR, ALP, ANG, BOE, BRI, BUR, BUT, CAM, CAN, CAS, CRE, CRS, GGT, GUM, KEF, KLS, LND, LNR, MLG, MOX, NBN, RME, SAR, SID, WAD, WYG (26/34 razze)

**Razze con errori di classificazione:**

| Razza | Recall | Confusa con |
|---|---|---|
| BRK | 90% | NBN (10%) |
| KAM | 90% | SAA (10%) |
| OSS | 90% | SID (10%) |
| PAT | 90% | OSS (10%) — confonde anche con CAS |
| RAN | 90% | ALP (10%) |
| SAA | 90% | CRS (10%) |
| SEA | 70% | WYG (30%) |
| TED | 90% | CRS (10%) |

> **SEA (Shell East African)** è la razza più problematica: il 30% viene confuso con WYG (West African Goat), probabilmente per vicinanza genetica geografica (Africa).

---

## 5. Visualizzazioni

### A. Distribuzione Dataset
Dataset perfettamente bilanciato: 30 campioni per ognuna delle 34 razze (1,020 totale).

![Distribuzione Dataset](results/embedding_paperfaithful.png)

### B. Embedding — Proiezione Equal Earth e Sfera 3D
Gli embedding 3D vengono proiettati sia su piano 2D tramite Equal Earth projection (Šavrič et al. 2019) sia visualizzati sulla sfera unitaria.

- **Equal Earth (sinistra):** I cluster sono ben separati, con poche sovrapposizioni
- **Sfera 3D (destra):** Le razze occupano regioni distinte sulla sfera

![Embedding Paper-Faithful](results/embedding_paperfaithful.png)

### C. Confusion Matrix KNN @3
Matrice normalizzata per riga (recall per classe). La diagonale scura indica alta precisione.

![Confusion Matrix](results/cm_paperfaithful.png)

## 7. Conclusioni

### Punti di forza:
1. **Compressione estrema:** 46,565 SNP → 3 dimensioni con 95.4% di accuracy — ogni razza occupa una regione distinta sulla sfera 3D
2. **26 razze su 34 classificate perfettamente** (100% recall)
3. **Interpretabilità:** La sfera 3D è direttamente visualizzabile e interpretabile (proiezione Equal Earth)
4. **Fedeltà al paper:** L'implementazione riproduce fedelmente il metodo originale (fix critico della loss function)

### Limitazioni:
1. **Silhouette moderato (0.54):** I cluster sono meno compatti rispetto al VAE (0.72) — probabile limite della compressione a sole 3 dimensioni
2. **SEA confusa con WYG (30%):** Alcune razze dell'Africa hanno embedding sovrapposti — possibile vicinanza genetica reale
3. **Training costoso:** 5,000 epoche full-batch con 59.7M parametri
4. **Nessuna capacità generativa:** A differenza del VAE, non può ricostruire il genotipo né generare nuovi campioni
