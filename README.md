# Representation Learning and Attribution-Based Feature Selection in Livestock Genomics

Code accompanying the MSc thesis *Representation Learning and Attribution-Based Feature Selection in Livestock Genomics: A Leakage-Controlled Benchmark*.

The repository implements a leakage-controlled experimental pipeline for:

- breed classification from high-density SNP genotype profiles;
- sample-efficiency and marker-efficiency analyses;
- comparison of SNP-ranking strategies ($F_{ST}$, Random Forest importance, SHAP, Integrated Gradients) through the downstream utility of the reduced panels they produce;
- cross-species validation on an independent ovine cohort.

It contains a VMGP-inspired variational model and a paper-faithful centroid N-pair contrastive model, together with the shared genomic utilities and the experiment scripts that produced the thesis results.

> Research code for a thesis, not a maintained library. Feature attributions are computational relevance signals, not causal biological evidence.

## Repository layout

| Path | Description |
| --- | --- |
| `vae/` | VMGP-inspired variational model: network, Lightning module, data module, training, attribution and metrics |
| `contrastive_learning/` | Paper-faithful centroid N-pair contrastive model (Thor & Nettelblad, 2025): augmentation, encoder, training and Integrated Gradients attribution |
| `genomic/` | Model-independent genomic utilities: cohort QC, preprocessing, splitting, label mapping, $F_{ST}$, ranking and panel evaluation |
| `scripts/` | Experiment entry points (RQ1–RQ4, ovine cross-species) and split generation |
| `configs/` | Split and cohort configuration files |
| `splits/` | Frozen split manifests for the primary caprine cohort |
| `tests/` | Unit tests for the `genomic/` utilities and model integrations |
| `thesis/` | LaTeX sources, figures and bibliography of the thesis |
| `data/`, `results/`, `checkpoints/`, `lightning_logs/` | Local data and generated artifacts; not versioned |

## Datasets

- **Primary (caprine): ADAPTmap.** 4,653 genotyped samples and 53,347 raw SNP markers; after individual and breed eligibility filtering, 2,863 individuals from 34 breeds (2,437 development / 426 held-out test).
- **Validation (ovine): ISGC Sheep HapMap / Sheep Breed Diversity**, ovine SNP50 BeadChip. 2,819 animals and 49,034 markers after consortium quality control; 1,863 individuals from 28 breeds after eligibility filtering (1,585 development / 278 held-out test).

Raw genotype data are **not distributed** with this repository. Place the PLINK files under `data/` as expected by `configs/splits.yaml` and `configs/splits_sheep.yaml` before running the pipelines.

Key references: Kijas et al. (2012) for the sheep HapMap; Stella et al. (2018) and Colli et al. (2018) for ADAPTmap.

## Experimental protocol

Every data-dependent transformation (marker missingness filtering, minor-allele-frequency filtering, mode imputation, LD pruning) is fitted on the training partition of each cross-validation fold only. The held-out test set is opened once, after model configuration is fixed. Marker rankings are constructed inside training partitions and evaluated on untouched validation individuals. The same splits, preprocessing, metrics, panel sizes and seeds are used for all compared methods.

- Three stratified development folds; outer and fold seeds fixed at 42.
- RQ2 subsampling seeds 42–46, nested per-breed samples, preprocessing refitted for every subsample.
- Ranking methods compared on nested panels $K \in \{50, 100, 200, 500, 1000, 2000, 5000\}$ with pre-registered retention tolerances ($\epsilon_P = 0.02$ primary, $0.05$ secondary).
- Robustness analyses (alternative evaluators, native retraining, association baseline, valid-baseline Integrated Gradients) are development-only and never read the held-out test set.

## Getting started

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Run the unit tests:

```bash
uv run pytest
```

Generate or regenerate the frozen splits:

```bash
uv run python scripts/make_split.py --config configs/splits.yaml
uv run python scripts/make_split_sheep.py --config configs/splits_sheep.yaml
```

Experiment entry points (each script documents its options with `--help`):

| Script | Experiment |
| --- | --- |
| `scripts/run_rq1_final.py` | Final development training and held-out test evaluation (VMGP and contrastive) |
| `scripts/run_rq2.py` | Sample-efficiency sweep ($N \in \{5, \dots, 30\}$ per breed; 3 folds × 5 seeds) |
| `scripts/run_rq3.py` | Marker-efficiency and ranking comparison under the frozen RQ3/RQ4 protocol |
| `scripts/run_rq4.py` | Companion entry point for the RQ4 ranking comparison |
| `scripts/run_rq4_extended.py` | Robustness analyses: evaluator sensitivity, native retraining, association baseline, valid-baseline Integrated Gradients |
| `scripts/run_sheep_rq1.py` | Cross-species RQ1 evaluation on the ovine cohort |
| `scripts/run_sheep_rq2_mini.py` | Bounded ovine sample-efficiency mini-sweep |
| `scripts/analyze_ranking_overlap.py` | Pairwise Jaccard overlap between ranking manifests |
| `scripts/generate_thesis_figures.py` | Regenerate the thesis result figures from `results/` |

Model code is importable as the `vae` and `contrastive_learning` Python
packages; the scripts above provide the reproducible entry points.

## Main results

Macro-averaged F1 under the frozen protocol, as reported in the thesis:

| Analysis | Result |
| --- | --- |
| RQ1, caprine held-out test | VMGP 0.9703; contrastive 0.9496 (full-panel Random Forest development-CV reference 0.7189) |
| RQ2, caprine sample efficiency | $N_{\min} = 10$ training individuals per breed retain 98.5% of the $N = 30$ reference |
| RQ3, marker efficiency (frozen Random Forest evaluator) | 100–200 markers match the full post-QC panel; SHAP needs 500; contrastive Integrated Gradients 2,000–5,000 |
| RQ4, ranking comparison | Under the frozen Random Forest the classical rankings are the most marker-efficient; SHAP becomes competitive under linear or native evaluators on the caprine cohort, while the ovine cohort preserves the classical ordering |
| Cross-species, ovine held-out test | VMGP 0.9555; contrastive 0.9148; Random Forest 0.9330; $N = 10$ retains 98.6% |

Caprine and ovine raw scores are not directly comparable: the two cohorts differ in breed count, class balance, marker array, LD structure and available sample sizes. The ovine experiment applies the same methodological pipeline to an independent cohort; it is not a transfer of a caprine-trained model.

## Reproducibility notes

- The final test partition is never used for training, hyperparameter selection, early stopping, preprocessing, SNP ranking, feature selection or panel construction.
- `results/` is generated and not versioned; each script accepts its own output directory.
- GPU acceleration is recommended for model training; LD pruning and network training used CUDA when available.
- Dependencies are specified in `pyproject.toml` and pinned in `uv.lock`.

## Thesis

LaTeX sources, TikZ figures and the bibliography are in `thesis/`. The document is compiled with the `sapthesis` class (distributed separately).

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{spataro2026representation,
  author = {Lorenzo Spataro},
  title  = {Representation Learning and Attribution-Based Feature Selection in Livestock Genomics: A Leakage-Controlled Benchmark},
  school = {Sapienza University of Rome},
  year   = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

## Disclaimer

This is research code developed for a Master's thesis. It is not a diagnostic or commercial tool, and a high attribution score does not establish that a locus is biologically causal.
