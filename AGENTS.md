# AGENTS.md

MSc thesis experiment code: learning compact genomic representations (a VMGP-inspired
VAE and a contrastive model) for livestock breed classification from SNP genotype
profiles. This is a research notebook repo, not a library or service.

## Read this first

- `docs/THESIS_CONTEXT.md` is auto-loaded as opencode instructions via `opencode.json`.
  It is the authoritative source for the research questions, genomic terminology, and the
  critical experimental rules (held-out test set, train-only preprocessing/balancing,
  seed consistency, no fabricated results). Follow it; do not restate it here.

## Environment & running

- Python 3.12 (` .python-version`), managed with `uv` (`uv sync` / `uv run ...`); a `.venv/`
  is already present.
- `requirements.txt` is stale (only a small sklearn/umap subset). The real dependency spec
  is `pyproject.toml` + `uv.lock`. Ignore `requirements.txt`.
- No README, tests, linter, formatter, typecheck, or CI config. `pyproject.toml` points at a
  nonexistent `README.md`.
- Everything is run through Jupyter notebooks, not CLI scripts:
  - `vae/vae_training.ipynb` — VMGP VAE experiments (grid search over `min_samples` × `latent_dim`).
  - `contrastive_learning/contrastive_adaptmap_with_integrated_gradients.ipynb` (and
    `..._centroid_ig.ipynb`) — paper-faithful contrastive model (Thor & Nettelblad 2025) +
    Integrated Gradients SNP attribution.
- The VAE notebook imports the `vae/` package after a `sys.path` tweak; run it with the repo
  root importable.

## Architecture

- `vae/` is the core library:
  - `data_module.py` — PLINK load + QC (missingness → imputation → MAF → LD pruning) + multi-label.
  - `network.py` — VAE encoder/decoder + breed predictor (VMGP architecture).
  - `lightning_module.py` — loss = `alpha*recon + (1-alpha)*pred`.
  - `experiment.py` — `run_experiment`, k-fold CV over train/val + held-out test eval.
  - `utils.py` — SMOTE-like balancing, GE / Local-Structure / Neighbor-Overlap metrics.
  - `analysis.py` — post-hoc metrics, confusion matrices, plots.
- `vae_copy/` is a near-duplicate of `vae/` (uncommitted). Prefer `vae/`; do not edit both.
- The contrastive notebooks are self-contained (own `GenomicDataModule`, encoder, loss inline);
  they do not reuse `vae/`.
- Data: PLINK binary goat ADAPTmap data in `data/` (gitignored). Path is hardcoded in the
  notebooks (`data/ADAPTmap_genotypeTOP_20160222_full.bed`); labels come from the `.fam` FID
  column; `vae/classificazione-capre.csv` maps breed → Continent/Caseina.

## Repo quirks / gotchas

- Branch `rework` is messy: staged deletions plus unstaged changes, tracked `__pycache__/`,
  and generated `.png`/`.pdf`/`.csv`/`lightning_logs/` in git. Check `git status` before
  assuming a clean tree.
- `.gitattributes` marks `*.ckpt` for Git LFS; `vae/checkpoints_vae/` is gitignored.
- Training uses `precision="16-mixed"`; DataLoaders hardcode `num_workers=4`; LD pruning uses
  CUDA when available.
- Docstrings and comments are in Italian.
- Markdown → PDF reports use `markdown` + `weasyprint` (`vae/generate_pdf.py`,
  `vae/convert_report.py`, `contrastive_learning/generate_pdf.py`).

# Thesis-Specific Instructions

This repository contains experimental code for an MSc thesis.

The scientific context and experimental constraints are documented in:

`docs/THESIS_CONTEXT.md`

Treat those constraints as authoritative for tasks involving:

- datasets;
- preprocessing;
- model architecture;
- training;
- evaluation;
- SNP ranking;
- feature selection;
- sample-size experiments;
- marker-reduction experiments.

## Priorities

When modifying scientific code, prioritize:

1. methodological correctness;
2. prevention of data leakage;
3. reproducibility;
4. fair model comparison;
5. generalization;
6. predictive performance.

Never optimize performance by using information from the final test set.

Do not silently change the scientific protocol.

If the existing implementation conflicts with THESIS_CONTEXT.md,
report the inconsistency explicitly.

Unknown scientific decisions must remain TODO rather than being guessed.