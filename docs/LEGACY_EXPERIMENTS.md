# Legacy Experiments

This file records experimental variants that were removed from the
working tree but are preserved in Git history.

## `vae_copy/` — focal-loss + center-loss variant (removed)

`vae_copy/` was a near-duplicate of the canonical `vae/` package that
implemented a later experimental training variant of the VMGP model.

### What it did

- Added **focal loss** for the breed-classification term, enabled via
  `use_focal=True`.
- Added a **center loss** term on the latent mean `mu`, weighted by
  `center_loss_weight=0.1`, applied during training only.
- These were wired into `vae_copy/lightning_module.py` through the
  `model.focal_loss_fn` and `model.center_loss_fn` attributes defined in
  `vae_copy/network.py`.

### Limitations of the variant

- It ran on an older `experiment.py` runner that did **not** perform the
  newer held-out-test evaluation present in the canonical `vae/`.
- Its grid-search notebook (`vae_copy/vae_training.ipynb`) enabled the
  focal/center terms by default.

### Status

- **Not part of the current VMGP methodology.** The canonical `vae/`
  model trains with plain cross-entropy for breed classification and no
  center loss.
- The variant is preserved in Git history under the legacy baseline
  snapshot and must be treated as obsolete unless it is explicitly
  reconsidered as a future ablation experiment.
