# Thesis Context

## Thesis

Working title:

**Representation Learning and Attribution-Based Feature Selection in
Livestock Genomics: A Leakage-Controlled Benchmark**

This repository contains the experimental code for an MSc thesis in
Computer Science concerning machine learning and deep representation
learning applied to livestock SNP genotype data.

The primary experimental analysis is performed on caprine genomic data
(ADAPTmap goat cohort). An independent ovine dataset (ISGC Sheep HapMap)
is used for methodological cross-species generalization validation.

Cross-species validation does NOT imply that the same SNP markers are
shared or expected to be informative across species.


## Main Computational Problem

Each individual is represented by a genotype profile containing P SNP
markers.

For N individuals, the genomic dataset can be represented as an
N x P matrix.

The problem is typically high-dimensional, with P potentially much
larger than N.

The main downstream task is:

    genotype profile -> breed classification


## Research Questions

### RQ1 — Breed classification

How effectively can machine-learning and deep representation-learning
methods classify livestock breeds from high-dimensional SNP genotype
profiles?


### RQ2 — Sample efficiency

How does breed-classification performance on unseen individuals change
as the number of training individuals per breed is reduced?

The variable being reduced is the number of TRAINING individuals.

Validation and test sets must remain held out.


### RQ3 — Marker efficiency

How does breed-classification performance change when the number of SNP
markers is progressively reduced?

The goal is to identify the smallest marker panel that retains most of
the predictive performance obtained from the full SNP panel.


### RQ4 — SNP ranking

Can SNP rankings obtained from learned representations and neural
feature-attribution methods identify informative reduced marker panels?

These rankings are compared with classical statistical and
machine-learning approaches.


## Genomic Terminology

DNA is organized into chromosomes.

A locus is a specific genomic position.

A SNP is a genomic position at which a single nucleotide varies among
individuals.

Alternative variants at a locus are called alleles.

For diploid autosomal loci, an individual carries two alleles, giving
rise to genotypes such as AA, AG and GG.

Genotypes may be represented computationally using allele dosage:

    AA -> 0
    AG -> 1
    GG -> 2

A genotype describes genetic variation carried by an individual.

A phenotype is an observable or measurable trait.

Traditional GWAS commonly studies:

    genotype -> phenotype

This thesis instead primarily studies:

    genotype profile -> breed


## Locus-wise and Multivariate Analysis

Traditional locus-wise approaches analyze loci individually.

Examples:
- locus-wise F_ST;
- traditional single-marker GWAS.

Multivariate approaches jointly process multiple SNP markers.

Examples relevant to this thesis:
- Canonical Discriminant Analysis;
- Random Forest;
- neural networks;
- variational representation learning;
- contrastive representation learning


## VMGP-Inspired Variational Model

The first deep representation-learning model is inspired by the
Variational Multi-view Genomics Predictor (VMGP).

IMPORTANT:

VMGP was originally developed for genomic prediction in PLANTS.

It must NOT be described as an already validated livestock
breed-classification model.

VMGP is used as a genomics-oriented architectural reference.

The adapted model investigates whether high-dimensional genotype
profiles can be mapped to a regularized lower-dimensional latent
representation while preserving information useful for breed
classification.

The current conceptual objective combines:

- genotype reconstruction;
- KL / variational regularization;
- breed classification.

Conceptually:

    genotype x
        -> encoder
        -> latent representation z
        -> reconstruction

and:

    latent representation z
        -> breed prediction


## Contrastive Representation Model

The second deep model follows a contrastive representation-learning
approach.

It represents a different inductive bias from the variational model.

Two stochastic transformations of the SAME individual genotype profile
form a positive pair:

    x
      -> augmentation A -> x_a -> encoder -> z_a
      -> augmentation B -> x_b -> encoder -> z_b

The current conceptual design follows a SimCLR-like contrastive
objective based on NT-Xent.

IMPORTANT:

Unless breed labels are explicitly used by the implementation in the
contrastive loss, do NOT claim that the model directly brings
individuals belonging to the same breed closer together.

The positive pair consists of two augmented views of the same
individual.

Representations are L2-normalized and therefore lie on a unit
hypersphere where cosine similarity can be used naturally.


## Why These Two Models?

The models represent two complementary hypotheses.

The variational model investigates whether useful genomic information
can be compressed into a regularized latent representation while
supporting reconstruction and breed prediction.

The contrastive model investigates which genomic information remains
stable across related augmented views without requiring reconstruction
of every SNP.

Their relative performance is an EMPIRICAL question.

Do not assume that either deep-learning approach must outperform
classical baselines.


## SNP Ranking

Candidate ranking approaches include:

- F_ST;
- Random Forest feature importance;
- Integrated Gradients;
- SHAP;
- other explicitly implemented baselines.

Feature attribution must NOT be interpreted as evidence of biological
causality.

A highly attributed SNP means that the model relied strongly on that
feature under the adopted experimental conditions.


## Critical Experimental Rules

The final test set must NEVER be used for:

- training;
- hyperparameter selection;
- early stopping;
- preprocessing parameter estimation;
- SNP ranking;
- feature selection;
- reduced-panel construction;
- model selection.

Any data-dependent preprocessing must be fitted on training data only.

Whenever possible, models must be compared using identical:

- train/validation/test splits;
- preprocessing;
- evaluation metrics;
- sample-size conditions;
- marker-panel sizes;
- random seeds.

Reproducibility and methodological correctness are more important than
maximizing a single accuracy value.


## Evaluation

Relevant classification metrics include:

- Accuracy;
- Macro-F1;
- Balanced Accuracy;
- per-class recall;
- confusion matrices.

Experiments should use repeated runs/seeds where appropriate and report
both average performance and variability.


## Minimum Requirements

If a minimum training sample size N_min or minimum SNP panel size P_min
is reported, its criterion must be defined BEFORE interpreting final
test results.

Conceptually:

    N_min = smallest training size retaining a predefined proportion
            of reference performance

    P_min = smallest SNP panel retaining a predefined proportion
            of full-panel performance

The tolerance threshold is still a methodological decision and must
not be invented by the agent.


## General Scientific Rule

Never fabricate missing experimental details or results.

If something cannot be determined from the code or thesis, mark it as:

    TODO / requires researcher decision