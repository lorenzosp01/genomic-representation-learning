"""Model-independent genomic utilities for the thesis experimental framework."""

from genomic.splitting import (
    PROTOCOL_VERSION,
    SplitIndex,
    build_split,
    load_split,
    save_split,
)
from genomic.cohort import compute_individual_missingness, individual_qc_mask
from genomic.preprocessing import GenomicPreprocessor, load_bim
from genomic.labels import CanonicalLabelMapper
from genomic.experiment_data import (
    FinalDevelopmentData,
    FoldData,
    GenomicExperimentData,
    LockedTestData,
    load_genotype_matrix,
    locked_test_overlap,
)
from genomic.classification_metrics import aggregate_folds, compute_classification_metrics
from genomic.fst import weir_cockerham_fst
from genomic.panel_evaluation import (
    aggregate_panel_records,
    evaluate_full_panel,
    evaluate_panel,
    minimum_panel_size,
)
from genomic.ranking import MarkerRankingResult, compute_marker_ranking
from genomic.selection import best_epoch_from_checkpoint_path, select_best_configuration
from genomic.training_diagnostics import (
    ConvergenceHistoryCallback,
    classify_ceiling,
    summarize_convergence,
)

__all__ = [
    "PROTOCOL_VERSION",
    "SplitIndex",
    "build_split",
    "load_split",
    "save_split",
    "compute_individual_missingness",
    "individual_qc_mask",
    "GenomicPreprocessor",
    "load_bim",
    "CanonicalLabelMapper",
    "GenomicExperimentData",
    "FoldData",
    "FinalDevelopmentData",
    "LockedTestData",
    "load_genotype_matrix",
    "locked_test_overlap",
    "compute_classification_metrics",
    "aggregate_folds",
    "weir_cockerham_fst",
    "evaluate_panel",
    "evaluate_full_panel",
    "aggregate_panel_records",
    "minimum_panel_size",
    "MarkerRankingResult",
    "compute_marker_ranking",
    "select_best_configuration",
    "best_epoch_from_checkpoint_path",
    "ConvergenceHistoryCallback",
    "summarize_convergence",
    "classify_ceiling",
]
