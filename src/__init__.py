"""
This is the src package for the Epicurious Hit/Miss project

Here we expose the public API that the notebooks and the CLI scripts import:

    from src import build_preprocessed_datasets
    from src import save_metrics, save_predictions, save_figure, fit_and_score
    from src import CulinaryFeatureExtractor, make_culinary_extractor

The project's "42 Guarantee" is bound in src/_constants.py::RANDOM_STATE —
the single place the literal 42 exists. src.train_utils re-exports it for
convenience, and every model in the pipeline reads from there
(see research.md section 2.6.1 for the audit).
"""

from src.data_foundation import (
    CULINARY_KEYWORDS,
    CulinaryFeatureExtractor,
    build_dataset,
    make_culinary_extractor,
)
from src.preprocessing import (
    build_preprocessed_datasets,
)
from src.train_utils import (
    DATASETS,
    PROJECT_ROOT,
    RANDOM_STATE,
    RESULTS_DIR,
    build_metrics_payload,
    confusion_matrix_figure,
    fit_and_score,
    load_metrics,
    load_preprocessed,
    model_results_dir,
    print_dataset_block,
    print_delta,
    roc_curve_figure,
    save_figure,
    save_metrics,
    save_predictions,
    save_test_index,
)

__all__ = [
    "CULINARY_KEYWORDS",
    "CulinaryFeatureExtractor",
    "DATASETS",
    "PROJECT_ROOT",
    "RANDOM_STATE",
    "RESULTS_DIR",
    "build_dataset",
    "build_metrics_payload",
    "build_preprocessed_datasets",
    "confusion_matrix_figure",
    "fit_and_score",
    "load_metrics",
    "load_preprocessed",
    "make_culinary_extractor",
    "model_results_dir",
    "print_dataset_block",
    "print_delta",
    "roc_curve_figure",
    "save_figure",
    "save_metrics",
    "save_predictions",
    "save_test_index",
]
