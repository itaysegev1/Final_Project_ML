"""
This is the src package for the Epicurious Hit/Miss project

Here we expose the public API that the notebooks and the CLI scripts import:

    from src import build_preprocessed_datasets
    from src import save_metrics, save_predictions, save_figure, fit_and_score
    from src import CulinaryFeatureExtractor

The project's "42 Guarantee" lives in src/train_utils.py::RANDOM_STATE — one
single source of truth that every model in the pipeline reads from
(see README section 3 for the audit).
"""

from src.data_foundation import (
    CulinaryFeatureExtractor,
    build_dataset,
)
from src.preprocessing import (
    build_preprocessed_datasets,
)
from src.train_utils import (
    DATASETS,
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
)

__all__ = [
    "CulinaryFeatureExtractor",
    "DATASETS",
    "RANDOM_STATE",
    "RESULTS_DIR",
    "build_dataset",
    "build_metrics_payload",
    "build_preprocessed_datasets",
    "confusion_matrix_figure",
    "fit_and_score",
    "load_metrics",
    "load_preprocessed",
    "model_results_dir",
    "print_dataset_block",
    "print_delta",
    "roc_curve_figure",
    "save_figure",
    "save_metrics",
    "save_predictions",
]
