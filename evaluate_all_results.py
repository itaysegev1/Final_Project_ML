"""
This script aggregates all the model results we have under results/<slug>/
It does not train anything, it just walks each model folder looking for the
metrics.json file and builds the side by side summary table for all the models
on both feature matrices (Baseline and Advanced).

The order matters: we need first to run the train scripts (or the notebooks) so
that each results/<slug>/metrics.json exist before we run this aggregator.
We made it pretty leniant - it reports whatever models it finds and just notes
which ones are missing, so we can work model by model and not need to re-run
all the fleet on every change.

The schema we expect for the input json is in src/train_utils.build_metrics_payload
The aggregator only uses:
    display_name           - the row label
    datasets.<ds>.accuracy
    datasets.<ds>.f1
    datasets.<ds>.fp_rate / fn_rate (for the asymmetry note)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.train_utils import DATASETS, RESULTS_DIR


METRICS_BASENAME = "metrics.json"

# the order we want to show the seven models in the table
PREFERRED_ORDER: tuple = (
    "Perceptron",
    "LogisticRegression",
    "AdaBoost",
    "PCA(0.90) + KNN",
    "PCA(0.90) + KNN (Improved)",
    "RandomForest",
    "MLP (128,64)",
)


# Loading
def load_metrics_files(results_dir: Path) -> List[Dict[str, Any]]:
    """
    This function reads every results/<slug>/metrics.json into a list of dicts
    :param results_dir: the path of the results folder
    :return: a list with all the payload dicts we managed to read
    """
    # if the folder doesnt exist yet we just return empty
    if not results_dir.exists():
        return []
    payloads: List[Dict[str, Any]] = []
    # go over every slug folder inside results in sorted order
    for slug_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        path = slug_dir / METRICS_BASENAME
        # skip the folders that dont have a metrics.json yet
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            # save the slug name on the payload so we know where it came from
            payload.setdefault("_slug", slug_dir.name)
            payloads.append(payload)
        except (json.JSONDecodeError, OSError) as exc:
            # we dont want to crash, just warn and move on
            print(f"  [warning] could not read {path}: {exc}", file=sys.stderr)
    return payloads


def order_payloads(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    This function orders the payloads by the preferred order, and anything
    that we didnt put in the preferred order goes after in alphabetical order
    :param payloads: the list of payload dicts
    :return: the ordered list
    """
    # making a dict from model_name to its payload for quick lookup. if two
    # payloads claim the same model_name we keep the last and warn, instead
    # of silently dropping one
    name_to_payload: Dict[str, Dict[str, Any]] = {}
    for p in payloads:
        name = p.get("model_name", "")
        if name in name_to_payload:
            print(
                f"  [warning] duplicate model_name {name!r} "
                f"(slugs: {name_to_payload[name].get('_slug')!r} and "
                f"{p.get('_slug')!r}) — keeping the latter.",
                file=sys.stderr,
            )
        name_to_payload[name] = p
    ordered: List[Dict[str, Any]] = []
    # first we put the ones from the preferred order
    for name in PREFERRED_ORDER:
        if name in name_to_payload:
            ordered.append(name_to_payload.pop(name))
    # the rest goes at the end sorted alphabeticly
    for name in sorted(name_to_payload):
        ordered.append(name_to_payload[name])
    return ordered


def missing_from_preferred(payloads: List[Dict[str, Any]]) -> List[str]:
    """
    This function returns which of the expected models we didnt find
    :param payloads: the list of payload dicts we found
    :return: a list of the names that are missing from the lineup
    """
    present = {p.get("model_name") for p in payloads}
    return [name for name in PREFERRED_ORDER if name not in present]


# Summary table
def row_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    This function makes one row of the summary table from a single payload
    :param payload: the dict we loaded from metrics.json
    :return: a dict with all the columns we want for this row
    """
    ds = payload.get("datasets", {})
    # pulling the baseline and advanced results (DATASETS comes from
    # src.train_utils so the key spelling can never drift from the writers)
    baseline_key, advanced_key = DATASETS
    baseline_metrics = ds.get(baseline_key, {})
    advanced_metrics = ds.get(advanced_key, {})
    return {
        "Model":           payload.get("display_name", payload.get("model_name", "?")),
        "Acc (Baseline)":  baseline_metrics.get("accuracy", float("nan")),
        "Acc (Advanced)":  advanced_metrics.get("accuracy", float("nan")),
        "Δ Acc":           _safe_delta(advanced_metrics.get("accuracy"), baseline_metrics.get("accuracy")),
        "F1 (Baseline)":   baseline_metrics.get("f1", float("nan")),
        "F1 (Advanced)":   advanced_metrics.get("f1", float("nan")),
        "Δ F1":            _safe_delta(advanced_metrics.get("f1"), baseline_metrics.get("f1")),
    }


def _safe_delta(a: Optional[float], b: Optional[float]) -> float:
    """
    This function calculates a-b but only if both are not None, else returns nan
    :param a: the first value
    :param b: the second value
    :return: a-b or nan if one of them is missing
    """
    if a is None or b is None:
        return float("nan")
    return float(a) - float(b)


def _format_summary_table(rows) -> str:
    """
    This function takes the rows and returns the table as a nice string
    :param rows: the list of row dicts
    :return: the formatted table string
    """
    df = pd.DataFrame(rows)
    # explicit per-column formatting: the Δ columns always carry a sign, the
    # metric columns never do (no magnitude-based magic)
    formatters = {
        col: (lambda x: f"{x:+.4f}") if col.startswith("Δ") else (lambda x: f"{x: .4f}")
        for col in df.columns
        if col != "Model"
    }
    return df.to_string(index=False, formatters=formatters)


def build_summary_dataframe(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """
    This is the public helper, the notebooks call this to get the summary as
    a DataFrame they can show in a cell
    :param results_dir: the results folder path
    :return: the summary DataFrame
    """
    payloads = order_payloads(load_metrics_files(results_dir))
    return pd.DataFrame([row_from_payload(p) for p in payloads])


# Cross-model verdict
def _print_linear_vs_nonlinear_verdict(
    payloads: List[Dict[str, Any]],
    meaningful: float = 0.01,
) -> None:
    """
    This function prints a small note comparing the non linear models (RF, MLP)
    against the calibrated linear baseline (LogisticRegression) on the Advanced
    feature set. The idea is to see if the non linear models break the ceiling
    of the linear one or just plateau within noise.
    :param payloads: the list of payloads we loaded
    :param meaningful: the minimal delta to consider as a real gain (default 1%)
    """
    by_name = {p.get("model_name"): p for p in payloads}
    lr  = by_name.get("LogisticRegression")
    rf  = by_name.get("RandomForest")
    mlp = by_name.get("MLP (128,64)")
    # if we dont have LR or non of the non-linear ones, we cant compare
    if lr is None or (rf is None and mlp is None):
        return

    lr_advanced = lr["datasets"]["Advanced"]
    print("\n" + "=" * 72)
    print("  CROSS-MODEL NOTE — Linear baseline vs Non-linear models")
    print("=" * 72)
    print(f"\n  Reference (calibrated linear): LR Advanced   "
          f"Acc {lr_advanced['accuracy']:.4f}   F1 {lr_advanced['f1']:.4f}")

    def _line(label: str, payload: Dict[str, Any]) -> None:
        """
        Inner helper, prints one line for a single non-linear model showing the
        delta against LR and the verdict for it
        """
        ad = payload["datasets"]["Advanced"]
        d_acc = ad["accuracy"] - lr_advanced["accuracy"]
        d_f1 = ad["f1"] - lr_advanced["f1"]
        # picking the verdict according to the deltas
        if d_acc >= meaningful and d_f1 >= meaningful:
            verdict = "BREAKS the linear ceiling on both metrics."
        elif d_acc >= meaningful or d_f1 >= meaningful:
            verdict = "PARTIAL gain (one metric only)."
        elif d_acc <= -meaningful or d_f1 <= -meaningful:
            verdict = "UNDERPERFORMS the linear baseline."
        else:
            verdict = "PLATEAUS within noise of LR — no meaningful gain."
        print(
            f"  {label:<22} Acc {ad['accuracy']:.4f} ({d_acc:+.4f})   "
            f"F1 {ad['f1']:.4f} ({d_f1:+.4f})   →  {verdict}"
        )

    # printing the lines for the models we have
    if rf is not None:
        _line("RandomForest:", rf)
    if mlp is not None:
        _line("MLP (128,64):", mlp)


# Entry point
def main() -> int:
    """
    The main function, this is what we run from the command line
    :return: 0 if everything is ok, 1 if we didnt find any metrics files
    """
    print("=" * 72)
    print("  AGGREGATOR — model results summary")
    print("=" * 72)
    print(f"  Reading metrics from: {RESULTS_DIR}/<model_slug>/metrics.json")

    payloads = load_metrics_files(RESULTS_DIR)
    # if we found nothing it means the train scripts didnt run yet
    if not payloads:
        print(
            "\n  No metrics files found. Run the train_<model>.py scripts "
            "(or the notebooks/) first.",
            file=sys.stderr,
        )
        return 1

    # ordering them and checking which ones are missing from the lineup
    payloads = order_payloads(payloads)
    missing = missing_from_preferred(payloads)

    print(f"  Found {len(payloads)} model result file(s).")
    if missing:
        print(f"  Missing from the expected lineup: {missing}")
    print()

    # building the table rows and printing the summary
    rows = [row_from_payload(p) for p in payloads]
    print("=" * 72)
    print("  SUMMARY — Feature Engineering A/B comparison")
    print("=" * 72)
    print(_format_summary_table(rows))
    print(
        "\n  Reading the table: positive Δ means the engineered culinary "
        "features improved that metric over the baseline."
    )

    # finally the cross model verdict comparing LR to RF and MLP
    _print_linear_vs_nonlinear_verdict(payloads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
