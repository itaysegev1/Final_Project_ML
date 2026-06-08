"""
Aggregator — compiles the side-by-side model summary from `results/<slug>/`.

This script does NOT train any model. It walks each model subdirectory
under `results/` looking for `metrics.json`, and assembles the summary
table comparing all available models on both feature matrices.

Run order matters: train scripts (or notebooks) must have written
`results/<slug>/metrics.json` before this script will surface that
model. The aggregator is intentionally lenient — it reports whichever
models it finds and notes which expected ones are missing, so you can
iterate model-by-model rather than re-running the whole fleet on every
change.

Schema contract for the input JSONs lives in
`src/train_utils.py::build_metrics_payload`. The aggregator depends only
on:
    * `display_name`           — row label
    * `datasets.<ds>.accuracy`
    * `datasets.<ds>.f1`
    * `datasets.<ds>.fp_rate` / `fn_rate` (for the asymmetry note)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.train_utils import RESULTS_DIR


METRICS_BASENAME = "metrics.json"
DATASETS: tuple = ("Baseline", "Advanced")

# Conventional display order for the seven-model lineup.
PREFERRED_ORDER: tuple = (
    "Perceptron",
    "LogisticRegression",
    "AdaBoost",
    "PCA(0.90) + KNN",
    "PCA(0.90) + KNN (Improved)",
    "RandomForest",
    "MLP (128,64)",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_metrics_files(results_dir: Path) -> List[Dict[str, Any]]:
    """Read every `results/<slug>/metrics.json` into a list of dicts."""
    if not results_dir.exists():
        return []
    payloads: List[Dict[str, Any]] = []
    for slug_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        path = slug_dir / METRICS_BASENAME
        if not path.exists():
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            payload.setdefault("_slug", slug_dir.name)
            payloads.append(payload)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [warning] could not read {path}: {exc}", file=sys.stderr)
    return payloads


def _order_payloads(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    name_to_payload = {p.get("model_name", ""): p for p in payloads}
    ordered: List[Dict[str, Any]] = []
    for name in PREFERRED_ORDER:
        if name in name_to_payload:
            ordered.append(name_to_payload.pop(name))
    for name in sorted(name_to_payload):
        ordered.append(name_to_payload[name])
    return ordered


def _missing_from_preferred(payloads: List[Dict[str, Any]]) -> List[str]:
    present = {p.get("model_name") for p in payloads}
    return [name for name in PREFERRED_ORDER if name not in present]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def _row_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    ds = payload.get("datasets", {})
    b = ds.get("Baseline", {})
    a = ds.get("Advanced", {})
    return {
        "Model":           payload.get("display_name", payload.get("model_name", "?")),
        "Acc (Baseline)":  b.get("accuracy", float("nan")),
        "Acc (Advanced)":  a.get("accuracy", float("nan")),
        "Δ Acc":           _safe_delta(a.get("accuracy"), b.get("accuracy")),
        "F1 (Baseline)":   b.get("f1", float("nan")),
        "F1 (Advanced)":   a.get("f1", float("nan")),
        "Δ F1":            _safe_delta(a.get("f1"), b.get("f1")),
    }


def _safe_delta(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return float("nan")
    return float(a) - float(b)


def _format_summary_table(rows) -> str:
    df = pd.DataFrame(rows)
    return df.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:+.4f}" if isinstance(x, float) and abs(x) < 0.05 else f"{x: .4f}"
        ),
    )


def build_summary_dataframe(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Public helper: notebooks can call this to get the summary as a DataFrame."""
    payloads = _order_payloads(_load_metrics_files(results_dir))
    return pd.DataFrame([_row_from_payload(p) for p in payloads])


# ---------------------------------------------------------------------------
# Cross-model verdict
# ---------------------------------------------------------------------------
def _print_linear_vs_nonlinear_verdict(
    payloads: List[Dict[str, Any]],
    meaningful: float = 0.01,
) -> None:
    by_name = {p.get("model_name"): p for p in payloads}
    lr  = by_name.get("LogisticRegression")
    rf  = by_name.get("RandomForest")
    mlp = by_name.get("MLP (128,64)")
    if lr is None or (rf is None and mlp is None):
        return

    lr_a = lr["datasets"]["Advanced"]
    print("\n" + "=" * 72)
    print("  CROSS-MODEL NOTE — Linear baseline vs Non-linear models")
    print("=" * 72)
    print(f"\n  Reference (calibrated linear): LR Advanced   "
          f"Acc {lr_a['accuracy']:.4f}   F1 {lr_a['f1']:.4f}")

    def _line(label: str, payload: Dict[str, Any]) -> None:
        ad = payload["datasets"]["Advanced"]
        d_acc = ad["accuracy"] - lr_a["accuracy"]
        d_f1 = ad["f1"] - lr_a["f1"]
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

    if rf is not None:
        _line("RandomForest:", rf)
    if mlp is not None:
        _line("MLP (128,64):", mlp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("  AGGREGATOR — model results summary")
    print("=" * 72)
    print(f"  Reading metrics from: {RESULTS_DIR}/<model_slug>/metrics.json")

    payloads = _load_metrics_files(RESULTS_DIR)
    if not payloads:
        print(
            "\n  No metrics files found. Run the train_<model>.py scripts "
            "(or the notebooks/) first.",
            file=sys.stderr,
        )
        return 1

    payloads = _order_payloads(payloads)
    missing = _missing_from_preferred(payloads)

    print(f"  Found {len(payloads)} model result file(s).")
    if missing:
        print(f"  Missing from the expected lineup: {missing}")
    print()

    rows = [_row_from_payload(p) for p in payloads]
    print("=" * 72)
    print("  SUMMARY — Feature Engineering A/B comparison")
    print("=" * 72)
    print(_format_summary_table(rows))
    print(
        "\n  Reading the table: positive Δ means the engineered culinary "
        "features improved that metric over the baseline."
    )

    _print_linear_vs_nonlinear_verdict(payloads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
