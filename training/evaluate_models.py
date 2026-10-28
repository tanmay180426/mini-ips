"""Comprehensive Model Evaluation Engine for Mini-IDS.

Calculates trustworthy, publication-grade cybersecurity evaluation metrics:
- Overall Accuracy & Balanced Accuracy
- Macro & Weighted Precision, Recall, and F1-Scores
- Per-Class Metrics Breakdown (Precision, Recall, F1, Support)
- Confusion Matrix (Actual vs Predicted)
- False Positive Rate (FPR / False Alarm Rate)
- False Negative Rate (FNR)
- Inference Latency (milliseconds per 1,000 samples)
- Artifact Storage Footprint (KB)
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Ensure project root in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mini_ids.training.evaluate_models")

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
METRICS_DIR = REPORTS_DIR / "metrics"


def evaluate_single_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    inv_map: Dict[int, str],
    benign_idx: int = 0,
    model_name: str = "Model",
    is_anomaly: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single model on test set and return detailed metric dictionary."""
    t0 = time.time()
    if is_anomaly:
        # 1 = inlier (benign), -1 = outlier (attack)
        raw_preds = model.predict(X_test)
        preds = np.where(raw_preds == -1, 1, 0)
        y_test_eval = np.where(y_test == benign_idx, 0, 1)
    else:
        preds = model.predict(X_test)
        y_test_eval = y_test
    eval_duration = time.time() - t0
    latency_ms_per_1k = (eval_duration / max(len(X_test), 1)) * 1000.0 * 1000.0

    acc = float(accuracy_score(y_test_eval, preds))
    b_acc = float(balanced_accuracy_score(y_test_eval, preds))
    f1_m = float(f1_score(y_test_eval, preds, average="macro", zero_division=0))
    f1_w = float(f1_score(y_test_eval, preds, average="weighted", zero_division=0))
    prec_m = float(precision_score(y_test_eval, preds, average="macro", zero_division=0))
    prec_w = float(precision_score(y_test_eval, preds, average="weighted", zero_division=0))
    rec_m = float(recall_score(y_test_eval, preds, average="macro", zero_division=0))
    rec_w = float(recall_score(y_test_eval, preds, average="weighted", zero_division=0))

    # Confusion Matrix
    unique_labels = sorted(list(set(np.unique(y_test_eval)).union(set(np.unique(preds)))))
    cm = confusion_matrix(y_test_eval, preds, labels=unique_labels)

    # False Alarm Rate (FPR) on benign samples
    benign_mask = (y_test_eval == 0)
    fpr = float((preds[benign_mask] != 0).mean()) if benign_mask.any() else 0.0

    # False Negative Rate (FNR) on attack samples
    attack_mask = (y_test_eval != 0)
    fnr = float((preds[attack_mask] == 0).mean()) if attack_mask.any() else 0.0

    # Per-class breakdown
    per_class = {}
    for idx in unique_labels:
        c_name = inv_map.get(idx, f"Class_{idx}")
        c_mask = (y_test_eval == idx)
        support = int(c_mask.sum())
        p_c = float(precision_score(y_test_eval == idx, preds == idx, zero_division=0))
        r_c = float(recall_score(y_test_eval == idx, preds == idx, zero_division=0))
        f_c = float(f1_score(y_test_eval == idx, preds == idx, zero_division=0))
        per_class[c_name] = {
            "precision": round(p_c, 4),
            "recall": round(r_c, 4),
            "f1": round(f_c, 4),
            "support": support,
        }

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(b_acc, 4),
        "f1_macro": round(f1_m, 4),
        "f1_weighted": round(f1_w, 4),
        "precision_macro": round(prec_m, 4),
        "precision_weighted": round(prec_w, 4),
        "recall_macro": round(rec_m, 4),
        "recall_weighted": round(rec_w, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "inference_latency_ms_per_1k": round(latency_ms_per_1k, 2),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": [inv_map.get(i, f"Class_{i}") for i in unique_labels],
    }


def evaluate_models_on_test_set(
    task_type: str = "multiclass",
    models_dir: Path = Path("models"),
    data_dir: Path = Path("data/datasets/cicids2017/processed"),
    output_report_path: Optional[Path] = None,
    dataset_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Load held-out test split, evaluate all models, and produce comprehensive benchmark report."""
    data_dir = Path(data_dir)
    # Check if data_dir has the files; if not, fall back to data/processed/<dataset_key>
    if not (data_dir / f"X_test_{task_type}.npy").exists():
        fallback = PROCESSED_DIR / (dataset_key or "cicids2017")
        if (fallback / f"X_test_{task_type}.npy").exists():
            data_dir = fallback

    logger.info("Loading test data for task: %s from %s", task_type, data_dir)
    X_test = np.load(data_dir / f"X_test_{task_type}.npy")
    y_test = np.load(data_dir / f"y_test_{task_type}.npy")

    meta_file = data_dir / f"metadata_{task_type}.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    inferred_dataset = dataset_key or meta_data.get("dataset_key", "cicids2017")
    label_map = meta_data.get("label_mapping", {"BENIGN": 0, "ATTACK": 1})
    inv_map = {int(k): v for k, v in meta_data.get("inverse_label_mapping", {"0": "BENIGN", "1": "ATTACK"}).items()}
    benign_idx = label_map.get("BENIGN", 0)

    report: Dict[str, Any] = {
        "dataset_key": inferred_dataset,
        "task_type": task_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_samples": int(len(X_test)),
        "class_distribution": {name: int((y_test == idx).sum()) for idx, name in inv_map.items()},
        "models": {},
    }

    models_dir = Path(models_dir)
    # Collect candidate models from both legacy dirs and models/supervised / models/anomaly
    candidate_paths: List[Tuple[str, Path, bool]] = []

    # 1. Check models/supervised/
    sup_dir = models_dir / "supervised"
    if sup_dir.exists():
        for f in sup_dir.glob("*.joblib"):
            if inferred_dataset in f.name and task_type in f.name:
                candidate_paths.append((f.stem, f, False))

    # 2. Check models/anomaly/
    anom_dir = models_dir / "anomaly"
    if anom_dir.exists():
        for f in anom_dir.glob("*.joblib"):
            if inferred_dataset in f.name:
                candidate_paths.append((f.stem, f, True))

    # 3. Check legacy directory bundles
    for mdir in sorted(models_dir.iterdir()):
        if mdir.is_dir() and (mdir / "model.joblib").exists():
            is_anom = "isolation_forest" in mdir.name
            if (task_type in mdir.name or is_anom):
                candidate_paths.append((mdir.name, mdir / "model.joblib", is_anom))

    logger.info("Found %d candidate models to evaluate for %s (%s)", len(candidate_paths), inferred_dataset, task_type)

    for name, mpath, is_anom in candidate_paths:
        try:
            model = joblib.load(mpath)
            res = evaluate_single_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
                inv_map=inv_map if not is_anom else {0: "BENIGN", 1: "ATTACK"},
                benign_idx=benign_idx,
                model_name=name,
                is_anomaly=is_anom,
            )
            res["artifact_path"] = str(mpath)
            res["artifact_size_kb"] = round(mpath.stat().st_size / 1024.0, 1)
            report["models"][name] = res
        except Exception as err:
            logger.warning("Could not evaluate model %s: %s", name, err)

    # Save report
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = output_report_path or (METRICS_DIR / f"evaluation_report_{inferred_dataset}_{task_type}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to: %s", out_path)

    # Print summary table
    print("\n" + "=" * 75)
    print(f"MODEL BENCHMARK RESULTS: {inferred_dataset.upper()} ({task_type.upper()}, N={len(X_test)})")
    print("=" * 75)
    print(f"{'Model Name':<35} | {'Acc':<6} | {'Bal Acc':<7} | {'F1-Macro':<8} | {'FAR (FPR)':<9} | {'Latency':<8}")
    print("-" * 75)
    for m_name, m_metrics in report["models"].items():
        short_name = m_name[:34]
        print(f"{short_name:<35} | {m_metrics['accuracy']:<6.4f} | {m_metrics['balanced_accuracy']:<7.4f} | {m_metrics['f1_macro']:<8.4f} | {m_metrics['false_positive_rate']:<9.4f} | {m_metrics['inference_latency_ms_per_1k']:<6.1f}ms")
    print("=" * 75)

    return report


def evaluate_all_datasets(models_dir: Path = Path("models")) -> Dict[str, Any]:
    """Evaluate models across all datasets and task modes."""
    datasets = ["cicids2017", "unsw_nb15", "csecicids2018", "unified"]
    all_reports = {}
    for ds in datasets:
        all_reports[ds] = {}
        for t in ["binary", "multiclass"]:
            d_dir = PROCESSED_DIR / ds
            if not (d_dir / f"X_test_{t}.npy").exists():
                continue
            rep = evaluate_models_on_test_set(
                task_type=t,
                models_dir=models_dir,
                data_dir=d_dir,
                dataset_key=ds,
            )
            all_reports[ds][t] = rep
    return all_reports


def main() -> None:
    """CLI Entrypoint for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate intrusion detection models on held-out test sets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "cicids2017", "unsw_nb15", "csecicids2018", "unified"],
        default="all",
        help="Dataset to evaluate on (default: all)",
    )
    parser.add_argument(
        "--task",
        choices=["both", "binary", "multiclass"],
        default="both",
        help="Classification task (default: both)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Custom output JSON path",
    )
    args = parser.parse_args()

    m_dir = Path(args.models_dir)
    selected_tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]

    if args.dataset == "all":
        evaluate_all_datasets(models_dir=m_dir)
    else:
        for t in selected_tasks:
            d_dir = PROCESSED_DIR / args.dataset
            out_p = Path(args.output_report) if args.output_report else None
            evaluate_models_on_test_set(
                task_type=t,
                models_dir=m_dir,
                data_dir=d_dir,
                output_report_path=out_p,
                dataset_key=args.dataset,
            )


if __name__ == "__main__":
    main()
