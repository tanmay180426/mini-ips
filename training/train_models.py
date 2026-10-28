"""Multi-Dataset Model Training Engine for Mini-IDS.

Trains and evaluates 5 machine learning estimators:
1. Random Forest (Supervised Ensemble)
2. Extra Trees (Extremely Randomized Trees)
3. Logistic Regression (Linear Classifier Baseline)
4. HistGradientBoosting (Gradient-Boosted Decision Trees)
5. Isolation Forest (Unsupervised / Benign-Trained Anomaly Detection)

Supports:
- Dataset-specific models (CIC-IDS2017, UNSW-NB15, CSE-CIC-IDS2018)
- Unified cross-dataset model (Common Semantic Core)
- Binary and Multiclass tasks
- Zero-leakage validation
- Artifact persistence into `models/supervised/`, `models/anomaly/`, and `models/metadata/`.
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Ensure project root in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mini_ids.training.train_models")

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("models")
SUPERVISED_DIR = MODELS_DIR / "supervised"
ANOMALY_DIR = MODELS_DIR / "anomaly"
META_DIR = MODELS_DIR / "metadata"

ALL_ALGORITHMS = [
    "random_forest",
    "extra_trees",
    "logistic_regression",
    "hist_gradient_boosting",
    "isolation_forest",
]


def train_and_register_models(
    task_type: str = "multiclass",
    models_to_train: Optional[List[str]] = None,
    data_dir: Path = Path("data/datasets/cicids2017/processed"),
    output_dir: Path = Path("models"),
    random_seed: int = 42,
    dataset_key: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Train estimators on processed splits and save artifacts.

    Preserves full backward compatibility with existing tests and scripts.
    """
    if models_to_train is None:
        models_to_train = ["random_forest", "logistic_regression", "isolation_forest"]
    elif "all" in models_to_train:
        models_to_train = ALL_ALGORITHMS

    data_dir = Path(data_dir)
    # Check if data_dir exists, if not fall back to data/processed/<dataset_key>
    if not (data_dir / f"X_train_{task_type}.npy").exists():
        fallback = PROCESSED_DIR / (dataset_key or "cicids2017")
        if (fallback / f"X_train_{task_type}.npy").exists():
            data_dir = fallback

    # 1. Load preprocessed splits
    logger.info("Loading processed dataset splits for task: %s from %s", task_type, data_dir)
    X_train = np.load(data_dir / f"X_train_{task_type}.npy")
    y_train = np.load(data_dir / f"y_train_{task_type}.npy")
    X_val = np.load(data_dir / f"X_val_{task_type}.npy")
    y_val = np.load(data_dir / f"y_val_{task_type}.npy")

    meta_file = data_dir / f"metadata_{task_type}.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    inferred_dataset = dataset_key or meta_data.get("dataset_key", "cicids2017")
    label_map = meta_data.get("label_mapping", {"BENIGN": 0, "ATTACK": 1})
    inv_map = {int(k): v for k, v in meta_data.get("inverse_label_mapping", {"0": "BENIGN", "1": "ATTACK"}).items()}
    feature_names = meta_data.get("feature_names", [f"feat_{i}" for i in range(X_train.shape[1])])
    benign_idx = label_map.get("BENIGN", 0)

    # Output directory structuring
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sup_out = output_dir / "supervised"
    anom_out = output_dir / "anomaly"
    meta_out = output_dir / "metadata"
    sup_out.mkdir(parents=True, exist_ok=True)
    anom_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    if db_path is not None:
        registry = ModelRegistry(models_dir=output_dir, db_path=db_path)
    else:
        registry = ModelRegistry(models_dir=output_dir)
    results: Dict[str, Dict[str, Any]] = {}
    timestamp_str = str(int(time.time()))

    # --- 1. Random Forest ---
    if "random_forest" in models_to_train:
        logger.info("--- Training Random Forest (%s, %s) ---", inferred_dataset, task_type)
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=2,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )
        t0 = time.time()
        rf.fit(X_train, y_train)
        duration = time.time() - t0

        y_pred = rf.predict(X_val)
        acc = float(accuracy_score(y_val, y_pred))
        b_acc = float(balanced_accuracy_score(y_val, y_pred))
        f1_m = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        prec_m = float(precision_score(y_val, y_pred, average="macro", zero_division=0))
        rec_m = float(recall_score(y_val, y_pred, average="macro", zero_division=0))

        version = f"random_forest_{inferred_dataset}_{task_type}_{timestamp_str}"
        metrics = {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(b_acc, 4),
            "f1_macro": round(f1_m, 4),
            "precision_macro": round(prec_m, 4),
            "recall_macro": round(rec_m, 4),
            "training_duration_s": round(duration, 3),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }
        logger.info("Random Forest Val Acc: %.4f, Balanced Acc: %.4f, F1: %.4f (%.2fs)", acc, b_acc, f1_m, duration)

        joblib.dump(rf, sup_out / f"{version}.joblib")
        meta = registry.save_model_bundle(
            name=f"Random Forest ({inferred_dataset.upper()})",
            version=version,
            model=rf,
            preprocessor=None,
            model_type=f"supervised_{task_type}",
            algorithm="Random Forest",
            dataset_name=inferred_dataset,
            task_type=task_type,
            feature_schema=feature_names,
            metrics=metrics,
            set_active=True,
        )
        results["random_forest"] = {"version": version, "metrics": metrics, "meta": meta}

    # --- 2. Extra Trees ---
    if "extra_trees" in models_to_train:
        logger.info("--- Training Extra Trees (%s, %s) ---", inferred_dataset, task_type)
        et = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=2,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )
        t0 = time.time()
        et.fit(X_train, y_train)
        duration = time.time() - t0

        y_pred = et.predict(X_val)
        acc = float(accuracy_score(y_val, y_pred))
        b_acc = float(balanced_accuracy_score(y_val, y_pred))
        f1_m = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        prec_m = float(precision_score(y_val, y_pred, average="macro", zero_division=0))
        rec_m = float(recall_score(y_val, y_pred, average="macro", zero_division=0))

        version = f"extra_trees_{inferred_dataset}_{task_type}_{timestamp_str}"
        metrics = {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(b_acc, 4),
            "f1_macro": round(f1_m, 4),
            "precision_macro": round(prec_m, 4),
            "recall_macro": round(rec_m, 4),
            "training_duration_s": round(duration, 3),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }
        logger.info("Extra Trees Val Acc: %.4f, Balanced Acc: %.4f, F1: %.4f (%.2fs)", acc, b_acc, f1_m, duration)

        joblib.dump(et, sup_out / f"{version}.joblib")
        meta = registry.save_model_bundle(
            name=f"Extra Trees ({inferred_dataset.upper()})",
            version=version,
            model=et,
            preprocessor=None,
            model_type=f"supervised_{task_type}",
            algorithm="Extra Trees",
            dataset_name=inferred_dataset,
            task_type=task_type,
            feature_schema=feature_names,
            metrics=metrics,
            set_active=False,
        )
        results["extra_trees"] = {"version": version, "metrics": metrics, "meta": meta}

    # --- 3. Logistic Regression ---
    if "logistic_regression" in models_to_train:
        logger.info("--- Training Logistic Regression (%s, %s) ---", inferred_dataset, task_type)
        lr = LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_seed,
        )
        t0 = time.time()
        lr.fit(X_train, y_train)
        duration = time.time() - t0

        y_pred = lr.predict(X_val)
        acc = float(accuracy_score(y_val, y_pred))
        b_acc = float(balanced_accuracy_score(y_val, y_pred))
        f1_m = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        prec_m = float(precision_score(y_val, y_pred, average="macro", zero_division=0))
        rec_m = float(recall_score(y_val, y_pred, average="macro", zero_division=0))

        version = f"logistic_regression_{inferred_dataset}_{task_type}_{timestamp_str}"
        metrics = {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(b_acc, 4),
            "f1_macro": round(f1_m, 4),
            "precision_macro": round(prec_m, 4),
            "recall_macro": round(rec_m, 4),
            "training_duration_s": round(duration, 3),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }
        logger.info("Logistic Regression Val Acc: %.4f, Balanced Acc: %.4f, F1: %.4f (%.2fs)", acc, b_acc, f1_m, duration)

        joblib.dump(lr, sup_out / f"{version}.joblib")
        meta = registry.save_model_bundle(
            name=f"Logistic Regression ({inferred_dataset.upper()})",
            version=version,
            model=lr,
            preprocessor=None,
            model_type=f"supervised_{task_type}",
            algorithm="Logistic Regression",
            dataset_name=inferred_dataset,
            task_type=task_type,
            feature_schema=feature_names,
            metrics=metrics,
            set_active=False,
        )
        results["logistic_regression"] = {"version": version, "metrics": metrics, "meta": meta}

    # --- 4. HistGradientBoosting ---
    if "hist_gradient_boosting" in models_to_train:
        logger.info("--- Training HistGradientBoosting (%s, %s) ---", inferred_dataset, task_type)
        hgb = HistGradientBoostingClassifier(
            max_iter=100,
            random_state=random_seed,
        )
        t0 = time.time()
        hgb.fit(X_train, y_train)
        duration = time.time() - t0

        y_pred = hgb.predict(X_val)
        acc = float(accuracy_score(y_val, y_pred))
        b_acc = float(balanced_accuracy_score(y_val, y_pred))
        f1_m = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        prec_m = float(precision_score(y_val, y_pred, average="macro", zero_division=0))
        rec_m = float(recall_score(y_val, y_pred, average="macro", zero_division=0))

        version = f"hist_gradient_boosting_{inferred_dataset}_{task_type}_{timestamp_str}"
        metrics = {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(b_acc, 4),
            "f1_macro": round(f1_m, 4),
            "precision_macro": round(prec_m, 4),
            "recall_macro": round(rec_m, 4),
            "training_duration_s": round(duration, 3),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }
        logger.info("HistGradientBoosting Val Acc: %.4f, Balanced Acc: %.4f, F1: %.4f (%.2fs)", acc, b_acc, f1_m, duration)

        joblib.dump(hgb, sup_out / f"{version}.joblib")
        meta = registry.save_model_bundle(
            name=f"HistGradientBoosting ({inferred_dataset.upper()})",
            version=version,
            model=hgb,
            preprocessor=None,
            model_type=f"supervised_{task_type}",
            algorithm="HistGradientBoosting",
            dataset_name=inferred_dataset,
            task_type=task_type,
            feature_schema=feature_names,
            metrics=metrics,
            set_active=False,
        )
        results["hist_gradient_boosting"] = {"version": version, "metrics": metrics, "meta": meta}

    # --- 5. Isolation Forest (Anomaly Detector) ---
    if "isolation_forest" in models_to_train:
        logger.info("--- Training Isolation Forest (%s) ---", inferred_dataset)
        # Train strictly on benign samples when available
        benign_mask = (y_train == benign_idx)
        X_benign_train = X_train[benign_mask] if benign_mask.any() else X_train
        logger.info("Training Isolation Forest on %d benign baseline samples...", len(X_benign_train))

        iso = IsolationForest(
            n_estimators=100,
            contamination=0.10,
            random_state=random_seed,
            n_jobs=-1,
        )
        t0 = time.time()
        iso.fit(X_benign_train)
        duration = time.time() - t0

        raw_scores = -iso.score_samples(X_val)
        y_val_binary = np.where(y_val == benign_idx, 0, 1)

        val_threshold = float(np.percentile(raw_scores[y_val == benign_idx], 90)) if (y_val == benign_idx).any() else 0.5
        anom_pred = (raw_scores >= val_threshold).astype(int)

        acc = float(accuracy_score(y_val_binary, anom_pred))
        b_acc = float(balanced_accuracy_score(y_val_binary, anom_pred))
        f1_m = float(f1_score(y_val_binary, anom_pred, average="macro", zero_division=0))
        prec_m = float(precision_score(y_val_binary, anom_pred, average="macro", zero_division=0))
        rec_m = float(recall_score(y_val_binary, anom_pred, average="macro", zero_division=0))

        version = f"isolation_forest_{inferred_dataset}_{timestamp_str}"
        metrics = {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(b_acc, 4),
            "f1_macro": round(f1_m, 4),
            "precision_macro": round(prec_m, 4),
            "recall_macro": round(rec_m, 4),
            "score_threshold": round(val_threshold, 4),
            "training_duration_s": round(duration, 3),
            "train_samples": len(X_benign_train),
            "val_samples": len(X_val),
        }
        logger.info("Isolation Forest Val Acc: %.4f, Balanced Acc: %.4f, Threshold: %.4f (%.2fs)", acc, b_acc, val_threshold, duration)

        joblib.dump(iso, anom_out / f"{version}.joblib")
        meta = registry.save_model_bundle(
            name=f"Isolation Forest Anomaly Detector ({inferred_dataset.upper()})",
            version=version,
            model=iso,
            preprocessor=None,
            model_type="anomaly_isolation_forest",
            algorithm="Isolation Forest",
            dataset_name=inferred_dataset,
            task_type="anomaly",
            feature_schema=feature_names,
            metrics=metrics,
            set_active=True,
        )
        results["isolation_forest"] = {"version": version, "metrics": metrics, "meta": meta}

    # Save summary report in metadata
    summary_path = meta_out / f"training_summary_{inferred_dataset}_{task_type}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({k: v["metrics"] for k, v in results.items()}, f, indent=2)

    logger.info("Training cycle complete. Registered %d models for %s (%s).", len(results), inferred_dataset, task_type)
    return results


def train_all_models_across_datasets(
    models: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    output_dir: Path = Path("models"),
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train models across all 3 source datasets and the unified dataset."""
    datasets = ["cicids2017", "unsw_nb15", "csecicids2018", "unified"]
    task_list = tasks or ["binary", "multiclass"]
    all_results = {}

    for ds in datasets:
        ds_results = {}
        for task in task_list:
            data_dir = PROCESSED_DIR / ds
            if not (data_dir / f"X_train_{task}.npy").exists():
                logger.warning("Processed data missing for %s (%s). Skipping.", ds, task)
                continue
            logger.info("========== TRAINING: Dataset=%s, Task=%s ==========", ds.upper(), task.upper())
            res = train_and_register_models(
                task_type=task,
                models_to_train=models or ALL_ALGORITHMS,
                data_dir=data_dir,
                output_dir=output_dir,
                dataset_key=ds,
                db_path=db_path,
            )
            ds_results[task] = res
        all_results[ds] = ds_results

    return all_results


def main() -> None:
    """CLI Entrypoint for multi-dataset model training."""
    parser = argparse.ArgumentParser(description="Train intrusion detection and anomaly models.")
    parser.add_argument(
        "--dataset",
        choices=["all", "cicids2017", "unsw_nb15", "csecicids2018", "unified"],
        default="all",
        help="Dataset to train models on (default: all)",
    )
    parser.add_argument(
        "--task",
        choices=["both", "binary", "multiclass"],
        default="both",
        help="Classification task mode (default: both)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=f"Models to train: all, {', '.join(ALL_ALGORITHMS)}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for model artifacts",
    )
    args = parser.parse_args()

    selected_tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    selected_models = ALL_ALGORITHMS if "all" in args.models else args.models
    out_dir = Path(args.output_dir)

    if args.dataset == "all":
        results = train_all_models_across_datasets(
            models=selected_models,
            tasks=selected_tasks,
            output_dir=out_dir,
        )
    else:
        results = {}
        for t in selected_tasks:
            d_dir = PROCESSED_DIR / args.dataset
            results[t] = train_and_register_models(
                task_type=t,
                models_to_train=selected_models,
                data_dir=d_dir,
                output_dir=out_dir,
                dataset_key=args.dataset,
            )

    print("\n" + "=" * 65)
    print("TRAINING CYCLE COMPLETE - MODEL BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Artifacts registered to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
