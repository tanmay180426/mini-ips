"""Cross-Dataset Generalization and Transferability Benchmarking Engine.

Executes rigorous cross-dataset domain adaptation experiments across
CIC-IDS2017, UNSW-NB15, and CSE-CIC-IDS2018 using the harmonized 11-feature core:
- Experiment 1: In-Dataset Baselines (Train & test within same collection environment)
- Experiment 2: Transfer CIC-IDS2017 -> UNSW-NB15 (Disparate collection topology & attacks)
- Experiment 3: Transfer CIC-IDS2017 -> CSE-CIC-IDS2018 (Temporal evolution & new lab architecture)
- Experiment 4: Multi-Source Fusion (Train on CIC-IDS2017 + CSE-CIC-IDS2018 -> test on UNSW-NB15)
- Experiment 5: Leave-One-Dataset-Out (LODO) Evaluation across all 3 source permutations.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import RobustScaler

# Ensure project root in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.prepare_datasets import (
    UNIFIED_CORE_FEATURES,
    extract_unified_features_from_row,
    load_raw_dataset,
    map_label_to_taxonomy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mini_ids.training.cross_dataset")

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
METRICS_DIR = REPORTS_DIR / "metrics"


def extract_unified_corpus(
    dataset_key: str,
    max_records: int = 15_000,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract harmonized 11-feature matrix and binary labels for a dataset."""
    df = load_raw_dataset(dataset_key, max_records=max_records, random_seed=random_seed)
    label_col = next((c for c in ["Label", "label", "attack_cat"] if c in df.columns), None)
    if not label_col:
        raise ValueError(f"No label column found for {dataset_key}")

    features = []
    labels = []
    for _, row in df.iterrows():
        feat = extract_unified_features_from_row(row, dataset_key)
        b_int, _, _, _ = map_label_to_taxonomy(row[label_col], dataset_key)
        features.append([feat[f] for f in UNIFIED_CORE_FEATURES])
        labels.append(b_int)

    X = np.array(features, dtype=np.float32)
    X[np.isinf(X)] = np.nan
    y = np.array(labels, dtype=np.int32)
    return X, y, UNIFIED_CORE_FEATURES


def fit_and_evaluate_transfer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "rf",
    random_seed: int = 42,
) -> Dict[str, float]:
    """Fit zero-leakage imputer + scaler on X_train, fit model, evaluate on X_test."""
    # Zero leakage: Preprocessors fit ONLY on X_train
    X_train = np.array(X_train, dtype=np.float32, copy=True)
    X_test = np.array(X_test, dtype=np.float32, copy=True)
    X_train[np.isinf(X_train)] = np.nan
    X_test[np.isinf(X_test)] = np.nan
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_train)
    X_te_imp = imputer.transform(X_test)

    scaler = RobustScaler()
    X_tr_scl = scaler.fit_transform(X_tr_imp)
    X_te_scl = scaler.transform(X_te_imp)

    t0 = time.time()
    if model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=80,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )
    elif model_type == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=80,
            random_state=random_seed,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    clf.fit(X_tr_scl, y_train)
    fit_duration = time.time() - t0

    t_eval = time.time()
    preds = clf.predict(X_te_scl)
    eval_latency_ms = (time.time() - t_eval) / max(len(X_test), 1) * 1_000_000.0

    acc = float(accuracy_score(y_test, preds))
    b_acc = float(balanced_accuracy_score(y_test, preds))
    f1_m = float(f1_score(y_test, preds, average="macro", zero_division=0))
    prec_m = float(precision_score(y_test, preds, average="macro", zero_division=0))
    rec_m = float(recall_score(y_test, preds, average="macro", zero_division=0))

    # Benign False Alarm Rate (FPR)
    benign_mask = (y_test == 0)
    fpr = float((preds[benign_mask] != 0).mean()) if benign_mask.any() else 0.0

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(b_acc, 4),
        "f1_macro": round(f1_m, 4),
        "precision_macro": round(prec_m, 4),
        "recall_macro": round(rec_m, 4),
        "false_positive_rate": round(fpr, 4),
        "fit_duration_s": round(fit_duration, 2),
        "latency_ms_per_1k": round(eval_latency_ms, 2),
    }


def run_all_cross_dataset_experiments(
    max_records: int = 15_000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete battery of in-dataset and cross-dataset transfer experiments."""
    logger.info("Loading unified feature datasets into memory (max_records=%d)...", max_records)
    data = {}
    for ds_key in ["cicids2017", "unsw_nb15", "csecicids2018"]:
        logger.info("Extracting unified features for %s...", ds_key)
        X, y, _ = extract_unified_corpus(ds_key, max_records=max_records, random_seed=random_seed)
        data[ds_key] = {"X": X, "y": y}

    experiments: List[Dict[str, Any]] = []

    # -------------------------------------------------------------
    # 1. In-Dataset Baselines (Train on 70% of A, Test on 30% of A)
    # -------------------------------------------------------------
    logger.info("=== Running Experiment 1: In-Dataset Baselines ===")
    from sklearn.model_selection import train_test_split
    in_dataset_baselines = {}

    for ds_key in ["cicids2017", "unsw_nb15", "csecicids2018"]:
        X, y = data[ds_key]["X"], data[ds_key]["y"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=random_seed, stratify=y)
        res = fit_and_evaluate_transfer(X_tr, y_tr, X_te, y_te, model_type="rf", random_seed=random_seed)
        in_dataset_baselines[ds_key] = res["f1_macro"]

        experiments.append({
            "experiment_id": "EXP-1",
            "name": f"In-Dataset Baseline: {ds_key.upper()}",
            "train_sources": [ds_key],
            "test_source": ds_key,
            "train_samples": len(X_tr),
            "test_samples": len(X_te),
            "model": "Random Forest",
            "metrics": res,
            "transfer_gap": 0.0,
        })
        logger.info("Baseline %s -> Acc: %.4f, F1: %.4f", ds_key, res["accuracy"], res["f1_macro"])

    # -------------------------------------------------------------
    # 2. Transfer: CIC-IDS2017 -> UNSW-NB15
    # -------------------------------------------------------------
    logger.info("=== Running Experiment 2: CIC-IDS2017 -> UNSW-NB15 ===")
    X_tr, y_tr = data["cicids2017"]["X"], data["cicids2017"]["y"]
    X_te, y_te = data["unsw_nb15"]["X"], data["unsw_nb15"]["y"]
    res_exp2 = fit_and_evaluate_transfer(X_tr, y_tr, X_te, y_te, model_type="rf", random_seed=random_seed)
    gap_exp2 = round(in_dataset_baselines["unsw_nb15"] - res_exp2["f1_macro"], 4)
    experiments.append({
        "experiment_id": "EXP-2",
        "name": "Cross-Dataset Transfer: CIC-IDS2017 -> UNSW-NB15",
        "train_sources": ["cicids2017"],
        "test_source": "unsw_nb15",
        "train_samples": len(X_tr),
        "test_samples": len(X_te),
        "model": "Random Forest",
        "metrics": res_exp2,
        "transfer_gap": gap_exp2,
    })
    logger.info("CIC17 -> UNSW15 -> Acc: %.4f, F1: %.4f (Gap: %.4f)", res_exp2["accuracy"], res_exp2["f1_macro"], gap_exp2)

    # -------------------------------------------------------------
    # 3. Transfer: CIC-IDS2017 -> CSE-CIC-IDS2018
    # -------------------------------------------------------------
    logger.info("=== Running Experiment 3: CIC-IDS2017 -> CSE-CIC-IDS2018 ===")
    X_te, y_te = data["csecicids2018"]["X"], data["csecicids2018"]["y"]
    res_exp3 = fit_and_evaluate_transfer(X_tr, y_tr, X_te, y_te, model_type="rf", random_seed=random_seed)
    gap_exp3 = round(in_dataset_baselines["csecicids2018"] - res_exp3["f1_macro"], 4)
    experiments.append({
        "experiment_id": "EXP-3",
        "name": "Cross-Dataset Transfer: CIC-IDS2017 -> CSE-CIC-IDS2018",
        "train_sources": ["cicids2017"],
        "test_source": "csecicids2018",
        "train_samples": len(X_tr),
        "test_samples": len(X_te),
        "model": "Random Forest",
        "metrics": res_exp3,
        "transfer_gap": gap_exp3,
    })
    logger.info("CIC17 -> CIC18 -> Acc: %.4f, F1: %.4f (Gap: %.4f)", res_exp3["accuracy"], res_exp3["f1_macro"], gap_exp3)

    # -------------------------------------------------------------
    # 4. Multi-Source Training: CIC17 + CIC18 -> Test on UNSW-NB15
    # -------------------------------------------------------------
    logger.info("=== Running Experiment 4: Multi-Source Fusion (CIC17 + CIC18) -> UNSW-NB15 ===")
    X_multi = np.vstack([data["cicids2017"]["X"], data["csecicids2018"]["X"]])
    y_multi = np.concatenate([data["cicids2017"]["y"], data["csecicids2018"]["y"]])
    X_te, y_te = data["unsw_nb15"]["X"], data["unsw_nb15"]["y"]
    res_exp4 = fit_and_evaluate_transfer(X_multi, y_multi, X_te, y_te, model_type="rf", random_seed=random_seed)
    gap_exp4 = round(in_dataset_baselines["unsw_nb15"] - res_exp4["f1_macro"], 4)
    experiments.append({
        "experiment_id": "EXP-4",
        "name": "Multi-Source Training: (CIC17 + CIC18) -> UNSW-NB15",
        "train_sources": ["cicids2017", "csecicids2018"],
        "test_source": "unsw_nb15",
        "train_samples": len(X_multi),
        "test_samples": len(X_te),
        "model": "Random Forest",
        "metrics": res_exp4,
        "transfer_gap": gap_exp4,
    })
    logger.info("Multi-Source -> UNSW15 -> Acc: %.4f, F1: %.4f (Gap: %.4f)", res_exp4["accuracy"], res_exp4["f1_macro"], gap_exp4)

    # -------------------------------------------------------------
    # 5. Leave-One-Dataset-Out (LODO) Evaluation Permutations
    # -------------------------------------------------------------
    logger.info("=== Running Experiment 5: Leave-One-Dataset-Out (LODO) Permutations ===")
    datasets_all = ["cicids2017", "unsw_nb15", "csecicids2018"]
    for held_out in datasets_all:
        train_dss = [d for d in datasets_all if d != held_out]
        X_lodo_tr = np.vstack([data[d]["X"] for d in train_dss])
        y_lodo_tr = np.concatenate([data[d]["y"] for d in train_dss])
        X_lodo_te = data[held_out]["X"]
        y_lodo_te = data[held_out]["y"]

        res_lodo = fit_and_evaluate_transfer(X_lodo_tr, y_lodo_tr, X_lodo_te, y_lodo_te, model_type="rf", random_seed=random_seed)
        gap_lodo = round(in_dataset_baselines[held_out] - res_lodo["f1_macro"], 4)

        experiments.append({
            "experiment_id": "EXP-5",
            "name": f"LODO: Train ({' + '.join(train_dss)}) -> Test on {held_out.upper()}",
            "train_sources": train_dss,
            "test_source": held_out,
            "train_samples": len(X_lodo_tr),
            "test_samples": len(X_lodo_te),
            "model": "Random Forest",
            "metrics": res_lodo,
            "transfer_gap": gap_lodo,
        })
        logger.info("LODO (held out %s) -> Acc: %.4f, F1: %.4f (Gap: %.4f)", held_out, res_lodo["accuracy"], res_lodo["f1_macro"], gap_lodo)

    report = {
        "title": "Mini-IDS Cross-Dataset Transferability & Generalization Benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "common_features": UNIFIED_CORE_FEATURES,
        "experiments": experiments,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = METRICS_DIR / "cross_dataset_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved cross-dataset report: %s", report_file)

    # Print summary table
    print("\n" + "=" * 85)
    print("CROSS-DATASET GENERALIZATION & TRANSFERABILITY MATRIX")
    print("=" * 85)
    print(f"{'Experiment':<32} | {'Train Source':<18} | {'Test':<10} | {'Acc':<6} | {'F1-Macro':<8} | {'FAR (FPR)':<9} | {'Gap':<6}")
    print("-" * 85)
    for exp in experiments:
        t_src = "+".join(exp["train_sources"])[:17]
        m = exp["metrics"]
        print(f"{exp['name'][:32]:<32} | {t_src:<18} | {exp['test_source'][:10]:<10} | {m['accuracy']:<6.4f} | {m['f1_macro']:<8.4f} | {m['false_positive_rate']:<9.4f} | {exp['transfer_gap']:<+6.4f}")
    print("=" * 85)

    return report


def main() -> None:
    """CLI Entrypoint for cross-dataset evaluation."""
    parser = argparse.ArgumentParser(description="Run cross-dataset transferability experiments.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=15_000,
        help="Max records per dataset to use in transfer experiments (default: 15,000)",
    )
    args = parser.parse_args()
    run_all_cross_dataset_experiments(max_records=args.max_records)


if __name__ == "__main__":
    main()
