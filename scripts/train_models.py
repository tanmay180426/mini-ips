"""CLI Script: Train supervised classifiers and Isolation Forest anomaly models."""

import argparse
from pathlib import Path
import sys
import time

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from ml.dataset_manager import DatasetManager
from ml.evaluation import ModelEvaluator
from ml.model_registry import ModelRegistry
from ml.preprocessing import FlowDataPreprocessor
from ml.train_anomaly import AnomalyDetectorTrainer
from ml.train_supervised import SupervisedModelTrainer


def main() -> int:
    parser = argparse.ArgumentParser(description="Train machine learning intrusion detection models.")
    parser.add_argument(
        "--algorithm",
        choices=["random_forest", "logistic_regression", "hist_gradient_boosting", "isolation_forest", "all"],
        default="random_forest",
        help="Model algorithm to train (default: random_forest).",
    )
    parser.add_argument("--dataset", choices=["cicids2017", "unsw_nb15"], default="cicids2017")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--activate", action="store_true", help="Set the newly trained model as active.")
    args = parser.parse_args()

    manager = DatasetManager()
    paths = manager.get_dataset_paths(args.dataset)
    registry = ModelRegistry()

    # Ensure processed data exists
    splits_file = paths["processed"] / f"splits_{args.task}.joblib"
    preproc_file = paths["processed"] / f"preprocessor_{args.task}.joblib"

    if not splits_file.exists() or not preproc_file.exists():
        print("Processed splits not found. Automatically running data preprocessing...")
        raw_files = list(paths["raw"].glob("*.csv"))
        if not raw_files:
            manager.generate_reproducible_sample_dataset(args.dataset)
            raw_files = list(paths["raw"].glob("*.csv"))
        df = pd.read_csv(raw_files[0])
        preproc = FlowDataPreprocessor(task_type=args.task, split_method="stratified")
        splits, _ = preproc.fit_transform(df)
        joblib.dump(preproc, preproc_file)
        joblib.dump(splits, splits_file)
    else:
        preproc = joblib.load(preproc_file)
        splits = joblib.load(splits_file)

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    algos = (
        ["random_forest", "logistic_regression", "isolation_forest"]
        if args.algorithm == "all"
        else [args.algorithm]
    )

    print("=" * 65)
    print(f"Mini-IDS: Model Training Workspace | Dataset: {args.dataset.upper()}")
    print("=" * 65)

    for algo in algos:
        print(f"\n---> Training Algorithm: {algo.upper()}")
        version = f"{algo}_{args.task}_{int(time.time())}"

        if algo == "isolation_forest":
            # Anomaly detection: train on benign portion of training data
            benign_mask = y_train == 0
            X_benign_train = X_train[benign_mask] if benign_mask.any() else X_train

            trainer = AnomalyDetectorTrainer(contamination=0.05, n_estimators=100)
            model, summary = trainer.train(X_benign_train, training_population="benign_baseline")

            # Evaluate on test set
            is_anomaly, scores = trainer.score_samples(X_test)
            # Binary comparison: true attack = 1, true benign = 0
            acc = float(np.mean((is_anomaly.astype(int) == y_test)))
            metrics = {
                "training_duration_seconds": summary.training_duration_seconds,
                "train_samples": summary.train_samples,
                "feature_count": summary.feature_count,
                "score_threshold": summary.score_threshold,
                "test_accuracy_estimate": acc,
            }

            meta = registry.save_model_bundle(
                name=f"IsolationForest_{args.dataset.upper()}",
                version=version,
                model=model,
                preprocessor=preproc,
                model_type="anomaly_isolation_forest",
                algorithm="Isolation Forest",
                dataset_name=args.dataset,
                task_type="anomaly",
                feature_schema=preproc.feature_names_,
                metrics=metrics,
                set_active=args.activate,
            )
            print(f"  * Training Duration:  {summary.training_duration_seconds:.3f} s")
            print(f"  * Saved Artifact:     {meta.artifact_dir}")
            print(f"  * Model Version:      {version}")

        else:
            # Supervised Classifier
            trainer = SupervisedModelTrainer(algorithm=algo, task_type=args.task)
            model, summary = trainer.train(X_train, y_train)

            class_names = [preproc.inverse_label_mapping_.get(i, f"Class_{i}") for i in sorted(list(set(y_train)))]
            eval_results = ModelEvaluator.evaluate_classifier(
                model=model,
                X_test=X_test,
                y_test=y_test,
                model_name=algo,
                model_version=version,
                dataset_name=args.dataset,
                task_type=args.task,
                split_method=preproc.split_method,
                class_names=class_names,
            )

            # Save evaluation report to data/reports/
            ModelEvaluator.save_reports(eval_results)

            meta = registry.save_model_bundle(
                name=f"{algo}_{args.dataset.upper()}",
                version=version,
                model=model,
                preprocessor=preproc,
                model_type=f"supervised_{args.task}",
                algorithm=algo.replace("_", " ").title(),
                dataset_name=args.dataset,
                task_type=args.task,
                feature_schema=preproc.feature_names_,
                metrics=eval_results.to_dict(),
                set_active=args.activate,
            )

            print(f"  * Training Duration:  {summary.training_duration_seconds:.3f} s")
            print(f"  * Accuracy:           {eval_results.accuracy * 100:.2f}%")
            print(f"  * Precision (macro):  {eval_results.precision_macro * 100:.2f}%")
            print(f"  * Recall (macro):     {eval_results.recall_macro * 100:.2f}%")
            print(f"  * F1 Score (macro):   {eval_results.f1_macro * 100:.2f}%")
            if eval_results.false_positive_rate is not None:
                print(f"  * FPR:                {eval_results.false_positive_rate * 100:.2f}%")
            print(f"  * Saved Artifact:     {meta.artifact_dir}")
            print(f"  * Model Version:      {version}")

    print("\n" + "=" * 65)
    print("Training job complete. Artifacts registered in data/models/ and database.")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
