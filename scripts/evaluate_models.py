"""CLI Script: Evaluate trained model artifacts and output benchmark reports."""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
from ml.dataset_manager import DatasetManager
from ml.evaluation import ModelEvaluator
from ml.model_registry import ModelRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate registered model artifacts on held-out test splits.")
    parser.add_argument("--version", type=str, default=None, help="Model version to evaluate (default: active model).")
    parser.add_argument("--dataset", choices=["cicids2017", "unsw_nb15"], default="cicids2017")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    args = parser.parse_args()

    registry = ModelRegistry()
    manager = DatasetManager()
    paths = manager.get_dataset_paths(args.dataset)

    bundle = registry.load_active_bundle()
    if not bundle:
        print("No active model found. Run scripts/train_models.py first.")
        return 1

    model, preprocessor, meta = bundle
    print("=" * 65)
    print(f"Mini-IDS: Evaluating Model: {meta['name']} (`{meta['version']}`)")
    print("=" * 65)

    splits_file = paths["processed"] / f"splits_{args.task}.joblib"
    if not splits_file.exists():
        print(f"Processed test split not found at {splits_file}. Run scripts/preprocess_datasets.py first.")
        return 1

    splits = joblib.load(splits_file)
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    class_names = [preprocessor.inverse_label_mapping_.get(i, f"Class_{i}") for i in sorted(list(set(y_test)))]
    eval_res = ModelEvaluator.evaluate_classifier(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name=meta["name"],
        model_version=meta["version"],
        dataset_name=meta.get("dataset_name", args.dataset),
        task_type=meta.get("task_type", args.task),
        split_method=getattr(preprocessor, "split_method", "stratified"),
        class_names=class_names,
    )

    json_p, md_p = ModelEvaluator.save_reports(eval_res)

    print(f"\nEvaluation Results on {len(y_test):,} held-out test flows:")
    print(f"  * Accuracy:           {eval_res.accuracy * 100:.2f}%")
    print(f"  * Precision (Macro):  {eval_res.precision_macro * 100:.2f}%")
    print(f"  * Recall (Macro):     {eval_res.recall_macro * 100:.2f}%")
    print(f"  * F1 Score (Macro):   {eval_res.f1_macro * 100:.2f}%")
    print(f"  * Precision (Wt):     {eval_res.precision_weighted * 100:.2f}%")
    print(f"  * Recall (Wt):        {eval_res.recall_weighted * 100:.2f}%")
    print(f"  * F1 Score (Wt):      {eval_res.f1_weighted * 100:.2f}%")
    if eval_res.false_positive_rate is not None:
        print(f"  * False Positive Rate: {eval_res.false_positive_rate * 100:.2f}%")
    if eval_res.false_negative_rate is not None:
        print(f"  * False Negative Rate: {eval_res.false_negative_rate * 100:.2f}%")
    if eval_res.roc_auc is not None:
        print(f"  * ROC-AUC:            {eval_res.roc_auc:.4f}")
    print(f"  * Latency:            {eval_res.inference_latency_ms_per_sample:.4f} ms/sample")

    print(f"\nSaved Reports:")
    print(f"  - JSON:     {json_p}")
    print(f"  - Markdown: {md_p}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
