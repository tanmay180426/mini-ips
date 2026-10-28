"""Comprehensive Model Evaluation and Metrics Reporting Module.

Computes actual classification metrics, confusion matrices, ROC-AUC, FPR/FNR,
latency benchmarks, and generates machine- and human-readable evaluation reports.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("data") / "reports"


@dataclass
class EvaluationResults:
    """Rigorous evaluation metrics measured on held-out test data."""
    model_name: str
    model_version: str
    dataset_name: str
    task_type: str
    split_method: str
    test_sample_count: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    false_positive_rate: Optional[float]
    false_negative_rate: Optional[float]
    roc_auc: Optional[float]
    confusion_matrix: List[List[int]]
    class_labels: List[str]
    per_class_metrics: Dict[str, Dict[str, float]]
    inference_latency_ms_per_sample: float
    evaluation_timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelEvaluator:
    """Evaluates classification and anomaly models on held-out test partitions."""

    @staticmethod
    def evaluate_classifier(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        model_version: str,
        dataset_name: str,
        task_type: str = "binary",
        split_method: str = "stratified",
        class_names: Optional[List[str]] = None,
    ) -> EvaluationResults:
        """Run complete evaluation and generate real, verifiable metrics."""
        if len(X_test) == 0 or len(y_test) == 0:
            raise ValueError("Test set cannot be empty.")

        # Measure inference latency
        t0 = time.perf_counter()
        y_pred = model.predict(X_test)
        latency = (time.perf_counter() - t0) * 1000.0 / len(X_test)

        # Probabilities if available
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                y_prob = None

        acc = float(accuracy_score(y_test, y_pred))
        prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1_mac = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1_wt = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # Confusion matrix
        unique_classes = np.unique(np.concatenate([y_test, y_pred]))
        cm = confusion_matrix(y_test, y_pred, labels=unique_classes)
        cm_list = cm.tolist()

        # Binary specific FPR / FNR / ROC-AUC
        fpr_val: Optional[float] = None
        fnr_val: Optional[float] = None
        auc_val: Optional[float] = None

        if task_type == "binary" and len(unique_classes) == 2:
            tn, fp, fn, tp = cm.ravel()
            fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            fnr_val = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

            if y_prob is not None:
                try:
                    auc_val = float(roc_auc_score(y_test, y_prob[:, 1]))
                except Exception:
                    auc_val = None
        elif y_prob is not None and len(unique_classes) > 2:
            try:
                auc_val = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
            except Exception:
                auc_val = None

        # Per-class report
        default_names = [f"Class_{c}" for c in unique_classes]
        resolved_names = class_names if class_names and len(class_names) == len(unique_classes) else default_names

        clf_rep = classification_report(
            y_test,
            y_pred,
            labels=unique_classes,
            target_names=resolved_names,
            output_dict=True,
            zero_division=0,
        )

        per_class: Dict[str, Dict[str, float]] = {}
        for cname in resolved_names:
            if cname in clf_rep:
                per_class[cname] = {
                    "precision": float(clf_rep[cname]["precision"]),
                    "recall": float(clf_rep[cname]["recall"]),
                    "f1-score": float(clf_rep[cname]["f1-score"]),
                    "support": float(clf_rep[cname]["support"]),
                }

        results = EvaluationResults(
            model_name=model_name,
            model_version=model_version,
            dataset_name=dataset_name,
            task_type=task_type,
            split_method=split_method,
            test_sample_count=len(y_test),
            accuracy=acc,
            precision_macro=prec_macro,
            recall_macro=rec_macro,
            f1_macro=f1_mac,
            precision_weighted=prec_weighted,
            recall_weighted=rec_weighted,
            f1_weighted=f1_wt,
            false_positive_rate=fpr_val,
            false_negative_rate=fnr_val,
            roc_auc=auc_val,
            confusion_matrix=cm_list,
            class_labels=resolved_names,
            per_class_metrics=per_class,
            inference_latency_ms_per_sample=latency,
            evaluation_timestamp=time.time(),
        )

        return results

    @staticmethod
    def save_reports(results: EvaluationResults, output_dir: Path = REPORTS_DIR) -> Tuple[Path, Path]:
        """Save machine-readable JSON and human-readable Markdown evaluation reports."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"eval_{results.model_version}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, indent=2)

        md_path = output_dir / f"eval_{results.model_version}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Model Evaluation Report: {results.model_name} (`{results.model_version}`)\n\n")
            f.write(f"- **Dataset:** `{results.dataset_name}`\n")
            f.write(f"- **Task Type:** `{results.task_type}`\n")
            f.write(f"- **Split Methodology:** `{results.split_method}`\n")
            f.write(f"- **Test Samples Evaluated:** {results.test_sample_count:,}\n")
            f.write(f"- **Inference Latency:** {results.inference_latency_ms_per_sample:.4f} ms/sample\n\n")

            f.write("## Overall Performance Summary\n\n")
            f.write("| Metric | Value |\n")
            f.write("| :--- | :--- |\n")
            f.write(f"| **Accuracy** | {results.accuracy * 100:.2f}% |\n")
            f.write(f"| **Precision (Weighted)** | {results.precision_weighted * 100:.2f}% |\n")
            f.write(f"| **Recall (Weighted)** | {results.recall_weighted * 100:.2f}% |\n")
            f.write(f"| **F1 Score (Weighted)** | {results.f1_weighted * 100:.2f}% |\n")
            f.write(f"| **F1 Score (Macro)** | {results.f1_macro * 100:.2f}% |\n")
            if results.false_positive_rate is not None:
                f.write(f"| **False Positive Rate (FPR)** | {results.false_positive_rate * 100:.2f}% |\n")
            if results.false_negative_rate is not None:
                f.write(f"| **False Negative Rate (FNR)** | {results.false_negative_rate * 100:.2f}% |\n")
            if results.roc_auc is not None:
                f.write(f"| **ROC-AUC Score** | {results.roc_auc:.4f} |\n")

            f.write("\n## Per-Class Classification Breakdown\n\n")
            f.write("| Class | Precision | Recall | F1-Score | Support |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for cname, m in results.per_class_metrics.items():
                f.write(
                    f"| `{cname}` | {m['precision'] * 100:.2f}% | {m['recall'] * 100:.2f}% | "
                    f"{m['f1-score'] * 100:.2f}% | {int(m['support']):,} |\n"
                )

            f.write("\n## Confusion Matrix\n\n")
            f.write("```\n")
            f.write(f"Classes: {results.class_labels}\n")
            f.write(str(np.array(results.confusion_matrix)))
            f.write("\n```\n")

        return json_path, md_path
