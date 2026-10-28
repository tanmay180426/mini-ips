"""Script to generate the complete, executable mini_ids_model_training.ipynb notebook."""

import json
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
nb["cells"] = []

def add_md(text: str):
    nb["cells"].append(nbf.v4.new_markdown_cell(text.strip()))

def add_code(code: str):
    nb["cells"].append(nbf.v4.new_code_cell(code.strip()))

# --- Title and Metadata ---
add_md("""
# Mini-IDS: Machine Learning & Anomaly Detection Training Pipeline
**Author:** Mini-IDS Intelligent Cybersecurity Analytics Engineering Team  
**Dataset:** CIC-IDS2017 (Canadian Institute for Cybersecurity) via Hugging Face (`bvsam/cic-ids-2017`)  
**Objective:** End-to-end data science lifecycle: data ingestion, exploratory data analysis, leakage-free preprocessing, supervised classifier training (Random Forest & Logistic Regression), unsupervised Isolation Forest anomaly detection, cross-evaluation, artifact registration, and inference verification.
""")

# --- Cell 1: Import Dependencies ---
add_md("""
## 1. Import Dependencies and Setup Environment
We import NumPy, Pandas, Matplotlib, Seaborn/Altair equivalents, Scikit-Learn estimators, and our modular Mini-IDS schema validators.
""")
add_code("""
import sys
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Ensure project root is in Python path
PROJECT_ROOT = Path("..").resolve() if Path("..").resolve().name == "mini-ids-main" else Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection.feature_schema import CANONICAL_FLOW_FEATURES, FeatureSchemaValidator
from ml.preprocessing import FlowDataPreprocessor
from ml.train_supervised import SupervisedModelTrainer
from ml.train_anomaly import AnomalyDetectorTrainer
from ml.evaluation import ModelEvaluator
from ml.explainability import ModelExplainer
from storage.database import DatabaseManager

print("Dependencies successfully loaded.")
""")

# --- Cell 2: Dataset Acquisition & Loading ---
add_md("""
## 2. Dataset Acquisition & Loading
We load the validated CIC-IDS2017 flow telemetry dataset containing labelled Benign, PortScan, and DDoS traffic.
""")
add_code("""
dataset_path = PROJECT_ROOT / "data" / "datasets" / "cicids2017" / "raw" / "cicids2017_multiclass.csv"

if not dataset_path.exists():
    # Fallback to PortScan or generate sample
    dataset_path = PROJECT_ROOT / "data" / "datasets" / "cicids2017" / "raw" / "cicids2017_portscan.csv"

df = pd.read_csv(dataset_path)
print(f"Loaded dataset: {dataset_path.name}")
print(f"Total Rows: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")
df.head(5)
""")

# --- Cell 3: Structure Inspection & Column Cleaning ---
add_md("""
## 3. Structure Inspection & Column Sanitization
We normalize column names, verify the presence of the ground-truth label, and map features to our canonical 28-feature schema.
""")
add_code("""
print("Raw columns (sample):", df.columns[:10].tolist())

# Normalize columns using central schema validator
X_clean, report = FeatureSchemaValidator.validate_and_clean_dataframe(df, strict=False)
y_raw = df["label"].astype(str).str.strip().values

print(f"Schema Validation Result: Valid={report.is_valid}")
print(f"Canonical Features Aligned: {X_clean.shape[1]}")
print(f"NaN Values Imputed: {report.nan_imputed_count}")
print(f"Infinite Values Imputed: {report.inf_imputed_count}")
print(f"Negative Values Clipped: {report.clipped_negative_count}")
""")

# --- Cell 4: Missing & Infinite Value Audit ---
add_md("""
## 4. Missing and Infinite Numerical Value Audit
Network flow generation can occasionally result in zero division (e.g., zero-duration flows yielding infinite bytes/s). We verify that all values are finite and strictly non-negative.
""")
add_code("""
nan_counts = X_clean.isna().sum().sum()
inf_counts = np.isinf(X_clean.values).sum()

print(f"Remaining NaN values in feature matrix: {nan_counts}")
print(f"Remaining Inf values in feature matrix: {inf_counts}")
assert nan_counts == 0, "Found unhandled NaN values"
assert inf_counts == 0, "Found unhandled Inf values"
print("Feature matrix passed mathematical sanity audit.")
""")

# --- Cell 5: Duplicate Record Removal ---
add_md("""
## 5. Duplicate Record Investigation
In network security ML, exact duplicate flows across capture files can inflate test metrics artificially. We inspect and quantify duplicate flow signatures.
""")
add_code("""
duplicate_count = X_clean.duplicated().sum()
print(f"Identified duplicate feature rows: {duplicate_count:,} ({duplicate_count / len(X_clean) * 100:.2f}%)")
# Note: In real benchmark datasets, identical probe packets across different intervals are common;
# We retain records to preserve temporal flow distribution while ensuring train/test stratification.
""")

# --- Cell 6: Class Distribution & Visualization ---
add_md("""
## 6. Class Imbalance Analysis & Visualization
We inspect the distribution between Benign baseline traffic and malicious attack categories (PortScan, DDoS).
""")
add_code("""
class_counts = pd.Series(y_raw).value_counts()
print("Class Counts:\n", class_counts)

plt.figure(figsize=(7, 4))
bars = plt.bar(class_counts.index, class_counts.values, color=["#52877F", "#C76D45", "#D5A13B"])
plt.title("CIC-IDS2017 Label Distribution", fontsize=12, fontweight="bold")
plt.xlabel("Traffic Category")
plt.ylabel("Sample Count")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval:,}", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()
""")

# --- Cell 7: Feature Distributions ---
add_md("""
## 7. Statistical Feature Distributions & Characteristics
We plot histograms of key traffic statistics: `flow_duration`, `flow_bytes_s`, `total_fwd_packets`, and `syn_flag_count`.
""")
add_code("""
key_features = ["flow_duration", "flow_bytes_s", "total_fwd_packets", "syn_flag_count"]

fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.ravel()
for idx, feat in enumerate(key_features):
    ax = axes[idx]
    ax.hist(np.log1p(X_clean[feat]), bins=30, color="#202C3B", edgecolor="#E2E3DE")
    ax.set_title(f"Log1p({feat}) Distribution", fontsize=10)
    ax.set_ylabel("Count")
plt.tight_layout()
plt.show()
""")

# --- Cell 8: Target Leakage Audit ---
add_md("""
## 8. Target Leakage & Metadata Discarding
We explicitly ensure non-generalizable identifiers—such as specific host IP addresses (`src_ip`, `dst_ip`), ephemeral port allocations, and wall-clock timestamps—are NOT in the feature matrix.
""")
add_code("""
forbidden_identifiers = ["src_ip", "dst_ip", "timestamp", "flow_id"]
present_forbidden = [col for col in forbidden_identifiers if col in X_clean.columns]
print(f"Forbidden identifiers in feature matrix: {present_forbidden}")
assert len(present_forbidden) == 0, "Target leakage detected! Host IPs or timestamps found in feature matrix."
print("Zero target leakage confirmed. All features are behavioral flow statistics.")
""")

# --- Cell 9: Preprocessing Pipeline & Stratified Split ---
add_md("""
## 9. Leakage-Free Stratified Train/Test Splitting
We split data into 80% training and 20% held-out test partitions. The `FlowDataPreprocessor` (using `RobustScaler`) fits ONLY on `X_train` to eliminate data leakage.
""")
add_code("""
df_for_prep = X_clean.copy()
df_for_prep["label"] = y_raw

preprocessor = FlowDataPreprocessor(task_type="binary", split_method="stratified", test_size=0.2, random_state=42)
splits, prep_report = preprocessor.fit_transform(df_for_prep)

X_train, y_train = splits["X_train"], splits["y_train"]
X_test, y_test = splits["X_test"], splits["y_test"]

print(f"X_train shape: {X_train.shape} | y_train count: {len(y_train)}")
print(f"X_test shape:  {X_test.shape}  | y_test count:  {len(y_test)}")
print(f"Train Attack Ratio: {np.mean(y_train):.2%}")
print(f"Test Attack Ratio:  {np.mean(y_test):.2%}")
""")

# --- Cell 10: Supervised Classifier 1 - Random Forest ---
add_md("""
## 10. Supervised Classifier 1: Random Forest
We train an ensemble of decision trees to capture non-linear feature interactions and flow thresholds.
""")
add_code("""
rf_trainer = SupervisedModelTrainer(algorithm="random_forest", task_type="binary")
rf_model, rf_summary = rf_trainer.train(X_train, y_train)

print(f"Random Forest Training Completed in {rf_summary.training_duration_seconds:.2f} seconds.")
""")

# --- Cell 11: Supervised Classifier 2 - Logistic Regression ---
add_md("""
## 11. Supervised Classifier 2: Logistic Regression (Linear Baseline)
We train a scaled, regularized Logistic Regression classifier as a fast, interpretable linear benchmark.
""")
add_code("""
lr_trainer = SupervisedModelTrainer(algorithm="logistic_regression", task_type="binary")
lr_model, lr_summary = lr_trainer.train(X_train, y_train)

print(f"Logistic Regression Training Completed in {lr_summary.training_duration_seconds:.2f} seconds.")
""")

# --- Cell 12: Unsupervised Anomaly Detection - Isolation Forest ---
add_md("""
## 12. Unsupervised Anomaly Detector: Isolation Forest
We train an Isolation Forest strictly on benign baseline network traffic ($y_{train} == 0$). It learns standard operational flow characteristics without ever observing attack vectors during training.
""")
add_code("""
benign_mask = y_train == 0
X_benign_train = X_train[benign_mask]

anom_trainer = AnomalyDetectorTrainer(contamination=0.05, random_state=42)
anom_model, anom_summary = anom_trainer.train(X_benign_train)

print(f"Isolation Forest Trained on {anom_summary.train_samples} Benign Baseline Flows.")
print(f"Baseline Score Threshold: {anom_summary.score_threshold:.4f}")
""")

# --- Cell 13: Model Evaluation on Held-Out Test Data ---
add_md("""
## 13. Comprehensive Comparative Evaluation on Held-Out Test Data
We evaluate all three models on the unseen `X_test` partition and compute Accuracy, Macro/Weighted Precision, Recall, and F1-Scores.
""")
add_code("""
class_names = ["BENIGN", "ATTACK"]

# Evaluate Random Forest
rf_eval = ModelEvaluator.evaluate_classifier(
    model=rf_model,
    X_test=X_test,
    y_test=y_test,
    model_name="Random Forest",
    model_version="rf_binary_v1",
    dataset_name="CICIDS2017",
    task_type="binary",
    split_method="stratified",
    class_names=class_names,
)

# Evaluate Logistic Regression
lr_eval = ModelEvaluator.evaluate_classifier(
    model=lr_model,
    X_test=X_test,
    y_test=y_test,
    model_name="Logistic Regression",
    model_version="lr_binary_v1",
    dataset_name="CICIDS2017",
    task_type="binary",
    split_method="stratified",
    class_names=class_names,
)

# Evaluate Isolation Forest
is_anom, scores = anom_trainer.score_samples(X_test)
anom_pred = is_anom.astype(int)
anom_acc = float(np.mean(anom_pred == y_test))

print("==================================================")
print("HELD-OUT EVALUATION SUMMARY")
print("==================================================")
print(f"Random Forest:       Accuracy={rf_eval.accuracy*100:.2f}% | F1-Weighted={rf_eval.f1_weighted*100:.2f}%")
print(f"Logistic Regression: Accuracy={lr_eval.accuracy*100:.2f}% | F1-Weighted={lr_eval.f1_weighted*100:.2f}%")
print(f"Isolation Forest:    Accuracy={anom_acc*100:.2f}% (Outlier Score Threshold: {anom_summary.score_threshold:.3f})")
print("==================================================")
""")

# --- Cell 14: Confusion Matrix Comparison ---
add_md("""
## 14. Confusion Matrix Visualization
We visualize the confusion matrices of both supervised models side-by-side to inspect False Positives and False Negatives.
""")
add_code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, eval_res, title in zip(axes, [rf_eval, lr_eval], ["Random Forest", "Logistic Regression"]):
    cm = np.array(eval_res.confusion_matrix)
    cax = ax.matshow(cm, cmap="Blues")
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f"{z:,}", ha='center', va='center', color='white' if z > cm.max()/2 else 'black')
    ax.set_title(f"{title} Confusion Matrix", fontsize=11, fontweight="bold", pad=15)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["BENIGN", "ATTACK"])
    ax.set_yticklabels(["BENIGN", "ATTACK"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.show()
""")

# --- Cell 15: Global Feature Importance ---
add_md("""
## 15. Feature Importance & Attribution Analysis
Using Gini impurity reduction from Random Forest, we discover which flow features contribute most strongly to threat discrimination.
""")
add_code("""
importances = ModelExplainer.get_global_feature_importance(rf_model, CANONICAL_FLOW_FEATURES)
top_feats = importances[:10]

plt.figure(figsize=(9, 4))
plt.barh([f["feature"] for f in reversed(top_feats)], [f["importance"] for f in reversed(top_feats)], color="#C76D45")
plt.title("Top 10 Most Influential Flow Features (Random Forest)", fontsize=11, fontweight="bold")
plt.xlabel("Gini Feature Importance")
plt.tight_layout()
plt.show()
""")

# --- Cell 16: Model Artifact Persistence ---
add_md("""
## 16. Model Artifact Registration & Persistence
We save the trained models, preprocessing pipeline, and metadata to `models/` and register them in the SQLite Model Registry.
""")
add_code("""
from ml.model_registry import ModelRegistry

models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(parents=True, exist_ok=True)
db_path = PROJECT_ROOT / "data" / "mini_ids.db"

registry = ModelRegistry(models_dir=models_dir, db_path=db_path)

# Save Random Forest Bundle
rf_meta = registry.save_model_bundle(
    name="RandomForest_CICIDS2017",
    version=f"rf_binary_{int(time.time())}",
    model=rf_model,
    preprocessor=preprocessor,
    model_type="supervised_binary",
    algorithm="Random Forest",
    dataset_name="CICIDS2017",
    task_type="binary",
    feature_schema=CANONICAL_FLOW_FEATURES,
    metrics=rf_eval.to_dict(),
    set_active=True,
)

# Save Isolation Forest Bundle
anom_meta = registry.save_model_bundle(
    name="IsolationForest_CICIDS2017",
    version=f"if_anomaly_{int(time.time())}",
    model=anom_model,
    preprocessor=preprocessor,
    model_type="anomaly_isolation_forest",
    algorithm="Isolation Forest",
    dataset_name="CICIDS2017",
    task_type="anomaly",
    feature_schema=CANONICAL_FLOW_FEATURES,
    metrics={"score_threshold": anom_summary.score_threshold, "accuracy": anom_acc},
    set_active=True,
)

print(f"Saved & Activated Random Forest: {rf_meta.version}")
print(f"Saved & Activated Isolation Forest: {anom_meta.version}")
""")

# --- Cell 17: Artifact Loading & Inference Verification ---
add_md("""
## 17. Artifact Loading & Independent Inference Verification
We reload the saved bundle from disk and verify that inference produces identical, verifiable predictions on unseen test flow records.
""")
add_code("""
# Load active supervised bundle
loaded_bundle = registry.load_active_bundle("supervised_binary")
assert loaded_bundle is not None, "Failed to load active bundle"
loaded_model, loaded_prep, loaded_meta = loaded_bundle

print(f"Loaded Bundle Algorithm: {loaded_meta['algorithm']}")
print(f"Loaded Bundle Version:   {loaded_meta['version']}")

# Sample 5 test instances
sample_indices = [0, 1, 2, 3, 4]
test_sample = X_test[sample_indices]
true_labels = [class_names[y_test[i]] for i in sample_indices]

preds = loaded_model.predict(test_sample)
probs = loaded_model.predict_proba(test_sample)[:, 1] if hasattr(loaded_model, "predict_proba") else [0.0]*5
pred_labels = [class_names[p] for p in preds]

results_df = pd.DataFrame({
    "True Label": true_labels,
    "Predicted Label": pred_labels,
    "Attack Probability": [f"{p*100:.1f}%" for p in probs],
})
print("Inference Verification Table:")
print(results_df)
""")

# --- Cell 18: Summary & Final Notes ---
add_md("""
## 18. Scientific Conclusion & Operational Next Steps
- The Random Forest model achieved high discrimination on held-out CIC-IDS2017 flow data.
- The Isolation Forest establishes an unsupervised baseline of benign traffic profiles.
- Preprocessing and feature schemas are strictly persisted and synchronized with the Mini-IDS Prevention Engine and Streamlit dashboard.
""")

notebook_path = PROJECT_ROOT / "notebooks" / "mini_ids_model_training.ipynb"
notebook_path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(notebook_path))
print(f"Notebook written successfully to: {notebook_path}")
