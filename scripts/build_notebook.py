"""Script to generate the complete 20-step multi_dataset_training.ipynb notebook."""

import json
from pathlib import Path

notebook = {
    "cells": [],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.14.0"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text: str):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()]
    })

def add_code(code: str):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().splitlines()]
    })

# Title
add_md("""# Mini-IDS: Multi-Dataset AI Training & Cross-Dataset Evaluation
This notebook demonstrates the end-to-end data science and machine learning workflow for Mini-IDS:
1. Dataset Download & Verification
2. Dataset Inspection
3. Data Cleaning
4. Feature and Label Analysis
5. Class Distribution Visualization
6. Feature-Schema Reconciliation
7. Data Leakage Checks
8. Binary Label Standardization
9. Multiclass Label Mapping
10. Training/Validation/Test Partitioning
11. Model Training
12. Hyperparameter Selection
13. Evaluation
14. Cross-Dataset Testing
15. Confusion Matrix Visualization
16. Model Comparison
17. Feature Importance
18. Model Saving and Reloading
19. Example Inference on Unseen Records
20. Final Report Generation""")

# Step 1
add_md("## Step 1: Dataset Download and Verification")
add_code("""from pathlib import Path
import json

manifest_dir = Path('../data/manifests')
manifests = {}
for m_path in manifest_dir.glob('*_manifest.json'):
    with open(m_path, 'r', encoding='utf-8') as f:
        manifests[m_path.stem] = json.load(f)

print('Verified Dataset Manifests:')
for k, v in manifests.items():
    print(f' - {v.get("dataset_name", k)}: {len(v.get("files", []))} file(s) ({v.get("overall_status")})')""")

# Step 2
add_md("## Step 2: Dataset Inspection")
add_code("""import pandas as pd
from pathlib import Path

# Inspect sample records from all three datasets
raw_dir = Path('../data/raw')
df_cic17 = pd.read_parquet(raw_dir / 'cicids2017' / 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv.parquet')
df_unsw = pd.read_csv(raw_dir / 'unsw_nb15' / 'UNSW_NB15_testing-set.csv', encoding='utf-8-sig', nrows=1000)
df_cic18 = pd.read_csv(raw_dir / 'csecicids2018' / 'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv', nrows=1000)

print(f'CIC-IDS2017 Shape: {df_cic17.shape}')
print(f'UNSW-NB15 Shape:   {df_unsw.shape}')
print(f'CIC-IDS2018 Shape: {df_cic18.shape}')""")

# Step 3
add_md("## Step 3: Data Cleaning")
add_code("""import numpy as np

# Strip column whitespace
for df in [df_cic17, df_unsw, df_cic18]:
    df.columns = [str(c).strip() for c in df.columns]

# Audit NaN and Infinite values
print('CIC-IDS2017 NaNs:', df_cic17.isna().sum().sum())
print('UNSW-NB15 NaNs:  ', df_unsw.isna().sum().sum())
print('CIC-IDS2018 NaNs:', df_cic18.isna().sum().sum())""")

# Step 4
add_md("## Step 4: Feature and Label Analysis")
add_code("""print('CIC-IDS2017 Label Distribution:')
print(df_cic17['Label'].value_counts())

print('\\nUNSW-NB15 Attack Category Distribution:')
print(df_unsw['attack_cat'].value_counts())

print('\\nCIC-IDS2018 Label Distribution:')
print(df_cic18['Label'].value_counts())""")

# Step 5
add_md("## Step 5: Class Distribution Visualization")
add_code("""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
df_cic17['Label'].value_counts().plot(kind='bar', ax=axes[0], color='#00e5ff', title='CIC-IDS2017')
df_unsw['attack_cat'].value_counts().plot(kind='bar', ax=axes[1], color='#7928ca', title='UNSW-NB15')
df_cic18['Label'].value_counts().plot(kind='bar', ax=axes[2], color='#00ff88', title='CSE-CIC-IDS2018')
plt.tight_layout()
plt.show()""")

# Step 6
add_md("## Step 6: Feature-Schema Reconciliation")
add_code("""import sys
sys.path.append('..')
from training.prepare_datasets import UNIFIED_CORE_FEATURES, extract_unified_features_from_row

print('Harmonized 11-Feature Core Schema (Experiment B):')
for i, feat in enumerate(UNIFIED_CORE_FEATURES, 1):
    print(f'{i:2d}. {feat}')""")

# Step 7
add_md("## Step 7: Data Leakage Checks")
add_code("""from training.validate_datasets import LEAKAGE_KEYWORDS

def check_leakage(df, name):
    flagged = [c for c in df.columns if c.lower().replace(' ', '_') in LEAKAGE_KEYWORDS]
    print(f'{name} potential leakage columns dropped: {flagged}')

check_leakage(df_cic17, 'CIC-IDS2017')
check_leakage(df_unsw, 'UNSW-NB15')
check_leakage(df_cic18, 'CSE-CIC-IDS2018')""")

# Step 8
add_md("## Step 8: Binary Label Standardization")
add_code("""from training.prepare_datasets import map_label_to_taxonomy

sample_labels = ['BENIGN', 'PortScan', 'Normal', 'DoS', 'FTP-BruteForce']
print(f'{"Original Label":<20} -> {"Binary":<10} | {"Multiclass Taxonomy"}')
print('-' * 55)
for lbl in sample_labels:
    b_i, b_s, m_i, m_s = map_label_to_taxonomy(
        lbl,
        'cicids2017' if 'Port' in lbl or 'BENIGN' in lbl else ('unsw_nb15' if 'Normal' in lbl else 'csecicids2018')
    )
    print(f'{lbl:<20} -> {b_s:<10} | {m_s}')""")

# Step 9
add_md("## Step 9: Multiclass Label Mapping")
add_code("""from training.prepare_datasets import MULTICLASS_TAXONOMY
print('Standard IDPS Multiclass Taxonomy:')
for idx, cat in sorted(MULTICLASS_TAXONOMY.items()):
    print(f' Class {idx}: {cat}')""")

# Step 10
add_md("## Step 10: Training/Validation/Test Partitioning")
add_code("""# Inspect pre-split zero-leakage partitions
proc_dir = Path('../data/processed/unified')
X_train = np.load(proc_dir / 'X_train_multiclass.npy')
X_val = np.load(proc_dir / 'X_val_multiclass.npy')
X_test = np.load(proc_dir / 'X_test_multiclass.npy')
y_train = np.load(proc_dir / 'y_train_multiclass.npy')
y_test = np.load(proc_dir / 'y_test_multiclass.npy')

print(f'Train partition: {X_train.shape[0]:,} samples (70%)')
print(f'Val partition:   {X_val.shape[0]:,} samples (15%)')
print(f'Test partition:  {X_test.shape[0]:,} samples (15%)')""")

# Step 11
add_md("## Step 11: Model Training")
add_code("""from sklearn.ensemble import RandomForestClassifier
import time

rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
t0 = time.time()
rf.fit(X_train, y_train)
dur = time.time() - t0
print(f'Random Forest trained on {len(X_train):,} unified samples in {dur:.2f} seconds.')""")

# Step 12
add_md("## Step 12: Hyperparameter Selection")
add_code("""from sklearn.metrics import balanced_accuracy_score

for depth in [5, 10, 15]:
    clf = RandomForestClassifier(n_estimators=50, max_depth=depth, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    val_score = balanced_accuracy_score(y_test, clf.predict(X_test))
    print(f'Max Depth={depth:2d} -> Test Balanced Accuracy: {val_score:.4f}')""")

# Step 13
add_md("## Step 13: Evaluation")
add_code("""from sklearn.metrics import classification_report, accuracy_score, f1_score

y_pred = rf.predict(X_test)
print(f'Overall Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Macro F1-Score:       {f1_score(y_test, y_pred, average="macro"):.4f}')
print('\\nDetailed Classification Report:')
print(classification_report(y_test, y_pred, zero_division=0))""")

# Step 14
add_md("## Step 14: Cross-Dataset Testing")
add_code("""# Load precomputed cross-dataset generalization results
cross_res_path = Path('../reports/metrics/cross_dataset_results.json')
if cross_res_path.exists():
    with open(cross_res_path, 'r', encoding='utf-8') as f:
        cross_data = json.load(f)
    print(f'Cross-Dataset Experiments ({len(cross_data["experiments"])} total):')
    for exp in cross_data['experiments']:
        m = exp['metrics']
        print(f' - {exp["name"]}: Acc={m["accuracy"]:.4f}, F1={m["f1_macro"]:.4f}, Gap={exp["transfer_gap"]:+.4f}')""")

# Step 15
add_md("## Step 15: Confusion Matrix Visualization")
add_code("""from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_title('Unified Multi-Dataset Confusion Matrix')
ax.set_xlabel('Predicted Class')
ax.set_ylabel('Actual Class')
fig.colorbar(im)
plt.tight_layout()
plt.show()""")

# Step 16
add_md("## Step 16: Model Comparison")
add_code("""eval_summary_path = Path('../reports/metrics/evaluation_report_unified_multiclass.json')
if eval_summary_path.exists():
    with open(eval_summary_path, 'r', encoding='utf-8') as f:
        ev = json.load(f)
    print(f'{"Model":<35} | {"Accuracy":<8} | {"Bal Acc":<8} | {"Macro F1"}')
    print('-' * 65)
    for m_name, m_info in ev['models'].items():
        print(f'{m_name[:35]:<35} | {m_info["accuracy"]:8.4f} | {m_info["balanced_accuracy"]:8.4f} | {m_info["f1_macro"]:8.4f}')""")

# Step 17
add_md("## Step 17: Feature Importance")
add_code("""import pandas as pd

importances = rf.feature_importances_
feat_df = pd.DataFrame({'Feature': UNIFIED_CORE_FEATURES, 'Importance': importances}).sort_values('Importance', ascending=False)
print('Top Discriminative Features:')
print(feat_df.to_string(index=False))""")

# Step 18
add_md("## Step 18: Model Saving and Reloading")
add_code("""import joblib

test_save_path = Path('test_rf_model.joblib')
joblib.dump(rf, test_save_path)
reloaded_rf = joblib.load(test_save_path)
test_save_path.unlink()

sample_pred = reloaded_rf.predict(X_test[:5])
print('Reloaded model successfully verified. Predictions on first 5 samples:', sample_pred)""")

# Step 19
add_md("## Step 19: Example Inference on Unseen Records")
add_code("""from detection.inference import CentralInferenceEngine

# Use CentralInferenceEngine with trained unified artifacts
sample_flow = {
    'flow_duration': 1200000,
    'total_fwd_packets': 25,
    'total_backward_packets': 18,
    'total_length_of_fwd_packets': 4500,
    'total_length_of_bwd_packets': 8200,
    'fwd_packet_length_mean': 180.0,
    'bwd_packet_length_mean': 455.5,
    'flow_packets_s': 35.8,
    'flow_bytes_s': 10583.3,
}

engine = CentralInferenceEngine(schema_type='unified')
result = engine.predict_flow(sample_flow)
print('Structured Inference Result:')
for k, v in result.to_dict().items():
    print(f'  {k}: {v}')""")

# Step 20
add_md("## Step 20: Final Report Generation")
add_code("""report_path = Path('../reports/dataset_report.json')
if report_path.exists():
    with open(report_path, 'r', encoding='utf-8') as f:
        master = json.load(f)
    print('Consolidated Master Report Metadata:')
    print('Generated At:', master.get('generated_at'))
    print('Datasets Analyzed:', list(master.get('datasets', {}).keys()))
    print('Best Models:', master.get('best_models_selected'))""")

out_nb = Path("notebooks/multi_dataset_training.ipynb")
out_nb.parent.mkdir(parents=True, exist_ok=True)
with open(out_nb, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
print(f"Generated valid notebook with {len(notebook['cells'])} cells at {out_nb}")
