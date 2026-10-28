"""CLI Script: Preprocess, clean, and partition network flow datasets."""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from ml.dataset_manager import DatasetManager
from ml.preprocessing import FlowDataPreprocessor


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess and clean network flow datasets.")
    parser.add_argument("--dataset", choices=["cicids2017", "unsw_nb15"], default="cicids2017")
    parser.add_argument("--file", type=str, default=None, help="Input CSV path.")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--split", choices=["stratified", "temporal"], default="stratified")
    args = parser.parse_args()

    manager = DatasetManager()
    paths = manager.get_dataset_paths(args.dataset)

    if args.file:
        input_csv = Path(args.file)
    else:
        raw_files = list(paths["raw"].glob("*.csv"))
        if not raw_files:
            print(f"No raw CSV files found in {paths['raw']}. Generating deterministic sample...")
            input_csv = manager.generate_reproducible_sample_dataset(args.dataset)
        else:
            input_csv = raw_files[0]

    print("=" * 65)
    print(f"Mini-IDS: Preprocessing Dataset: {input_csv.name}")
    print(f"Task: {args.task.upper()} | Split: {args.split.upper()}")
    print("=" * 65)

    df = pd.read_csv(input_csv)
    preprocessor = FlowDataPreprocessor(
        task_type=args.task,
        split_method=args.split,
        scale_features=True,
    )

    data_splits, report = preprocessor.fit_transform(df)

    # Save processed splits and preprocessor
    out_dir = paths["processed"]
    preproc_file = out_dir / f"preprocessor_{args.task}.joblib"
    splits_file = out_dir / f"splits_{args.task}.joblib"

    joblib.dump(preprocessor, preproc_file)
    joblib.dump(data_splits, splits_file)

    print("\nPreprocessing Results:")
    print(f"  * Initial Data Shape:        {report.initial_rows:,} rows, {report.initial_cols} cols")
    print(f"  * Duplicates Removed:        {report.duplicate_rows_removed:,}")
    print(f"  * Infinite Values Handled:   {report.inf_values_handled:,}")
    print(f"  * Leakage Columns Dropped:   {len(report.leakage_columns_dropped)}")
    print(f"  * Constant Columns Dropped:  {len(report.constant_columns_dropped)}")
    print(f"  * Final Usable Features:     {report.final_feature_count}")
    print(f"  * Training Set Size:         {report.train_rows:,} samples")
    print(f"  * Validation Set Size:       {report.val_rows:,} samples")
    print(f"  * Test Set Size:             {report.test_rows:,} samples")
    print(f"\nArtifacts persisted:")
    print(f"  - Pipeline: {preproc_file}")
    print(f"  - Splits:   {splits_file}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
