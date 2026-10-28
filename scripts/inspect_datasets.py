"""CLI Script: Inspect dataset schemas, quality, and class distributions."""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from analytics.exploratory_analysis import DatasetProfiler
from ml.dataset_manager import DatasetManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect dataset quality, metadata, and distributions.")
    parser.add_argument("--file", type=str, default=None, help="Path to specific CSV file to inspect.")
    parser.add_argument("--dataset", choices=["cicids2017", "unsw_nb15"], default="cicids2017", help="Dataset to inspect.")
    args = parser.parse_args()

    manager = DatasetManager()
    paths = manager.get_dataset_paths(args.dataset)

    if args.file:
        csv_file = Path(args.file)
    else:
        # Pick first available raw file
        raws = list(paths["raw"].glob("*.csv"))
        if not raws:
            print(f"No CSV files found in {paths['raw']}. Run download_datasets.py first.")
            return 1
        csv_file = raws[0]

    print("=" * 65)
    print(f"Mini-IDS: Inspecting Dataset File: {csv_file.name}")
    print("=" * 65)

    df = pd.read_csv(csv_file)
    profile = DatasetProfiler.profile_dataframe(df)

    print(f"Total Rows:            {profile['total_rows']:,}")
    print(f"Total Columns:         {profile['total_cols']:,}")
    print(f"Duplicate Rows:        {profile['duplicate_rows']:,}")
    print(f"Label Column:          {profile['label_column']}")

    print("\nClass Distribution:")
    for k, v in profile["class_distribution"].items():
        pct = (v / profile['total_rows']) * 100.0
        print(f"  - {k:<25}: {v:>6,} ({pct:>5.1f}%)")

    if profile["constant_columns"]:
        print(f"\nConstant Columns ({len(profile['constant_columns'])}): {profile['constant_columns']}")

    missing_over_zero = {k: v for k, v in profile["missing_counts"].items() if v > 0}
    if missing_over_zero:
        print(f"\nMissing Values Detected in {len(missing_over_zero)} column(s):")
        for k, v in list(missing_over_zero.items())[:5]:
            print(f"  - {k}: {v} missing ({profile['missing_pcts'][k]}%)")
    else:
        print("\nMissing Values: None detected.")

    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
