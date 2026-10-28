"""CLI Script: Download or acquire public datasets and verify manifests."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.dataset_manager import DatasetManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire and verify network intrusion datasets.")
    parser.add_argument(
        "--dataset",
        choices=["cicids2017", "unsw_nb15", "all"],
        default="cicids2017",
        help="Target dataset to acquire (default: cicids2017).",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Generate a deterministic, verified sample dataset for testing/offline use.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1200,
        help="Number of samples to generate if generating a sample dataset (default: 1200).",
    )
    args = parser.parse_args()

    manager = DatasetManager()
    datasets = ["cicids2017", "unsw_nb15"] if args.dataset == "all" else [args.dataset]

    print("=" * 65)
    print("Mini-IDS: Dataset Acquisition & Verification Tool")
    print("=" * 65)

    for ds in datasets:
        print(f"\nProcessing dataset: {ds.upper()}")
        if args.sample_only:
            print(f"Generating deterministic sample dataset ({args.samples:,} records)...")
            path = manager.generate_reproducible_sample_dataset(ds, n_samples=args.samples)
            print(f"Sample generated and manifest recorded at: {path}")
        else:
            success, msg, path = manager.download_dataset(ds)
            if success:
                print(f"[SUCCESS] {msg}")
            else:
                print(f"[NOTICE] {msg}")
                print(f"Generating deterministic sample fixture for {ds} to enable offline training...")
                s_path = manager.generate_reproducible_sample_dataset(ds, n_samples=args.samples)
                print(f"Sample fixture ready at: {s_path}")

    print("\nDataset status summary:")
    listing = manager.list_datasets()
    for name, info in listing.items():
        print(f"  * {info['name']}: {info['raw_count']} raw files, {info['processed_count']} processed files, {info['manifest_count']} manifests")

    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
