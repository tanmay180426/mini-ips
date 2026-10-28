"""CLI Script: Validate integrity, schemas, and inference health of saved model artifacts."""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.model_registry import ModelRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate saved model artifacts in data/models/.")
    parser.add_argument("--version", type=str, default=None, help="Specific model version directory to check.")
    args = parser.parse_args()

    registry = ModelRegistry()
    print("=" * 65)
    print("Mini-IDS: Model Artifact Validator")
    print("=" * 65)

    if args.version:
        targets = [args.version]
    else:
        targets = [d.name for d in registry.models_dir.iterdir() if d.is_dir()]

    if not targets:
        print("No model artifacts found in data/models/.")
        return 0

    valid_count = 0
    for ver in targets:
        valid, msg, meta = registry.validate_artifact(ver)
        status_str = "[VALID]" if valid else "[INVALID]"
        print(f"\n{status_str} Model Version: {ver}")
        print(f"  * Details: {msg}")
        if meta:
            print(f"  * Algorithm:  {meta.get('algorithm')}")
            print(f"  * Task:       {meta.get('task_type')}")
            print(f"  * Features:   {len(meta.get('feature_schema', []))} inputs")
            print(f"  * Created At: {meta.get('created_at')}")
        if valid:
            valid_count += 1

    print("\n" + "=" * 65)
    print(f"Artifact Validation Summary: {valid_count} / {len(targets)} artifacts passed integrity checks.")
    print("=" * 65)
    return 0 if valid_count == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
