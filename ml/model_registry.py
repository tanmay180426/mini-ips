"""Model Registry and Artifact Lifecycle Management Module.

Handles saving, validating, versioning, activating, and loading trained ML models
and preprocessors from disk and SQLite metadata storage.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data") / "mini_ids.db"
MODELS_DIR = Path("data") / "models"


@dataclass
class ModelArtifactMeta:
    """Detailed registry record for a saved model artifact bundle."""
    name: str
    version: str
    model_type: str  # 'supervised_binary', 'supervised_multiclass', 'anomaly_isolation_forest'
    algorithm: str
    dataset_name: str
    task_type: str
    feature_schema: List[str]
    metrics: Dict[str, Any]
    artifact_dir: str
    model_path: str
    preprocessor_path: str
    created_at: float
    is_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """Manages model artifacts in `data/models/` and records metadata in SQLite."""

    def __init__(self, models_dir: Path = MODELS_DIR, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)

    def save_model_bundle(
        self,
        name: str,
        version: str,
        model: Any,
        preprocessor: Any,
        model_type: str,
        algorithm: str,
        dataset_name: str,
        task_type: str,
        feature_schema: List[str],
        metrics: Dict[str, Any],
        set_active: bool = False,
    ) -> ModelArtifactMeta:
        """Serialize model, preprocessor, and metadata into a versioned artifact folder."""
        version_dir = self.models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_file = version_dir / "model.joblib"
        preproc_file = version_dir / "preprocessor.joblib"
        meta_file = version_dir / "metadata.json"

        # 1. Serialize model and preprocessor safely
        if not HAS_JOBLIB:
            raise RuntimeError("Cannot save model bundle: optional dependency 'joblib' is not installed.")
        joblib.dump(model, model_file)
        if preprocessor is not None:
            joblib.dump(preprocessor, preproc_file)

        meta = ModelArtifactMeta(
            name=name,
            version=version,
            model_type=model_type,
            algorithm=algorithm,
            dataset_name=dataset_name,
            task_type=task_type,
            feature_schema=feature_schema,
            metrics=metrics,
            artifact_dir=str(version_dir),
            model_path=str(model_file),
            preprocessor_path=str(preproc_file) if preprocessor is not None else "",
            created_at=time.time(),
            is_active=set_active,
        )

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2)

        # 2. Persist to SQLite models table if database exists
        try:
            from storage.database import DatabaseManager
            with DatabaseManager(self.db_path) as db:
                db.init_db()
                db.insert_model({
                    "name": name,
                    "version": version,
                    "model_type": model_type,
                    "algorithm": algorithm,
                    "dataset_name": dataset_name,
                    "task_type": task_type,
                    "metrics": metrics,
                    "feature_schema": feature_schema,
                    "artifact_path": str(version_dir),
                    "created_at": meta.created_at,
                    "is_active": set_active,
                })
                if set_active:
                    db.set_active_model(version, is_active=True)
        except Exception as db_err:
            logger.warning("Could not write model metadata to DB (%s): %s", self.db_path, db_err)

        return meta

    def validate_artifact(self, version_or_dir: Union[str, Path]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate artifact integrity before loading into inference pipelines."""
        path = self.models_dir / str(version_or_dir) if not Path(version_or_dir).is_dir() else Path(version_or_dir)
        if not path.is_dir():
            return False, f"Artifact directory not found: {path}", None

        model_file = path / "model.joblib"
        preprocessor_file = path / "preprocessor.joblib"
        meta_file = path / "metadata.json"

        if not model_file.is_file():
            return False, f"Missing model binary at {model_file}", None
        if not meta_file.is_file():
            return False, f"Missing metadata file at {meta_file}", None

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as err:
            return False, f"Corrupted metadata JSON: {err}", None
        if meta.get("version") and str(meta["version"]) != path.name:
            return False, "Artifact directory name does not match the version in metadata.", None

        # Verify the complete inference bundle, including ordered preprocessing
        # compatibility, before reporting the model as usable.
        if not HAS_JOBLIB:
            return False, "Cannot validate model artifact: optional dependency 'joblib' is not installed.", None

        try:
            loaded_model = joblib.load(model_file)
            feature_schema = list(meta.get("feature_schema") or [])
            if not feature_schema:
                return False, "Artifact metadata has no ordered feature_schema.", None
            if not preprocessor_file.is_file():
                return False, f"Missing preprocessor bundle at {preprocessor_file}", None

            preprocessor = joblib.load(preprocessor_file)
            preprocessor_features = list(getattr(preprocessor, "feature_names_", []))
            if preprocessor_features != feature_schema:
                return False, "Metadata feature_schema does not match the fitted preprocessor order.", None

            expected_width = getattr(loaded_model, "n_features_in_", None)
            if expected_width is not None and int(expected_width) != len(feature_schema):
                return False, "Model input width does not match the recorded feature schema.", None

            sample_features = {name: 0.0 for name in feature_schema}
            sample = preprocessor.transform_inference(sample_features)
            if sample.ndim != 2 or sample.shape != (1, len(feature_schema)):
                return False, "Preprocessor dry run returned an incompatible feature shape.", None
            if not hasattr(loaded_model, "predict"):
                return False, "Loaded model does not provide predict().", None
            loaded_model.predict(sample)
        except Exception as dry_err:
            return False, f"Failed dry-run validation inference: {dry_err}", None

        return True, "Artifact verified and schema validated successfully.", meta

    def load_active_bundle(
        self,
        model_type: Optional[str] = None,
    ) -> Optional[Tuple[Any, Any, Dict[str, Any]]]:
        """Load the active model and preprocessor bundle from disk and database."""
        if not HAS_JOBLIB:
            logger.warning("Cannot load model bundle: optional dependency 'joblib' is not installed.")
            return None

        target_version = None
        target_artifact_path: Optional[Path] = None
        active_model_type: Optional[str] = None

        # Registry lookups must not initialize or migrate the database. Open an
        # existing registry read-only; a missing/incompatible registry falls
        # back to a matching artifact only.
        try:
            if self.db_path.is_file():
                db_uri = self.db_path.resolve().as_uri() + "?mode=ro"
                conn = sqlite3.connect(db_uri, uri=True)
                try:
                    if model_type:
                        row = conn.execute(
                            "SELECT version, artifact_path, model_type FROM models "
                            "WHERE is_active = 1 AND model_type = ? "
                            "ORDER BY created_at DESC LIMIT 1",
                            (str(model_type),),
                        ).fetchone()
                    else:
                        row = conn.execute(
                            "SELECT version, artifact_path, model_type FROM models WHERE is_active = 1 "
                            "ORDER BY created_at DESC LIMIT 1"
                        ).fetchone()
                    if row:
                        target_version = row[0]
                        active_model_type = row[2]
                        if row[1]:
                            target_artifact_path = Path(row[1])
                            if not target_artifact_path.is_absolute():
                                target_artifact_path = Path.cwd() / target_artifact_path
                finally:
                    conn.close()
        except Exception as err:
            logger.debug("Database query for active model failed: %s", err)

        # Fallback: scan disk for most recent verified model
        if not target_version:
            versions = []
            for directory in self.models_dir.iterdir() if self.models_dir.is_dir() else []:
                if not directory.is_dir() or not (directory / "model.joblib").is_file():
                    continue
                try:
                    with (directory / "metadata.json").open("r", encoding="utf-8") as f:
                        candidate_meta = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                if model_type is None or candidate_meta.get("model_type") == model_type:
                    versions.append(directory.name)
            versions.sort(reverse=True)
            if not versions:
                return None
            target_version = versions[0]
            target_artifact_path = self.models_dir / target_version

        artifact_path = target_artifact_path or (self.models_dir / target_version)
        valid, msg, meta = self.validate_artifact(artifact_path)
        if not valid or meta is None:
            logger.warning("Active artifact %s failed validation: %s", target_version, msg)
            return None
        if meta.get("version") and meta["version"] != target_version:
            logger.warning("Active artifact %s metadata reports version %s", target_version, meta["version"])
            return None
        expected_type = model_type or active_model_type
        if expected_type and meta.get("model_type") != expected_type:
            logger.warning(
                "Active artifact %s has type %s, expected %s",
                target_version,
                meta.get("model_type"),
                expected_type,
            )
            return None

        version_dir = artifact_path
        model = joblib.load(version_dir / "model.joblib")
        preprocessor = None
        if (version_dir / "preprocessor.joblib").exists():
            preprocessor = joblib.load(version_dir / "preprocessor.joblib")

        return model, preprocessor, meta
