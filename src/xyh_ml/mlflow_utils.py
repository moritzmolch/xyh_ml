"""Utilities for MLflow integration."""

import pickle
from pathlib import Path
from typing import Any

import mlflow


def log_sklearn_artifacts(
    artifact_dir: str | Path,
    artifacts: dict[str, Any],
    run_id: str | None = None,
) -> None:
    """
    Log sklearn artifacts (scalers, encoders, etc.) to MLflow.

    Parameters
    ----------
    artifact_dir : str | Path
        Directory to save artifacts before logging to MLflow.
    artifacts : dict[str, Any]
        Dictionary of artifact names and objects to log.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    for artifact_name, artifact_obj in artifacts.items():
        artifact_path = artifact_dir / f"{artifact_name}.pkl"
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact_obj, f)
        mlflow.log_artifact(
            local_path=str(artifact_path),
            artifact_path="preprocessors",
            run_id=run_id,
        )


def get_sqlite_tracking_uri(db_dir: str | Path) -> str:
    """
    Get a proper SQLite tracking URI.

    Parameters
    ----------
    db_dir : str | Path
        Directory to store the SQLite database.

    Returns
    -------
    str
        SQLite tracking URI in the format 'sqlite:////path/to/db'.
    """
    db_path = Path(db_dir).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Use four slashes for absolute Unix paths (sqlite:////path)
    # or five on Windows (sqlite://///C:/path)
    return f"sqlite:///{db_path}"
