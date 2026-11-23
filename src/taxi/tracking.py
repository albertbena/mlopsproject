"""What every MLflow run must be able to answer: which code (git commit) and which data (hash) made it."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import mlflow
import pandas as pd

from taxi.config import ROOT, tracking_uri

EXPERIMENT = "nyc-taxi-demand"


def setup() -> None:
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT)


def git_commit() -> str:
    """The commit the code runs from, with "-dirty" when there are uncommitted changes; "unknown" outside Git."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_md5(file: Path) -> str:
    """MD5 of a file: the same hash DVC writes for it in dvc.lock, so a run can be matched to a DVC version."""
    h = hashlib.md5()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    """A fingerprint of the exact rows a model was trained on (changes if one count changes)."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()
