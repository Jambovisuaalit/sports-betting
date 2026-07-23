"""Run a versioned sports-betting data extraction pipeline.

The pipeline uses the repository's existing ``SoccerDataLoader`` instead of a
fragile browser scraper. Every run writes immutable CSV snapshots and a JSON
manifest containing configuration, row counts, quality metrics and SHA-256
checksums.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sportsbet.datasets import SoccerDataLoader

SCHEMA_VERSION = "sports-data-snapshot/v1"
DEFAULT_OUTPUT_ROOT = "artifacts/sports-data"


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration parsed from environment variables."""

    leagues: list[str]
    divisions: list[int]
    years: list[int]
    odds_type: str
    output_root: Path
    run_id: str


def _csv_strings(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(name: str, default: str) -> list[int]:
    values = _csv_strings(name, default)
    try:
        return [int(value) for value in values]
    except ValueError as exc:
        msg = f"{name} must contain comma-separated integers: {values!r}"
        raise ValueError(msg) from exc


def load_config() -> PipelineConfig:
    """Load and validate pipeline configuration."""

    generated = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    github_run = os.getenv("GITHUB_RUN_ID")
    github_attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1")
    run_id = f"gh-{github_run}-{github_attempt}" if github_run else generated

    config = PipelineConfig(
        leagues=_csv_strings("PIPELINE_LEAGUES", "England,Germany,Italy,France,Spain"),
        divisions=_csv_ints("PIPELINE_DIVISIONS", "1"),
        years=_csv_ints("PIPELINE_YEARS", "2023,2024"),
        odds_type=os.getenv("PIPELINE_ODDS_TYPE", "market_maximum").strip(),
        output_root=Path(os.getenv("PIPELINE_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)),
        run_id=run_id,
    )
    if not config.leagues:
        raise ValueError("PIPELINE_LEAGUES cannot be empty")
    if not config.divisions:
        raise ValueError("PIPELINE_DIVISIONS cannot be empty")
    if not config.years:
        raise ValueError("PIPELINE_YEARS cannot be empty")
    if not config.odds_type:
        raise ValueError("PIPELINE_ODDS_TYPE cannot be empty")
    return config


def sha256_file(path: Path) -> str:
    """Return a SHA-256 checksum for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_columns(frame: pd.DataFrame) -> list[str]:
    return [" | ".join(map(str, column)) if isinstance(column, tuple) else str(column) for column in frame.columns]


def frame_quality(frame: pd.DataFrame) -> dict[str, Any]:
    """Calculate source-independent data-quality metrics."""

    stringified = frame.astype("string")
    numeric = frame.select_dtypes(include=[np.number])
    return {
        "rows": int(len(frame)),
        "columns": _string_columns(frame),
        "column_count": int(len(frame.columns)),
        "duplicate_rows": int(stringified.duplicated().sum()) if len(frame) else 0,
        "null_cells": int(frame.isna().sum().sum()),
        "infinite_numeric_cells": int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
        if not numeric.empty
        else 0,
    }


def _validate_unique_columns(name: str, frame: pd.DataFrame) -> None:
    columns = _string_columns(frame)
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        msg = f"{name} contains duplicate columns: {duplicates}"
        raise ValueError(msg)


def _validate_odds(name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    observed = numeric.stack(future_stack=True).dropna()
    if observed.empty:
        msg = f"{name} contains no numeric odds"
        raise ValueError(msg)
    invalid = observed[observed <= 1.0]
    if not invalid.empty:
        msg = f"{name} contains {len(invalid)} decimal odds values <= 1.0"
        raise ValueError(msg)


def validate_frames(frames: dict[str, pd.DataFrame]) -> None:
    """Apply hard quality gates before publishing a snapshot."""

    required = {
        "train_features",
        "train_targets",
        "train_odds",
        "fixtures_features",
        "fixtures_targets",
        "fixtures_odds",
    }
    missing = required.difference(frames)
    if missing:
        msg = f"Missing required frames: {sorted(missing)}"
        raise ValueError(msg)

    if frames["train_features"].empty:
        raise ValueError("Training features are empty")

    train_rows = {len(frames[name]) for name in ("train_features", "train_targets", "train_odds")}
    if len(train_rows) != 1:
        raise ValueError(f"Training frame row counts are not aligned: {sorted(train_rows)}")

    fixture_rows = {len(frames[name]) for name in ("fixtures_features", "fixtures_targets", "fixtures_odds")}
    if len(fixture_rows) != 1:
        raise ValueError(f"Fixture frame row counts are not aligned: {sorted(fixture_rows)}")

    for name, frame in frames.items():
        _validate_unique_columns(name, frame)
        metrics = frame_quality(frame)
        if metrics["infinite_numeric_cells"]:
            raise ValueError(f"{name} contains infinite numeric values")

    _validate_odds("train_odds", frames["train_odds"])
    _validate_odds("fixtures_odds", frames["fixtures_odds"])


def extract_frames(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    """Download historical and upcoming soccer data."""

    parameter_grid = {
        "league": config.leagues,
        "division": config.divisions,
        "year": config.years,
    }
    loader = SoccerDataLoader(param_grid=parameter_grid)
    x_train, y_train, o_train = loader.extract_train_data(odds_type=config.odds_type)
    x_fixtures, y_fixtures, o_fixtures = loader.extract_fixtures_data()
    return {
        "train_features": x_train,
        "train_targets": y_train,
        "train_odds": o_train,
        "fixtures_features": x_fixtures,
        "fixtures_targets": y_fixtures,
        "fixtures_odds": o_fixtures,
    }


def write_snapshot(config: PipelineConfig, frames: dict[str, pd.DataFrame]) -> Path:
    """Write an immutable snapshot atomically and return its directory."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    final_directory = config.output_root / config.run_id
    temporary_directory = config.output_root / f".{config.run_id}.tmp"

    if final_directory.exists():
        raise FileExistsError(f"Snapshot already exists: {final_directory}")
    shutil.rmtree(temporary_directory, ignore_errors=True)
    temporary_directory.mkdir(parents=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    file_entries: dict[str, Any] = {}
    try:
        for name, frame in frames.items():
            path = temporary_directory / f"{name}.csv"
            frame.to_csv(path, index=True)
            file_entries[name] = {
                "path": path.name,
                "sha256": sha256_file(path),
                "quality": frame_quality(frame),
            }

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": config.run_id,
            "generated_at": generated_at,
            "source": "sportsbet.datasets.SoccerDataLoader",
            "configuration": {
                "leagues": config.leagues,
                "divisions": config.divisions,
                "years": config.years,
                "odds_type": config.odds_type,
            },
            "files": file_entries,
        }
        manifest_path = temporary_directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        temporary_directory.rename(final_directory)
        latest_path = config.output_root / "latest.json"
        latest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise

    return final_directory


def main() -> int:
    """Execute the pipeline and return a process exit code."""

    try:
        config = load_config()
        frames = extract_frames(config)
        validate_frames(frames)
        snapshot = write_snapshot(config, frames)
    except Exception as exc:  # noqa: BLE001 - CLI boundary must report all failures.
        print(f"sports-data pipeline failed: {exc}", file=sys.stderr)
        return 1

    print(f"sports-data snapshot published: {snapshot}")
    for name, frame in frames.items():
        print(f"- {name}: {len(frame)} rows, {len(frame.columns)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
