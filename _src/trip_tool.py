"""Utility functions for computing aggregates on TLC trip data."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time

from crewai.utilities.logger import PrinterColor
import pandas as pd


DEFAULT_TRIP_DATA_PATH = Path("trip_data.parquet")


class TripDataNotFoundError(FileNotFoundError):
    """Raised when the trip data parquet file cannot be located."""


class TripDataEmptyError(ValueError):
    """Raised when the trip data parquet file has no valid duration rows."""


@dataclass(slots=True)
class TripDurationAggregate:
    average_minutes: float
    row_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "average_trip_duration_minutes": self.average_minutes,
            "row_count": self.row_count,
        }

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return json.dumps(self.as_dict())


def _resolve_trip_data_path(path: Optional[Path | str] = None) -> Path:
    if path is None:
        env_path = os.getenv("TRIP_DATA_PATH")
        path = Path(env_path) if env_path else DEFAULT_TRIP_DATA_PATH
    else:
        path = Path(path)

    if not path.exists():
        raise TripDataNotFoundError(f"trip data parquet not found at {path}")
    if not path.is_file():
        raise TripDataNotFoundError(f"trip data path is not a file: {path}")
    return path


def compute_average_trip_duration_minutes(
    path: Optional[Path | str] = None,
) -> TripDurationAggregate:
    """Compute the average trip duration in minutes from the TLC parquet dataset."""
    file_path = _resolve_trip_data_path(path)
    start_time = time.time()

    df = pd.read_parquet(
        file_path,
        columns=["pickup_datetime", "dropOff_datetime"],
    )

    df = df.dropna(subset=["pickup_datetime", "dropOff_datetime"])
    if df.empty:
        raise TripDataEmptyError("no valid pickup/dropoff rows found")

    # Ensure columns are in datetime format before subtraction.
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df["dropOff_datetime"] = pd.to_datetime(
        df["dropOff_datetime"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["pickup_datetime", "dropOff_datetime"])
    if df.empty:
        raise TripDataEmptyError("no valid datetime rows after conversion")

    durations = df["dropOff_datetime"] - df["pickup_datetime"]
    durations_minutes = durations.dt.total_seconds() / 60.0
    durations_minutes = durations_minutes[durations_minutes >= 0]
    if durations_minutes.empty:
        raise TripDataEmptyError("no non-negative duration rows available")

    average = float(durations_minutes.mean())

    # print("Average:", average)

    end_time = time.time()
    execution_time = end_time - start_time
    # print(f"Execution time: {execution_time:.2f} Secs")

    return TripDurationAggregate(
        average_minutes=average, row_count=int(len(durations_minutes))
    )


def format_average_trip_duration(path: Optional[Path | str] = None) -> str:
    aggregate = compute_average_trip_duration_minutes(path)
    return (
        f"Average trip duration across {aggregate.row_count} trips: "
        f"{aggregate.average_minutes:.2f} minutes."
    )


__all__ = [
    "TripDurationAggregate",
    "TripDataNotFoundError",
    "TripDataEmptyError",
    "compute_average_trip_duration_minutes",
    "format_average_trip_duration",
]
