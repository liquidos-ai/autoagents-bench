use chrono::{DateTime, FixedOffset, NaiveDateTime};
use polars::prelude::*;
use polars::prelude::{ChunkAgg, ChunkCompareIneq, NamedFrom};
use serde::Serialize;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tokio::time::Instant;

#[derive(Clone, Debug, Serialize)]
pub struct TripDurationAggregate {
    pub average_trip_duration_minutes: f64,
    pub row_count: usize,
}

#[derive(Debug, Error)]
pub enum TripAggregationError {
    #[error("trip data parquet not found at {0}")]
    NotFound(PathBuf),
    #[error("trip data parquet did not contain any valid pickup/dropoff timestamps")]
    Empty,
    #[error("I/O error while reading trip data: {0}")]
    Io(#[from] io::Error),
    #[error("failed to read trip data: {0}")]
    Read(#[from] PolarsError),
}

/// Computes average trip duration in minutes from a Parquet file.
pub fn compute_average_trip_duration(
    path: &Path,
) -> Result<TripDurationAggregate, TripAggregationError> {
    let _start = Instant::now();
    if !path.exists() {
        return Err(TripAggregationError::NotFound(path.to_path_buf()));
    }

    let path_str = path
        .to_str()
        .ok_or_else(|| TripAggregationError::NotFound(path.to_path_buf()))?;

    let df = LazyFrame::scan_parquet(PlPath::new(path_str), ScanArgsParquet::default())?
        .select([col("pickup_datetime"), col("dropOff_datetime")])
        .filter(
            col("pickup_datetime")
                .is_not_null()
                .and(col("dropOff_datetime").is_not_null()),
        )
        .collect()?;

    if df.height() == 0 {
        return Err(TripAggregationError::Empty);
    }

    let pickup = df
        .column("pickup_datetime")?
        .as_materialized_series()
        .clone();
    let dropoff = df
        .column("dropOff_datetime")?
        .as_materialized_series()
        .clone();

    let pickup_ms = series_to_millis(&pickup)?;
    let dropoff_ms = series_to_millis(&dropoff)?;

    let durations_ms_series = (&dropoff_ms - &pickup_ms)?;
    let mask = durations_ms_series.gt_eq(0)?;
    let durations_ms_filtered_series = durations_ms_series.filter(&mask)?;
    let durations_ms_filtered = durations_ms_filtered_series
        .i64()
        .map_err(|_| TripAggregationError::Empty)?
        .drop_nulls();

    if durations_ms_filtered.is_empty() {
        return Err(TripAggregationError::Empty);
    }

    let row_count = durations_ms_filtered.len();
    let average_minutes = durations_ms_filtered
        .mean()
        .map(|avg_ms| avg_ms / 60_000.0)
        .ok_or(TripAggregationError::Empty)?;

    // println!("Time Taken {}", _start.elapsed().as_millis());

    Ok(TripDurationAggregate {
        average_trip_duration_minutes: average_minutes,
        row_count,
    })
}

fn series_to_millis(series: &Series) -> Result<Series, TripAggregationError> {
    match series.dtype() {
        DataType::Datetime(time_unit, tz) => {
            let datetime_ms = if *time_unit == TimeUnit::Milliseconds {
                series.clone()
            } else {
                series
                    .cast(&DataType::Datetime(TimeUnit::Milliseconds, tz.clone()))
                    .map_err(|_| TripAggregationError::Empty)?
            };

            datetime_ms
                .cast(&DataType::Int64)
                .map_err(|_| TripAggregationError::Empty)
        }
        DataType::Int64 => Ok(series.clone()),
        DataType::String => {
            let values = series
                .str()
                .map_err(|_| TripAggregationError::Empty)?
                .into_iter()
                .map(|opt| opt.and_then(parse_datetime_to_millis))
                .collect::<Vec<_>>();

            Ok(Series::new(series.name().clone(), values))
        }
        _ => Err(TripAggregationError::Empty),
    }
}

fn parse_datetime_to_millis(value: &str) -> Option<i64> {
    const FORMATS: &[&str] = &[
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
    ];

    for format in FORMATS {
        if let Ok(dt) = NaiveDateTime::parse_from_str(value, format) {
            return Some(dt.and_utc().timestamp_millis());
        }
    }

    value
        .parse::<DateTime<FixedOffset>>()
        .ok()
        .map(|dt| dt.timestamp_millis())
}
