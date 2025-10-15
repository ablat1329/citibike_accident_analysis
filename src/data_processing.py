from typing import Tuple
import polars as pl
import h3
import pandas as pd
from .features import hour_of_week_expr, haversine_km

def clean_bike_trips(
    trips: pl.DataFrame,
    tz: str = "America/New_York",
    h3_res: int = 9
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Clean CitiBike trips using Polars.
    Returns: trips_df, agg_trips_hex_hour, start_hex_total, end_hex_total, stations
    """

    df = trips.clone()
    print(f"Original: {df.shape[0]} rows, {df.shape[1]} columns")

    # Ensure string station IDs
    df = df.with_columns([
        pl.col("start_station_id").cast(pl.Utf8),
        pl.col("end_station_id").cast(pl.Utf8)
    ])

    # Remove missing datetimes
    df = df.filter(pl.col("started_at").is_not_null() & pl.col("ended_at").is_not_null())
    print(f"After dropping null times: {df.shape}")

    # Duration in minutes
    df = df.with_columns(
        ((pl.col("ended_at") - pl.col("started_at")).dt.total_seconds() / 60)
        .alias("duration_minutes")
    ).filter(pl.col("duration_minutes").is_between(2, 720))

    # Valid coordinates
    df = df.filter(
        pl.col("start_lat").is_between(-90, 90)
        & pl.col("start_lng").is_between(-180, 180)
        & pl.col("end_lat").is_between(-90, 90)
        & pl.col("end_lng").is_between(-180, 180)
    )

    # Drop duplicates
    df = df.unique(subset=["ride_id"])

    # ---- Temporal & spatial features ----
    df = df.with_columns([
        pl.col("started_at").dt.date().alias("date"),
        pl.col("started_at").dt.year().alias("year"),
        pl.col("started_at").dt.month().alias("month"),
        pl.col("started_at").dt.day().alias("day"),
        pl.col("started_at").dt.weekday().alias("dow"),
        pl.col("started_at").dt.hour().alias("hour"),
        hour_of_week_expr("started_at").alias("hour_of_week"),
        (pl.col("started_at").dt.weekday().is_in([5, 6]).cast(pl.Int8)).alias("is_weekend"),


        # Spatial H3 hexes
        pl.struct(["start_lat", "start_lng"])
        .map_elements(lambda r: h3.latlng_to_cell(r["start_lat"], r["start_lng"], h3_res),
                      return_dtype=pl.Utf8)
        .alias("h3_start"),

        pl.struct(["end_lat", "end_lng"])
        .map_elements(lambda r: h3.latlng_to_cell(r["end_lat"], r["end_lng"], h3_res),
                      return_dtype=pl.Utf8)
        .alias("h3_end"),
    ])
    
    # add distance_km
    df = df.with_columns([
        pl.struct(["start_lat", "start_lng", "end_lat", "end_lng"])
        .map_elements(
            lambda r: haversine_km(r["start_lat"], r["start_lng"], r["end_lat"], r["end_lng"]),
            return_dtype=pl.Float64
        )
        .alias("distance_km")
    ])

    # add speed_kmh and the rest
    df = df.with_columns([
        ((pl.col("distance_km") / (pl.col("duration_minutes") / 60.0))
        .cast(pl.Float64)
        .replace([float("inf"), float("-inf")], None))
        .alias("speed_kmh")
    ])

    # ---- Aggregations ----
    agg_trips_hex_hour = (
        df.group_by(["h3_start", "hour"])
          .agg(pl.len().alias("trip_count"))
          .drop_nulls("h3_start")
    )

    start_hex_total = (
        df.group_by("h3_start")
          .agg(pl.len().alias("start_trip_count"))
          .drop_nulls("h3_start")
    )

    end_hex_total = (
        df.group_by("h3_end")
          .agg(pl.len().alias("end_trip_count"))
          .drop_nulls("h3_end")
    )

    # ---- Station reference table ----
    start_stations = df.select([
        pl.col("start_station_id").alias("station_id"),
        pl.col("start_station_name").alias("station_name"),
        pl.col("start_lat").alias("lat"),
        pl.col("start_lng").alias("lng")
    ]).drop_nulls().unique()

    end_stations = df.select([
        pl.col("end_station_id").alias("station_id"),
        pl.col("end_station_name").alias("station_name"),
        pl.col("end_lat").alias("lat"),
        pl.col("end_lng").alias("lng")
    ]).drop_nulls().unique()

    stations = pl.concat([start_stations, end_stations]).unique()
    stations = stations.filter(pl.col("lat").is_between(40.48, 40.95) & pl.col("lng").is_between(-74.27, -73.68))
    stations = stations.with_columns(pl.col("station_id").cast(pl.Utf8))
    print(df.columns())
    return df, agg_trips_hex_hour, start_hex_total, end_hex_total, stations


def clean_crash_data(crash: pl.DataFrame, h3_res: int = 9) -> pl.DataFrame:
    df = crash.clone()
    print(f"Original crash data shape: {df.shape}")


    # Parse datetime correctly
    df = df.with_columns([
    (pl.col("CRASH DATE").cast(pl.Utf8) + " " + pl.col("CRASH TIME").cast(pl.Utf8))
    .str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M:%S", strict=False)
    .alias("CRASH_DATETIME")])

    print("After datetime parsing:", df.shape)

    # Remove rows with null datetimes    

    df = df.filter(pl.col("CRASH_DATETIME").is_not_null())
    print("After removing null datetimes:", df.shape)

    # Keep only valid NYC coordinates
    df = df.filter(
        pl.col("LATITUDE").is_between(40, 42)
        & pl.col("LONGITUDE").is_between(-75, -72)
    )
    print("After coordinate filter:", df.shape)

    # Add cyclist columns if missing
    for col in ["NUMBER OF CYCLIST INJURED", "NUMBER OF CYCLIST KILLED", "NUMBER OF PERSONS INJURED", "NUMBER OF PERSONS KILLED"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0).alias(col))

    # Compute cyclist impact
    df = df.with_columns([
        (pl.col("NUMBER OF CYCLIST INJURED").fill_null(0)
         + pl.col("NUMBER OF CYCLIST KILLED").fill_null(0))
        .alias("CYCLIST_IMPACT")
    ])

    print("After adding cyclist impact:", df.shape)

    # Drop duplicates
    if "COLLISION_ID" in df.columns:
        df = df.unique(subset=["COLLISION_ID"])
    print("After deduplication:", df.shape)

    # Add calendar fields
    df = df.with_columns([
        pl.col("CRASH_DATETIME").dt.date().alias("date"),
        pl.col("CRASH_DATETIME").dt.year().alias("year"),
        pl.col("CRASH_DATETIME").dt.month().alias("month"),
        pl.col("CRASH_DATETIME").dt.hour().alias("hour"),
        pl.col("CRASH_DATETIME").dt.weekday().alias("dow"),
        hour_of_week_expr("CRASH_DATETIME").alias("hour_of_week")
    ])
    df = df.with_columns([
        (pl.col("dow").is_in([5, 6])).cast(pl.Int8).alias("is_weekend")
    ])

    # Compute severity metrics
    df = df.with_columns([
        (5 * pl.col("NUMBER OF CYCLIST KILLED").fill_null(0)
         + pl.col("NUMBER OF CYCLIST INJURED").fill_null(0)).alias("severity_score"),
        (5 * pl.col("NUMBER OF PERSONS KILLED").fill_null(0)
         + pl.col("NUMBER OF PERSONS INJURED").fill_null(0)).alias("severity_score_all")
    ])

    df = df.with_columns([
        pl.struct(['LATITUDE', 'LONGITUDE'])
        .map_elements(
            lambda row: (
                h3.latlng_to_cell(row['LATITUDE'], row['LONGITUDE'], h3_res)
                if row['LATITUDE'] is not None and row['LONGITUDE'] is not None
                else None
            ),
            return_dtype=pl.Utf8
        )
        .alias("h3")
    ])

     # Keep only crashes from 2023
    df = df.filter(pl.col("year") == 2023)
    print("After filtering for year 2023:", df.height)
    print("Final:", df.shape)
    return df

def create_accident_label(trips: pd.DataFrame, crashes: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """
    Create an 'accident' label in trips if a crash occurred in the same h3_start and hour.
    """
    
    crashes_lookup = set([(r[0], int(r[1])) for r in trips[['h3','hour']].to_numpy()])
    trips['accident'] = trips.apply(lambda r: 1 if (r['h3_start'], int(r['hour'])) in crashes_lookup else 0, axis=1)
    
    return trips