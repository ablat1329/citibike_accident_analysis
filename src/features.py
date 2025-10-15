import logging
from typing import Union
import numpy as np
import pandas as pd
import polars as pl
import h3
from .spatial_h3 import assign_h3, h3_k_ring

logger = logging.getLogger(__name__)

def hour_of_week(ts: pd.Series) -> pd.Series:
    """
    Compute hour-of-week index: Monday 00:00 = 0, Sunday 23:00 = 167.
    This helps encode cyclical time — 23:00 and 00:00 are neighbors.
    """
    return ts.dt.dayofweek * 24 + ts.dt.hour

def hour_of_week_expr(col: str) -> pl.Expr:
    """
    Return a Polars expression for hour-of-week: Monday 00:00 = 0, Sunday 23:00 = 167.
    """
    return pl.col(col).dt.weekday() * 24 + pl.col(col).dt.hour()

def add_severity(
    crash_df: pl.DataFrame,
    inj_col: str = "NUMBER OF CYCLIST INJURED",
    kill_col: str = "NUMBER OF CYCLIST KILLED",
    kill_weight: int = 5,
) -> pl.DataFrame:
    """
    Compute a severity score for each crash:
    severity_score = injured + kill_weight * killed
    Works efficiently with Polars DataFrames.
    """
    df = crash_df.clone()  # safe lightweight copy (not deep like pandas)

    # Ensure injury & killed columns exist
    for c in [inj_col, kill_col]:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0).alias(c))

    # Fill nulls with 0 and compute severity score
    df = df.with_columns([
        pl.col(inj_col).fill_null(0),
        pl.col(kill_col).fill_null(0),
        (
            pl.col(inj_col).fill_null(0) + kill_weight * pl.col(kill_col).fill_null(0)
        ).alias("severity_score"),
    ])

    return df



def station_day_features_h3(trips: pd.DataFrame, stations: pd.DataFrame, h3_res: int) -> pd.DataFrame:
    """
    Build compact station-day features using H3:
    - attach station_h3 to each station
    - aggregate trip starts per station per day (exposure and behavior)
    - include hour_of_week mode (how_mode) to capture peak timing
    """
    logger.info("Engineering station-day features (H3-based)...")
    st = stations[["station_id","lat","lng"]].dropna().copy()
    st = assign_h3(st, "lat", "lng", "station_h3", res=h3_res)

    df = trips.dropna(subset=["start_station_id","started_at"]).copy()
    df["date"] = df["started_at"].dt.date
    df["hour"] = df["started_at"].dt.hour
    df["dow"] = df["started_at"].dt.dayofweek
    df["hour_of_week"] = hour_of_week(df["started_at"])
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)

    grp = df.groupby(["start_station_id","date"])
    agg = grp.agg(
        trips_start_count=("ride_id","count"),
        dur_start_median=("duration_minutes","median"),
        dur_start_mean=("duration_minutes","mean"),
        member_share_start=("member_casual", lambda s: np.mean(s=="member")),
        rt_electric_share=("rideable_type", lambda s: np.mean(s.astype(str).str.contains("electric", case=False, na=False))),
        morning_share_start=("hour", lambda s: np.mean(s.between(7,10))),
        evening_share_start=("hour", lambda s: np.mean(s.between(16,19))),
        night_share_start=("hour", lambda s: np.mean((s >= 22) | (s <= 5))),
    ).reset_index()

    #most common hour_of_week per group
    how_mode = grp["hour_of_week"].agg(lambda s: s.mode().iloc[0] if len(s) else np.nan).reset_index(name="how_mode")

    features = agg.merge(how_mode, on=["start_station_id","date"], how="left")
    features = features.rename(columns={"start_station_id":"station_id"})
    features["date"] = pd.to_datetime(features["date"])

    features = features.merge(st, on="station_id", how="left")

    for c in ["dur_start_median","dur_start_mean","member_share_start","rt_electric_share","morning_share_start","evening_share_start","night_share_start","how_mode"]:
        if c in features.columns:
            features[c] = features[c].fillna(0.0)
    features["trips_start_count"] = features["trips_start_count"].fillna(0).astype(int)

    logger.info(f"Built station-day features: {features.shape}")
    return features


def crash_hex_daily_counts(crash: pd.DataFrame, h3_res: int) -> pd.DataFrame:
    """
    Aggregate cyclist-involved crash counts by crash_h3 and date.
    """
    logger.info("Aggregating crashes by H3 and date...")
    cr = crash[["LATITUDE","LONGITUDE","date"]].dropna().copy()
    cr["date"] = pd.to_datetime(cr["date"])
    cr = assign_h3(cr, "LATITUDE", "LONGITUDE", "crash_h3", res=h3_res).dropna(subset=["crash_h3"])
    cr_cnt = cr.groupby(["crash_h3","date"]).size().reset_index(name="crash_count")
    logger.info(f"Crash hex-day counts: {cr_cnt.shape}")
    c = crash.copy()
    if "CRASH DATETIME" in c.columns:
        c["date"] = pd.to_datetime(c["CRASH DATETIME"]).dt.date
    elif "date" in c.columns:
        c["date"] = pd.to_datetime(c["date"]).dt.date
    c["h3"] = c.apply(lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], 9) if pd.notnull(r["LATITUDE"]) else None, axis=1)
    sev_agg = c.groupby(["h3","date"])["severity_score"].sum().reset_index().rename(columns={"h3":"crash_h3"})
    # merge into crash_hex_day
    crash_hex_day = cr_cnt.merge(sev_agg, on=["crash_h3","date"], how="left")
    crash_hex_day["severity_score"] = crash_hex_day["severity_score"].fillna(0)

    return crash_hex_day


ArrayLike = Union[float, list[float], np.ndarray]

def haversine_km(
    lat1: ArrayLike, 
    lon1: ArrayLike, 
    lat2: ArrayLike, 
    lon2: ArrayLike
) -> np.ndarray:
    """
    Compute the great-circle distance between two latitude/longitude points using
    the Haversine formula.

    Parameters
    ----------
    lat1 : float | list[float] | np.ndarray
        Latitude(s) of the first point(s) in degrees.
    lon1 : float | list[float] | np.ndarray
        Longitude(s) of the first point(s) in degrees.
    lat2 : float | list[float] | np.ndarray
        Latitude(s) of the second point(s) in degrees.
    lon2 : float | list[float] | np.ndarray
        Longitude(s) of the second point(s) in degrees.

    Returns
    -------
    np.ndarray
        Great-circle distance(s) between the input points, in kilometers.

    Notes
    -----
    - Uses a vectorized implementation for efficiency.
    - Earth radius (R) is set to 6371.0088 km (WGS84 standard).
    """

    # Mean radius of Earth in kilometers
    R = 6371.0088  

    # Convert to numpy arrays for vectorized math
    lat1_arr = np.asarray(lat1, dtype=float)
    lon1_arr = np.asarray(lon1, dtype=float)
    lat2_arr = np.asarray(lat2, dtype=float)
    lon2_arr = np.asarray(lon2, dtype=float)

    # Convert degrees to radians
    phi1 = np.radians(lat1_arr)
    phi2 = np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1_arr)
    dlambda = np.radians(lon2_arr - lon1_arr)

    # Apply Haversine formula
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    distance_km = 2 * R * np.arcsin(np.sqrt(a))

    return distance_km

