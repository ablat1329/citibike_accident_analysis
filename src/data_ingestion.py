import os
import numpy as np
import glob
from typing import List, Tuple
import pandas as pd
from zipfile import ZipFile
import polars as pl


def unzip_citibike_nested(main_zip_path: str, data_dir: str, all_csv_dir: str) -> None:
    """
    Unzip a main zip (containing monthly zips) and extract all CSVs into all_csv_dir.
    Uses system unzip and find via subprocess, as requested.
    """
    print(os.path.abspath(main_zip_path))
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(all_csv_dir, exist_ok=True)
    # Unzip the main zip into data_dir
    #subprocess.run(["unzip", main_zip_path, "-d", data_dir], check=True)
    # Find all nested zips and unzip into all_csv_dir
    #subprocess.run(["find", data_dir, "-name", "*.zip", "-exec", "unzip", "-o", "{}", "-d", all_csv_dir, ";"], check=True)
    #shutil.unpack_archive(main_zip_path, data_dir)
     # 1. Unzip the main zip into data_dir
    with ZipFile(main_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    # 2. Recursively find all nested zip files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".zip"):
                nested_zip_path = os.path.join(root, file)
                
                # 3. Unzip each nested zip into all_csv_dir
                with ZipFile(nested_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(all_csv_dir)

    print("All zips extracted successfully.")

def list_trip_csvs(all_csv_dir: str) -> List[str]:
    """
    List all CitiBike trip CSV files after extraction.
    """
    patterns = [
        os.path.join(all_csv_dir, "*.csv"),
        os.path.join(all_csv_dir, "*trip*.csv"),
        os.path.join(all_csv_dir, "*Trip*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    # De-duplicate preserving order
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique

def load_bike_trips(csv_paths: List[str], parse_dates: Tuple[str, str] = ("started_at", "ended_at")) -> pl.DataFrame:
    """
    Load CitiBike CSVs (or ZIPs containing CSVs) as a Polars DataFrame.
    """
    usecols = [
        "ride_id","rideable_type","started_at","ended_at",
        "start_station_name","start_station_id","end_station_name","end_station_id",
        "start_lat","start_lng","end_lat","end_lng","member_casual",
    ]

    dtype_dict = {
        "ride_id": pl.Utf8,
        "rideable_type": pl.Utf8,
        "start_station_id": pl.Utf8,
        "end_station_id": pl.Utf8,
        "start_station_name": pl.Utf8,
        "end_station_name": pl.Utf8,
        "member_casual": pl.Utf8,
        "start_lat": pl.Float64,
        "start_lng": pl.Float64,
        "end_lat": pl.Float64,
        "end_lng": pl.Float64,
    }

    dfs = []
    for p in csv_paths:
        try:
            if p.endswith(".zip"):
                with ZipFile(p, "r") as z:
                    for name in z.namelist():
                        if name.endswith(".csv"):
                            with z.open(name) as f:
                                df = pl.read_csv(f, columns=usecols, dtypes=dtype_dict, try_parse_dates=True)
                                dfs.append(df)
            else:
                df = pl.read_csv(p, columns=usecols, dtypes=dtype_dict, try_parse_dates=True)
                dfs.append(df)
        except Exception as e:
            print(f"Warning: failed reading {p}: {e}")

    if not dfs:
        raise RuntimeError("No trip CSV files could be read.")

    return pl.concat(dfs, rechunk=True)


def load_crash_csv(crash_file: str) -> pl.DataFrame:
    """
    Load NYPD crash CSV(s) from crash_dir as a Polars DataFrame.
    Supports multiple CSV files in the directory.
    """
    cols_keep = ['CRASH DATE', 'CRASH TIME', 'BOROUGH', 'LATITUDE',
       'LONGITUDE', 'NUMBER OF PERSONS INJURED',
       'NUMBER OF PERSONS KILLED', 'NUMBER OF CYCLIST INJURED',
       'NUMBER OF CYCLIST KILLED', 'NUMBER OF MOTORIST INJURED',
       'NUMBER OF MOTORIST KILLED', 'CONTRIBUTING FACTOR VEHICLE 1',
       'COLLISION_ID', 'VEHICLE TYPE CODE 1', 'VEHICLE TYPE CODE 2',
       'VEHICLE TYPE CODE 3', 'VEHICLE TYPE CODE 4', 'VEHICLE TYPE CODE 5',
       'CRASH_DATETIME', 'CYCLIST_IMPACT']
    try:
        df = pl.read_csv(crash_file, try_parse_dates=True, low_memory=False)
         # Keep only relevant columns
        df = df.select([c for c in cols_keep if c in df.columns])
        return df
        
    except Exception as e:
        print(f"Warning: failed reading crash file {crash_file}: {e}")

def sample_trips_polars(path: str, sample_frac: float = 0.4, seed: int = 42) -> pd.DataFrame:
    """
    Randomly sample a fraction of trips from a Parquet or CSV file using Polars.
    Keeps only a random subset of the data for faster downstream processing.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".parquet":
        df = pl.read_parquet(path)
    elif ext == ".csv":
        df = pl.read_csv(path, try_parse_dates=True, low_memory=False)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    print(f"Loaded {path} with shape {df.shape}")

    # Randomly sample fraction of rows
    df_sampled = df.sample(fraction=sample_frac, shuffle=True, seed=seed)

    print(f"Sampled {sample_frac*100:.0f}% of data → shape {df_sampled.shape}")
    return df_sampled.to_pandas()

def load_clean_data(path: str) -> pd.DataFrame:
    """
    Load a cleaned Parquet file (trips, stations, crashes).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df_polars = pl.read_parquet(path)
        df_pandas = df_polars.to_pandas()
        print(f"Loaded cleaned data {path} with shape {df_pandas.shape} and top 2 rows:\n{df_pandas.head(2)}")
        return df_pandas
    elif ext == ".csv":
        df_polars= pl.read_csv(path, try_parse_dates=True, low_memory=False)
        df_pandas = df_polars.to_pandas()
        print(f"Loaded cleaned data {path} with shape {df_pandas.shape} and top 2 rows:\n{df_pandas.head(2)}")

        return df_pandas
    else:
        raise ValueError(f"Unsupported cleaned data file extension: {ext}")
    
    
def load_clean_trips_aggr(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a cleaned Parquet file (trips).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df_polars = pl.read_parquet(path)
        #df_polars = pl.read_parquet(path, n_rows=1_800_000) #n_rows=1_800_000
        df_pandas = df_polars.to_pandas()
        print(f"Loaded cleaned data {path} with shape {df_pandas.shape} and top 2 rows:\n{df_pandas.head(2)}")
       
    elif ext == ".csv":
        df_polars= pl.read_csv(path, try_parse_dates=True, low_memory=False)
        #df_pandas = df_polars.to_pandas()
        print(f"Loaded cleaned data {path} with shape {df_pandas.shape} and top 2 rows:\n{df_pandas.head(2)}")
    else:
        raise ValueError(f"Unsupported cleaned data file extension: {ext}")
    print(f"columns of the dataframe are {df_polars.columns}")
    # ---- Aggregations ----
    agg_trips_hex_hour = (
        df_polars.group_by(["h3_start", "hour"])
          .agg(pl.len().alias("trip_count"))
          .drop_nulls("h3_start")
    )

    start_hex_total = (
        df_polars.group_by("h3_start")
          .agg(pl.len().alias("start_trip_count"))
          .drop_nulls("h3_start")
    )

    end_hex_total = (
        df_polars.group_by("h3_end")
          .agg(pl.len().alias("end_trip_count"))
          .drop_nulls("h3_end")
    )

    df_pandas = df_polars.to_pandas()
    agg_trips_hex_hour_df = agg_trips_hex_hour.to_pandas()
    start_hex_total_df = start_hex_total.to_pandas()
    end_hex_total_df = end_hex_total.to_pandas()   
    return df_pandas, agg_trips_hex_hour_df, start_hex_total_df, end_hex_total_df
    
def load_clean_crash_aggr(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load a cleaned Parquet file (crashes).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df_polars = pl.read_parquet(path, n_rows=1_800_000)
        df_pandas = df_polars.to_pandas()
        print(f"Loaded cleaned data {path} with shape {df_pandas.shape} and top 2 rows:\n{df_pandas.head(2)}")
       
    elif ext == ".csv":
        df_polars= pl.read_csv(path, try_parse_dates=True, low_memory=False)
        #df_pandas = df_polars.to_pandas()
        print(f"Loaded cleaned data {path} with shape {df_pandas.shape} and top 2 rows:\n{df_pandas.head(2)}")
    else:
        raise ValueError(f"Unsupported cleaned data file extension: {ext}")
    print(f"columns of the dataframe are {df_polars.columns}")
    # ---- Aggregations ----
    agg_crash_hex_hour = (
        df_polars.group_by(["h3", "hour"])
        .agg([
            pl.len().alias("crashes"),
            pl.sum("severity_score").alias("severity_sum")
        ])
        .drop_nulls("h3")
    )

    df_pandas = df_polars.to_pandas()
    agg_crash_her_hour_pandas = agg_crash_hex_hour.to_pandas()

    return df_pandas, agg_crash_her_hour_pandas

def stratified_sample(df: pd.DataFrame, group_col: str, frac: float = 0.6, seed: int = 42) -> pd.DataFrame:
    """
    Memory-efficient stratified sampling per group (e.g., per month).
    """
    rng = np.random.default_rng(seed)
    samples = []
    for key, group in df.groupby(group_col, sort=False):
        n = int(len(group) * frac)
        if n == 0:
            continue
        sampled_idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        samples.append(df.loc[sampled_idx])
    return pd.concat(samples, ignore_index=True)