import os
import json
import pickle
from typing import Any, Dict
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl

def ensure_dir(path: str) -> None:
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)

def save_json(d: Dict[str, Any], path: str) -> None:
    """Save dictionary as JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def save_pickle(obj: Any, path: str) -> None:
    """Pickle any Python object."""
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str) -> Any:
    """Load a pickled Python object."""
    with open(path, "rb") as f:
        return pickle.load(f)

def save_df_pd(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame as parquet or CSV based on extension."""
    ensure_dir(os.path.dirname(path))
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df.to_parquet(path, index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported DataFrame extension: {ext}")
    
def save_df(df: pl.DataFrame, path: str) -> None:
    """Save DataFrame as parquet or CSV based on extension."""
    ensure_dir(os.path.dirname(path))
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df.write_parquet(path)
    elif ext == ".csv":
        
        df.write_csv(path)
    else:
        raise ValueError(f"Unsupported DataFrame extension: {ext}")
    
def save_fig(path: str, dpi: int = 150, bbox_inches: str = "tight") -> None:
    """Save the current matplotlib figure and close it."""
    ensure_dir(os.path.dirname(path))
    plt.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

def save_geojson(geojson_obj: Dict[str, Any], path: str) -> None:
    """Save a GeoJSON FeatureCollection to disk."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f)
        
def plot_and_save(fig, out_path):
        """Tight layout + save figure to out_path, then close."""
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)