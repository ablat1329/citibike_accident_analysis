import logging
from typing import Iterable, List, Dict
import pandas as pd
import h3

logger = logging.getLogger(__name__)

def assign_h3(df: pd.DataFrame, lat_col: str, lng_col: str, h3_col: str, res: int) -> pd.DataFrame:
    """
    Assign H3 index to each row at resolution res using df[lat_col], df[lng_col].
    """
    logger.info(f"Assigning H3 (res={res}) to {len(df):,} rows...")
    out = df.copy()
    out[h3_col] = [
        h3.latlng_to_cell(lat, lng, res) if pd.notnull(lat) and pd.notnull(lng) else None
        for lat, lng in zip(out[lat_col].values, out[lng_col].values)
    ]
    logger.info(f"H3 assigned to {out[h3_col].notna().sum():,} rows.")
    return out

def h3_k_ring(h: str, k: int) -> List[str]:
    """
    Return neighbors within k rings of hex cell h.
    """
    if h is None or pd.isna(h):
        return []
    return list(h3.grid_disk(h, k))

def h3_polygon_geojson(h: str) -> List[List[List[float]]]:
    """
    Convert an H3 cell index to a GeoJSON polygon ring (lon, lat).
    """
    boundary = h3.cell_to_boundary(h)  # list of (lat, lon)
    ring = [[pt[1], pt[0]] for pt in boundary]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return [ring]

def hex_feature_collection(df: pd.DataFrame, h3_col: str, props: Dict[str, str]) -> Dict:
    """
    Build GeoJSON FeatureCollection of H3 cells with attached properties.
    props: mapping property_name -> df column to include in feature properties.
    """
    logger.info("Building GeoJSON FeatureCollection for hex grid...")
    features = []
    for _, row in df.iterrows():
        h = row[h3_col]
        if pd.isna(h):
            continue
        geometry = {"type": "Polygon", "coordinates": h3_polygon_geojson(h)}
        properties = {k: row[v] for k, v in props.items()}
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})
    return {"type": "FeatureCollection", "features": features}
