import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils_io import save_df_pd, save_geojson
from src.spatial_h3 import hex_feature_collection
from src.visualization_map import build_hex_risk_map
sns.set(style="whitegrid")



def plot_and_save(fig, out_path):
    """Tight layout + save figure to out_path, then close."""
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def build_hex_level_crash_risk_map(crash_hex_day: pd.DataFrame, out_dir: str,map_center_lat: float, map_center_lng: float, map_zoom_start: float) -> None:
    """
    Build and save a hex-level average daily crash risk map (interactive HTML + GeoJSON).

    Parameters
    ----------
    crash_hex_day : pd.DataFrame
        DataFrame containing daily crash counts aggregated by H3 hexagon ID.
        Expected columns: ['crash_h3', 'date', 'crash_count']

    -------
    None
        Saves Parquet/CSV, GeoJSON, and interactive HTML map to disk.
    """
    if crash_hex_day.empty:
        print("No crash hex data available — skipping risk map generation.")
        return

    # Compute total days span
    days_span = (crash_hex_day["date"].max() - crash_hex_day["date"].min()).days + 1

    # Aggregate total crashes per hexagon
    hex_agg = (
        crash_hex_day.groupby("crash_h3", as_index=False)["crash_count"]
        .sum()
        .rename(columns={"crash_h3": "hex_id"})
    )

    # Compute average daily crash rate
    hex_agg["avg_daily_crash"] = hex_agg["crash_count"] / max(days_span, 1)

    # --- Save tabular data ---
    save_path = f"{out_dir}/geo/hex_risk_agg.parquet"
    save_df_pd(hex_agg, save_path)

    # --- Save GeoJSON ---
    fc = hex_feature_collection(
        hex_agg,
        h3_col="hex_id",
        props={
            "hex_id": "hex_id",
            "crash_count": "crash_count",
            "avg_daily_crash": "avg_daily_crash",
        },
    )
    geojson_path = f"{out_dir}/geo/hex_risk_agg.geojson"
    save_geojson(fc, geojson_path)

    # --- Build interactive HTML map ---
    html_path = f"{out_dir}/plots/hex_risk_map.html"
    build_hex_risk_map(
        hex_df=hex_agg,
        h3_col="hex_id",
        value_col="avg_daily_crash",
        out_html=html_path,
        map_center=(map_center_lat, map_center_lng),
        zoom_start=map_zoom_start,
        tooltip_fields=["hex_id", "avg_daily_crash", "crash_count"],
    )

    print(f" Hex risk map successfully generated:\n- Data: {save_path}\n- GeoJSON: {geojson_path}\n- Map: {html_path}")