import logging
from typing import List
import pandas as pd
import folium
from folium.features import GeoJson, GeoJsonTooltip
from branca.colormap import linear
from .spatial_h3 import hex_feature_collection

logger = logging.getLogger(__name__)

def build_hex_risk_map(hex_df: pd.DataFrame, h3_col: str, value_col: str, out_html: str, map_center: tuple, zoom_start: int = 12, tooltip_fields: List[str] = None) -> None:
    """
    Build an interactive Folium map of H3 hexagons colored by value_col, with tooltips.
    Saves HTML to out_html.
    """
    tooltip_fields = tooltip_fields or [h3_col, value_col]
    logger.info("Creating Folium hex risk map...")

    fc = hex_feature_collection(hex_df, h3_col=h3_col, props={c: c for c in hex_df.columns})
    vals = hex_df[value_col].values
    vmin, vmax = float(vals.min()), float(vals.max() if vals.max() > 0 else 1.0)
    cmap = linear.Reds_09.scale(vmin, vmax)
    cmap.caption = f"{value_col}"

    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="cartodbpositron")
    def style_fn(feature):
        return {
            "fillColor": cmap(feature["properties"][value_col]),
            "color": "#555555",
            "weight": 0.5,
            "fillOpacity": 0.6,
        }
    gj = GeoJson(data=fc, style_function=style_fn, tooltip=GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_fields, sticky=True))
    gj.add_to(m)
    cmap.add_to(m)
    m.save(out_html)
    logger.info(f"Saved Folium map to {out_html}")
