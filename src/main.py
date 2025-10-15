import os
import logging
from src.config import ProjectConfig
from src.utils_io import ensure_dir, save_df
from src.data_ingestion import unzip_citibike_nested, list_trip_csvs, load_bike_trips, load_crash_csv, load_clean_data, sample_trips_polars
from src.data_processing import clean_bike_trips, clean_crash_data, create_accident_label
from src.features import crash_hex_daily_counts
from src.visualization import build_hex_level_crash_risk_map
from src.modeling import trip_level_accidient_classifier,predict_crash_and_severity
from src.feature_selection import feature_selection_stats
from src.edv_analysis import generate_eda_plots, analyze_associations

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def main():
    setup_logging()
    logger = logging.getLogger("main")
    cfg = ProjectConfig()
    saving_suffix = "parquet"  # "csv" or "parquet"
    logger.info("Starting CitiBike + NYPD pipeline (H3 edition)...")

    # Prepare output directories
    for sub in ["data_clean","features","models","plots","metrics","predictions","geo"]:
        ensure_dir(os.path.join(cfg.out_dir, sub))
    
    # 1) Ingest: unzip and load
    unzip_citibike_nested(cfg.main_zip_path, cfg.data_dir, cfg.all_csv_dir)
    trip_csvs = list_trip_csvs(cfg.all_csv_dir)
    logger.info(f"Found {len(trip_csvs)} trip CSVs.")
    trips_raw = load_bike_trips(trip_csvs)
    logger.info("Clean and feature engineering the bike trips...")
    trips_clean, agg_trips_hex_hour, start_hex_total, end_hex_total, stations = clean_bike_trips(trips_raw, tz="America/New_York",h3_res=9)
    
    save_df(trips_clean, f"{cfg.out_dir}/data_clean/trips_clean.{saving_suffix}")
    save_df(stations, f"{cfg.out_dir}/data_clean/stations.{saving_suffix}")
    
    logger.info("Subsample trips for EDA and modeling...")
    trips_clean_pd = sample_trips_polars(f"{cfg.out_dir}/data_clean/trips_clean2.{saving_suffix}")
    
    #read and clean crash data
    logger.info("Load, clean and feature engineering crash data...")
    crash_raw = load_crash_csv(cfg.crash_file)
    crash_clean_pl = clean_crash_data(crash_raw, h3_res=9)
    save_df(crash_clean_pl, f"{cfg.out_dir}/data_clean/crash_clean2.{saving_suffix}")
    
    #crash_clean, agg_crash_hex_hour = load_clean_crash_aggr(f"{cfg.out_dir}/data_clean/crash_clean2.{saving_suffix}")
    crash_clean = load_clean_data(f"{cfg.out_dir}/data_clean/crash_clean2.{saving_suffix}")
    

    # 2) EDA and visualization
    logger.info("Exploratory data analysis and visualization...")
    agg_results = generate_eda_plots(trips_clean_pd, crash_clean, os.path.join(cfg.out_dir, "plots/"))
    hex_hour = analyze_associations(trips_clean_pd, crash_clean, agg_results, f"{cfg.out_dir}/plots/")
    crash_hex_day = crash_hex_daily_counts(crash_clean, h3_res=cfg.h3_res)
    build_hex_level_crash_risk_map(crash_hex_day, f"{cfg.out_dir}/plots/",cfg.map_center_lat,cfg.map_center_lng,cfg.map_zoom_start)
    
   
    # 3) Feature selection for trip-level accident prediction
    logger.info("Feature selection analysis...")
    trip_accident_label_df = create_accident_label(trips_clean_pd, crash_clean, f"{cfg.out_dir}/data_clean/")
    kruskal_df, chi2_df = feature_selection_stats(trip_accident_label_df, os.path.join(cfg.out_dir, "features/"))
    logger.info("Trip level crash classification ...")
    trip_level_accidient_classifier(trips_clean_pd, crash_clean, cfg.significant_features, cfg.grid_search_params, cfg.test_size, f"{cfg.out_dir}/models2/")
   
    # 4) Severity prediction
    logger.info("Severity prediction ...")
    predict_crash_and_severity(hex_hour, cfg.kmeans_params, cfg.test_size, cfg.random_state, False, cfg.grid_search_params, cfg.base_price, cfg.risk_multiplier, cfg.num_price_bins, f"{cfg.out_dir}/models2/")

    logger.info("Pipeline complete.")
    logger.info(f"Saved outputs to: {cfg.out_dir}")

if __name__ == "__main__":
    main()
