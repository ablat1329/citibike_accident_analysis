import os
from dataclasses import dataclass

@dataclass
class ProjectConfig:
    """
    Central configuration for paths, modeling, spatial, and map parameters.
    Values can be overridden via environment variables.
    """
    # IO
    data_dir: str = os.environ.get("DATA_DIR", "data")
    out_dir: str = os.environ.get("OUT_DIR", "outputs")
    main_zip_path: str = os.environ.get("MAIN_ZIP_PATH", os.path.join("data", "2023-citibike-tripdata.zip"))
    all_csv_dir: str = os.environ.get("ALL_CSV_DIR", os.path.join("data", "all"))
    crash_dir: str = os.environ.get("CRASH_DIR", os.path.join("data", "crash"))
    crash_file: str = os.environ.get("crash_file", os.path.join("data","crash", "Motor_Vehicle_Collisions_-_Crashes_20251008.csv"))
    # Temporal granularity flag (for future hourly extension)
    granularity: str = os.environ.get("GRANULARITY", "daily")

    # H3 spatial params
    h3_res: int = int(os.environ.get("H3_RES", "9"))      # r9: ~170m edges
    h3_k_ring: int = int(os.environ.get("H3_K_RING", "1"))  # 1 -> ~300m neighborhood at r9

    # significant features
    significant_features_str = os.environ.get(
    "SIGNIFICANT_FEATURES","hour,speed_kmh,distance_km,dow,duration_minutes,hour_of_week,is_weekend,month"
)
# ['duration_minutes','distance_km','speed_kmh','hour','dow','is_weekend']
#"start_lng,end_lng,hour,speed_kmh,end_lat,start_lat,distance_km,dow,duration_minutes,hour_of_week,is_weekend,month,day,year,member_casual,h3"
    significant_features = [f.strip() for f in significant_features_str.split(",") if f.strip()]
    # Modeling
    test_months: int = int(os.environ.get("TEST_MONTHS", "2"))  # last N months as test
    test_size: float = float(os.environ.get("TEST_SIZE", "0.2"))  # test size for random split
    random_state: int = int(os.environ.get("RANDOM_STATE", "42"))
    n_jobs: int = int(os.environ.get("N_JOBS", "4"))
    base_premium_cents: float = float(os.environ.get("base_premium_cents", "0.2"))
    base_price: float = float(os.environ.get("base_price", "0.5"))
    risk_multiplier = float(os.environ.get("risk_multiplier", "0.25"))
    location_clusters = int(os.environ.get("location_clusters", "100"))
    num_price_bins = int(os.environ.get("num_price_bins", "10")) #used to calculate deciles for pricing

    # XGBoost defaults (tune as needed)
    xgb_regression_params: dict = None
    xgb_classification_params: dict = None
    grid_search_params: dict = None
    # kmeans clustering for h3_index based clusters
    kmeans_params: dict = None

    # Map settings
    map_center_lat: float = float(os.environ.get("MAP_CENTER_LAT", "40.75"))
    map_center_lng: float = float(os.environ.get("MAP_CENTER_LNG", "-73.98"))
    map_zoom_start: int = int(os.environ.get("MAP_ZOOM_START", "12"))

    def __post_init__(self):
        if self.xgb_regression_params is None:
            self.xgb_regression_params = {
                "n_estimators": 600,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
                "min_child_weight": 1.0,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            }

        if self.xgb_classification_params is None:
            self.xgb_classification_params = {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1.0,
                "objective": "binary:logistic",
                "tree_method": "hist",
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            }

        if self.kmeans_params is None:
            self.kmeans_params = {
                "n_clusters": self.location_clusters,
                "random_state": self.random_state,
                "n_init": 10,
                "max_iter": 5
            }
        
        if self.grid_search_params is None:
            self.grid_search_params = {
                'n_estimators':[100,200,400, 600, 800, 1000], 
                'max_depth': [2, 4, 6, 8],
                'learning_rate': [0.005, 0.01, 0.05, 0.1],
                'subsample': [0.8]#[0.8, 1.0] 
            }

