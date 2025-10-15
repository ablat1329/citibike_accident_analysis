from typing import Dict
import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.cluster import KMeans
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

import h3


def train_trip_classifier(trips_df: pd.DataFrame, significant_features: list, param_grid: Dict,  test_size: float, out_dir: str, random_state=42):
    # trips_df: pandas DataFrame with per-trip features and a binary 'accident' column (1 if trip matched crash)
    df = trips_df.dropna().copy()
    # choose features
    #feat_cols = ['duration_minutes','distance_km','speed_kmh','hour','dow','is_weekend']
    
    feat_cols = significant_features
    X = df[feat_cols].fillna(0)
    y = df['accident'].astype(int)
    if y.nunique() < 2:
        return None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    xgb_classifier_model = xgb.XGBClassifier()
    grid_search = GridSearchCV(xgb_classifier_model, param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    param_set_log = os.path.join(out_dir, "models", "xgb_prediction_best_paramset.txt")
    with open(param_set_log, "w") as f:
        f.write(f"Best parameters: {grid_search.best_params_}")
        
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(out_dir, "models", "xgb_model.joblib"))
    best_model.fit(X_train, y_train)
    pred = best_model.predict_proba(X_test)[:,1]
    precision_score_v = precision_score(y_test, pred, average='weighted')
    recall_score_val = recall_score(y_test, pred, average='weighted')
    f1 = f1_score(y_test, pred, average='weighted')
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(out_dir, "models", "xgb_trip_classifier.joblib"))
    metrics_path = os.path.join(out_dir, "models", "xgb_trip_classifier_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"precision: {precision_score_v:.4f}\n")
        f.write(f"Recall: {recall_score_val:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")

    return best_model, {'precision': float(precision_score), 'recall_score_val': float(recall_score_val), 'F1-score': float{f1}}


def trip_level_accidient_classifier(trips: pd.DataFrame, crashes: pd.DataFrame, significant_features: list, param_grid: Dict, test_size: float, out_dir: str):
    trips_pd = trips[significant_features]
    #trips_pd = trips[['duration_minutes','distance_km','speed_kmh','hour','dow','is_weekend','h3_start']]
    # mark accident if same h3 and hour has a crash (simplistic)
    
    crashes_lookup = set([(r[0], int(r[1])) for r in crashes[['h3','hour']].to_numpy()])
    trips_pd['accident'] = trips_pd.apply(lambda r: 1 if (r['h3'], int(r['hour'])) in crashes_lookup else 0, axis=1)
    #trips_pd.to_csv(os.path.join(out_dir, "features", "trips_with_accident_label.csv"), index=False)
    
    trip_model, trip_metrics = train_trip_classifier(trips_pd, significant_features, param_grid, test_size, str(out_dir))

    print('Trip classifier metrics:', trip_metrics)


def predict_crash_and_severity(
    df: pd.DataFrame,
    kmeans_params: dict = None,
    test_size: float = 0.1,
    random_state: int = 42,
    log_transform: bool = False,
    param_grid: Dict = None,
    base_price: float = 5.0,
    risk_multiplier: float = 0.25,
    num_price_bins: int = 10,
    out_dir: str = "model_outputs",
):
    """
    Train XGBoost models to predict both crash rate and severity rate per 10k trips.
    Produces evaluation metrics, test predictions, and pricing tables for each target.
    """

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n Starting dual-model prediction | log_transform={log_transform}")

    # ======================================================
    # 1. CLEANUP
    # ======================================================
    
    df = df.dropna(subset=[
        'trip_count', 'avg_speed', 'avg_distance_km',
        'avg_duration_minutes', 'weekend_share', 'casual_share',
        'crash_rate_per_10k', 'severity_per_10k'
    ])
  
    df = df[df['trip_count'] > 0].copy()

    # ======================================================
    # 2. SPATIAL CLUSTERING
    # ======================================================
    print("Encoding spatial clusters from H3 indices...")
    df['lat'], df['lng'] = zip(*df['h3'].map(lambda x: h3.cell_to_latlng(x)))
    coords = df[['lat', 'lng']].values

    kmeans = KMeans(**kmeans_params)
    df['spatial_cluster'] = kmeans.fit_predict(coords)

    # ======================================================
    # 3. COMMON FEATURES
    # ======================================================
    
    feature_cols = [
        'hour', 'trip_count', 'avg_speed', 'avg_distance_km',
        'avg_duration_minutes', 'weekend_share', 'casual_share', 'spatial_cluster'
    ]
    

    results = {}

    for target_col in ['crash_rate_per_10k', 'severity_per_10k']:
        print(f"\n Training model for: {target_col}")

        data = df.copy()

        # Transform target if requested
        if log_transform:
            data['target'] = np.log1p(data[target_col])
        else:
            data['target'] = data[target_col]

        # ======================================================
        # 4. TRAIN/TEST SPLIT
        # ======================================================
        train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
        X_train = train_df[feature_cols]
        y_train = train_df['target']
        X_test = test_df[feature_cols]
        y_test = test_df['target']
        # ======================================================
        # 5. TRAIN XGBOOST
        # ======================================================
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(xgb_model, param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        param_set_log = os.path.join(out_dir, "models", target_col + "xgb_prediction_best_paramset.txt")
        with open(param_set_log, "w") as f:
            f.write(f"Best parameters: {grid_search.best_params_}")
            
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(best_model, os.path.join(out_dir, "models",  target_col + "xgb_model.joblib"))
        # Predict
        preds = best_model.predict(X_test)
        
        # Inverse transform if needed
        if log_transform:
            preds = np.expm1(preds)
        preds = np.maximum(preds, 0)

        # ======================================================
        # 6. EVALUATE
        # ======================================================
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"   MAE: {mae:.4f}")
        print(f"   R²:  {r2:.4f}")
        param_set_log = os.path.join(out_dir, "models", target_col + "xgb_prediction_best_paramset.txt")
        with open(param_set_log, "w") as f:
            f.write(f"Best parameters: {grid_search.best_params_}")
            f.write(f"\nMAE: {mae:.4f}\nR²: {r2:.4f}\n")

        # ======================================================
        # 7. SAVE TEST PREDICTIONS
        # ======================================================
        test_df[f'pred_{target_col}'] = preds
        test_csv = os.path.join(out_dir, "predictions",  f"pred_{target_col}_{'log' if log_transform else 'raw'}.csv")
        test_df.to_csv(test_csv, index=False)
        print(f"   Predictions saved → {test_csv}")

        # ======================================================
        # 8. PRICING TABLE
        # ======================================================
        def minmax_scale(series):
            return (series - series.min()) / (series.max() - series.min() + 1e-9)

        test_df['risk_index'] = minmax_scale(test_df[f'pred_{target_col}'])
        test_df['risk_decile'] = pd.qcut(test_df['risk_index'], num_price_bins, labels=False, duplicates='drop')
        test_df['price'] = base_price * (1 + test_df['risk_decile'] * risk_multiplier)

        pricing_table = test_df.groupby('spatial_cluster').agg(
            avg_pred_rate=(f'pred_{target_col}', 'mean'),
            avg_risk_index=('risk_index', 'mean'),
            avg_price=('price', 'mean'),
            total_hexes=('h3', 'nunique')
        ).reset_index()

        out_pricing = os.path.join(out_dir, "predictions", f"pricing_table_{target_col}_{'log' if log_transform else 'raw'}.csv")
        pricing_table.to_csv(out_pricing, index=False)
        print(f"   Pricing table saved → {out_pricing}")

        # Store results
        results[target_col] = {
            'model': best_model,
            'test_df': test_df,
            'pricing_table': pricing_table,
            'mae': mae,
            'r2': r2
        }

    print("Both models trained successfully!")

    return results

