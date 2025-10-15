import pandas as pd
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from .utils_io import plot_and_save

sns.set(style="whitegrid")


def generate_eda_plots(trips: pd.DataFrame, crashes: pd.DataFrame, out_dir: str):
    """Generate time-based EDA plots (trip and crash patterns) and return aggregated data."""
    os.makedirs(out_dir, exist_ok=True)
    results = {}
       # === Sanitize datatypes ===
    for col in ['hour', 'month', 'day', 'dow']:
        if col in trips.columns:
            trips[col] = pd.to_numeric(trips[col], errors='coerce').astype('Int8')
    if 'speed_kmh' in trips.columns:
        trips['speed_kmh'] = pd.to_numeric(trips['speed_kmh'], errors='coerce')
    trips = trips[(trips['speed_kmh'] >= 3) & (trips['speed_kmh'] <= 45)] 
    
    # === 1. Time-based aggregations ===
    trip_hourly = trips.groupby('hour')['ride_id'].count().reset_index(name='trip_count')
    trip_daily = trips.groupby('dow')['ride_id'].count().reset_index(name='trip_count')
    trip_monthly = trips.groupby('month')['ride_id'].count().reset_index(name='trip_count')

    crash_hourly = crashes.groupby('hour')['COLLISION_ID'].count().reset_index(name='crash_count')
    crash_daily = crashes.groupby('dow')['COLLISION_ID'].count().reset_index(name='crash_count')
    crash_monthly = crashes.groupby('month')['COLLISION_ID'].count().reset_index(name='crash_count')

    # merge to compute crash_rate per period
    hourly = pd.merge(trip_hourly, crash_hourly, on='hour', how='left').fillna(0)
    hourly['crash_rate_per_10k'] = (hourly['crash_count'] / hourly['trip_count']) * 10000

    daily = pd.merge(trip_daily, crash_daily, on='dow', how='left').fillna(0)
    daily['crash_rate_per_10k'] = (daily['crash_count'] / daily['trip_count']) * 10000

    monthly = pd.merge(trip_monthly, crash_monthly, on='month', how='left').fillna(0)
    monthly['crash_rate_per_10k'] = (monthly['crash_count'] / monthly['trip_count']) * 10000

    results.update({
        'hourly': hourly, 'daily': daily, 'monthly': monthly
    })

    # === 2. Crash rate plots ===
    for name, df, x in [('hour', hourly, 'hour'), ('day', daily, 'dow'), ('month', monthly, 'month')]:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(data=df, x=x, y='crash_rate_per_10k', marker='o', ax=ax)
        ax.set_title(f"Crash rate vs {name}")
        ax.set_ylabel("Crashes per 10k trips")
        plot_and_save(fig, os.path.join(out_dir, f"crash_rate_vs_{name}.png"))

    # === 3. Trip volume plots ===
    for name, df, x in [('hour', hourly, 'hour'), ('day', daily, 'dow'), ('month', monthly, 'month')]:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(data=df, x=x, y='trip_count', color='steelblue', ax=ax)
        ax.set_title(f"Trip count vs {name}")
        ax.set_ylabel("Trips")
        plot_and_save(fig, os.path.join(out_dir, f"trip_count_vs_{name}.png"))

    trips_sample = trips.sample(n=min(2_800_000, len(trips)), random_state=42)
    # === 4. Speed analysis ===
    if 'member_casual' in trips.columns:
        fig, ax = plt.subplots(figsize=(8,6))
        sns.boxplot(data=trips_sample, x='member_casual', y='speed_kmh', ax=ax)
        ax.set_title("Speed by Rider Type")
        plot_and_save(fig, os.path.join(out_dir, "speed_by_ridertype.png"))

    fig, ax = plt.subplots(figsize=(10,6))
    
    sns.violinplot(data=trips_sample, x='hour', y='speed_kmh', inner='quartile', cut=0, ax=ax)
    ax.set_title("Speed distribution by hour")
    plot_and_save(fig, os.path.join(out_dir, "speed_by_hour.png"))
    
    # === 5. Precompute H3 x hour exposure for later use ===
    if 'h3_start' in trips.columns:
        trips_hex_hour = trips.groupby(['h3_start', 'hour']).size().reset_index(name='trip_count')
        results['trips_hex_hour'] = trips_hex_hour
        trips_hex_hour.to_csv(os.path.join(out_dir, "eda_aggregated_hex_hour.csv"), index=False)

    return results

def analyze_associations(trips: pd.DataFrame, crashes: pd.DataFrame, agg_results: dict, out_dir: str):
    """Associate bike trip features with crash rates and visualize relationships."""
    os.makedirs(out_dir, exist_ok=True)

    trips_hex_hour = agg_results.get('trips_hex_hour')
    if trips_hex_hour is None:
        raise ValueError("Missing trips_hex_hour in precomputed results from generate_eda_plots()")

    # === Merge crash counts into trip exposure ===
    crash_hex_hour = crashes.groupby(['h3', 'hour']).agg(
        crashes=('COLLISION_ID', 'count'),
        severity=('severity_score', 'sum')
    ).reset_index()

    trips_hex_hour['h3'] = trips_hex_hour['h3_start']
    hex_hour = trips_hex_hour.merge(crash_hex_hour, on=['h3', 'hour'], how='left').fillna({'crashes':0, 'severity':0})
    hex_hour = hex_hour[hex_hour['trip_count'] > 0].copy()
    hex_hour['crash_rate_per_10k'] = (hex_hour['crashes'] / hex_hour['trip_count']) * 10000
    hex_hour['severity_per_10k'] = (hex_hour['severity'] / hex_hour['trip_count']) * 10000

    # === Aggregate average bike features ===
    trips['h3'] = trips['h3_start']
      # === Aggregate average bike features efficiently ===
    group_cols = ['h3', 'hour']
    add_aggs = {
        'speed_kmh': 'mean',
        'distance_km': 'mean',
        'duration_minutes': 'mean',
        'is_weekend': 'mean'
    }

    # Downcast numeric types to save memory
    trips = trips.astype({
        'speed_kmh': 'float32',
        'distance_km': 'float32',
        'duration_minutes': 'float32'
    })

    # Compute means without unnecessary copies
    add_feats = (
        trips.groupby(group_cols, observed=True)
        .agg(add_aggs)
        .rename(columns={
            'speed_kmh': 'avg_speed',
            'distance_km': 'avg_distance_km',
            'duration_minutes': 'avg_duration_minutes',
            'is_weekend': 'weekend_share'
        })
        .reset_index()
    )

    # Compute share of casual riders safely
    if 'member_casual' in trips.columns:
        if trips['member_casual'].notna().any():
            # Use boolean indexing (no copy)
            casual_mask = trips['member_casual'] == 'casual'
            casual_data = trips.loc[casual_mask, group_cols]
            casual_data['is_casual'] = 1
            # Count total and casual
            total_per_group = trips.groupby(group_cols, observed=True).size().rename("n_total")
            casual_per_group = casual_data.groupby(group_cols, observed=True).size().rename("n_casual")
            # Merge and compute share
            casual_share = (
                pd.concat([total_per_group, casual_per_group], axis=1)
                .fillna(0)
                .eval("casual_share = n_casual / n_total")
                .reset_index()[group_cols + ['casual_share']]
            )
            add_feats = add_feats.merge(casual_share, on=group_cols, how='left')
        else:
            add_feats['casual_share'] = np.nan
    
    hexh = hex_hour.merge(add_feats, on=['h3', 'hour'], how='left')
    hexh.to_csv(os.path.join(out_dir, "assoc_hex_hour.csv"), index=False)

    
    # === Correlation analysis ===
    corr_vars = ['trip_count','avg_speed','avg_distance_km','avg_duration_minutes','weekend_share','casual_share']
    corr_df = hexh[corr_vars + ['crash_rate_per_10k','severity_per_10k']].dropna()
    if not corr_df.empty:
        spearman = corr_df.corr(method='spearman')
        pearson = corr_df.corr(method='pearson')
        spearman.to_csv(os.path.join(out_dir, "assoc_corr_spearman.csv"))
        pearson.to_csv(os.path.join(out_dir, "assoc_corr_pearson.csv"))

        # visualize correlation matrix
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(spearman, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title("Spearman Correlation Matrix — Trip vs Crash Features")
        plot_and_save(fig, os.path.join(out_dir, "assoc_corr_heatmap.png"))
    
    # === Visual relationships ===
    plt.figure(figsize=(8,6))
    sns.regplot(data=hexh, x='trip_count', y='crash_rate_per_10k',
                scatter_kws={'s':10, 'alpha':0.3}, line_kws={'color':'black'})
    plt.title("Crash rate vs Trip Count (per H3 x hour)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "assoc_scatter_trips_vs_rate.png"), dpi=150, bbox_inches='tight')
    plt.close()

    if 'avg_speed' in hexh.columns:
        plt.figure(figsize=(8,6))
        sns.regplot(data=hexh, x='avg_speed', y='crash_rate_per_10k',
                    scatter_kws={'s':10, 'alpha':0.3}, line_kws={'color':'black'})
        plt.title("Crash rate vs Average Speed (per H3 x hour)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "assoc_scatter_speed_vs_rate.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    
    return hexh

