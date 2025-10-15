import pandas as pd
import numpy as np 
import os
from scipy.stats import kruskal, chi2_contingency


def feature_selection_stats(df: pd.DataFrame, out_dir: str, target_col: str = 'accident', 
                            alpha: float = 0.05, categorical_threshold: int = 15):
    """
    Perform feature selection:
      - Kruskal–Wallis test for numeric features
      - Chi-squared test for categorical features
    Returns two DataFrames with results.
    """

    # ---  Separate numeric and categorical features ---
    numeric_feats = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore')
    categorical_feats = [col for col in df.columns 
                         if col not in numeric_feats and col != target_col]

    # --- Kruskal–Wallis for numeric features ---
    kruskal_results = []
    for col in numeric_feats:
        try:
            group0 = df.loc[df[target_col] == 0, col].dropna()
            group1 = df.loc[df[target_col] == 1, col].dropna()
            if len(group0) > 5 and len(group1) > 5:
                stat, p = kruskal(group0, group1)
                kruskal_results.append((col, stat, p))
        except Exception as e:
            print(f" Skipped numeric feature {col}: {e}")

    kruskal_df = pd.DataFrame(kruskal_results, columns=["feature", "statistic", "p_value"])
    kruskal_df["significant"] = kruskal_df["p_value"] < alpha
    kruskal_df = kruskal_df.sort_values("p_value")

    # ---  Chi-squared test for categorical features ---
    chi2_results = []
    for col in categorical_feats:
        try:
            # only test columns with few unique categories
            if df[col].nunique() <= categorical_threshold:
                contingency = pd.crosstab(df[col], df[target_col])
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2, p, dof, ex = chi2_contingency(contingency)
                    chi2_results.append((col, chi2, p, dof))
        except Exception as e:
            print(f" Skipped categorical feature {col}: {e}")

    chi2_df = pd.DataFrame(chi2_results, columns=["feature", "chi2", "p_value", "dof"])
    chi2_df["significant"] = chi2_df["p_value"] < alpha
    chi2_df = chi2_df.sort_values("p_value")

    # --- 4️ Print summary ---
    print(f"\n Kruskal–Wallis (numeric features): {len(kruskal_df)} tested")
    print(kruskal_df[kruskal_df["significant"]].head(10))

    print(f"\n Chi-squared (categorical features): {len(chi2_df)} tested")
    print(chi2_df[chi2_df["significant"]].head(10))

    kruskal_results_f = os.path.join(out_dir, "features", "kruskal_results.csv")
    chi2_results_f = os.path.join(out_dir, "features", "chi2_results.csv")

    
    kruskal_df.to_csv(kruskal_results_f, index=None)
    chi2_df.to_csv(chi2_results_f, index=None)
    
    return kruskal_df, chi2_df
