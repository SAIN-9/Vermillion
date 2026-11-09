"""
modules/ml_engine.py
────────────────────
Machine Learning workloads:
  • RFM Customer Segmentation  (K-Means)
  • Revenue Time-Series Forecasting  (Holt-Winters / Exponential Smoothing)
  • Market Basket Analysis  (Apriori + Association Rules)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ════════════════════════════════════════════════════════════════════
# 1.  RFM  CUSTOMER  SEGMENTATION
# ════════════════════════════════════════════════════════════════════

def compute_rfm(df: pd.DataFrame,
                reference_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Build the RFM feature table.
    Expects canonical columns: CustomerID, InvoiceNo, InvoiceDate, Revenue.
    """
    required = {"CustomerID", "InvoiceNo", "InvoiceDate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for RFM: {missing}")

    if "Revenue" not in df.columns:
        df = df.copy()
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
          .agg(
              Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
              Frequency=("InvoiceNo", "nunique"),
              Monetary=("Revenue", "sum"),
          )
          .reset_index()
    )

    # Drop customers with non-positive monetary
    rfm = rfm[rfm["Monetary"] > 0].reset_index(drop=True)
    return rfm


def run_kmeans_rfm(rfm: pd.DataFrame,
                   n_clusters: int = 4,
                   random_state: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalise RFM features → K-Means → assign cluster labels + personas.
    Returns (rfm_with_clusters, cluster_summary_dict).
    """
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = km.fit_predict(scaled)

    # ── Assign business personas ──
    # Strategy: rank clusters by composite score (low recency is good,
    # high frequency & monetary are good).
    cluster_stats = (
        rfm.groupby("Cluster")
           .agg(
               AvgRecency=("Recency", "mean"),
               AvgFrequency=("Frequency", "mean"),
               AvgMonetary=("Monetary", "mean"),
               Count=("CustomerID", "count"),
           )
           .reset_index()
    )

    # Composite score: normalise each metric 0-1, then combine
    r_min, r_max = cluster_stats["AvgRecency"].min(), cluster_stats["AvgRecency"].max()
    f_min, f_max = cluster_stats["AvgFrequency"].min(), cluster_stats["AvgFrequency"].max()
    m_min, m_max = cluster_stats["AvgMonetary"].min(), cluster_stats["AvgMonetary"].max()

    def _safe_norm(val, lo, hi, invert=False):
        if hi == lo:
            return 0.5
        n = (val - lo) / (hi - lo)
        return 1 - n if invert else n

    cluster_stats["Score"] = cluster_stats.apply(
        lambda row: (
            _safe_norm(row["AvgRecency"],   r_min, r_max, invert=True) * 0.3 +
            _safe_norm(row["AvgFrequency"], f_min, f_max) * 0.35 +
            _safe_norm(row["AvgMonetary"],  m_min, m_max) * 0.35
        ), axis=1
    )
    cluster_stats = cluster_stats.sort_values("Score", ascending=False).reset_index(drop=True)

    # Map persona labels
    _PERSONAS = [
        "Champions",
        "Loyal Accounts",
        "At-Risk",
        "Hibernating",
        "Emerging",
        "Requires Attention",
    ]
    persona_map = {}
    for idx, row in cluster_stats.iterrows():
        persona_map[row["Cluster"]] = _PERSONAS[idx] if idx < len(_PERSONAS) else f"Segment {idx}"

    rfm["Segment"] = rfm["Cluster"].map(persona_map)
    cluster_stats["Segment"] = cluster_stats["Cluster"].map(persona_map)

    return rfm, cluster_stats


# ════════════════════════════════════════════════════════════════════
# 2.  TIME-SERIES  FORECASTING  (Holt-Winters)
# ════════════════════════════════════════════════════════════════════

def forecast_revenue(df: pd.DataFrame,
                     periods: int = 30,
                     seasonal_periods: int = 7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate daily revenue → fit Exponential Smoothing → forecast.
    Returns (history_df, forecast_df) both with columns: Date, Revenue,
    and forecast_df additionally has Lower, Upper.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if "Revenue" not in df.columns:
        df = df.copy()
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    daily = (
        df.set_index("InvoiceDate")
          .resample("D")["Revenue"]
          .sum()
          .reset_index()
          .rename(columns={"InvoiceDate": "Date"})
    )
    daily = daily.sort_values("Date").reset_index(drop=True)

    # Need enough data points for seasonal decomposition
    min_points = seasonal_periods * 2 + 1
    if len(daily) < min_points:
        seasonal_periods = max(2, len(daily) // 3)

    # Fit model — use additive trend and seasonal if enough data
    try:
        model = ExponentialSmoothing(
            daily["Revenue"],
            trend="add",
            seasonal="add" if len(daily) >= seasonal_periods * 2 else None,
            seasonal_periods=seasonal_periods if len(daily) >= seasonal_periods * 2 else None,
        ).fit(optimized=True)
    except Exception:
        # Fallback: trend-only
        model = ExponentialSmoothing(
            daily["Revenue"],
            trend="add",
            seasonal=None,
        ).fit(optimized=True)

    forecast = model.forecast(periods)
    residuals = model.resid
    std_resid = residuals.std() if len(residuals) > 1 else 0

    future_dates = pd.date_range(
        start=daily["Date"].iloc[-1] + pd.Timedelta(days=1),
        periods=periods,
        freq="D",
    )

    forecast_df = pd.DataFrame({
        "Date":    future_dates,
        "Revenue": forecast.values,
        "Lower":   forecast.values - 1.96 * std_resid,
        "Upper":   forecast.values + 1.96 * std_resid,
    })
    # Clip lower bound at 0
    forecast_df["Lower"] = forecast_df["Lower"].clip(lower=0)

    return daily, forecast_df


# ════════════════════════════════════════════════════════════════════
# 3.  MARKET  BASKET  ANALYSIS  (Apriori)
# ════════════════════════════════════════════════════════════════════

def market_basket_analysis(df: pd.DataFrame,
                           min_support: float = 0.02,
                           min_confidence: float = 0.3,
                           max_items: int = 500) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Run Apriori + association rules on basket data.
    Returns (rules_df | None, status_message).
    """
    from mlxtend.frequent_patterns import apriori, association_rules

    required = {"InvoiceNo", "Description", "Quantity"}
    missing = required - set(df.columns)
    if missing:
        return None, f"Missing columns: {missing}"

    # Build basket matrix  (invoice × product → 0/1)
    basket = (
        df.groupby(["InvoiceNo", "Description"])["Quantity"]
          .sum()
          .unstack()
          .fillna(0)
    )

    # Limit to top products by frequency to keep computation tractable
    product_freq = (basket > 0).sum().sort_values(ascending=False)
    top_products = product_freq.head(max_items).index
    basket = basket[top_products]

    # Encode as boolean
    basket = basket > 0

    # Drop invoices with fewer than 2 items
    basket = basket[basket.sum(axis=1) >= 2]

    if basket.empty:
        return None, "Not enough multi-item transactions for analysis."

    # Run Apriori
    try:
        freq_items = apriori(basket, min_support=min_support, use_colnames=True)
    except Exception as e:
        return None, f"Apriori failed: {e}"

    if freq_items.empty:
        return None, "No frequent itemsets found. Try lowering minimum support."

    # Generate rules
    try:
        rules = association_rules(freq_items, metric="confidence",
                                  min_threshold=min_confidence)
    except Exception as e:
        return None, f"Association rules failed: {e}"

    if rules.empty:
        return None, "No association rules met the confidence threshold."

    # Clean up for display
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    rules_display = rules[display_cols].copy()
    rules_display.columns = ["If Customer Buys", "They Also Buy",
                             "Support", "Confidence", "Lift"]

    # Round
    for c in ["Support", "Confidence", "Lift"]:
        rules_display[c] = rules_display[c].round(3)

    return rules_display, f"Found {len(rules_display)} association rules."
