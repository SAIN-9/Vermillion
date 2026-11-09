"""
modules/analytics.py
────────────────────
KPI computation, time-series metrics, and geographic breakdowns.
Uses DuckDB in-memory SQL where aggregation is heavy, with Pandas fallbacks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────
# HELPER: quick DuckDB query on a DataFrame
# ────────────────────────────────────────────────────────────────────
def _q(sql: str, **tables) -> pd.DataFrame:
    """
    Run *sql* against named DataFrames passed as keyword arguments.
    Example:  _q("SELECT * FROM df WHERE Revenue > 100", df=my_df)
    """
    con = duckdb.connect()
    for name, frame in tables.items():
        con.register(name, frame)
    result = con.execute(sql).fetchdf()
    con.close()
    return result


# ────────────────────────────────────────────────────────────────────
# 1.  EXECUTIVE KPIs
# ────────────────────────────────────────────────────────────────────
def compute_kpis(df: pd.DataFrame) -> Dict:
    """Return a flat dict of high-level KPIs."""
    if "Revenue" not in df.columns:
        df = df.copy()
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    total_revenue = float(df["Revenue"].sum())

    # Total unique orders
    total_orders = int(df["InvoiceNo"].nunique()) if "InvoiceNo" in df.columns else 0

    # Unique customers
    unique_customers = int(df["CustomerID"].nunique()) if "CustomerID" in df.columns else 0

    # AOV
    aov = total_revenue / total_orders if total_orders > 0 else 0.0

    # Repeat customer rate
    if "CustomerID" in df.columns and "InvoiceNo" in df.columns:
        cust_orders = df.groupby("CustomerID")["InvoiceNo"].nunique()
        repeat_rate = float((cust_orders > 1).sum() / len(cust_orders) * 100) if len(cust_orders) > 0 else 0.0
    else:
        repeat_rate = 0.0

    return {
        "total_revenue":     round(total_revenue, 2),
        "total_orders":      total_orders,
        "unique_customers":  unique_customers,
        "aov":               round(aov, 2),
        "repeat_rate":       round(repeat_rate, 1),
    }


# ────────────────────────────────────────────────────────────────────
# 2.  REVENUE OVER TIME
# ────────────────────────────────────────────────────────────────────
def revenue_over_time(df: pd.DataFrame,
                      freq: str = "D") -> pd.DataFrame:
    """
    Aggregate revenue by date frequency (D / W / M).
    Returns a DataFrame with columns: Date, Revenue, RollingAvg.
    """
    if "Revenue" not in df.columns:
        df = df.copy()
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    if "InvoiceDate" not in df.columns:
        raise ValueError("A transaction date column is required for the revenue trend.")

    working = df.copy()
    invoice_dates = working.loc[:, "InvoiceDate"]
    if isinstance(invoice_dates, pd.DataFrame):
        invoice_dates = invoice_dates.iloc[:, 0]
    working["InvoiceDate"] = pd.to_datetime(
        invoice_dates, errors="coerce", utc=True
    ).dt.tz_convert(None)
    working = working.dropna(subset=["InvoiceDate"])
    if working.empty:
        raise ValueError("No valid transaction dates were found in the uploaded file.")

    ts = (working.set_index("InvoiceDate")
            .resample(freq)["Revenue"]
            .sum()
            .reset_index()
            .rename(columns={"InvoiceDate": "Date"}))

    window = {"D": 7, "W": 4, "M": 3, "ME": 3}.get(freq, 7)
    # Use 'ME' for month-end frequency in newer pandas
    ts["RollingAvg"] = ts["Revenue"].rolling(window=window, min_periods=1).mean()
    return ts


# ────────────────────────────────────────────────────────────────────
# 3.  TOP PRODUCTS BY REVENUE
# ────────────────────────────────────────────────────────────────────
def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top *n* products by total revenue."""
    result = _q(f"""
        SELECT   Description          AS Product,
                 SUM(Quantity * UnitPrice) AS Revenue
        FROM     df
        WHERE    Description IS NOT NULL
        GROUP BY Description
        ORDER BY Revenue DESC
        LIMIT    {n}
    """, df=df)
    return result


# ────────────────────────────────────────────────────────────────────
# 4.  TOP CUSTOMERS BY LTV
# ────────────────────────────────────────────────────────────────────
def top_customers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top *n* customers by lifetime spend."""
    result = _q(f"""
        SELECT   CAST(CustomerID AS VARCHAR) AS CustomerID,
                 SUM(Quantity * UnitPrice)   AS LifetimeValue
        FROM     df
        WHERE    CustomerID IS NOT NULL
        GROUP BY CustomerID
        ORDER BY LifetimeValue DESC
        LIMIT    {n}
    """, df=df)
    return result


# ────────────────────────────────────────────────────────────────────
# 5.  GEOGRAPHIC BREAKDOWN
# ────────────────────────────────────────────────────────────────────
def geo_breakdown(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Revenue by country, top *n*."""
    result = _q(f"""
        SELECT   Country,
                 SUM(Quantity * UnitPrice) AS Revenue,
                 COUNT(DISTINCT InvoiceNo) AS Orders
        FROM     df
        WHERE    Country IS NOT NULL
        GROUP BY Country
        ORDER BY Revenue DESC
        LIMIT    {n}
    """, df=df)
    return result


# ────────────────────────────────────────────────────────────────────
# 6.  PARETO (80 / 20) DATA
# ────────────────────────────────────────────────────────────────────
def pareto_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative revenue share by product — for the 80/20 chart.
    Returns: Product, Revenue, CumRevenue, CumPct
    """
    prod = _q("""
        SELECT   Description          AS Product,
                 SUM(Quantity * UnitPrice) AS Revenue
        FROM     df
        WHERE    Description IS NOT NULL
        GROUP BY Description
        ORDER BY Revenue DESC
    """, df=df)

    prod["CumRevenue"] = prod["Revenue"].cumsum()
    total = prod["Revenue"].sum()
    prod["CumPct"] = prod["CumRevenue"] / total * 100 if total > 0 else 0
    prod = prod.reset_index(drop=True)
    prod.index = prod.index + 1      # 1-based product rank
    prod.index.name = "Rank"
    return prod.reset_index()


# ────────────────────────────────────────────────────────────────────
# 7.  AUTOMATED INSIGHT BULLETS
# ────────────────────────────────────────────────────────────────────
class Insight(dict):
    """Structured insight object with clean string fallback."""
    def __init__(self, category: str, headline: str, detail: str, tag: str = "Observation"):
        super().__init__(category=category, headline=headline, detail=detail, tag=tag)

    def __str__(self) -> str:
        return f"**{self['headline']}**: {self['detail']}"


def generate_insights(df: pd.DataFrame, kpis: Dict) -> List[Insight]:
    """Rule-based executive summary structured insights."""
    insights: List[Insight] = []

    # Peak revenue month
    if "InvoiceDate" in df.columns and "Revenue" in df.columns:
        monthly = df.set_index("InvoiceDate").resample("ME")["Revenue"].sum()
        if len(monthly) > 0:
            peak = monthly.idxmax()
            avg_mo = monthly.mean()
            diff_pct = ((monthly.max() - avg_mo) / avg_mo * 100) if avg_mo > 0 else 0
            insights.append(Insight(
                category="Seasonality",
                headline=f"Peak Month: {peak.strftime('%B %Y')}",
                detail=f"Recorded highest revenue at £{monthly.max():,.0f} (+{diff_pct:.0f}% vs monthly average).",
                tag="Growth",
            ))

    # Revenue concentration
    if "CustomerID" in df.columns and "Revenue" in df.columns:
        cust_rev = df.groupby("CustomerID")["Revenue"].sum().sort_values(ascending=False)
        total = cust_rev.sum()
        if total > 0:
            top10_n = max(1, len(cust_rev) // 10)
            top10_pct = cust_rev.head(top10_n).sum() / total * 100
            insights.append(Insight(
                category="Client Concentration",
                headline=f"Top 10% Accounts Drive {top10_pct:.1f}% Revenue",
                detail=f"{top10_n:,} high-volume accounts represent the majority of settled capital.",
                tag="Risk Benchmark",
            ))

    # Average basket size
    if "InvoiceNo" in df.columns and "Quantity" in df.columns:
        basket = df.groupby("InvoiceNo")["Quantity"].sum().mean()
        insights.append(Insight(
            category="Order Velocity",
            headline=f"Average Basket Size: {basket:.1f} Units",
            detail=f"Cross-selling and inventory bundling directly influence this transaction baseline.",
            tag="Yield Index",
        ))

    # Repeat customers
    if kpis.get("repeat_rate", 0) > 0:
        insights.append(Insight(
            category="Client Retention",
            headline=f"{kpis['repeat_rate']:.1f}% Repeat Purchaser Index",
            detail="Repurchase frequency demonstrates sustained client loyalty and catalog demand.",
            tag="Retention",
        ))

    # Geographic
    if "Country" in df.columns and "Revenue" in df.columns:
        geo = df.groupby("Country")["Revenue"].sum().sort_values(ascending=False)
        if len(geo) > 0:
            top_country = geo.index[0]
            top_share = geo.iloc[0] / geo.sum() * 100
            insights.append(Insight(
                category="Territory Dispersion",
                headline=f"{top_country} Generates {top_share:.1f}% Total Capital",
                detail=f"Primary sovereign market. Secondary regional expansions offer incremental growth headroom.",
                tag="Market Share",
            ))

    # Fastest-growing product
    if "InvoiceDate" in df.columns and "Revenue" in df.columns and "Description" in df.columns:
        mid = df["InvoiceDate"].min() + (df["InvoiceDate"].max() - df["InvoiceDate"].min()) / 2
        first_half = df[df["InvoiceDate"] <= mid].groupby("Description")["Revenue"].sum()
        second_half = df[df["InvoiceDate"] > mid].groupby("Description")["Revenue"].sum()
        common = first_half.index.intersection(second_half.index)
        if len(common) > 0:
            growth = ((second_half[common] - first_half[common]) / first_half[common].replace(0, np.nan)).dropna()
            if len(growth) > 0:
                fastest = growth.idxmax()
                pct = growth.max() * 100
                if pct > 0:
                    insights.append(Insight(
                        category="Inventory Velocity",
                        headline=f"Accelerating Asset: {fastest}",
                        detail=f"Surged +{pct:.0f}% in revenue velocity during the second half of observed timeline.",
                        tag="Trending",
                    ))

    return insights
