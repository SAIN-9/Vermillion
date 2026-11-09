"""
modules/cleaner.py
──────────────────
Smart column auto-mapping, data health audit, and cleaning pipeline.
Works purely in-memory with Pandas — zero database overhead.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ────────────────────────────────────────────────────────────────────
# 1.  CANONICAL SCHEMA  — the column names the rest of the app uses
# ────────────────────────────────────────────────────────────────────
CANONICAL_COLUMNS = {
    "invoice":     "InvoiceNo",
    "customer":    "CustomerID",
    "stockcode":   "StockCode",
    "description": "Description",
    "quantity":    "Quantity",
    "unitprice":   "UnitPrice",
    "date":        "InvoiceDate",
    "country":     "Country",
}

# Synonyms per canonical role  (lowercase → canonical key)
_SYNONYMS: Dict[str, List[str]] = {
    "invoice":     ["invoice", "invoiceno", "invoice_no", "invoice_id",
                    "invoiceid", "order", "order_id", "orderid",
                    "order_no", "orderno", "transaction_id", "transactionid"],
    "customer":    ["customer", "customerid", "customer_id", "cust_id",
                    "custid", "client_id", "clientid", "buyer_id"],
    "stockcode":   ["stockcode", "stock_code", "productcode", "product_code",
                    "sku", "item_code", "itemcode", "product_id", "productid"],
    "description": ["description", "product_name", "productname",
                    "item_name", "itemname", "product", "item_description",
                    "item"],
    "quantity":    ["quantity", "qty", "units", "amount", "units_sold",
                    "order_quantity", "orderquantity"],
    "unitprice":   ["unitprice", "unit_price", "price", "item_price",
                    "itemprice", "sell_price", "sellprice", "cost"],
    "date":        ["invoicedate", "invoice_date", "orderdate", "order_date",
                    "date", "transaction_date", "transactiondate",
                    "purchase_date", "purchasedate"],
    "country":     ["country", "region", "nation", "location", "geo",
                    "ship_country", "shipcountry", "billing_country"],
}


# ────────────────────────────────────────────────────────────────────
# 2.  AUTO-DETECTION  — fuzzy match incoming columns → canonical names
# ────────────────────────────────────────────────────────────────────
def _normalise(name: object) -> str:
    """Lower-case, strip whitespace & common separators."""
    return re.sub(r"[\s\-_]+", "", str(name).strip().lower())


def auto_detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Return {canonical_key: matched_original_column_name | None}.
    Matching is purely synonym-based; ambiguities are left as None.
    """
    mapping: Dict[str, Optional[str]] = {k: None for k in CANONICAL_COLUMNS}
    used: set = set()

    for canon_key, synonyms in _SYNONYMS.items():
        norm_syns = {_normalise(s) for s in synonyms}
        for col in df.columns:
            if col in used:
                continue
            if _normalise(col) in norm_syns:
                mapping[canon_key] = col
                used.add(col)
                break

    # ── Heuristic fall-backs (type-based guessing) ──
    # Date column — pick the first datetime-parseable column
    if mapping["date"] is None:
        for col in df.columns:
            if col in used:
                continue
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                mapping["date"] = col
                used.add(col)
                break
            try:
                pd.to_datetime(df[col].dropna().head(50), infer_datetime_format=True)
                mapping["date"] = col
                used.add(col)
                break
            except Exception:
                continue

    return mapping


# ────────────────────────────────────────────────────────────────────
# 3.  RENAME TO CANONICAL NAMES
# ────────────────────────────────────────────────────────────────────
def rename_to_canonical(df: pd.DataFrame,
                        mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Rename detected columns while preserving one column per canonical field."""
    rename_map = {}
    for canon_key, orig_col in mapping.items():
        target = CANONICAL_COLUMNS[canon_key]
        # Never rename a second input column over a canonical column that exists.
        if orig_col is not None and orig_col in df.columns and orig_col != target and target not in df.columns:
            rename_map[orig_col] = target
    renamed = df.rename(columns=rename_map)
    # Repeated source headers are legal in CSV/XLSX files but break date indexing.
    return renamed.loc[:, ~renamed.columns.duplicated(keep="first")].copy()


# ────────────────────────────────────────────────────────────────────
# 4.  DATA HEALTH AUDIT
# ────────────────────────────────────────────────────────────────────
def data_health_audit(df: pd.DataFrame) -> Dict:
    """
    Return a dict with counts & a composite quality score.
    Expects columns already renamed to canonical names.
    """
    total_rows = len(df)
    report: Dict = {
        "total_rows":       total_rows,
        "missing_values":   {},
        "total_missing":    0,
        "negative_qty":     0,
        "negative_price":   0,
        "cancelled_orders": 0,
        "duplicates":       0,
        "quality_score":    100.0,
    }

    # Missing values per column
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n > 0:
            report["missing_values"][col] = n
    report["total_missing"] = int(df.isna().sum().sum())

    # Negative quantities
    if "Quantity" in df.columns:
        quantity = pd.to_numeric(df["Quantity"], errors="coerce")
        report["negative_qty"] = int((quantity < 0).sum())

    # Negative prices
    if "UnitPrice" in df.columns:
        unit_price = pd.to_numeric(df["UnitPrice"], errors="coerce")
        report["negative_price"] = int((unit_price < 0).sum())

    # Cancelled orders (InvoiceNo starts with 'C')
    if "InvoiceNo" in df.columns:
        inv_str = df["InvoiceNo"].astype(str)
        report["cancelled_orders"] = int(inv_str.str.startswith("C").sum())

    # Duplicates
    report["duplicates"] = int(df.duplicated().sum())

    # ── Quality Score (simple weighted deduction) ──
    if total_rows > 0:
        penalty = 0.0
        penalty += (report["total_missing"]    / (total_rows * len(df.columns))) * 30
        penalty += (report["negative_qty"]     / total_rows) * 20
        penalty += (report["negative_price"]   / total_rows) * 10
        penalty += (report["cancelled_orders"] / total_rows) * 20
        penalty += (report["duplicates"]       / total_rows) * 20
        report["quality_score"] = round(max(0.0, 100.0 - penalty * 100), 1)

    return report


# ────────────────────────────────────────────────────────────────────
# 5.  AUTO-CLEAN PIPELINE
# ────────────────────────────────────────────────────────────────────
def auto_clean(df: pd.DataFrame,
               remove_negative_qty: bool = True,
               drop_missing_ids: bool = True,
               remove_cancellations: bool = True,
               remove_duplicates: bool = True,
               remove_negative_price: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean the dataset in-place and return (cleaned_df, summary_dict).
    """
    summary: Dict[str, int] = {}
    n_before = len(df)
    df = df.copy()

    for col in ["Quantity", "UnitPrice"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. Remove cancelled orders
    if remove_cancellations and "InvoiceNo" in df.columns:
        mask = df["InvoiceNo"].astype(str).str.startswith("C")
        summary["cancelled_removed"] = int(mask.sum())
        df = df[~mask]

    # 2. Remove negative quantities
    if remove_negative_qty and "Quantity" in df.columns:
        mask = df["Quantity"] < 0
        summary["negative_qty_removed"] = int(mask.sum())
        df = df[~mask]

    # 3. Remove negative prices
    if remove_negative_price and "UnitPrice" in df.columns:
        mask = df["UnitPrice"] < 0
        summary["negative_price_removed"] = int(mask.sum())
        df = df[~mask]

    # 4. Drop rows missing critical IDs
    if drop_missing_ids:
        for col in ["InvoiceNo", "CustomerID"]:
            if col in df.columns:
                before = len(df)
                df = df.dropna(subset=[col])
                summary[f"missing_{col}_dropped"] = before - len(df)

    # 5. Deduplicate
    if remove_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        summary["duplicates_removed"] = before - len(df)

    # 6. Parse date column
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(
            df["InvoiceDate"], errors="coerce", utc=True
        ).dt.tz_convert(None)
        dropped = df["InvoiceDate"].isna().sum()
        if dropped > 0:
            summary["unparseable_dates_dropped"] = int(dropped)
            df = df.dropna(subset=["InvoiceDate"])

    # 7. Ensure numeric types
    for col in ["Quantity", "UnitPrice"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=[col])

    # 8. Compute Revenue helper
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    summary["rows_before"] = n_before
    summary["rows_after"] = len(df)
    summary["rows_removed"] = n_before - len(df)

    df = df.reset_index(drop=True)
    return df, summary
