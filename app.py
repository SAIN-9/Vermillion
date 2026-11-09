"""Vermillion — a focused retail analytics workspace."""

from __future__ import annotations

import io
from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules.cleaner import (
    auto_detect_columns,
    auto_clean,
    data_health_audit,
    rename_to_canonical,
)
from modules.analytics import (
    compute_kpis,
    generate_insights,
    geo_breakdown,
    pareto_data,
    revenue_over_time,
    top_customers,
    top_products,
)
from modules.ml_engine import (
    compute_rfm,
    forecast_revenue,
    market_basket_analysis,
    run_kmeans_rfm,
)
from ui import (
    load_css,
    header,
    hero,
    render_3d_sphere,
)

# ════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG & GLOBAL THEME
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Vermillion",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject external modular CSS (styles.css)
load_css("styles.css")

# ════════════════════════════════════════════════════════════════════
# 2. PLOTLY MINIMALIST ARCHIVE THEME
# ════════════════════════════════════════════════════════════════════
def apply_chart_style(fig: go.Figure, height: int = 350) -> go.Figure:
    """Applies a minimalist Ethan Vale archive theme across Plotly charts."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#8c8783", size=11),
        margin=dict(l=35, r=20, t=35, b=35),
        height=height,
        hoverlabel=dict(
            bgcolor="#070707",
            bordercolor="rgba(244, 242, 239, 0.28)",
            font_size=11,
            font_family="Inter, sans-serif",
            font_color="#f4f2ef",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#8c8783", size=10, family="Inter, sans-serif"),
        title_font=dict(color="#f4f2ef", size=11, family="Inter, sans-serif"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(244, 242, 239, 0.08)",
        zeroline=False,
        tickfont=dict(color="#8c8783", size=10, family="Inter, sans-serif"),
        title_font=dict(color="#f4f2ef", size=11, family="Inter, sans-serif"),
    )
    return fig


_COLOR_PRIMARY = "#f4f2ef"
_COLOR_ACCENT = "rgba(244, 242, 239, 0.45)"
_COLOR_MUTED = "#8c8783"
_PALETTE = ["#f4f2ef", "#c4bfb9", "#8c8783", "#5c5753", "#3a3633", "#e0ded9"]


# ════════════════════════════════════════════════════════════════════
# 3. CACHED ENGINE CALLS
# ════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def cached_rfm_analysis(df: pd.DataFrame, n_clusters: int):
    rfm = compute_rfm(df)
    return run_kmeans_rfm(rfm, n_clusters=n_clusters)


@st.cache_data(show_spinner=False)
def cached_forecast_revenue(df: pd.DataFrame, periods: int):
    return forecast_revenue(df, periods=periods)


@st.cache_data(show_spinner=False)
def cached_market_basket(df: pd.DataFrame, min_support: float, min_confidence: float):
    return market_basket_analysis(df, min_support=min_support, min_confidence=min_confidence)


# ════════════════════════════════════════════════════════════════════
# 4. FORMATTING HELPERS
# ════════════════════════════════════════════════════════════════════
def _fmt_currency(val: float) -> str:
    if abs(val) >= 1_000_000:
        return f"£{val/1_000_000:,.1f}M"
    if abs(val) >= 1_000:
        return f"£{val/1_000:,.1f}K"
    return f"£{val:,.2f}"


def _fmt_int(val: int) -> str:
    if val >= 1_000_000:
        return f"{val/1_000_000:,.1f}M"
    if val >= 1_000:
        return f"{val/1_000:,.1f}K"
    return f"{val:,}"


# ════════════════════════════════════════════════════════════════════
# 5. DATA CONTROLS
# ════════════════════════════════════════════════════════════════════
def render_data_controls() -> bool:
    """Render only the controls needed to prepare or replace uploaded data."""
    control_col, action_col = st.columns([3, 1])
    with control_col:
        clean_active = st.toggle(
            "Clean cancellations, invalid rows, duplicates, and missing customer IDs",
            value=True,
        )
    with action_col:
        if st.button("Change data"):
            st.session_state["raw_df"] = None
            st.session_state["data_source_name"] = ""
            st.rerun()
    return clean_active


# ════════════════════════════════════════════════════════════════════
# 6. TAB 1: EXECUTIVE DASHBOARD
# ════════════════════════════════════════════════════════════════════
def render_executive_dashboard(df: pd.DataFrame):
    kpis = compute_kpis(df)

    # A concise operating snapshot makes the first screen useful at a glance.
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Gross revenue</div>
            <div class="kpi-value">{_fmt_currency(kpis["total_revenue"])}</div>
            <div class="kpi-detail">Across the active dataset</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Orders</div>
            <div class="kpi-value">{_fmt_int(kpis["total_orders"])}</div>
            <div class="kpi-detail">Unique transactions</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Customers</div>
            <div class="kpi-value">{_fmt_int(kpis["unique_customers"])}</div>
            <div class="kpi-detail">Unique purchasers</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Average order</div>
            <div class="kpi-value">{_fmt_currency(kpis["aov"])}</div>
            <div class="kpi-detail">Revenue per order</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Repeat rate</div>
            <div class="kpi-value">{kpis["repeat_rate"]:.1f}%</div>
            <div class="kpi-detail">Customers with repeat orders</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Revenue Trajectory Over Time ──
    header_col, control_col = st.columns([3, 1])
    with header_col:
        st.markdown('<div class="ev-section-title">Revenue over time</div><div class="ev-section-subtitle">Trend and rolling average</div>', unsafe_allow_html=True)
    with control_col:
        freq_choice = st.selectbox(
            "View", ["Daily", "Weekly", "Monthly"], index=0, label_visibility="collapsed"
        )
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    freq = freq_map.get(freq_choice, "D")

    try:
        ts = revenue_over_time(df, freq=freq)
    except ValueError as exc:
        st.info(f"Revenue trend unavailable: {exc}")
        ts = None

    if ts is not None:
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Scatter(
            x=ts["Date"], y=ts["Revenue"],
            mode="lines", name="Observed Revenue",
            line=dict(color=_COLOR_PRIMARY, width=1.4),
            fill="tozeroy",
            fillcolor="rgba(244, 242, 239, 0.04)",
            hovertemplate="%{x|%b %d, %Y}<br><b>£%{y:,.2f}</b><extra></extra>",
        ))
        fig_rev.add_trace(go.Scatter(
            x=ts["Date"], y=ts["RollingAvg"],
            mode="lines", name="Rolling Benchmark",
            line=dict(color=_COLOR_ACCENT, width=1.4, dash="dot"),
            hovertemplate="Benchmark: <b>£%{y:,.2f}</b><extra></extra>",
        ))
        fig_rev.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        apply_chart_style(fig_rev, height=330)
        st.plotly_chart(fig_rev, use_container_width=True)

    # ── Top Products & Top Customers (2-Column Grid) ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="ev-section-title">Top products</div><div class="ev-section-subtitle">By revenue</div>', unsafe_allow_html=True)
        tp = top_products(df, 10)
        fig_tp = px.bar(
            tp, y="Product", x="Revenue", orientation="h",
            color_discrete_sequence=[_COLOR_PRIMARY],
        )
        fig_tp.update_layout(xaxis_title="", yaxis_title="")
        fig_tp.update_yaxes(autorange="reversed", tickfont=dict(size=10, family="Inter, sans-serif"))
        apply_chart_style(fig_tp, height=360)
        fig_tp.update_traces(
            hovertemplate="<b>%{y}</b><br>Yield: £%{x:,.2f}<extra></extra>"
        )
        st.plotly_chart(fig_tp, use_container_width=True)

    with col2:
        st.markdown('<div class="ev-section-title">Top customers</div><div class="ev-section-subtitle">Lifetime value</div>', unsafe_allow_html=True)
        tc = top_customers(df, 10)
        fig_tc = px.bar(
            tc, y="CustomerID", x="LifetimeValue", orientation="h",
            color_discrete_sequence=[_COLOR_ACCENT],
        )
        fig_tc.update_layout(xaxis_title="", yaxis_title="")
        fig_tc.update_yaxes(autorange="reversed", type="category", tickfont=dict(size=10, family="Inter, sans-serif"))
        apply_chart_style(fig_tc, height=360)
        fig_tc.update_traces(
            hovertemplate="Account #%{y}<br>LTV: £%{x:,.2f}<extra></extra>"
        )
        st.plotly_chart(fig_tc, use_container_width=True)

    # ── Geographic Revenue + Pareto Distribution ──
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="ev-section-title">Revenue by market</div><div class="ev-section-subtitle">Top countries</div>', unsafe_allow_html=True)
        if "Country" in df.columns:
            geo = geo_breakdown(df, 12)
            fig_geo = px.bar(
                geo, x="Country", y="Revenue",
                color_discrete_sequence=[_COLOR_PRIMARY],
            )
            fig_geo.update_layout(xaxis_title="", yaxis_title="")
            fig_geo.update_xaxes(tickangle=-35)
            apply_chart_style(fig_geo, height=340)
            fig_geo.update_traces(
                hovertemplate="<b>%{x}</b><br>Yield: £%{y:,.2f}<extra></extra>"
            )
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.info("Add a country or region column to view revenue by market.")

    with col4:
        st.markdown('<div class="ev-section-title">Revenue concentration</div><div class="ev-section-subtitle">Cumulative product contribution</div>', unsafe_allow_html=True)
        par = pareto_data(df)
        fig_par = go.Figure()
        fig_par.add_trace(go.Bar(
            x=par["Rank"], y=par["Revenue"],
            name="Individual Yield", marker_color="rgba(244, 242, 239, 0.28)",
            hovertemplate="Rank #%{x}<br>Yield: £%{y:,.2f}<extra></extra>",
        ))
        fig_par.add_trace(go.Scatter(
            x=par["Rank"], y=par["CumPct"],
            name="Cumulative Share", yaxis="y2",
            line=dict(color=_COLOR_PRIMARY, width=1.6),
            hovertemplate="Cumulative: %{y:.1f}%<extra></extra>",
        ))
        fig_par.add_hline(y=80, line_dash="dash", line_color="rgba(244, 242, 239, 0.4)", yref="y2")
        fig_par.update_layout(
            yaxis2=dict(
                overlaying="y", side="right", range=[0, 105],
                showgrid=False, tickfont=dict(color="#8c8783", size=10),
            ),
            xaxis_title="Catalog Rank",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        apply_chart_style(fig_par, height=340)
        st.plotly_chart(fig_par, use_container_width=True)

    st.markdown('<div class="ev-section-title">Product landscape</div><div class="ev-section-subtitle">A spatial view of the leading products</div>', unsafe_allow_html=True)
    sphere_items = [
        {"title": str(row["Product"])[:28], "value": _fmt_currency(float(row["Revenue"]))}
        for _, row in tp.head(16).iterrows()
    ]
    render_3d_sphere(sphere_items)

# ════════════════════════════════════════════════════════════════════
# 7. TAB 2: MACHINE LEARNING & DATA SCIENCE
# ════════════════════════════════════════════════════════════════════
def render_ml_tab(df: pd.DataFrame):
    subtab1, subtab2, subtab3 = st.tabs([
        "Customer Personas (RFM)",
        "Revenue Forecast",
        "Market Basket Recommendations",
    ])

    # ── RFM Segmentation ──
    with subtab1:
        c1, c2 = st.columns([1, 3])
        with c1:
            clusters_k = st.selectbox("Customer groups", options=[3, 4, 5, 6], index=1)
        with c2:
            st.caption("Accounts are grouped by recency, purchase frequency, and lifetime value.")

        try:
            rfm_clustered, cluster_stats = cached_rfm_analysis(df, n_clusters=clusters_k)
            st.session_state["rfm_clustered"] = rfm_clustered

            # Persona status chips
            persona_html = '<div style="display:flex; gap:10px; margin-bottom:18px; flex-wrap:wrap;">'
            for _, r in cluster_stats.iterrows():
                persona_html += f"""
                <div class="ev-insight-tag" style="padding: 6px 12px; font-size: 0.72rem;">
                    <b>{r['Segment']}</b> · {r['Count']:,} clients (avg {_fmt_currency(r['AvgMonetary'])})
                </div>
                """
            persona_html += '</div>'
            st.markdown(persona_html, unsafe_allow_html=True)

            scatter_col, table_col = st.columns([3, 2])
            with scatter_col:
                fig_rfm = px.scatter(
                    rfm_clustered,
                    x="Frequency", y="Monetary",
                    color="Segment",
                    color_discrete_sequence=_PALETTE,
                    hover_data=["CustomerID", "Recency"],
                )
                apply_chart_style(fig_rfm, height=440)
                st.plotly_chart(fig_rfm, use_container_width=True)

            with table_col:
                st.markdown('<div class="ev-section-title">Cohort Architecture</div>', unsafe_allow_html=True)
                show_tbl = cluster_stats[["Segment", "Count", "AvgRecency", "AvgFrequency", "AvgMonetary"]].copy()
                show_tbl.columns = ["Persona", "Accounts", "Recency (d)", "Frequency", "Monetary (£)"]
                for col in ["Recency (d)", "Frequency", "Monetary (£)"]:
                    show_tbl[col] = show_tbl[col].round(1)
                st.dataframe(show_tbl, use_container_width=True, hide_index=True)

        except Exception as e:
            st.warning(f"RFM analysis unavailable: {e}")

    # ── Time-Series Forecasting ──
    with subtab2:
        h_col1, h_col2 = st.columns([1, 3])
        with h_col1:
            horizon = st.selectbox("Projection window", options=[14, 30, 60, 90], index=1)
        with h_col2:
            st.caption("Holt-Winters Exponential Smoothing evaluating historical trend patterns with 95% confidence cones.")

        try:
            history, forecast = cached_forecast_revenue(df, periods=int(horizon or 30))

            m1, m2, m3 = st.columns(3)
            tot_pred = forecast["Revenue"].sum()
            avg_pred = forecast["Revenue"].mean()
            m1.markdown(f"""
            <div class="ev-kpi-card">
                <div class="ev-kpi-label">Projected Yield</div>
                <div class="ev-kpi-val">{_fmt_currency(tot_pred)}</div>
                <div class="ev-kpi-desc">Total forecasted period gross</div>
            </div>
            """, unsafe_allow_html=True)
            m2.markdown(f"""
            <div class="ev-kpi-card">
                <div class="ev-kpi-label">Daily Expected Mean</div>
                <div class="ev-kpi-val">{_fmt_currency(avg_pred)}</div>
                <div class="ev-kpi-desc">Smoothed daily benchmark</div>
            </div>
            """, unsafe_allow_html=True)
            m3.markdown(f"""
            <div class="ev-kpi-card">
                <div class="ev-kpi-label">Projection Horizon</div>
                <div class="ev-kpi-val">{horizon} Days</div>
                <div class="ev-kpi-desc">Forward modeling window</div>
            </div>
            """, unsafe_allow_html=True)

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(
                x=history["Date"], y=history["Revenue"],
                mode="lines", name="Historical",
                line=dict(color=_COLOR_MUTED, width=1.4),
            ))
            fig_fc.add_trace(go.Scatter(
                x=forecast["Date"], y=forecast["Revenue"],
                mode="lines", name="Predicted Trajectory",
                line=dict(color=_COLOR_PRIMARY, width=1.8, dash="dash"),
            ))
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([forecast["Date"], forecast["Date"][::-1]]),
                y=pd.concat([forecast["Upper"], forecast["Lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(244, 242, 239, 0.06)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% Confidence Cone",
            ))
            fig_fc.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            apply_chart_style(fig_fc, height=380)
            st.plotly_chart(fig_fc, use_container_width=True)

        except Exception as e:
            st.warning(f"Forecasting unavailable: {e}")

    # ── Market Basket Analysis ──
    with subtab3:
        p_col1, p_col2, p_col3 = st.columns([1.5, 1.5, 1])
        with p_col1:
            min_sup = st.selectbox(
                "Minimum support", [0.01, 0.02, 0.03, 0.05, 0.10], index=1,
                help="Frequency with which item combinations appear together.",
            )
        with p_col2:
            min_conf = st.selectbox(
                "Minimum confidence", [0.10, 0.20, 0.30, 0.40, 0.50], index=2,
                help="Probability that item B is purchased given item A is in transaction.",
            )
        with p_col3:
            st.write("")
            run_btn = st.button("Execute Apriori Mining", type="primary", use_container_width=True)

        if run_btn or st.session_state.get("mba_run"):
            st.session_state["mba_run"] = True
            with st.spinner("Mining association rules from catalog transactions…"):
                rules, msg = cached_market_basket(df, min_support=min_sup, min_confidence=min_conf)
                if rules is not None and not rules.empty:
                    st.caption(f"Rules identified: {msg}")
                    st.dataframe(rules.head(25), use_container_width=True, hide_index=True)
                else:
                    st.info(msg or "No rules met the selected support and confidence thresholds.")


# ════════════════════════════════════════════════════════════════════
# 8. TAB 3: STRATEGIC INSIGHTS & EXPORT CENTER
# ════════════════════════════════════════════════════════════════════
def render_insights_export_tab(df: pd.DataFrame):
    kpis = compute_kpis(df)
    insights = generate_insights(df, kpis)

    st.markdown('<div class="ev-section-title">Strategic Executive Brief</div>', unsafe_allow_html=True)
    st.markdown('<div class="ev-section-subtitle">Algorithmic diagnostics &amp; pattern observations</div>', unsafe_allow_html=True)

    if insights:
        cards_html = '<div class="ev-insight-grid">'
        for ins in insights:
            cards_html += f"""
            <div class="ev-insight-card">
                <div class="ev-insight-top">
                    <span class="ev-insight-cat">{ins.get('category', 'Diagnostic')}</span>
                    <span class="ev-insight-tag">{ins.get('tag', 'Observation')}</span>
                </div>
                <div class="ev-insight-headline">{ins.get('headline', '')}</div>
                <div class="ev-insight-detail">{ins.get('detail', '')}</div>
            </div>
            """
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)
    else:
        st.info("Insufficient variance in repository to derive algorithmic insights.")

    # ── Export Center ──
    st.markdown('<div class="ev-section-title" style="margin-top: 24px;">Export Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="ev-section-subtitle">Download standardized structured outputs</div>', unsafe_allow_html=True)

    ex1, ex2 = st.columns(2)

    with ex1:
        st.markdown("""
        <div class="ev-card">
            <div style="font-family: var(--font-serif); font-size: 1.15rem; color: var(--ink); margin-bottom: 4px;">Sanitized Repository</div>
            <div style="font-size: 0.78rem; color: var(--dim); margin-bottom: 14px;">Cleaned transactional matrix with anomalies resolved.</div>
        """, unsafe_allow_html=True)
        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download Cleaned CSV",
            data=csv_buf.getvalue(),
            file_name="vermillion_cleaned_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with ex2:
        st.markdown("""
        <div class="ev-card">
            <div style="font-family: var(--font-serif); font-size: 1.15rem; color: var(--ink); margin-bottom: 4px;">Client Persona Architecture</div>
            <div style="font-size: 0.78rem; color: var(--dim); margin-bottom: 14px;">Account roster tagged with behavioral RFM cluster personas.</div>
        """, unsafe_allow_html=True)
        rfm_export = st.session_state.get("rfm_clustered")
        if rfm_export is not None:
            rfm_buf = io.BytesIO()
            rfm_export.to_csv(rfm_buf, index=False)
            st.download_button(
                label="Download Personas CSV",
                data=rfm_buf.getvalue(),
                file_name="vermillion_customer_personas.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.caption("Execute RFM segmentation in Tab 2 to compile personas.")
        st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 9. ONBOARDING & EMPTY STATE
# ════════════════════════════════════════════════════════════════════
def render_onboarding():
    st.markdown("""
    <section class="landing">
        <div class="landing-copy">
            <h1>Know what is<br>moving your business.</h1>
            <p>Bring in a transaction export to explore revenue, customer behaviour, and product performance.</p>
        </div>
    </section>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Select a file to begin",
        type=["csv", "xlsx", "parquet"],
    )
    if uploaded is not None:
        try:
            extension = uploaded.name.rsplit(".", 1)[-1].lower()
            if extension == "csv":
                df = pd.read_csv(uploaded, encoding="latin-1")
            elif extension == "xlsx":
                df = pd.read_excel(uploaded, engine="openpyxl")
            elif extension == "parquet":
                df = pd.read_parquet(uploaded)
            else:
                st.error("Unsupported file extension.")
                return
            st.session_state["raw_df"] = df
            st.session_state["data_source_name"] = uploaded.name
            st.rerun()
        except Exception as e:
            st.error(f"Error reading source file: {e}")


# ════════════════════════════════════════════════════════════════════
# 10. MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════════
def main():
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None
    if "data_source_name" not in st.session_state:
        st.session_state["data_source_name"] = ""

    header()

    raw_df = st.session_state["raw_df"]

    # If no repository loaded yet, render onboarding hero
    if raw_df is None:
        render_onboarding()
        return

    hero()

    # Normalize column mapping
    mapping = auto_detect_columns(raw_df)
    df = rename_to_canonical(raw_df, mapping)

    required_columns = {
        "InvoiceNo",
        "CustomerID",
        "Description",
        "Quantity",
        "UnitPrice",
        "InvoiceDate",
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        st.error(
            "This file is missing required transaction columns: "
            + ", ".join(missing_columns)
        )
        st.caption(
            "Include an order ID, customer ID, product description, quantity, "
            "unit price, and transaction date to continue."
        )
        return

    # Health audit
    audit = data_health_audit(df)
    quality_score = audit["quality_score"]

    # Keep the active dataset visible without competing with the analysis.
    st.markdown(f"""
    <div class="workspace-intro">
        <div class="workspace-title">Active workspace</div>
        <div class="dataset-pills">
          <span class="pill">{escape(st.session_state["data_source_name"])}</span>
          <span class="pill">{len(df):,} records</span>
          <span class="pill">{quality_score:.0f}% quality</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    clean_active = render_data_controls()
    df, cleaning_summary = auto_clean(
        df,
        remove_negative_qty=clean_active,
        remove_negative_price=clean_active,
        drop_missing_ids=clean_active,
        remove_cancellations=clean_active,
        remove_duplicates=clean_active,
    )
    if df.empty:
        st.warning(
            "No usable transactions remain after parsing dates and numeric values. "
            "Check the uploaded file and try again."
        )
        return
    if cleaning_summary["rows_removed"]:
        st.caption(
            f"Prepared dataset excludes {cleaning_summary['rows_removed']:,} "
            "rows that were invalid or removed by the selected cleaning options."
        )

    # Main Tabs
    tab1, tab2, tab3 = st.tabs([
        "Executive Dashboard",
        "Machine Learning & Forecasts",
        "Strategic Insights & Export",
    ])

    with tab1:
        render_executive_dashboard(df)

    with tab2:
        render_ml_tab(df)

    with tab3:
        render_insights_export_tab(df)


if __name__ == "__main__":
    main()
