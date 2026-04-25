"""
app.py — Household Power Consumption Forecast Dashboard
========================================================
Streamlit UI for real-time next-hour electricity prediction.
Models: Ridge · Lasso · PLS · PCR
"""

import streamlit as st
import numpy as np
import joblib
import os
import json
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ─────────────────────────────────── */
html, body, .stApp { background-color: #0d0d0d !important; color: #e2e8f0; }
[data-testid="collapsedControl"] { display: none; }

/* ── Top nav ──────────────────────────────── */
.topnav {
    display: flex; align-items: center;
    background: #111111; border-bottom: 1px solid #222;
    padding: 13px 28px; margin: -1rem -1rem 1.5rem -1rem;
}
.topnav-logo { font-size: 1.15rem; font-weight: 700; color: #f1f5f9; letter-spacing:.02em; }
.topnav-sub  { color: #4b5563; font-size: 0.8rem; margin-left: 10px; }
.topnav-badge {
    margin-left: auto; background: #1c1c1c; border: 1px solid #2a2a2a;
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.75rem; color: #6b7280; letter-spacing:.05em;
}

/* ── Tabs ─────────────────────────────────── */
[data-testid="stTabs"] { border-bottom: 1px solid #1e1e1e; }
button[data-baseweb="tab"] { color: #4b5563 !important; font-size:.85rem; font-weight:500; padding:10px 18px; }
button[data-baseweb="tab"][aria-selected="true"] { color: #f1f5f9 !important; border-bottom: 2px solid #e2e8f0 !important; }

/* ── Cards ────────────────────────────────── */
.card {
    background: #111111; border: 1px solid #1e1e1e;
    border-radius: 12px; padding: 18px 22px; margin-bottom: 14px;
}
.card-label {
    font-size: .7rem; font-weight: 600; color: #4b5563;
    letter-spacing: .1em; text-transform: uppercase; margin-bottom: 10px;
}

/* ── Metrics ──────────────────────────────── */
[data-testid="stMetric"] {
    background: #111111 !important; border: 1px solid #1e1e1e !important;
    border-radius: 10px !important; padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] { color: #4b5563 !important; font-size:.72rem !important; letter-spacing:.06em; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size:1.45rem !important; font-weight:700 !important; }

/* ── Prediction hero ──────────────────────── */
.pred-hero {
    background: #111111; border: 1px solid #222;
    border-radius: 14px; padding: 30px 24px; text-align: center;
}
.ph-label { color: #4b5563; font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; }
.ph-value { color: #f1f5f9; font-size: 3.2rem; font-weight: 800; line-height:1.05; margin: 6px 0 2px; }
.ph-unit  { color: #6b7280; font-size:.88rem; }
.ph-badge {
    display:inline-block; margin-top:12px; background:#1c1c1c;
    border:1px solid #2a2a2a; border-radius:20px;
    padding:3px 14px; font-size:.72rem; color:#9ca3af;
}

/* ── Section divider ──────────────────────── */
.sh {
    font-size:.7rem; font-weight:600; color:#374151;
    letter-spacing:.1em; text-transform:uppercase;
    margin:1.4rem 0 .7rem; padding-bottom:5px;
    border-bottom:1px solid #1a1a1a;
}

/* ── Info / tip box ───────────────────────── */
.tip {
    background:#111; border-left:3px solid #1e1e1e;
    border-radius:0 8px 8px 0; padding:10px 14px;
    color:#4b5563; font-size:.8rem; margin-bottom:1rem;
}

/* ── Vertical divider ─────────────────────── */
.vdiv { border-left:1px solid #1a1a1a; min-height:460px; margin:0 6px; }

/* ── Getting-started cards ────────────────── */
.gs { background:#0f0f0f; border:1px solid #1a1a1a; border-radius:12px; padding:20px; }
.gs-n { font-size:2rem; font-weight:800; color:#1e1e1e; line-height:1; margin-bottom:8px; }
.gs-t { font-size:.88rem; font-weight:600; color:#e2e8f0; margin-bottom:5px; }
.gs-b { font-size:.8rem; color:#4b5563; line-height:1.55; }

/* ── Model eval table ─────────────────────── */
.eval-row {
    display:flex; align-items:center; justify-content:space-between;
    background:#111; border:1px solid #1e1e1e; border-radius:10px;
    padding:14px 20px; margin-bottom:10px;
}
.eval-name { font-size:.9rem; font-weight:600; color:#e2e8f0; min-width:80px; }
.eval-bar-wrap { flex:1; margin:0 20px; background:#1a1a1a; border-radius:4px; height:6px; }
.eval-bar { height:6px; border-radius:4px; background:#374151; }
.eval-val { font-size:.85rem; color:#9ca3af; min-width:110px; text-align:right; }

/* ── Global text ──────────────────────────── */
.stApp p, .stApp li, .stMarkdown p { color: #e2e8f0; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILE CHECK
# ─────────────────────────────────────────────
# FILE CHECK  (only core files are required)
# ─────────────────────────────────────────────
required = [
    "scaler.pkl", "features.pkl",
    "models/ridge.pkl", "models/lasso.pkl", "models/pls.pkl",
]
missing = [f for f in required if not os.path.exists(f)]
if missing:
    st.error(f"Missing: {', '.join(missing)}  —  run `python train.py` first.")
    st.stop()

# ─────────────────────────────────────────────
# LOAD ARTEFACTS  (pcr + results are optional)
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    sc  = joblib.load("scaler.pkl")
    ft  = joblib.load("features.pkl")

    # Load whichever models exist
    mdl = {}
    for m in ["ridge", "lasso", "pls", "pcr"]:
        path = f"models/{m}.pkl"
        if os.path.exists(path):
            mdl[m] = joblib.load(path)

    # Load CV results if available, else build placeholder
    res_path = "models/results.json"
    if os.path.exists(res_path):
        with open(res_path) as f:
            res = json.load(f)
    else:
        res = {m: {"mse": 0.0, "rmse": 0.0} for m in mdl}

    return sc, ft, mdl, res

scaler, features, all_models, cv_results = load_artifacts()

# Build label/color dicts from whatever models loaded
_ALL_LABELS = {"ridge": "Ridge", "lasso": "Lasso", "pls": "PLS", "pcr": "PCR"}
_ALL_COLORS = {"ridge": "#e2e8f0", "lasso": "#94a3b8", "pls": "#cbd5e1", "pcr": "#64748b"}
MODEL_LABELS = {m: _ALL_LABELS[m] for m in all_models}
MODEL_COLORS = {m: _ALL_COLORS[m] for m in all_models}
DAY_NAMES    = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

CHART_LAYOUT = dict(
    paper_bgcolor="#111111", plot_bgcolor="#111111",
    font=dict(color="#9ca3af", size=11),
    margin=dict(l=20, r=20, t=14, b=20),
)

def dark_axes(fig, grid=True):
    fig.update_xaxes(color="#374151", showgrid=False, zeroline=False)
    fig.update_yaxes(color="#374151", gridcolor="#1a1a1a" if grid else "rgba(0,0,0,0)",
                     zeroline=False)

# ─────────────────────────────────────────────
# TOP NAV
# ─────────────────────────────────────────────
st.markdown("""
<div class="topnav">
  <span class="topnav-logo">⚡ Energy Forecast</span>
  <span class="topnav-sub">Household Power Consumption · UCI Dataset</span>
  <span class="topnav-badge">Ridge · Lasso · PLS · PCR</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡  Forecast",
    "📊  Model Comparison",
    "📈  Evaluation",
    "ℹ️  About",
])

# ══════════════════════════════════════════════
# TAB 1 — FORECAST
# ══════════════════════════════════════════════
with tab1:
    left, div, right = st.columns([1, 0.02, 1.7])

    # ── Inputs ───────────────────────────────
    with left:
        st.markdown('<div class="sh">⚙️ Configure Forecast</div>', unsafe_allow_html=True)

        model_choice = st.selectbox(
            "Algorithm",
            list(MODEL_LABELS.keys()),
            format_func=lambda x: MODEL_LABELS[x],
        )

        st.markdown('<div class="sh">Time Context</div>', unsafe_allow_html=True)
        hour = st.slider("Hour of Day", 0, 23, 14)
        day  = st.slider("Day of Week  (0 = Mon)", 0, 6, 2)
        st.caption(f"📅 **{DAY_NAMES[day]}** · **{hour:02d}:00**")

        st.markdown('<div class="sh">Recent Readings</div>', unsafe_allow_html=True)
        lag_1  = st.number_input("Last Hour (kW)",           value=1.50, step=0.05, format="%.2f")
        lag_24 = st.number_input("Same Hour Yesterday (kW)", value=1.20, step=0.05, format="%.2f")

        st.markdown("")
        run = st.button("⚡ Run Forecast", use_container_width=True, type="primary")

    with div:
        st.markdown('<div class="vdiv"></div>', unsafe_allow_html=True)

    # ── Results ───────────────────────────────
    with right:
        # Derived features (always computed so Tab 2 can use them)
        rolling_mean_3  = (lag_1 + lag_24) / 2
        rolling_std_3   = abs(lag_1 - lag_24)
        rolling_mean_6  = (lag_1 + lag_24) / 2          # simplified proxy
        rolling_mean_24 = (lag_1 + lag_24) / 2
        lag_2           = lag_1 * 0.98                  # proxy
        lag_168         = lag_24 * 1.02                 # proxy
        month           = 6                             # default June
        is_weekend      = int(day >= 5)

        feature_map = {
            "hour": hour, "day": day, "month": month, "is_weekend": is_weekend,
            "lag_1": lag_1, "lag_2": lag_2, "lag_24": lag_24, "lag_168": lag_168,
            "rolling_mean_3": rolling_mean_3, "rolling_std_3": rolling_std_3,
            "rolling_mean_6": rolling_mean_6, "rolling_mean_24": rolling_mean_24,
        }
        input_vec    = np.array([feature_map[f] for f in features]).reshape(1, -1)
        input_scaled = scaler.transform(input_vec)

        if run:
            pred = all_models[model_choice].predict(input_scaled)
            prediction = float(pred.ravel()[0])
            d_last = prediction - lag_1
            d_yest = prediction - lag_24

            # Hero
            st.markdown(f"""
<div class="pred-hero">
  <div class="ph-label">Forecasted Next-Hour Consumption</div>
  <div class="ph-value">{prediction:.2f}</div>
  <div class="ph-unit">kilowatts (kW)</div>
  <div class="ph-badge">{MODEL_LABELS[model_choice]} Regression</div>
</div>""", unsafe_allow_html=True)

            st.markdown("")

            # KPI 2×2
            st.markdown('<div class="sh">Key Metrics</div>', unsafe_allow_html=True)
            r1c1, r1c2 = st.columns(2)
            r2c1, r2c2 = st.columns(2)
            r1c1.metric("Forecast",       f"{prediction:.2f} kW")
            r1c2.metric("Rolling Mean",   f"{rolling_mean_3:.2f} kW")
            r2c1.metric("vs Last Hour",   f"{lag_1:.2f} kW",  delta=f"{d_last:+.2f} kW")
            r2c2.metric("vs Yesterday",   f"{lag_24:.2f} kW", delta=f"{d_yest:+.2f} kW")

            # Trend chart
            st.markdown('<div class="sh">Consumption Trend</div>', unsafe_allow_html=True)
            labels = ["Yesterday", "Last Hour", "Forecast"]
            values = [lag_24, lag_1, prediction]
            dot_c  = ["#1e1e1e", "#2e2e2e", MODEL_COLORS[model_choice]]

            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(
                x=labels, y=values,
                fill="tozeroy", fillcolor="rgba(55,65,81,0.08)",
                mode="lines+markers+text",
                line=dict(color=MODEL_COLORS[model_choice], width=2.5),
                marker=dict(size=[10, 10, 14], color=dot_c,
                            line=dict(color="#e2e8f0", width=1.5)),
                text=[f"{v:.2f}" for v in values],
                textposition="top center",
                textfont=dict(color="#9ca3af", size=11),
            ))
            fig_t.update_layout(height=230, **CHART_LAYOUT)
            dark_axes(fig_t)
            fig_t.update_yaxes(title_text="kW")
            st.plotly_chart(fig_t, use_container_width=True)

        else:
            st.markdown('<div class="sh">Results</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="tip">Set your inputs on the left and click <strong>⚡ Run Forecast</strong>.</div>',
                unsafe_allow_html=True,
            )
            fig_ph = go.Figure()
            fig_ph.update_layout(
                height=300, **CHART_LAYOUT,
                annotations=[dict(text="No forecast yet", x=.5, y=.5,
                                  xref="paper", yref="paper", showarrow=False,
                                  font=dict(color="#1e1e1e", size=15))],
            )
            dark_axes(fig_ph, grid=False)
            st.plotly_chart(fig_ph, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sh">All Models · Current Inputs</div>', unsafe_allow_html=True)

    # Recompute for current inputs (same feature_map from Tab 1 scope)
    rolling_mean_3  = (lag_1 + lag_24) / 2
    rolling_std_3   = abs(lag_1 - lag_24)
    rolling_mean_6  = (lag_1 + lag_24) / 2
    rolling_mean_24 = (lag_1 + lag_24) / 2
    lag_2           = lag_1 * 0.98
    lag_168         = lag_24 * 1.02
    month           = 6
    is_weekend      = int(day >= 5)

    feature_map2 = {
        "hour": hour, "day": day, "month": month, "is_weekend": is_weekend,
        "lag_1": lag_1, "lag_2": lag_2, "lag_24": lag_24, "lag_168": lag_168,
        "rolling_mean_3": rolling_mean_3, "rolling_std_3": rolling_std_3,
        "rolling_mean_6": rolling_mean_6, "rolling_mean_24": rolling_mean_24,
    }
    iv2     = np.array([feature_map2[f] for f in features]).reshape(1, -1)
    is2     = scaler.transform(iv2)
    preds2  = {m: float(all_models[m].predict(is2).ravel()[0]) for m in all_models}

    # 4 metric cards
    mc = st.columns(4)
    for col, (m, lbl) in zip(mc, MODEL_LABELS.items()):
        col.metric(lbl, f"{preds2[m]:.2f} kW")

    st.markdown("")
    g1, g2 = st.columns(2)

    with g1:
        st.markdown('<div class="card-label">Prediction by Model</div>', unsafe_allow_html=True)
        fig_b = go.Figure(go.Bar(
            x=[MODEL_LABELS[m] for m in preds2],
            y=list(preds2.values()),
            marker_color=[MODEL_COLORS[m] for m in preds2],
            marker_line_color="#0d0d0d", marker_line_width=2,
            text=[f"{v:.2f}" for v in preds2.values()],
            textposition="outside", textfont=dict(color="#9ca3af"),
        ))
        fig_b.update_layout(height=300, **CHART_LAYOUT)
        dark_axes(fig_b)
        fig_b.update_yaxes(title_text="kW")
        st.plotly_chart(fig_b, use_container_width=True)

    with g2:
        st.markdown('<div class="card-label">Input Feature Profile</div>', unsafe_allow_html=True)
        rv = [feature_map2[f] for f in features]
        mn, mx = min(rv), max(rv)
        norm = [(v - mn) / (mx - mn + 1e-9) for v in rv]

        fig_r = go.Figure()
        for m in all_models:
            fig_r.add_trace(go.Scatterpolar(
                r=norm + [norm[0]], theta=features + [features[0]],
                fill="toself", fillcolor=MODEL_COLORS[m], opacity=0.12,
                line=dict(color=MODEL_COLORS[m], width=1.5),
                name=MODEL_LABELS[m],
            ))
        fig_r.update_layout(
            height=300,
            polar=dict(
                bgcolor="#111111",
                radialaxis=dict(visible=True, color="#374151", gridcolor="#1a1a1a"),
                angularaxis=dict(color="#4b5563"),
            ),
            legend=dict(bgcolor="#111", bordercolor="#1e1e1e",
                        font=dict(color="#6b7280", size=10)),
            **CHART_LAYOUT,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with st.expander("🔍 Feature Values Used"):
        fc = st.columns(len(features))
        for col, f in zip(fc, features):
            col.metric(f, f"{feature_map2[f]:.3f}")

# ══════════════════════════════════════════════
# TAB 3 — EVALUATION
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sh">Cross-Validation Results · 5-Fold TimeSeriesSplit</div>',
                unsafe_allow_html=True)

    # Best model
    best = min(cv_results, key=lambda m: cv_results[m]["mse"]) if any(cv_results[m]["mse"] > 0 for m in cv_results) else list(cv_results.keys())[0]
    has_results = any(cv_results[m]["rmse"] > 0 for m in cv_results)
    if has_results:
        st.markdown(
            f'<div class="tip">🏆 Best model: <strong>{MODEL_LABELS[best]}</strong> '
            f'— RMSE {cv_results[best]["rmse"]:.4f} kW</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="tip">⚠️ No evaluation results found. Push <code>models/results.json</code> to GitHub or re-run <code>train.py</code>.</div>',
            unsafe_allow_html=True,
        )

    # Eval rows with inline bar
    max_rmse = max(v["rmse"] for v in cv_results.values())
    for m, res in cv_results.items():
        pct = int(res["rmse"] / max_rmse * 100) if max_rmse > 0 else 0
        st.markdown(f"""
<div class="eval-row">
  <span class="eval-name">{MODEL_LABELS[m]}</span>
  <div class="eval-bar-wrap"><div class="eval-bar" style="width:{pct}%"></div></div>
  <span class="eval-val">MSE {res['mse']:.4f} · RMSE {res['rmse']:.4f}</span>
</div>""", unsafe_allow_html=True)

    st.markdown("")

    # Side-by-side bar charts
    e1, e2 = st.columns(2)

    with e1:
        st.markdown('<div class="card-label">MSE by Model</div>', unsafe_allow_html=True)
        fig_mse = go.Figure(go.Bar(
            x=[MODEL_LABELS[m] for m in cv_results],
            y=[cv_results[m]["mse"] for m in cv_results],
            marker_color=[MODEL_COLORS[m] for m in cv_results],
            marker_line_color="#0d0d0d", marker_line_width=2,
            text=[f"{cv_results[m]['mse']:.4f}" for m in cv_results],
            textposition="outside", textfont=dict(color="#9ca3af"),
        ))
        fig_mse.update_layout(height=280, **CHART_LAYOUT)
        dark_axes(fig_mse)
        fig_mse.update_yaxes(title_text="MSE")
        st.plotly_chart(fig_mse, use_container_width=True)

    with e2:
        st.markdown('<div class="card-label">RMSE by Model</div>', unsafe_allow_html=True)
        fig_rmse = go.Figure(go.Bar(
            x=[MODEL_LABELS[m] for m in cv_results],
            y=[cv_results[m]["rmse"] for m in cv_results],
            marker_color=[MODEL_COLORS[m] for m in cv_results],
            marker_line_color="#0d0d0d", marker_line_width=2,
            text=[f"{cv_results[m]['rmse']:.4f}" for m in cv_results],
            textposition="outside", textfont=dict(color="#9ca3af"),
        ))
        fig_rmse.update_layout(height=280, **CHART_LAYOUT)
        dark_axes(fig_rmse)
        fig_rmse.update_yaxes(title_text="RMSE (kW)")
        st.plotly_chart(fig_rmse, use_container_width=True)

    # Scatter: MSE vs RMSE
    st.markdown('<div class="card-label">MSE vs RMSE Scatter</div>', unsafe_allow_html=True)
    fig_sc = go.Figure()
    for m in cv_results:
        fig_sc.add_trace(go.Scatter(
            x=[cv_results[m]["mse"]], y=[cv_results[m]["rmse"]],
            mode="markers+text",
            marker=dict(size=14, color=MODEL_COLORS[m],
                        line=dict(color="#0d0d0d", width=2)),
            text=[MODEL_LABELS[m]], textposition="top center",
            textfont=dict(color="#9ca3af", size=11),
            name=MODEL_LABELS[m],
        ))
    fig_sc.update_layout(
        height=260, showlegend=False, **CHART_LAYOUT,
        xaxis_title="MSE", yaxis_title="RMSE (kW)",
    )
    dark_axes(fig_sc)
    st.plotly_chart(fig_sc, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════
with tab4:
    st.markdown("")

    # How to use
    st.markdown('<div class="sh">How to Use</div>', unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    steps = [
        ("01", "Set Inputs", "Choose a model, set the hour and day, then enter last hour's and yesterday's consumption readings."),
        ("02", "Run Forecast", "Click ⚡ Run Forecast to get the next-hour prediction with trend chart and KPI metrics."),
        ("03", "Compare & Evaluate", "Use the Model Comparison and Evaluation tabs to see how Ridge, Lasso, PLS, and PCR differ."),
    ]
    for col, (n, t, b) in zip([h1, h2, h3], steps):
        col.markdown(f'<div class="gs"><div class="gs-n">{n}</div><div class="gs-t">{t}</div><div class="gs-b">{b}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("")

    # Models
    st.markdown('<div class="sh">Models</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    model_info = [
        ("Ridge", "L2 regularisation. Shrinks all coefficients evenly. Stable baseline with low variance."),
        ("Lasso", "L1 regularisation. Drives irrelevant coefficients to zero — automatic feature selection."),
        ("PLS", "Partial Least Squares. Finds latent components that maximise covariance with the target."),
        ("PCR", "Principal Component Regression. PCA for dimensionality reduction, then Ridge on components."),
    ]
    for col, (name, desc) in zip([m1, m2, m3, m4], model_info):
        col.markdown(f'<div class="card"><div class="card-label">{name}</div>{desc}</div>',
                     unsafe_allow_html=True)

    st.markdown("")

    # Pipeline
    st.markdown('<div class="sh">Pipeline</div>', unsafe_allow_html=True)
    p1, p2, p3, p4, p5 = st.columns(5)
    pipeline_steps = [
        ("1", "Load & Clean", "Handle missing values with forward-fill + back-fill"),
        ("2", "Resample", "Minute-level → hourly mean aggregation"),
        ("3", "Features", "Hour, day, month, weekend flag, lag-1/2/24/168h, rolling mean/std"),
        ("4", "Selection", "Forward sequential feature selection (6 best features)"),
        ("5", "Train", "StandardScaler + TimeSeriesSplit (5-fold) + MSE/RMSE evaluation"),
    ]
    for col, (n, t, b) in zip([p1, p2, p3, p4, p5], pipeline_steps):
        col.markdown(f'<div class="gs"><div class="gs-n">{n}</div><div class="gs-t">{t}</div><div class="gs-b">{b}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("")
    st.markdown("""
<div class="tip">
  📦 <strong>Dataset:</strong> UCI Individual Household Electric Power Consumption —
  ~2 million minute-level readings (Dec 2006 – Nov 2010), resampled to hourly.
  Source: <a href="https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set"
  style="color:#6b7280">Kaggle / UCI ML Repository</a>
</div>""", unsafe_allow_html=True)
