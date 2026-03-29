"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  app.py  —  CV Quantum Information Dashboard  (STATIC / CLOUD-SAFE v3.0)   ║
║  IIT Jodhpur · m25iqt013                                                    ║
║                                                                              ║
║  ARCHITECTURE: Zero heavy imports at runtime.                                ║
║  All quantum data is pre-computed (generate_data.py) → saved to data/*.pkl  ║
║  Only needs: streamlit · plotly · numpy · pandas                             ║
║  100% compatible with Streamlit Cloud free tier.                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import pickle, math, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ════════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  ── must be the very first Streamlit call
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CV Quantum Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "CV Quantum Information · IIT Jodhpur · m25iqt013"},
)

# ════════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — deep-space scientific aesthetic
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

/* ── Base ── */
:root {
  --bg:       #020817;
  --bg2:      #060e24;
  --bg3:      #0b1635;
  --border:   #1a2d5a;
  --border2:  #2a4080;
  --accent:   #6366f1;   /* indigo */
  --accent2:  #22d3ee;   /* cyan */
  --accent3:  #f472b6;   /* pink */
  --accent4:  #a3e635;   /* lime */
  --gold:     #fbbf24;
  --text:     #e2e8f0;
  --text-dim: #64748b;
  --mono:     'Space Mono', monospace;
  --sans:     'Syne', sans-serif;
}

[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stSidebar"]          { background: var(--bg2); border-right: 1px solid var(--border); }
[data-testid="stHeader"]           { background: transparent; }
html, body, [class*="css"]         { color: var(--text); }

/* ── Typography ── */
h1,h2,h3,h4 { font-family: var(--sans); letter-spacing: -0.02em; }
h1 { font-size: 2.1rem; font-weight: 800; color: #fff; }
h2 { font-size: 1.4rem; font-weight: 700; color: var(--accent2); }
h3 { font-size: 1.05rem; font-weight: 600; color: var(--accent); }
p, li, label { font-family: var(--sans); font-size: 0.88rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-family: var(--mono); font-size: 0.75rem; }
[data-testid="stSidebar"] .stRadio > label { color: #94a3b8 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: 0.76rem; color: var(--text-dim); }

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, var(--bg2), var(--bg3));
    border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] { font-family: var(--mono); color: var(--accent2); font-size: 1.3rem; font-weight: 700; }
[data-testid="stMetricLabel"] { color: var(--text-dim); font-size: 0.7rem; font-family: var(--mono); }
[data-testid="stMetricDelta"] { font-size: 0.7rem; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #312e81, var(--accent));
    color: white; border: none; border-radius: 8px;
    font-family: var(--mono); font-size: 0.78rem; font-weight: 700;
    padding: 9px 22px; letter-spacing: 0.05em; transition: all 0.2s;
}
.stButton > button:hover { background: linear-gradient(135deg, var(--accent), #818cf8); transform: translateY(-1px); }

/* ── Selectbox / slider ── */
[data-baseweb="select"] > div { background: var(--bg3) !important; border-color: var(--border) !important; border-radius: 8px !important; font-family: var(--mono); font-size: 0.8rem; }
.stSlider [data-testid="stThumb"] { background: var(--accent) !important; }
.stSlider [data-testid="stSliderTrack"] > div:first-child { background: var(--border2) !important; }
.stSlider [data-testid="stSliderTrack"] > div:last-child  { background: var(--accent) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]  { background: var(--bg2); border-bottom: 2px solid var(--border); gap: 4px; padding: 0 4px; }
.stTabs [data-baseweb="tab"]       { color: var(--text-dim); font-family: var(--mono); font-size: 0.78rem; padding: 10px 16px; border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"]     { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: var(--bg3); }

/* ── Custom components ── */
.page-banner {
    background: linear-gradient(135deg, var(--bg2) 0%, var(--bg3) 100%);
    border: 1px solid var(--border2); border-radius: 16px;
    padding: 24px 32px; margin-bottom: 20px;
    position: relative; overflow: hidden;
}
.page-banner::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 180px; height: 180px; border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.15), transparent 70%);
}
.page-banner h1 { margin: 0 0 4px 0; font-size: 1.8rem; color: #fff; }
.page-banner p  { margin: 3px 0; color: var(--text-dim); font-family: var(--mono); font-size: 0.78rem; }
.page-banner .tag { 
    display: inline-block; background: var(--bg3); border: 1px solid var(--border2);
    border-radius: 20px; padding: 2px 10px; font-size: 0.7rem; font-family: var(--mono);
    color: var(--accent2); margin: 4px 4px 0 0;
}

.eq-card {
    background: var(--bg2); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 0 10px 10px 0;
    padding: 12px 16px; margin: 8px 0;
    font-family: var(--mono); font-size: 0.82rem; color: #c7d2fe;
    line-height: 1.6;
}

.insight-card {
    background: linear-gradient(135deg, rgba(34,211,238,0.05), rgba(99,102,241,0.05));
    border: 1px solid rgba(34,211,238,0.2); border-radius: 12px;
    padding: 14px 18px; margin: 8px 0;
}
.insight-card .title { font-family: var(--mono); font-size: 0.7rem; color: var(--accent2); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
.insight-card p { margin: 0; font-size: 0.82rem; color: #94a3b8; }

.warn-card {
    background: rgba(251,191,36,0.07); border: 1px solid rgba(251,191,36,0.25);
    border-radius: 10px; padding: 12px 16px; margin: 8px 0;
}

.section-divider {
    border: none; border-top: 1px solid var(--border);
    margin: 20px 0;
}

.state-badge {
    display: inline-block; padding: 3px 12px;
    border-radius: 20px; font-family: var(--mono); font-size: 0.72rem;
    font-weight: 700; letter-spacing: 0.05em;
}
.badge-classical { background: rgba(251,191,36,0.15); color: var(--gold); border: 1px solid rgba(251,191,36,0.3); }
.badge-nonclassical { background: rgba(244,114,182,0.15); color: var(--accent3); border: 1px solid rgba(244,114,182,0.3); }

hr { border-color: var(--border); margin: 1rem 0; }
.stDataFrame { font-family: var(--mono); font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ════════════════════════════════════════════════════════════════════════════════
_LAY = dict(
    paper_bgcolor="#020817",
    plot_bgcolor="#060e24",
    font=dict(color="#e2e8f0", family="Space Mono, monospace", size=11),
    xaxis=dict(gridcolor="#0b1635", zerolinecolor="#1a2d5a", title_font=dict(size=11), tickfont=dict(size=9)),
    yaxis=dict(gridcolor="#0b1635", zerolinecolor="#1a2d5a", title_font=dict(size=11), tickfont=dict(size=9)),
    legend=dict(bgcolor="#060e24", bordercolor="#1a2d5a", borderwidth=1, font=dict(size=9)),
    margin=dict(l=44, r=24, t=46, b=36),
    colorway=["#6366f1","#22d3ee","#f472b6","#a3e635","#fbbf24","#60a5fa","#fb923c"],
)

# Colormaps
W_CS   = [[0.0,"#0a0f2e"],[0.35,"#1e1b4b"],[0.5,"#020817"],[0.65,"#4c0519"],[1.0,"#db2777"]]
W_CS2  = [[0.0,"#042f2e"],[0.5,"#020817"],[1.0,"#0891b2"]]
Q_CS   = [[0.0,"#020817"],[0.4,"#312e81"],[0.7,"#7c3aed"],[1.0,"#fbbf24"]]
DM_CS  = "RdBu_r"

# ════════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ════════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path(__file__).parent / "data"

@st.cache_resource(show_spinner=False)
def load_data():
    """Load all pre-computed pickle files once at startup."""
    out = {}
    for name in ("states", "channels", "gbs"):
        p = DATA_DIR / f"{name}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                out[name] = pickle.load(f)
        else:
            out[name] = None
    return out

DATA = load_data()

def _check(key):
    if DATA[key] is None:
        st.error(f"""
        **Data file `data/{key}.pkl` not found.**  
        Run `python generate_data.py` locally (requires QuTiP), then commit the `data/` folder to GitHub.
        """)
        return False
    return True

# ════════════════════════════════════════════════════════════════════════════════
# FIGURE HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _wigner_range(W):
    wmin, wmax = float(W.min()), float(W.max())
    if wmin >= 0: wmin = -1e-9
    if wmax <= 0: wmax = 1e-9
    return wmin, wmax

def fig_wigner(W_list, xvec, title="W(x,p)", height=400):
    W    = np.array(W_list)
    xv   = np.array(xvec)
    wmin, wmax = _wigner_range(W)
    fig  = go.Figure()
    fig.add_trace(go.Heatmap(
        z=W, x=xv, y=xv, colorscale=W_CS, zmin=wmin, zmax=wmax,
        colorbar=dict(title="W", len=0.8, thickness=12, tickfont=dict(size=9, color="#e2e8f0")),
        hovertemplate="x=%{x:.2f}  p=%{y:.2f}  W=%{z:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Contour(
        z=W, x=xv, y=xv,
        contours=dict(start=0, end=0, size=1, coloring="none"),
        line=dict(color="rgba(255,255,255,0.4)", width=1),
        showscale=False,
    ))
    fig.update_layout(**_LAY, height=height,
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#6366f1"), x=0.5),
        xaxis_title="x (position)", yaxis_title="p (momentum)")
    return fig

def fig_husimi(Q_list, xvec, title="Q(α)", height=400):
    fig = go.Figure(go.Heatmap(
        z=np.array(Q_list), x=np.array(xvec), y=np.array(xvec),
        colorscale=Q_CS,
        colorbar=dict(title="Q", len=0.8, thickness=12, tickfont=dict(size=9, color="#e2e8f0")),
        hovertemplate="Re(α)=%{x:.2f}  Im(α)=%{y:.2f}  Q=%{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**_LAY, height=height,
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#22d3ee"), x=0.5),
        xaxis_title="Re(α)", yaxis_title="Im(α)")
    return fig

def fig_density_matrix(rho_list, title="ρ", height=360):
    rho = np.array(rho_list)
    sub = make_subplots(1, 2, subplot_titles=["Re(ρ)", "Im(ρ)"], horizontal_spacing=0.08)
    for ci, data in enumerate([np.real(rho), np.imag(rho)], 1):
        vm = max(float(np.abs(data).max()), 1e-6)
        sub.add_trace(go.Heatmap(
            z=data, colorscale=DM_CS, zmin=-vm, zmax=vm, showscale=(ci==1),
            hovertemplate=f"n=%{{y}}  m=%{{x}}  val=%{{z:.4f}}<extra></extra>",
        ), row=1, col=ci)
    sub.update_layout(**_LAY, height=height,
        title=dict(text=f"<b>Density Matrix {title}</b>", font=dict(size=12, color="#6366f1"), x=0.5))
    return sub

def fig_photon_dist(probs_list, mean_n, title="Photon distribution P(n)", height=320):
    probs = np.array(probs_list)
    k     = np.arange(len(probs))
    poi   = np.array([
        math.exp(-max(mean_n,1e-9)) * (max(mean_n,1e-9)**ki) / math.factorial(int(ki))
        if int(ki) < 170 else 0.0 for ki in k
    ])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=k, y=probs, name="P(n)", marker_color="#6366f1", opacity=0.85))
    if mean_n > 0.01:
        fig.add_trace(go.Scatter(x=k, y=poi, mode="lines+markers", name="Poisson ref",
            line=dict(color="#fbbf24", width=2, dash="dot"), marker=dict(size=3)))
    fig.update_layout(**_LAY, height=height,
        title=dict(text=f"<b>{title}</b>", font=dict(size=12, color="#6366f1"), x=0.5),
        xaxis_title="Photon number n", yaxis_title="Probability P(n)",
        bargap=0.15)
    return fig

def fig_wigner_3d(W_list, xvec, title="W(x,p) 3-D", height=460):
    W  = np.array(W_list)
    xv = np.array(xvec)
    wa = max(float(np.abs(W).max()), 1e-6)
    fig = go.Figure(go.Surface(
        z=W, x=xv, y=xv, colorscale=W_CS, cmin=-wa, cmax=wa,
        showscale=True, opacity=0.93,
        colorbar=dict(title="W", len=0.7, thickness=12, tickfont=dict(size=9, color="#e2e8f0")),
        hovertemplate="x=%{x:.2f}  p=%{y:.2f}  W=%{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**_LAY, height=height,
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#6366f1"), x=0.5),
        scene=dict(
            xaxis=dict(title="x", gridcolor="#0b1635", backgroundcolor="#020817"),
            yaxis=dict(title="p", gridcolor="#0b1635", backgroundcolor="#020817"),
            zaxis=dict(title="W(x,p)", gridcolor="#0b1635", backgroundcolor="#020817"),
            bgcolor="#020817", camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
        ))
    return fig

def metrics_row(m: dict, wnv: float):
    """Render a row of metric cards."""
    cols = st.columns(6)
    items = [
        ("⟨n⟩", m.get('mean_n','—'), "Mean photon number"),
        ("Purity", m.get('purity','—'), "Tr(ρ²)  [1=pure]"),
        ("Entropy", m.get('entropy','—'), "von Neumann S(ρ)"),
        ("Mandel Q", m.get('mandel_Q','N/A'), "<0 sub-Poissonian"),
        ("Δx·Δp", m.get('heis_prod','—'), "Heisenberg product"),
        ("WNV", round(wnv,5), "Wigner negativity vol."),
    ]
    for col, (lbl, val, help_txt) in zip(cols, items):
        col.metric(label=lbl, value=str(val), help=help_txt)

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.markdown("""
    <div style="text-align:center;padding:18px 0 10px">
        <div style="font-size:2.4rem;filter:drop-shadow(0 0 12px #6366f1)">⚛️</div>
        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.05rem;
                    color:#e2e8f0;letter-spacing:0.05em;margin-top:6px">CV QUANTUM</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#475569;
                    margin-top:2px">IIT Jodhpur · m25iqt013</div>
    </div>
    <hr style="border-color:#1a2d5a;margin:8px 0 16px">
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "NAVIGATE",
        ["🔬 State Explorer",
         "🌌 Phase Space Zoo",
         "🧪 Witness Lab",
         "⚡ Channel Simulator",
         "🔭 GBS & Quantum ML"],
        label_visibility="visible",
    )

    st.sidebar.markdown("<hr style='border-color:#1a2d5a;margin:14px 0'>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#334155;text-align:center;line-height:1.8">
    DEPS: streamlit · plotly · numpy · pandas<br>
    DATA: pre-computed (QuTiP + SF)<br>
    © 2025 m25iqt013
    </div>
    """, unsafe_allow_html=True)
    return page

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STATE EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
STATE_INFO = {
    "Fock |n⟩": {
        "emoji": "🎯",
        "color": "#f472b6",
        "eq": "|n⟩  —  energy eigenstate  â†â|n⟩ = n|n⟩",
        "desc": "Fock states have exactly **n** photons — perfect number certainty, maximum phase uncertainty. They are deeply non-classical: their Wigner function goes **negative**, a hallmark of quantum weirdness.",
        "classical": False,
    },
    "Coherent |α⟩": {
        "emoji": "💡",
        "color": "#fbbf24",
        "eq": "|α⟩ = e^{-|α|²/2} Σ (αⁿ/√n!) |n⟩",
        "desc": "Coherent states are the **most classical** quantum states — laser light! Their Wigner function is a Gaussian, always positive. Minimum uncertainty: ΔxΔp = 1/2.",
        "classical": True,
    },
    "Squeezed |r,φ⟩": {
        "emoji": "🔧",
        "color": "#6366f1",
        "eq": "|r,φ⟩ = S(ξ)|0⟩,  ξ = r·e^{iφ}",
        "desc": "Squeezing trades noise from one quadrature into the other. Below the **shot-noise limit** in x → noise in p amplified. Used in gravitational-wave detectors (LIGO).",
        "classical": False,
    },
    "Thermal ρ_th": {
        "emoji": "🌡️",
        "color": "#fb923c",
        "eq": "ρ_th = Σ (n̄ⁿ/(1+n̄)ⁿ⁺¹) |n⟩⟨n|",
        "desc": "Thermal (blackbody) radiation. Mixed state with super-Poissonian photon statistics (Mandel Q > 0). The Wigner function is a broad Gaussian — classical, but noisy.",
        "classical": True,
    },
    "Cat State": {
        "emoji": "🐱",
        "color": "#a3e635",
        "eq": "|cat⟩ = N(|α⟩ ± |−α⟩)  (Schrödinger's cat)",
        "desc": "Superposition of two coherent states — macroscopic quantum superposition. Shows spectacular **interference fringes** in the Wigner function, proving non-classicality.",
        "classical": False,
    },
    "Displaced-Squeezed": {
        "emoji": "🌀",
        "color": "#22d3ee",
        "eq": "|α,r⟩ = D(α)S(r)|0⟩",
        "desc": "A squeezed state shifted in phase space. Combines displacement D(α) and squeezing S(r). Used in CV quantum key distribution (CV-QKD) protocols.",
        "classical": False,
    },
    "GKP State": {
        "emoji": "🛡️",
        "color": "#c084fc",
        "eq": "|GKP⟩ ∝ Σ_n e^{-δ²n²} D(n√π)|0⟩",
        "desc": "Gottesman-Kitaev-Preskill code states: a superposition of displaced vacua forming a grid in phase space. The go-to state for **photonic quantum error correction**.",
        "classical": False,
    },
}

def page_state_explorer():
    if not _check("states"): return
    SD = DATA["states"]
    xvec = SD["xvec"]

    st.markdown("""
    <div class="page-banner">
        <h1>🔬 State Explorer</h1>
        <p>Interactively explore all 8 continuous-variable quantum states.</p>
        <p>Each state shows its Wigner function W(x,p), Husimi Q function, density matrix, and quantum metrics.</p>
        <span class="tag">Wigner function</span><span class="tag">Husimi Q</span>
        <span class="tag">Density matrix</span><span class="tag">Photon statistics</span>
        <span class="tag">8 quantum states</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Control panel ──
    col_ctrl, col_main = st.columns([1, 3.2], gap="medium")

    with col_ctrl:
        st.markdown("### ⚙️ Controls")
        state_type = st.selectbox("**Quantum State**", list(STATE_INFO.keys()))
        info = STATE_INFO[state_type]

        st.markdown(f"""
        <div style="background:rgba({
            '99,102,241' if not info['classical'] else '251,191,36'
        },0.08);border:1px solid rgba({
            '99,102,241' if not info['classical'] else '251,191,36'
        },0.25);border-radius:10px;padding:12px 14px;margin:10px 0">
        <div style="font-size:1.4rem">{info['emoji']}</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#94a3b8;margin:6px 0">
            {'⚡ NON-CLASSICAL' if not info['classical'] else '☀️ CLASSICAL'}</div>
        <div style="font-size:0.8rem;color:#cbd5e1;line-height:1.5">{info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""<div class="eq-card">{info['eq']}</div>""", unsafe_allow_html=True)

        # State-specific parameters (presets only — data already computed)
        st.markdown("**Choose preset:**")
        if state_type == "Fock |n⟩":
            preset = st.selectbox("Photon number n", [0,1,2,3,5])
            key = preset; group = "fock"
        elif state_type == "Coherent |α⟩":
            preset = st.selectbox("α (Re, Im)", ["0,0","1,0","2,0","1,1","2,2","-2,0","0,2"])
            key = preset; group = "coherent"
        elif state_type == "Squeezed |r,φ⟩":
            preset = st.selectbox("(r, φ)", ["0.5,0","1.0,0","1.5,0","2.0,0","1.0,1.57"])
            key = preset; group = "squeezed"
        elif state_type == "Thermal ρ_th":
            preset = st.selectbox("n̄ (mean photons)", [0.5,1,2,5,10])
            key = preset; group = "thermal"
        elif state_type == "Cat State":
            preset = st.selectbox("Cat type", ["even (|α⟩+|−α⟩)","odd (|α⟩−|−α⟩)","even α=2","odd α=2"])
            key = {"even (|α⟩+|−α⟩)":"even","odd (|α⟩−|−α⟩)":"odd","even α=2":"even_2","odd α=2":"odd_2"}[preset]
            group = "cat"
        elif state_type == "Displaced-Squeezed":
            preset = st.selectbox("(α, r, φ)", ["(1+i, 0.5, 0)","(2, 1.0, 0)","(1+2i, 0.8, π/2)"])
            key = {"(1+i, 0.5, 0)":"(1+1j),0.5,0",
                   "(2, 1.0, 0)":"(2+0j),1.0,0",
                   "(1+2i, 0.8, π/2)":"(1+2j),0.8,1.57"}[preset]
            group = "displaced_squeezed"
        else:  # GKP
            preset = st.selectbox("(δ, peaks)", ["δ=0.3, 3 peaks","δ=0.5, 3 peaks"])
            key = "0.3,3" if "0.3" in preset else "0.5,3"
            group = "gkp"

        view_3d = st.checkbox("🏔️ 3-D Wigner", value=False)
        show_dm = st.checkbox("🔢 Show density matrix", value=True)

    # ── Load state ──
    try:
        state_d = SD[group][key]
    except KeyError:
        keys = list(SD[group].keys())
        state_d = SD[group][keys[0]]

    m   = state_d['metrics']
    wnv = state_d.get('wnv', 0.0)

    with col_main:
        metrics_row(m, wnv)
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["🌊 Wigner W(x,p)", "🌀 Husimi Q(α)", "📊 Photon dist.", "📐 Density matrix"])

        with tab1:
            if view_3d:
                st.plotly_chart(fig_wigner_3d(state_d['W'], xvec, title=f"Wigner — {state_type}"), use_container_width=True)
            else:
                st.plotly_chart(fig_wigner(state_d['W'], xvec, title=f"Wigner — {state_type}"), use_container_width=True)

            neg_str = f"{wnv:.5f}" if wnv > 0.001 else "≈ 0"
            # Physics-correct non-classicality: WNV>0 OR state is intrinsically non-classical (squeezed, cat, DS, GKP, Fock)
            is_nonclassical = not info.get('classical', True)
            if wnv > 0.001:
                nc_text = "This state is <b>non-classical</b> — Wigner function goes negative. ⚡"
            elif is_nonclassical:
                nc_text = ("This state is <b>non-classical</b> (quantum). "
                           "No Wigner negativity for this preset, but non-classicality is present "
                           "(e.g. squeezed noise below shot-noise, quantum coherence, or sub-Poissonian statistics).")
            else:
                nc_text = "This state is <b>classical / Gaussian</b> — Wigner function is always positive."
            st.markdown(f"""
            <div class="insight-card">
                <div class="title">💡 What you're seeing</div>
                <p>The Wigner function W(x,p) is a quasi-probability distribution in phase space.
                <b>Red regions are negative</b> — impossible for classical states.
                Wigner Negativity Volume = <b>{neg_str}</b>.
                {nc_text}</p>
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.plotly_chart(fig_husimi(state_d['Q'], xvec, title=f"Husimi Q — {state_type}"), use_container_width=True)
            st.markdown("""
            <div class="insight-card">
                <div class="title">💡 Husimi Q-function</div>
                <p>Q(α) = ⟨α|ρ|α⟩/π — always non-negative (unlike Wigner).
                Brighter regions show where the state "lives" in phase space.
                The Q-function is smoother and easier to measure but loses some quantum information.</p>
            </div>""", unsafe_allow_html=True)

        with tab3:
            st.plotly_chart(fig_photon_dist(m['probs'], m['mean_n'],
                title=f"Photon number distribution P(n) — {state_type}"), use_container_width=True)
            mq = m.get('mandel_Q')
            if mq is not None:
                regime = "sub-Poissonian (non-classical)" if mq < 0 else ("super-Poissonian" if mq > 0 else "Poissonian (coherent)")
                st.markdown(f"""
                <div class="insight-card">
                    <div class="title">💡 Photon Statistics</div>
                    <p>Mandel Q = {mq:.4f} → <b>{regime}</b>.
                    Yellow line = Poisson reference (coherent state).
                    Q &lt; 0 is a quantum signature: fewer photon-number fluctuations than a laser.</p>
                </div>""", unsafe_allow_html=True)

        with tab4:
            if show_dm and 'rho' in state_d:
                st.plotly_chart(fig_density_matrix(state_d['rho'], title=f"— {state_type}"), use_container_width=True)
                st.markdown("""
                <div class="insight-card">
                    <div class="title">💡 Density matrix ρ</div>
                    <p>ρ encodes everything about the quantum state. The diagonal gives P(n).
                    Off-diagonal elements (coherences) are the signature of superposition.
                    Pure states have Tr(ρ²)=1; mixed states have Tr(ρ²) &lt; 1.</p>
                </div>""", unsafe_allow_html=True)
            elif 'rho' not in state_d:
                st.info("Density matrix not stored for this state type (reduce memory). Shown in Fock and Coherent states.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PHASE SPACE ZOO
# ════════════════════════════════════════════════════════════════════════════════
def page_phase_space_zoo():
    if not _check("states"): return
    SD = DATA["states"]
    xvec = SD["xvec"]

    st.markdown("""
    <div class="page-banner">
        <h1>🌌 Phase Space Zoo</h1>
        <p>Side-by-side comparison of Wigner, Husimi Q, and P-function for all quantum states.</p>
        <p>Spot the non-classical signatures at a glance.</p>
        <span class="tag">Side-by-side comparison</span><span class="tag">All 8 states</span>
        <span class="tag">Non-classicality</span>
    </div>
    """, unsafe_allow_html=True)

    rep = st.radio("**Representation**", ["Wigner W(x,p)", "Husimi Q(α)"], horizontal=True)

    # Build a 2×4 grid of all states
    state_configs = [
        ("fock",              0,      "Fock |0⟩"),
        ("fock",              2,      "Fock |2⟩"),
        ("coherent",         "2,0",   "Coherent |2⟩"),
        ("squeezed",         "1.0,0", "Squeezed r=1"),
        ("thermal",           1,      "Thermal n̄=1"),
        ("cat",              "even",  "Cat (even)"),
        ("displaced_squeezed","(1+1j),0.5,0", "Disp-Sq"),
        ("gkp",              "0.3,3", "GKP δ=0.3"),
    ]

    # Physics-correct non-classicality per state group (not just WNV>0)
    _NC_GROUPS = {"fock", "squeezed", "cat", "displaced_squeezed", "gkp"}

    cols = st.columns(4)
    for idx, (group, key, label) in enumerate(state_configs):
        try:
            sd = SD[group][key]
        except KeyError:
            keys = list(SD[group].keys())
            sd = SD[group][keys[0]]

        wnv = sd.get('wnv', 0.0)
        is_nonclassical = (group in _NC_GROUPS) or (wnv > 0.001)
        data_arr = sd['W'] if "Wigner" in rep else sd['Q']
        cs  = W_CS if "Wigner" in rep else Q_CS

        W_np = np.array(data_arr)
        xv   = np.array(xvec)
        wmin, wmax = _wigner_range(W_np) if "Wigner" in rep else (float(W_np.min()), float(W_np.max()))

        fig = go.Figure(go.Heatmap(
            z=W_np, x=xv, y=xv, colorscale=cs, zmin=wmin, zmax=wmax,
            showscale=False, hovertemplate=f"{label}<br>x=%{{x:.1f}}  p=%{{y:.1f}}<br>val=%{{z:.3f}}<extra></extra>",
        ))
        if "Wigner" in rep:
            fig.add_trace(go.Contour(z=W_np, x=xv, y=xv,
                contours=dict(start=0,end=0,size=1,coloring="none"),
                line=dict(color="rgba(255,255,255,0.5)",width=1), showscale=False))
        badge = "⚡ Non-classical" if is_nonclassical else "☀️ Classical / Gaussian"
        fig.update_layout(
            paper_bgcolor="#020817", plot_bgcolor="#060e24",
            font=dict(color="#e2e8f0", size=9),
            xaxis=dict(gridcolor="#0b1635", title="x/Re(α)", title_font=dict(size=9), tickfont=dict(size=7)),
            yaxis=dict(gridcolor="#0b1635", title="p/Im(α)", title_font=dict(size=9), tickfont=dict(size=7)),
            title=dict(text=f"<b>{label}</b><br><sup>{badge}</sup>", font=dict(size=11,color="#e2e8f0"), x=0.5),
            margin=dict(l=32,r=8,t=50,b=30), height=260,
        )
        cols[idx % 4].plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### 📊 Non-classicality Scorecard")

    rows = []
    labels_full = ["Fock |0⟩","Fock |2⟩","Coherent |2⟩","Squeezed r=1","Thermal n̄=1","Cat (even)","Disp-Sq","GKP δ=0.3"]
    for (group,key,label),lf in zip(state_configs, labels_full):
        try:
            sd = SD[group][key]
        except KeyError:
            sd = list(SD[group].values())[0]
        m   = sd['metrics']
        wnv = sd.get('wnv', 0.0)
        is_nc = (group in _NC_GROUPS) or (wnv > 0.001)
        rows.append({
            "State": lf,
            "Purity": m.get('purity','—'),
            "Entropy S(ρ)": m.get('entropy','—'),
            "Mandel Q": m.get('mandel_Q','—'),
            "WNV": round(wnv,5),
            "Non-classical": "✅" if is_nc else "❌",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="insight-card" style="margin-top:12px">
        <div class="title">💡 Reading the table</div>
        <p><b>WNV</b> (Wigner Negativity Volume) > 0 is a rigorous proof of non-classicality.
        <b>Mandel Q &lt; 0</b> = sub-Poissonian photon statistics (another quantum signature).
        <b>Purity = 1</b> means pure state; <b>Entropy = 0</b> means no classical uncertainty.</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WITNESS LAB
# ════════════════════════════════════════════════════════════════════════════════
def page_witness_lab():
    if not _check("states"): return
    SD = DATA["states"]

    st.markdown("""
    <div class="page-banner">
        <h1>🧪 Witness Lab</h1>
        <p>Live quantum metrics and non-classicality witnesses across all states.</p>
        <p>Compare purity, entropy, Mandel Q, Wigner negativity, and Heisenberg uncertainty.</p>
        <span class="tag">Quantum witnesses</span><span class="tag">Metrics dashboard</span>
        <span class="tag">Heisenberg uncertainty</span>
    </div>
    """, unsafe_allow_html=True)

    _NC_GROUPS_W = {"fock", "squeezed", "cat", "displaced_squeezed", "gkp"}
    all_states = []
    STATE_KEYS = [
        ("fock",0,"Fock |0⟩"),("fock",1,"Fock |1⟩"),("fock",2,"Fock |2⟩"),("fock",3,"Fock |3⟩"),
        ("coherent","0,0","Coherent |0⟩"),("coherent","1,0","Coherent |1⟩"),("coherent","2,0","Coherent |2⟩"),
        ("squeezed","0.5,0","Squeezed r=0.5"),("squeezed","1.0,0","Squeezed r=1"),("squeezed","2.0,0","Squeezed r=2"),
        ("thermal",0.5,"Thermal n̄=0.5"),("thermal",1,"Thermal n̄=1"),("thermal",5,"Thermal n̄=5"),
        ("cat","even","Cat even α=1.5"),("cat","odd","Cat odd α=1.5"),("cat","even_2","Cat even α=2"),
        ("displaced_squeezed","(1+1j),0.5,0","Disp-Sq (1+i,0.5)"),
        ("gkp","0.3,3","GKP δ=0.3"),
    ]
    for group, key, label in STATE_KEYS:
        try:
            sd = SD[group][key]
        except KeyError:
            continue
        m   = sd['metrics']
        wnv = sd.get('wnv', 0.0)
        is_nc = (group in _NC_GROUPS_W) or (wnv > 0.001)
        all_states.append({
            "State": label, "⟨n⟩": m['mean_n'], "Purity": m['purity'],
            "Entropy": m['entropy'], "Mandel Q": m.get('mandel_Q', float('nan')),
            "Δx": m['delta_x'], "Δp": m['delta_p'], "ΔxΔp": m['heis_prod'],
            "WNV": round(wnv, 5), "Non-classical": is_nc,
        })

    df = pd.DataFrame(all_states)

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Full table", "🌊 WNV comparison", "🔭 Purity vs Entropy", "⚖️ Heisenberg"])

    with tab1:
        # Color-coded dataframe
        def color_nc(val):
            return "color: #a3e635; font-weight:700" if val else "color: #64748b"
        st.dataframe(
            df.style.map(color_nc, subset=["Non-classical"])
                    .format({"⟨n⟩":"{:.3f}","Purity":"{:.4f}","Entropy":"{:.4f}",
                              "Mandel Q": lambda v: f"{v:.4f}" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "—",
                              "Δx":"{:.4f}","Δp":"{:.4f}","ΔxΔp":"{:.4f}","WNV":"{:.5f}"}),
            use_container_width=True, height=480,
        )

    with tab2:
        fig = go.Figure()
        colors_nc = ["#f472b6" if row["Non-classical"] else "#fbbf24" for _, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df["State"], y=df["WNV"], marker_color=colors_nc,
            text=[f"{v:.4f}" for v in df["WNV"]], textposition="outside",
            textfont=dict(size=8), name="WNV",
        ))
        fig.add_hline(y=0, line_color="white", line_dash="dash", line_width=1)
        fig.update_layout(**_LAY, height=420,
            title=dict(text="<b>Wigner Negativity Volume</b> — pink=non-classical, gold=classical",
                       font=dict(size=13,color="#6366f1"), x=0.5),
            xaxis_tickangle=-40, xaxis_title="", yaxis_title="WNV")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-card">
            <div class="title">💡 Wigner Negativity Volume (WNV)</div>
            <p>WNV = ∫|W(x,p)|dxdp − 1.  Any WNV > 0 proves the state cannot be described classically.
            Fock states, cat states, and GKP states all have WNV > 0.
            Coherent and thermal states have WNV = 0.</p>
        </div>""", unsafe_allow_html=True)

    with tab3:
        fig = go.Figure()
        for nc, color, label in [(True,"#f472b6","Non-classical"),(False,"#fbbf24","Classical")]:
            mask = df["Non-classical"]==nc
            sub  = df[mask]
            fig.add_trace(go.Scatter(
                x=sub["Purity"], y=sub["Entropy"], mode="markers+text",
                marker=dict(color=color, size=12, symbol="diamond" if nc else "circle",
                            line=dict(color="white",width=0.5)),
                text=sub["State"], textposition="top center",
                textfont=dict(size=7), name=label,
            ))
        fig.update_layout(**_LAY, height=430,
            title=dict(text="<b>Purity Tr(ρ²) vs von Neumann Entropy S(ρ)</b>",
                       font=dict(size=13,color="#6366f1"), x=0.5),
            xaxis_title="Purity Tr(ρ²)", yaxis_title="Entropy S(ρ) [bits]")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="insight-card">
            <div class="title">💡 Purity vs Entropy</div>
            <p>Pure states (Purity=1, Entropy=0) sit top-left. Thermal states are the most mixed.
            For pure states: Purity = 1 always. For mixed states: Purity &lt; 1 and Entropy > 0.
            Note: non-classicality ≠ purity — a pure coherent state is classical!</p>
        </div>""", unsafe_allow_html=True)

    with tab4:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["State"], y=df["ΔxΔp"],
            marker=dict(color=df["ΔxΔp"], colorscale="Viridis", showscale=True,
                        colorbar=dict(title="ΔxΔp", thickness=10, len=0.7)),
            text=[f"{v:.4f}" for v in df["ΔxΔp"]], textposition="outside", textfont=dict(size=7),
        ))
        fig.add_hline(y=0.5, line_color="#a3e635", line_dash="dash",
                      annotation_text="Heisenberg limit: ΔxΔp = ½",
                      annotation_font=dict(color="#a3e635", size=10))
        fig.update_layout(**_LAY, height=420,
            title=dict(text="<b>Heisenberg Uncertainty Product Δx·Δp</b>",
                       font=dict(size=13,color="#6366f1"), x=0.5),
            xaxis_tickangle=-40, xaxis_title="", yaxis_title="Δx · Δp")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="eq-card">
        Heisenberg Uncertainty Principle:  Δx · Δp ≥ ½<br>
        Minimum uncertainty states (Δx·Δp = ½): Coherent, Squeezed (along squeezed direction)<br>
        Thermal/mixed states: Δx·Δp > ½  (excess classical noise)
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CHANNEL SIMULATOR
# ════════════════════════════════════════════════════════════════════════════════
def page_channel_simulator():
    if not _check("channels"): return
    CD = DATA["channels"]
    xvec = CD["xvec"]

    st.markdown("""
    <div class="page-banner">
        <h1>⚡ Channel Simulator</h1>
        <p>Watch a coherent state evolve through displacement D(α), squeezing S(r), phase shifts R(φ), and loss channels.</p>
        <p>Each operation is pre-computed with the full Lindblad master equation.</p>
        <span class="tag">Displacement D(α)</span><span class="tag">Squeezing S(r)</span>
        <span class="tag">Phase shift R(φ)</span><span class="tag">Loss channel (Lindblad)</span>
    </div>
    """, unsafe_allow_html=True)

    channel = st.radio("**Select Channel**",
        ["💫 Displacement D(α)", "🔧 Squeezing S(r)", "🔄 Phase shift R(φ)", "📉 Loss (Lindblad)"],
        horizontal=True)

    tab1, tab2 = st.tabs(["🌊 Wigner evolution", "📊 Metrics evolution"])

    if "Displacement" in channel:
        series = CD["displacement_sweep"]
        param  = [s["alpha_re"] for s in series]
        param_label = "α (real part)"
        title_fn = lambda v: f"D({v:.1f}) — Displacement"
        x_label = "Displacement α"
    elif "Squeezing" in channel:
        series = CD["squeezing_sweep"]
        param  = [s["r"] for s in series]
        param_label = "Squeezing r"
        title_fn = lambda v: f"S({v:.1f}) — Squeezed coherent"
        x_label = "Squeezing parameter r"
    elif "Phase" in channel:
        series = CD["phase_sweep"]
        param  = [round(s["phi"],2) for s in series]
        param_label = "Phase φ"
        title_fn = lambda v: f"R({v:.2f}rad) — Phase rotated"
        x_label = "Phase shift φ (rad)"
    else:
        series = CD["loss_sweep"]
        param  = [s["gamma_t"] for s in series]
        param_label = "γt (loss)"
        title_fn = lambda v: f"Loss γt={v:.1f}"
        x_label = "γt (loss parameter)"

    with tab1:
        idx = st.slider(f"**{param_label}**", 0, len(series)-1, 0,
                        format=f"step %d/{len(series)-1}")
        s = series[idx]
        pval = param[idx]
        st.plotly_chart(fig_wigner(s["W"], xvec, title=title_fn(pval), height=440), use_container_width=True)

        # Show 3 snapshots side-by-side
        st.markdown("**Evolution snapshots:**")
        snap_idx = [0, len(series)//2, len(series)-1]
        snap_cols = st.columns(3)
        for ci, si in enumerate(snap_idx):
            sv = series[si]
            pv = param[si]
            W_np = np.array(sv["W"], dtype=float)
            xv   = np.array(xvec, dtype=float)
            if W_np.ndim != 2:
                continue
            wmin, wmax = _wigner_range(W_np)
            fig = go.Figure(go.Heatmap(z=W_np, x=xv, y=xv, colorscale=W_CS, zmin=wmin, zmax=wmax, showscale=False))
            fig.update_layout(
                paper_bgcolor="#020817", plot_bgcolor="#060e24",
                xaxis=dict(gridcolor="#0b1635", tickfont=dict(size=7)),
                yaxis=dict(gridcolor="#0b1635", tickfont=dict(size=7)),
                title=dict(text=f"<b>{title_fn(pv)}</b>", font=dict(size=10,color="#22d3ee"), x=0.5),
                margin=dict(l=24,r=8,t=36,b=20), height=220,
            )
            snap_cols[ci].plotly_chart(fig, use_container_width=True)

    with tab2:
        purities  = [s["metrics"]["purity"]  for s in series]
        entropies = [s["metrics"]["entropy"]  for s in series]
        mean_ns   = [s["metrics"]["mean_n"]   for s in series]

        fig = make_subplots(1, 3, subplot_titles=["Purity Tr(ρ²)","von Neumann Entropy","⟨n⟩ Mean photons"])
        for col, (ys, color, name) in enumerate([
            (purities,  "#6366f1", "Purity"),
            (entropies, "#f472b6", "Entropy"),
            (mean_ns,   "#22d3ee", "⟨n⟩"),
        ], 1):
            fig.add_trace(go.Scatter(x=param, y=ys, mode="lines+markers",
                line=dict(color=color, width=2.5), marker=dict(size=5), name=name), row=1, col=col)

        fig.update_layout(**_LAY, height=340,
            title=dict(text=f"<b>Metrics vs {param_label}</b>", font=dict(size=13,color="#6366f1"), x=0.5))
        st.plotly_chart(fig, use_container_width=True)

    # Educational box
    channel_theory = {
        "Displacement": (
            "D(α) = exp(αa† − α*a)",
            "Displacement shifts the state rigidly in phase space. |⟨n⟩| increases with |α|. "
            "Purity is unchanged — a pure state stays pure. Used in CV teleportation correction."
        ),
        "Squeezing": (
            "S(r) = exp(r(a² − a†²)/2)",
            "Squeezing stretches the Wigner function in one direction and compresses the other. "
            "The area (and purity) is conserved. Below-shot-noise noise is the key resource for LIGO."
        ),
        "Phase": (
            "R(φ) = exp(iφ a†a)",
            "Phase rotation spins the Wigner function in phase space. Purity is perfectly conserved. "
            "A full 2π rotation returns the state to itself."
        ),
        "Loss": (
            "dρ/dt = γ(aρa† − ½a†aρ − ½ρa†a)",
            "The loss channel is the most physically relevant: every photon absorbed by the environment "
            "increases entropy and decreases purity. A coherent state remains coherent but its amplitude decays."
        ),
    }
    key_ch = [k for k in channel_theory if k.lower() in channel.lower()][0]
    eq, desc = channel_theory[key_ch]
    st.markdown(f"""
    <div class="eq-card" style="margin-top:16px">
        <b>Theory:</b>  {eq}
    </div>
    <div class="insight-card">
        <div class="title">💡 Physical intuition</div>
        <p>{desc}</p>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — GBS & QUANTUM ML
# ════════════════════════════════════════════════════════════════════════════════
def page_gbs_sampler():
    if not _check("gbs"): return
    GD = DATA["gbs"]

    st.markdown("""
    <div class="page-banner">
        <h1>🔭 Gaussian Boson Sampling & CV-QML</h1>
        <p>Xanadu-grade: GBS circuits, hafnian sampling, photon-number distributions, and CV quantum machine learning.</p>
        <p>Data from real Strawberry Fields simulation (or analytic computation if SF unavailable).</p>
        <span class="tag">Strawberry Fields</span><span class="tag">Hafnian (#P-hard)</span>
        <span class="tag">PennyLane CV-QML</span><span class="tag">GBS advantage</span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🔭 GBS Circuit", "📊 Photon statistics", "∑ Hafnian", "🤖 CV-QML"])

    # Pick dataset
    avail = {k:v for k,v in GD.items() if isinstance(v,dict) and 'mean_photons' in v}
    mode_options = list(avail.keys())

    with tab1:
        st.markdown("### Gaussian Boson Sampling Circuit")
        if mode_options:
            sel_key = st.selectbox("**GBS configuration**", mode_options,
                                    format_func=lambda k: f"{'SF' if 'sf' in k else 'Analytic'} — {GD[k]['N_modes']} modes, r≈{GD[k]['r_vals'][0]:.2f}")
            gbs = GD[sel_key]
            source_tag = "🍓 Strawberry Fields (real simulation)" if gbs.get('source')=='strawberryfields' else "📐 Analytic GBS (exact Gaussian computation)"
            st.markdown(f"<span class='state-badge badge-classical' style='margin-bottom:10px;display:inline-block'>{source_tag}</span>", unsafe_allow_html=True)

            # Circuit diagram via plotly
            N = gbs['N_modes']
            fig_circ = go.Figure()
            # Mode wires
            for i in range(N):
                fig_circ.add_shape(type="line", x0=0, x1=4.5, y0=i, y1=i,
                                    line=dict(color="#1a2d5a", width=2))
                fig_circ.add_annotation(x=-0.2, y=i, text=f"|0⟩", showarrow=False,
                                         font=dict(color="#22d3ee", size=11, family="Space Mono"))
                # Squeezing gate
                r_i = gbs['r_vals'][i] if i < len(gbs['r_vals']) else gbs['r_vals'][0]
                fig_circ.add_shape(type="rect", x0=0.3, x1=1.2, y0=i-0.28, y1=i+0.28,
                    line=dict(color="#6366f1", width=2), fillcolor="rgba(99,102,241,0.2)")
                fig_circ.add_annotation(x=0.75, y=i, text=f"S({r_i:.1f})", showarrow=False,
                    font=dict(color="#c7d2fe", size=9, family="Space Mono"))
                # Interferometer block
                fig_circ.add_annotation(x=4.8, y=i, text=f"n̄={gbs['mean_photons'][i]:.2f}", showarrow=False,
                    font=dict(color="#fbbf24", size=10, family="Space Mono"))
            # Interferometer box
            fig_circ.add_shape(type="rect", x0=1.5, x1=3.5, y0=-0.5, y1=N-0.5,
                line=dict(color="#22d3ee", width=2), fillcolor="rgba(34,211,238,0.06)")
            fig_circ.add_annotation(x=2.5, y=(N-1)/2, text="Interferometer U", showarrow=False,
                font=dict(color="#22d3ee", size=12, family="Space Mono"))
            # Measurement
            for i in range(N):
                fig_circ.add_shape(type="rect", x0=3.7, x1=4.5, y0=i-0.28, y1=i+0.28,
                    line=dict(color="#f472b6",width=2), fillcolor="rgba(244,114,182,0.1)")
                fig_circ.add_annotation(x=4.1, y=i, text="⟨n⟩", showarrow=False,
                    font=dict(color="#f9a8d4", size=9, family="Space Mono"))
            fig_circ.update_layout(
                paper_bgcolor="#020817", plot_bgcolor="#060e24",
                xaxis=dict(visible=False, range=[-0.5, 6]),
                yaxis=dict(visible=False, range=[-0.8, N+0.1]),
                height=max(240, N*55+40),
                title=dict(text=f"<b>GBS Circuit — {N} modes</b>", font=dict(size=13,color="#6366f1"), x=0.5),
                margin=dict(l=10,r=10,t=50,b=10),
            )
            st.plotly_chart(fig_circ, use_container_width=True)

            # Wigner functions per mode
            st.markdown("**Wigner functions per mode (after interferometer):**")
            xvec_sf = np.array(gbs['xvec'])
            n_show  = min(N, 4)
            w_cols  = st.columns(n_show)
            for i in range(n_show):
                W_i  = np.array(gbs['wigners'][i if isinstance(list(gbs['wigners'].keys())[0], int) else str(i)])
                wmin, wmax = _wigner_range(W_i)
                fig_wi = go.Figure(go.Heatmap(z=W_i, x=xvec_sf, y=xvec_sf, colorscale=W_CS,
                    zmin=wmin, zmax=wmax, showscale=False))
                fig_wi.update_layout(paper_bgcolor="#020817", plot_bgcolor="#060e24",
                    xaxis=dict(gridcolor="#0b1635", tickfont=dict(size=7)),
                    yaxis=dict(gridcolor="#0b1635", tickfont=dict(size=7)),
                    title=dict(text=f"<b>Mode {i}</b>", font=dict(size=10,color="#22d3ee"), x=0.5),
                    margin=dict(l=20,r=6,t=34,b=16), height=200)
                w_cols[i].plotly_chart(fig_wi, use_container_width=True)

        st.markdown("""
        <div class="eq-card">
        GBS = N squeezed modes → random interferometer → photon-number detection<br>
        Quantum advantage: sampling from this distribution is #P-hard classically!<br>
        Applications: molecular vibronic spectra, graph optimization, drug discovery
        </div>""", unsafe_allow_html=True)

    with tab2:
        if mode_options:
            gbs = GD[mode_options[0]]
            # Mean photons per mode
            fig_mn = go.Figure(go.Bar(
                x=[f"Mode {i}" for i in range(len(gbs['mean_photons']))],
                y=[float(v) for v in gbs['mean_photons']],
                marker_color="#6366f1", opacity=0.85,
                text=[f"{v:.3f}" for v in gbs['mean_photons']], textposition="outside",
            ))
            fig_mn.update_layout(**_LAY, height=300,
                title=dict(text="<b>Mean photon number ⟨n⟩ per mode</b>", font=dict(size=12,color="#6366f1"), x=0.5),
                xaxis_title="Mode", yaxis_title="⟨n⟩")
            st.plotly_chart(fig_mn, use_container_width=True)

            # Single-mode photon distribution
            pn = np.array(gbs['photon_dist']); ns = np.array(gbs['photon_ns'])
            fig_pn = go.Figure(go.Bar(x=ns, y=pn, marker_color="#f472b6", opacity=0.85, name="P(n)"))
            nbar = gbs['mean_photons'][0]
            if nbar > 0:
                poi = (float(nbar)/(1+float(nbar)))**ns/(1+float(nbar))
                fig_pn.add_trace(go.Scatter(x=ns, y=poi, mode="lines", name="Thermal ref",
                    line=dict(color="#fbbf24", dash="dot", width=2)))
            fig_pn.update_layout(**_LAY, height=300,
                title=dict(text="<b>Mode 0 — Photon number distribution P(n)</b>", font=dict(size=12,color="#f472b6"), x=0.5),
                xaxis_title="n", yaxis_title="P(n)")
            st.plotly_chart(fig_pn, use_container_width=True)

            st.markdown("""
            <div class="insight-card">
                <div class="title">💡 GBS photon statistics</div>
                <p>Each squeezed mode after the interferometer follows a thermal-like distribution with n̄ = sinh²(r).
                The <b>joint</b> distribution across all modes is what's hard to sample — its probability is proportional to |Haf(A_S)|².</p>
            </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("### Hafnian — the #P-hard function at the heart of GBS")
        st.markdown("""
        <div class="eq-card">
        Haf(A) = Σ_{perfect matchings M} Π_{(i,j)∈M} A_{ij}<br>
        GBS probability:  P(S) = |Haf(A_S)|² / (s₁!·s₂!·… · √det(σ_Q))<br>
        Computing Haf(A) for large n is #P-hard — requires exponential classical time.
        </div>""", unsafe_allow_html=True)

        c_haf1, c_haf2 = st.columns(2)
        with c_haf1:
            st.markdown("**Verification table:**")
            df_haf = pd.DataFrame(GD.get('hafnian_table', []))
            if not df_haf.empty:
                st.dataframe(df_haf, use_container_width=True, hide_index=True)

        with c_haf2:
            # Scaling plot
            if mode_options:
                gbs0 = GD[mode_options[0]]
                n_arr  = np.array(gbs0['hafnian_n'])
                t_ryser = np.array(gbs0['hafnian_ops'])
                fig_haf = go.Figure(go.Scatter(x=n_arr, y=t_ryser, mode="lines+markers",
                    line=dict(color="#6366f1", width=2.5), marker=dict(size=6, color="#f472b6"), name="Ryser O(2ⁿn²)"))
                fig_haf.update_layout(**_LAY, height=300, yaxis_type="log",
                    title=dict(text="<b>Hafnian complexity scaling</b>", font=dict(size=12,color="#6366f1"), x=0.5),
                    xaxis_title="n (matrix size)", yaxis_title="Operations (log scale)")
                st.plotly_chart(fig_haf, use_container_width=True)

    with tab4:
        st.markdown("### Continuous-Variable Quantum Machine Learning")
        st.markdown("""
        <div class="eq-card">
        CV QNode:  U(θ) = D(α)·S(r)·R(φ)  acting on |0⟩<br>
        Expectation:  ⟨X⟩(θ) = √2·r·cos(θ)  (displacement after squeezing)<br>
        Parameter-shift rule:  ∂⟨O⟩/∂θ = ½[⟨O⟩_{θ+π/2} − ⟨O⟩_{θ-π/2}]
        </div>""", unsafe_allow_html=True)

        theta = np.array(GD['qml_theta'])
        exp_x = np.array(GD['qml_exp_x'])
        grad  = np.array(GD['qml_gradient'])

        fig_qml = make_subplots(1, 2, subplot_titles=["⟨X̂⟩ vs θ", "∂⟨X̂⟩/∂θ (gradient)"])
        fig_qml.add_trace(go.Scatter(x=theta, y=exp_x, mode="lines",
            line=dict(color="#6366f1", width=2.5), name="⟨X⟩"), row=1, col=1)
        fig_qml.add_trace(go.Scatter(x=theta, y=grad, mode="lines",
            line=dict(color="#22d3ee", width=2.5), name="gradient"), row=1, col=2)
        fig_qml.update_layout(**_LAY, height=320,
            title=dict(text="<b>CV QNode — Parameter landscape</b>", font=dict(size=13,color="#6366f1"), x=0.5))
        st.plotly_chart(fig_qml, use_container_width=True)

        steps = np.array(GD['qml_steps']); loss = np.array(GD['qml_loss'])
        fig_tr = go.Figure(go.Scatter(x=steps, y=loss, mode="lines",
            line=dict(color="#f472b6", width=2.5),
            fill="tozeroy", fillcolor="rgba(244,114,182,0.08)"))
        fig_tr.update_layout(**_LAY, height=280, yaxis_type="log",
            title=dict(text="<b>CV-QNN Training Loss (Adam optimizer)</b>", font=dict(size=12,color="#f472b6"), x=0.5),
            xaxis_title="Training step", yaxis_title="MSE Loss (log scale)")
        st.plotly_chart(fig_tr, use_container_width=True)

        st.markdown("""
        <div class="insight-card">
            <div class="title">💡 CV-QML — Why it matters</div>
            <p>Continuous-variable quantum neural networks use <b>Gaussian gates</b> (displacement, squeezing, rotation)
            and <b>non-Gaussian gates</b> (Kerr) as trainable layers.
            The parameter-shift rule enables exact quantum gradients on real hardware.
            Xanadu's Borealis processor demonstrated quantum advantage using GBS in 2022.</p>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ════════════════════════════════════════════════════════════════════════════════
def main():
    page = render_sidebar()
    if   "State Explorer"    in page: page_state_explorer()
    elif "Phase Space Zoo"   in page: page_phase_space_zoo()
    elif "Witness Lab"       in page: page_witness_lab()
    elif "Channel Simulator" in page: page_channel_simulator()
    elif "GBS"               in page: page_gbs_sampler()

if __name__ == "__main__":
    main()
