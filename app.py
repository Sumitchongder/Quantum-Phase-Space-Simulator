"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   CV Quantum Information — Streamlit Dashboard                               ║
║   IIT Jodhpur · Course: Continuous-Variable Quantum Information              ║
║   Author : m25iqt013                                                         ║
║   5 Pages: State Explorer · Phase Space Zoo · Witness Lab ·                  ║
║             Channel Simulator · GBS Sampler                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ════════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════════════
import sys, os, math, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

import qutip as qt
from qutip import (
    basis, ket2dm, coherent, coherent_dm, thermal_dm,
    expect, destroy, num, displace, squeeze,
    tensor, qeye, mesolve, Options,
)
from scipy.linalg import sqrtm, expm, eigvalsh
from scipy.special import eval_laguerre
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

# ════════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be FIRST streamlit call)
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CV Quantum Dashboard — m25iqt013",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/XanaduAI/strawberryfields",
        "Report a bug": None,
        "About": "CV Quantum Information Dashboard · IIT Jodhpur · m25iqt013",
    },
)

# ════════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  — Xanadu-inspired dark theme
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Root background ───────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #0a0a1a; }
[data-testid="stSidebar"]          { background: #0d0d24; border-right: 1px solid #2d1b69; }
[data-testid="stHeader"]           { background: transparent; }

/* ── Typography ────────────────────────────────────────────── */
html, body, [class*="css"]         { color: #e2e8f0; font-family: 'Inter', 'DejaVu Sans', sans-serif; }
h1 { font-size: 2.0rem; font-weight: 800; letter-spacing: 1px; }
h2 { font-size: 1.4rem; font-weight: 700; color: #a78bfa; }
h3 { font-size: 1.1rem; font-weight: 600; color: #7dd3fc; }

/* ── Sidebar labels ────────────────────────────────────────── */
[data-testid="stSidebar"] label    { color: #c4b5fd !important; font-weight: 600; }
[data-testid="stSidebar"] .stSlider > label { color: #7dd3fc !important; }

/* ── Metric cards ──────────────────────────────────────────── */
[data-testid="metric-container"]   {
    background: linear-gradient(135deg, #12124a, #1a0a2e);
    border: 1px solid #2d1b69;
    border-radius: 12px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"]      { color: #a78bfa; font-weight: 800; font-size: 1.4rem; }
[data-testid="stMetricLabel"]      { color: #94a3b8; font-size: 0.75rem; }

/* ── Buttons ───────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #4c1d95, #7c3aed);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 8px 20px;
    transition: all 0.2s;
}
.stButton > button:hover { background: linear-gradient(135deg, #6d28d9, #9333ea); transform: translateY(-1px); }

/* ── Selectbox / sliders ───────────────────────────────────── */
.stSelectbox > div > div { background: #12124a; border: 1px solid #2d1b69; border-radius: 8px; color: #e2e8f0; }
.stSlider > div > div > div { background: #7c3aed; }

/* ── Tabs ──────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]  { background: #0d0d24; border-bottom: 2px solid #2d1b69; }
.stTabs [data-baseweb="tab"]       { color: #94a3b8; font-weight: 600; padding: 10px 20px; }
.stTabs [aria-selected="true"]     { color: #a78bfa !important; border-bottom: 2px solid #a78bfa; }

/* ── Info / warning boxes ──────────────────────────────────── */
.stInfo    { background: #0f172a; border-left: 4px solid #60a5fa; color: #bfdbfe; border-radius: 0 8px 8px 0; }
.stSuccess { background: #052e16; border-left: 4px solid #34d399; color: #a7f3d0; border-radius: 0 8px 8px 0; }
.stWarning { background: #1c1400; border-left: 4px solid #fbbf24; color: #fde68a; border-radius: 0 8px 8px 0; }

/* ── DataFrames ────────────────────────────────────────────── */
[data-testid="stDataFrame"] { background: #0d0d24; border-radius: 10px; border: 1px solid #2d1b69; }

/* ── Divider ───────────────────────────────────────────────── */
hr { border-color: #2d1b69; margin: 1.2rem 0; }

/* ── Banner header ─────────────────────────────────────────── */
.banner {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0a1a2e 100%);
    border: 1px solid #4c1d95;
    border-radius: 16px;
    padding: 28px 36px;
    text-align: center;
    margin-bottom: 1.5rem;
}
.banner h1 { color: #a78bfa; margin: 0 0 6px 0; }
.banner p  { color: #94a3b8; margin: 4px 0; font-size: 0.92rem; }

/* ── Equation box ──────────────────────────────────────────── */
.eq-box {
    background: #0f0f2a;
    border: 1px solid #2d1b69;
    border-radius: 10px;
    padding: 14px 18px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    color: #c4b5fd;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PLOTLY DARK TEMPLATE  — used by every figure
# ════════════════════════════════════════════════════════════════════════════════
PLOTLY_TEMPLATE = {
    "layout": go.Layout(
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0d0d24",
        font=dict(color="#e2e8f0", family="Inter, DejaVu Sans"),
        xaxis=dict(gridcolor="#1e1e3f", zerolinecolor="#2d1b69",
                   title_font=dict(size=12), tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1e1e3f", zerolinecolor="#2d1b69",
                   title_font=dict(size=12), tickfont=dict(size=10)),
        legend=dict(bgcolor="#0d0d24", bordercolor="#2d1b69", borderwidth=1),
        margin=dict(l=50, r=30, t=50, b=40),
        colorway=["#a78bfa","#22d3ee","#f472b6","#34d399","#fbbf24","#60a5fa","#fb923c"],
    )
}

WIGNER_CS  = [[0.0,"#1e3a5f"],[0.5,"#0a0a1a"],[1.0,"#e879f9"]]
WIGNER_CS2 = [[0.0,"#0c1445"],[0.5,"#0a0a1a"],[1.0,"#38bdf8"]]
HUSIMI_CS  = [[0.0,"#0a0a1a"],[0.5,"#7c3aed"],[1.0,"#fbbf24"]]

# ════════════════════════════════════════════════════════════════════════════════
# CACHED CORE FUNCTIONS  (st.cache_data / st.cache_resource)
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def _compute_wigner(rho_arr: np.ndarray, xvec: np.ndarray) -> np.ndarray:
    rho = qt.Qobj(rho_arr)
    rho.dims = [[rho_arr.shape[0]], [rho_arr.shape[0]]]
    return np.array(qt.wigner(rho, xvec, xvec, g=2))

@st.cache_data(ttl=600, show_spinner=False)
def _compute_husimi(rho_arr: np.ndarray, xvec: np.ndarray) -> np.ndarray:
    rho = qt.Qobj(rho_arr)
    rho.dims = [[rho_arr.shape[0]], [rho_arr.shape[0]]]
    result = qt.qfunc(rho, xvec, xvec)
    return np.array(result[0] if isinstance(result, tuple) else result)

@st.cache_data(ttl=600, show_spinner=False)
def _state_metrics(rho_arr: np.ndarray) -> dict:
    rho  = qt.Qobj(rho_arr); rho.dims = [[rho_arr.shape[0]],[rho_arr.shape[0]]]
    dim  = rho.shape[0]
    a    = destroy(dim); n_op = num(dim)
    x_op = (a + a.dag()) / np.sqrt(2)
    p_op = 1j * (a.dag() - a) / np.sqrt(2)
    mn   = float(expect(n_op, rho).real)
    mn2  = float(expect(n_op*n_op, rho).real)
    vn   = mn2 - mn**2
    pur  = float((rho*rho).tr().real)
    ent  = float(qt.entropy_vn(rho, base=2))
    mq   = (vn - mn)/mn if mn > 1e-10 else float('nan')
    mx   = float(expect(x_op, rho).real)
    mp   = float(expect(p_op, rho).real)
    vx   = float(expect(x_op*x_op, rho).real) - mx**2
    vp   = float(expect(p_op*p_op, rho).real) - mp**2
    xpc  = float(expect((x_op*p_op + p_op*x_op)/2, rho).real) - mx*mp
    dx   = math.sqrt(max(vx, 0)); dp = math.sqrt(max(vp, 0))
    probs= np.array([float(rho[n,n].real) for n in range(min(dim,30))])
    return dict(mean_n=round(mn,5), var_n=round(vn,5), purity=round(pur,6),
                entropy=round(ent,6), mandel_Q=round(mq,5) if not math.isnan(mq) else None,
                mean_x=round(mx,5), mean_p=round(mp,5), delta_x=round(dx,6),
                delta_p=round(dp,6), heis_prod=round(dx*dp,6),
                var_x=round(vx,6), var_p=round(vp,6), xpc=round(xpc,6), probs=probs)

def wigner_neg_vol(W, xvec):
    dx = xvec[1]-xvec[0]
    return float(np.sum(np.abs(W))*dx**2 - 1.0)

def safe_cmap_norm(W):
    wmin, wmax = float(W.min()), float(W.max())
    if wmin >= 0: wmin = -1e-9
    if wmax <= 0: wmax =  1e-9
    return wmin, wmax

# ── State builders (hashable inputs → cached) ──────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def build_fock(n: int, dim: int) -> np.ndarray:
    return ket2dm(basis(dim, n)).full()

@st.cache_data(ttl=600, show_spinner=False)
def build_coherent(alpha_re: float, alpha_im: float, dim: int) -> np.ndarray:
    return coherent_dm(dim, alpha_re + 1j*alpha_im).full()

@st.cache_data(ttl=600, show_spinner=False)
def build_squeezed(r: float, phi: float, dim: int) -> np.ndarray:
    xi  = r * np.exp(1j*phi)
    psi = squeeze(dim, xi) * basis(dim, 0)
    return ket2dm(psi).full()

@st.cache_data(ttl=600, show_spinner=False)
def build_thermal(nbar: float, dim: int) -> np.ndarray:
    return thermal_dm(dim, nbar).full()

@st.cache_data(ttl=600, show_spinner=False)
def build_cat(alpha_re: float, alpha_im: float, sign: int, dim: int) -> np.ndarray:
    alpha = alpha_re + 1j*alpha_im
    psi   = (coherent(dim, alpha) + sign*coherent(dim, -alpha)).unit()
    return ket2dm(psi).full()

@st.cache_data(ttl=600, show_spinner=False)
def build_displaced_squeezed(alpha_re: float, alpha_im: float, r: float, phi: float, dim: int) -> np.ndarray:
    xi  = r*np.exp(1j*phi); alpha = alpha_re + 1j*alpha_im
    psi = displace(dim, alpha) * squeeze(dim, xi) * basis(dim, 0)
    return ket2dm(psi).full()

@st.cache_data(ttl=600, show_spinner=False)
def build_tmsv(r: float, dim: int) -> np.ndarray:
    lam = np.tanh(r); norm = np.sqrt(1-lam**2)
    psi = sum(norm*(lam**n)*tensor(basis(dim,n), basis(dim,n)) for n in range(dim))
    return ket2dm(psi).full()

@st.cache_data(ttl=600, show_spinner=False)
def build_gkp(delta: float, n_peaks: int, dim: int) -> np.ndarray:
    psi = sum(np.exp(-delta**2*n**2)*displace(dim, n*np.sqrt(np.pi))*basis(dim,0)
              for n in range(-n_peaks, n_peaks+1))
    return ket2dm(psi.unit()).full()

@st.cache_data(ttl=600, show_spinner=False)
def apply_loss(rho_arr: np.ndarray, gamma_t: float) -> np.ndarray:
    rho = qt.Qobj(rho_arr); rho.dims = [[rho_arr.shape[0]],[rho_arr.shape[0]]]
    dim = rho.shape[0]; a = destroy(dim)
    H   = qt.Qobj(np.zeros((dim,dim)))
    opts = Options(nsteps=15000)
    res  = mesolve(H, rho, np.linspace(0, gamma_t, 12), [np.sqrt(1.0)*a], [], options=opts)
    return res.states[-1].full()

@st.cache_data(ttl=600, show_spinner=False)
def apply_displacement_op(rho_arr: np.ndarray, alpha_re: float, alpha_im: float) -> np.ndarray:
    rho = qt.Qobj(rho_arr); rho.dims = [[rho_arr.shape[0]],[rho_arr.shape[0]]]
    D   = displace(rho.shape[0], alpha_re+1j*alpha_im)
    return (D*rho*D.dag()).full()

@st.cache_data(ttl=600, show_spinner=False)
def apply_squeeze_op(rho_arr: np.ndarray, r: float, phi: float) -> np.ndarray:
    rho = qt.Qobj(rho_arr); rho.dims = [[rho_arr.shape[0]],[rho_arr.shape[0]]]
    S   = squeeze(rho.shape[0], r*np.exp(1j*phi))
    return (S*rho*S.dag()).full()

@st.cache_data(ttl=600, show_spinner=False)
def apply_phase_shift_op(rho_arr: np.ndarray, phi: float) -> np.ndarray:
    rho = qt.Qobj(rho_arr); rho.dims = [[rho_arr.shape[0]],[rho_arr.shape[0]]]
    n_op = num(rho.shape[0]); R = (-1j*phi*n_op).expm()
    return (R*rho*R.dag()).full()

# ════════════════════════════════════════════════════════════════════════════════
# PLOTLY FIGURE BUILDERS
# ════════════════════════════════════════════════════════════════════════════════

def fig_wigner(W, xvec, title="W(x,p)", cs=WIGNER_CS, height=420):
    wmin, wmax = safe_cmap_norm(W)
    fig = go.Figure(go.Heatmap(
        z=W, x=xvec, y=xvec,
        colorscale=cs, zmin=wmin, zmax=wmax,
        colorbar=dict(title="W", len=0.8, thickness=14,
                      tickfont=dict(color="#e2e8f0",size=10)),
        hoverongaps=False,
        hovertemplate="x=%{x:.2f}  p=%{y:.2f}  W=%{z:.4f}<extra></extra>",
    ))
    # Zero contour
    fig.add_contour(z=W, x=xvec, y=xvec,
                    contours=dict(start=0, end=0, size=1, coloring="none",
                                  showlabels=False),
                    line=dict(color="rgba(255,255,255,0.5)", width=1.2),
                    showscale=False)
    fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                      title=dict(text=f"<b>{title}</b>", font=dict(size=14,color="#a78bfa"), x=0.5),
                      xaxis_title="x (position)", yaxis_title="p (momentum)",
                      height=height)
    return fig

def fig_husimi(Q, xvec, title="Q(α)", height=420):
    fig = go.Figure(go.Heatmap(
        z=Q, x=xvec, y=xvec,
        colorscale=HUSIMI_CS,
        colorbar=dict(title="Q", len=0.8, thickness=14,
                      tickfont=dict(color="#e2e8f0",size=10)),
        hovertemplate="x=%{x:.2f}  p=%{y:.2f}  Q=%{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                      title=dict(text=f"<b>{title}</b>", font=dict(size=14,color="#22d3ee"), x=0.5),
                      xaxis_title="Re(α)", yaxis_title="Im(α)", height=height)
    return fig

def fig_density_matrix(rho_arr, display_dim=15, title="ρ", height=380):
    mat  = np.real(rho_arr[:display_dim, :display_dim])
    wmax = max(abs(mat).max(), 1e-6); wmin = -wmax
    ticks = list(range(0, display_dim, max(1, display_dim//6)))
    fig  = make_subplots(1, 2, subplot_titles=["Re(ρ)", "Im(ρ)"],
                          horizontal_spacing=0.08)
    for col_i, (data, clr) in enumerate([(np.real(rho_arr[:display_dim,:display_dim]),
                                           "#a78bfa"),
                                          (np.imag(rho_arr[:display_dim,:display_dim]),
                                           "#22d3ee")], 1):
        vm = max(abs(data).max(),1e-6)
        fig.add_trace(go.Heatmap(z=data, colorscale="RdBu_r",
                                  zmin=-vm, zmax=vm,
                                  showscale=(col_i==1),
                                  hovertemplate=f"n=%{{y}}  m=%{{x}}  val=%{{z:.4f}}<extra></extra>"),
                      row=1, col=col_i)
    fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                      title=dict(text=f"<b>Density Matrix {title}</b>",
                                  font=dict(size=13,color="#a78bfa"), x=0.5),
                      height=height)
    return fig

def fig_photon_dist(probs, mean_n, title="P(n)", height=340):
    k    = np.arange(len(probs))
    poi  = np.exp(-mean_n) * mean_n**k / np.array([math.factorial(int(ki)) for ki in k])
    fig  = go.Figure()
    fig.add_trace(go.Bar(x=k, y=probs, name="P(n)",
                          marker_color="#a78bfa", marker_line_color="#c4b5fd",
                          marker_line_width=0.5, opacity=0.85))
    if mean_n > 0.01:
        fig.add_trace(go.Scatter(x=k, y=poi, mode="lines+markers",
                                  name="Poisson", line=dict(color="#fbbf24",width=2),
                                  marker=dict(size=4)))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                      title=dict(text=f"<b>{title}</b>", font=dict(size=13,color="#a78bfa"), x=0.5),
                      xaxis_title="Photon number n", yaxis_title="P(n)",
                      barmode="overlay", height=height)
    return fig

def fig_wigner_3d(W, xvec, title="W(x,p) — 3D", height=480):
    wabs = max(abs(W.min()), abs(W.max()), 1e-6)
    fig  = go.Figure(go.Surface(
        z=W, x=xvec, y=xvec,
        colorscale=WIGNER_CS, cmin=-wabs, cmax=wabs,
        showscale=True, opacity=0.94,
        colorbar=dict(title="W", len=0.7, thickness=14,
                      tickfont=dict(color="#e2e8f0",size=9)),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
        hovertemplate="x=%{x:.2f}  p=%{y:.2f}  W=%{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"],
                      title=dict(text=f"<b>{title}</b>", font=dict(size=14,color="#a78bfa"), x=0.5),
                      scene=dict(
                          xaxis=dict(title="x", gridcolor="#1e1e3f", backgroundcolor="#0a0a1a"),
                          yaxis=dict(title="p", gridcolor="#1e1e3f", backgroundcolor="#0a0a1a"),
                          zaxis=dict(title="W(x,p)", gridcolor="#1e1e3f", backgroundcolor="#0a0a1a"),
                          bgcolor="#0a0a1a",
                          camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
                      ), height=height)
    return fig

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — global state builder
# ════════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.markdown("""
    <div style="text-align:center;padding:14px 0 8px">
        <div style="font-size:2.2rem">⚛️</div>
        <div style="color:#a78bfa;font-weight:800;font-size:1.1rem;letter-spacing:1px">
            CV Quantum Dashboard
        </div>
        <div style="color:#64748b;font-size:0.75rem;margin-top:4px">
            IIT Jodhpur · m25iqt013
        </div>
    </div>
    <hr style="border-color:#2d1b69;margin:10px 0 16px">
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "📑 **Navigate**",
        ["🔬 Page 1 — State Explorer",
         "🌌 Page 2 — Phase Space Zoo",
         "🧪 Page 3 — Witness Lab",
         "⚡ Page 4 — Channel Simulator",
         "🔭 Page 5 — GBS Sampler"],
        label_visibility="visible",
    )

    st.sidebar.markdown("<hr style='border-color:#2d1b69;margin:14px 0'>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='color:#64748b;font-size:0.72rem;text-align:center'>"
                        "QuTiP 5.2 · NumPy 2.x · Plotly · Streamlit<br>"
                        "© 2025 m25iqt013 — IIT Jodhpur</div>", unsafe_allow_html=True)
    return page

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STATE EXPLORER
# ════════════════════════════════════════════════════════════════════════════════

def page_state_explorer():
    st.markdown("""
    <div class="banner">
        <h1>🔬 State Explorer</h1>
        <p>Interactive phase-space explorer for all 8 CV quantum states</p>
        <p>Live Wigner function · Husimi Q · Density matrix · Photon statistics · Metrics</p>
    </div>""", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    col_ctrl, col_main = st.columns([1, 3], gap="medium")
    with col_ctrl:
        st.markdown("### ⚙️ State Controls")
        state_type = st.selectbox("**Quantum State**", [
            "Fock |n⟩", "Coherent |α⟩", "Squeezed |r,φ⟩",
            "Thermal ρ_th", "Cat State", "Displaced–Squeezed",
            "TMSV (EPR)", "GKP Grid State",
        ])
        dim = st.slider("**Hilbert dim**", 20, 60, 35, 5)
        xres= st.slider("**Grid resolution**", 80, 220, 140, 20)
        xvec= np.linspace(-6, 6, xres)

        st.markdown("---")
        # Dynamic parameter widgets
        rho_arr = None
        params  = {}

        if state_type == "Fock |n⟩":
            params["n"] = st.slider("**n** (photon number)", 0, min(dim-1,15), 3)
            st.markdown("""<div class='eq-box'>|n⟩⟨n|<br>Purity = 1 · Non-Gaussian</div>""", unsafe_allow_html=True)
            rho_arr = build_fock(params["n"], dim)

        elif state_type == "Coherent |α⟩":
            params["re"] = st.slider("**Re(α)**", -3.0, 3.0, 1.5, 0.1)
            params["im"] = st.slider("**Im(α)**", -3.0, 3.0, 0.5, 0.1)
            st.markdown(f"""<div class='eq-box'>|α={params['re']:+.2f}{params['im']:+.2f}i⟩<br>Gaussian · Purity=1</div>""", unsafe_allow_html=True)
            rho_arr = build_coherent(params["re"], params["im"], dim)

        elif state_type == "Squeezed |r,φ⟩":
            params["r"]   = st.slider("**r** (squeezing)", 0.0, 2.0, 0.8, 0.05)
            params["phi"] = st.slider("**φ** (angle, π units)", 0.0, 2.0, 0.0, 0.1)
            phi_rad = params["phi"] * math.pi
            dB = round(10*math.log10(math.exp(2*params["r"])),2) if params["r"]>0 else 0
            st.markdown(f"""<div class='eq-box'>S(ξ)|0⟩  ξ=r·e^iφ<br>r={params['r']:.2f} ({dB} dB) · φ={params['phi']:.1f}π</div>""", unsafe_allow_html=True)
            rho_arr = build_squeezed(params["r"], phi_rad, dim)

        elif state_type == "Thermal ρ_th":
            params["nbar"] = st.slider("**n̄** (mean photon number)", 0.1, 5.0, 1.0, 0.1)
            st.markdown(f"""<div class='eq-box'>ρ_th = Σ [n̄ⁿ/(n̄+1)^(n+1)] |n⟩⟨n|<br>n̄={params['nbar']:.2f} · Mixed state</div>""", unsafe_allow_html=True)
            rho_arr = build_thermal(params["nbar"], dim)

        elif state_type == "Cat State":
            params["re"]   = st.slider("**Re(α)**", 0.5, 3.5, 2.0, 0.1)
            params["im"]   = st.slider("**Im(α)**", -2.0, 2.0, 0.0, 0.1)
            params["sign"] = st.radio("**Parity**", ["+1 (even)", "-1 (odd)"])
            sign = +1 if "+1" in params["sign"] else -1
            label = "even" if sign>0 else "odd"
            st.markdown(f"""<div class='eq-box'>N±(|α⟩ ± |-α⟩)  [{label}]<br>α={params['re']:+.2f}{params['im']:+.2f}i</div>""", unsafe_allow_html=True)
            rho_arr = build_cat(params["re"], params["im"], sign, dim)

        elif state_type == "Displaced–Squeezed":
            params["a_re"] = st.slider("**Re(α)**", -3.0, 3.0, 2.0, 0.1)
            params["a_im"] = st.slider("**Im(α)**", -3.0, 3.0, 0.0, 0.1)
            params["r"]    = st.slider("**r** (squeezing)", 0.0, 2.0, 0.8, 0.05)
            params["phi"]  = st.slider("**φ** (angle, π)", 0.0, 2.0, 0.0, 0.1)
            st.markdown(f"""<div class='eq-box'>D(α)S(ξ)|0⟩ · α={params['a_re']:+.2f}{params['a_im']:+.2f}i<br>r={params['r']:.2f} · φ={params['phi']:.1f}π</div>""", unsafe_allow_html=True)
            rho_arr = build_displaced_squeezed(params["a_re"], params["a_im"],
                                               params["r"], params["phi"]*math.pi, dim)

        elif state_type == "TMSV (EPR)":
            params["r"]    = st.slider("**r** (two-mode squeezing)", 0.1, 2.0, 1.0, 0.05)
            dim2 = min(dim, 20)
            EN   = 2*params["r"]/math.log(2)
            SE   = math.cosh(params["r"])**2*math.log2(math.cosh(params["r"])**2) - \
                   math.sinh(params["r"])**2*math.log2(math.sinh(params["r"])**2+1e-15)
            st.markdown(f"""<div class='eq-box'>S₂(r)|00⟩ · r={params['r']:.2f}<br>E_N={EN:.3f} · S={SE:.3f} bits</div>""", unsafe_allow_html=True)
            rho_arr = build_tmsv(params["r"], dim2)
            # Show reduced state
            rho_2m = qt.Qobj(rho_arr)
            rho_2m.dims = [[dim2,dim2],[dim2,dim2]]
            rho_arr = rho_2m.ptrace(0).full()

        elif state_type == "GKP Grid State":
            params["delta"]  = st.slider("**δ** (envelope width)", 0.1, 0.6, 0.25, 0.05)
            params["npeaks"] = st.slider("**n_max** (peaks)", 2, 6, 4, 1)
            st.markdown(f"""<div class='eq-box'>Σ exp(-δ²n²) D(2n√π)|0⟩<br>δ={params['delta']:.2f} · Non-Gaussian grid</div>""", unsafe_allow_html=True)
            rho_arr = build_gkp(params["delta"], params["npeaks"], dim)

        show_3d = st.checkbox("**Show 3D Wigner**", False)
        rep_tabs = st.multiselect("**Representations**",
                                   ["Wigner","Husimi Q","Density Matrix","Photon Dist"],
                                   default=["Wigner","Husimi Q"])

    # ── Main panel ────────────────────────────────────────────────────────────
    with col_main:
        if rho_arr is None:
            st.info("Select a state from the sidebar.")
            return

        with st.spinner("Computing phase space..."):
            W  = _compute_wigner(rho_arr, xvec)
            Q  = _compute_husimi(rho_arr, xvec)
            m  = _state_metrics(rho_arr)
            dW = wigner_neg_vol(W, xvec)

        # ── Metric strip ─────────────────────────────────────────────────────
        mc = st.columns(7)
        mc[0].metric("⟨n⟩",      f"{m['mean_n']:.4f}")
        mc[1].metric("Purity",    f"{m['purity']:.5f}")
        mc[2].metric("Entropy",   f"{m['entropy']:.4f} bits")
        mc[3].metric("Δx",        f"{m['delta_x']:.5f}")
        mc[4].metric("Δp",        f"{m['delta_p']:.5f}")
        mc[5].metric("ΔxΔp",      f"{m['heis_prod']:.5f}")
        mc[6].metric("W_neg δ",   f"{dW:.5f}")

        mq_val = m['mandel_Q']
        mc2 = st.columns(4)
        mc2[0].metric("Mandel Q", f"{mq_val:.4f}" if mq_val is not None else "N/A",
                       help="Q<0: sub-Poissonian (non-classical)")
        mc2[1].metric("Var(n)",   f"{m['var_n']:.5f}")
        mc2[2].metric("⟨x⟩",     f"{m['mean_x']:.4f}")
        mc2[3].metric("⟨p⟩",     f"{m['mean_p']:.4f}")

        # non-classicality badge
        is_nc = dW > 1e-4 or (mq_val is not None and mq_val < -0.01)
        badge = ("🟢 **Non-classical** — Wigner negativity detected" if dW > 1e-4 else
                 "🟡 **Squeezed** — sub-shot-noise quadrature" if m["heis_prod"] < 0.499 else
                 "⚪ **Classical / Gaussian**")
        st.markdown(f"<div style='background:#0f172a;border-left:3px solid #a78bfa;"
                    f"padding:8px 14px;border-radius:0 8px 8px 0;margin:8px 0'>{badge}</div>",
                    unsafe_allow_html=True)

        # ── Representations ───────────────────────────────────────────────────
        if show_3d:
            st.plotly_chart(fig_wigner_3d(W, xvec, title=f"W(x,p) — {state_type}"),
                            use_container_width=True)

        if "Wigner" in rep_tabs and "Husimi Q" in rep_tabs:
            c1, c2 = st.columns(2)
            c1.plotly_chart(fig_wigner(W, xvec, title=f"Wigner — {state_type}"),
                            use_container_width=True)
            c2.plotly_chart(fig_husimi(Q, xvec, title=f"Husimi Q — {state_type}"),
                            use_container_width=True)
        elif "Wigner" in rep_tabs:
            st.plotly_chart(fig_wigner(W, xvec, title=f"Wigner — {state_type}", height=500),
                            use_container_width=True)
        elif "Husimi Q" in rep_tabs:
            st.plotly_chart(fig_husimi(Q, xvec, title=f"Husimi Q — {state_type}", height=500),
                            use_container_width=True)

        if "Density Matrix" in rep_tabs:
            st.plotly_chart(fig_density_matrix(rho_arr, min(20,dim), title=state_type),
                            use_container_width=True)

        if "Photon Dist" in rep_tabs:
            st.plotly_chart(fig_photon_dist(m["probs"], m["mean_n"],
                                            title=f"P(n) — {state_type}"),
                            use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PHASE SPACE ZOO
# ════════════════════════════════════════════════════════════════════════════════

def page_phase_space_zoo():
    st.markdown("""
    <div class="banner">
        <h1>🌌 Phase Space Zoo</h1>
        <p>W · Q · P side-by-side for all 8 quantum states simultaneously</p>
        <p>Toggle representations · Compare non-classicality · Full parameter control</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🌌 Zoo Controls")
        xres   = st.slider("Grid resolution", 60, 160, 90, 10)
        dim_z  = st.slider("Hilbert dim", 20, 50, 30, 5)
        rep    = st.radio("Representation", ["Wigner W(x,p)","Husimi Q(α)","Compare W vs Q"])
        n_fock = st.slider("n (Fock)", 0, 8, 3)
        alpha_z= st.slider("|α| (Coherent/Cat)", 0.5, 3.0, 2.0, 0.1)
        r_sq   = st.slider("r (Squeezed)", 0.1, 2.0, 0.9, 0.05)
        nbar_th= st.slider("n̄ (Thermal)", 0.1, 4.0, 1.0, 0.1)
        r_cat  = st.slider("|α| (Cat)", 0.5, 3.0, 2.0, 0.1)
        r_ds   = st.slider("r (Disp-Sq)", 0.1, 1.5, 0.8, 0.05)
        r_tmsv = st.slider("r (TMSV)", 0.1, 1.5, 0.8, 0.05)
        d_gkp  = st.slider("δ (GKP)", 0.1, 0.5, 0.25, 0.05)

    xvec = np.linspace(-6, 6, xres)
    with st.spinner("Building all states..."):
        states_raw = {
            f"Fock |{n_fock}⟩"    : build_fock(n_fock, dim_z),
            f"Coherent |{alpha_z}⟩": build_coherent(alpha_z, 0.0, dim_z),
            f"Squeezed r={r_sq}"   : build_squeezed(r_sq, 0.0, dim_z),
            f"Thermal n̄={nbar_th}" : build_thermal(nbar_th, dim_z),
            f"Cat |{r_cat}⟩ even"  : build_cat(r_cat, 0.0, +1, dim_z),
            f"Disp-Sq r={r_ds}"    : build_displaced_squeezed(1.5, 0.0, r_ds, 0.0, dim_z),
            "TMSV reduced"         : _tmsv_reduced(r_tmsv, min(dim_z,18)),
            f"GKP δ={d_gkp}"       : build_gkp(d_gkp, 4, dim_z),
        }

        Ws, Qs, mets = {}, {}, {}
        for lbl, arr in states_raw.items():
            Ws[lbl]  = _compute_wigner(arr, xvec)
            Qs[lbl]  = _compute_husimi(arr, xvec)
            mets[lbl]= _state_metrics(arr)

    labels = list(states_raw.keys())

    # ── Grid of plots ─────────────────────────────────────────────────────────
    st.markdown("### Phase Space Representations — All 8 States")
    cols_per_row = 4

    if rep in ["Wigner W(x,p)", "Compare W vs Q"]:
        st.markdown("#### 🌀 Wigner Function W(x,p)")
        rows = [labels[i:i+cols_per_row] for i in range(0,8,cols_per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for col, lbl in zip(cols, row):
                W = Ws[lbl]; neg = wigner_neg_vol(W, xvec)
                fig = fig_wigner(W, xvec, title=f"{lbl}<br>δ={neg:.4f}", height=300)
                col.plotly_chart(fig, use_container_width=True)

    if rep in ["Husimi Q(α)", "Compare W vs Q"]:
        st.markdown("#### 🔶 Husimi Q Function Q(α)")
        rows = [labels[i:i+cols_per_row] for i in range(0,8,cols_per_row)]
        for row in rows:
            cols = st.columns(len(row))
            for col, lbl in zip(cols, row):
                col.plotly_chart(fig_husimi(Qs[lbl], xvec, title=lbl, height=300),
                                 use_container_width=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown("### 📊 Metrics Comparison Table")
    rows_t = []
    for lbl in labels:
        m   = mets[lbl]; W = Ws[lbl]; neg = wigner_neg_vol(W, xvec)
        nc  = "✅ Non-classical" if neg>1e-4 else ("🟡 Squeezed" if m["heis_prod"]<0.499 else "⚪ Classical")
        rows_t.append({
            "State": lbl, "⟨n⟩": m["mean_n"], "Purity": m["purity"],
            "Entropy": m["entropy"], "Δx": m["delta_x"], "Δp": m["delta_p"],
            "ΔxΔp": m["heis_prod"], "W_neg δ": round(neg,5),
            "Mandel Q": m["mandel_Q"] if m["mandel_Q"] is not None else "N/A",
            "Class": nc,
        })
    df = pd.DataFrame(rows_t).set_index("State")
    st.dataframe(df.style.background_gradient(cmap="Purples", subset=["W_neg δ"])
                          .background_gradient(cmap="Blues",   subset=["Purity"])
                          .format({"⟨n⟩":":.4f","Purity":":.5f","Entropy":":.4f",
                                   "Δx":":.5f","Δp":":.5f","ΔxΔp":":.5f","W_neg δ":":.5f"}),
                 use_container_width=True)


def _tmsv_reduced(r, dim2):
    """Helper — TMSV reduced state."""
    arr    = build_tmsv(r, dim2)
    rho_2m = qt.Qobj(arr); rho_2m.dims = [[dim2,dim2],[dim2,dim2]]
    return rho_2m.ptrace(0).full()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WITNESS LAB
# ════════════════════════════════════════════════════════════════════════════════

def page_witness_lab():
    st.markdown("""
    <div class="banner">
        <h1>🧪 Witness Lab</h1>
        <p>Live non-classicality witnesses · All 8 states · All measures simultaneously</p>
        <p>Wigner neg. volume · Mandel Q · Purity · Entropy · Fidelity · Log-negativity · QFI</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🧪 Witness Controls")
        dim_w   = st.slider("Hilbert dim", 20, 50, 30, 5)
        xres_w  = st.slider("Grid resolution", 60, 140, 80, 10)
        alpha_w = st.slider("|α|", 0.5, 3.0, 2.0, 0.1)
        r_w     = st.slider("r (squeezing)", 0.1, 2.0, 1.0, 0.05)
        nbar_w  = st.slider("n̄ (thermal)", 0.1, 4.0, 1.0, 0.1)
        cat_a_w = st.slider("|α| cat", 0.5, 3.0, 2.0, 0.1)

    xvec_w = np.linspace(-6, 6, xres_w)

    with st.spinner("Computing all witnesses..."):
        states_w = {
            "Vacuum |0⟩"          : build_fock(0, dim_w),
            f"Fock |3⟩"           : build_fock(3, dim_w),
            f"Coherent |{alpha_w}⟩": build_coherent(alpha_w, 0.0, dim_w),
            f"Squeezed r={r_w}"   : build_squeezed(r_w, 0.0, dim_w),
            f"Thermal n̄={nbar_w}" : build_thermal(nbar_w, dim_w),
            f"Even Cat |{cat_a_w}⟩":build_cat(cat_a_w, 0.0, +1, dim_w),
            f"Odd Cat |{cat_a_w}⟩" :build_cat(cat_a_w, 0.0, -1, dim_w),
            "Disp-Sq r=0.8"       : build_displaced_squeezed(2.0, 0.0, 0.8, 0.0, dim_w),
        }

        data_rows = []
        W_all     = {}
        for lbl, arr in states_w.items():
            W   = _compute_wigner(arr, xvec_w)
            m   = _state_metrics(arr)
            neg = wigner_neg_vol(W, xvec_w)
            W_all[lbl] = W
            # QFI (approximated via variance of photon number for speed)
            n_op = num(dim_w)
            rho_q= qt.Qobj(arr); rho_q.dims = [[dim_w],[dim_w]]
            mn   = float(expect(n_op, rho_q).real)
            mn2  = float(expect(n_op*n_op, rho_q).real)
            qfi_approx = 4*(mn2 - mn**2)   # 4·Var(n) for phase estimation
            data_rows.append({
                "State": lbl, "W_neg δ": round(neg,6),
                "W_min": round(float(W.min()),6), "W_max": round(float(W.max()),5),
                "Purity": m["purity"], "Entropy": m["entropy"],
                "Δx": m["delta_x"], "Δp": m["delta_p"], "ΔxΔp": m["heis_prod"],
                "Mandel Q": round(m["mandel_Q"],5) if m["mandel_Q"] is not None else float("nan"),
                "⟨n⟩": m["mean_n"], "Var(n)": m["var_n"],
                "QFI≈4Var(n)": round(qfi_approx,4),
                "Non-classical": bool(neg>1e-4),
                "Sub-Poissonian": bool(m["mandel_Q"] is not None and m["mandel_Q"]<-0.01),
                "Squeezed": bool(m["heis_prod"] < 0.499),
            })

    df_w = pd.DataFrame(data_rows).set_index("State")

    # ── Tab layout ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Witness Table", "📈 Bar Charts",
                                        "🌀 Wigner Gallery", "🔢 QFI Analysis"])

    with tab1:
        st.markdown("### Complete Non-Classicality Witness Table")
        def color_nc(val):
            if val is True:  return "background-color:#14532d;color:#86efac"
            if val is False: return "background-color:#1c1c2e;color:#64748b"
            return ""
        numeric_cols = ["W_neg δ","W_min","Purity","Entropy","Δx","Δp","ΔxΔp","Mandel Q","⟨n⟩","Var(n)","QFI≈4Var(n)"]
        bool_cols    = ["Non-classical","Sub-Poissonian","Squeezed"]
        st.dataframe(
            df_w.style
                .background_gradient(cmap="Purples",  subset=["W_neg δ"])
                .background_gradient(cmap="RdYlGn_r", subset=["Purity"])
                .applymap(color_nc, subset=bool_cols)
                .format({c:":.5f" for c in numeric_cols if c in df_w.columns}),
            use_container_width=True, height=340
        )
        st.download_button("⬇️ Download CSV", df_w.to_csv().encode(),
                           "witness_table.csv", "text/csv")

    with tab2:
        st.markdown("### Non-Classicality Witnesses — Bar Charts")
        labels_w = list(states_w.keys())
        witnesses_plot = [
            ("W_neg δ",   "Wigner Negativity Volume δ",     "#a78bfa", "δ=0 → classical"),
            ("Purity",    "Purity Tr(ρ²)",                  "#34d399", "1=pure, 0.5=maximally mixed"),
            ("Entropy",   "von Neumann Entropy (bits)",      "#22d3ee", "0=pure"),
            ("ΔxΔp",      "Heisenberg Product ΔxΔp",         "#f472b6", "0.5=minimum uncertainty"),
            ("Mandel Q",  "Mandel Q (sub-Poissonian<0)",     "#fbbf24", "Q=0 Poissonian"),
            ("QFI≈4Var(n)","Quantum Fisher Info ≈ 4Var(n)",  "#60a5fa", "Higher → more metrological"),
        ]
        rows_b = [witnesses_plot[:3], witnesses_plot[3:]]
        for row_b in rows_b:
            cols_b = st.columns(3)
            for col_b, (metric, title, color, note) in zip(cols_b, row_b):
                vals = df_w[metric].tolist()
                fig  = go.Figure(go.Bar(
                    x=labels_w, y=vals, marker_color=color,
                    marker_line_color="white", marker_line_width=0.4,
                    text=[f"{v:.3f}" for v in vals], textposition="outside",
                    textfont=dict(size=9),
                ))
                fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=280,
                                   title=dict(text=f"<b>{title}</b><br><sup>{note}</sup>",
                                              font=dict(size=11,color=color), x=0.5),
                                   xaxis=dict(tickangle=-30, tickfont=dict(size=8)))
                col_b.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Wigner Functions — All Witness States")
        lbl_list = list(W_all.keys())
        rows_g   = [lbl_list[i:i+4] for i in range(0,8,4)]
        for row_g in rows_g:
            cols_g = st.columns(4)
            for col_g, lbl in zip(cols_g, row_g):
                W   = W_all[lbl]; neg = wigner_neg_vol(W, xvec_w)
                title = f"{lbl}<br>δ={neg:.4f}"
                col_g.plotly_chart(fig_wigner(W, xvec_w, title=title, height=260),
                                   use_container_width=True)

    with tab4:
        st.markdown("### Quantum Fisher Information Analysis")
        st.info("QFI ≈ 4·Var(n) for pure states — exact for coherent, thermal, Fock, cat. "
                "QFI quantifies metrological usefulness for phase estimation (Cramér-Rao bound).")
        r_qfi = np.linspace(0.01, 2.5, 60)
        qfi_coh = 4*r_qfi**2               # F_Q[coherent|r] = 4r² (Fock number noise)
        qfi_sq  = 4*np.sinh(r_qfi)**2      # squeezed vacuum
        fig_qfi = go.Figure()
        for y, name, col in [(qfi_coh,"Coherent (4|α|²)","#22d3ee"),
                              (qfi_sq, "Squeezed vacuum (4sinh²r)","#a78bfa")]:
            fig_qfi.add_trace(go.Scatter(x=r_qfi, y=y, name=name,
                                          line=dict(color=col, width=2.5)))
        fig_qfi.update_layout(**PLOTLY_TEMPLATE["layout"], height=340,
                               title=dict(text="<b>QFI vs Parameter</b>", font=dict(size=13,color="#a78bfa"),x=0.5),
                               xaxis_title="Parameter (|α| or r)", yaxis_title="QFI")
        st.plotly_chart(fig_qfi, use_container_width=True)

        qfi_vals  = df_w["QFI≈4Var(n)"].tolist()
        fig_radar = go.Figure(go.Scatterpolar(
            r=qfi_vals, theta=labels_w, fill="toself",
            line_color="#a78bfa", fillcolor="rgba(167,139,250,0.15)",
        ))
        fig_radar.update_layout(**PLOTLY_TEMPLATE["layout"], height=400,
                                 polar=dict(bgcolor="#0d0d24",
                                            radialaxis=dict(gridcolor="#2d1b69"),
                                            angularaxis=dict(gridcolor="#2d1b69")),
                                 title=dict(text="<b>QFI Radar — All States</b>",
                                            font=dict(size=13,color="#a78bfa"),x=0.5))
        st.plotly_chart(fig_radar, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CHANNEL SIMULATOR
# ════════════════════════════════════════════════════════════════════════════════

def page_channel_simulator():
    st.markdown("""
    <div class="banner">
        <h1>⚡ Channel Simulator</h1>
        <p>Apply quantum channels to any input state · Watch Wigner evolve</p>
        <p>D(α) · S(r) · R(φ) · Loss (Lindblad) · Observe decoherence in real time</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚡ Channel Controls")
        dim_c    = st.slider("Hilbert dim", 20, 50, 30, 5)
        xres_c   = st.slider("Grid resolution", 60, 140, 90, 10)
        st.markdown("**Input State**")
        in_type  = st.selectbox("Input", ["Coherent |α⟩","Fock |n⟩","Squeezed",
                                           "Cat State","Thermal"])
        if in_type == "Coherent |α⟩":
            a_in = st.slider("Re(α)", -3.0, 3.0, 2.0, 0.1)
            rho_in = build_coherent(a_in, 0.0, dim_c)
        elif in_type == "Fock |n⟩":
            n_in = st.slider("n", 0, 8, 3)
            rho_in = build_fock(n_in, dim_c)
        elif in_type == "Squeezed":
            r_in = st.slider("r", 0.0, 2.0, 0.8, 0.05)
            rho_in = build_squeezed(r_in, 0.0, dim_c)
        elif in_type == "Cat State":
            a_cat = st.slider("|α|", 0.5, 3.0, 2.0, 0.1)
            rho_in = build_cat(a_cat, 0.0, +1, dim_c)
        else:
            nb_in = st.slider("n̄", 0.1, 3.0, 1.0, 0.1)
            rho_in = build_thermal(nb_in, dim_c)

        st.markdown("**Channel**")
        channel  = st.selectbox("Channel", [
            "Displacement D(α)", "Squeezing S(r)", "Phase Shift R(φ)",
            "Photon Loss", "Sequential D→S→R",
        ])
        if channel == "Displacement D(α)":
            ch_alpha = st.slider("Re(α)", -3.0, 3.0, 1.0, 0.1)
            ch_ai    = st.slider("Im(α)", -3.0, 3.0, 0.0, 0.1)
        elif channel == "Squeezing S(r)":
            ch_r   = st.slider("r", 0.0, 2.0, 0.5, 0.05)
            ch_phi = st.slider("φ (π)", 0.0, 2.0, 0.0, 0.1)
        elif channel == "Phase Shift R(φ)":
            ch_phi = st.slider("φ (π)", 0.0, 2.0, 0.5, 0.05)
        elif channel == "Photon Loss":
            ch_gt  = st.slider("γt (loss)", 0.0, 3.0, 0.5, 0.05)
        elif channel == "Sequential D→S→R":
            ch_alpha = st.slider("D: Re(α)", -2.0, 2.0, 1.0, 0.1)
            ch_r     = st.slider("S: r",     0.0,  1.5, 0.5, 0.05)
            ch_phi   = st.slider("R: φ (π)", 0.0,  2.0, 0.3, 0.05)

    xvec_c = np.linspace(-6, 6, xres_c)

    with st.spinner("Applying channel..."):
        # Build output state
        if channel == "Displacement D(α)":
            rho_out = apply_displacement_op(rho_in, ch_alpha, ch_ai)
            ch_label= f"D(α={ch_alpha:+.2f}{ch_ai:+.2f}i)"
        elif channel == "Squeezing S(r)":
            rho_out = apply_squeeze_op(rho_in, ch_r, ch_phi*math.pi)
            ch_label= f"S(r={ch_r:.2f}, φ={ch_phi:.1f}π)"
        elif channel == "Phase Shift R(φ)":
            rho_out = apply_phase_shift_op(rho_in, ch_phi*math.pi)
            ch_label= f"R(φ={ch_phi:.2f}π)"
        elif channel == "Photon Loss":
            rho_out = apply_loss(rho_in, ch_gt)
            ch_label= f"Loss(γt={ch_gt:.2f})"
        else:
            tmp1 = apply_displacement_op(rho_in, ch_alpha, 0.0)
            tmp2 = apply_squeeze_op(tmp1, ch_r, 0.0)
            rho_out = apply_phase_shift_op(tmp2, ch_phi*math.pi)
            ch_label= f"D({ch_alpha:.1f})→S({ch_r:.2f})→R({ch_phi:.1f}π)"

        W_in  = _compute_wigner(rho_in,  xvec_c)
        W_out = _compute_wigner(rho_out, xvec_c)
        m_in  = _state_metrics(rho_in)
        m_out = _state_metrics(rho_out)
        neg_in  = wigner_neg_vol(W_in,  xvec_c)
        neg_out = wigner_neg_vol(W_out, xvec_c)

    # ── Layout ──────────────────────────────────────────────────────────────────
    st.markdown(f"### {in_type}  →  **{ch_label}**  →  Output")

    # Metric delta strip
    mc = st.columns(6)
    mc[0].metric("Purity IN",    f"{m_in['purity']:.5f}")
    mc[1].metric("Purity OUT",   f"{m_out['purity']:.5f}",
                  delta=f"{m_out['purity']-m_in['purity']:+.5f}")
    mc[2].metric("Entropy OUT",  f"{m_out['entropy']:.4f} bits",
                  delta=f"{m_out['entropy']-m_in['entropy']:+.4f}")
    mc[3].metric("W_neg IN",     f"{neg_in:.5f}")
    mc[4].metric("W_neg OUT",    f"{neg_out:.5f}",
                  delta=f"{neg_out-neg_in:+.5f}")
    mc[5].metric("ΔxΔp OUT",     f"{m_out['heis_prod']:.5f}",
                  delta=f"{m_out['heis_prod']-m_in['heis_prod']:+.5f}")

    # Before / After Wigner
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_wigner(W_in,  xvec_c, title=f"Input: {in_type}"),
                    use_container_width=True)
    c2.plotly_chart(fig_wigner(W_out, xvec_c, title=f"Output: {ch_label}"),
                    use_container_width=True)

    # W difference
    W_diff = W_out - W_in
    st.plotly_chart(fig_wigner(W_diff, xvec_c,
                               title="ΔW = W_out − W_in (channel effect on phase space)",
                               cs=[[0,"#ef4444"],[0.5,"#0a0a1a"],[1,"#22d3ee"]]),
                    use_container_width=True)

    # ── Photon distribution comparison ────────────────────────────────────────
    st.markdown("### 📊 Photon Distribution: Before vs After")
    k    = np.arange(len(m_in["probs"]))
    fig_p= go.Figure()
    fig_p.add_trace(go.Bar(x=k, y=m_in["probs"],  name="Input",
                            marker_color="#60a5fa", opacity=0.75, width=0.4,
                            offset=-0.2))
    fig_p.add_trace(go.Bar(x=k, y=m_out["probs"], name="Output",
                            marker_color="#f472b6", opacity=0.75, width=0.4,
                            offset=0.2))
    fig_p.update_layout(**PLOTLY_TEMPLATE["layout"], barmode="overlay", height=300,
                         title=dict(text="<b>P(n): Input vs Output</b>",
                                    font=dict(size=13,color="#a78bfa"),x=0.5),
                         xaxis_title="Photon number n", yaxis_title="P(n)")
    st.plotly_chart(fig_p, use_container_width=True)

    # ── Loss evolution sweep ─────────────────────────────────────────────────
    if channel == "Photon Loss":
        st.markdown("### 🔻 Loss Channel Evolution Sweep")
        gt_vals = np.linspace(0, 3.0, 8)
        purity_traj, neg_traj, entropy_traj = [], [], []
        with st.spinner("Computing loss trajectory..."):
            for gt_ in gt_vals:
                r_ = apply_loss(rho_in, float(gt_))
                m_ = _state_metrics(r_)
                W_ = _compute_wigner(r_, xvec_c)
                purity_traj.append(m_["purity"])
                neg_traj.append(wigner_neg_vol(W_, xvec_c))
                entropy_traj.append(m_["entropy"])

        fig_ev = make_subplots(1, 3, subplot_titles=["Purity","W Negativity","Entropy"])
        for col_i, (vals, col, name) in enumerate([
            (purity_traj,  "#34d399", "Purity"),
            (neg_traj,     "#a78bfa", "W_neg"),
            (entropy_traj, "#22d3ee", "Entropy"),
        ], 1):
            fig_ev.add_trace(go.Scatter(x=gt_vals, y=vals, mode="lines+markers",
                                         line=dict(color=col,width=2.5),
                                         marker=dict(size=7), name=name),
                              row=1, col=col_i)
        fig_ev.update_layout(**PLOTLY_TEMPLATE["layout"], height=300, showlegend=False,
                               title=dict(text="<b>Decoherence under Photon Loss</b>",
                                          font=dict(size=13,color="#a78bfa"),x=0.5))
        st.plotly_chart(fig_ev, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — GBS SAMPLER
# ════════════════════════════════════════════════════════════════════════════════

def page_gbs_sampler():
    st.markdown("""
    <div class="banner">
        <h1>🔭 GBS Sampler</h1>
        <p>Gaussian Boson Sampling · Hafnian probabilities · Photon statistics</p>
        <p>Strawberry Fields backend · Thewalrus hafnian · PennyLane CV-QML</p>
    </div>""", unsafe_allow_html=True)

    # Check SF availability
    sf_ok = False
    try:
        import strawberryfields as sf
        from strawberryfields import ops as sf_ops
        sf_ok = True
    except ImportError:
        pass

    tw_ok = False
    try:
        from thewalrus import hafnian
        tw_ok = True
    except ImportError:
        pass

    if not sf_ok:
        st.warning("⚠️ Strawberry Fields not installed. "
                   "Run `pip install strawberryfields thewalrus` on the HPC. "
                   "Showing analytical GBS simulation below.")

    with st.sidebar:
        st.markdown("### 🔭 GBS Controls")
        N_modes  = st.slider("N modes", 2, 6, 4, 1)
        r_gbs    = st.slider("Squeezing r", 0.1, 2.0, 0.8, 0.05)
        asym_r   = st.checkbox("Asymmetric squeezing", False)
        cutoff   = st.slider("Fock cutoff", 4, 10, 6, 1)
        xres_g   = st.slider("Grid resolution", 60, 120, 80, 10)
        dim_g    = 30

    xvec_g = np.linspace(-5, 5, xres_g)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📡 GBS Circuit", "📊 Photon Statistics",
                                       "🔢 Hafnian",      "🤖 CV-QML"])

    with tab1:
        st.markdown("### GBS Circuit Architecture")
        st.markdown("""
        <div class='eq-box'>
        Squeezed vacuum inputs  →  Haar-random interferometer  →  PNR detection<br>
        P(n₁,...,nₘ) = |Haf(A_S)|² / (n₁!···nₘ! · √det(σ_Q))
        </div>""", unsafe_allow_html=True)

        r_vals = ([r_gbs]*N_modes if not asym_r else
                  [r_gbs*(0.5 + 0.5*i/(N_modes-1)) for i in range(N_modes)])

        if sf_ok:
            with st.spinner("Running SF Gaussian backend..."):
                result_sf = _run_sf_gbs(N_modes, r_vals)
            cov   = result_sf["cov"]
            means = result_sf["means"]
            mn_per_mode = result_sf["mean_photons"]

            st.success(f"✅ Strawberry Fields: {N_modes}-mode GBS executed")
            mc = st.columns(N_modes)
            for i, mn in enumerate(mn_per_mode):
                mc[i].metric(f"⟨n_{i}⟩", f"{mn:.4f}")

            # Wigner of output modes
            st.markdown("#### Output Mode Wigner Functions")
            rows_sf = [list(range(N_modes))[i:i+4] for i in range(0,N_modes,4)]
            for row_sf in rows_sf:
                cols_sf = st.columns(len(row_sf))
                for col_sf, mi in zip(cols_sf, row_sf):
                    W_sf = result_sf["wigners"][mi]
                    col_sf.plotly_chart(fig_wigner(W_sf, xvec_g,
                                                    title=f"Mode {mi}<br>⟨n⟩={mn_per_mode[mi]:.4f}",
                                                    height=280),
                                        use_container_width=True)
        else:
            # Analytical fallback
            st.info("SF not available — showing analytical mean photon numbers.")
            mn_vals = [math.sinh(r)**2 for r in r_vals]
            mc = st.columns(N_modes)
            for i, mn in enumerate(mn_vals):
                mc[i].metric(f"⟨n_{i}⟩ analytic", f"{mn:.4f}")

        # Circuit diagram (text art)
        st.markdown("#### Circuit Diagram")
        circuit_lines = []
        for i, r in enumerate(r_vals):
            circuit_lines.append(
                f"Mode {i}: |0⟩ ──── S(r={r:.2f}) ──── [  U  ] ──── PNR"
            )
        circuit_str = "\n".join(circuit_lines)
        st.code(f"""
GBS Circuit ({N_modes} modes):
{'─'*55}
{circuit_str}
{'─'*55}
U = Haar-random {N_modes}×{N_modes} unitary (Clements decomposition)
PNR = photon-number-resolving detector
Output: click pattern (n₀, n₁, ..., n_{N_modes-1})
""", language="")

    with tab2:
        st.markdown("### Photon Number Statistics")

        # Analytical GBS statistics
        r_arr = np.array(r_vals)
        mean_n= np.sinh(r_arr)**2
        var_n = np.sinh(r_arr)**2 * np.cosh(r_arr)**2 * 2  # approx
        mandel_Q_gbs = (var_n - mean_n) / np.maximum(mean_n, 1e-10)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            fig_mn = go.Figure(go.Bar(
                x=[f"Mode {i}" for i in range(N_modes)],
                y=mean_n.tolist(),
                marker_color=["#a78bfa","#22d3ee","#f472b6","#34d399","#fbbf24","#60a5fa"][:N_modes],
                text=[f"{v:.4f}" for v in mean_n], textposition="outside",
            ))
            fig_mn.update_layout(**PLOTLY_TEMPLATE["layout"], height=300,
                                  title=dict(text="<b>⟨n⟩ per Mode (sinh²r)</b>",
                                             font=dict(size=13,color="#a78bfa"),x=0.5),
                                  yaxis_title="⟨n⟩")
            st.plotly_chart(fig_mn, use_container_width=True)

        with col_s2:
            fig_mq = go.Figure(go.Bar(
                x=[f"Mode {i}" for i in range(N_modes)],
                y=mandel_Q_gbs.tolist(),
                marker_color="#fbbf24",
                text=[f"{v:.3f}" for v in mandel_Q_gbs], textposition="outside",
            ))
            fig_mq.add_hline(y=0, line_color="white", line_dash="dash",
                              annotation_text="Poissonian Q=0")
            fig_mq.update_layout(**PLOTLY_TEMPLATE["layout"], height=300,
                                  title=dict(text="<b>Mandel Q (super-Poissonian)</b>",
                                             font=dict(size=13,color="#fbbf24"),x=0.5),
                                  yaxis_title="Q_M")
            st.plotly_chart(fig_mq, use_container_width=True)

        # Mean photon vs squeezing sweep
        r_sweep  = np.linspace(0, 2.5, 100)
        n_sweep  = np.sinh(r_sweep)**2
        dB_sweep = 10*np.log10(np.exp(2*r_sweep)+1e-10)
        fig_sw   = go.Figure()
        fig_sw.add_trace(go.Scatter(x=r_sweep, y=n_sweep, mode="lines",
                                     line=dict(color="#a78bfa",width=2.5),
                                     name="⟨n⟩=sinh²r"))
        fig_sw.add_vline(x=r_gbs, line_color="#fbbf24", line_dash="dash",
                          annotation_text=f"r={r_gbs:.2f}", annotation_font_color="#fbbf24")
        fig_sw.update_layout(**PLOTLY_TEMPLATE["layout"], height=300,
                              title=dict(text="<b>Mean Photon Number vs Squeezing</b>",
                                         font=dict(size=13,color="#a78bfa"),x=0.5),
                              xaxis_title="r", yaxis_title="⟨n⟩")
        st.plotly_chart(fig_sw, use_container_width=True)

    with tab3:
        st.markdown("### Hafnian Computation")
        st.markdown("""
        <div class='eq-box'>
        Haf(A) = Σ_{perfect matchings M} Π_{(i,j)∈M} A_{ij}<br>
        GBS: P(S) = |Haf(A_S)|² / (S! · √det(σ_Q))  · Computing Haf is #P-hard
        </div>""", unsafe_allow_html=True)

        # Demonstrate hafnian on small matrices
        st.markdown("#### Hafnian Examples")
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            st.markdown("**Verification on known matrices:**")
            test_mats = {
                "2×2 Identity": np.eye(2),
                "2×2 Ones":     np.ones((2,2)),
                "4×4 Random":   np.random.RandomState(42).randn(4,4),
            }
            rows_haf = []
            for name_, A_ in test_mats.items():
                A_sym = (A_ + A_.T)/2
                haf_b = _hafnian_brute(A_sym)
                if tw_ok:
                    from thewalrus import hafnian as tw_haf
                    haf_tw = tw_haf(A_sym)
                else:
                    haf_tw = "N/A (install thewalrus)"
                rows_haf.append({"Matrix": name_,
                                   "Brute-force Haf": f"{haf_b:.6f}",
                                   "Thewalrus Haf": f"{float(np.real(haf_tw)):.6f}" if isinstance(haf_tw, (int,float,complex,np.floating)) else haf_tw})
            st.dataframe(pd.DataFrame(rows_haf), use_container_width=True, hide_index=True)

        with col_h2:
            st.markdown("**GBS output probabilities (2-mode):**")
            r2  = 0.8; sq2 = math.sinh(r2); cq2 = math.cosh(r2)
            patterns_2 = [(0,0),(1,0),(0,1),(1,1),(2,0),(2,2)]
            rows_p = []
            for pat in patterns_2:
                n_tot = sum(pat)
                if n_tot == 0:
                    prob = 1/(cq2**2)
                else:
                    # Simple 2-mode analytic prob
                    prob = (math.tanh(r2)**(2*min(pat)))/(cq2**2 * math.factorial(pat[0]) * math.factorial(pat[1]))
                rows_p.append({"Pattern": str(pat), "P(pattern)≈": f"{abs(prob):.8f}"})
            st.dataframe(pd.DataFrame(rows_p), use_container_width=True, hide_index=True)

        # Hardness scaling
        st.markdown("#### Computational Complexity Scaling")
        n_vals  = np.arange(2, 22, 2)
        t_ryser = 2**n_vals * n_vals**2          # Ryser O(2^n · n²)
        t_class = np.exp(n_vals * math.log(10))  # classical (rough)
        fig_cx  = go.Figure()
        fig_cx.add_trace(go.Scatter(x=n_vals, y=t_ryser, mode="lines+markers",
                                     name="Hafnian (Ryser) O(2ⁿn²)",
                                     line=dict(color="#a78bfa",width=2.5)))
        fig_cx.update_layout(**PLOTLY_TEMPLATE["layout"], height=300,
                              title=dict(text="<b>Hafnian Computation Scaling</b>",
                                         font=dict(size=13,color="#a78bfa"),x=0.5),
                              xaxis_title="Matrix size n (photons)", yaxis_title="Operations",
                              yaxis_type="log")
        st.plotly_chart(fig_cx, use_container_width=True)

    with tab4:
        st.markdown("### PennyLane CV-QML")
        pl_ok = False
        try:
            import pennylane as qml
            pl_ok = True
        except ImportError:
            pass

        if not pl_ok:
            st.warning("PennyLane not installed. Run `pip install pennylane pennylane-sf`")

        st.markdown("""
        <div class='eq-box'>
        CV Quantum Kernel: K(x,x') = |⟨0|U†(x)U(x')|0⟩|²<br>
        Parameter-shift rule: ∂⟨O⟩/∂θ = ½[⟨O⟩_{θ+π/2} - ⟨O⟩_{θ-π/2}]
        </div>""", unsafe_allow_html=True)

        # Show gradient landscape analytically
        theta_vals = np.linspace(-math.pi, math.pi, 200)
        # Simulate expectation of X after squeezing + displacement
        r_demo = 0.5; alpha_demo = 1.5
        expect_x = alpha_demo * np.sqrt(2) * np.cos(theta_vals)  # rotated coherent
        grad_x   = -alpha_demo * np.sqrt(2) * np.sin(theta_vals)

        fig_qml = make_subplots(1, 2, subplot_titles=["⟨X⟩ vs θ", "Gradient ∂⟨X⟩/∂θ"])
        fig_qml.add_trace(go.Scatter(x=theta_vals, y=expect_x, mode="lines",
                                      line=dict(color="#a78bfa",width=2.5), name="⟨X⟩"), row=1, col=1)
        fig_qml.add_trace(go.Scatter(x=theta_vals, y=grad_x, mode="lines",
                                      line=dict(color="#22d3ee",width=2.5), name="Gradient"), row=1, col=2)
        fig_qml.update_layout(**PLOTLY_TEMPLATE["layout"], height=320,
                               title=dict(text="<b>CV QNode: Parameter-Shift Landscape</b>",
                                          font=dict(size=13,color="#a78bfa"),x=0.5))
        st.plotly_chart(fig_qml, use_container_width=True)

        # Training curve simulation
        st.markdown("#### Simulated CV-QNN Training (Adam Optimizer)")
        steps = np.arange(120)
        loss  = 2.5 * np.exp(-steps/30) + 0.05 * np.random.RandomState(42).randn(120)*np.exp(-steps/60)
        loss  = np.maximum(loss, 0.001)
        fig_tr= go.Figure(go.Scatter(x=steps, y=loss, mode="lines",
                                      line=dict(color="#f472b6",width=2.5), fill="tozeroy",
                                      fillcolor="rgba(244,114,182,0.1)"))
        fig_tr.update_layout(**PLOTLY_TEMPLATE["layout"], height=280,
                              title=dict(text="<b>CV-QNN Loss Curve (Adam, lr=0.05)</b>",
                                         font=dict(size=13,color="#f472b6"),x=0.5),
                              xaxis_title="Training step", yaxis_title="MSE Loss",
                              yaxis_type="log")
        st.plotly_chart(fig_tr, use_container_width=True)

        st.info("💡 Full PennyLane execution with parameter-shift gradients runs in Notebook 08. "
                "See `08_GBS_SF.ipynb` sections 13–16.")


# ── Helper: cached SF GBS run ──────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def _run_sf_gbs(N_modes: int, r_vals: list) -> dict:
    import strawberryfields as sf
    from strawberryfields import ops as sf_ops
    from strawberryfields.utils import random_interferometer

    U = random_interferometer(N_modes)
    prog = sf.Program(N_modes)
    with prog.context as q:
        for i, r in enumerate(r_vals):
            sf_ops.Squeezed(r, 0) | q[i]
        sf_ops.Interferometer(U) | tuple(q[i] for i in range(N_modes))

    eng    = sf.Engine("gaussian")
    result = eng.run(prog)
    state  = result.state
    cov    = state.cov()
    means  = state.means()
    mn_per = [state.mean_photon(i)[0] for i in range(N_modes)]
    xvec_sf= np.linspace(-5, 5, 80)
    wigners= {}
    for i in range(N_modes):
        W_, _ = state.wigner(i, xvec_sf, xvec_sf)
        wigners[i] = W_
    return dict(cov=cov, means=means, mean_photons=mn_per, wigners=wigners, xvec=xvec_sf)


def _hafnian_brute(A: np.ndarray) -> float:
    """Brute-force hafnian for small matrices."""
    n2 = A.shape[0]
    if n2 % 2 != 0: return 0.0
    idx = list(range(n2)); haf = 0.0+0j
    def matchings(lst):
        if not lst: yield []; return
        first = lst[0]; rest = lst[1:]
        for i, v in enumerate(rest):
            for m in matchings(rest[:i]+rest[i+1:]):
                yield [(first,v)]+m
    for m in matchings(idx):
        term = 1.0+0j
        for (i,j) in m: term *= A[i,j]
        haf += term
    return float(np.real(haf))


# ════════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ════════════════════════════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    if   "State Explorer"    in page: page_state_explorer()
    elif "Phase Space Zoo"   in page: page_phase_space_zoo()
    elif "Witness Lab"       in page: page_witness_lab()
    elif "Channel Simulator" in page: page_channel_simulator()
    elif "GBS Sampler"       in page: page_gbs_sampler()

if __name__ == "__main__":
    main()
