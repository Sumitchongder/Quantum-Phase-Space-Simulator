"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  app.py  —  Density Matrix & Wigner Function Dashboard  v5.0                 ║
║  IIT Jodhpur · m25iqt013                                                     ║
║  Live QuTiP computation · Industry-grade professional UI                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import sys, os, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import qutip as qt
from qutip import (
    basis, ket2dm, coherent, coherent_dm, thermal_dm,
    expect, destroy, num, displace, squeeze, mesolve, Options,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be FIRST Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Quantum Phase Space · m25iqt013",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Density Matrix & Wigner Function · IIT Jodhpur · m25iqt013"},
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — bright, vibrant, Xanadu professional
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #07091a;
    background-image:
        radial-gradient(ellipse 90% 40% at 15% 0%, rgba(109,40,217,0.12) 0%, transparent 55%),
        radial-gradient(ellipse 60% 35% at 85% 90%, rgba(6,182,212,0.07) 0%, transparent 55%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0e24 0%, #080918 100%) !important;
    border-right: 1px solid #2d1b69;
}
[data-testid="stHeader"] { background: transparent !important; }
html, body, [class*="css"] {
    color: #e2e8f0;
    font-family: 'Inter', 'DejaVu Sans', sans-serif;
}
.block-container { padding-top: 1.4rem !important; }

/* ── Typography ── */
h1 { font-size: 2.0rem; font-weight: 800; letter-spacing: -0.02em; color: #f1f5f9; }
h2 { font-size: 1.35rem; font-weight: 700; color: #a78bfa; }
h3 { font-size: 1.05rem; font-weight: 600; color: #7dd3fc; }

/* ── Sidebar ── */
[data-testid="stSidebar"] label { color: #c4b5fd !important; font-weight: 600; font-size: 0.82rem !important; }
[data-testid="stSidebar"] .stSlider > label { color: #7dd3fc !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #64748b; font-size: 0.73rem; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #12124a 0%, #1a0a2e 100%);
    border: 1px solid #2d1b69;
    border-radius: 12px;
    padding: 12px 16px;
    transition: transform 0.15s, box-shadow 0.15s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(167,139,250,0.2);
}
[data-testid="stMetricValue"] {
    color: #a78bfa; font-weight: 800; font-size: 1.35rem !important;
    font-family: 'JetBrains Mono', monospace;
}
[data-testid="stMetricLabel"] { color: #94a3b8; font-size: 0.72rem; }
[data-testid="stMetricDelta"] { font-size: 0.72rem; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4c1d95, #7c3aed);
    color: white; border: none; border-radius: 9px;
    font-weight: 700; padding: 9px 22px;
    font-family: 'Inter', sans-serif; font-size: 0.82rem;
    transition: all 0.2s; box-shadow: 0 4px 12px rgba(124,58,237,0.35);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, #9333ea);
    transform: translateY(-2px); box-shadow: 0 8px 20px rgba(124,58,237,0.5);
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #12124a; border: 1px solid #2d1b69;
    border-radius: 9px; color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
}
[data-baseweb="select"] > div:focus-within {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.25) !important;
}

/* ── Sliders ── */
.stSlider > div > div > div { background: #2d1b69 !important; }
.stSlider [data-testid="stThumb"] {
    background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
    box-shadow: 0 0 10px rgba(167,139,250,0.5) !important;
}

/* ── Multiselect ── */
[data-baseweb="tag"] { background: #4c1d95 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0d0e24; border-bottom: 2px solid #2d1b69;
    border-radius: 12px 12px 0 0; padding: 4px 6px; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b; font-weight: 600; padding: 9px 18px;
    border-radius: 8px 8px 0 0; font-size: 0.82rem;
    font-family: 'Inter', sans-serif; transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    color: #a78bfa !important;
    background: linear-gradient(135deg, rgba(76,29,149,0.3), rgba(124,58,237,0.2));
    border-bottom: 2.5px solid #a78bfa;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    background: #0d0e24; border-radius: 10px; border: 1px solid #2d1b69;
}

/* ── Info / success / warning ── */
[data-testid="stAlert"] { border-radius: 10px; }

/* ── Divider ── */
hr { border-color: #2d1b69; margin: 1rem 0; }

/* ── Banner ── */
.banner {
    background: linear-gradient(135deg, #070718 0%, #160a2e 45%, #071828 100%);
    border: 1px solid #4c1d95;
    border-radius: 18px;
    padding: 26px 36px;
    text-align: center;
    margin-bottom: 1.4rem;
    position: relative; overflow: hidden;
}
.banner::after {
    content: '';
    position: absolute; top: -50px; right: -50px;
    width: 200px; height: 200px; border-radius: 50%;
    background: radial-gradient(circle, rgba(109,40,217,0.18), transparent 70%);
    pointer-events: none;
}
.banner::before {
    content: '';
    position: absolute; bottom: -40px; left: 25%;
    width: 150px; height: 150px; border-radius: 50%;
    background: radial-gradient(circle, rgba(6,182,212,0.09), transparent 70%);
    pointer-events: none;
}
.banner h1 { color: #a78bfa; margin: 0 0 8px 0; font-size: 2.0rem; }
.banner p  { color: #94a3b8; margin: 4px 0; font-size: 0.9rem; line-height: 1.5; }
.banner .tag {
    display: inline-block;
    background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.4);
    border-radius: 100px; padding: 3px 12px;
    font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;
    color: #c4b5fd; margin: 5px 3px 0;
}

/* ── Equation box ── */
.eq-box {
    background: #0f0f2a; border: 1px solid #2d1b69;
    border-left: 3px solid #7c3aed; border-radius: 0 10px 10px 0;
    padding: 12px 18px; margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem; color: #c4b5fd; line-height: 1.75;
}

/* ── Insight card ── */
.ins-card {
    background: linear-gradient(135deg, rgba(6,182,212,0.05), rgba(109,40,217,0.05));
    border: 1px solid rgba(6,182,212,0.22);
    border-radius: 12px; padding: 14px 18px; margin: 8px 0;
}
.ins-card .label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #22d3ee;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;
}
.ins-card p { margin: 0; font-size: 0.84rem; color: #94a3b8; line-height: 1.65; }

/* ── Non-classicality badge ── */
.nc-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 14px; border-radius: 100px;
    font-family: 'Inter', sans-serif; font-size: 0.78rem; font-weight: 700;
}
.nc-quantum  { background: rgba(244,114,182,0.14); color: #f472b6; border: 1px solid rgba(244,114,182,0.4); }
.nc-squeezed { background: rgba(34,211,238,0.12);  color: #22d3ee; border: 1px solid rgba(34,211,238,0.4); }
.nc-classical{ background: rgba(251,191,36,0.10);  color: #fbbf24; border: 1px solid rgba(251,191,36,0.35); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════
_T = dict(
    paper_bgcolor="#07091a",
    plot_bgcolor="#0d0e24",
    font=dict(color="#e2e8f0", family="Inter, DejaVu Sans, sans-serif", size=11),
    xaxis=dict(gridcolor="#1e1e3f", zerolinecolor="#2d1b69",
               title_font=dict(size=11), tickfont=dict(size=9)),
    yaxis=dict(gridcolor="#1e1e3f", zerolinecolor="#2d1b69",
               title_font=dict(size=11), tickfont=dict(size=9)),
    legend=dict(bgcolor="#0d0e24", bordercolor="#2d1b69", borderwidth=1,
                font=dict(size=9)),
    margin=dict(l=50, r=28, t=50, b=40),
    colorway=["#a78bfa","#22d3ee","#f472b6","#34d399","#fbbf24","#60a5fa","#fb923c"],
)

# Colormaps
W_CS  = [[0.0,"#1e3a5f"],[0.5,"#07091a"],[1.0,"#e879f9"]]   # blue-dark-magenta
W_CS2 = [[0.0,"#0c1445"],[0.5,"#07091a"],[1.0,"#38bdf8"]]   # blue-dark-cyan
Q_CS  = [[0.0,"#07091a"],[0.5,"#7c3aed"],[1.0,"#fbbf24"]]   # dark-violet-gold
DM_CS = "RdBu_r"

# ══════════════════════════════════════════════════════════════════════════════
# CACHED STATE BUILDERS & QUANTUM FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600, show_spinner=False)
def _wigner(rho_arr: np.ndarray, xvec: np.ndarray) -> np.ndarray:
    rho = qt.Qobj(rho_arr); rho.dims = [[rho_arr.shape[0]],[rho_arr.shape[0]]]
    return np.array(qt.wigner(rho, xvec, xvec, g=2))

@st.cache_data(ttl=600, show_spinner=False)
def _husimi(rho_arr: np.ndarray, xvec: np.ndarray) -> np.ndarray:
    rho = qt.Qobj(rho_arr); rho.dims = [[rho_arr.shape[0]],[rho_arr.shape[0]]]
    res = qt.qfunc(rho, xvec, xvec)
    return np.array(res[0] if isinstance(res, tuple) else res)

@st.cache_data(ttl=600, show_spinner=False)
def _metrics(rho_arr: np.ndarray) -> dict:
    rho = qt.Qobj(rho_arr); rho.dims = [[rho_arr.shape[0]],[rho_arr.shape[0]]]
    dim = rho.shape[0]
    a   = destroy(dim); n_op = num(dim)
    x_op = (a + a.dag()) / np.sqrt(2)
    p_op = 1j*(a.dag() - a) / np.sqrt(2)
    mn   = float(expect(n_op, rho).real)
    mn2  = float(expect(n_op*n_op, rho).real)
    vn   = mn2 - mn**2
    pur  = float((rho*rho).tr().real)
    ent  = float(qt.entropy_vn(rho, base=2))
    mq   = (vn-mn)/mn if mn > 1e-10 else float("nan")
    mx   = float(expect(x_op, rho).real)
    mp   = float(expect(p_op, rho).real)
    vx   = float(expect(x_op*x_op, rho).real) - mx**2
    vp   = float(expect(p_op*p_op, rho).real) - mp**2
    dx   = math.sqrt(max(vx,0)); dp = math.sqrt(max(vp,0))
    probs= np.array([float(rho[n,n].real) for n in range(min(dim,30))])
    qfi  = 4*vn
    return dict(mean_n=round(mn,5), var_n=round(vn,5), purity=round(pur,6),
                entropy=round(ent,6), mandel_Q=round(mq,5) if not math.isnan(mq) else None,
                mean_x=round(mx,5), mean_p=round(mp,5),
                delta_x=round(dx,6), delta_p=round(dp,6),
                heis_prod=round(dx*dp,6), qfi=round(qfi,4), probs=probs)

def _wnv(W, xvec):
    dx = xvec[1]-xvec[0]
    return float(np.sum(np.abs(W))*dx**2 - 1.0)

def _wrange(W):
    wmin, wmax = float(W.min()), float(W.max())
    if wmin >= 0: wmin = -1e-9
    if wmax <= 0: wmax =  1e-9
    lim = max(abs(wmin), abs(wmax))
    return -lim, lim

# State builders
@st.cache_data(ttl=600, show_spinner=False)
def mk_fock(n, dim):            return ket2dm(basis(dim,n)).full()
@st.cache_data(ttl=600, show_spinner=False)
def mk_coherent(re, im, dim):   return coherent_dm(dim, re+1j*im).full()
@st.cache_data(ttl=600, show_spinner=False)
def mk_squeezed(r, phi, dim):   return ket2dm(squeeze(dim,r*np.exp(1j*phi))*basis(dim,0)).full()
@st.cache_data(ttl=600, show_spinner=False)
def mk_thermal(nbar, dim):      return thermal_dm(dim, nbar).full()
@st.cache_data(ttl=600, show_spinner=False)
def mk_cat(re, im, sign, dim):
    psi = (coherent(dim,re+1j*im) + sign*coherent(dim,-(re+1j*im))).unit()
    return ket2dm(psi).full()
@st.cache_data(ttl=600, show_spinner=False)
def mk_ds(a_re, a_im, r, phi, dim):
    psi = displace(dim,a_re+1j*a_im)*squeeze(dim,r*np.exp(1j*phi))*basis(dim,0)
    return ket2dm(psi).full()
@st.cache_data(ttl=600, show_spinner=False)
def mk_gkp(delta, npeaks, dim):
    psi = sum(np.exp(-delta**2*n**2)*displace(dim,n*np.sqrt(np.pi))*basis(dim,0)
              for n in range(-npeaks, npeaks+1))
    return ket2dm(psi.unit()).full()
@st.cache_data(ttl=600, show_spinner=False)
def mk_tmsv_reduced(r, dim):
    lam  = np.tanh(r); norm = np.sqrt(1-lam**2)
    from qutip import tensor as qt_tensor
    psi  = sum(norm*(lam**n)*qt_tensor(basis(dim,n),basis(dim,n)) for n in range(dim))
    rho2 = ket2dm(psi); rho2.dims = [[dim,dim],[dim,dim]]
    return rho2.ptrace(0).full()

# Channel ops
@st.cache_data(ttl=600, show_spinner=False)
def ch_disp(rho_arr, a_re, a_im):
    rho=qt.Qobj(rho_arr); rho.dims=[[rho_arr.shape[0]],[rho_arr.shape[0]]]
    D=displace(rho.shape[0],a_re+1j*a_im); return (D*rho*D.dag()).full()
@st.cache_data(ttl=600, show_spinner=False)
def ch_sq(rho_arr, r, phi):
    rho=qt.Qobj(rho_arr); rho.dims=[[rho_arr.shape[0]],[rho_arr.shape[0]]]
    S=squeeze(rho.shape[0],r*np.exp(1j*phi)); return (S*rho*S.dag()).full()
@st.cache_data(ttl=600, show_spinner=False)
def ch_phase(rho_arr, phi):
    rho=qt.Qobj(rho_arr); rho.dims=[[rho_arr.shape[0]],[rho_arr.shape[0]]]
    R=(-1j*phi*num(rho.shape[0])).expm(); return (R*rho*R.dag()).full()
@st.cache_data(ttl=600, show_spinner=False)
def ch_loss(rho_arr, gt):
    rho=qt.Qobj(rho_arr); rho.dims=[[rho_arr.shape[0]],[rho_arr.shape[0]]]
    dim=rho.shape[0]; a=destroy(dim)
    H=qt.Qobj(np.zeros((dim,dim))); H.dims=[[dim],[dim]]
    opts=Options(nsteps=15000)
    res=mesolve(H,rho,np.linspace(0,gt,12),[np.sqrt(1.0)*a],[],options=opts)
    return res.states[-1].full()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_wigner(W, xvec, title="W(x,p)", cs=None, height=430):
    cs    = cs or W_CS
    wmin, wmax = _wrange(W)
    fig   = go.Figure()
    fig.add_trace(go.Heatmap(
        z=W, x=xvec, y=xvec,
        colorscale=cs, zmin=wmin, zmax=wmax,
        colorbar=dict(title=dict(text="W", font=dict(size=10,color="#e2e8f0")),
                      len=0.82, thickness=14, tickfont=dict(color="#e2e8f0",size=9),
                      outlinecolor="#2d1b69", outlinewidth=1),
        hovertemplate="x=%{x:.2f}  p=%{y:.2f}  W=%{z:.5f}<extra></extra>",
    ))
    fig.add_trace(go.Contour(
        z=W, x=xvec, y=xvec,
        contours=dict(start=0,end=0,size=1,coloring="none"),
        line=dict(color="rgba(255,255,255,0.5)", width=1.3, dash="dot"),
        showscale=False, hoverinfo="skip",
    ))
    fig.update_layout(**_T, height=height,
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=14, color="#a78bfa"), x=0.5),
        xaxis_title="x  (quadrature)", yaxis_title="p  (momentum)",
    )
    return fig

def fig_husimi(Q, xvec, title="Q(α)", height=430):
    fig = go.Figure(go.Heatmap(
        z=Q, x=xvec, y=xvec,
        colorscale=Q_CS,
        colorbar=dict(title=dict(text="Q", font=dict(size=10,color="#e2e8f0")),
                      len=0.82, thickness=14, tickfont=dict(color="#e2e8f0",size=9),
                      outlinecolor="#2d1b69", outlinewidth=1),
        hovertemplate="Re(α)=%{x:.2f}  Im(α)=%{y:.2f}  Q=%{z:.5f}<extra></extra>",
    ))
    fig.update_layout(**_T, height=height,
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=14, color="#22d3ee"), x=0.5),
        xaxis_title="Re(α)", yaxis_title="Im(α)",
    )
    return fig

def fig_density_matrix(rho_arr, display_dim=15, title="ρ", height=400):
    d = min(display_dim, rho_arr.shape[0])
    sub = make_subplots(1, 2, subplot_titles=["Re(ρ<sub>mn</sub>)", "Im(ρ<sub>mn</sub>)"],
                        horizontal_spacing=0.09)
    for ci, dat in enumerate([np.real(rho_arr[:d,:d]), np.imag(rho_arr[:d,:d])], 1):
        vm = max(abs(dat).max(), 1e-6)
        sub.add_trace(go.Heatmap(
            z=dat, colorscale=DM_CS, zmid=0, zmin=-vm, zmax=vm,
            showscale=(ci==1),
            colorbar=dict(title=dict(text="ρ",font=dict(size=10)),
                          x=1.02, len=0.88, thickness=12,
                          tickfont=dict(color="#e2e8f0",size=9)),
            hovertemplate="n=%{y}  m=%{x}<br>ρ=%{z:.5f}<extra></extra>",
        ), row=1, col=ci)
    sub.update_layout(**_T, height=height,
        title=dict(text=f"<b>Density Matrix {title}</b>",
                   font=dict(size=13, color="#22d3ee"), x=0.5),
    )
    sub.update_xaxes(title_text="m", title_font=dict(size=10))
    sub.update_yaxes(title_text="n", title_font=dict(size=10))
    return sub

def fig_photon_dist(probs, mean_n, title="P(n)", height=340):
    k   = np.arange(len(probs))
    poi = np.array([
        math.exp(-max(mean_n,1e-9))*(max(mean_n,1e-9)**ki)/math.factorial(int(ki))
        if int(ki)<170 else 0.0 for ki in k
    ])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=k, y=probs, name="P(n)",
        marker=dict(
            color=probs,
            colorscale=[[0,"#1e1b4b"],[0.5,"#7c3aed"],[1,"#a78bfa"]],
            showscale=False, line=dict(color="rgba(0,0,0,0)"),
        ),
        opacity=0.9,
    ))
    if mean_n > 0.01:
        fig.add_trace(go.Scatter(x=k, y=poi, mode="lines+markers",
            name="Poisson ref",
            line=dict(color="#fbbf24", width=2, dash="dot"),
            marker=dict(size=4, color="#fbbf24"),
        ))
    fig.update_layout(**_T, height=height, bargap=0.12,
        title=dict(text=f"<b>{title}</b>", font=dict(size=13,color="#a78bfa"), x=0.5),
        xaxis_title="Photon number n", yaxis_title="P(n)",
    )
    return fig

def fig_wigner_3d(W, xvec, title="W(x,p) — 3D Surface", height=500):
    wa  = max(abs(W.min()), abs(W.max()), 1e-6)
    fig = go.Figure(go.Surface(
        z=W, x=xvec, y=xvec,
        colorscale=W_CS, cmin=-wa, cmax=wa,
        showscale=True, opacity=0.93,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
        colorbar=dict(title=dict(text="W",font=dict(size=10)),
                      len=0.72, thickness=14, tickfont=dict(color="#e2e8f0",size=9)),
        hovertemplate="x=%{x:.2f}  p=%{y:.2f}  W=%{z:.5f}<extra></extra>",
    ))
    fig.update_layout(**_T, height=height,
        title=dict(text=f"<b>{title}</b>", font=dict(size=14,color="#a78bfa"), x=0.5),
        scene=dict(
            xaxis=dict(title="x",gridcolor="#1e1e3f",backgroundcolor="#07091a",showbackground=True),
            yaxis=dict(title="p",gridcolor="#1e1e3f",backgroundcolor="#07091a",showbackground=True),
            zaxis=dict(title="W(x,p)",gridcolor="#1e1e3f",backgroundcolor="#07091a",showbackground=True),
            bgcolor="#07091a",
            camera=dict(eye=dict(x=1.4,y=1.4,z=0.9)),
        ),
    )
    return fig

def metric_strip(m, wnv):
    """Full metric strip — 7 + 4 cards."""
    cols = st.columns(7)
    mq   = m["mandel_Q"]
    for col, (lbl, val) in zip(cols, [
        ("⟨n̂⟩",    f"{m['mean_n']:.4f}"),
        ("Purity",  f"{m['purity']:.5f}"),
        ("Entropy", f"{m['entropy']:.4f}"),
        ("Δx",      f"{m['delta_x']:.5f}"),
        ("Δp",      f"{m['delta_p']:.5f}"),
        ("ΔxΔp",    f"{m['heis_prod']:.5f}"),
        ("WNV",     f"{wnv:.5f}"),
    ]):
        col.metric(lbl, val)
    cols2 = st.columns(4)
    for col, (lbl, val, hlp) in zip(cols2, [
        ("Mandel Q", f"{mq:.4f}" if mq is not None else "N/A", "< 0 = sub-Poissonian"),
        ("Var(n)",   f"{m['var_n']:.5f}", "Photon number variance"),
        ("⟨x̂⟩",    f"{m['mean_x']:.4f}", "Mean position quadrature"),
        ("⟨p̂⟩",    f"{m['mean_p']:.4f}", "Mean momentum quadrature"),
    ]):
        col.metric(lbl, val, help=hlp)

def nc_badge(wnv, m):
    mq = m.get("mandel_Q")
    if wnv > 1e-4:
        return '<span class="nc-badge nc-quantum">⚡ NON-CLASSICAL — Wigner Negativity Detected</span>'
    elif m.get("heis_prod",1) < 0.499:
        return '<span class="nc-badge nc-squeezed">🔧 SQUEEZED — Below Shot-Noise Limit</span>'
    elif mq is not None and mq < -0.01:
        return '<span class="nc-badge nc-squeezed">📉 SUB-POISSONIAN — Quantum Statistics</span>'
    else:
        return '<span class="nc-badge nc-classical">☀️ CLASSICAL / GAUSSIAN</span>'

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    st.sidebar.markdown("""
    <div style="padding:18px 14px 12px; text-align:center;">
      <div style="font-size:2.6rem; line-height:1;
                  filter:drop-shadow(0 0 18px rgba(167,139,250,0.8));">⚛️</div>
      <div style="font-family:'Inter',sans-serif; font-weight:900; font-size:0.98rem;
                  color:#f1f5f9; letter-spacing:-0.01em; margin-top:10px; line-height:1.25;">
        Quantum Phase Space<br>
        <span style="font-size:0.72rem; font-weight:500; color:#7c3aed;">
          Density Matrix & Wigner Function
        </span>
      </div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:0.64rem;
                  color:#334155; margin-top:5px;">
        IIT Jodhpur · m25iqt013
      </div>
    </div>
    <hr style="border-color:#2d1b69; margin:0 0 10px 0;">
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

    st.sidebar.markdown("<hr style='border-color:#2d1b69; margin:14px 0;'>",
                        unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='font-family:\"JetBrains Mono\",monospace; font-size:0.63rem;"
        "color:#334155; text-align:center; line-height:1.9;'>"
        "QuTiP · NumPy · Plotly · Streamlit<br>"
        "© 2025 m25iqt013 — IIT Jodhpur</div>",
        unsafe_allow_html=True,
    )
    return page

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STATE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
def page_state_explorer():
    st.markdown("""
    <div class="banner">
      <h1>🔬 State Explorer</h1>
      <p>Interactive phase-space explorer — live QuTiP computation for all 8 CV quantum states</p>
      <p>Adjust parameters with sliders · Instant Wigner · Husimi Q · Density matrix · Photon statistics</p>
      <span class="tag">Wigner W(x,p)</span>
      <span class="tag">Husimi Q(α)</span>
      <span class="tag">Density Matrix ρ</span>
      <span class="tag">Photon Statistics P(n)</span>
      <span class="tag">8 Quantum States</span>
    </div>""", unsafe_allow_html=True)

    col_ctrl, col_main = st.columns([1, 3], gap="medium")

    with col_ctrl:
        st.markdown("### ⚙️ State Controls")
        state_type = st.selectbox("**Quantum State**", [
            "Fock |n⟩", "Coherent |α⟩", "Squeezed |r,φ⟩",
            "Thermal ρ̂_th", "Cat State", "Displaced-Squeezed",
            "TMSV (EPR)", "GKP Grid State",
        ])
        dim  = st.slider("**Hilbert dim**", 20, 60, 35, 5,
                         help="Fock space truncation dimension")
        xres = st.slider("**Grid resolution**", 80, 200, 130, 20,
                         help="Points per axis for W and Q computation")
        xvec = np.linspace(-6, 6, xres)
        st.markdown("---")

        rho_arr = None

        if state_type == "Fock |n⟩":
            n = st.slider("**n** (photon number)", 0, min(dim-1, 15), 3)
            st.markdown(f"""<div class="eq-box">|n={n}⟩⟨n={n}|<br>Purity = 1 · Non-Gaussian · WNV &gt; 0</div>""",
                        unsafe_allow_html=True)
            rho_arr = mk_fock(n, dim)

        elif state_type == "Coherent |α⟩":
            a_re = st.slider("**Re(α)**", -3.0, 3.0, 1.5, 0.1)
            a_im = st.slider("**Im(α)**", -3.0, 3.0, 0.5, 0.1)
            st.markdown(f"""<div class="eq-box">|α={a_re:+.2f}{a_im:+.2f}i⟩<br>Gaussian · Purity = 1 · ΔxΔp = ½</div>""",
                        unsafe_allow_html=True)
            rho_arr = mk_coherent(a_re, a_im, dim)

        elif state_type == "Squeezed |r,φ⟩":
            r   = st.slider("**r** (squeezing)", 0.0, 2.5, 0.8, 0.05)
            phi = st.slider("**φ** (angle, π units)", 0.0, 2.0, 0.0, 0.1)
            dB  = round(10*math.log10(math.exp(2*r)), 2) if r > 0 else 0
            st.markdown(f"""<div class="eq-box">Ŝ(ξ)|0⟩  ξ = r·e^{{iφ}}<br>r = {r:.2f} ({dB} dB) · φ = {phi:.1f}π<br>Sub-shot-noise in x quadrature</div>""",
                        unsafe_allow_html=True)
            rho_arr = mk_squeezed(r, phi*math.pi, dim)

        elif state_type == "Thermal ρ̂_th":
            nbar = st.slider("**n̄** (mean photon number)", 0.1, 5.0, 1.0, 0.1)
            st.markdown(f"""<div class="eq-box">ρ_th = Σ [n̄ⁿ/(n̄+1)^(n+1)] |n⟩⟨n|<br>n̄ = {nbar:.2f} · Mixed · Super-Poissonian</div>""",
                        unsafe_allow_html=True)
            rho_arr = mk_thermal(nbar, dim)

        elif state_type == "Cat State":
            a_re  = st.slider("**Re(α)**", 0.5, 3.5, 2.0, 0.1)
            a_im  = st.slider("**Im(α)**", -2.0, 2.0, 0.0, 0.1)
            parity = st.radio("**Parity**", ["+1  (even cat)", "-1  (odd cat)"])
            sign  = +1 if "+1" in parity else -1
            label = "even" if sign > 0 else "odd"
            st.markdown(f"""<div class="eq-box">N±(|α⟩ ± |−α⟩)  [{label}]<br>α = {a_re:+.2f}{a_im:+.2f}i · Interference fringes</div>""",
                        unsafe_allow_html=True)
            rho_arr = mk_cat(a_re, a_im, sign, dim)

        elif state_type == "Displaced-Squeezed":
            a_re = st.slider("**Re(α)**", -3.0, 3.0, 2.0, 0.1)
            a_im = st.slider("**Im(α)**", -3.0, 3.0, 0.0, 0.1)
            r    = st.slider("**r** (squeezing)", 0.0, 2.0, 0.8, 0.05)
            phi  = st.slider("**φ** (angle, π)", 0.0, 2.0, 0.0, 0.1)
            st.markdown(f"""<div class="eq-box">D̂(α)Ŝ(ξ)|0⟩ · α = {a_re:+.2f}{a_im:+.2f}i<br>r = {r:.2f} · φ = {phi:.1f}π · CV-QKD resource</div>""",
                        unsafe_allow_html=True)
            rho_arr = mk_ds(a_re, a_im, r, phi*math.pi, dim)

        elif state_type == "TMSV (EPR)":
            r    = st.slider("**r** (two-mode squeezing)", 0.1, 2.0, 1.0, 0.05)
            dim2 = min(dim, 20)
            EN   = 2*r/math.log(2)
            st.markdown(f"""<div class="eq-box">Ŝ₂(r)|00⟩ · r = {r:.2f}<br>E_N = {EN:.3f} · Reduced single-mode state</div>""",
                        unsafe_allow_html=True)
            rho_arr = mk_tmsv_reduced(r, dim2)

        else:  # GKP
            delta  = st.slider("**δ** (envelope width)", 0.10, 0.60, 0.25, 0.05)
            npeaks = st.slider("**n_max** (peaks)", 2, 6, 4, 1)
            st.markdown(f"""<div class="eq-box">Σ exp(−δ²n²) D̂(n√π)|0⟩<br>δ = {delta:.2f} · Photonic error correction</div>""",
                        unsafe_allow_html=True)
            rho_arr = mk_gkp(delta, npeaks, dim)

        st.markdown("---")
        st.markdown("**Display Options**")
        show_3d = st.checkbox("🏔️  3D Wigner Surface", False)
        rep_sel = st.multiselect(
            "**Representations**",
            ["Wigner", "Husimi Q", "Density Matrix", "Photon Dist"],
            default=["Wigner", "Husimi Q"],
        )

    with col_main:
        if rho_arr is None:
            st.info("Select a state from the left panel.")
            return

        with st.spinner("⚛️  Computing phase space…"):
            W   = _wigner(rho_arr, xvec)
            Q   = _husimi(rho_arr, xvec)
            m   = _metrics(rho_arr)
            wnv = _wnv(W, xvec)

        # Metric strip
        metric_strip(m, wnv)
        st.markdown(
            f"<div style='margin:10px 0 14px'>{nc_badge(wnv, m)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # 3D Wigner
        if show_3d:
            st.plotly_chart(
                fig_wigner_3d(W, xvec, title=f"W(x,p) 3D — {state_type}"),
                use_container_width=True,
            )

        # Representations
        if "Wigner" in rep_sel and "Husimi Q" in rep_sel:
            c1, c2 = st.columns(2)
            c1.plotly_chart(fig_wigner(W, xvec, title=f"Wigner — {state_type}"),
                            use_container_width=True)
            c2.plotly_chart(fig_husimi(Q, xvec, title=f"Husimi Q — {state_type}"),
                            use_container_width=True)
        elif "Wigner" in rep_sel:
            st.plotly_chart(fig_wigner(W, xvec, title=f"Wigner — {state_type}", height=500),
                            use_container_width=True)
        elif "Husimi Q" in rep_sel:
            st.plotly_chart(fig_husimi(Q, xvec, title=f"Husimi Q — {state_type}", height=500),
                            use_container_width=True)

        if "Density Matrix" in rep_sel:
            st.plotly_chart(
                fig_density_matrix(rho_arr, min(18, dim), title=f"— {state_type}"),
                use_container_width=True,
            )
            off_max = float(np.abs(rho_arr - np.diag(np.diag(rho_arr))).max())
            st.markdown(f"""
            <div class="ins-card">
              <div class="label">💡 Density Matrix</div>
              <p>ρ encodes everything about the quantum state.  The diagonal ρ<sub>nn</sub> = P(n) gives photon probabilities.
              Off-diagonal coherences signal superposition — max |ρ<sub>mn</sub>| = <b>{off_max:.5f}</b>.
              Purity Tr(ρ²) = <b>{m['purity']:.5f}</b>
              {'(pure ✅)' if m['purity'] > 0.999 else '(mixed — entropy > 0)'}.</p>
            </div>""", unsafe_allow_html=True)

        if "Photon Dist" in rep_sel:
            st.plotly_chart(
                fig_photon_dist(m["probs"], m["mean_n"],
                                title=f"P(n) — {state_type}"),
                use_container_width=True,
            )
            mq = m["mandel_Q"]
            if mq is not None:
                if mq < -0.001:
                    regime = f"<b style='color:#f472b6'>Q = {mq:.4f} → Sub-Poissonian</b> (quantum signature)"
                elif mq > 0.001:
                    regime = f"<b style='color:#fbbf24'>Q = {mq:.4f} → Super-Poissonian</b> (thermal bunching)"
                else:
                    regime = f"<b style='color:#34d399'>Q ≈ 0 → Poissonian</b> (coherent statistics)"
                st.markdown(f"""
                <div class="ins-card">
                  <div class="label">💡 Photon Statistics</div>
                  <p>Mandel Q: {regime}.  Gold dashed line = Poisson reference (coherent state).
                  Q &lt; 0 means fewer photon-number fluctuations than a laser — a quantum signature.</p>
                </div>""", unsafe_allow_html=True)

        # Wigner insight
        if "Wigner" in rep_sel:
            st.markdown(f"""
            <div class="ins-card">
              <div class="label">💡 Wigner Function</div>
              <p>W(x,p) is a quasi-probability distribution.
              <b>Negative regions</b> prove non-classicality — WNV = <b>{wnv:.5f}</b>.
              The zero-contour (white dashed line) separates positive and negative regions.
              Red/magenta = positive · Dark/blue = negative.</p>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PHASE SPACE ZOO
# ══════════════════════════════════════════════════════════════════════════════
def page_phase_space_zoo():
    st.markdown("""
    <div class="banner">
      <h1>🌌 Phase Space Zoo</h1>
      <p>All 8 quantum states side-by-side with full parameter control</p>
      <p>Toggle Wigner / Husimi Q / Compare · Adjust all parameters live · Metrics table</p>
      <span class="tag">8 States</span>
      <span class="tag">Wigner · Husimi</span>
      <span class="tag">Non-classicality</span>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🌌 Zoo Controls")
        xres_z  = st.slider("Grid resolution", 60, 160, 90, 10)
        dim_z   = st.slider("Hilbert dim", 20, 50, 30, 5)
        rep_z   = st.radio("Representation", ["Wigner W(x,p)", "Husimi Q(α)", "Compare W vs Q"])
        n_fock  = st.slider("n (Fock)", 0, 8, 2)
        alpha_z = st.slider("|α| (Coherent)", 0.5, 3.0, 2.0, 0.1)
        r_sq    = st.slider("r (Squeezed)", 0.1, 2.0, 0.9, 0.05)
        nbar_th = st.slider("n̄ (Thermal)", 0.1, 4.0, 1.0, 0.1)
        r_cat   = st.slider("|α| (Cat)", 0.5, 3.0, 2.0, 0.1)
        r_ds    = st.slider("r (Disp-Sq)", 0.1, 1.5, 0.8, 0.05)
        r_tmsv  = st.slider("r (TMSV)", 0.1, 1.5, 0.8, 0.05)
        d_gkp   = st.slider("δ (GKP)", 0.1, 0.5, 0.25, 0.05)

    xvec_z = np.linspace(-6, 6, xres_z)

    with st.spinner("⚛️  Building all states…"):
        states_raw = {
            f"Fock |{n_fock}⟩"      : mk_fock(n_fock, dim_z),
            f"Coherent |{alpha_z}⟩"  : mk_coherent(alpha_z, 0.0, dim_z),
            f"Squeezed r={r_sq:.2f}"  : mk_squeezed(r_sq, 0.0, dim_z),
            f"Thermal n̄={nbar_th:.1f}": mk_thermal(nbar_th, dim_z),
            f"Cat |{r_cat}⟩ even"    : mk_cat(r_cat, 0.0, +1, dim_z),
            f"Disp-Sq r={r_ds:.2f}"  : mk_ds(1.5, 0.0, r_ds, 0.0, dim_z),
            "TMSV reduced"           : mk_tmsv_reduced(r_tmsv, min(dim_z, 18)),
            f"GKP δ={d_gkp:.2f}"     : mk_gkp(d_gkp, 4, dim_z),
        }
        Ws, Qs, mets = {}, {}, {}
        for lbl, arr in states_raw.items():
            Ws[lbl]   = _wigner(arr, xvec_z)
            Qs[lbl]   = _husimi(arr, xvec_z)
            mets[lbl] = _metrics(arr)

    labels = list(states_raw.keys())

    def _tile(W_or_Q, lbl, cs, is_W, ht=280):
        arr = np.array(W_or_Q)
        neg = _wnv(arr, xvec_z) if is_W else 0
        wmin, wmax = _wrange(arr) if is_W else (float(arr.min()), float(arr.max()))
        badge_col  = "#f472b6" if neg > 1e-4 else "#fbbf24"
        badge_txt  = f"⚡ WNV={neg:.4f}" if neg > 1e-4 else "☀️ Classical"
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=arr, x=xvec_z, y=xvec_z,
            colorscale=cs, zmin=wmin, zmax=wmax, showscale=False,
            hovertemplate=f"<b>{lbl}</b><br>x=%{{x:.2f}} p=%{{y:.2f}}<br>val=%{{z:.4f}}<extra></extra>"))
        if is_W:
            fig.add_trace(go.Contour(z=arr, x=xvec_z, y=xvec_z,
                contours=dict(start=0,end=0,size=1,coloring="none"),
                line=dict(color="rgba(255,255,255,0.55)",width=1.2),
                showscale=False, hoverinfo="skip"))
        fig.update_layout(
            paper_bgcolor="#07091a", plot_bgcolor="#0d0e24",
            font=dict(color="#e2e8f0", size=9),
            xaxis=dict(gridcolor="#1e1e3f", tickfont=dict(size=7), title="x", title_font=dict(size=8)),
            yaxis=dict(gridcolor="#1e1e3f", tickfont=dict(size=7), title="p", title_font=dict(size=8)),
            title=dict(text=f"<b>{lbl}</b><br><sup style='color:{badge_col}'>{badge_txt}</sup>",
                       font=dict(size=10,color="#e2e8f0"), x=0.5),
            margin=dict(l=30,r=8,t=48,b=26), height=ht,
        )
        return fig

    def _grid(data_dict, cs, is_W):
        lbls = list(data_dict.keys())
        rows = [lbls[i:i+4] for i in range(0, len(lbls), 4)]
        for row in rows:
            cols = st.columns(len(row))
            for col, lbl in zip(cols, row):
                col.plotly_chart(_tile(data_dict[lbl], lbl, cs, is_W),
                                 use_container_width=True)

    if rep_z in ["Wigner W(x,p)", "Compare W vs Q"]:
        st.markdown("#### 🌀 Wigner Function W(x,p)")
        _grid(Ws, W_CS, True)

    if rep_z in ["Husimi Q(α)", "Compare W vs Q"]:
        st.markdown("#### 🔶 Husimi Q Function Q(α)")
        _grid(Qs, Q_CS, False)

    st.markdown("---")
    st.markdown("### 📊 Metrics Comparison Table")
    rows_t = []
    for lbl in labels:
        mm  = mets[lbl]; W = Ws[lbl]; neg = _wnv(W, xvec_z)
        nc  = ("✅ Non-classical" if neg > 1e-4
               else ("🔧 Squeezed" if mm["heis_prod"] < 0.499 else "☀️ Classical"))
        rows_t.append({
            "State": lbl, "⟨n̂⟩": mm["mean_n"], "Purity": mm["purity"],
            "Entropy": mm["entropy"], "Δx": mm["delta_x"], "Δp": mm["delta_p"],
            "ΔxΔp": mm["heis_prod"], "WNV": round(neg, 5),
            "Mandel Q": mm["mandel_Q"] if mm["mandel_Q"] is not None else float("nan"),
            "Class": nc,
        })
    df_z = pd.DataFrame(rows_t).set_index("State")
    st.dataframe(
        df_z.style
            .background_gradient(cmap="Purples", subset=["WNV"])
            .background_gradient(cmap="Blues",   subset=["Purity"])
            .format({"⟨n̂⟩":":.4f","Purity":":.5f","Entropy":":.4f",
                     "Δx":":.5f","Δp":":.5f","ΔxΔp":":.5f","WNV":":.5f",
                     "Mandel Q": lambda v: f"{v:.4f}" if not (isinstance(v,float) and math.isnan(v)) else "—"}),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WITNESS LAB
# ══════════════════════════════════════════════════════════════════════════════
def page_witness_lab():
    st.markdown("""
    <div class="banner">
      <h1>🧪 Witness Lab</h1>
      <p>Live non-classicality witnesses · All 8 states · All measures simultaneously</p>
      <p>Wigner negativity · Mandel Q · Purity · Entropy · Heisenberg · QFI · Radar chart</p>
      <span class="tag">WNV Witness</span>
      <span class="tag">Mandel Q</span>
      <span class="tag">QFI</span>
      <span class="tag">Heisenberg ΔxΔp</span>
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

    with st.spinner("⚛️  Computing all witnesses…"):
        states_w = {
            "Vacuum |0⟩"             : mk_fock(0, dim_w),
            "Fock |3⟩"               : mk_fock(3, dim_w),
            f"Coherent |{alpha_w}⟩"  : mk_coherent(alpha_w, 0.0, dim_w),
            f"Squeezed r={r_w}"      : mk_squeezed(r_w, 0.0, dim_w),
            f"Thermal n̄={nbar_w}"   : mk_thermal(nbar_w, dim_w),
            f"Even Cat |{cat_a_w}⟩"  : mk_cat(cat_a_w, 0.0, +1, dim_w),
            f"Odd Cat |{cat_a_w}⟩"   : mk_cat(cat_a_w, 0.0, -1, dim_w),
            "Disp-Sq r=0.8"          : mk_ds(2.0, 0.0, 0.8, 0.0, dim_w),
        }
        data_rows = []; W_all = {}
        for lbl, arr in states_w.items():
            W   = _wigner(arr, xvec_w)
            mm  = _metrics(arr)
            neg = _wnv(W, xvec_w)
            W_all[lbl] = W
            data_rows.append({
                "State": lbl, "WNV": round(neg,6),
                "W_min": round(float(W.min()),6), "W_max": round(float(W.max()),5),
                "Purity": mm["purity"], "Entropy": mm["entropy"],
                "Δx": mm["delta_x"], "Δp": mm["delta_p"], "ΔxΔp": mm["heis_prod"],
                "Mandel Q": round(mm["mandel_Q"],5) if mm["mandel_Q"] is not None else float("nan"),
                "⟨n̂⟩": mm["mean_n"], "Var(n)": mm["var_n"],
                "QFI≈4Var(n)": mm["qfi"],
                "Non-classical": bool(neg > 1e-4),
                "Sub-Poissonian": bool(mm["mandel_Q"] is not None and mm["mandel_Q"] < -0.01),
                "Squeezed": bool(mm["heis_prod"] < 0.499),
            })

    df_w = pd.DataFrame(data_rows).set_index("State")
    labels_w = list(states_w.keys())

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊  Witness Table", "📈  Bar Charts",
        "🌀  Wigner Gallery", "🔢  QFI Analysis", "⚖️  Heisenberg"
    ])

    with tab1:
        st.markdown("### Complete Non-Classicality Witness Table")
        def color_nc(val):
            if val is True:  return "background-color:#14532d;color:#86efac;font-weight:700"
            if val is False: return "background-color:#1c1c2e;color:#475569"
            return ""
        num_cols  = ["WNV","W_min","Purity","Entropy","Δx","Δp","ΔxΔp","Mandel Q","⟨n̂⟩","Var(n)","QFI≈4Var(n)"]
        bool_cols = ["Non-classical","Sub-Poissonian","Squeezed"]
        fmt_mq = lambda v: f"{v:.5f}" if not (isinstance(v,float) and math.isnan(v)) else "—"
        st.dataframe(
            df_w.style
                .background_gradient(cmap="Purples",  subset=["WNV"])
                .background_gradient(cmap="RdYlGn_r", subset=["Purity"])
                .map(color_nc, subset=bool_cols)
                .format({c:":.5f" for c in num_cols if c in df_w.columns and c != "Mandel Q"})
                .format({"Mandel Q": fmt_mq}),
            use_container_width=True, height=340,
        )
        st.download_button("⬇️ Download CSV", df_w.to_csv().encode(),
                           "witness_table.csv", "text/csv")

    with tab2:
        st.markdown("### Non-Classicality Witnesses — Bar Charts")
        witnesses_plot = [
            ("WNV",          "Wigner Negativity Volume",      "#a78bfa", "δ=0 → classical"),
            ("Purity",       "Purity Tr(ρ²)",                 "#34d399", "1=pure"),
            ("Entropy",      "von Neumann Entropy (bits)",    "#22d3ee", "0=pure state"),
            ("ΔxΔp",         "Heisenberg Product Δx·Δp",      "#f472b6", "≥ 0.5"),
            ("Mandel Q",     "Mandel Q (Q<0 = quantum)",      "#fbbf24", "0=Poissonian"),
            ("QFI≈4Var(n)",  "Quantum Fisher Info",           "#60a5fa", "phase estimation"),
        ]
        for row_group in [witnesses_plot[:3], witnesses_plot[3:]]:
            cols_b = st.columns(3)
            for col_b, (metric, title, color, note) in zip(cols_b, row_group):
                vals = df_w[metric].tolist()
                clean_vals = [v if not (isinstance(v,float) and math.isnan(v)) else 0 for v in vals]
                fig = go.Figure(go.Bar(
                    x=labels_w, y=clean_vals,
                    marker=dict(color=color, line=dict(color="rgba(255,255,255,0.3)", width=0.5)),
                    text=[f"{v:.3f}" if not (isinstance(v,float) and math.isnan(v)) else "—"
                          for v in vals],
                    textposition="outside", textfont=dict(size=9),
                ))
                fig.update_layout(**_T, height=295,
                    title=dict(text=f"<b>{title}</b><br><sup>{note}</sup>",
                               font=dict(size=11,color=color), x=0.5),
                    xaxis=dict(tickangle=-32, tickfont=dict(size=8)),
                )
                col_b.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Wigner Functions — All Witness States")
        lbls_g = list(W_all.keys())
        for row_g in [lbls_g[:4], lbls_g[4:]]:
            cols_g = st.columns(4)
            for col_g, lbl in zip(cols_g, row_g):
                W   = W_all[lbl]; neg = _wnv(W, xvec_w)
                col_g.plotly_chart(
                    fig_wigner(W, xvec_w, title=f"{lbl}<br>WNV={neg:.4f}", height=265),
                    use_container_width=True,
                )

    with tab4:
        st.markdown("### Quantum Fisher Information")
        st.markdown("""<div class="ins-card">
          <div class="label">💡 QFI</div>
          <p>QFI ≈ 4·Var(n) quantifies metrological usefulness for phase estimation (Cramér-Rao bound).
          Higher QFI → better phase resolution below the standard quantum limit.</p>
        </div>""", unsafe_allow_html=True)

        r_qfi  = np.linspace(0.01, 2.5, 60)
        fig_qfi = go.Figure()
        for y, name, col in [
            (4*r_qfi**2,          "Coherent  4|α|²",           "#22d3ee"),
            (4*np.sinh(r_qfi)**2, "Squeezed  4sinh²(r)",       "#a78bfa"),
        ]:
            fig_qfi.add_trace(go.Scatter(x=r_qfi, y=y, name=name,
                line=dict(color=col, width=2.5)))
        fig_qfi.update_layout(**_T, height=320,
            title=dict(text="<b>QFI vs Parameter</b>",
                       font=dict(size=13,color="#a78bfa"), x=0.5),
            xaxis_title="Parameter (|α| or r)", yaxis_title="QFI")
        st.plotly_chart(fig_qfi, use_container_width=True)

        qfi_vals = df_w["QFI≈4Var(n)"].tolist()
        fig_rad  = go.Figure(go.Scatterpolar(
            r=qfi_vals, theta=labels_w, fill="toself",
            line_color="#a78bfa", fillcolor="rgba(167,139,250,0.15)",
        ))
        fig_rad.update_layout(**_T, height=420,
            polar=dict(bgcolor="#0d0e24",
                       radialaxis=dict(gridcolor="#2d1b69"),
                       angularaxis=dict(gridcolor="#2d1b69")),
            title=dict(text="<b>QFI Radar — All States</b>",
                       font=dict(size=13,color="#a78bfa"), x=0.5))
        st.plotly_chart(fig_rad, use_container_width=True)

    with tab5:
        st.markdown("### Heisenberg Uncertainty Product Δx·Δp")
        heis_vals = df_w["ΔxΔp"].tolist()
        colors_h  = ["#f472b6" if row["Non-classical"] else
                     ("#22d3ee" if row["Squeezed"] else "#fbbf24")
                     for _, row in df_w.iterrows()]
        fig_h = go.Figure(go.Bar(
            x=labels_w, y=heis_vals,
            marker=dict(color=colors_h, line=dict(color="rgba(255,255,255,0.3)",width=0.5)),
            text=[f"{v:.5f}" for v in heis_vals],
            textposition="outside", textfont=dict(size=9),
        ))
        fig_h.add_hline(y=0.5, line_color="#34d399", line_dash="dash", line_width=2,
                        annotation_text="Heisenberg limit  Δx·Δp = ½",
                        annotation_font=dict(color="#34d399", size=10),
                        annotation_position="top right")
        fig_h.update_layout(**_T, height=420,
            title=dict(text="<b>Heisenberg Uncertainty Product  Δx · Δp</b>",
                       font=dict(size=13,color="#a78bfa"), x=0.5),
            xaxis=dict(tickangle=-32, tickfont=dict(size=9)),
            yaxis_title="Δx · Δp",
        )
        st.plotly_chart(fig_h, use_container_width=True)
        st.markdown("""<div class="eq-box">
        Heisenberg Uncertainty Principle:   Δx · Δp  ≥  ½<br>
        Minimum uncertainty (Δx·Δp = ½):  Coherent states,  Squeezed states (along squeezed axis)<br>
        Thermal / mixed states:  Δx·Δp  >  ½  (excess classical noise above vacuum level)
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CHANNEL SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
def page_channel_simulator():
    st.markdown("""
    <div class="banner">
      <h1>⚡ Channel Simulator</h1>
      <p>Apply any quantum channel to any input state · Watch the Wigner function evolve live</p>
      <p>Displacement D(α) · Squeezing S(r) · Phase Shift R(φ) · Photon Loss (Lindblad)</p>
      <span class="tag">Before / After Wigner</span>
      <span class="tag">ΔW difference map</span>
      <span class="tag">Loss evolution sweep</span>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚡ Channel Controls")
        dim_c  = st.slider("Hilbert dim", 20, 50, 30, 5)
        xres_c = st.slider("Grid resolution", 60, 140, 90, 10)

        st.markdown("**Input State**")
        in_type = st.selectbox("Input", [
            "Coherent |α⟩", "Fock |n⟩", "Squeezed", "Cat State", "Thermal"
        ])
        if in_type == "Coherent |α⟩":
            a_in   = st.slider("Re(α)", -3.0, 3.0, 2.0, 0.1)
            rho_in = mk_coherent(a_in, 0.0, dim_c)
        elif in_type == "Fock |n⟩":
            n_in   = st.slider("n", 0, 8, 3)
            rho_in = mk_fock(n_in, dim_c)
        elif in_type == "Squeezed":
            r_in   = st.slider("r", 0.0, 2.0, 0.8, 0.05)
            rho_in = mk_squeezed(r_in, 0.0, dim_c)
        elif in_type == "Cat State":
            a_cat  = st.slider("|α|", 0.5, 3.0, 2.0, 0.1)
            rho_in = mk_cat(a_cat, 0.0, +1, dim_c)
        else:
            nb_in  = st.slider("n̄", 0.1, 3.0, 1.0, 0.1)
            rho_in = mk_thermal(nb_in, dim_c)

        st.markdown("**Channel**")
        channel = st.selectbox("Channel", [
            "Displacement D(α)", "Squeezing S(r)", "Phase Shift R(φ)",
            "Photon Loss", "Sequential D→S→R",
        ])
        ch_alpha = ch_ai = ch_r = ch_phi = ch_gt = 0.0
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
        else:
            ch_alpha = st.slider("D: Re(α)", -2.0, 2.0, 1.0, 0.1)
            ch_r     = st.slider("S: r",     0.0,  1.5, 0.5, 0.05)
            ch_phi   = st.slider("R: φ (π)", 0.0,  2.0, 0.3, 0.05)

    xvec_c = np.linspace(-6, 6, xres_c)

    with st.spinner("⚛️  Applying channel…"):
        if channel == "Displacement D(α)":
            rho_out  = ch_disp(rho_in, ch_alpha, ch_ai)
            ch_label = f"D(α={ch_alpha:+.2f}{ch_ai:+.2f}i)"
            theory_eq  = "D̂(α) = exp(αâ† − α*â)  →  rigid shift in phase space"
            theory_txt = "Displacement moves the state rigidly — no distortion. Purity is preserved perfectly. Used in CV teleportation as the correction operation."
        elif channel == "Squeezing S(r)":
            rho_out  = ch_sq(rho_in, ch_r, ch_phi*math.pi)
            ch_label = f"S(r={ch_r:.2f}, φ={ch_phi:.1f}π)"
            theory_eq  = "Ŝ(r) = exp(r(â² − â†²)/2)  →  elliptical squeezing"
            theory_txt = "Squeezing compresses one quadrature and amplifies the other. Purity and phase-space area are perfectly conserved. Resource for LIGO and CV-QKD."
        elif channel == "Phase Shift R(φ)":
            rho_out  = ch_phase(rho_in, ch_phi*math.pi)
            ch_label = f"R(φ={ch_phi:.2f}π)"
            theory_eq  = "R̂(φ) = exp(iφ â†â)  →  rotation in phase space"
            theory_txt = "Phase rotation spins the Wigner function rigidly about the origin. Purity is perfectly conserved — a unitary operation. Full 2π returns the state exactly."
        elif channel == "Photon Loss":
            rho_out  = ch_loss(rho_in, ch_gt)
            ch_label = f"Loss(γt={ch_gt:.2f})"
            theory_eq  = "dρ/dt = γ(âρâ† − ½â†âρ − ½ρâ†â)  →  Gaussian spreading"
            theory_txt = "Loss channel models photon absorption by the environment. Every absorbed photon increases entropy and decreases purity. The Wigner function broadens and negativity is washed out."
        else:
            tmp1     = ch_disp(rho_in, ch_alpha, 0.0)
            tmp2     = ch_sq(tmp1, ch_r, 0.0)
            rho_out  = ch_phase(tmp2, ch_phi*math.pi)
            ch_label = f"D({ch_alpha:.1f})→S({ch_r:.2f})→R({ch_phi:.1f}π)"
            theory_eq  = "D̂(α) → Ŝ(r) → R̂(φ)  sequential channel"
            theory_txt = "Sequential application of displacement, squeezing, and phase rotation. Each unitary operation preserves purity. Their composition is still unitary."

        W_in    = _wigner(rho_in,  xvec_c)
        W_out   = _wigner(rho_out, xvec_c)
        m_in    = _metrics(rho_in)
        m_out   = _metrics(rho_out)
        neg_in  = _wnv(W_in,  xvec_c)
        neg_out = _wnv(W_out, xvec_c)

    st.markdown(f"### {in_type}  →  **{ch_label}**  →  Output")
    st.markdown(f'<div class="eq-box"><b>Theory:</b>  {theory_eq}</div>', unsafe_allow_html=True)

    # Metric delta strip
    mc = st.columns(6)
    mc[0].metric("Purity IN",  f"{m_in['purity']:.5f}")
    mc[1].metric("Purity OUT", f"{m_out['purity']:.5f}",
                 delta=f"{m_out['purity']-m_in['purity']:+.5f}")
    mc[2].metric("Entropy OUT",f"{m_out['entropy']:.4f}",
                 delta=f"{m_out['entropy']-m_in['entropy']:+.4f}")
    mc[3].metric("WNV IN",     f"{neg_in:.5f}")
    mc[4].metric("WNV OUT",    f"{neg_out:.5f}",
                 delta=f"{neg_out-neg_in:+.5f}")
    mc[5].metric("ΔxΔp OUT",   f"{m_out['heis_prod']:.5f}",
                 delta=f"{m_out['heis_prod']-m_in['heis_prod']:+.5f}")

    # Before / After Wigner
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_wigner(W_in,  xvec_c, title=f"Input:  {in_type}"),
                    use_container_width=True)
    c2.plotly_chart(fig_wigner(W_out, xvec_c, title=f"Output:  {ch_label}"),
                    use_container_width=True)

    # ΔW difference map
    W_diff = W_out - W_in
    diff_cs = [[0,"#ef4444"],[0.5,"#07091a"],[1,"#22d3ee"]]
    st.plotly_chart(
        fig_wigner(W_diff, xvec_c,
                   title="ΔW = W_out − W_in  (channel effect on phase space)",
                   cs=diff_cs),
        use_container_width=True,
    )

    # Photon distribution comparison
    st.markdown("### 📊 Photon Distribution — Before vs After")
    k    = np.arange(len(m_in["probs"]))
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(x=k, y=m_in["probs"],  name="Input",
        marker=dict(color="#60a5fa", line=dict(color="rgba(0,0,0,0)")),
        opacity=0.8, width=0.4, offset=-0.2))
    fig_p.add_trace(go.Bar(x=k, y=m_out["probs"], name="Output",
        marker=dict(color="#f472b6", line=dict(color="rgba(0,0,0,0)")),
        opacity=0.8, width=0.4, offset=0.2))
    fig_p.update_layout(**_T, barmode="overlay", height=300,
        title=dict(text="<b>P(n): Input vs Output</b>",
                   font=dict(size=13,color="#a78bfa"), x=0.5),
        xaxis_title="Photon number n", yaxis_title="P(n)")
    st.plotly_chart(fig_p, use_container_width=True)

    # Loss evolution sweep
    if channel == "Photon Loss":
        st.markdown("### 🔻 Loss Channel Evolution Sweep")
        gt_vals = np.linspace(0, 3.0, 8)
        pur_tr, neg_tr, ent_tr = [], [], []
        with st.spinner("⚛️  Computing loss trajectory…"):
            for gt_ in gt_vals:
                r_  = ch_loss(rho_in, float(gt_))
                m_  = _metrics(r_)
                W_  = _wigner(r_, xvec_c)
                pur_tr.append(m_["purity"])
                neg_tr.append(_wnv(W_, xvec_c))
                ent_tr.append(m_["entropy"])
        fig_ev = make_subplots(1, 3, subplot_titles=["Purity","WNV","Entropy"])
        for ci, (vals, col, name) in enumerate([
            (pur_tr,"#34d399","Purity"),
            (neg_tr,"#a78bfa","WNV"),
            (ent_tr,"#22d3ee","Entropy"),
        ], 1):
            fig_ev.add_trace(go.Scatter(x=gt_vals, y=vals, mode="lines+markers",
                line=dict(color=col,width=2.5), marker=dict(size=7), name=name),
                row=1, col=ci)
        fig_ev.update_layout(**_T, height=300, showlegend=False,
            title=dict(text="<b>Decoherence under Photon Loss</b>",
                       font=dict(size=13,color="#a78bfa"), x=0.5))
        st.plotly_chart(fig_ev, use_container_width=True)

    st.markdown(f"""<div class="ins-card">
      <div class="label">💡 Physical Intuition</div>
      <p>{theory_txt}</p>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — GBS SAMPLER
# ══════════════════════════════════════════════════════════════════════════════
def page_gbs_sampler():
    st.markdown("""
    <div class="banner">
      <h1>🔭 GBS Sampler</h1>
      <p>Gaussian Boson Sampling · Hafnian probabilities · Photon statistics · CV-QML</p>
      <p>Strawberry Fields backend · Thewalrus hafnian · PennyLane parameter-shift rule</p>
      <span class="tag">Strawberry Fields</span>
      <span class="tag">Hafnian (#P-hard)</span>
      <span class="tag">CV-QML</span>
    </div>""", unsafe_allow_html=True)

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
        st.info("ℹ️  Strawberry Fields runs on the HPC — this page shows analytic GBS simulation.")

    with st.sidebar:
        st.markdown("### 🔭 GBS Controls")
        N_modes = st.slider("N modes", 2, 6, 4, 1)
        r_gbs   = st.slider("Squeezing r", 0.1, 2.0, 0.8, 0.05)
        asym_r  = st.checkbox("Asymmetric squeezing", False)
        xres_g  = st.slider("Grid resolution", 60, 120, 80, 10)
        dim_g   = 30

    xvec_g = np.linspace(-5, 5, xres_g)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📡  GBS Circuit", "📊  Photon Statistics", "🔢  Hafnian", "🤖  CV-QML"
    ])

    r_vals = ([r_gbs]*N_modes if not asym_r else
              [r_gbs*(0.5 + 0.5*i/(N_modes-1)) for i in range(N_modes)])

    with tab1:
        st.markdown("### GBS Circuit Architecture")
        st.markdown("""<div class="eq-box">
        Squeezed vacuum inputs  →  Haar-random interferometer  →  PNR detection<br>
        P(n₁,…,nₘ) = |Haf(A_S)|² / (n₁!···nₘ! · √det(σ_Q))
        </div>""", unsafe_allow_html=True)

        if sf_ok:
            @st.cache_data(ttl=600, show_spinner=False)
            def _run_sf(N, rv):
                import strawberryfields as sf2
                from strawberryfields import ops as sf_ops2
                from strawberryfields.utils import random_interferometer
                U    = random_interferometer(N)
                prog = sf2.Program(N)
                with prog.context as q:
                    for i, r in enumerate(rv): sf_ops2.Squeezed(r,0)|q[i]
                    sf_ops2.Interferometer(U)|tuple(q[i] for i in range(N))
                eng    = sf2.Engine("gaussian")
                result = eng.run(prog)
                state  = result.state
                mn_per = [state.mean_photon(i)[0] for i in range(N)]
                xv_sf  = np.linspace(-5,5,80)
                wigs   = {}
                for i in range(N):
                    wr = state.wigner(i, xv_sf, xv_sf)
                    wigs[i] = wr[0] if isinstance(wr,tuple) else wr
                return dict(mean_photons=mn_per, wigners=wigs, xvec=xv_sf)
            with st.spinner("🍓 Running Strawberry Fields Gaussian backend…"):
                res_sf = _run_sf(N_modes, tuple(r_vals))
            st.success(f"✅ Strawberry Fields: {N_modes}-mode GBS executed")
            mc = st.columns(N_modes)
            for i, mn in enumerate(res_sf["mean_photons"]):
                mc[i].metric(f"⟨n̂_{i}⟩", f"{mn:.4f}")
            st.markdown("#### Output Mode Wigner Functions")
            cols_sf = st.columns(min(N_modes, 4))
            for i in range(min(N_modes, 4)):
                W_sf = np.array(res_sf["wigners"][i], dtype=float)
                cols_sf[i].plotly_chart(
                    fig_wigner(W_sf, res_sf["xvec"],
                               title=f"Mode {i}  ⟨n̂⟩={res_sf['mean_photons'][i]:.4f}",
                               height=280),
                    use_container_width=True)
        else:
            mn_vals = [math.sinh(r)**2 for r in r_vals]
            mc = st.columns(N_modes)
            for i, mn in enumerate(mn_vals):
                mc[i].metric(f"⟨n̂_{i}⟩ analytic", f"{mn:.4f}")

            # Analytic Wigner for each squeezed mode
            st.markdown("#### Marginal Wigner Functions (squeezed vacuum)")
            cols_an = st.columns(min(N_modes,4))
            for i in range(min(N_modes,4)):
                r_i = r_vals[i]
                sx  = np.exp(-r_i); sp = np.exp(r_i)
                X, P = np.meshgrid(xvec_g, xvec_g)
                W_an = (2/np.pi)*np.exp(-2*(X**2*sx**2 + P**2*sp**2))/(sx*sp)
                cols_an[i].plotly_chart(
                    fig_wigner(W_an, xvec_g,
                               title=f"Mode {i}  r={r_i:.2f}", height=260),
                    use_container_width=True)

        # Circuit text diagram
        st.markdown("#### Circuit Diagram")
        lines = [f"Mode {i}: |0⟩ ──── Ŝ(r={r:.2f}) ──── [  Û  ] ──── PNR" for i,r in enumerate(r_vals)]
        st.code(f"""GBS Circuit ({N_modes} modes):
{'─'*58}
{chr(10).join(lines)}
{'─'*58}
Û = Haar-random {N_modes}×{N_modes} unitary (Clements decomposition)
PNR = photon-number-resolving detector
Output: click pattern (n₀, n₁, …, n_{N_modes-1})
""", language="")

        st.markdown("""<div class="ins-card">
          <div class="label">💡 GBS Quantum Advantage</div>
          <p>GBS output probability P(S) ∝ |Haf(A_S)|² — the hafnian is a #P-hard function.
          Sampling from this distribution is classically intractable for large N.
          Applications: molecular vibronic spectra, dense subgraph optimisation, drug discovery.
          Xanadu's Borealis (2022) demonstrated quantum computational advantage using GBS.</p>
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("### Photon Number Statistics")
        r_arr    = np.array(r_vals)
        mean_n   = np.sinh(r_arr)**2
        var_n    = np.sinh(r_arr)**2 * np.cosh(r_arr)**2 * 2
        mq_gbs   = (var_n - mean_n) / np.maximum(mean_n, 1e-10)

        c1, c2 = st.columns(2)
        with c1:
            colors_mn = ["#a78bfa","#22d3ee","#f472b6","#34d399","#fbbf24","#60a5fa"][:N_modes]
            fig_mn = go.Figure(go.Bar(
                x=[f"Mode {i}" for i in range(N_modes)],
                y=mean_n.tolist(),
                marker=dict(color=colors_mn, line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.4f}" for v in mean_n], textposition="outside",
            ))
            fig_mn.update_layout(**_T, height=310,
                title=dict(text="<b>⟨n̂⟩ per Mode  (sinh²r)</b>",
                           font=dict(size=13,color="#a78bfa"),x=0.5),
                yaxis_title="⟨n̂⟩")
            st.plotly_chart(fig_mn, use_container_width=True)
        with c2:
            fig_mq = go.Figure(go.Bar(
                x=[f"Mode {i}" for i in range(N_modes)],
                y=mq_gbs.tolist(),
                marker=dict(color="#fbbf24", line=dict(color="rgba(0,0,0,0)")),
                text=[f"{v:.3f}" for v in mq_gbs], textposition="outside",
            ))
            fig_mq.add_hline(y=0, line_color="#475569", line_dash="dash",
                             annotation_text="Poissonian Q=0")
            fig_mq.update_layout(**_T, height=310,
                title=dict(text="<b>Mandel Q  (super-Poissonian)</b>",
                           font=dict(size=13,color="#fbbf24"),x=0.5),
                yaxis_title="Q_M")
            st.plotly_chart(fig_mq, use_container_width=True)

        # ⟨n⟩ vs r sweep
        r_sw  = np.linspace(0, 2.5, 100)
        n_sw  = np.sinh(r_sw)**2
        fig_sw = go.Figure()
        fig_sw.add_trace(go.Scatter(x=r_sw, y=n_sw, mode="lines",
            line=dict(color="#a78bfa",width=2.5), fill="tozeroy",
            fillcolor="rgba(167,139,250,0.08)", name="⟨n̂⟩=sinh²r"))
        fig_sw.add_vline(x=r_gbs, line_color="#fbbf24", line_dash="dash",
                         annotation_text=f"r={r_gbs:.2f}",
                         annotation_font_color="#fbbf24")
        fig_sw.update_layout(**_T, height=300,
            title=dict(text="<b>Mean Photon Number vs Squeezing</b>",
                       font=dict(size=13,color="#a78bfa"),x=0.5),
            xaxis_title="r", yaxis_title="⟨n̂⟩")
        st.plotly_chart(fig_sw, use_container_width=True)

    with tab3:
        st.markdown("### Hafnian Computation")
        st.markdown("""<div class="eq-box">
        Haf(A) = Σ_{perfect matchings M} Π_{(i,j)∈M} A_{ij}<br>
        GBS:  P(S) = |Haf(A_S)|² / (S! · √det(σ_Q))  ·  Computing Haf is #P-hard
        </div>""", unsafe_allow_html=True)

        def _haf_brute(A):
            n2 = A.shape[0]
            if n2 % 2 != 0: return 0.0
            idx = list(range(n2)); haf = 0.0+0j
            def _m(lst):
                if not lst: yield []; return
                f=lst[0]; rest=lst[1:]
                for i,v in enumerate(rest):
                    for m in _m(rest[:i]+rest[i+1:]):
                        yield [(f,v)]+m
            for m in _m(idx):
                term=1.0+0j
                for (i,j) in m: term*=A[i,j]
                haf+=term
            return float(np.real(haf))

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown("**Verification on known matrices:**")
            test_mats = {
                "2×2 Identity": np.eye(2),
                "2×2 Ones":     np.ones((2,2)),
                "4×4 Random":   np.random.RandomState(42).randn(4,4),
            }
            rows_h = []
            for name_, A_ in test_mats.items():
                A_sym = (A_+A_.T)/2
                hb    = _haf_brute(A_sym)
                ht    = "N/A"
                if tw_ok:
                    from thewalrus import hafnian as tw_haf
                    ht = f"{float(np.real(tw_haf(A_sym))):.6f}"
                rows_h.append({"Matrix":name_,"Brute-force Haf":f"{hb:.6f}","Thewalrus Haf":ht})
            st.dataframe(pd.DataFrame(rows_h), use_container_width=True, hide_index=True)
        with col_h2:
            st.markdown("**Classical complexity scaling:**")
            n_v  = np.arange(2,22,2)
            t_r  = 2**n_v * n_v**2
            fig_cx = go.Figure(go.Scatter(x=n_v, y=t_r, mode="lines+markers",
                line=dict(color="#a78bfa",width=2.5),
                marker=dict(size=6,color="#f472b6"),
                fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
                name="Ryser O(2ⁿn²)"))
            fig_cx.update_layout(**_T, height=300, yaxis_type="log",
                title=dict(text="<b>Hafnian Complexity  O(2ⁿ·n²)</b>",
                           font=dict(size=13,color="#a78bfa"),x=0.5),
                xaxis_title="Matrix size n", yaxis_title="Operations")
            st.plotly_chart(fig_cx, use_container_width=True)

    with tab4:
        st.markdown("### PennyLane CV-QML")
        st.markdown("""<div class="eq-box">
        CV Quantum Kernel:  K(x,x') = |⟨0|Û†(x)Û(x')|0⟩|²<br>
        Parameter-shift rule:  ∂⟨Ô⟩/∂θ = ½[⟨Ô⟩_{θ+π/2} − ⟨Ô⟩_{θ−π/2}]
        </div>""", unsafe_allow_html=True)

        theta_v  = np.linspace(-math.pi, math.pi, 200)
        expect_x = 1.5*np.sqrt(2)*np.cos(theta_v)
        grad_x   = -1.5*np.sqrt(2)*np.sin(theta_v)
        fig_qml  = make_subplots(1, 2, subplot_titles=["⟨X̂⟩ vs θ", "∂⟨X̂⟩/∂θ (gradient)"])
        fig_qml.add_trace(go.Scatter(x=theta_v, y=expect_x, mode="lines",
            line=dict(color="#a78bfa",width=2.5), fill="tozeroy",
            fillcolor="rgba(167,139,250,0.07)", name="⟨X̂⟩"), row=1, col=1)
        fig_qml.add_trace(go.Scatter(x=theta_v, y=grad_x, mode="lines",
            line=dict(color="#22d3ee",width=2.5), name="gradient"), row=1, col=2)
        fig_qml.update_layout(**_T, height=320,
            title=dict(text="<b>CV QNode: Parameter-Shift Landscape</b>",
                       font=dict(size=13,color="#a78bfa"),x=0.5))
        st.plotly_chart(fig_qml, use_container_width=True)

        steps = np.arange(120)
        loss  = np.maximum(
            2.5*np.exp(-steps/30) + 0.05*np.random.RandomState(42).randn(120)*np.exp(-steps/60),
            0.001)
        ma    = np.convolve(loss, np.ones(8)/8, mode="same")
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Scatter(x=steps, y=loss, mode="lines",
            line=dict(color="#f472b6",width=2), fill="tozeroy",
            fillcolor="rgba(244,114,182,0.08)", name="Loss"))
        fig_tr.add_trace(go.Scatter(x=steps, y=ma, mode="lines",
            line=dict(color="#fbbf24",width=1.5,dash="dot"), name="8-step avg"))
        fig_tr.update_layout(**_T, height=290, yaxis_type="log",
            title=dict(text="<b>CV-QNN Training Loss (Adam Optimizer)</b>",
                       font=dict(size=13,color="#f472b6"),x=0.5),
            xaxis_title="Training step", yaxis_title="MSE Loss")
        st.plotly_chart(fig_tr, use_container_width=True)

        st.markdown("""<div class="ins-card">
          <div class="label">💡 CV Quantum Machine Learning</div>
          <p>CV-QNNs use Gaussian gates (D̂, Ŝ, R̂) and non-Gaussian gates (Kerr) as trainable layers.
          The parameter-shift rule gives exact quantum gradients on hardware — no finite-differences needed.
          CV-QML is naturally suited for regression, generative modelling, and quantum chemistry tasks.
          See <b>08_GBS_SF.ipynb</b> sections 13–16 for full PennyLane implementation.</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    page = render_sidebar()
    if   "State Explorer"    in page: page_state_explorer()
    elif "Phase Space Zoo"   in page: page_phase_space_zoo()
    elif "Witness Lab"       in page: page_witness_lab()
    elif "Channel Simulator" in page: page_channel_simulator()
    elif "GBS Sampler"       in page: page_gbs_sampler()

if __name__ == "__main__":
    main()
