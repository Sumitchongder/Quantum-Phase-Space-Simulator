"""
CV Quantum Information Dashboard — Static v3.0
IIT Jodhpur · m25iqt013
Zero heavy deps: only streamlit + plotly + numpy + pandas
All quantum data from pre-computed data/*.pkl files
"""
 
import pickle, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Quantum Dashboard · IIT Jodhpur",
    page_icon="⚛️", layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
:root{
  --bg:#020817;--bg2:#060e24;--bg3:#0c1a3a;
  --border:#1a2d5a;--border2:#2a4585;
  --indigo:#6366f1;--cyan:#22d3ee;--pink:#f472b6;--lime:#a3e635;--gold:#fbbf24;
  --text:#e2e8f0;--dim:#64748b;
  --mono:'Space Mono',monospace;--sans:'Syne',sans-serif;
}
[data-testid="stAppViewContainer"]{background:var(--bg);}
[data-testid="stSidebar"]{background:var(--bg2);border-right:1px solid var(--border);}
[data-testid="stHeader"]{background:transparent;}
html,body,[class*="css"]{color:var(--text);}
h1,h2,h3,h4{font-family:var(--sans);letter-spacing:-0.02em;}
h1{font-size:2rem;font-weight:800;}
h2{font-size:1.3rem;font-weight:700;color:var(--cyan);}
h3{font-size:1rem;font-weight:600;color:var(--indigo);}
p,li{font-family:var(--sans);font-size:0.87rem;}
 
/* sidebar */
[data-testid="stSidebar"] label{color:#94a3b8!important;font-family:var(--mono);font-size:0.73rem;}
[data-testid="stSidebar"] p{font-size:0.73rem;color:var(--dim);}
 
/* metrics */
[data-testid="metric-container"]{
  background:linear-gradient(135deg,var(--bg2),var(--bg3));
  border:1px solid var(--border);border-radius:10px;padding:12px 14px;
}
[data-testid="stMetricValue"]{font-family:var(--mono);color:var(--cyan);font-size:1.15rem;font-weight:700;}
[data-testid="stMetricLabel"]{color:var(--dim);font-size:0.68rem;font-family:var(--mono);}
 
/* buttons */
.stButton>button{
  background:linear-gradient(135deg,#312e81,var(--indigo));
  color:#fff;border:none;border-radius:8px;
  font-family:var(--mono);font-size:0.75rem;font-weight:700;
  padding:8px 20px;letter-spacing:0.04em;transition:all .2s;
}
.stButton>button:hover{background:linear-gradient(135deg,var(--indigo),#818cf8);transform:translateY(-1px);}
 
/* selectbox */
[data-baseweb="select"]>div{background:var(--bg3)!important;border-color:var(--border)!important;border-radius:8px!important;font-family:var(--mono);font-size:0.78rem;}
 
/* tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--bg2);border-bottom:2px solid var(--border);gap:2px;padding:0 4px;}
.stTabs [data-baseweb="tab"]{color:var(--dim);font-family:var(--mono);font-size:0.75rem;padding:9px 14px;border-radius:8px 8px 0 0;}
.stTabs [aria-selected="true"]{color:var(--indigo)!important;border-bottom:2px solid var(--indigo)!important;background:var(--bg3);}
 
/* slider */
.stSlider [data-testid="stThumb"]{background:var(--indigo)!important;}
 
/* custom cards */
.banner{
  background:linear-gradient(135deg,var(--bg2),var(--bg3));
  border:1px solid var(--border2);border-radius:16px;
  padding:22px 28px;margin-bottom:18px;position:relative;overflow:hidden;
}
.banner::after{
  content:'';position:absolute;top:-50px;right:-50px;
  width:200px;height:200px;border-radius:50%;
  background:radial-gradient(circle,rgba(99,102,241,.12),transparent 70%);
}
.banner h1{margin:0 0 4px;font-size:1.7rem;color:#fff;}
.banner p{margin:2px 0;color:var(--dim);font-family:var(--mono);font-size:0.72rem;}
.tag{display:inline-block;background:var(--bg3);border:1px solid var(--border2);
  border-radius:20px;padding:2px 10px;font-size:0.68rem;font-family:var(--mono);
  color:var(--cyan);margin:4px 3px 0 0;}
 
.eq{background:var(--bg2);border:1px solid var(--border);border-left:3px solid var(--indigo);
  border-radius:0 8px 8px 0;padding:10px 14px;margin:8px 0;
  font-family:var(--mono);font-size:0.8rem;color:#c7d2fe;line-height:1.6;}
 
.insight{background:linear-gradient(135deg,rgba(34,211,238,.04),rgba(99,102,241,.04));
  border:1px solid rgba(34,211,238,.2);border-radius:10px;
  padding:12px 16px;margin:8px 0;}
.insight .t{font-family:var(--mono);font-size:0.67rem;color:var(--cyan);
  text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px;}
.insight p{margin:0;font-size:0.8rem;color:#94a3b8;}
 
.warn{background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.25);
  border-radius:8px;padding:10px 14px;margin:6px 0;font-size:0.8rem;color:#fcd34d;}
 
hr{border-color:var(--border);margin:.8rem 0;}
</style>
""", unsafe_allow_html=True)
 
# ── Plotly theme ──────────────────────────────────────────────────────────────
_L = dict(
    paper_bgcolor="#020817", plot_bgcolor="#060e24",
    font=dict(color="#e2e8f0", family="Space Mono, monospace", size=10),
    xaxis=dict(gridcolor="#0b1635", zerolinecolor="#1a2d5a",
               title_font=dict(size=10), tickfont=dict(size=8)),
    yaxis=dict(gridcolor="#0b1635", zerolinecolor="#1a2d5a",
               title_font=dict(size=10), tickfont=dict(size=8)),
    legend=dict(bgcolor="#060e24", bordercolor="#1a2d5a", borderwidth=1,
                font=dict(size=8)),
    margin=dict(l=42, r=20, t=44, b=34),
    colorway=["#6366f1","#22d3ee","#f472b6","#a3e635","#fbbf24","#60a5fa","#fb923c"],
)
 
W_CS   = [[0,"#041030"],[.25,"#1e1b4b"],[.45,"#0f172a"],[.55,"#0f172a"],[.75,"#7f1d1d"],[1,"#db2777"]]
W_CS2  = [[0,"#042f2e"],[.5,"#020817"],[1,"#0891b2"]]
Q_CS   = [[0,"#020817"],[.35,"#1e1b4b"],[.65,"#6d28d9"],[.85,"#c026d3"],[1,"#fbbf24"]]
DM_CS  = "RdBu_r"
 
# ── Data loader ───────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
 
@st.cache_resource(show_spinner=False)
def load_all():
    out = {}
    for name in ("states","channels","gbs"):
        p = DATA_DIR / f"{name}.pkl"
        if p.exists():
            with open(p,"rb") as f:
                out[name] = pickle.load(f)
        else:
            out[name] = None
    return out
 
DATA = load_all()
 
def need(key):
    if DATA[key] is None:
        st.error(f"**`data/{key}.pkl` not found.** Place the pkl files in a `data/` folder next to `app.py`.")
        st.stop()
 
# ── Figure helpers ────────────────────────────────────────────────────────────
def _wr(W):
    wmin, wmax = float(W.min()), float(W.max())
    if wmin >= 0: wmin = -1e-9
    if wmax <= 0: wmax =  1e-9
    return wmin, wmax
 
def fw(W, xv, title="W(x,p)", h=420, show3d=False):
    W = np.array(W); xv = np.array(xv)
    wmin, wmax = _wr(W)
    if show3d:
        wa = max(abs(wmin), abs(wmax), 1e-6)
        fig = go.Figure(go.Surface(
            z=W, x=xv, y=xv, colorscale=W_CS, cmin=-wa, cmax=wa,
            showscale=True, opacity=0.93,
            colorbar=dict(title="W",len=0.7,thickness=11,tickfont=dict(size=8,color="#e2e8f0")),
            hovertemplate="x=%{x:.2f}  p=%{y:.2f}  W=%{z:.4f}<extra></extra>"))
        fig.update_layout(**_L, height=h,
            title=dict(text=f"<b>{title} — 3D</b>",font=dict(size=13,color="#6366f1"),x=0.5),
            scene=dict(
                xaxis=dict(title="x",gridcolor="#0b1635",backgroundcolor="#020817"),
                yaxis=dict(title="p",gridcolor="#0b1635",backgroundcolor="#020817"),
                zaxis=dict(title="W",gridcolor="#0b1635",backgroundcolor="#020817"),
                bgcolor="#020817",camera=dict(eye=dict(x=1.4,y=1.4,z=0.9))))
        return fig
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=W, x=xv, y=xv, colorscale=W_CS, zmin=wmin, zmax=wmax,
        colorbar=dict(title="W",len=0.78,thickness=11,tickfont=dict(size=8,color="#e2e8f0")),
        hovertemplate="x=%{x:.2f}  p=%{y:.2f}  W=%{z:.4f}<extra></extra>"))
    fig.add_trace(go.Contour(z=W, x=xv, y=xv,
        contours=dict(start=0,end=0,size=1,coloring="none"),
        line=dict(color="rgba(255,255,255,0.4)",width=1),showscale=False))
    fig.update_layout(**_L, height=h,
        title=dict(text=f"<b>{title}</b>",font=dict(size=13,color="#6366f1"),x=0.5),
        xaxis_title="x (position)", yaxis_title="p (momentum)")
    return fig
 
def fq(Q, xv, title="Q(α)", h=420):
    fig = go.Figure(go.Heatmap(z=np.array(Q), x=np.array(xv), y=np.array(xv),
        colorscale=Q_CS,
        colorbar=dict(title="Q",len=0.78,thickness=11,tickfont=dict(size=8,color="#e2e8f0")),
        hovertemplate="Re(α)=%{x:.2f}  Im(α)=%{y:.2f}  Q=%{z:.4f}<extra></extra>"))
    fig.update_layout(**_L, height=h,
        title=dict(text=f"<b>{title}</b>",font=dict(size=13,color="#22d3ee"),x=0.5),
        xaxis_title="Re(α)", yaxis_title="Im(α)")
    return fig
 
def fdm(rho, title="ρ", h=360):
    rho = np.array(rho)
    sub = make_subplots(1,2,subplot_titles=["Re(ρ)","Im(ρ)"],horizontal_spacing=0.07)
    for ci,data in enumerate([np.real(rho),np.imag(rho)],1):
        vm = max(float(np.abs(data).max()),1e-6)
        sub.add_trace(go.Heatmap(z=data,colorscale=DM_CS,zmin=-vm,zmax=vm,
            showscale=(ci==1),
            hovertemplate=f"n=%{{y}}  m=%{{x}}  val=%{{z:.4f}}<extra></extra>"),row=1,col=ci)
    sub.update_layout(**_L,height=h,
        title=dict(text=f"<b>Density matrix {title}</b>",font=dict(size=12,color="#6366f1"),x=0.5))
    return sub
 
# ── Density matrix reconstruction from photon probabilities ─────────────────
def reconstruct_rho(sd, n_max=20):
    """
    Reconstruct density matrix from pkl data.
    - If rho stored directly: use it.
    - Pure states (purity≈1): rho_mn = sqrt(P(m)*P(n)) * phase_correction
    - Mixed states: rho = diag(P(n))  [only diagonal available]
    - Special: cat states have alternating-sign coherences
    - Special: thermal is purely diagonal (classical mixture)
    """
    if "rho" in sd:
        return np.array(sd["rho"])
    
    m   = sd["metrics"]
    probs = np.array(m["probs"][:n_max], dtype=float)
    purity = m.get("purity", 0.0)
    
    # Detect state type from structure of probs
    # Squeezed vacuum: only even Fock components non-zero
    # Cat even: only even components
    # Cat odd: only odd components
    probs_safe = np.maximum(probs, 0)
 
    # Normalise to account for truncation at n=29 (high-r squeezed, high-nbar thermal)
    total = probs_safe.sum()
    if total > 1e-10:
        probs_safe = probs_safe / total
 
    if purity > 0.98:
        # Pure state: rho_mn = psi_m * psi_n^*
        odd_sum  = probs_safe[1::2].sum()
        even_sum = probs_safe[0::2].sum()
        psi = np.sqrt(probs_safe)
 
        if odd_sum > 0.98 and even_sum < 0.02:
            # Odd cat: only odd Fock components non-zero
            psi[0::2] = 0
        elif even_sum > 0.98 and odd_sum < 0.02:
            # Even cat / squeezed vacuum: only even Fock components non-zero
            psi[1::2] = 0
        # else: general pure state (displaced-squeezed, GKP, etc.)
 
        rho = np.outer(psi, psi)
    else:
        # Mixed state: only diagonal available (no phase coherence)
        rho = np.diag(probs_safe)
 
    # Final safety: ensure Tr(rho)=1
    tr = float(np.trace(rho))
    if tr > 1e-10:
        rho = rho / tr
    return rho
 
def fpn(probs, mean_n, title="P(n)", h=320):
    probs = np.array(probs); k = np.arange(len(probs))
    poi = np.array([math.exp(-max(mean_n,1e-9))*(max(mean_n,1e-9)**ki)/math.factorial(int(ki))
                    if int(ki)<170 else 0. for ki in k])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=k,y=probs,name="P(n)",marker_color="#6366f1",opacity=0.85))
    if mean_n > 0.01:
        fig.add_trace(go.Scatter(x=k,y=poi,mode="lines+markers",name="Poisson ref",
            line=dict(color="#fbbf24",width=2,dash="dot"),marker=dict(size=3)))
    fig.update_layout(**_L,height=h,
        title=dict(text=f"<b>{title}</b>",font=dict(size=12,color="#6366f1"),x=0.5),
        xaxis_title="Photon number n",yaxis_title="P(n)",bargap=0.15)
    return fig
 
def mrow(m, wnv):
    cols = st.columns(6)
    vals = [
        ("⟨n⟩",    m.get('mean_n','—'),   "Mean photon number"),
        ("Purity",  m.get('purity','—'),   "Tr(ρ²)  [1 = pure state]"),
        ("Entropy", m.get('entropy','—'),  "von Neumann S(ρ) in bits"),
        ("Mandel Q",m.get('mandel_Q','—'),"<0 sub-Poissonian / >0 super"),
        ("Δx·Δp",   m.get('heis_prod','—'),"Heisenberg product ≥ 0.5"),
        ("WNV",     round(wnv,5),          "Wigner Negativity Volume"),
    ]
    for col,(lbl,val,hlp) in zip(cols,vals):
        col.metric(lbl, "—" if val is None else str(val), help=hlp)
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.markdown("""
    <div style="text-align:center;padding:16px 0 8px">
      <div style="font-size:2.2rem;filter:drop-shadow(0 0 10px #6366f1)">⚛️</div>
      <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1rem;
                  color:#e2e8f0;letter-spacing:.06em;margin-top:6px">CV QUANTUM</div>
      <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                  color:#475569;margin-top:2px">IIT Jodhpur · m25iqt013</div>
    </div>
    <hr style="border-color:#1a2d5a;margin:8px 0 14px">
    """, unsafe_allow_html=True)
 
    page = st.sidebar.radio("NAVIGATE", [
        "🔬 State Explorer",
        "🌌 Phase Space Zoo",
        "🧪 Witness Lab",
        "⚡ Channel Simulator",
        "🔭 GBS & CV-QML",
    ])
 
    st.sidebar.markdown("<hr style='border-color:#1a2d5a;margin:12px 0'>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                color:#334155;text-align:center;line-height:1.9">
    runtime deps<br>streamlit · plotly · numpy · pandas<br>
    data: pre-computed pkl (QuTiP+SF)<br>
    © 2025 m25iqt013
    </div>""", unsafe_allow_html=True)
    return page
 
# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — STATE EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
STATE_META = {
    "Fock |n⟩": {
        "emoji":"🎯","color":"#f472b6","classical":False,
        "eq":"â†â|n⟩ = n|n⟩  —  exact n photons  (|0⟩ = vacuum = classical)",
        "desc":"Fock states have <b>exactly n photons</b> — perfect number certainty, total phase uncertainty. "
               "For <b>n ≥ 1</b>: Wigner function goes <b>negative</b> — a rigorous proof of non-classicality. "
               "<b>|0⟩ (vacuum)</b> is a special case: it is classical (P = δ²(α), W = Gaussian). "
               "Used in photonic quantum computing as qubit encodings.",
    },
    "Coherent |α⟩": {
        "emoji":"💡","color":"#fbbf24","classical":True,
        "eq":"|α⟩ = e^{−|α|²/2} Σ (αⁿ/√n!) |n⟩",
        "desc":"Coherent states are <b>laser light</b> — the most classical quantum state. "
               "Gaussian Wigner function, always positive. Minimum uncertainty: ΔxΔp = ½. "
               "Displaced in phase space by Re(α), Im(α).",
    },
    "Squeezed |r,φ⟩": {
        "emoji":"🔧","color":"#6366f1","classical":False,
        "eq":"S(ξ)|0⟩,  ξ = r·e^{iφ}  —  Δx = e^{−r}/√2,  Δp = e^{r}/√2",
        "desc":"Squeezing compresses noise in one quadrature (Δx = e^{−r}/√2 < shot noise) "
               "while amplifying the other (Δp = e^{r}/√2). "
               "The Wigner function is a <b>Gaussian ellipse</b> — always non-negative! "
               "Non-classicality is detected via the <b>P-function</b> (singular), not Wigner negativity. "
               "Δx·Δp = ½ (minimum uncertainty). Used in <b>LIGO</b> for gravitational-wave detection.",
    },
    "Thermal ρ_th": {
        "emoji":"🌡️","color":"#fb923c","classical":True,
        "eq":"ρ_th = Σ [n̄ⁿ/(1+n̄)ⁿ⁺¹] |n⟩⟨n|  —  n̄ = ⟨n⟩",
        "desc":"Thermal (blackbody) radiation — a <b>mixed state</b>. "
               "Gaussian Wigner function (always positive). Super-Poissonian stats: Mandel Q = n̄ > 0. "
               "Maximum von Neumann entropy for a given ⟨n⟩: S = (n̄+1)log(n̄+1) − n̄ log(n̄). "
               "Symplectic eigenvalue ν = 2n̄+1. Classical via both Wigner (≥0) and P-function (Gaussian).",
    },
    "Cat State": {
        "emoji":"🐱","color":"#a3e635","classical":False,
        "eq":"|cat±⟩ = N(|α⟩ ± |−α⟩),  N = 1/√(2±2e^{−2|α|²})",
        "desc":"Quantum superposition of two coherent states. Shows spectacular "
               "<b>interference fringes</b> in the Wigner function between the two Gaussian blobs — "
               "a direct proof of quantum coherence. "
               "Even (+) cat: only even photon numbers. Odd (−) cat: only odd photon numbers. "
               "Fringe visibility ∝ e^{−2|α|²} (decays with decoherence). "
               "Non-classicality: WNV > 0, Mandel Q &lt; −1.",
    },
    "Displaced-Squeezed": {
        "emoji":"🌀","color":"#22d3ee","classical":False,
        "eq":"|α,r⟩ = D(α)S(r)|0⟩",
        "desc":"A squeezed vacuum displaced in phase space. Combines displacement D(α) "
               "and squeezing S(r). Key resource in <b>CV quantum key distribution (QKD)</b>. "
               "Purity = 1 (pure state).",
    },
    "GKP State": {
        "emoji":"🛡️","color":"#c084fc","classical":False,
        "eq":"|GKP⟩ ∝ Σ_n e^{−δ²n²} D(n√π)|0⟩  (δ = envelope width)",
        "desc":"Gottesman-Kitaev-Preskill code states form a <b>grid in phase space</b> "
               "(comb of displaced squeezed states separated by √π). "
               "Encodes a <b>logical qubit</b> in a harmonic oscillator, "
               "correcting displacement errors up to √π/2. "
               "δ controls the envelope: small δ = sharper peaks, better error correction, more energy. "
               "The leading approach for photonic <b>quantum error correction</b> (IBM, Xanadu).",
    },
}
 
def page_state_explorer():
    need("states")
    SD = DATA["states"]
    xvec = SD["xvec"]
 
    st.markdown("""
    <div class="banner">
      <h1>🔬 State Explorer</h1>
      <p>Explore all 8 CV quantum states interactively — Wigner function, Husimi Q, density matrix, photon statistics.</p>
      <span class="tag">Wigner W(x,p)</span><span class="tag">Husimi Q(α)</span>
      <span class="tag">Density matrix ρ</span><span class="tag">Photon dist.</span>
      <span class="tag">8 quantum states</span>
    </div>""", unsafe_allow_html=True)
 
    # ── Controls ──
    col_c, col_m = st.columns([1, 3.2], gap="medium")
 
    with col_c:
        st.markdown("### ⚙️ Controls")
        state_type = st.selectbox("**Quantum State**", list(STATE_META.keys()), key="se_state")
        meta = STATE_META[state_type]
 
        # State-specific selector
        st.markdown("**Preset:**")
        if state_type == "Fock |n⟩":
            n_sel = st.select_slider("Photon number n", options=[0,1,2,3,5], key="se_fock")
            grp, key = "fock", n_sel
            
            # INNER CHECK: If n is 0, override the metadata to show as classical
            if n_sel == 0:
                meta = meta.copy()  # Create a local copy to avoid modifying the global STATE_META
                meta["classical"] = True
                meta["emoji"] = "⭕" # Optional: distinct emoji for vacuum
                meta["desc"] = "The <b>Vacuum State</b> |0⟩ is the lowest energy state. It is technically a Fock state but is Gaussian and purely classical."
 
        elif state_type == "Coherent |α⟩":
            alpha_opts = {"α=0 (vacuum)":"0,0","α=1":"1,0","α=2":"2,0",
                          "α=1+i":"1,1","α=2+2i":"2,2","α=−2":"−2,0","α=2i":"0,2"}
            sel = st.selectbox("Amplitude α", list(alpha_opts.keys()), key="se_coh")
            grp, key = "coherent", alpha_opts[sel]
 
        elif state_type == "Squeezed |r,φ⟩":
            sq_opts = {"r=0.5, φ=0":"0.5,0","r=1.0, φ=0":"1.0,0","r=1.5, φ=0":"1.5,0",
                       "r=2.0, φ=0":"2.0,0","r=1.0, φ=π/2":"1.0,1.57"}
            sel = st.selectbox("Squeezing (r, φ)", list(sq_opts.keys()), key="se_sq")
            grp, key = "squeezed", sq_opts[sel]
 
        elif state_type == "Thermal ρ_th":
            nbar_opts = {"n̄=0.5":0.5,"n̄=1":1,"n̄=2":2,"n̄=5":5,"n̄=10":10}
            sel = st.selectbox("Mean photon number n̄", list(nbar_opts.keys()), key="se_th")
            grp, key = "thermal", nbar_opts[sel]
 
        elif state_type == "Cat State":
            cat_opts = {"Even cat |α|=1.5":"even","Odd cat |α|=1.5":"odd",
                        "Even cat |α|=2.0":"even_2","Odd cat |α|=2.0":"odd_2"}
            sel = st.selectbox("Cat type", list(cat_opts.keys()), key="se_cat")
            grp, key = "cat", cat_opts[sel]
 
        elif state_type == "Displaced-Squeezed":
            ds_opts = {"α=1+i, r=0.5":"(1+1j),0.5,0",
                       "α=2, r=1.0":"(2+0j),1.0,0",
                       "α=1+2i, r=0.8, φ=π/2":"(1+2j),0.8,1.57"}
            sel = st.selectbox("Parameters (α, r, φ)", list(ds_opts.keys()), key="se_ds")
            grp, key = "displaced_squeezed", ds_opts[sel]
 
        else:  # GKP
            gkp_opts = {"δ=0.3 (sharp)":"0.3,3","δ=0.5 (broad)":"0.5,3"}
            sel = st.selectbox("Envelope δ", list(gkp_opts.keys()), key="se_gkp")
            grp, key = "gkp", gkp_opts[sel]
 
        view3d = st.checkbox("🏔️ 3D Wigner surface", value=False, key="se_3d")
        show_dm = st.checkbox("🔢 Show density matrix", value=True, key="se_dm")
 
        # Info card
        cl_txt  = "☀️ CLASSICAL" if meta["classical"] else "⚡ NON-CLASSICAL"
        cl_col  = "#fbbf24" if meta["classical"] else "#f472b6"
        st.markdown(f"""
        <div style="background:rgba(99,102,241,.07);border:1px solid rgba(99,102,241,.25);
                    border-radius:10px;padding:12px 14px;margin-top:12px">
          <div style="font-size:1.5rem">{meta['emoji']}</div>
          <div style="font-family:'Space Mono',monospace;font-size:0.68rem;
                      color:{cl_col};margin:5px 0;font-weight:700">{cl_txt}</div>
          <div style="font-size:0.79rem;color:#cbd5e1;line-height:1.5">{meta['desc']}</div>
        </div>
        <div class="eq" style="margin-top:8px">{meta['eq']}</div>
        """, unsafe_allow_html=True)
 
    # ── Load data ──
    try:
        sd = SD[grp][key]
    except KeyError:
        sd = list(SD[grp].values())[0]
 
    m   = sd["metrics"]
    wnv = sd.get("wnv", 0.0)
 
    with col_m:
        mrow(m, wnv)
        st.markdown("<hr>", unsafe_allow_html=True)
 
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌊 Wigner W(x,p)", "🌀 Husimi Q(α)",
            "📊 Photon dist.", "📐 Density matrix"])
 
        with tab1:
            st.plotly_chart(fw(sd["W"], xvec, f"Wigner — {state_type}", h=440, show3d=view3d),
                            use_container_width=True)
 
            neg = f"{wnv:.5f}" if wnv>0.001 else "≈ 0"
            is_nonclassical = (wnv > 0.001) or (not meta["classical"])
            # Squeezed vacuum: W is analytically Gaussian (always positive)
            # Any small WNV is a grid discretization artifact — correct this display
            is_squeezed_vacuum = (grp == "squeezed")
            if is_squeezed_vacuum:
                wnv_display = "≈ 0 (grid artifact — squeezed vacuum W is Gaussian)"
                nc_verdict = "Non-classical ⚡ — but via <b>P-function</b> (singular), not Wigner negativity"
            elif wnv > 0.001:
                wnv_display = f"{wnv:.5f}"
                nc_verdict = "Non-classical ⚡ — Wigner negativity proven"
            else:
                wnv_display = "≈ 0 (classical)"
                nc_verdict = "Classical ☀️ — W(x,p) ≥ 0 everywhere"
 
            st.markdown(f"""
            <div class="insight"><div class="t">💡 What you're seeing</div>
            <p>W(x,p) is a quasi-probability distribution in phase space.
            <b>Pink/red = negative</b> — impossible for classical states.
            Wigner Negativity Volume = <b>{wnv_display}</b>. {nc_verdict}.<br>
            <b>Hudson's theorem:</b> W ≥ 0 everywhere ⟺ state is a <i>pure Gaussian</i> (coherent or squeezed vacuum).
            </p></div>""", unsafe_allow_html=True)
 
 
        with tab2:
            st.plotly_chart(fq(sd["Q"], xvec, f"Husimi Q — {state_type}", h=440),
                            use_container_width=True)
            st.markdown("""<div class="insight"><div class="t">💡 Husimi Q-function</div>
            <p>Q(α) = ⟨α|ρ|α⟩/π ≥ 0 always — normalized so ∫Q(α)d²α = 1.
            Smoother than Wigner (never negative), so <b>cannot witness non-classicality</b> directly.
            Brighter regions show where the state has highest overlap with coherent states.
            Measured experimentally via <b>heterodyne detection</b>.</p></div>""", unsafe_allow_html=True)
 
        with tab3:
            st.plotly_chart(fpn(m["probs"], m["mean_n"], f"P(n) — {state_type}", h=360),
                            use_container_width=True)
            mq = m.get("mandel_Q")
            if mq is not None:
                regime = "sub-Poissonian ⚡ non-classical" if mq<0 else ("super-Poissonian" if mq>0 else "Poissonian")
                st.markdown(f"""<div class="insight"><div class="t">💡 Photon Statistics</div>
                <p>Mandel Q = {mq:.4f} → <b>{regime}</b>.
                Yellow dashed = Poisson reference (coherent state).
                Q &lt; 0 means fewer fluctuations than a laser — quantum signature.</p></div>""",
                unsafe_allow_html=True)
 
        with tab4:
            if show_dm:
                rho_arr = reconstruct_rho(sd, n_max=20)
                is_stored = "rho" in sd
                is_pure   = m.get("purity", 0) > 0.98
                is_mixed  = not is_pure
 
                # Source note
                if is_stored:
                    src_note = "✅ Exact density matrix (stored in pkl)"
                    src_col  = "#a3e635"
                elif is_pure:
                    src_note = "🔄 Reconstructed from P(n) — pure state: ρ_mn = √(P(m)·P(n))"
                    src_col  = "#22d3ee"
                else:
                    src_note = "🔄 Reconstructed: diagonal only (mixed state — off-diagonals not stored)"
                    src_col  = "#fbbf24"
 
                st.markdown(f'<div style="font-family:var(--mono);font-size:0.7rem;color:{src_col};'
                            f'margin-bottom:8px;padding:6px 10px;background:rgba(0,0,0,0.3);'
                            f'border-radius:6px">{src_note}</div>', unsafe_allow_html=True)
 
                st.plotly_chart(fdm(rho_arr.tolist(), f"— {state_type}", h=380),
                                use_container_width=True)
 
                # Metrics below
                tr_rho2 = float(np.trace(rho_arr @ rho_arr))
                c1dm, c2dm, c3dm = st.columns(3)
                c1dm.metric("Tr(ρ²) reconstructed", f"{tr_rho2:.4f}", help="Should equal purity")
                c2dm.metric("Tr(ρ) check", f"{float(np.trace(rho_arr)):.4f}", help="Should be 1.0")
                c3dm.metric("Max |ρ_mn|", f"{float(np.max(np.abs(rho_arr))):.4f}", help="Largest matrix element")
 
                if is_pure:
                    ptype = "pure"
                    coh_msg = ("Off-diagonal elements ρ_mn = √(P(m)·P(n)) show <b>quantum coherence</b> — "
                               "the hallmark of a pure superposition state.")
                else:
                    ptype = "mixed"
                    coh_msg = ("Only diagonal P(n) shown — mixed states have no phase coherence between "
                               "Fock components. This is <b>classical statistical mixture</b>, not superposition.")
 
                st.markdown(f"""<div class="insight"><div class="t">💡 Density matrix ρ — {ptype} state</div>
                <p>Diagonal elements = photon-number probabilities P(n). {coh_msg}
                Pure states: Tr(ρ²)=1 &nbsp;|&nbsp; Mixed states: Tr(ρ²) &lt; 1.</p></div>""",
                unsafe_allow_html=True)
 
# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PHASE SPACE ZOO
# ════════════════════════════════════════════════════════════════════════════════
ZOO_CONFIGS = [
    ("fock",0,"Fock |0⟩"),("fock",2,"Fock |2⟩"),
    ("coherent","2,0","Coherent |2⟩"),("squeezed","1.0,0","Squeezed r=1"),
    ("thermal",1,"Thermal n̄=1"),("cat","even","Cat (even)"),
    ("displaced_squeezed","(1+1j),0.5,0","Disp-Sq"),("gkp","0.3,3","GKP δ=0.3"),
]
 
def page_phase_space_zoo():
    need("states")
    SD = DATA["states"]; xvec = np.array(SD["xvec"])
 
    st.markdown("""
    <div class="banner">
      <h1>🌌 Phase Space Zoo</h1>
      <p>All 8 quantum states side-by-side — spot the non-classical signatures at a glance.</p>
      <span class="tag">Side-by-side grid</span><span class="tag">Non-classicality</span>
      <span class="tag">Wigner · Husimi Q</span>
    </div>""", unsafe_allow_html=True)
 
    c1, c2 = st.columns([1,3])
    with c1:
        rep  = st.radio("**Representation**", ["Wigner W(x,p)","Husimi Q(α)"], key="zoo_rep")
        ncol = st.radio("**Grid columns**", [2,4], index=1, horizontal=True, key="zoo_cols")
        show_score = st.checkbox("Show scorecard table", value=True, key="zoo_tbl")
 
    # ── Grid ──
    cols = st.columns(ncol)
    for idx,(grp,key,label) in enumerate(ZOO_CONFIGS):
        try: sd = SD[grp][key]
        except KeyError: sd = list(SD[grp].values())[0]
 
        wnv   = sd.get("wnv",0.0)
        arr   = np.array(sd["W"] if "Wigner" in rep else sd["Q"])
        cs    = W_CS if "Wigner" in rep else Q_CS
        wmin  = float(arr.min()); wmax = float(arr.max())
        if "Wigner" in rep:
            if wmin>=0: wmin=-1e-9
            if wmax<=0: wmax=1e-9
 
        # Decide badge and color
        if wnv > 0.001 or "Disp-Sq" in label or "Squeezed" in label or "Cat" in label or "GKP" in label:
            badge = "⚡ Non-cl."
            col   = "rgba(244,114,182,0.9)"
        else:
            badge = "☀️ Classical"
            col   = "rgba(251,191,36,0.9)"
 
        fig = go.Figure(go.Heatmap(z=arr,x=xvec,y=xvec,colorscale=cs,
            zmin=wmin,zmax=wmax,showscale=False,
            hovertemplate=f"{label}<br>x=%{{x:.1f}} p=%{{y:.1f}} val=%{{z:.3f}}<extra></extra>"))
        if "Wigner" in rep and wnv>0.001:
            fig.add_trace(go.Contour(z=arr,x=xvec,y=xvec,
                contours=dict(start=0,end=0,size=1,coloring="none"),
                line=dict(color="rgba(255,255,255,0.5)",width=1),showscale=False))
        fig.update_layout(
            paper_bgcolor="#020817",plot_bgcolor="#060e24",
            font=dict(color="#e2e8f0",size=8),
            xaxis=dict(gridcolor="#0b1635",tickfont=dict(size=7),title="x" if "Wigner" in rep else "Re(α)"),
            yaxis=dict(gridcolor="#0b1635",tickfont=dict(size=7),title="p" if "Wigner" in rep else "Im(α)"),
            title=dict(text=f"<b>{label}</b><br><sup style='color:{col}'>{badge}</sup>",
                       font=dict(size=11,color="#e2e8f0"),x=0.5),
            margin=dict(l=28,r=6,t=46,b=22),height=240)
        cols[idx%ncol].plotly_chart(fig, use_container_width=True)
 
    # ── Scorecard ──
    if show_score:
        st.markdown("### 📊 Non-classicality Scorecard")
        rows = []
        for (grp, key, label) in ZOO_CONFIGS:
            try:
                sd = SD[grp][key]
            except:
                sd = list(SD[grp].values())[0]
        
            m = sd["metrics"]
            wnv = sd.get("wnv", 0.0)
            mandel_q = m.get("mandel_Q", None)
        
            # Decision logic for non-classicality
            if wnv > 0.001:
                is_nonclassical = True
            elif mandel_q is not None and mandel_q < 0:
                is_nonclassical = True
            elif "Disp-Sq" in label or "Squeezed" in label or "Cat" in label or "GKP" in label:
                # Explicitly mark these families as non-classical
                is_nonclassical = True
            elif "Thermal" in label or "Fock |0⟩" in label or "Coherent" in label:
                # Classical states
                is_nonclassical = False
            else:
                is_nonclassical = False
        
            rows.append({
                "State": label,
                "⟨n⟩": m["mean_n"],
                "Purity": m["purity"],
                "Entropy S(ρ)": m["entropy"],
                "Mandel Q": mandel_q if mandel_q is not None else "—",
                "WNV": round(wnv, 5),
                "Non-classical": "✅" if is_nonclassical else "❌"
            })
        
        # Render the table only once
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=320)
        st.markdown("""<div class="insight"><div class="t">💡 Reading the table</div>
        <p><b>WNV > 0</b> rigorously proves non-classicality.<br>
        <b>Mandel Q &lt; 0</b> = sub-Poissonian photon statistics (non-classical).<br>
        <b>Purity = 1</b> = pure state. <b>Entropy = 0</b> = no classical uncertainty.<br>
        Thermal, coherent, and vacuum states are classical mixtures even if Mandel Q ≥ 0.<br>
        Displaced squeezed, squeezed, cat, and GKP states are always non-classical.</p>
        </div>""", unsafe_allow_html=True)
 
 
 
# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WITNESS LAB
# ════════════════════════════════════════════════════════════════════════════════
ALL_STATES_WIT = [
    ("fock",0,"Fock |0⟩"),("fock",1,"Fock |1⟩"),("fock",2,"Fock |2⟩"),("fock",3,"Fock |3⟩"),
    ("coherent","0,0","Coherent α=0"),("coherent","1,0","Coherent α=1"),
    ("coherent","2,0","Coherent α=2"),("coherent","2,2","Coherent α=2+2i"),
    ("squeezed","0.5,0","Squeezed r=0.5"),("squeezed","1.0,0","Squeezed r=1"),
    ("squeezed","2.0,0","Squeezed r=2"),("squeezed","1.0,1.57","Squeezed r=1,φ=π/2"),
    ("thermal",0.5,"Thermal n̄=0.5"),("thermal",1,"Thermal n̄=1"),
    ("thermal",5,"Thermal n̄=5"),("thermal",10,"Thermal n̄=10"),
    ("cat","even","Cat even α=1.5"),("cat","odd","Cat odd α=1.5"),
    ("cat","even_2","Cat even α=2"),("cat","odd_2","Cat odd α=2"),
    ("displaced_squeezed","(1+1j),0.5,0","DispSq α=1+i"),
    ("displaced_squeezed","(2+0j),1.0,0","DispSq α=2,r=1"),
    ("gkp","0.3,3","GKP δ=0.3"),("gkp","0.5,3","GKP δ=0.5"),
]
 
def page_witness_lab():
    need("states")
    SD = DATA["states"]
 
    st.markdown("""
    <div class="banner">
      <h1>🧪 Witness Lab</h1>
      <p>Complete quantum metrics and non-classicality witnesses for all 24 states.</p>
      <span class="tag">Wigner negativity</span><span class="tag">Purity vs Entropy</span>
      <span class="tag">Heisenberg</span><span class="tag">Mandel Q</span>
    </div>""", unsafe_allow_html=True)
 
    # Build dataframe
    rows=[]
    for grp,key,label in ALL_STATES_WIT:
        try: sd=SD[grp][key]
        except: continue
        m=sd["metrics"]; wnv=sd.get("wnv",0.0)
        # Determine P-function classicality
        P_CLASS = {
            "coherent":          "δ²(α−α₀) — classical ✅",
            "squeezed":          "Singular / negative — non-classical",
            "thermal":           "Gaussian ≥ 0 — classical ✅",
            "cat":               "Highly singular — non-classical",
            "displaced_squeezed":"Singular — non-classical",
            "gkp":               "Comb of singularities — non-classical",
        }
        # Fock |0⟩ (vacuum) = coherent state α=0  →  P = δ²(α)  →  CLASSICAL
        # Fock |n≥1⟩  →  P involves nth-order derivatives of δ  →  NON-CLASSICAL
        if grp == "fock":
            p_func = "δ²(α) — classical ✅ (vacuum = coherent α=0)" if key == 0 else "∂²ⁿδ/∂αⁿ∂α*ⁿ — singular, non-classical"
        else:
            p_func = P_CLASS.get(grp, "Unknown")
 
        rows.append({"State":label,"⟨n⟩":m["mean_n"],"Purity":m["purity"],
            "Entropy":m["entropy"],"Mandel Q":m.get("mandel_Q",float("nan")),
            "Δx":m["delta_x"],"Δp":m["delta_p"],"ΔxΔp":m["heis_prod"],
            "WNV":round(wnv,5),"P-function":p_func,"NC":wnv>0.001})
    df=pd.DataFrame(rows)
 
    tab1,tab2,tab3,tab4,tab5=st.tabs(["📋 Full table","🌊 WNV bars","🔭 Purity vs Entropy","⚖️ Heisenberg","🔮 P-function"])
 
    with tab1:
        filter_nc = st.checkbox("Show only non-classical states", key="wit_nc")
        dff = df[df["NC"]] if filter_nc else df
        st.dataframe(
            dff.style.applymap(lambda v: "color:#a3e635;font-weight:700" if v else "color:#475569",
                               subset=["NC"])
                     .format({"⟨n⟩":"{:.3f}","Purity":"{:.4f}","Entropy":"{:.4f}",
                               "Mandel Q":"{:.4f}","Δx":"{:.4f}","Δp":"{:.4f}",
                               "ΔxΔp":"{:.4f}","WNV":"{:.5f}"}),
            use_container_width=True, height=480)
        
        # P-function explanation card
        st.markdown("""<div class="insight" style="margin-top:8px">
        <div class="t">💡 Glauber-Sudarshan P-function as a non-classicality witness</div>
        <p>The P-function ρ = ∫P(α)|α⟩⟨α|d²α is the most sensitive non-classicality witness:
        <b>If P(α) is a non-negative regular function → state is CLASSICAL.</b>
        Any negativity or singularity (more singular than a Dirac delta) → <b>NON-CLASSICAL</b>.<br>
        • <b>Coherent</b>: P = δ²(α−α₀) — delta function, non-negative → classical<br>
        • <b>Thermal</b>: P = Gaussian ≥ 0 → classical<br>
        • <b>Fock |0⟩</b>: P = δ²(α) — delta at origin → classical ✅<br>
        • <b>Fock |n≥1⟩</b>: P involves nth-order derivatives of δ → singular → non-classical ⚡<br>
        • <b>Squeezed/Cat/GKP</b>: P is negative or highly singular → non-classical<br>
        <b>Hudson's theorem</b>: W(x,p) ≥ 0 everywhere ⟺ state is a <i>pure Gaussian state</i> (coherent or squeezed vacuum).<br>
        <b>P-function criterion</b>: P(α) ≥ 0 and regular ⟺ state is <i>classical</i> (coherent or thermal mixture).<br>
        Note: squeezed vacuum satisfies Hudson (W≥0) but is non-classical (P singular).
        </p></div>""", unsafe_allow_html=True)
 
    with tab2:
        sort_by = st.radio("Sort by", ["WNV (default)","State name"], horizontal=True, key="wit_sort")
        dfs = df.sort_values("WNV",ascending=False) if "WNV" in sort_by else df.sort_values("State")
        colors = ["#f472b6" if nc else "#fbbf24" for nc in dfs["NC"]]
        fig=go.Figure(go.Bar(x=dfs["State"],y=dfs["WNV"],marker_color=colors,
            text=[f"{v:.4f}" for v in dfs["WNV"]],textposition="outside",textfont=dict(size=7),name="WNV"))
        fig.add_hline(y=0,line_color="white",line_dash="dash",line_width=1)
        fig.update_layout(**_L,height=460,
            title=dict(text="<b>Wigner Negativity Volume</b>  (pink = non-classical)",
                       font=dict(size=13,color="#6366f1"),x=0.5),
            xaxis_tickangle=-45,xaxis_title="",yaxis_title="WNV")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("""<div class="insight"><div class="t">💡 WNV</div>
        <p>WNV = ∫|W(x,p)|dxdp − 1. Any WNV > 0 is a sufficient (not necessary) condition for non-classicality.
        <b>Hudson's theorem</b> (separate): W(x,p) ≥ 0 everywhere ⟺ the state is a pure Gaussian (coherent or squeezed vacuum).
        Coherent and thermal states have WNV = 0. Note: squeezed vacuum also has WNV ≈ 0 but is non-classical via its P-function.</p></div>""",unsafe_allow_html=True)
 
    with tab3:
        grp_sel = st.multiselect("Highlight groups",
            ["fock","coherent","squeezed","thermal","cat","displaced_squeezed","gkp"],
            default=["fock","coherent","cat","thermal"], key="wit_grp")
        PAL={"fock":"#f472b6","coherent":"#fbbf24","squeezed":"#6366f1",
             "thermal":"#fb923c","cat":"#a3e635","displaced_squeezed":"#22d3ee","gkp":"#c084fc"}
        fig=go.Figure()
        for g in grp_sel:
            sub=df[[g in r["State"].lower() or
                    any(g==grp for grp,_,label in ALL_STATES_WIT if label==r["State"])
                    for _,r in df.iterrows()]]
            # simpler: group by checking ALL_STATES_WIT
            rows_g = [(label,r) for (grp,key,label),(_,r) in zip(ALL_STATES_WIT,df.iterrows()) if grp==g]
            if not rows_g: continue
            labels_g=[lb for lb,_ in rows_g]
            rs=[r for _,r in rows_g]
            fig.add_trace(go.Scatter(
                x=[r["Purity"] for r in rs], y=[r["Entropy"] for r in rs],
                mode="markers+text", text=labels_g,
                textposition="top center", textfont=dict(size=7),
                marker=dict(color=PAL.get(g,"#6366f1"),size=11,
                            symbol="diamond" if any(r["NC"] for r in rs) else "circle",
                            line=dict(color="white",width=0.5)),
                name=g))
        fig.update_layout(**_L,height=440,
            title=dict(text="<b>Purity Tr(ρ²) vs von Neumann Entropy S(ρ)</b>",
                       font=dict(size=13,color="#6366f1"),x=0.5),
            xaxis_title="Purity Tr(ρ²)",yaxis_title="Entropy S(ρ) [bits]")
        st.plotly_chart(fig,use_container_width=True)
 
    with tab4:
        fig=go.Figure()
        fig.add_trace(go.Bar(x=df["State"],y=df["ΔxΔp"],
            marker=dict(color=df["ΔxΔp"],colorscale="Plasma",showscale=True,
                colorbar=dict(title="ΔxΔp",thickness=10,len=0.7,tickfont=dict(size=8))),
            text=[f"{v:.4f}" for v in df["ΔxΔp"]],textposition="outside",textfont=dict(size=7)))
        fig.add_hline(y=0.5,line_color="#a3e635",line_dash="dash",
            annotation_text="Heisenberg limit ΔxΔp = ½",
            annotation_font=dict(color="#a3e635",size=10))
        fig.update_layout(**_L,height=460,
            title=dict(text="<b>Heisenberg Uncertainty Product Δx·Δp</b>",
                       font=dict(size=13,color="#6366f1"),x=0.5),
            xaxis_tickangle=-45,xaxis_title="",yaxis_title="Δx · Δp")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("""<div class="eq">
        Heisenberg Uncertainty:  Δx · Δp ≥ ½<br>
        Minimum uncertainty (= ½): Coherent states, Squeezed states (along squeezed axis)<br>
        Thermal / mixed states: Δx·Δp > ½  (excess classical noise)
        </div>""",unsafe_allow_html=True)
 
    with tab5:
        st.markdown("### 🔮 Glauber-Sudarshan P-function Witness")
        st.markdown("""<div class="eq">
        ρ = ∫ P(α)|α⟩⟨α| d²α  — P-representation<br>
        Classical state ⟺  P(α) ≥ 0 everywhere and regular<br>
        Non-classical state ⟺  P(α) &lt; 0  OR  more singular than δ²(α)<br>
        Hudson's theorem:  W(x,p) ≥ 0 everywhere ⟺ state is a <b>pure Gaussian</b> (coherent or squeezed vacuum)<br>
        P-function criterion (stronger):  P(α) ≥ 0 and regular ⟺ state is <b>classical</b> (coherent or thermal)
        </div>""", unsafe_allow_html=True)
 
        # P-function classification chart
        P_LABELS = {
            "coherent":           ("Classical ✅", "#fbbf24", "δ²(α−α₀) — Delta function (non-negative)"),
            "thermal":            ("Classical ✅", "#fb923c", "Gaussian ≥ 0 (non-negative)"),
            "fock_0":             ("Classical ✅", "#34d399", "δ²(α) — vacuum = coherent α=0\n(non-negative delta at origin)"),
            "fock_n":             ("Non-classical ⚡", "#f472b6", "∂²ⁿδ/∂αⁿ∂α*ⁿ — nth-order derivatives of δ\n(singular for n≥1)"),
            "squeezed":           ("Non-classical ⚡", "#6366f1", "Squeezed Gaussian — P < 0 in some region"),
            "cat":                ("Non-classical ⚡", "#a3e635", "Sum of singular terms with interference"),
            "displaced_squeezed": ("Non-classical ⚡", "#22d3ee", "Shifted singular Gaussian with P < 0"),
            "gkp":                ("Non-classical ⚡", "#c084fc", "Comb of singularities"),
        }
 
        # Visual: two-column layout
        cl_grps  = ["coherent","thermal","fock_0"]
        ncl_grps = ["fock_n","squeezed","cat","displaced_squeezed","gkp"]
 
        G_DISPLAY_NAMES = {
            "coherent":"COHERENT |α⟩","thermal":"THERMAL ρ_th",
            "fock_0":"FOCK |0⟩  (vacuum)","fock_n":"FOCK |n≥1⟩",
            "squeezed":"SQUEEZED |r,φ⟩","cat":"CAT STATE",
            "displaced_squeezed":"DISPLACED-SQUEEZED","gkp":"GKP STATE",
        }
        col_cl, col_ncl = st.columns(2)
        with col_cl:
            st.markdown("#### ☀️ Classical states (P ≥ 0 everywhere)")
            for g in cl_grps:
                lbl, col, desc = P_LABELS[g]
                disp = G_DISPLAY_NAMES.get(g, g.upper())
                st.markdown(f"""
                <div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);
                            border-radius:10px;padding:12px 14px;margin:8px 0">
                  <div style="font-family:var(--mono);font-size:0.7rem;color:{col};font-weight:700">{disp}</div>
                  <div style="font-size:0.78rem;color:#94a3b8;white-space:pre-line;margin-top:4px">{desc}</div>
                </div>""", unsafe_allow_html=True)
        with col_ncl:
            st.markdown("#### ⚡ Non-classical states (P singular or negative)")
            for g in ncl_grps:
                lbl, col, desc = P_LABELS[g]
                disp = G_DISPLAY_NAMES.get(g, g.upper())
                st.markdown(f"""
                <div style="background:rgba(244,114,182,0.06);border:1px solid rgba(244,114,182,0.25);
                            border-radius:10px;padding:12px 14px;margin:8px 0">
                  <div style="font-family:var(--mono);font-size:0.7rem;color:{col};font-weight:700">{disp}</div>
                  <div style="font-size:0.78rem;color:#94a3b8;white-space:pre-line;margin-top:4px">{desc}</div>
                </div>""", unsafe_allow_html=True)
 
        # Summary bar chart: P-function classicality score
        st.markdown("#### Classicality comparison across all states")
        # We use WNV as proxy — P-nonclassical <=> WNV > 0 for pure states
        # For thermal we need a different score  
        p_rows = []
        for grp, key, label in ALL_STATES_WIT:
            try: sd=DATA["states"][grp][key]
            except: continue
            wnv=sd.get("wnv",0.0)
            # Fock |0⟩ (vacuum) is classical — P = δ²(α), same as coherent α=0
            # Squeezed vacuum: W≥0 (Gaussian) so WNV≈0, but P-function singular → non-classical
            # We mark squeezed as non-classical correctly but via P-function
            is_classical = (grp in ["coherent","thermal"]) or (grp == "fock" and key == 0)
            p_rows.append({"State":label,"P-classical":is_classical,
                           "WNV":round(wnv,5),
                           "P-function type":P_LABELS.get(grp,("?","#fff","?"))[2].replace("\n"," ")})
        df_p = pd.DataFrame(p_rows)
        
        fig_p = go.Figure()
        for is_cl, color, name in [(True,"#fbbf24","P ≥ 0 (classical)"),(False,"#f472b6","P singular/negative (non-classical)")]:
            mask = df_p["P-classical"]==is_cl
            sub  = df_p[mask]
            fig_p.add_trace(go.Bar(
                x=sub["State"], y=sub["WNV"],
                name=name, marker_color=color, opacity=0.85,
                text=[f"{v:.4f}" for v in sub["WNV"]],
                textposition="outside", textfont=dict(size=7)))
        fig_p.add_hline(y=0, line_color="white", line_dash="dash", line_width=1)
        fig_p.update_layout(**_L, height=380,
            title=dict(text="<b>WNV coloured by P-function classicality</b>",
                       font=dict(size=13,color="#6366f1"),x=0.5),
            xaxis_tickangle=-45, xaxis_title="", yaxis_title="WNV",
            barmode="group")
        st.plotly_chart(fig_p, use_container_width=True)
 
        # Analytic P-function plots for thermal and coherent (the two tractable cases)
        st.markdown("#### Analytic P-functions for tractable cases")
        alpha_grid = np.linspace(-4,4,200)
        X,Y = np.meshgrid(alpha_grid,alpha_grid)
        Z_abs = X**2+Y**2
 
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            nbar_p = st.slider("Thermal n̄", 0.5, 5.0, 1.0, 0.5, key="p_nbar")
            P_th = np.exp(-Z_abs/nbar_p)/(np.pi*nbar_p)
            fig_pth = go.Figure(go.Heatmap(z=P_th,x=alpha_grid,y=alpha_grid,
                colorscale=Q_CS,
                colorbar=dict(title="P(α)",len=0.8,thickness=11,tickfont=dict(size=8,color="#e2e8f0")),
                hovertemplate="Re(α)=%{x:.2f}  Im(α)=%{y:.2f}  P=%{z:.4f}<extra></extra>"))
            fig_pth.update_layout(**_L,height=320,
                title=dict(text=f"<b>Thermal P(α) — n̄={nbar_p:.1f}</b>",
                           font=dict(size=12,color="#fbbf24"),x=0.5),
                xaxis_title="Re(α)",yaxis_title="Im(α)")
            st.plotly_chart(fig_pth,use_container_width=True)
            st.markdown('<div class="eq">P_th(α) = (1/πn̄)·exp(−|α|²/n̄) ≥ 0 always → CLASSICAL</div>',
                        unsafe_allow_html=True)
 
        with col_p2:
            alpha0_re = st.slider("Coherent Re(α₀)", -2.0, 2.0, 1.0, 0.5, key="p_re")
            alpha0_im = st.slider("Coherent Im(α₀)", -2.0, 2.0, 0.0, 0.5, key="p_im")
            # Regularised coherent P: approximate delta with narrow Gaussian
            sigma2 = 0.04
            P_coh = np.exp(-((X-alpha0_re)**2+(Y-alpha0_im)**2)/sigma2)/(np.pi*sigma2)
            P_coh /= P_coh.max()
            fig_pcoh = go.Figure(go.Heatmap(z=P_coh,x=alpha_grid,y=alpha_grid,
                colorscale=Q_CS,
                colorbar=dict(title="P(α)",len=0.8,thickness=11,tickfont=dict(size=8,color="#e2e8f0")),
                hovertemplate="Re(α)=%{x:.2f}  Im(α)=%{y:.2f}  P=%{z:.4f}<extra></extra>"))
            fig_pcoh.update_layout(**_L,height=320,
                title=dict(text=f"<b>Coherent P(α) — regularised δ² at α₀={alpha0_re:.1f}+{alpha0_im:.1f}i</b>",
                           font=dict(size=11,color="#fbbf24"),x=0.5),
                xaxis_title="Re(α)",yaxis_title="Im(α)")
            st.plotly_chart(fig_pcoh,use_container_width=True)
            st.markdown('<div class="eq">P_coh(α) = δ²(α−α₀)  [regularised Gaussian shown] → CLASSICAL</div>',
                        unsafe_allow_html=True)
 
# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CHANNEL SIMULATOR
# ════════════════════════════════════════════════════════════════════════════════
CHANNEL_THEORY = {
    "Displacement D(α)": {
        "eq":"D(α) = exp(αa† − α*a)   ⟨n⟩ → ⟨n⟩ + |α|²",
        "desc":"Shifts the state rigidly in phase space. Purity is <b>perfectly conserved</b>. "
               "Used in CV quantum teleportation as a corrective operation.",
        "key":"displacement_sweep","param":"alpha_re","label":"Displacement α",
    },
    "Squeezing S(r)": {
        "eq":"S(r) = exp(r(a²−a†²)/2)   Δx → e^{−r}Δx",
        "desc":"Stretches Wigner in one axis, compresses in the other. "
               "Purity conserved (unitary). Key resource for <b>LIGO</b> and CV-QKD.",
        "key":"squeezing_sweep","param":"r","label":"Squeezing r",
    },
    "Phase shift R(φ)": {
        "eq":"R(φ) = exp(iφ a†a)   — rotates in phase space",
        "desc":"Pure rotation of the Wigner function. Purity is <b>exactly conserved</b>. "
               "2π rotation = identity. Implemented with a mirror delay line.",
        "key":"phase_sweep","param":"phi","label":"Phase φ (rad)",
    },
    "Loss channel (Lindblad)": {
        "eq":"dρ/dt = γ(aρa† − ½{a†a,ρ})   — Lindblad master equation",
        "desc":"Every photon lost to environment increases <b>entropy</b> and decreases <b>purity</b>. "
               "The most physically important channel (mirrors, fibre, detectors). "
               "Coherent state stays coherent but amplitude decays: α → αe^{−γt/2}. "
               "Fock states and cat states rapidly decohere → classical mixtures.",
        "key":"loss_sweep","param":"gamma_t","label":"γt (loss)",
    },
}
 
def page_channel_simulator():
    need("channels")
    CD = DATA["channels"]; xvec = CD["xvec"]
 
    st.markdown("""
    <div class="banner">
      <h1>⚡ Channel Simulator</h1>
      <p>Watch a coherent state evolve through quantum channels. All data from full Lindblad mesolve (QuTiP).</p>
      <span class="tag">Displacement D(α)</span><span class="tag">Squeezing S(r)</span>
      <span class="tag">Phase shift R(φ)</span><span class="tag">Loss (Lindblad)</span>
    </div>""", unsafe_allow_html=True)
 
    # Sidebar controls
    csel = st.selectbox("**Select channel**", list(CHANNEL_THEORY.keys()), key="ch_sel")
    info = CHANNEL_THEORY[csel]
    series = CD[info["key"]]
    params = [s[info["param"]] for s in series]
 
    tab1,tab2 = st.tabs(["🌊 Wigner evolution","📊 Metrics evolution"])
 
    with tab1:
        c1,c2 = st.columns([1,3])
        with c1:
            idx = st.slider(f"**{info['label']}**", 0, len(series)-1, 0,
                format=f"step %d", key="ch_slider")
            s   = series[idx]
            pv  = params[idx]
            st.metric(info["param"], f"{pv:.3f}")
            st.markdown(f'<div class="eq">{info["eq"]}</div>',unsafe_allow_html=True)
            st.markdown(f'<div class="insight"><div class="t">💡 Physics</div><p>{info["desc"]}</p></div>',
                        unsafe_allow_html=True)
        with c2:
            st.plotly_chart(fw(s["W"],xvec,f"{csel.split('(')[0].strip()}  [{info['param']}={pv:.3f}]",h=460),
                            use_container_width=True)
 
        # Snapshots row
        st.markdown("**Evolution snapshots (start → mid → end):**")
        scols = st.columns(len(series))
        for ci,sv in enumerate(series):
            pv2=params[ci]; arr=np.array(sv["W"]); xv=np.array(xvec)
            wmin,wmax=_wr(arr)
            fig=go.Figure(go.Heatmap(z=arr,x=xv,y=xv,colorscale=W_CS,zmin=wmin,zmax=wmax,showscale=False))
            fig.update_layout(paper_bgcolor="#020817",plot_bgcolor="#060e24",
                xaxis=dict(gridcolor="#0b1635",tickfont=dict(size=6),showticklabels=False),
                yaxis=dict(gridcolor="#0b1635",tickfont=dict(size=6),showticklabels=False),
                title=dict(text=f"<b>{pv2:.2f}</b>",font=dict(size=10,color="#22d3ee"),x=0.5),
                margin=dict(l=4,r=4,t=28,b=4),height=160)
            scols[ci].plotly_chart(fig,use_container_width=True)
 
    with tab2:
        purs   = [s["metrics"]["purity"]  for s in series]
        ents   = [s["metrics"]["entropy"] for s in series]
        mns    = [s["metrics"]["mean_n"]  for s in series]
        heisps = [s["metrics"]["heis_prod"] for s in series]
 
        fig=make_subplots(2,2,subplot_titles=["Purity Tr(ρ²)","von Neumann Entropy",
                                               "Mean photons ⟨n⟩","Heisenberg Δx·Δp"])
        for (ri,ci),ys,col,nm in [
            ((1,1),purs,"#6366f1","Purity"),((1,2),ents,"#f472b6","Entropy"),
            ((2,1),mns,"#22d3ee","⟨n⟩"),((2,2),heisps,"#a3e635","ΔxΔp")]:
            fig.add_trace(go.Scatter(x=params,y=ys,mode="lines+markers",
                line=dict(color=col,width=2.5),marker=dict(size=6),name=nm),row=ri,col=ci)
        if "Loss" in csel:
            fig.add_hline(y=0.5,line_color="#a3e635",line_dash="dot",row=2,col=2)
        fig.update_layout(**_L,height=480,
            title=dict(text=f"<b>Metrics vs {info['label']}</b>",
                       font=dict(size=13,color="#6366f1"),x=0.5))
        st.plotly_chart(fig,use_container_width=True)
 
# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 — GBS & CV-QML
# ════════════════════════════════════════════════════════════════════════════════
def page_gbs():
    need("gbs")
    GD = DATA["gbs"]
 
    st.markdown("""
    <div class="banner">
      <h1>🔭 Gaussian Boson Sampling & CV-QML</h1>
      <p>Real Strawberry Fields GBS circuits, hafnian computation, photon statistics, and PennyLane CV-QML.</p>
      <span class="tag">Strawberry Fields</span><span class="tag">Hafnian (#P-hard)</span>
      <span class="tag">PennyLane CV-QML</span><span class="tag">GBS quantum advantage</span>
    </div>""", unsafe_allow_html=True)
 
    tab1,tab2,tab3,tab4=st.tabs(["🔭 GBS Circuit","📊 Photon statistics","∑ Hafnian","🤖 CV-QML"])
 
    avail={k:v for k,v in GD.items() if isinstance(v,dict) and "mean_photons" in v}
 
    with tab1:
        c1,c2=st.columns([1,3])
        with c1:
            sel_key=st.selectbox("**GBS configuration**",list(avail.keys()),
                format_func=lambda k:(
                    f"{'🍓 SF' if 'sf' in k else '📐 Analytic'} — "
                    f"{GD[k]['N_modes']} modes, r={GD[k]['r_vals'][0]:.2f}"),
                key="gbs_sel")
            gbs=GD[sel_key]
            src="🍓 Strawberry Fields" if gbs.get("source","")=="strawberryfields" else "📐 Analytic"
            st.markdown(f'<div class="warn">{src}</div>',unsafe_allow_html=True)
            N=gbs["N_modes"]
            st.metric("Modes",N)
            st.metric("r per mode",f"{gbs['r_vals'][0]:.2f}")
            st.metric("⟨n⟩ mode 0",f"{gbs['mean_photons'][0]:.4f}")
            st.markdown("""<div class="eq">
            GBS = N squeezed modes<br>→ random interferometer U<br>→ photon-number detection<br>
            P(S) ∝ |Haf(A_S)|² / s!√det(σ_Q)<br>
            Sampling = #P-hard classically!
            </div>""",unsafe_allow_html=True)
 
        with c2:
            # Circuit diagram
            xvec_sf=np.array(gbs["xvec"])
            fig_c=go.Figure()
            for i in range(N):
                fig_c.add_shape(type="line",x0=0,x1=4.8,y0=i,y1=i,line=dict(color="#1a2d5a",width=2))
                fig_c.add_annotation(x=-0.2,y=i,text="|0⟩",showarrow=False,
                    font=dict(color="#22d3ee",size=10,family="Space Mono"))
                r_i=gbs["r_vals"][i] if i<len(gbs["r_vals"]) else gbs["r_vals"][0]
                fig_c.add_shape(type="rect",x0=0.2,x1=1.1,y0=i-.25,y1=i+.25,
                    line=dict(color="#6366f1",width=2),fillcolor="rgba(99,102,241,0.18)")
                fig_c.add_annotation(x=0.65,y=i,text=f"S({r_i:.1f})",showarrow=False,
                    font=dict(color="#c7d2fe",size=8,family="Space Mono"))
                fig_c.add_shape(type="rect",x0=3.6,x1=4.4,y0=i-.25,y1=i+.25,
                    line=dict(color="#f472b6",width=2),fillcolor="rgba(244,114,182,0.1)")
                fig_c.add_annotation(x=4.0,y=i,text="⟨n⟩",showarrow=False,
                    font=dict(color="#f9a8d4",size=8,family="Space Mono"))
                fig_c.add_annotation(x=5.1,y=i,text=f"n̄={gbs['mean_photons'][i]:.3f}",
                    showarrow=False,font=dict(color="#fbbf24",size=9,family="Space Mono"))
            fig_c.add_shape(type="rect",x0=1.3,x1=3.4,y0=-.5,y1=N-.5,
                line=dict(color="#22d3ee",width=2),fillcolor="rgba(34,211,238,0.05)")
            fig_c.add_annotation(x=2.35,y=(N-1)/2,text="Interferometer U",showarrow=False,
                font=dict(color="#22d3ee",size=12,family="Space Mono"))
            fig_c.update_layout(paper_bgcolor="#020817",plot_bgcolor="#060e24",
                xaxis=dict(visible=False,range=[-0.5,6.5]),
                yaxis=dict(visible=False,range=[-0.8,N+.2]),
                height=max(260,N*60+40),margin=dict(l=10,r=10,t=40,b=10),
                title=dict(text=f"<b>GBS Circuit — {N} modes</b>",
                           font=dict(size=13,color="#6366f1"),x=0.5))
            st.plotly_chart(fig_c,use_container_width=True)
 
            # Wigner per mode
            n_show=min(N,4)
            st.markdown("**Wigner functions per mode:**")
            wcols=st.columns(n_show)
            for i in range(n_show):
                k_w=i if i in gbs["wigners"] else list(gbs["wigners"].keys())[i]
                W_i=np.array(gbs["wigners"][k_w])
                wmin,wmax=_wr(W_i)
                fig_wi=go.Figure(go.Heatmap(z=W_i,x=xvec_sf,y=xvec_sf,
                    colorscale=W_CS,zmin=wmin,zmax=wmax,showscale=False))
                fig_wi.update_layout(paper_bgcolor="#020817",plot_bgcolor="#060e24",
                    xaxis=dict(gridcolor="#0b1635",tickfont=dict(size=6),showticklabels=False),
                    yaxis=dict(gridcolor="#0b1635",tickfont=dict(size=6),showticklabels=False),
                    title=dict(text=f"<b>Mode {i}</b>",font=dict(size=10,color="#22d3ee"),x=0.5),
                    margin=dict(l=4,r=4,t=28,b=4),height=180)
                wcols[i].plotly_chart(fig_wi,use_container_width=True)
 
    with tab2:
        c1,c2=st.columns(2)
        gbs0=GD[sel_key] if "sel_key" in dir() else GD[list(avail.keys())[0]]
 
        with c1:
            # Mean photons bar
            fig_mn=go.Figure(go.Bar(
                x=[f"Mode {i}" for i in range(len(gbs0["mean_photons"]))],
                y=[float(v) for v in gbs0["mean_photons"]],
                marker_color="#6366f1",opacity=0.85,
                text=[f"{v:.3f}" for v in gbs0["mean_photons"]],textposition="outside"))
            fig_mn.update_layout(**_L,height=300,
                title=dict(text="<b>⟨n⟩ per mode</b>",font=dict(size=12,color="#6366f1"),x=0.5),
                xaxis_title="Mode",yaxis_title="⟨n⟩")
            st.plotly_chart(fig_mn,use_container_width=True)
 
        with c2:
            # Mandel Q
            fig_mq=go.Figure(go.Bar(
                x=[f"Mode {i}" for i in range(len(gbs0["mq_vals"]))],
                y=[float(v) for v in gbs0["mq_vals"]],
                marker_color="#fbbf24",opacity=0.85,
                text=[f"{v:.3f}" for v in gbs0["mq_vals"]],textposition="outside"))
            fig_mq.add_hline(y=0,line_color="white",line_dash="dash")
            fig_mq.update_layout(**_L,height=300,
                title=dict(text="<b>Mandel Q per mode</b>",font=dict(size=12,color="#fbbf24"),x=0.5),
                xaxis_title="Mode",yaxis_title="Q")
            st.plotly_chart(fig_mq,use_container_width=True)
 
        # Photon number distribution
        pn=np.array(gbs0["photon_dist"]); ns=np.array(gbs0["photon_ns"])
        nbar=float(gbs0["mean_photons"][0])
        therm=np.array([(nbar/(1+nbar))**n/(1+nbar) if nbar>0 else (1.0 if n==0 else 0.0) for n in ns])
        fig_pn=go.Figure()
        fig_pn.add_trace(go.Bar(x=ns,y=pn,name="GBS P(n)",marker_color="#f472b6",opacity=0.85))
        fig_pn.add_trace(go.Scatter(x=ns,y=therm,mode="lines",name="Thermal ref",
            line=dict(color="#fbbf24",dash="dot",width=2)))
        fig_pn.update_layout(**_L,height=300,
            title=dict(text="<b>Mode 0 — Photon number distribution</b>",
                       font=dict(size=12,color="#f472b6"),x=0.5),
            xaxis_title="n",yaxis_title="P(n)")
        st.plotly_chart(fig_pn,use_container_width=True)
        st.markdown("""<div class="insight"><div class="t">💡 GBS photon statistics</div>
        <p>Each mode follows a thermal-like distribution with n̄ = sinh²(r).
        The <b>joint</b> multi-mode distribution is what makes GBS hard to simulate — 
        its probability is |Haf(A_S)|², requiring exponential classical resources.</p>
        </div>""",unsafe_allow_html=True)
 
    with tab3:
        st.markdown("### Hafnian — the #P-hard engine of GBS")
        st.markdown("""<div class="eq">
        Haf(A) = Σ_{perfect matchings M} Π_{(i,j)∈M} A_{ij}<br>
        P(S) = |Haf(A_S)|² / (s₁!·s₂!·… · √det(σ_Q))<br>
        Classical best: Ryser's algorithm O(2ⁿ·n²)  —  intractable for n≳50
        </div>""",unsafe_allow_html=True)
 
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Verification table (brute-force):**")
            df_h=pd.DataFrame(GD.get("hafnian_table",[]))
            if not df_h.empty:
                st.dataframe(df_h,use_container_width=True,hide_index=True)
            st.markdown("""<div class="insight"><div class="t">💡 What is a hafnian?</div>
            <p>For a symmetric matrix A, Haf(A) counts weighted perfect matchings of a graph.
            For a 2×2 identity: only one matching (0,1) → Haf = A₀₁ = 0 (off-diagonal).
            For 2×2 all-ones: Haf = 1. The hafnian generalises the permanent to symmetric matrices.</p>
            </div>""",unsafe_allow_html=True)
 
        with c2:
            # Pick any GBS config that has hafnian data
            gbs_h=GD[list(avail.keys())[0]]
            n_arr=np.array(gbs_h["hafnian_n"]); t_rys=np.array(gbs_h["hafnian_ops"])
            fig_haf=go.Figure()
            fig_haf.add_trace(go.Scatter(x=n_arr,y=t_rys,mode="lines+markers",
                line=dict(color="#6366f1",width=2.5),marker=dict(size=7,color="#f472b6"),
                name="Ryser O(2ⁿn²)"))
            # Classical computer reference (1 TFLOP ≈ 1e12 ops/s)
            fig_haf.add_hline(y=1e12,line_color="#fbbf24",line_dash="dot",
                annotation_text="1 TFLOP/s reference",
                annotation_font=dict(color="#fbbf24",size=9))
            fig_haf.update_layout(**_L,height=340,yaxis_type="log",
                title=dict(text="<b>Hafnian complexity scaling</b>",
                           font=dict(size=12,color="#6366f1"),x=0.5),
                xaxis_title="n (matrix size)",yaxis_title="Operations (log scale)")
            st.plotly_chart(fig_haf,use_container_width=True)
 
        # Squeezing vs mean photon theory curve
        r_th=np.linspace(0,2.5,200)
        fig_sw=go.Figure()
        fig_sw.add_trace(go.Scatter(x=r_th,y=np.sinh(r_th)**2,mode="lines",
            line=dict(color="#6366f1",width=2.5),name="⟨n⟩=sinh²(r)"))
        fig_sw.add_trace(go.Scatter(x=r_th,y=np.cosh(r_th)**2-1,mode="lines",
            line=dict(color="#22d3ee",width=2,dash="dot"),name="⟨n²⟩−⟨n⟩²"))
        for gbs_c in list(avail.values()):
            for i,r_i in enumerate(gbs_c["r_vals"][:1]):
                fig_sw.add_vline(x=r_i,line_color="#fbbf24",line_dash="dash",
                    annotation_text=f"r={r_i:.1f}",annotation_font=dict(color="#fbbf24",size=9))
            break
        fig_sw.update_layout(**_L,height=300,
            title=dict(text="<b>⟨n⟩ = sinh²(r) — squeezing theory</b>",
                       font=dict(size=12,color="#6366f1"),x=0.5),
            xaxis_title="Squeezing r",yaxis_title="⟨n⟩")
        st.plotly_chart(fig_sw,use_container_width=True)
 
    with tab4:
        st.markdown("### Continuous-Variable Quantum Machine Learning")
        st.markdown("""<div class="eq">
        CV QNode circuit:  |ψ(θ)⟩ = D(α)·S(r)·R(φ)·|0⟩<br>
        Observable:  ⟨X̂⟩(θ) = √2·r·cos(θ)<br>
        Parameter-shift rule:  ∂⟨Ô⟩/∂θ = ½[⟨Ô⟩_{θ+π/2} − ⟨Ô⟩_{θ−π/2}]<br>
        CV-QNN layers:  D(α)·S(r)·R(φ) + Kerr nonlinearity
        </div>""",unsafe_allow_html=True)
 
        theta=np.array(GD["qml_theta"])
        exp_x=np.array(GD["qml_exp_x"])
        grad =np.array(GD["qml_gradient"])
 
        c1,c2=st.columns(2)
        with c1:
            fig_q=go.Figure()
            fig_q.add_trace(go.Scatter(x=theta,y=exp_x,mode="lines",
                line=dict(color="#6366f1",width=2.5),name="⟨X̂⟩(θ)"))
            fig_q.update_layout(**_L,height=300,
                title=dict(text="<b>⟨X̂⟩ vs parameter θ</b>",
                           font=dict(size=12,color="#6366f1"),x=0.5),
                xaxis_title="θ (rad)",yaxis_title="⟨X̂⟩")
            st.plotly_chart(fig_q,use_container_width=True)
 
        with c2:
            fig_g=go.Figure()
            fig_g.add_trace(go.Scatter(x=theta,y=grad,mode="lines",
                line=dict(color="#22d3ee",width=2.5),name="∂⟨X̂⟩/∂θ"))
            fig_g.add_hline(y=0,line_color="white",line_dash="dash",line_width=1)
            fig_g.update_layout(**_L,height=300,
                title=dict(text="<b>Gradient ∂⟨X̂⟩/∂θ (parameter-shift)</b>",
                           font=dict(size=12,color="#22d3ee"),x=0.5),
                xaxis_title="θ (rad)",yaxis_title="Gradient")
            st.plotly_chart(fig_g,use_container_width=True)
 
        steps=np.array(GD["qml_steps"]); loss=np.array(GD["qml_loss"])
        show_log = st.checkbox("Log scale loss", value=True, key="gbs_log")
        fig_tr=go.Figure(go.Scatter(x=steps,y=loss,mode="lines",
            line=dict(color="#f472b6",width=2.5),
            fill="tozeroy",fillcolor="rgba(244,114,182,0.08)"))
        fig_tr.update_layout(**_L,height=300,
            yaxis_type="log" if show_log else "linear",
            title=dict(text="<b>CV-QNN Training Loss (Adam optimizer)</b>",
                       font=dict(size=12,color="#f472b6"),x=0.5),
            xaxis_title="Training step",yaxis_title="MSE Loss")
        st.plotly_chart(fig_tr,use_container_width=True)
 
        st.markdown("""<div class="insight"><div class="t">💡 Why CV-QML?</div>
        <p>CV quantum neural networks use <b>Gaussian gates</b> (displacement, squeezing, rotation)
        and <b>non-Gaussian gates</b> (Kerr) as trainable layers.
        The parameter-shift rule gives <b>exact analytic gradients</b> on real quantum hardware.
        Xanadu's Borealis processor demonstrated quantum advantage via GBS in 2022.</p>
        </div>""",unsafe_allow_html=True)
 
# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
def main():
    page = sidebar()
    if   "State Explorer"    in page: page_state_explorer()
    elif "Phase Space Zoo"   in page: page_phase_space_zoo()
    elif "Witness Lab"       in page: page_witness_lab()
    elif "Channel Simulator" in page: page_channel_simulator()
    elif "GBS"               in page: page_gbs()
 
if __name__ == "__main__":
    main()
