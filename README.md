<h1>⚛️ CV Quantum Phase Space Explorer</h1>

<div align="center">
  
<h3><em>An Interactive Platform for Exploring Continuous-Variable Quantum Information across Diverse Quantum State Representations</em></h3>

</div>

<br/>

<table align="center">
<tr>
<td align="center">
<a href="https://quantum-phase-space-explorer.streamlit.app/">
<img src="https://img.shields.io/badge/%F0%9F%9A%80%20LIVE%20DASHBOARD-Open%20Now-white?style=for-the-badge&color=6366f1" alt="Live Dashboard"/>
</a>
</td>
<td align="center">
<img src="https://img.shields.io/badge/Strawberry%20Fields-0.23.0-ff4d6d?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" alt="Strawberry Fields"/>
</td>
<td align="center">
<img src="https://img.shields.io/badge/QuTiP-5.2.3-a78bfa?style=for-the-badge&labelColor=1a1a2e" alt="QuTiP"/>
</td>
<td align="center">
<img src="https://img.shields.io/badge/Pennylane-0.42.0-ff6f61?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" alt="Pennylane"/>
</td>
<td align="center">
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" alt="Python"/>
</td>
</tr>
</table>

<br/>

---

## 🌐 Live Dashboard

<div align="center">
  
### Try it instantly -> no installation, no login, no API key

<br/>
  
<a href="https://quantum-phase-space-explorer.streamlit.app/">
<img src="https://img.shields.io/badge/CLICK%20TO%20OPEN%20LIVE%20DASHBOARD%20%E2%86%97-quantum--phase--space--explorer.streamlit.app-6366f1?style=for-the-badge&labelColor=020817" width="80%"/>
</a>

<br/><br/>

**[`🔗 https://quantum-phase-space-explorer.streamlit.app/`](https://quantum-phase-space-explorer.streamlit.app/)**

<br/>

*Explore 8 quantum states · 4 phase-space representations · live Wigner functions · Gaussian Boson Sampling · CV-QML*

</div>

<br/>

<div align="center">

<img src="assets/dashboard_preview.png" alt="CV Quantum Phase Space Explorer Dashboard" width="90%"/>

<br/><br/>

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Dashboard Pages](#-dashboard-pages)
- [Physics Coverage](#-physics-coverage)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Software Stack](#-software-stack)
- [Key Results](#-key-results)
- [Contributing](#-contributing)
- [Code of Conduct](#-code-of-conduct)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔭 Overview

**CV Quantum Phase Space Explorer** is a research-grade interactive platform for the comprehensive study of **continuous-variable (CV) quantum information theory**. Built on QuTiP 5.2, Strawberry Fields 0.23, and PennyLane 0.42, it provides deep quantum-optical simulation and visualisation in a zero-friction web interface.

The project systematically covers **8 quantum state classes**, **6 quantum channel families**, **Gaussian Boson Sampling**, and **CV quantum machine learning**, all backed by full Lindblad master-equation simulations on the **IIT Jodhpur HPC cluster** and served via a **pre-computation architecture** achieving ~3 s dashboard load time with zero heavy-library runtime dependencies.

```
8 quantum states · 4 representations · 6 channels · 124 figures · 1978-line utility library
5 dashboard pages · 14 publication figures · 67-page technical report · 25/25 tests passing
```

| Metric | Value |
|--------|-------|
| Quantum states | 8 (Fock, Coherent, Squeezed, Thermal, Cat, Displaced-Squeezed, GKP, TMSV) |
| Representations | 4 (Wigner W, Husimi Q, Glauber-Sudarshan P, Characteristic χ) |
| Channels simulated | 6 (Displacement, Squeezing, Phase shift, Beam splitter, Lindblad loss, Amplification) |
| Figures generated | 124 across 8 Jupyter notebooks |
| Utility library | `quantum_utils.py` 1978 lines, 15 modules, 25 validation tests |
| Dashboard load | ~3 s (pre-computation architecture) |
| Compute platform | IIT Jodhpur HPC (cn03), QuTiP 5.2.3 + NumPy 2.2.6 |

---

## 🖥️ Dashboard Pages

> **Live at [`quantum-phase-space-explorer.streamlit.app`](https://quantum-phase-space-explorer.streamlit.app/) -> no setup required.**

### 🔬 Page 1 - State Explorer

Interactive exploration of all 8 CV quantum states with live parameter controls.

- State-type dropdown and preset parameter selectors
- Wigner function W(x,p) - 2D heatmap **or** full interactive 3D surface toggle
- Husimi Q-function Q(α)
- Photon number distribution P(n) with Poisson reference overlay
- Density matrix ρ (exact stored or analytically reconstructed from P(n))
- 6 live quantum metric cards: ⟨n⟩, Purity Tr(ρ²), von Neumann entropy, Mandel Q, ΔxΔp, WNV

### 🌌 Page 2 - Phase Space Zoo

Side-by-side 2×4 comparison grid of all 8 states simultaneously.

- Toggle between Wigner W(x,p) and Husimi Q(α) representations
- Non-classicality badge on each panel (classical / non-classical)
- Interactive 2 or 4 column grid control
- Full scorecard comparison table with all metrics and CSV download

### 🧪 Page 3 - Witness Lab

Five-tab quantitative non-classicality dashboard.

- Full witness table with Pandas styling, P-function column, CSV export
- Wigner Negativity Volume (WNV) bar chart with Hudson's theorem correctly stated
- Purity vs. von Neumann Entropy scatter with multiselect group highlighting
- Heisenberg uncertainty product with ΔxΔp = ½ limit annotated
- **Glauber-Sudarshan P-function tab** - analytic interactive plots with live sliders for n̄ and α₀

### ⚡ Page 4 - Channel Simulator

Apply quantum channels and watch the Wigner function evolve.

- Displacement D(α), Squeezing S(r), Phase shift R(φ), Lindblad photon loss
- Parameter slider traverses precomputed Wigner snapshots instantly
- All-snapshots thumbnail strip for at-a-glance evolution view
- 2×2 metric evolution grid: purity, entropy, ⟨n⟩, Heisenberg product vs. channel parameter

### 🔭 Page 5 - GBS & CV-QML

Four-tab Gaussian Boson Sampling and quantum machine learning page.

- GBS circuit visualisation (real Strawberry Fields 4-mode and 6-mode results)
- Photon statistics: mean ⟨n⟩ per mode, Mandel Q, P(n) distribution
- Hafnian complexity: brute-force verification vs. Thewalrus, Ryser O(2ⁿn²) scaling
- CV-QML: ⟨X̂⟩(θ) parameter landscape, parameter-shift gradient, Adam training loss

---

## ⚛️ Physics Coverage

### Quantum States

| State | Key Property | Classical? | WNV | Detected by |
|-------|-------------|-----------|-----|------------|
| Fock \|0⟩ (vacuum) | Ground state, P = δ²(α) | ✅ Yes | 0.000 | — |
| Fock \|n≥1⟩ | Q_M = −1, sub-Poissonian | ❌ No | > 0 | WNV, Mandel Q |
| Coherent \|α⟩ | Laser light, Q_M = 0 | ✅ Yes | 0.000 | — |
| Squeezed \|r,φ⟩ | Sub-shot-noise, LIGO resource | ❌ No* | 0.000 | P-function, Q_M |
| Thermal ρ_th | Bose-Einstein, Q_M = n̄ | ✅ Yes | 0.000 | — |
| Cat \|cat±⟩ | Schrödinger superposition | ❌ No | 0.468 | WNV, P-function |
| Displaced-Squeezed | CV-QKD resource | ❌ No* | 0.000 | P-function, Q_M |
| GKP \|GKP⟩ | Error correction encoding | ❌ No | **0.893** | WNV (largest) |
| TMSV (reduced) | EPR, E_N = 2r/ln2 | ❌ No** | 0.000 | Log-negativity |

*Non-classical via P-function singularity (Gaussian state, W ≥ 0 — consistent with Hudson's theorem).
**Non-classical via bipartite entanglement.

### Phase-Space Representations

| Representation | Formula | Always ≥ 0? | Witnesses NC? |
|---------------|---------|------------|--------------|
| Wigner W(x,p) | Weyl-Wigner transform of ρ | ❌ Can be negative | ✅ WNV > 0 |
| Husimi Q(α) | ⟨α\|ρ\|α⟩/π | ✅ Always | ❌ No |
| P-function P(α) | Diagonal coherent expansion | ❌ Can be singular | ✅ Most sensitive |
| Characteristic χ(ξ) | Tr[ρ D(ξ)] | Complex-valued | ✅ Non-Gaussianity |

### Non-Classicality Witnesses

```
WNV      δ[W] = ∫|W(x,p)| dxdp − 1 > 0    ⟹  non-classical (sufficient, not necessary)
Hudson   W(x,p) ≥ 0 everywhere              ⟺  pure Gaussian state (NOT same as classical)
P-fn     P(α) < 0 or singular               ⟹  non-classical (necessary and sufficient)
Mandel   Q_M < 0                             ⟹  sub-Poissonian, non-classical
QFI      F_Q > n̄                            ⟹  quantum-enhanced sensing beyond SQL
E_N      E_N > 0                             ⟹  entangled (two-mode states)
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard  (app.py v5.0)                  │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐ ┌────────┐   │
│  │  State    │ │  Phase    │ │  Witness  │ │ Channel  │ │  GBS   │   │
│  │ Explorer  │ │Space Zoo  │ │   Lab     │ │   Sim    │ │ CV-QML │   │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └────┬─────┘ └───┬────┘   │
│        └─────────────┴─────────────┴────────────┴───────────┘        │
│                  @st.cache_resource  (~3 s load)                     │
├──────────────────────────────────────────────────────────────────────┤
│              Pre-computed Data Layer  (data/*.pkl)                   │
│  states.pkl (7.9 MB)  ·  channels.pkl (3.8 MB)  ·  gbs.pkl (1.6 MB)  │
├──────────────────────────────────────────────────────────────────────┤
│         quantum_utils.py  (v1.0.1 · 1978 lines · 15 modules)         │
├──────────────────────────────────────────────────────────────────────┤
│              HPC Computation Layer  (generate_data.py)               │
│       QuTiP 5.2.3  ·  Strawberry Fields 0.23  ·  PennyLane 0.42      │
│        NumPy 2.2.6  ·  SciPy 1.13.1  ·  IIT Jodhpur HPC (cn03)       │
└──────────────────────────────────────────────────────────────────────┘
```

**Why pre-computation?** Heavy quantum libraries are incompatible with free-tier cloud memory limits. All quantum computations run once on IIT Jodhpur HPC, serialised to `.pkl` files, committed to the repository, and loaded at dashboard startup in ~3 s. Runtime needs only `streamlit + plotly + numpy + pandas`.

---

## 📁 Repository Structure

```
quantum-phase-space-explorer/
|-- app.py                       # Streamlit dashboard (v5.0, 1349 lines)
|-- quantum_utils.py             # Shared utility library (v1.0.1, 1978 lines)
|-- generate_data.py             # HPC pre-computation script
|-- requirements.txt             # Runtime: streamlit, plotly, numpy, pandas
|-- requirements_hpc.txt         # HPC: qutip, strawberryfields, pennylane
|-- assets/
|   `-- dashboard_preview.png    # Dashboard screenshot (see note below)
|-- notebooks/
|   |-- 01_fock_states.ipynb         (5.5 MB, 14 figures)
|   |-- 02_coherent_states.ipynb     (5.2 MB, 13 figures)
|   |-- 03_squeezed_states.ipynb     (5.7 MB, 13 figures)
|   |-- 04_thermal_states.ipynb      (3.9 MB, 18 figures)
|   |-- 05_cat_noon_states.ipynb     (4.6 MB, 17 figures)
|   |-- 06_quantum_channels.ipynb   (13.3 MB, 19 figures)
|   |-- 07_advanced_states.ipynb     (7.6 MB, 15 figures)
|   `-- 08_GBS_SF.ipynb              (3.0 MB, 15 figures)
|-- data/
|   |-- states.pkl       (7935 KB - all 8 state classes)
|   |-- channels.pkl     (3824 KB - 4 channel sweeps)
|   `-- gbs.pkl          (1597 KB - GBS + CV-QML data)
`-- .streamlit/config.toml
```

---

## 🚀 Installation

### Option A - Live dashboard *(recommended - zero setup)*

Open **[quantum-phase-space-explorer.streamlit.app](https://quantum-phase-space-explorer.streamlit.app/)** in any browser. Done in 3 seconds.

### Option B - Run locally

```bash
git clone https://github.com/Sumitchongder/quantum-phase-space-explorer.git
cd quantum-phase-space-explorer
pip install -r requirements.txt
streamlit run app.py
```

Requires Python ≥ 3.10. No GPU needed.

### Option C - Full HPC / regenerate data

```bash
pip install -r requirements_hpc.txt
python generate_data.py       # ~15-30 min on HPC, produces data/*.pkl
streamlit run app.py
```

### Option D - Jupyter notebooks

```bash
pip install -r requirements_hpc.txt
jupyter notebook notebooks/
```

---

## 📦 Software Stack

| Package | Version | Role |
|---------|---------|------|
| [QuTiP](https://qutip.org/) | 5.2.3 | Quantum states, Wigner/Husimi, mesolve |
| [NumPy](https://numpy.org/) | 2.2.6 | Array operations and linear algebra |
| [SciPy](https://scipy.org/) | 1.13.1 | Special functions, matrix operations |
| [Strawberry Fields](https://strawberryfields.ai/) | 0.23.0 | GBS Gaussian backend |
| [Thewalrus](https://the-walrus.readthedocs.io/) | 0.21.0 | Hafnian computation (#P-hard) |
| [PennyLane](https://pennylane.ai/) | 0.42.0 | CV-QML, parameter-shift gradients |
| [Matplotlib](https://matplotlib.org/) | 3.8.4 | Notebook publication figures |
| [Plotly](https://plotly.com/) | 5.21.0 | Interactive dashboard visualisations |
| [Streamlit](https://streamlit.io/) | 1.35.0 | Web dashboard framework |
| [Pandas](https://pandas.pydata.org/) | 2.2.2 | Data tables and CSV export |

---

## 📊 Key Results

### Wigner Negativity Volume - State Ranking

```
Fock  |5⟩  ████████████████████████  1.062
GKP δ=0.3  ████████████████████░░░░  0.893
Fock  |3⟩  ██████████████████░░░░░░  0.721
Cat  even  █████████████░░░░░░░░░░░  0.468
Fock  |1⟩  █████████░░░░░░░░░░░░░░░  0.318
Squeezed   ░░░░░░░░░░░░░░░░░░░░░░░░  0.000  ← non-classical via P-function only
Coherent   ░░░░░░░░░░░░░░░░░░░░░░░░  0.000  ← classical
Thermal    ░░░░░░░░░░░░░░░░░░░░░░░░  0.000  ← classical
```

### Comprehensive State Metrics

| State | Purity | Entropy | Mandel Q | WNV | ΔxΔp | QFI |
|-------|--------|---------|---------|-----|------|-----|
| Fock \|0⟩ (vacuum) | 1.000 | 0.000 | N/A | 0.000 | 0.500 | 0.000 |
| Fock \|1⟩ | 1.000 | 0.000 | −1.000 | 0.318 | 1.118 | 4.000 |
| Fock \|3⟩ | 1.000 | 0.000 | −1.000 | 0.721 | 1.803 | 12.000 |
| Coherent α=2 | 1.000 | 0.000 | 0.000 | 0.000 | 0.500 | 16.000 |
| Squeezed r=1 | 1.000 | 0.000 | −0.482 | 0.000 | 0.500 | 13.807 |
| Thermal n̄=2 | 0.333 | 1.585 | +2.000 | 0.000 | 1.225 | 8.000 |
| Cat even α=2 | 1.000 | 0.000 | −0.963 | 0.468 | 0.558 | 64.000 |
| GKP δ=0.3 | 0.921 | 0.115 | −0.612 | 0.893 | 0.622 | 47.200 |

### Validation - 25/25 Tests Passed

```
✅  All state traces Tr(ρ) = 1.000           ✅  Coherent Mandel Q = 0.000 exactly
✅  All pure-state purities = 1.000          ✅  Fock |n≥1⟩ WNV > 0 confirmed
✅  Wigner normalisation ∫W dxdp = 1.0000    ✅  Cat state WNV > 0 confirmed
✅  Husimi Q(α) ≥ 0 everywhere               ✅  Fidelity F(ρ,ρ) = 1 for all states
✅  Vacuum Heisenberg ΔxΔp = 0.5000          ✅  TMSV log-negativity E_N > 0
✅  Squeezed Δx < 1/√2 (below shot noise)    ✅  Hudson's theorem verified numerically
```

---

## 🤝 Contributing

Contributions of all kinds are welcome - new quantum states, channels, visualisations, theory corrections, or documentation.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the full guide including physics standards, QuTiP coding conventions, commit format, and PR checklist.

```bash
git clone https://github.com/<your-username>/quantum-phase-space-explorer.git
git checkout -b feature/your-feature
# make changes, then verify
python -c "from quantum_utils import run_validation_suite; run_validation_suite()"
git commit -m "feat: your description"
```

---

## 📜 Code of Conduct

This project follows the [Contributor Covenant v2.1](CODE_OF_CONDUCT.md). We are committed to a welcoming environment for everyone regardless of background or experience level. Scientific integrity is explicitly required.

---

## 📖 Citation

```bibtex
@software{chongder2025cvqpe,
  author      = {Chongder, Sumit Tapas},
  title       = {{CV Quantum Phase Space Explorer}},
  year        = {2025},
  institution = {Indian Institute of Technology Jodhpur},
  url         = {https://github.com/Sumitchongder/quantum-phase-space-explorer},
  note        = {M.Tech. Quantum Technologies. Live: https://quantum-phase-space-explorer.streamlit.app/}
}
```

| Reference | Relevance |
|-----------|-----------|
| Wigner (1932), *Phys. Rev.* **40**, 749 | Wigner function definition |
| Hudson (1974), *Rep. Math. Phys.* **6**, 249 | Hudson's theorem |
| Kenfack & Życzkowski (2004), *J. Opt. B* **6**, 396 | WNV measure |
| Gottesman, Kitaev & Preskill (2001), *PRA* **64**, 012310 | GKP states |
| Hamilton et al. (2017), *PRL* **119**, 170501 | Gaussian Boson Sampling |
| Johansson, Nation & Nori (2012), *CPC* **183**, 1760 | QuTiP |
| Killoran et al. (2019), *Quantum* **3**, 129 | Strawberry Fields |

---

## 📄 License

MIT License - see [LICENSE](LICENSE). Free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

| | |
|--|--|
| **Supervisor** | Dr. V. Narayanan, Department of Physics, IIT Jodhpur |
| **HPC** | IIT Jodhpur High-Performance Computing facility (cn03) |
| **QuTiP Team** | Johansson, Nation, Nori and collaborators |
| **Xanadu AI** | Strawberry Fields and PennyLane development teams |
| **IIT Jodhpur** | Interdisciplinary Research Platform on Quantum Information & Computation |

---

<div align="center">

<br/>

**Built with ⚛️ at IIT Jodhpur · M.Tech. Quantum Technologies · 2025–26**

<br/>

<a href="https://quantum-phase-space-explorer.streamlit.app/">
<img src="https://img.shields.io/badge/%F0%9F%9A%80%20Open%20Live%20Dashboard-quantum--phase--space--explorer.streamlit.app-6366f1?style=for-the-badge&labelColor=020817" alt="Open Live Dashboard"/>
</a>

<br/><br/>

*If this project helped you, please consider giving it a ⭐ — it helps others discover it*

</div>
