<div align="center">

<img src="https://raw.githubusercontent.com/Sumitchongder/quantum-phase-space-explorer/main/assets/banner.png" alt="CV Quantum Phase Space Explorer" width="100%"/>

# ⚛️ CV Quantum Phase Space Explorer

### *Industrial-grade interactive dashboard for Continuous-Variable Quantum Information*

<br/>

[![Live Dashboard](https://img.shields.io/badge/🚀%20LIVE%20DASHBOARD-Open%20App-6366f1?style=for-the-badge&labelColor=0f0f2a&logoColor=white)](https://quantum-phase-space-explorer.streamlit.app/)
&nbsp;&nbsp;
[![GitHub Stars](https://img.shields.io/github/stars/Sumitchongder/quantum-phase-space-explorer?style=for-the-badge&color=fbbf24&labelColor=0f0f2a)](https://github.com/Sumitchongder/quantum-phase-space-explorer/stargazers)
&nbsp;&nbsp;
[![License](https://img.shields.io/badge/License-MIT-22d3ee?style=for-the-badge&labelColor=0f0f2a)](LICENSE)
&nbsp;&nbsp;
[![Python](https://img.shields.io/badge/Python-3.11-a3e635?style=for-the-badge&logo=python&logoColor=white&labelColor=0f0f2a)](https://python.org)

<br/>

> **Explore, visualise, and deeply understand the full landscape of CV quantum states —**
> **Wigner functions, density matrices, phase-space representations, quantum channels,**
> **Gaussian Boson Sampling, and CV-QML — all in one live interactive platform.**

<br/>

---

## 🌐 Live Dashboard

<br/>

<a href="https://quantum-phase-space-explorer.streamlit.app/" target="_blank">
  <img src="https://img.shields.io/badge/──────────────────────────────────────────────────────────────-CLICK%20TO%20OPEN%20LIVE%20DASHBOARD-6366f1?style=for-the-badge&labelColor=020817" width="100%"/>
</a>

<br/><br/>

### 👇 Click below — the dashboard loads in your browser instantly

<br/>

**[`🔬 https://quantum-phase-space-explorer.streamlit.app/`](https://quantum-phase-space-explorer.streamlit.app/)**

<br/>

*No installation. No login. No API key. Opens in 3 seconds.*

<br/>

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Dashboard](#-live-dashboard)
- [Dashboard Pages](#-dashboard-pages)
- [Physics Coverage](#-physics-coverage)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Computational Infrastructure](#-computational-infrastructure)
- [Software Stack](#-software-stack)
- [Key Results](#-key-results)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [Code of Conduct](#-code-of-conduct)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🔭 Overview

**CV Quantum Phase Space Explorer** is a research-grade interactive platform for the comprehensive study of **continuous-variable (CV) quantum information theory**. Built on top of QuTiP 5.2, Strawberry Fields 0.23, and PennyLane 0.42, it provides an unparalleled depth of quantum-optical simulation and visualisation in a zero-friction web interface.

The project systematically covers **8 quantum state classes**, **4 quantum channel families**, **Gaussian Boson Sampling**, and **CV quantum machine learning** — all backed by full Lindblad master equation simulations executed on the **IIT Jodhpur HPC cluster** and served via a **pre-computation architecture** that achieves ~3 s dashboard load time with zero heavy-library runtime dependencies.

```
8 quantum states  ·  4 channels  ·  124 figures  ·  1978-line utility library
5 dashboard pages  ·  14 publication figures  ·  67-page technical report
```

| Metric | Value |
|--------|-------|
| States covered | 8 (Fock, Coherent, Squeezed, Thermal, Cat, Displaced-Squeezed, GKP, TMSV) |
| Representations | 4 (Wigner W, Husimi Q, Glauber-Sudarshan P, Characteristic χ) |
| Channels simulated | 6 (Displacement, Squeezing, Phase shift, Beam splitter, Lindblad loss, Amplification) |
| Figures generated | 124 across 8 Jupyter notebooks |
| Library size | `quantum_utils.py` — 1978 lines, 15 modules |
| Dashboard pages | 5 interactive pages, ~3 s load time |
| Compute platform | IIT Jodhpur HPC (cn03), QuTiP 5.2.3 + NumPy 2.2.6 |

---

## 🖥️ Dashboard Pages

The dashboard is deployed publicly at **[quantum-phase-space-explorer.streamlit.app](https://quantum-phase-space-explorer.streamlit.app/)** and requires no installation.

### Page 1 — 🔬 State Explorer
Interactive exploration of all 8 CV quantum states with live parameter controls.
- State-type dropdown + preset parameter selectors
- Wigner function W(x,p) — 2D heatmap or full 3D surface
- Husimi Q-function Q(α)
- Photon number distribution P(n) with Poisson reference
- Density matrix ρ (exact or reconstructed from P(n))
- 6 live quantum metric cards: ⟨n⟩, Purity, Entropy, Mandel Q, ΔxΔp, WNV

### Page 2 — 🌌 Phase Space Zoo
Side-by-side 2×4 comparison grid of all 8 states.
- Toggle between Wigner W(x,p) and Husimi Q(α) representations
- Non-classicality badge on each panel (classical / non-classical)
- Interactive grid column control (2 or 4 columns)
- Full scorecard comparison table with all metrics

### Page 3 — 🧪 Witness Lab
Five-tab quantitative non-classicality dashboard.
- Full witness table with Pandas styling and CSV download
- Wigner Negativity Volume (WNV) bar chart
- Purity vs. von Neumann Entropy scatter with multiselect group highlighting
- Heisenberg uncertainty product with ½ limit annotated
- **Glauber-Sudarshan P-function tab** — analytic interactive plots for thermal and coherent states with live sliders

### Page 4 — ⚡ Channel Simulator
Apply any quantum channel to a coherent base state.
- Displacement D(α), Squeezing S(r), Phase shift R(φ), Lindblad photon loss
- Parameter slider traverses precomputed Wigner snapshots
- 2×2 metric evolution grid: purity, entropy, ⟨n⟩, Heisenberg product
- All-snapshots thumbnail strip for at-a-glance evolution

### Page 5 — 🔭 GBS & CV-QML
Four-tab Gaussian Boson Sampling and quantum machine learning.
- Circuit visualisation (real Strawberry Fields 4- and 6-mode GBS)
- Photon statistics: mean ⟨n⟩ per mode, Mandel Q, P(n) distribution
- Hafnian verification table (brute-force vs. Thewalrus)
- CV-QML parameter landscape: ⟨X̂⟩(θ), gradient, Adam training loss

---

## ⚛️ Physics Coverage

### Quantum States

| State | Key Property | Classical? | WNV |
|-------|-------------|-----------|-----|
| Fock \|0⟩ (vacuum) | Gaussian ground state | ✅ Yes | 0.000 |
| Fock \|n≥1⟩ | Sub-Poissonian, Q_M = −1 | ❌ No | > 0 |
| Coherent \|α⟩ | Laser light, Q_M = 0, P = δ²(α−α₀) | ✅ Yes | 0.000 |
| Squeezed \|r,φ⟩ | Sub-shot-noise, LIGO resource | ❌ No* | 0.000 |
| Thermal ρ_th | Bose-Einstein stats, Q_M = n̄ | ✅ Yes | 0.000 |
| Cat \|cat±⟩ | Schrödinger superposition, WNV > 0 | ❌ No | 0.468 |
| Displaced-Squeezed | CV-QKD resource | ❌ No* | 0.000 |
| GKP \|GKP⟩ | Quantum error correction encoding | ❌ No | 0.893 |
| TMSV (reduced) | EPR entanglement, E_N = 2r/ln2 | ❌ No** | 0.000 |

*Non-classical via P-function (singular), not Wigner negativity — consistent with Hudson's theorem.
**Non-classical via entanglement (log-negativity > 0).

### Phase-Space Representations

| Representation | Definition | Non-negativity | Detects NC? |
|---------------|-----------|---------------|------------|
| Wigner W(x,p) | Weyl-Wigner transform | Can be negative | ✅ Yes (WNV > 0) |
| Husimi Q(α) | ⟨α\|ρ\|α⟩/π | Always ≥ 0 | ❌ No |
| P-function P(α) | Coherent-state diagonal | Can be singular | ✅ Yes (most sensitive) |
| Characteristic χ(ξ) | Tr[ρ D(ξ)] | Complex-valued | ✅ Yes (non-Gaussianity) |

### Non-Classicality Witnesses

- **Wigner Negativity Volume (WNV)**: δ[W] = ∫|W(x,p)|dxdp − 1
- **Hudson's Theorem**: W ≥ 0 everywhere ⟺ pure Gaussian state
- **Mandel Q Parameter**: Q_M < 0 ⟹ sub-Poissonian, non-classical
- **Glauber-Sudarshan P-function**: P < 0 or singular ⟹ non-classical
- **Quantum Fisher Information**: F_Q > n̄ ⟹ quantum-enhanced sensing
- **Log-Negativity**: E_N > 0 ⟹ entangled (two-mode states)
- **Symplectic Eigenvalues**: ν < ½ ⟹ non-physical (uncertainty principle violation)

### Quantum Channels

```
Displacement  D(α)  →  rigid phase-space translation, purity preserved
Squeezing     S(r)  →  elliptical stretching, purity preserved
Phase shift   R(φ)  →  rotation in phase space, P(n) unchanged
Beam splitter B(θ)  →  two-mode mixing, Hong-Ou-Mandel effect
Photon loss         →  Lindblad master equation dρ/dt = γ(aρa† − ½{a†a, ρ})
Amplification       →  phase-insensitive gain, noise injection
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Streamlit Dashboard (app.py, v5.0)                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │  State   │ │  Phase   │ │ Witness  │ │ Channel  │ │ GBS  │ │
│  │ Explorer │ │Space Zoo │ │   Lab    │ │   Sim    │ │ QML  │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──┬───┘ │
│       └─────────────┴───────────┴─────────────┴──────────┘     │
│                    @st.cache_resource (~3 s load)               │
├─────────────────────────────────────────────────────────────────┤
│              Pre-computed Data Layer (data/*.pkl)               │
│    states.pkl (7.9 MB)  channels.pkl (3.8 MB)  gbs.pkl (1.6 MB)│
├─────────────────────────────────────────────────────────────────┤
│         quantum_utils.py  (v1.0.1, 1978 lines, 15 modules)     │
├─────────────────────────────────────────────────────────────────┤
│              HPC Computation Layer (generate_data.py)           │
│  QuTiP 5.2.3 · Strawberry Fields 0.23 · PennyLane 0.42        │
│  NumPy 2.2.6 · SciPy 1.13.1 · IIT Jodhpur HPC (cn03)         │
└─────────────────────────────────────────────────────────────────┘
```

**Why pre-computation?**
The dashboard runs on Streamlit Cloud's free tier. Heavy quantum libraries (QuTiP, SF, PennyLane) are incompatible with free-tier resource limits. All quantum computations are run once on IIT Jodhpur HPC, serialised to `.pkl` files, committed to the repository, and loaded at dashboard startup in ~3 seconds. The runtime only needs `streamlit`, `plotly`, `numpy`, and `pandas`.

---

## 📁 Repository Structure

```
quantum-phase-space-explorer/
|-- app.py                  # Streamlit dashboard (v5.0, 1349 lines)
|-- quantum_utils.py        # Shared utility library (v1.0.1, 1978 lines)
|-- generate_data.py        # HPC pre-computation script
|-- requirements.txt        # Runtime deps: streamlit, plotly, numpy, pandas
|-- requirements_hpc.txt    # HPC deps: qutip, strawberryfields, pennylane
|-- notebooks/
|   |-- 01_fock_states.ipynb         (5.5 MB, 14 figures)
|   |-- 02_coherent_states.ipynb     (5.2 MB, 13 figures)
|   |-- 03_squeezed_states.ipynb     (5.7 MB, 13 figures)
|   |-- 04_thermal_states.ipynb      (3.9 MB, 18 figures)
|   |-- 05_cat_noon_states.ipynb     (4.6 MB, 17 figures)
|   |-- 06_quantum_channels.ipynb    (13.3 MB, 19 figures)
|   |-- 07_advanced_states.ipynb     (7.6 MB, 15 figures)
|   `-- 08_GBS_SF.ipynb              (3.0 MB, 15 figures)
|-- data/
|   |-- states.pkl       (7935 KB - all 8 state classes)
|   |-- channels.pkl     (3824 KB - 4 channel sweeps)
|   `-- gbs.pkl          (1597 KB - GBS + CV-QML data)
|-- assets/
|   `-- banner.png
`-- .streamlit/config.toml
```

---

## 🚀 Installation

### Option A — Use the live dashboard (recommended)

No installation needed. Open **[quantum-phase-space-explorer.streamlit.app](https://quantum-phase-space-explorer.streamlit.app/)** in any browser.

### Option B — Run locally (dashboard only)

```bash
git clone https://github.com/Sumitchongder/quantum-phase-space-explorer.git
cd quantum-phase-space-explorer
pip install -r requirements.txt
streamlit run app.py
```

Requirements: Python ≥ 3.10, ~50 MB disk. No GPU needed.

### Option C — Full HPC setup (regenerate data)

```bash
# On HPC or local machine with sufficient RAM (≥16 GB recommended)
pip install -r requirements_hpc.txt

# Generate all pre-computed data (takes ~15-30 min on HPC)
python generate_data.py

# data/ folder now contains states.pkl, channels.pkl, gbs.pkl
# Run the dashboard
streamlit run app.py
```

### Option D — Jupyter notebooks

```bash
pip install -r requirements_hpc.txt
jupyter notebook notebooks/
```

Run notebooks in order: `01_fock_states.ipynb` → `08_GBS_SF.ipynb`. Each notebook is self-contained and imports `quantum_utils.py`.

---

## 🖥️ Computational Infrastructure

### quantum_utils.py — Shared Library (v1.0.1)

The entire project is built on a single shared utility library of 1978 lines organised into 15 modules:

| Module | Functions | Purpose |
|--------|-----------|---------|
| State construction | `make_fock`, `make_coherent`, `make_squeezed`, `make_cat`, `make_gkp`, `make_tmsv` | Build density matrices |
| Phase-space | `compute_wigner`, `compute_husimi`, `compute_p_function`, `compute_characteristic` | All 4 representations |
| Metrics | `state_metrics` | Purity, entropy, Mandel Q, quadratures, covariance |
| Witnesses | `wigner_neg_volume`, `nonclassicality_witnesses` | WNV, sub-Poissonian, P-function test |
| Channels | `apply_displacement`, `apply_squeezing`, `apply_loss`, `apply_beam_splitter` | Full CPTP maps |
| Entanglement | `log_negativity`, `entanglement_entropy` | Two-mode measures |
| QFI | `quantum_fisher_info` | Metrological bounds |
| Fidelity | `q_fidelity`, `bures_distance`, `trace_distance` | Distance measures |
| Validation | `run_validation_suite` | 25 automated tests |

### Critical Bug Fixes (NumPy 2.x / QuTiP 5.x compatibility)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `compute_husimi` returned wrong type | QuTiP 5.x `qfunc()` returns array, not tuple | Runtime type-check with fallback |
| `math.factorial` crash | `np.math` removed in NumPy 2.0 | Replaced all with `math.factorial` |
| `TwoSlopeNorm` crash | Requires vmin < 0 < vmax strictly | `snorm(vm)` helper enforces this |
| Wigner rescaled | `g` parameter default changed | Added `g=2` everywhere explicitly |
| `mesolve` non-convergence | Default nsteps too low for γt > 1.5 | `Options(nsteps=15000)` |
| Fidelity out of [0,1] | `scipy.linalg.sqrtm` imaginary residuals on rank-deficient matrices | Replaced with `eigh`-based matrix sqrt |

---

## 📦 Software Stack

| Package | Version | Role |
|---------|---------|------|
| Python | 3.11.9 | Runtime |
| [QuTiP](https://qutip.org/) | 5.2.3 | Quantum states, Wigner, mesolve |
| [NumPy](https://numpy.org/) | 2.2.6 | Array operations |
| [SciPy](https://scipy.org/) | 1.13.1 | Special functions, matrix ops |
| [Strawberry Fields](https://strawberryfields.ai/) | 0.23.0 | GBS Gaussian backend |
| [Thewalrus](https://the-walrus.readthedocs.io/) | 0.21.0 | Hafnian computation |
| [PennyLane](https://pennylane.ai/) | 0.42.0 | CV-QML, parameter-shift |
| [Matplotlib](https://matplotlib.org/) | 3.8.4 | Notebook figures |
| [Plotly](https://plotly.com/) | 5.21.0 | Interactive dashboard plots |
| [Streamlit](https://streamlit.io/) | 1.35.0 | Web dashboard |
| [Pandas](https://pandas.pydata.org/) | 2.2.2 | Data tables |

---

## 📊 Key Results

### Wigner Negativity Volume (WNV) — State Ranking

```
GKP δ=0.3          ████████████████████░░░  0.893
Fock |5⟩           ████████████████████████  1.062
Fock |3⟩           ██████████████████░░░░░░  0.721
Cat even α=2       █████████████░░░░░░░░░░░  0.468
Fock |1⟩           █████████░░░░░░░░░░░░░░░  0.318
Coherent, Thermal  ░░░░░░░░░░░░░░░░░░░░░░░░  0.000 (classical)
Squeezed           ░░░░░░░░░░░░░░░░░░░░░░░░  0.000 (Gaussian — non-classical via P-function)
```

### Comprehensive State Metrics

| State | Purity | Entropy | Mandel Q | WNV | Δx·Δp | QFI |
|-------|--------|---------|---------|-----|-------|-----|
| Fock \|0⟩ (vacuum) | 1.000 | 0.000 | N/A | 0.000 | 0.500 | 0.000 |
| Fock \|1⟩ | 1.000 | 0.000 | −1.000 | 0.318 | 1.118 | 4.000 |
| Fock \|3⟩ | 1.000 | 0.000 | −1.000 | 0.721 | 1.803 | 12.000 |
| Coherent α=2 | 1.000 | 0.000 | 0.000 | 0.000 | 0.500 | 16.000 |
| Squeezed r=1 | 1.000 | 0.000 | −0.482 | 0.000 | 0.500 | 13.807 |
| Thermal n̄=2 | 0.333 | 1.585 | +2.000 | 0.000 | 1.225 | 8.000 |
| Cat even α=2 | 1.000 | 0.000 | −0.963 | 0.468 | 0.558 | 64.000 |
| GKP δ=0.3 | 0.921 | 0.115 | −0.612 | 0.893 | 0.622 | 47.200 |

### Validation — 25/25 Tests Passed

```
✅ All state purities and traces verified
✅ Wigner normalization ∫W dxdp = 1.0000
✅ Husimi non-negativity Q(α) ≥ 0 everywhere
✅ Vacuum Heisenberg product ΔxΔp = 0.5000
✅ Squeezed Δx < 1/√2 (below shot noise)
✅ Coherent Mandel Q = 0 exactly
✅ Fock |n≥1⟩ WNV > 0 confirmed
✅ Fidelity F(ρ,ρ) = 1 for all states
✅ TMSV log-negativity E_N > 0 confirmed
✅ Hudson's theorem verified computationally
```

---

## 📸 Screenshots

### State Explorer — Cat State (3D Wigner)

The cat state Wigner function showing the two coherent-state lobes and central interference fringes, rendered as an interactive 3D surface.

> *Live at: [quantum-phase-space-explorer.streamlit.app](https://quantum-phase-space-explorer.streamlit.app/)*

---

## 🤝 Contributing

We welcome contributions of all kinds — bug fixes, new quantum states, additional channels, improved visualisations, or documentation improvements.

### Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/<your-username>/quantum-phase-space-explorer.git
   cd quantum-phase-space-explorer
   ```
3. **Create a branch** with a descriptive name:
   ```bash
   git checkout -b feature/add-fock-malabar-state
   # or
   git checkout -b fix/wigner-normalization-thermal
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt         # dashboard only
   pip install -r requirements_hpc.txt     # full quantum stack
   ```
5. **Make your changes** and run the validation suite:
   ```bash
   python -c "from quantum_utils import run_validation_suite; run_validation_suite()"
   ```
6. **Commit** with a clear message following [Conventional Commits](https://www.conventionalcommits.org/):
   ```bash
   git commit -m "feat: add NOON state to State Explorer page"
   git commit -m "fix: correct Husimi normalisation for large alpha"
   git commit -m "docs: add derivation of cat state fringe spacing"
   ```
7. **Push** and open a **Pull Request** against `main`.

### Contribution Areas

| Area | Description | Skills needed |
|------|-------------|--------------|
| 🔬 New quantum states | NOON states, photon-subtracted squeezed, compass states | QuTiP, quantum optics |
| 📊 New channels | Amplification channel, thermal noise, beam-splitter network | QuTiP mesolve |
| 🌐 Dashboard UX | New visualisation tabs, improved controls, mobile layout | Streamlit, Plotly |
| 📐 Theory | Derivations, corrections, additional witnesses | Quantum information theory |
| 🧪 Tests | Additional validation tests in `quantum_utils.py` | Python, pytest |
| 📖 Documentation | Notebook documentation, README improvements | Markdown, LaTeX |
| ⚡ Performance | Faster Wigner computation, smaller pkl files | NumPy, optimisation |

### Pull Request Checklist

- [ ] All 25 existing validation tests pass
- [ ] New features include corresponding validation tests
- [ ] Code follows the existing style in `quantum_utils.py`
- [ ] Physics claims are cited or derived explicitly
- [ ] Dashboard changes tested with `streamlit run app.py`
- [ ] Any new pkl data committed alongside code

### Reporting Issues

Please use [GitHub Issues](https://github.com/Sumitchongder/quantum-phase-space-explorer/issues) and include:
- Python / QuTiP / NumPy version (`python -c "import qutip; print(qutip.__version__)"`)
- Full traceback
- Minimal reproducible example

---

## 📜 Code of Conduct

This project is committed to providing a welcoming and inclusive environment for everyone — regardless of experience level, background, nationality, or affiliation.

### Our Standards

**We encourage:**
- Respectful and constructive technical discussion
- Welcoming contributions from beginners and experts alike
- Accurate physics — citing sources and acknowledging uncertainty
- Clear, well-documented code that teaches as well as functions
- Credit to prior work, open-source libraries, and collaborators

**We do not tolerate:**
- Harassment, discrimination, or personal attacks of any kind
- Dismissiveness towards questions from those learning quantum physics
- Misrepresentation of physics, fabrication of results, or plagiarism
- Hostile or unconstructive code review
- Spam, self-promotion unrelated to the project

### Enforcement

Violations may be reported to the project maintainer at the email listed in the GitHub profile. All reports will be reviewed confidentially. Maintainers reserve the right to remove comments, close issues, or ban contributors who violate this code.

This Code of Conduct is adapted from the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

---

## 📖 Citation

If this project contributes to your research, teaching, or work, please cite it:

```bibtex
@software{chongder2025cvqpe,
  author       = {Chongder, Sumit Tapas},
  title        = {{CV Quantum Phase Space Explorer: An Interactive Dashboard
                   for Continuous-Variable Quantum Information}},
  year         = {2025},
  publisher    = {GitHub},
  institution  = {Indian Institute of Technology Jodhpur},
  url          = {https://github.com/Sumitchongder/quantum-phase-space-explorer},
  note         = {M.Tech. Project, Quantum Technologies, IIT Jodhpur.
                  Live dashboard: https://quantum-phase-space-explorer.streamlit.app/}
}
```

### Related Work

This project builds on:

- Wigner, E. P. (1932). *On the quantum correction for thermodynamic equilibrium.* Physical Review, 40, 749.
- Hudson, R. L. (1974). *When is the Wigner quasi-probability density non-negative?* Reports on Mathematical Physics, 6, 249.
- Kenfack, A. & Życzkowski, K. (2004). *Negativity of the Wigner function as an indicator of non-classicality.* J. Optics B, 6, 396.
- Gottesman, D., Kitaev, A. & Preskill, J. (2001). *Encoding a qubit in an oscillator.* Physical Review A, 64, 012310.
- Hamilton, C. S. et al. (2017). *Gaussian Boson Sampling.* Physical Review Letters, 119, 170501.
- Johansson, J. R., Nation, P. D. & Nori, F. (2012). *QuTiP.* Computer Physics Communications, 183, 1760.
- Killoran, N. et al. (2019). *Strawberry Fields.* Quantum, 3, 129.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for full details.

```
MIT License  Copyright (c) 2025  Sumit Tapas Chongder, IIT Jodhpur

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, subject to the conditions in LICENSE.
```

---

## 🙏 Acknowledgements

| | |
|--|--|
| **Supervisor** | Dr. V. Narayanan, Department of Physics, IIT Jodhpur — for expert guidance in quantum optics and photonic quantum systems |
| **HPC** | IIT Jodhpur High-Performance Computing facility (cn03) — for the computational resources that made large-scale QuTiP and Strawberry Fields simulations possible |
| **QuTiP Team** | Johansson, Nation, Nori and collaborators — for an outstanding open-source quantum toolbox |
| **Xanadu AI** | Strawberry Fields and PennyLane teams — for world-class photonic quantum software |
| **Streamlit** | For making research-grade interactive dashboards deployable in minutes |
| **IIT Jodhpur** | Interdisciplinary Research Platform on Quantum Information & Computation — for the academic environment and course framework |

---

<div align="center">

<br/>

**Built with ⚛️ at IIT Jodhpur · Quantum Technologies Programme · 2025–26**

<br/>

[![Live Dashboard](https://img.shields.io/badge/🚀%20OPEN%20LIVE%20DASHBOARD-quantum--phase--space--explorer.streamlit.app-6366f1?style=for-the-badge&labelColor=020817)](https://quantum-phase-space-explorer.streamlit.app/)

<br/>

*If this project helped you, please consider giving it a ⭐ on GitHub*

<br/>

</div>
