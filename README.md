# ⚛️ CV Quantum Information Dashboard

**IIT Jodhpur · m25iqt013**  
*Continuous-Variable Quantum Information — Interactive Streamlit Dashboard*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quantum-phase-space-explorer.streamlit.app/)

---

## 🏗️ Architecture

This dashboard uses a **pre-computed data** architecture to be completely error-free on Streamlit Cloud:

```
generate_data.py    ← runs ONCE on HPC (needs QuTiP, SF, etc.)
       ↓
   data/*.pkl       ← committed to GitHub
       ↓
    app.py          ← Streamlit Cloud (only needs numpy, plotly, pandas)
```

**Why this design?**
- Streamlit Cloud can't install QuTiP, Strawberry Fields, PennyLane reliably
- Pre-computed `.pkl` files = instant load, zero errors, blazing fast
- 100% compatible with free-tier Streamlit Cloud

---

## 🚀 Deployment

### Step 1 — Generate data locally/on HPC

```bash
# Install heavy deps
pip install -r requirements_generate.txt

# Run once — creates data/ folder with .pkl files
python generate_data.py
```

This produces:
- `data/states.pkl` — Wigner, Husimi, density matrices, metrics for all 8 states
- `data/channels.pkl` — Channel evolution sweeps (displacement, squeezing, phase, loss)
- `data/gbs.pkl` — GBS circuits, hafnian tables, CV-QML curves

### Step 2 — Commit everything to GitHub

```bash
git add app.py requirements.txt data/
git commit -m "Add pre-computed quantum data"
git push
```

### Step 3 — Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repo
3. Set main file: `app.py`
4. Done — no build errors!

---

## 📄 Dashboard Pages

| Page | Description |
|------|-------------|
| 🔬 State Explorer | Interactive Wigner/Husimi/density matrix for all 8 states |
| 🌌 Phase Space Zoo | Side-by-side comparison grid, non-classicality scorecard |
| 🧪 Witness Lab | Full metrics table, WNV comparison, purity vs entropy, Heisenberg |
| ⚡ Channel Simulator | Displacement, squeezing, phase, and loss (Lindblad) evolution |
| 🔭 GBS & CV-QML | GBS circuits, hafnian scaling, photon distributions, CV-QNN training |

---

## 🔬 Physics covered

**Quantum states:** Fock |n⟩, Coherent |α⟩, Squeezed |r,φ⟩, Thermal ρ_th, Cat/NOON, Displaced-Squeezed, GKP

**Representations:** Wigner W(x,p), Husimi Q(α), Density matrix ρ, Characteristic function

**Non-classicality:** Wigner negativity volume, Hudson's theorem, Mandel Q, sub-Poissonian stats, stellar rank

**Quantum channels:** D(α), S(r), BS(θ), R(φ), Loss (Lindblad mesolve), Amplification

**GBS:** Strawberry Fields circuits, hafnian computation (thewalrus), photon sampling, #P-hardness

**CV-QML:** PennyLane CV QNodes, parameter-shift gradients, CV-QNN training

---

## 📦 File structure

```
.
├── app.py                    # Main Streamlit app (cloud-safe)
├── generate_data.py          # Run locally to create data/*.pkl
├── requirements.txt          # Cloud runtime deps (lean)
├── requirements_generate.txt # HPC/local deps (heavy)
├── data/
│   ├── states.pkl            # All 8 quantum states
│   ├── channels.pkl          # Channel evolution
│   └── gbs.pkl               # GBS + CV-QML
└── README.md
```

---

*Built with QuTiP · Strawberry Fields · PennyLane · Plotly · Streamlit*
