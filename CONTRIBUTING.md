# Contributing to CV Quantum Phase Space Explorer

Thank you for your interest in contributing. This document provides everything you need to make a high-quality contribution — whether you are fixing a typo, adding a new quantum state, or extending the Streamlit dashboard.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Types of Contributions](#types-of-contributions)
- [Coding Standards](#coding-standards)
- [Physics Standards](#physics-standards)
- [Testing](#testing)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

---

## Quick Start

```bash
# 1. Fork on GitHub, then clone your fork
git clone https://github.com/<your-username>/quantum-phase-space-explorer.git
cd quantum-phase-space-explorer

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Install dashboard-only deps (no heavy quantum stack needed for UI changes)
pip install -r requirements.txt

# 4. Install full quantum stack (needed for quantum_utils.py changes)
pip install -r requirements_hpc.txt

# 5. Run the dashboard
streamlit run app.py

# 6. Run the validation suite before committing
python -c "from quantum_utils import run_validation_suite; run_validation_suite()"
```

---

## Development Setup

### Dashboard-only development

If you are working only on `app.py` or visualisation:

```bash
pip install streamlit>=1.32 plotly>=5.18 numpy>=1.24 pandas>=2.0
streamlit run app.py
```

The dashboard loads data from pre-computed `data/*.pkl` files already in the repository. No QuTiP or quantum computation needed.

### Full quantum stack development

For changes to `quantum_utils.py`, notebooks, or `generate_data.py`:

```bash
pip install -r requirements_hpc.txt
# Recommended: Python 3.11, numpy<2.0 for QuTiP 5.x compatibility
```

### Environment verification

```bash
python -c "
import qutip, numpy, scipy, plotly, streamlit
print(f'QuTiP:      {qutip.__version__}')
print(f'NumPy:      {numpy.__version__}')
print(f'SciPy:      {scipy.__version__}')
print(f'Plotly:     {plotly.__version__}')
print(f'Streamlit:  {streamlit.__version__}')
"
```

Expected: QuTiP ≥ 5.0, NumPy ≥ 1.24 (< 2.0 for full QuTiP compatibility), SciPy ≥ 1.11.

---

## Project Architecture

```
quantum-phase-space-explorer/
|-- app.py                  ← Streamlit dashboard (do not import qutip here)
|-- quantum_utils.py        ← All quantum computation lives here
|-- generate_data.py        ← Run on HPC to regenerate data/*.pkl
|-- data/                   ← Pre-computed pickle files (committed to repo)
|-- notebooks/              ← Jupyter notebooks (01–08, independent)
`-- requirements*.txt
```

**Critical rule**: `app.py` must **never** import `qutip`, `strawberryfields`, or `pennylane`. All quantum computation goes in `quantum_utils.py` and is called via `generate_data.py` on HPC. The dashboard only loads `data/*.pkl`.

---

## Types of Contributions

### 🔬 New quantum states

To add a new state (e.g. NOON state, photon-subtracted squeezed):

1. Add a `make_<state>()` function in `quantum_utils.py` following the existing pattern
2. Ensure it returns a valid `qt.Qobj` with `dims` set correctly
3. Add the state to `generate_data.py` with Wigner, Husimi, metrics, and WNV
4. Add a corresponding preset in `app.py` `page_state_explorer()`
5. Add validation tests in `run_validation_suite()`
6. Document the physical formula and properties in a docstring

### 📊 New channels

To add a new channel:

1. Add `apply_<channel>()` in `quantum_utils.py`
2. Add the channel sweep to `generate_data.py`
3. Add to `CHANNEL_THEORY` dict in `app.py`

### 🌐 Dashboard improvements

- Keep all Plotly figures using the established dark theme `_L` layout dict
- All new metrics should appear in `mrow()` or as `st.metric()` cards
- Insight cards use the `.insight` CSS class for consistency

### 📖 Documentation and theory

- Physics claims must be supported by a citation or derivation
- Add references to `references.bib` if modifying the LaTeX report
- Notebook markdown cells should explain the physics before showing code

---

## Coding Standards

### Python style

- Follow **PEP 8** with 4-space indentation
- Maximum line length: 100 characters
- All public functions in `quantum_utils.py` must have docstrings including:
  - Parameters (type and meaning)
  - Returns (type and meaning)
  - Physical formula or reference
  - Known limitations or edge cases

```python
def compute_wigner(rho: qt.Qobj, xvec: np.ndarray) -> np.ndarray:
    """
    Compute the Wigner quasi-probability distribution W(x,p).

    Uses QuTiP's wigner() with g=2 convention (hbar=1, [x,p]=i).
    Formula: W(x,p) = (2/pi) * Tr[rho * D(alpha) * Pi * D†(alpha)]

    Parameters
    ----------
    rho : qt.Qobj
        Density matrix with dims [[N],[N]] set explicitly.
    xvec : np.ndarray
        Phase-space grid points (same for x and p axes).

    Returns
    -------
    np.ndarray
        2D Wigner function array of shape (len(xvec), len(xvec)).
        Normalised so that integral W dx dp = 1.

    References
    ----------
    Wigner (1932), Phys. Rev. 40, 749.
    """
```

### QuTiP conventions

- Always set `rho.dims = [[N], [N]]` after constructing `qt.Qobj(arr)`
- Always use `g=2` in `qt.wigner()` — this enforces [x,p]=i normalisation
- Use `math.factorial` not `np.math.factorial` (removed in NumPy 2.0)
- Wrap `mesolve` calls with `Options(nsteps=15000)` for loss channel stability

---

## Physics Standards

This project is submitted to and reviewed by quantum physics professors. Physics accuracy is non-negotiable.

1. **Every formula must be cited or derived.** Use `\cite{}` in LaTeX, or add a docstring reference in Python.
2. **Hudson's theorem** is about pure Gaussian states — do not state it as "W ≥ 0 ⟺ classical". The correct statement is "W ≥ 0 everywhere ⟺ state is a pure Gaussian state".
3. **Wigner negativity ≠ only non-classicality criterion.** Squeezed vacuum has W ≥ 0 (Gaussian) but is non-classical via its P-function.
4. **Vacuum |0⟩ is classical.** Its P-function is δ²(α) — a non-negative delta function — same as a coherent state at α=0.
5. **Fock states |n≥1⟩ are non-classical.** Their P-function involves nth-order derivatives of a delta function.
6. **Mandel Q = −1 is the minimum**, achieved by all Fock states — not a free parameter to vary.
7. **WNV formula**: δ[W] = ∫|W(x,p)|dxdp − 1, not δ[W] = (1/2)(∫|W|dxdp − 1). Check the convention before reporting.

---

## Testing

Run the full validation suite before every pull request:

```bash
python -c "from quantum_utils import run_validation_suite; run_validation_suite()"
```

All 25 tests must pass. The suite checks:
- Trace = 1 for all states
- Purity = 1 for all pure states
- Wigner normalisation ∫W dxdp ≈ 1 (tolerance 5%)
- Husimi non-negativity Q(α) ≥ 0
- Vacuum Heisenberg product ΔxΔp = 0.500
- Squeezed sub-shot-noise Δx < 1/√2
- WNV > 0 for Fock |3⟩ and cat states
- Fidelity F(ρ,ρ) = 1
- TMSV log-negativity > 0

When adding new features, add corresponding tests to `run_validation_suite()`.

---

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat:  new quantum state or channel
fix:   bug fix
docs:  documentation only
test:  adding or fixing tests
perf:  performance improvement
style: formatting, no logic change
chore: dependency updates, config
```

Examples:
```
feat: add NOON state (N=1..5) to State Explorer
fix: correct WNV formula — use ∫|W|dxdp-1 not (∫|W|dxdp-1)/2
docs: add Hudson theorem derivation to theoretical background
test: add validation for displaced-squeezed Heisenberg product
```

---

## Pull Request Process

1. Ensure all validation tests pass
2. Update `CHANGELOG.md` (or create one) with a summary of changes
3. If adding a new state or channel, update `README.md` tables
4. If changing `quantum_utils.py` in a way that invalidates the pkl files, run `generate_data.py` on HPC and commit the new pkl files
5. Request review from the maintainer

Pull requests are typically reviewed within 3–5 days. Feedback will be constructive and specific.

---

## Issue Reporting

Use [GitHub Issues](https://github.com/Sumitchongder/quantum-phase-space-explorer/issues). Please include:

**For bugs:**
```
Environment:
  Python version: 3.x.x
  QuTiP version:  5.x.x
  NumPy version:  2.x.x
  OS:             Ubuntu 24 / macOS 15 / Windows 11

Expected behaviour:
  [what should happen]

Actual behaviour:
  [what actually happened]

Traceback:
  [paste full traceback]

Minimal reproducible example:
  [shortest code that reproduces the issue]
```

**For feature requests:**
- Describe the physics motivation
- Reference relevant literature
- Indicate whether you are willing to implement it

---

*Thank you for contributing to open quantum science.*
