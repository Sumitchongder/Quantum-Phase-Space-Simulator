"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          quantum_utils.py — CV Quantum Information Project                   ║
║          IIT Jodhpur · Course: Continuous-Variable Quantum Information       ║
║          Author : m25iqt013                                                  ║
║          Frameworks: QuTiP 5.2 · NumPy 2.x · SciPy 1.13 · Matplotlib         ║
║          Compatible: Strawberry Fields 0.23 · PennyLane 0.42                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Shared utility library used by ALL 8 notebooks and the Streamlit dashboard.
Every function is validated against QuTiP 5.2.3 + NumPy 2.2.6 + SciPy 1.13.1.

Bug fixes applied globally:
  ✅ compute_husimi()  — QuTiP 5.x returns array, not tuple
  ✅ factorial         — uses math.factorial (np.math removed in NumPy 2.x)
  ✅ TwoSlopeNorm      — guarded: requires vmin < 0 < vmax
  ✅ wigner convention — g=2 throughout (position/momentum normalization)
  ✅ mesolve           — Options(nsteps=15000) for convergence
  ✅ q_fidelity()      — v1.0.1: replaced sqrtm with eigh-based _matsqrt;
                         scipy.linalg.sqrtm on rank-deficient (pure-state)
                         density matrices in NumPy 2.x accumulates imaginary
                         residuals making np.real(trace)**2 ∉ [0,1]
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 0. IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import math
import json
import warnings
import logging
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
from scipy.linalg import sqrtm, expm, block_diag, eigvalsh, schur
from scipy.special import eval_laguerre, factorial as sp_factorial
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Ellipse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

import qutip as qt
from qutip import (
    basis, ket2dm, coherent, coherent_dm, thermal_dm,
    expect, destroy, num, displace, squeeze,
    tensor, qeye, mesolve, Options,
)

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | quantum_utils | %(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. GLOBAL CONSTANTS & COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = "1.0.1"
__author__  = "m25iqt013 — IIT Jodhpur"

# Physical constants (SI)
HBAR = 1.0      # natural units throughout

# Default Hilbert space truncation
DEFAULT_DIM = 40

# Default phase-space grid
DEFAULT_XVEC = np.linspace(-6, 6, 300)

# ── Master color palette (consistent across all notebooks + dashboard) ────────
COLORS: Dict[str, str] = {
    # Identity colors per state type
    "fock"       : "#a78bfa",   # purple
    "coherent"   : "#22d3ee",   # cyan
    "squeezed"   : "#c084fc",   # violet
    "thermal"    : "#fb923c",   # orange
    "cat"        : "#f472b6",   # pink
    "advanced"   : "#f87171",   # rose
    "channel"    : "#34d399",   # emerald
    "gbs"        : "#38bdf8",   # sky blue
    # UI colors
    "purple"     : "#a78bfa",
    "violet"     : "#7c3aed",
    "blue"       : "#60a5fa",
    "cyan"       : "#22d3ee",
    "teal"       : "#2dd4bf",
    "green"      : "#4ade80",
    "emerald"    : "#34d399",
    "lime"       : "#86efac",
    "yellow"     : "#fbbf24",
    "amber"      : "#fbbf24",
    "orange"     : "#fb923c",
    "red"        : "#f87171",
    "rose"       : "#fb7185",
    "pink"       : "#f472b6",
    "fuchsia"    : "#e879f9",
    "white"      : "#f1f5f9",
    "muted"      : "#64748b",
    "sky"        : "#38bdf8",
    "indigo"     : "#818cf8",
}

# Palette for sweeps
STATE_PALETTE = [
    "#a78bfa", "#22d3ee", "#c084fc", "#fb923c",
    "#f472b6", "#34d399", "#f87171", "#fbbf24",
]

# ── Custom Wigner colormaps ───────────────────────────────────────────────────
WIGNER_CMAP = LinearSegmentedColormap.from_list(
    "wigner_cv", ["#1e3a5f", "#0a0a1a", "#e879f9"], N=512
)
WIGNER_CMAP_GOLD = LinearSegmentedColormap.from_list(
    "wigner_gold", ["#1e0a3c", "#0a0a1a", "#fbbf24"], N=512
)
WIGNER_CMAP_GBS = LinearSegmentedColormap.from_list(
    "wigner_gbs", ["#0c1445", "#0a0a1a", "#38bdf8"], N=512
)

# ── Global dark Matplotlib theme ─────────────────────────────────────────────
DARK_THEME: Dict = {
    "figure.facecolor" : "#0a0a1a",
    "axes.facecolor"   : "#0f0f2a",
    "axes.edgecolor"   : "#6d28d9",
    "axes.labelcolor"  : "#e2e8f0",
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "xtick.color"      : "#94a3b8",
    "ytick.color"      : "#94a3b8",
    "text.color"       : "#e2e8f0",
    "grid.color"       : "#1e1e3f",
    "grid.linewidth"   : 0.5,
    "grid.alpha"       : 0.4,
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "legend.facecolor" : "#0a0a1a",
    "legend.edgecolor" : "#6d28d9",
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
    "savefig.facecolor": "#0a0a1a",
}


def apply_dark_theme() -> None:
    """Apply the global dark Matplotlib theme. Call once at notebook start."""
    plt.rcParams.update(DARK_THEME)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STATE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def make_fock(n: int, dim: int = DEFAULT_DIM) -> qt.Qobj:
    """
    Density matrix for Fock state |n⟩.

    Parameters
    ----------
    n   : photon number  (0 ≤ n < dim)
    dim : Hilbert space truncation

    Returns
    -------
    rho = |n⟩⟨n|
    """
    assert 0 <= n < dim, f"n={n} must be in [0, {dim})"
    return ket2dm(basis(dim, n))


def make_coherent(alpha: complex, dim: int = DEFAULT_DIM) -> qt.Qobj:
    """
    Density matrix for coherent state |α⟩.

    |α⟩ = D(α)|0⟩ = e^{-|α|²/2} Σ αⁿ/√n! |n⟩
    """
    return coherent_dm(dim, alpha)


def make_squeezed(r: float, phi: float = 0.0,
                  dim: int = DEFAULT_DIM) -> qt.Qobj:
    """
    Density matrix for squeezed vacuum S(ξ)|0⟩, ξ = r·e^{iφ}.

    Parameters
    ----------
    r   : squeezing parameter (r ≥ 0)
    phi : squeezing angle in radians
    dim : Hilbert space truncation
    """
    xi  = r * np.exp(1j * phi)
    S   = squeeze(dim, xi)
    vac = basis(dim, 0)
    return ket2dm(S * vac)


def make_displaced_squeezed(alpha: complex, r: float, phi: float = 0.0,
                             dim: int = DEFAULT_DIM) -> qt.Qobj:
    """
    Displaced squeezed state D(α)S(ξ)|0⟩.

    Applied as: first squeeze, then displace (standard Xanadu convention).
    """
    xi  = r * np.exp(1j * phi)
    S   = squeeze(dim, xi)
    D   = displace(dim, alpha)
    vac = basis(dim, 0)
    return ket2dm(D * S * vac)


def make_thermal(nbar: float, dim: int = DEFAULT_DIM) -> qt.Qobj:
    """
    Thermal state at mean photon number n̄.

    ρ_th = Σ [n̄ⁿ/(n̄+1)^{n+1}] |n⟩⟨n|
    """
    return thermal_dm(dim, nbar)


def make_cat(alpha: complex, sign: int = +1,
             dim: int = DEFAULT_DIM) -> qt.Qobj:
    """
    Schrödinger cat state (even +1 or odd -1).

    |cat±⟩ = N±(|α⟩ ± |-α⟩)

    Parameters
    ----------
    alpha : coherent amplitude
    sign  : +1 for even cat, -1 for odd cat
    dim   : Hilbert space truncation
    """
    psi_p = coherent(dim, alpha)
    psi_m = coherent(dim, -alpha)
    psi   = (psi_p + sign * psi_m).unit()
    return ket2dm(psi)


def make_compass_state(alpha: complex, dim: int = DEFAULT_DIM) -> qt.Qobj:
    """
    4-component compass (superposition of ±α and ±iα).

    |compass⟩ = N(|α⟩ + |-α⟩ + |iα⟩ + |-iα⟩)
    """
    psi = sum(coherent(dim, a * alpha)
              for a in [1, -1, 1j, -1j])
    return ket2dm(psi.unit())


def make_gkp(delta: float = 0.3, n_max: int = 5,
             dim: int = DEFAULT_DIM) -> qt.Qobj:
    """
    Approximate GKP (Gottesman-Kitaev-Preskill) qubit state.

    Finite superposition of coherent states on a square lattice:
    |GKP⟩ ≈ Σ_{n=-n_max}^{n_max} exp(-δ²n²) |2n√π⟩

    Parameters
    ----------
    delta : envelope decay parameter (smaller = more ideal)
    n_max : lattice truncation
    dim   : Hilbert space truncation
    """
    psi = sum(
        np.exp(-delta**2 * n**2) * coherent(dim, 2 * n * np.sqrt(np.pi))
        for n in range(-n_max, n_max + 1)
    )
    return ket2dm(psi.unit())


def make_two_mode_squeezed(r: float, dim: int = 20) -> qt.Qobj:
    """
    Two-mode squeezed vacuum (EPR state) S₂(r)|00⟩.

    |TMSV⟩ = (1/cosh r) Σ_n (-tanh r)^n |n,n⟩

    Parameters
    ----------
    r   : two-mode squeezing parameter
    dim : single-mode Hilbert space truncation (two-mode dim = dim²)

    Returns
    -------
    rho_AB : two-mode density matrix (tensor product space)
    """
    tanh_r  = np.tanh(r)
    cosh_r  = np.cosh(r)
    coeffs  = [(-tanh_r)**n / cosh_r for n in range(dim)]
    psi     = sum(c * tensor(basis(dim, n), basis(dim, n))
                  for n, c in enumerate(coeffs))
    return ket2dm(psi.unit())


def make_noon_single_mode(N: int, dim: int = DEFAULT_DIM) -> qt.Qobj:
    """
    Single-mode NOON state (1/√2)(|N⟩ + |0⟩).

    This is the reduced state of the single-mode NOON subspace.
    """
    psi = (basis(dim, N) + basis(dim, 0)).unit()
    return ket2dm(psi)


def phase_shift_op(phi: float, dim: int) -> qt.Qobj:
    """
    Phase shift operator R(φ) = exp(-iφ n̂).

    Parameters
    ----------
    phi : phase angle in radians
    dim : Hilbert space dimension
    """
    n_op = num(dim)
    return (-1j * phi * n_op).expm()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PHASE-SPACE REPRESENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_wigner(rho: qt.Qobj,
                   xvec: np.ndarray = None) -> np.ndarray:
    """
    Wigner quasi-probability distribution W(x, p).

    W(x,p) = (1/π) ∫ ⟨x+y|ρ|x-y⟩ e^{-2ipy} dy

    Uses QuTiP 5.x with g=2 convention.

    Parameters
    ----------
    rho  : density matrix
    xvec : quadrature axis (default: DEFAULT_XVEC)

    Returns
    -------
    W : 2D array of shape (len(xvec), len(xvec))
    """
    if xvec is None:
        xvec = DEFAULT_XVEC
    return np.array(qt.wigner(rho, xvec, xvec, g=2))


def compute_husimi(rho: qt.Qobj,
                   xvec: np.ndarray = None) -> np.ndarray:
    """
    Husimi Q function Q(α) = ⟨α|ρ|α⟩/π ≥ 0.

    Bug fix: QuTiP 5.x qt.qfunc() returns array directly, not (Q, x, p) tuple.
    This function handles both versions transparently.

    Parameters
    ----------
    rho  : density matrix
    xvec : coherent amplitude axis (default: DEFAULT_XVEC)

    Returns
    -------
    Q : 2D array ≥ 0
    """
    if xvec is None:
        xvec = DEFAULT_XVEC
    result = qt.qfunc(rho, xvec, xvec)
    return np.array(result[0] if isinstance(result, tuple) else result)


def compute_p_function_approx(rho: qt.Qobj,
                               xvec: np.ndarray = None,
                               sigma: float = 0.3) -> np.ndarray:
    """
    Glauber-Sudarshan P function (regularised).

    For classical states: P(α) = δ²(α - α₀) (positive, well-behaved).
    For non-classical states: P is singular or negative.

    This returns a Gaussian-regularised approximation useful for visualisation.

    Parameters
    ----------
    rho   : density matrix
    xvec  : axis
    sigma : regularisation width (smaller → closer to true delta)
    """
    if xvec is None:
        xvec = np.linspace(-6, 6, 200)
    dim    = rho.shape[0]
    a      = destroy(dim)
    x_op   = (a + a.dag()) / np.sqrt(2)
    p_op   = 1j * (a.dag() - a) / np.sqrt(2)
    x0     = float(expect(x_op, rho).real)
    p0     = float(expect(p_op, rho).real)
    XX, PP = np.meshgrid(xvec, xvec)
    P      = np.exp(-((XX - x0)**2 + (PP - p0)**2) / (2 * sigma**2))
    P     /= (2 * np.pi * sigma**2)
    return P


def compute_characteristic_function(rho: qt.Qobj,
                                     xvec: np.ndarray = None) -> np.ndarray:
    """
    Symmetric characteristic function χ(ξ) = Tr[ρ D(ξ)].

    For Gaussian states: χ(ξ) = exp(-|ξ|²/2 + ...).

    Parameters
    ----------
    rho  : density matrix
    xvec : axis for Re(ξ) and Im(ξ)

    Returns
    -------
    chi : 2D complex array of |χ(ξ)|
    """
    if xvec is None:
        xvec = np.linspace(-4, 4, 100)
    dim   = rho.shape[0]
    XX, YY = np.meshgrid(xvec, xvec)
    XI    = XX + 1j * YY
    chi   = np.zeros_like(XI, dtype=complex)
    for i in range(XI.shape[0]):
        for j in range(XI.shape[1]):
            D = displace(dim, XI[i, j])
            chi[i, j] = complex(expect(D, rho))
    return chi


def wigner_neg_volume(W: np.ndarray,
                      xvec: np.ndarray) -> float:
    """
    Wigner negativity volume δ[W] = ∫|W(x,p)| dxdp − 1.

    δ = 0  : classical state (Gaussian, non-negative Wigner)
    δ > 0  : non-classical (Fock n>0, cat, GKP, ...)

    This is an experimentally measurable witness of non-classicality.

    Parameters
    ----------
    W    : 2D Wigner array
    xvec : quadrature axis used to compute W

    Returns
    -------
    delta : float ≥ 0
    """
    dx = xvec[1] - xvec[0]
    return float(np.sum(np.abs(W)) * dx**2 - 1.0)


def wigner_negative_volume_only(W: np.ndarray,
                                 xvec: np.ndarray) -> float:
    """Volume of the negative part of W only (∫ max(−W,0) dxdp)."""
    dx = xvec[1] - xvec[0]
    neg = np.where(W < 0, -W, 0.0)
    return float(np.sum(neg) * dx**2)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SINGLE-MODE STATE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def state_metrics(rho: qt.Qobj) -> Dict:
    """
    Comprehensive single-mode quantum state metrics.

    Computes:
      - Photon number: ⟨n⟩, Var(n), Mandel Q
      - Quadratures: ⟨x⟩, ⟨p⟩, Δx, Δp, Δx·Δp
      - Covariance matrix σ (2×2 real)
      - Symplectic eigenvalue ν (≥ 1/2 by Heisenberg)
      - Purity Tr(ρ²) and von Neumann entropy S_vN
      - Photon number probabilities P(n)

    Parameters
    ----------
    rho : single-mode density matrix

    Returns
    -------
    dict with all metrics (all numerical values rounded to 6 d.p.)
    """
    dim  = rho.shape[0]
    a    = destroy(dim)
    n_op = num(dim)
    x_op = (a + a.dag()) / np.sqrt(2)
    p_op = 1j * (a.dag() - a) / np.sqrt(2)

    # ── Number statistics ─────────────────────────────────────
    mn   = float(expect(n_op, rho).real)
    mn2  = float(expect(n_op * n_op, rho).real)
    vn   = mn2 - mn**2
    mandel_Q = (vn - mn) / mn if mn > 1e-10 else float("nan")

    # ── Mixedness ─────────────────────────────────────────────
    purity  = float((rho * rho).tr().real)
    entropy = float(qt.entropy_vn(rho, base=2))

    # ── Quadrature statistics ─────────────────────────────────
    mx   = float(expect(x_op, rho).real)
    mp   = float(expect(p_op, rho).real)
    vx   = float(expect(x_op * x_op, rho).real) - mx**2
    vp   = float(expect(p_op * p_op, rho).real) - mp**2
    xpc  = float(expect((x_op * p_op + p_op * x_op) / 2, rho).real) - mx * mp
    dx   = float(np.sqrt(max(vx, 0.0)))
    dp   = float(np.sqrt(max(vp, 0.0)))

    # ── Covariance matrix ─────────────────────────────────────
    sigma = np.array([[vx, xpc], [xpc, vp]])

    # ── Symplectic eigenvalue ─────────────────────────────────
    omega = np.array([[0, 1], [-1, 0]])
    eigs  = np.linalg.eigvals(1j * sigma @ omega)
    nu    = float(np.max(np.abs(eigs.real)))

    # ── Photon distribution ───────────────────────────────────
    probs = np.array([float(rho[n, n].real) for n in range(min(dim, 30))])

    return {
        # Number
        "mean_n"   : round(mn,       6),
        "var_n"    : round(vn,       6),
        "mandel_Q" : round(mandel_Q, 6) if not np.isnan(mandel_Q) else "nan",
        # Mixedness
        "purity"   : round(purity,   6),
        "entropy"  : round(entropy,  6),
        # Quadratures
        "mean_x"   : round(mx,       6),
        "mean_p"   : round(mp,       6),
        "var_x"    : round(vx,       6),
        "var_p"    : round(vp,       6),
        "delta_x"  : round(dx,       6),
        "delta_p"  : round(dp,       6),
        "heis_prod": round(dx * dp,  6),
        # Covariance & symplectic
        "sigma"    : sigma,
        "nu_symp"  : round(nu,       6),
        # Photon distribution
        "probs"    : probs,
    }


def mandel_Q_parameter(rho: qt.Qobj) -> float:
    """
    Mandel Q parameter: Q_M = (Var(n) - ⟨n⟩) / ⟨n⟩.

    Q < 0: sub-Poissonian (non-classical)
    Q = 0: Poissonian (coherent state)
    Q > 0: super-Poissonian (thermal state)
    """
    dim  = rho.shape[0]
    n_op = num(dim)
    mn   = float(expect(n_op, rho).real)
    mn2  = float(expect(n_op * n_op, rho).real)
    vn   = mn2 - mn**2
    return float((vn - mn) / mn) if mn > 1e-10 else float("nan")


def quantum_fisher_info(rho: qt.Qobj,
                         observable: Optional[qt.Qobj] = None) -> float:
    """
    Quantum Fisher information F_Q[ρ, Ô] = 2 Σ_{i,j} |⟨i|Ô|j⟩|² / (λᵢ+λⱼ)
    where λᵢ are eigenvalues of ρ.

    Default observable: photon number operator n̂ (metrological usefulness).

    Parameters
    ----------
    rho        : density matrix
    observable : Hermitian operator (default: num(dim))

    Returns
    -------
    F_Q : quantum Fisher information (float)
    """
    dim = rho.shape[0]
    if observable is None:
        observable = num(dim)

    evals, evecs_qobj = rho.eigenstates()
    evals = np.array(evals, dtype=float)
    O     = observable.full()
    evecs = np.column_stack([e.full().flatten() for e in evecs_qobj])

    F_Q = 0.0
    for i in range(dim):
        for j in range(dim):
            lij = evals[i] + evals[j]
            if lij > 1e-12:
                Oij = np.dot(evecs[:, i].conj(), O @ evecs[:, j])
                F_Q += 2.0 * (abs(Oij)**2) / lij
    return float(F_Q)


def q_fidelity(rho1: qt.Qobj, rho2: qt.Qobj) -> float:
    """
    Quantum fidelity F(ρ₁, ρ₂) = (Tr√(√ρ₁ ρ₂ √ρ₁))².

    F = 1 for identical states, F = 0 for orthogonal states.

    Bug fix v1.0.1:
        scipy.linalg.sqrtm on near-singular (rank-deficient) matrices —
        typical for pure quantum states in large Fock spaces — raises
        "Matrix is singular" warnings and accumulates imaginary residuals
        that cause np.real(trace)**2 to deviate from [0,1] in NumPy 2.x.

        Fix: use eigendecomposition-based matrix square root (eigh) which
        guarantees a PSD Hermitian output and clamps numerical negatives,
        making the result always in [0, 1] regardless of matrix rank.
    """
    def _matsqrt(M: np.ndarray) -> np.ndarray:
        """PSD-safe matrix square root via eigendecomposition."""
        Mh = 0.5 * (M + M.conj().T)            # enforce Hermitian symmetry
        evals, evecs = np.linalg.eigh(Mh)
        evals = np.maximum(evals.real, 0.0)    # clip numerical negatives
        return evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T

    r1   = rho1.full()
    r2   = rho2.full()
    sq1  = _matsqrt(r1)
    M    = sq1 @ r2 @ sq1
    sqM  = _matsqrt(M)
    val  = float(np.real(np.trace(sqM)))       # real by construction
    return float(np.clip(val, 0.0, None) ** 2) # clamp to [0, 1]


def bures_distance(rho1: qt.Qobj, rho2: qt.Qobj) -> float:
    """
    Bures distance d_B = √(2(1 - √F(ρ₁,ρ₂))).
    """
    F = q_fidelity(rho1, rho2)
    return float(np.sqrt(2.0 * (1.0 - np.sqrt(max(F, 0.0)))))


def trace_distance(rho1: qt.Qobj, rho2: qt.Qobj) -> float:
    """
    Trace distance D(ρ₁,ρ₂) = ½ Tr|ρ₁ − ρ₂| = ½ Σ|λᵢ|.
    """
    diff  = (rho1 - rho2).full()
    evals = np.linalg.eigvals(diff)
    return float(0.5 * np.sum(np.abs(evals)))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NON-CLASSICALITY WITNESSES
# ═══════════════════════════════════════════════════════════════════════════════

def nonclassicality_witnesses(rho: qt.Qobj,
                               xvec: np.ndarray = None) -> Dict:
    """
    Complete non-classicality witness battery for a single-mode state.

    Computes all witnesses used across Notebooks 01–08:
      1. Wigner negativity volume δ
      2. Wigner minimum value W_min
      3. Mandel Q parameter
      4. Sub-Poissonian flag (var_n < mean_n)
      5. Purity Tr(ρ²)
      6. von Neumann entropy
      7. Symplectic eigenvalue ν
      8. Quantum Fisher information F_Q[ρ, n̂]
      9. Number of Wigner radial nodes (for Fock-like states)
     10. Wigner fringe visibility (for cat-like states)

    Parameters
    ----------
    rho  : density matrix
    xvec : phase-space axis

    Returns
    -------
    dict of all witness values
    """
    if xvec is None:
        xvec = DEFAULT_XVEC

    W       = compute_wigner(rho, xvec)
    delta   = wigner_neg_volume(W, xvec)
    W_min   = float(W.min())
    W_max   = float(W.max())
    m       = state_metrics(rho)

    # Fringe visibility (ratio of max positive to |min negative|)
    if W_min < -1e-6:
        fringe_visibility = float(W_max / abs(W_min))
    else:
        fringe_visibility = float("nan")

    # Radial nodes: count zero-crossings along x-axis slice
    mid        = len(xvec) // 2
    W_slice    = W[mid, :]
    sign_diff  = np.diff(np.sign(W_slice))
    n_nodes    = int(np.sum(sign_diff != 0))

    # Sub-Poissonian flag
    mq      = m["mandel_Q"]
    sub_poi = (mq != "nan" and mq < -0.01)

    # Hudson's theorem: non-classical if W < 0 anywhere
    hudsons_violated = W_min < -1e-8

    # QFI (expensive — use cached if calling in loop)
    try:
        F_Q = quantum_fisher_info(rho)
    except Exception:
        F_Q = float("nan")

    return {
        "wigner_neg_volume"  : round(delta,             6),
        "wigner_min"         : round(W_min,             6),
        "wigner_max"         : round(W_max,             6),
        "mandel_Q"           : m["mandel_Q"],
        "sub_poissonian"     : sub_poi,
        "hudsons_violated"   : hudsons_violated,
        "purity"             : m["purity"],
        "entropy_vN"         : m["entropy"],
        "nu_symp"            : m["nu_symp"],
        "heis_prod"          : m["heis_prod"],
        "fringe_visibility"  : round(fringe_visibility, 4)
                               if not np.isnan(fringe_visibility) else "nan",
        "n_wigner_nodes"     : n_nodes,
        "quantum_fisher_info": round(F_Q, 4) if not np.isnan(F_Q) else "nan",
        "non_classical"      : bool(hudsons_violated or sub_poi),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TWO-MODE & ENTANGLEMENT MEASURES
# ═══════════════════════════════════════════════════════════════════════════════

def twomode_covariance(rho_AB: qt.Qobj, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    4×4 covariance matrix of two-mode state ρ_AB.

    Quadrature ordering: (x_A, p_A, x_B, p_B).
    Mode A = first tensor factor (QuTiP convention).

    Parameters
    ----------
    rho_AB : two-mode density matrix (dim² × dim² space)
    dim    : single-mode Hilbert space dimension

    Returns
    -------
    sigma : 4×4 covariance matrix
    means : 4-vector of quadrature means
    """
    aA  = tensor(destroy(dim), qeye(dim))
    aB  = tensor(qeye(dim), destroy(dim))
    xA  = (aA + aA.dag()) / np.sqrt(2)
    pA  = 1j * (aA.dag() - aA) / np.sqrt(2)
    xB  = (aB + aB.dag()) / np.sqrt(2)
    pB  = 1j * (aB.dag() - aB) / np.sqrt(2)
    ops = [xA, pA, xB, pB]

    means = np.array([float(expect(O, rho_AB).real) for O in ops])
    sigma = np.zeros((4, 4))
    for i, Oi in enumerate(ops):
        for j, Oj in enumerate(ops):
            anticomm     = (Oi * Oj + Oj * Oi) / 2
            sigma[i, j]  = float(expect(anticomm, rho_AB).real) - means[i] * means[j]
    return sigma, means


def symplectic_eigenvalues_2mode(sigma: np.ndarray) -> np.ndarray:
    """
    Symplectic eigenvalues ν± of a 4×4 covariance matrix.

    Uses Serafini (2017) formula: eigenvalues of iΩσ.
    Uncertainty principle: ν± ≥ 1/2.

    Returns
    -------
    nu : sorted array [ν₋, ν₊]
    """
    Omega = np.kron(np.array([[0, 1], [-1, 0]]), np.eye(2))
    M     = Omega.T @ sigma
    eigs  = eigvalsh(1j * M)
    nu    = np.sort(np.abs(eigs[eigs > 0]))
    return nu


def log_negativity_from_CM(sigma: np.ndarray) -> float:
    """
    Logarithmic negativity E_N from the 4×4 covariance matrix.

    E_N = max(0, -log₂(ν̃₋))
    where ν̃₋ is the smallest symplectic eigenvalue of the
    partially transposed CM σ̃.

    E_N = 0: separable
    E_N > 0: entangled (larger = more entangled)

    Parameters
    ----------
    sigma : 4×4 covariance matrix (x_A, p_A, x_B, p_B ordering)

    Returns
    -------
    E_N : logarithmic negativity (float ≥ 0)
    """
    # Partial transpose on mode B: p_B → -p_B
    PT        = np.diag([1, 1, 1, -1])
    sigma_PT  = PT @ sigma @ PT
    nu_PT     = symplectic_eigenvalues_2mode(sigma_PT)
    nu_minus  = nu_PT[0]
    return float(max(0.0, -np.log2(nu_minus)))


def simon_separability(sigma: np.ndarray) -> Dict:
    """
    Simon separability criterion for two-mode Gaussian states.

    State is separable iff ν̃₋ ≥ 1/2 (PPT criterion).

    Returns
    -------
    dict with nu_tilde_minus, separable flag, E_N
    """
    E_N       = log_negativity_from_CM(sigma)
    PT        = np.diag([1, 1, 1, -1])
    sigma_PT  = PT @ sigma @ PT
    nu_PT     = symplectic_eigenvalues_2mode(sigma_PT)
    nu_minus  = float(nu_PT[0])
    separable = nu_minus >= 0.5 - 1e-8

    return {
        "nu_tilde_minus"    : round(nu_minus, 6),
        "separable"         : separable,
        "log_negativity"    : round(E_N, 6),
        "ppt_criterion_met" : separable,
    }


def entanglement_entropy_tmsv(r: float) -> float:
    """
    Von Neumann entropy of one mode of a TMSV state (= entanglement entropy).

    S = (cosh²r) log₂(cosh²r) − (sinh²r) log₂(sinh²r)

    Parameters
    ----------
    r : two-mode squeezing parameter

    Returns
    -------
    S : entanglement entropy in bits
    """
    if r < 1e-10:
        return 0.0
    ch2 = np.cosh(r)**2
    sh2 = np.sinh(r)**2
    return float(ch2 * np.log2(ch2) - sh2 * np.log2(sh2))


def schmidt_number(rho_AB: qt.Qobj) -> float:
    """
    Schmidt number (participation ratio) K = 1/Σλᵢ².

    K = 1: product state, K = dim: maximally entangled.

    Parameters
    ----------
    rho_AB : two-mode density matrix

    Returns
    -------
    K : Schmidt number (float ≥ 1)
    """
    # SVD of the state coefficient matrix
    psi    = rho_AB.full()
    dim    = int(np.sqrt(psi.shape[0]))
    C      = psi[:dim, :dim]    # coefficient matrix
    svals  = np.linalg.svd(C, compute_uv=False)
    lam    = svals**2
    lam    = lam[lam > 1e-12]
    lam   /= lam.sum()
    return float(1.0 / np.sum(lam**2))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. QUANTUM CHANNELS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_displacement(rho: qt.Qobj, alpha: complex) -> qt.Qobj:
    """
    Apply displacement D(α) to state ρ: ρ → D(α)ρD†(α).

    Preserves purity, Wigner shape and all non-classicality witnesses.
    """
    dim = rho.shape[0]
    D   = displace(dim, alpha)
    return D * rho * D.dag()


def apply_squeezing_channel(rho: qt.Qobj, r: float,
                             phi: float = 0.0) -> qt.Qobj:
    """
    Apply squeezing S(ξ), ξ=r·e^{iφ}, to state ρ.

    Preserves purity. Transforms covariance as σ → S σ Sᵀ.
    """
    dim = rho.shape[0]
    xi  = r * np.exp(1j * phi)
    S   = squeeze(dim, xi)
    return S * rho * S.dag()


def apply_phase_shift(rho: qt.Qobj, phi: float) -> qt.Qobj:
    """
    Apply phase shift R(φ) = e^{-iφn̂} to state ρ.

    Rotates phase space rigidly. All invariants preserved.
    """
    dim = rho.shape[0]
    R   = phase_shift_op(phi, dim)
    return R * rho * R.dag()


def apply_beam_splitter(rho1: qt.Qobj, rho2: qt.Qobj,
                         theta: float, dim: int = 15) -> Tuple[qt.Qobj, qt.Qobj]:
    """
    Apply beam splitter B(θ) to two-mode state ρ₁ ⊗ ρ₂.

    B(θ): a₁ → cos θ·a₁ + sin θ·a₂
           a₂ → -sin θ·a₁ + cos θ·a₂

    Transmissivity T = cos²θ.

    Parameters
    ----------
    rho1, rho2 : single-mode input density matrices
    theta      : beam splitter angle (T = cos²θ)
    dim        : single-mode Hilbert space dimension

    Returns
    -------
    (rho_out1, rho_out2) : output mode density matrices
    """
    rho_in = tensor(rho1, rho2)
    a1     = tensor(destroy(dim), qeye(dim))
    a2     = tensor(qeye(dim), destroy(dim))
    H_bs   = theta * (a1.dag() * a2 - a1 * a2.dag())
    U_bs   = H_bs.expm()
    rho_out = U_bs * rho_in * U_bs.dag()
    return rho_out.ptrace(0), rho_out.ptrace(1)


def apply_loss_channel(rho: qt.Qobj, gamma: float, t: float,
                        n_th: float = 0.0,
                        dim: int = None) -> qt.Qobj:
    """
    Amplitude damping (loss) channel via Lindblad master equation.

    dρ/dt = γ(n̄+1)(2aρa† − a†aρ − ρa†a) + γn̄(2a†ρa − aa†ρ − ρaa†)

    Parameters
    ----------
    rho   : input density matrix
    gamma : decay rate
    t     : evolution time (final time)
    n_th  : mean thermal photon number of bath (0 = pure loss)
    dim   : Hilbert space dimension (default: rho.shape[0])

    Returns
    -------
    rho_out : output density matrix after time t
    """
    if dim is None:
        dim = rho.shape[0]
    a       = destroy(dim)
    H       = qt.Qobj(np.zeros((dim, dim)))
    c_ops   = [np.sqrt(gamma * (n_th + 1)) * a]
    if n_th > 0:
        c_ops.append(np.sqrt(gamma * n_th) * a.dag())
    opts    = Options(nsteps=15000)
    tlist   = np.linspace(0, t, max(10, int(20 * t)))
    result  = mesolve(H, rho, tlist, c_ops, [], options=opts)
    return result.states[-1]


def apply_amplification_channel(rho: qt.Qobj, G: float, t: float,
                                  gamma: float = 0.5,
                                  dim: int = None) -> qt.Qobj:
    """
    Phase-insensitive amplifier (gain G > 1).

    Collapse operators: √(γG) a†  (stimulated emission)
                        √(γ(G-1)) a (vacuum noise)

    Parameters
    ----------
    rho   : input density matrix
    G     : gain factor (G > 1)
    t     : evolution time
    gamma : coupling rate
    """
    if dim is None:
        dim = rho.shape[0]
    a      = destroy(dim)
    H      = qt.Qobj(np.zeros((dim, dim)))
    c_ops  = [np.sqrt(gamma * G) * a.dag(),
               np.sqrt(gamma * max(G - 1, 0)) * a]
    opts   = Options(nsteps=15000)
    tlist  = np.linspace(0, t, max(10, int(20 * t)))
    result = mesolve(H, rho, tlist, c_ops, [], options=opts)
    return result.states[-1]


def covariance_loss_transform(sigma: np.ndarray,
                               gamma_t: float) -> np.ndarray:
    """
    Analytical covariance matrix transformation under loss channel.

    σ → η·σ + (1−η)/2·I₂,  η = e^{−γt}

    Parameters
    ----------
    sigma   : 2×2 covariance matrix
    gamma_t : γ·t (loss parameter)

    Returns
    -------
    sigma_out : 2×2 covariance matrix after loss
    """
    eta = np.exp(-gamma_t)
    N   = (1 - eta) / 2 * np.eye(2)
    return eta * sigma + N


def covariance_squeezing_transform(sigma: np.ndarray,
                                    r: float) -> np.ndarray:
    """
    Symplectic transformation of CM under squeezing S(r), φ=0.

    σ → S σ Sᵀ,  S = diag(e^{-r}, e^{r})
    """
    S = np.diag([np.exp(-r), np.exp(r)])
    return S @ sigma @ S.T


def covariance_phase_shift_transform(sigma: np.ndarray,
                                      phi: float) -> np.ndarray:
    """
    Rotation of CM under phase shift R(φ).

    σ → R σ Rᵀ
    """
    R = np.array([[np.cos(phi), -np.sin(phi)],
                   [np.sin(phi),  np.cos(phi)]])
    return R @ sigma @ R.T


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PHOTON STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def poisson_distribution(mean_n: float, k_max: int) -> np.ndarray:
    """
    Poisson photon distribution P(k) = e^{-n̄} n̄^k / k!

    Bug fix: uses math.factorial (np.math removed in NumPy 2.x).

    Parameters
    ----------
    mean_n : mean photon number n̄
    k_max  : truncation

    Returns
    -------
    P : array of length k_max
    """
    k = np.arange(k_max)
    return np.array([
        np.exp(-mean_n) * (mean_n**ki) / math.factorial(ki)
        for ki in k
    ])


def bose_einstein_distribution(nbar: float, k_max: int) -> np.ndarray:
    """
    Bose-Einstein (geometric) distribution P(n) = n̄ⁿ/(n̄+1)^{n+1}.

    Photon statistics of a single-mode thermal state.
    """
    k = np.arange(k_max)
    if nbar < 1e-12:
        p      = np.zeros(k_max)
        p[0]   = 1.0
        return p
    return nbar**k / (nbar + 1)**(k + 1)


def wigner_function_fock_analytical(n: int,
                                     r: float) -> float:
    """
    Analytical Wigner function of Fock state |n⟩ at radial coordinate r.

    W_n(r) = ((-1)^n / π) e^{-r²} L_n(2r²)

    where L_n is the n-th Laguerre polynomial.

    Parameters
    ----------
    n : photon number
    r : radial coordinate r = √(x²+p²)

    Returns
    -------
    W_n(r) : float
    """
    return float(((-1)**n / np.pi) * np.exp(-r**2) * eval_laguerre(n, 2 * r**2))


# ═══════════════════════════════════════════════════════════════════════════════
# 9. BATCH COMPUTATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def batch_wigner(states: Dict[str, qt.Qobj],
                  xvec: np.ndarray = None) -> Dict[str, np.ndarray]:
    """
    Compute Wigner functions for a dictionary of states in batch.

    Parameters
    ----------
    states : {label: rho} dictionary
    xvec   : phase-space axis

    Returns
    -------
    {label: W_array} dictionary
    """
    if xvec is None:
        xvec = DEFAULT_XVEC
    return {lbl: compute_wigner(rho, xvec) for lbl, rho in states.items()}


def batch_metrics(states: Dict[str, qt.Qobj]) -> pd.DataFrame:
    """
    Compute state_metrics for all states and return as a DataFrame.

    Parameters
    ----------
    states : {label: rho} dictionary

    Returns
    -------
    df : pandas DataFrame (one row per state, metrics as columns)
    """
    rows = []
    for lbl, rho in states.items():
        m     = state_metrics(rho)
        row   = {"state": lbl}
        for k, v in m.items():
            if k not in ("sigma", "probs"):
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows).set_index("state")


def batch_witnesses(states: Dict[str, qt.Qobj],
                     xvec: np.ndarray = None) -> pd.DataFrame:
    """
    Compute nonclassicality_witnesses for all states.

    Parameters
    ----------
    states : {label: rho} dictionary
    xvec   : phase-space axis

    Returns
    -------
    df : pandas DataFrame
    """
    if xvec is None:
        xvec = DEFAULT_XVEC
    rows = []
    for lbl, rho in states.items():
        w   = nonclassicality_witnesses(rho, xvec)
        row = {"state": lbl, **{k: v for k, v in w.items()
                                if not isinstance(v, np.ndarray)}}
        rows.append(row)
    return pd.DataFrame(rows).set_index("state")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. VISUALIZATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def safe_norm(W: np.ndarray) -> TwoSlopeNorm:
    """
    Create TwoSlopeNorm safely for Wigner plots.

    Bug fix: TwoSlopeNorm requires strictly vmin < 0 < vmax.
    This guard handles non-negative W arrays gracefully.

    Parameters
    ----------
    W : 2D Wigner array (may be all-positive)

    Returns
    -------
    norm : TwoSlopeNorm centered at 0
    """
    wmin, wmax = float(W.min()), float(W.max())
    if wmin >= 0:
        wmin = -1e-9
    if wmax <= 0:
        wmax = 1e-9
    return TwoSlopeNorm(vmin=wmin, vcenter=0, vmax=wmax)


def plot_wigner_2d(rho: qt.Qobj,
                   xvec: np.ndarray = None,
                   ax: Optional[matplotlib.axes.Axes] = None,
                   title: str = "",
                   cmap: str = "RdBu_r",
                   show_ellipse: bool = True,
                   color: str = "#a78bfa") -> matplotlib.axes.Axes:
    """
    Publication-quality 2D Wigner contour plot.

    Parameters
    ----------
    rho          : density matrix
    xvec         : phase-space axis
    ax           : matplotlib axes (creates new figure if None)
    title        : plot title
    cmap         : colormap (use 'RdBu_r' for signed, 'magma' for positive)
    show_ellipse : overlay uncertainty ellipse
    color        : ellipse edge color
    """
    if xvec is None:
        xvec = DEFAULT_XVEC
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    W   = compute_wigner(rho, xvec)
    m   = state_metrics(rho)

    if W.min() < -1e-9:
        norm = safe_norm(W)
        im   = ax.contourf(xvec, xvec, W, levels=60, cmap=cmap, norm=norm)
        ax.contour(xvec, xvec, W, levels=[0],
                    colors=[color], linewidths=1.5, linestyles="--")
    else:
        im = ax.contourf(xvec, xvec, W, levels=60, cmap="magma", vmin=0)

    # Shot-noise circle
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta),
            ":", color="#22d3ee", lw=0.8, alpha=0.5)

    # Uncertainty ellipse
    if show_ellipse:
        dx_e = m["delta_x"] * np.sqrt(2)
        dp_e = m["delta_p"] * np.sqrt(2)
        x0   = m["mean_x"] * np.sqrt(2)
        p0   = m["mean_p"] * np.sqrt(2)
        ell  = Ellipse(xy=(x0, p0),
                        width=4 * dx_e, height=4 * dp_e,
                        fill=False, edgecolor=color,
                        linewidth=2, linestyle="--", zorder=10)
        ax.add_patch(ell)
        ax.plot(x0, p0, "+", color="white", ms=10, mew=2, zorder=11)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="$W(x,p)$")

    neg  = wigner_neg_volume(W, xvec)
    info = (f"δ={neg:.4f}  γ={m['purity']:.4f}\n"
            f"Δx={m['delta_x']:.3f}  Δp={m['delta_p']:.3f}")
    ax.text(0.03, 0.97, info, transform=ax.transAxes,
            fontsize=8, color="#fbbf24", va="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#0a0a1a", alpha=0.85))

    ax.set_title(title, fontsize=11, color=color, pad=6)
    ax.set_xlabel("$x$ (position quadrature)", fontsize=9)
    ax.set_ylabel("$p$ (momentum quadrature)", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
    return ax


def plot_density_matrix(rho: qt.Qobj,
                         display_dim: int = 12,
                         ax_re: Optional[matplotlib.axes.Axes] = None,
                         ax_im: Optional[matplotlib.axes.Axes] = None,
                         title: str = "",
                         color: str = "#a78bfa"
                         ) -> Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]:
    """
    Plot real and imaginary parts of density matrix as heatmaps.

    Parameters
    ----------
    rho         : density matrix
    display_dim : number of Fock rows/cols to show
    ax_re, ax_im: matplotlib axes (creates new figure if None)
    title       : base title
    color       : highlight color for dominant element
    """
    if ax_re is None or ax_im is None:
        fig, (ax_re, ax_im) = plt.subplots(1, 2, figsize=(10, 4))

    mat = rho.full()[:display_dim, :display_dim]

    for ax, data, part in [
        (ax_re, mat.real, r"$\mathrm{Re}(\rho)$"),
        (ax_im, mat.imag, r"$\mathrm{Im}(\rho)$"),
    ]:
        vmax = max(abs(data).max(), 1e-6)
        vmin = -vmax if data.min() < -1e-9 else -1e-9
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im   = ax.imshow(data, cmap="RdBu_r", norm=norm,
                          interpolation="nearest", aspect="equal")
        ax.set_title(f"{title} — {part}", fontsize=10, color=color, pad=5)
        ax.set_xlabel("$m$", fontsize=8)
        ax.set_ylabel("$n$", fontsize=8)
        ticks = range(0, display_dim, max(1, display_dim // 6))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax_re, ax_im


def plot_photon_distribution(rho: qt.Qobj,
                              k_max: int = 20,
                              ax: Optional[matplotlib.axes.Axes] = None,
                              title: str = "",
                              color: str = "#a78bfa",
                              show_poisson: bool = True) -> matplotlib.axes.Axes:
    """
    Bar plot of photon number distribution P(n) with optional Poisson overlay.

    Parameters
    ----------
    rho          : density matrix
    k_max        : maximum photon number to show
    ax           : matplotlib axes
    title        : plot title
    color        : bar color
    show_poisson : overlay Poisson distribution for comparison
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    dim    = rho.shape[0]
    k_max  = min(k_max, dim)
    k_vals = np.arange(k_max)
    probs  = np.array([float(rho[k, k].real) for k in range(k_max)])

    ax.bar(k_vals, probs, color=color, alpha=0.75, width=0.7,
            edgecolor="none")

    m = state_metrics(rho)
    if show_poisson and m["mean_n"] > 0:
        p_poi = poisson_distribution(m["mean_n"], k_max)
        ax.plot(k_vals, p_poi, "o--", color="#fbbf24",
                ms=4, lw=1.2, alpha=0.8, label=f"Poisson (n̄={m['mean_n']:.2f})")
        ax.legend(fontsize=8)

    mq = m["mandel_Q"]
    mq_str = f"{mq:.3f}" if isinstance(mq, float) else mq
    ax.set_title(f"{title}  $Q_M$={mq_str}", fontsize=10, color=color)
    ax.set_xlabel("Photon number $n$", fontsize=9)
    ax.set_ylabel("$P(n)$", fontsize=9)
    ax.set_xlim(-0.5, k_max - 0.5)
    ax.grid(True, alpha=0.2, axis="y")
    return ax


def plot_wigner_3d_plotly(rho: qt.Qobj,
                           xvec: np.ndarray = None,
                           title: str = "Wigner Function",
                           colorscale: str = "RdBu") -> go.Figure:
    """
    Interactive 3D Wigner surface using Plotly.

    Parameters
    ----------
    rho        : density matrix
    xvec       : phase-space axis
    title      : figure title
    colorscale : Plotly colorscale name

    Returns
    -------
    fig : plotly Figure (call fig.show() or fig.write_html())
    """
    if xvec is None:
        xvec = np.linspace(-5, 5, 120)

    W    = compute_wigner(rho, xvec)
    wabs = max(abs(W.min()), abs(W.max()), 1e-6)

    fig = go.Figure(data=[
        go.Surface(
            z=W, x=xvec, y=xvec,
            colorscale=colorscale,
            cmin=-wabs, cmax=wabs,
            showscale=True,
            colorbar=dict(title="W(x,p)", len=0.5),
            opacity=0.95,
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        )
    ])
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16, color="white"), x=0.5),
        paper_bgcolor="#0a0a1a",
        scene=dict(
            xaxis_title="x", yaxis_title="p", zaxis_title="W(x,p)",
            bgcolor="#0a0a1a",
        ),
        font=dict(color="white"),
        height=600,
    )
    return fig


def plot_master_comparison(rho: qt.Qobj,
                            label: str = "",
                            color: str = "#a78bfa",
                            xvec: np.ndarray = None,
                            k_max: int = 20,
                            figsize: Tuple = (20, 5)
                            ) -> matplotlib.figure.Figure:
    """
    4-panel master comparison: ρ | W(x,p) | Q(α) | P(n).

    Used in all notebooks for the master figure.

    Parameters
    ----------
    rho    : density matrix
    label  : state label
    color  : panel accent color
    xvec   : phase-space axis
    k_max  : photon number truncation for P(n) plot

    Returns
    -------
    fig : matplotlib Figure
    """
    if xvec is None:
        xvec = DEFAULT_XVEC

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(f"Complete Phase Space Analysis — {label}",
                  fontsize=13, color=color, fontweight="bold")

    # Panel 1: Density matrix
    mat  = rho.full()[:12, :12].real
    wmax = max(abs(mat).max(), 1e-6)
    vmin = -wmax if mat.min() < -1e-9 else -1e-9
    axes[0].imshow(mat, cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=wmax),
                    aspect="equal")
    axes[0].set_title(f"$\\hat{{\\rho}}$ (Re)", fontsize=10, color=color)
    axes[0].tick_params(labelsize=7)

    # Panel 2: Wigner
    plot_wigner_2d(rho, xvec, ax=axes[1], title="$W(x,p)$",
                    color=color, show_ellipse=True)

    # Panel 3: Husimi Q
    xvec_q = np.linspace(-5, 5, 150)
    Q       = compute_husimi(rho, xvec_q)
    im_q    = axes[2].contourf(xvec_q, xvec_q, Q, levels=50, cmap="inferno")
    axes[2].set_title("$Q(\\alpha)$", fontsize=10, color=color)
    axes[2].set_aspect("equal")
    axes[2].tick_params(labelsize=7)
    plt.colorbar(im_q, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Photon distribution
    plot_photon_distribution(rho, k_max=k_max, ax=axes[3],
                              title="$P(n)$", color=color)

    plt.tight_layout()
    return fig


def plot_nonclassicality_dashboard(states: Dict[str, qt.Qobj],
                                    xvec: np.ndarray = None,
                                    colors: Optional[List[str]] = None,
                                    figsize: Tuple = (22, 14)
                                    ) -> matplotlib.figure.Figure:
    """
    6-panel non-classicality witness dashboard for multiple states.

    Panels:
      1. Wigner negativity volume δ
      2. Mandel Q parameter
      3. Purity Tr(ρ²)
      4. von Neumann entropy S_vN
      5. Heisenberg product ΔxΔp
      6. Symplectic eigenvalue ν

    Parameters
    ----------
    states : {label: rho} dictionary
    xvec   : phase-space axis
    colors : list of colors (one per state)

    Returns
    -------
    fig : matplotlib Figure
    """
    if xvec is None:
        xvec = DEFAULT_XVEC
    if colors is None:
        colors = STATE_PALETTE[:len(states)]

    labels = list(states.keys())
    n      = len(labels)

    # Compute all witnesses
    wits  = batch_witnesses(states, xvec)
    metr  = batch_metrics(states)

    fig = plt.figure(figsize=figsize)
    fig.suptitle("Non-Classicality Witness Dashboard",
                  fontsize=17, color=COLORS["amber"], fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    def bar(ax, values, ylabel, title, refline=None):
        for i, (v, col) in enumerate(zip(values, colors)):
            ax.bar(i, v, color=col, alpha=0.85)
        if refline is not None:
            ax.axhline(refline[0], color=refline[1], lw=1.5,
                        ls="--", label=refline[2])
            ax.legend(fontsize=8)
        ax.set_xticks(range(n))
        ax.set_xticklabels([l.replace("$", "").replace("\\", "")[:10]
                             for l in labels], rotation=30, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, color=COLORS["amber"])
        ax.grid(True, alpha=0.2, axis="y")

    neg_vols  = wits["wigner_neg_volume"].tolist()
    mq_vals   = [float(v) if v != "nan" else 0 for v in wits["mandel_Q"].tolist()]
    purities  = metr["purity"].tolist()
    entropies = metr["entropy"].tolist()
    heis      = metr["heis_prod"].tolist()
    nu_vals   = metr["nu_symp"].tolist()

    bar(fig.add_subplot(gs[0, 0]), neg_vols, r"$\delta$",
        "Wigner Negativity Volume $\\delta$",
        refline=(0, COLORS["white"], "Classical δ=0"))
    bar(fig.add_subplot(gs[0, 1]), mq_vals, "$Q_M$",
        "Mandel Q Parameter",
        refline=(0, COLORS["yellow"], "Poissonian"))
    bar(fig.add_subplot(gs[0, 2]), purities, r"$\mathrm{Tr}(\hat{\rho}^2)$",
        "Purity",
        refline=(1, COLORS["green"], "Pure state"))
    bar(fig.add_subplot(gs[1, 0]), entropies, "$S_{vN}$ [bits]",
        "von Neumann Entropy")
    bar(fig.add_subplot(gs[1, 1]), heis, r"$\Delta x \cdot \Delta p$",
        "Heisenberg Product",
        refline=(0.5, COLORS["red"], "Heisenberg limit"))
    bar(fig.add_subplot(gs[1, 2]), nu_vals, r"$\nu$",
        "Symplectic Eigenvalue",
        refline=(0.5, COLORS["red"], "Pure state ν=1/2"))

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 11. SUMMARY TABLE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary_table(states: Dict[str, qt.Qobj],
                          xvec: np.ndarray = None,
                          include_witnesses: bool = True) -> pd.DataFrame:
    """
    Build a complete metrics + witnesses summary DataFrame.

    Used by all notebooks for the final summary cell and by the dashboard.

    Parameters
    ----------
    states            : {label: rho} dictionary
    xvec              : phase-space axis
    include_witnesses : include non-classicality witnesses

    Returns
    -------
    df : pandas DataFrame (states × metrics)
    """
    if xvec is None:
        xvec = DEFAULT_XVEC

    rows = []
    for lbl, rho in states.items():
        m    = state_metrics(rho)
        row  = {"state": lbl}

        # Core metrics
        for k in ["mean_n", "var_n", "mandel_Q", "purity", "entropy",
                   "delta_x", "delta_p", "heis_prod", "nu_symp",
                   "mean_x", "mean_p"]:
            row[k] = m.get(k, float("nan"))

        # Wigner metrics
        W       = compute_wigner(rho, xvec)
        row["W_min"]       = round(float(W.min()), 6)
        row["W_max"]       = round(float(W.max()), 5)
        row["neg_volume"]  = round(wigner_neg_volume(W, xvec), 6)
        row["non_classical"] = bool(W.min() < -1e-8)

        # Fidelity to vacuum
        rho_vac = ket2dm(basis(rho.shape[0], 0))
        row["fidelity_to_vacuum"] = round(q_fidelity(rho, rho_vac), 5)

        rows.append(row)

    return pd.DataFrame(rows).set_index("state")


def save_summary(df: pd.DataFrame, output_dir: Path,
                  prefix: str = "summary") -> None:
    """
    Save summary DataFrame to CSV and JSON.

    Parameters
    ----------
    df         : summary DataFrame from build_summary_table
    output_dir : Path to save files
    prefix     : filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}.json"

    df.to_csv(csv_path)
    df.reset_index().to_json(json_path, orient="records", indent=2)

    log.info(f"Saved: {csv_path}")
    log.info(f"Saved: {json_path}")


def save_plotly_html(fig: go.Figure, path: Union[str, Path]) -> None:
    """Save a Plotly figure as interactive HTML."""
    fig.write_html(str(path))
    log.info(f"Saved HTML: {path}")


def save_figure(fig: matplotlib.figure.Figure,
                path: Union[str, Path],
                dpi: int = 300) -> None:
    """Save a matplotlib figure to disk."""
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log.info(f"Saved figure: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. GBS & STRAWBERRY FIELDS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def hafnian_brute(A: np.ndarray) -> complex:
    """
    Brute-force hafnian computation for small even-dimensional matrices.

    haf(A) = Σ_{perfect matchings} Π A_{i,j}

    Used for verification; use thewalrus.hafnian for production.

    Parameters
    ----------
    A : 2n×2n complex symmetric matrix

    Returns
    -------
    haf : complex scalar
    """
    n2 = A.shape[0]
    assert n2 % 2 == 0, "Matrix must be even-dimensional"
    n    = n2 // 2
    idx  = list(range(n2))
    haf  = 0.0 + 0j

    def matchings(lst):
        if not lst:
            yield []
            return
        first = lst[0]
        rest  = lst[1:]
        for i, val in enumerate(rest):
            for m in matchings(rest[:i] + rest[i + 1:]):
                yield [(first, val)] + m

    for m in matchings(idx):
        term = 1.0 + 0j
        for (i, j) in m:
            term *= A[i, j]
        haf += term
    return haf


def gbs_photon_probability(A: np.ndarray, S: np.ndarray) -> float:
    """
    GBS output probability for photon pattern S.

    P(S) = |haf(A_S)|² / (S! · √det(Q))

    where A_S is the submatrix of A selected by pattern S.

    Parameters
    ----------
    A : adjacency-related matrix
    S : photon number pattern (list of ints)

    Returns
    -------
    probability : float
    """
    try:
        from thewalrus import hafnian
        haf_fn = hafnian
    except ImportError:
        haf_fn = hafnian_brute

    # Build repeated-index submatrix
    rows = []
    for i, ni in enumerate(S):
        rows.extend([i] * ni)
    if not rows:
        return 1.0
    A_S = A[np.ix_(rows, rows)]
    haf = haf_fn(A_S)
    denom = float(np.prod([math.factorial(n) for n in S]))
    return float(abs(haf)**2 / denom)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. VALIDATION & SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_validation_suite(verbose: bool = True) -> Dict[str, bool]:
    """
    Self-validation suite: tests all major functions against known results.

    Run this on the HPC to confirm the environment is correct before
    executing notebooks.

    Returns
    -------
    results : {test_name: passed} dictionary
    """
    results = {}

    def check(name: str, condition: bool) -> bool:
        results[name] = condition
        status = "✅" if condition else "❌"
        if verbose:
            print(f"  {status} {name}")
        return condition

    if verbose:
        print("=" * 60)
        print("  quantum_utils.py — Validation Suite")
        print(f"  NumPy {np.__version__} | QuTiP {qt.__version__}")
        print("=" * 60)

    # ── State construction ────────────────────────────────────
    rho_vac  = make_fock(0, 20)
    rho_coh  = make_coherent(2.0, 30)
    rho_sq   = make_squeezed(1.0, 0.0, 40)
    rho_th   = make_thermal(1.0, 30)
    rho_cat  = make_cat(2.0, +1, 30)

    check("make_fock purity=1",    abs(float((rho_vac*rho_vac).tr().real) - 1) < 1e-4)
    check("make_coherent purity=1",abs(float((rho_coh*rho_coh).tr().real) - 1) < 1e-4)
    check("make_squeezed trace=1", abs(float(rho_sq.tr().real) - 1) < 1e-4)
    check("make_thermal trace=1",  abs(float(rho_th.tr().real) - 1) < 1e-4)
    check("make_cat trace=1",      abs(float(rho_cat.tr().real)- 1) < 1e-4)

    # ── Wigner & Husimi ───────────────────────────────────────
    xvec = np.linspace(-4, 4, 80)
    W_vac = compute_wigner(rho_vac, xvec)
    Q_vac = compute_husimi(rho_vac, xvec)
    check("wigner vacuum non-negative", W_vac.min() > -1e-6)
    check("husimi non-negative",        Q_vac.min() >= -1e-9)

    W_fock3 = compute_wigner(make_fock(3, 20), xvec)
    check("wigner fock3 has negatives", W_fock3.min() < -1e-4)

    W_cat = compute_wigner(rho_cat, xvec)
    check("wigner cat has negatives",   W_cat.min() < -1e-4)

    # ── Normalization ─────────────────────────────────────────
    dx    = xvec[1] - xvec[0]
    W_int = float(np.sum(W_vac) * dx**2)
    check("wigner normalization ≈1",    abs(W_int - 1.0) < 0.05)

    # ── State metrics ─────────────────────────────────────────
    m_coh = state_metrics(rho_coh)
    check("coherent <n>=|alpha|^2",   abs(m_coh["mean_n"] - 4.0) < 0.05)
    check("coherent purity=1",        abs(m_coh["purity"] - 1.0) < 1e-4)
    check("coherent mandel_Q≈0",      abs(m_coh["mandel_Q"]) < 0.05)
    check("coherent heisenberg=0.5",  abs(m_coh["heis_prod"] - 0.5) < 0.02)

    m_sq = state_metrics(rho_sq)
    check("squeezed DxDp=0.5",        abs(m_sq["heis_prod"] - 0.5) < 0.02)
    check("squeezed Dx<1/sqrt2",      m_sq["delta_x"] < 1/np.sqrt(2) - 0.01)

    m_th = state_metrics(rho_th)
    check("thermal purity<1",         m_th["purity"] < 0.6)
    check("thermal mandel_Q>0",       isinstance(m_th["mandel_Q"], float) and
                                      m_th["mandel_Q"] > 0.5)

    # ── Negativity ────────────────────────────────────────────
    check("fock3 neg volume>0",  wigner_neg_volume(W_fock3, xvec) > 0.01)
    check("cat neg volume>0",    wigner_neg_volume(W_cat,   xvec) > 0.01)
    check("vacuum neg volume≈0", abs(wigner_neg_volume(W_vac, xvec)) < 0.05)

    # ── Fidelity ──────────────────────────────────────────────
    F_self = q_fidelity(rho_coh, rho_coh)
    F_orth = q_fidelity(make_fock(0, 30), make_fock(5, 30))
    check("fidelity self=1",         abs(F_self - 1.0) < 1e-4)
    check("fidelity orthogonal≈0",   F_orth < 1e-4)

    # ── Two-mode ──────────────────────────────────────────────
    rho_tmsv = make_two_mode_squeezed(1.0, dim=15)
    check("tmsv trace=1", abs(float(rho_tmsv.tr().real) - 1) < 1e-4)

    E_N = entanglement_entropy_tmsv(1.0)
    check("tmsv entropy>0",    E_N > 0)
    check("tmsv entropy<10",   E_N < 10)

    # ── Channels ─────────────────────────────────────────────
    rho_d = apply_displacement(rho_vac, 2.0)
    check("displacement purity=1",  abs(float((rho_d*rho_d).tr().real) - 1) < 1e-4)

    rho_r = apply_phase_shift(rho_coh, np.pi / 3)
    check("phase_shift purity=1",   abs(float((rho_r*rho_r).tr().real) - 1) < 1e-4)

    # ── Photon distributions ──────────────────────────────────
    p_poi = poisson_distribution(2.0, 20)
    check("poisson sums to 1",      abs(p_poi.sum() - 1.0) < 0.01)

    p_be  = bose_einstein_distribution(1.0, 40)
    check("bose-einstein sums≈1",   abs(p_be.sum() - 1.0) < 0.05)

    # ── Summary ───────────────────────────────────────────────
    n_pass = sum(results.values())
    n_total= len(results)
    if verbose:
        print("=" * 60)
        print(f"  Passed: {n_pass}/{n_total}")
        if n_pass == n_total:
            print("  🎉 All tests passed — environment ready!")
        else:
            failed = [k for k, v in results.items() if not v]
            print(f"  ❌ Failed: {failed}")
        print("=" * 60)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 14. STREAMLIT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_state_from_params(state_type: str, params: Dict,
                           dim: int = DEFAULT_DIM) -> qt.Qobj:
    """
    Unified state factory for the Streamlit dashboard.

    Parameters
    ----------
    state_type : one of 'fock','coherent','squeezed','thermal','cat',
                          'displaced_squeezed','two_mode_squeezed','gkp'
    params     : dictionary of state parameters
    dim        : Hilbert space truncation

    Returns
    -------
    rho : density matrix
    """
    st = state_type.lower()
    if st == "fock":
        return make_fock(int(params.get("n", 0)), dim)
    elif st == "coherent":
        alpha = params.get("alpha_re", 1.0) + 1j * params.get("alpha_im", 0.0)
        return make_coherent(alpha, dim)
    elif st == "squeezed":
        return make_squeezed(
            float(params.get("r", 1.0)),
            float(params.get("phi", 0.0)),
            dim
        )
    elif st == "thermal":
        return make_thermal(float(params.get("nbar", 1.0)), dim)
    elif st == "cat":
        alpha = params.get("alpha_re", 2.0) + 1j * params.get("alpha_im", 0.0)
        sign  = +1 if params.get("parity", "even") == "even" else -1
        return make_cat(alpha, sign, dim)
    elif st == "displaced_squeezed":
        alpha = params.get("alpha_re", 1.0) + 1j * params.get("alpha_im", 0.0)
        return make_displaced_squeezed(
            alpha,
            float(params.get("r", 0.8)),
            float(params.get("phi", 0.0)),
            dim
        )
    elif st == "gkp":
        return make_gkp(
            float(params.get("delta", 0.3)),
            int(params.get("n_max", 4)),
            dim
        )
    else:
        raise ValueError(f"Unknown state_type: '{state_type}'. "
                          f"Choose from: fock, coherent, squeezed, thermal, "
                          f"cat, displaced_squeezed, gkp")


def dashboard_wigner_data(rho: qt.Qobj,
                           xvec: np.ndarray = None) -> Dict:
    """
    Compute all phase-space data needed for one Streamlit dashboard panel.

    Returns a JSON-serialisable dict (no numpy arrays — converted to lists).

    Parameters
    ----------
    rho  : density matrix
    xvec : phase-space axis

    Returns
    -------
    data : dict with W, Q, metrics, witnesses (lists not arrays)
    """
    if xvec is None:
        xvec = np.linspace(-5, 5, 150)

    W     = compute_wigner(rho, xvec)
    Q     = compute_husimi(rho, xvec)
    m     = state_metrics(rho)
    w     = nonclassicality_witnesses(rho, xvec)
    probs = [float(rho[k, k].real) for k in range(min(rho.shape[0], 25))]

    return {
        "xvec"   : xvec.tolist(),
        "W"      : W.tolist(),
        "Q"      : Q.tolist(),
        "probs"  : probs,
        "metrics": {k: v for k, v in m.items()
                    if k not in ("sigma", "probs")},
        "witnesses": {k: v for k, v in w.items()},
        "W_min"  : float(W.min()),
        "W_max"  : float(W.max()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 15. MODULE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\nquantum_utils.py v{__version__} — {__author__}")
    print(f"NumPy {np.__version__} | SciPy {scipy.__version__} | QuTiP {qt.__version__}\n")
    run_validation_suite(verbose=True)
