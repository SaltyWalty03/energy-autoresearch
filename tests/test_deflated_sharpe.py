"""
Unit tests for the corrected PSR / Deflated Sharpe implementation.

Reference: Bailey & López de Prado (2014) "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
Journal of Portfolio Management, 40(5).

Key identity being tested
--------------------------
For Gaussian returns (skew=0, excess_kurtosis=0) and SR0=0:

    adj = 1 + (0 + 2)/4 * sr^2 = 1 + 0.5 * sr^2

    z   = (sr - 0) * sqrt(T-1) / sqrt(adj)
        ≈ sr * sqrt(T-1)          for small sr

    PSR = Phi(z)

Special case: sr = 1/sqrt(T-1) → z = 1 → PSR = Phi(1) ≈ 0.84135
"""
import math
import sys
import os

import pytest
from scipy.stats import norm as _norm

_PHI1 = float(_norm.cdf(1.0))   # 0.84134474606...

# ── path so tests can import phase1_eval from the project root ──────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from phase1_eval import _psr


# ---------------------------------------------------------------------------
# Case 1: trivial — SR=0 → PSR=0.5 for any distribution with SR0=0
# ---------------------------------------------------------------------------
def test_psr_zero_sharpe_is_half():
    """PSR(sr0=0) should be 0.5 when the observed SR is exactly 0."""
    result = _psr(sr_daily=0.0, T=1000, skew=0.0, excess_kurtosis=0.0, sr0=0.0)
    assert abs(result - 0.5) < 1e-6, f"Expected 0.5, got {result}"


# ---------------------------------------------------------------------------
# Case 2: Gaussian returns, SR=1/sqrt(T-1), SR0=0 → z=1 → PSR≈0.84135
#
# With sr = 1/sqrt(T-1), skew=0, excess_kurtosis=0:
#   adj = 1 + 0.5 * (1/(T-1))  — very close to 1 for large T
#   z   = (1/sqrt(T-1)) * sqrt(T-1) / sqrt(adj) = 1 / sqrt(adj) ≈ 1.0
#
# We test with T=100001 so 0.5/(T-1) < 5e-6 and adj ≈ 1 to high precision.
# ---------------------------------------------------------------------------
def test_psr_gaussian_sr1_sigma():
    T  = 100_001
    sr = 1.0 / math.sqrt(T - 1)
    result = _psr(sr_daily=sr, T=T, skew=0.0, excess_kurtosis=0.0, sr0=0.0)
    phi1   = _PHI1  # Phi(1.0) — standard normal CDF at z=1
    assert abs(result - phi1) < 1e-5, f"Expected ~{phi1:.6f}, got {result:.6f}"


# ---------------------------------------------------------------------------
# Case 3: excess_kurtosis=-2 makes the raw kurtosis = 1.  Then
#   (excess_kurtosis + 2)/4 = 0 → adj = 1 exactly.
#   With sr = 1/sqrt(T-1): z = 1.0 exactly → PSR = Phi(1) exactly.
# ---------------------------------------------------------------------------
def test_psr_exact_with_kurtosis_correction():
    """When (excess_kurtosis+2)/4=0, adj=1 and z is exact."""
    T  = 1_000
    sr = 1.0 / math.sqrt(T - 1)
    result = _psr(sr_daily=sr, T=T, skew=0.0, excess_kurtosis=-2.0, sr0=0.0)
    # adj = 1 exactly, so z = sr * sqrt(T-1) = 1.0 exactly → PSR = Phi(1)
    assert abs(result - _PHI1) < 1e-6, f"Expected {_PHI1:.8f}, got {result:.8f}"


# ---------------------------------------------------------------------------
# Case 4: monotonicity — higher SR → higher PSR (all else equal)
# ---------------------------------------------------------------------------
def test_psr_monotone_in_sr():
    kwargs = dict(T=500, skew=0.0, excess_kurtosis=0.0, sr0=0.0)
    vals = [_psr(sr_daily=sr, **kwargs) for sr in [0.0, 0.01, 0.05, 0.10]]
    assert vals == sorted(vals), f"PSR not monotone: {vals}"


# ---------------------------------------------------------------------------
# Case 5: higher sr0 → lower PSR (harder benchmark)
# ---------------------------------------------------------------------------
def test_psr_decreases_with_higher_sr0():
    kwargs = dict(sr_daily=0.05, T=500, skew=0.0, excess_kurtosis=0.0)
    p_easy = _psr(**kwargs, sr0=0.00)
    p_hard = _psr(**kwargs, sr0=0.04)
    assert p_easy > p_hard, "PSR should fall as sr0 rises"


# ---------------------------------------------------------------------------
# Case 6: fat tails reduce PSR (for positive SR)
#   More fat tails → larger adj → smaller z → lower PSR.
# ---------------------------------------------------------------------------
def test_psr_fat_tails_reduce_psr():
    base = dict(sr_daily=0.05, T=500, skew=0.0, sr0=0.0)
    p_normal = _psr(**base, excess_kurtosis=0.0)
    p_fat    = _psr(**base, excess_kurtosis=3.0)
    assert p_normal > p_fat, (
        f"Fat tails should reduce PSR for positive SR: {p_normal:.4f} vs {p_fat:.4f}"
    )
