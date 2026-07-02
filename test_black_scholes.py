"""Tests: no-arbitrage identities, limiting behavior, Greeks vs finite
differences, and implied-vol round-trips."""

import numpy as np
import pytest

from black_scholes import greeks, implied_vol, intrinsic, price

RNG = np.random.default_rng(7)

# 200 random-but-sane parameter sets: S, K, T, sigma, r, q
PARAMS = [
    (float(RNG.uniform(20, 400)), float(RNG.uniform(20, 400)),
     float(RNG.uniform(0.02, 3.0)), float(RNG.uniform(0.05, 0.9)),
     float(RNG.uniform(0.0, 0.10)), float(RNG.uniform(0.0, 0.06)))
    for _ in range(200)
]


@pytest.mark.parametrize("S,K,T,sigma,r,q", PARAMS)
def test_put_call_parity(S, K, T, sigma, r, q):
    """C - P = S e^{-qT} - K e^{-rT}, exactly, for every parameter set."""
    c = price("call", S, K, T, sigma, r, q)
    p = price("put", S, K, T, sigma, r, q)
    forward = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert c - p == pytest.approx(forward, abs=1e-9)


def test_known_value():
    """The classic textbook case: S=K=100, T=1, sigma=20%, r=5%, q=0."""
    assert price("call", 100, 100, 1.0, 0.20, 0.05) == pytest.approx(10.4506, abs=1e-4)
    assert price("put", 100, 100, 1.0, 0.20, 0.05) == pytest.approx(5.5735, abs=1e-4)


def test_zero_vol_limit_is_discounted_forward_intrinsic():
    """As sigma -> 0 the option is worth the discounted intrinsic of the forward."""
    S, K, T, r, q = 120.0, 100.0, 0.5, 0.04, 0.01
    expected = np.exp(-r * T) * max(S * np.exp((r - q) * T) - K, 0.0)
    assert price("call", S, K, T, 1e-8, r, q) == pytest.approx(expected, abs=1e-8)


def test_short_expiry_limit_is_intrinsic():
    for S in (80.0, 100.0, 120.0):
        for ot in ("call", "put"):
            assert price(ot, S, 100.0, 1e-8, 0.2, 0.05) == pytest.approx(
                intrinsic(ot, S, 100.0), abs=1e-3)


def test_delta_bounds_and_deep_limits():
    assert greeks("call", 300, 100, 1.0, 0.2, 0.05).delta == pytest.approx(1.0, abs=1e-6)
    assert greeks("call", 30, 100, 1.0, 0.2, 0.05).delta == pytest.approx(0.0, abs=1e-6)
    assert greeks("put", 30, 100, 1.0, 0.2, 0.05).delta == pytest.approx(-1.0, abs=1e-6)
    for S, K, T, sigma, r, q in PARAMS[:50]:
        d = greeks("call", S, K, T, sigma, r, q).delta
        assert 0.0 <= d <= 1.0


def test_monotone_in_vol():
    """Vega > 0: price strictly increases with sigma (both types)."""
    sigmas = np.linspace(0.05, 1.0, 40)
    for ot in ("call", "put"):
        prices = price(ot, 100, 110, 0.7, sigmas, 0.03, 0.01)
        assert np.all(np.diff(prices) > 0)


@pytest.mark.parametrize("ot", ["call", "put"])
@pytest.mark.parametrize("S,K,T,sigma,r,q", PARAMS[:40])
def test_greeks_match_finite_differences(ot, S, K, T, sigma, r, q):
    """Analytic Greeks vs central finite differences of the price function."""
    g = greeks(ot, S, K, T, sigma, r, q)
    h = 1e-4

    delta_fd = (price(ot, S + h, K, T, sigma, r, q) - price(ot, S - h, K, T, sigma, r, q)) / (2 * h)
    gamma_fd = (price(ot, S + h, K, T, sigma, r, q) - 2 * price(ot, S, K, T, sigma, r, q)
                + price(ot, S - h, K, T, sigma, r, q)) / h**2
    vega_fd = (price(ot, S, K, T, sigma + h, r, q) - price(ot, S, K, T, sigma - h, r, q)) / (2 * h)
    rho_fd = (price(ot, S, K, T, sigma, r + h, q) - price(ot, S, K, T, sigma, r - h, q)) / (2 * h)
    theta_fd = -(price(ot, S, K, T + h, sigma, r, q) - price(ot, S, K, T - h, sigma, r, q)) / (2 * h)

    assert g.delta == pytest.approx(delta_fd, abs=1e-5)
    assert g.gamma == pytest.approx(gamma_fd, abs=1e-4)
    assert g.vega == pytest.approx(vega_fd / 100.0, abs=1e-6)
    assert g.rho == pytest.approx(rho_fd / 100.0, abs=1e-6)
    assert g.theta == pytest.approx(theta_fd / 365.0, abs=1e-6)


# IV is only identifiable where price actually moves with sigma. Where vega
# underflows (very deep ITM/OTM), every sigma yields the same float price and
# no solver can recover the input - so restrict the round-trip to cases with
# meaningful vega.
PARAMS_IV = [ps for ps in PARAMS
             if greeks("call", *ps).vega * 100.0 > 1e-2][:60]


@pytest.mark.parametrize("ot", ["call", "put"])
@pytest.mark.parametrize("S,K,T,sigma,r,q", PARAMS_IV)
def test_implied_vol_round_trip(ot, S, K, T, sigma, r, q):
    """price(sigma) -> implied_vol -> recovers sigma."""
    p = price(ot, S, K, T, sigma, r, q)
    res = implied_vol(ot, p, S, K, T, r, q, tol=1e-12)
    assert res is not None and res.converged
    assert res.sigma == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_arbitrage_violations():
    """A call priced below its no-arbitrage floor has no implied vol."""
    S, K, T, r, q = 100.0, 80.0, 0.5, 0.05, 0.0
    floor = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert implied_vol("call", floor - 0.01, S, K, T, r, q) is None
    assert implied_vol("call", -1.0, S, K, T, r, q) is None


def test_implied_vol_handles_tiny_vega():
    """Deep OTM short-dated: vega ~ 0, Newton alone would diverge; the
    bisection safeguard must still converge."""
    S, K, T, r, q = 100.0, 150.0, 0.05, 0.05, 0.0
    p = price("call", S, K, T, 0.35, r, q)  # ~ nothing, but positive
    res = implied_vol("call", p, S, K, T, r, q, tol=1e-12)
    assert res is not None and res.converged
    assert res.sigma == pytest.approx(0.35, abs=1e-6)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        price("straddle", 100, 100, 1.0, 0.2, 0.05)
