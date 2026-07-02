"""Black-Scholes-Merton pricing, Greeks, and implied volatility.

European options on an asset with continuous dividend yield q. All functions
are vectorized over their inputs via NumPy broadcasting.

Conventions (trading units):
- vega  is per 1 volatility point (i.e. d price / d sigma, divided by 100)
- theta is per calendar day (annual theta / 365)
- rho   is per 1 rate point (d price / d r, divided by 100)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, sigma, r, q):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def price(option_type: str, S, K, T, sigma, r, q=0.0):
    """Black-Scholes price of a European call or put."""
    d1, d2 = _d1_d2(S, K, T, sigma, r, q)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    if option_type == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    if option_type == "put":
        return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


@dataclass
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float    # per 1 vol point
    theta: float   # per calendar day
    rho: float     # per 1 rate point


def greeks(option_type: str, S, K, T, sigma, r, q=0.0) -> Greeks:
    """Price and Greeks, in the trading units documented at module top."""
    d1, d2 = _d1_d2(S, K, T, sigma, r, q)
    sqrt_T = np.sqrt(T)
    df_r, df_q = np.exp(-r * T), np.exp(-q * T)
    pdf_d1 = norm.pdf(d1)

    gamma = df_q * pdf_d1 / (S * sigma * sqrt_T)
    vega = S * df_q * pdf_d1 * sqrt_T

    if option_type == "call":
        p = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
        delta = df_q * norm.cdf(d1)
        theta = (-S * df_q * pdf_d1 * sigma / (2 * sqrt_T)
                 - r * K * df_r * norm.cdf(d2)
                 + q * S * df_q * norm.cdf(d1))
        rho = K * T * df_r * norm.cdf(d2)
    elif option_type == "put":
        p = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
        delta = df_q * (norm.cdf(d1) - 1.0)
        theta = (-S * df_q * pdf_d1 * sigma / (2 * sqrt_T)
                 + r * K * df_r * norm.cdf(-d2)
                 - q * S * df_q * norm.cdf(-d1))
        rho = -K * T * df_r * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    return Greeks(price=p, delta=delta, gamma=gamma,
                  vega=vega / 100.0, theta=theta / 365.0, rho=rho / 100.0)


def intrinsic(option_type: str, S, K):
    """Payoff at expiry."""
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


@dataclass
class IVResult:
    sigma: float
    iterations: int
    trace: list  # (iteration, sigma, model_price, error) per step
    converged: bool


def implied_vol(option_type: str, target_price: float, S, K, T, r, q=0.0,
                tol: float = 1e-9, max_iter: int = 60,
                lo: float = 1e-4, hi: float = 5.0) -> IVResult | None:
    """Invert Black-Scholes for volatility with a safeguarded Newton's method.

    Newton's method uses vega as the derivative and is seeded with the
    Brenner-Subrahmanyam ATM approximation sigma ~ price/S * sqrt(2*pi/T).
    Every iterate also tightens a [lo, hi] bracket (price is monotone in
    sigma); if a Newton step leaves the bracket - which happens where vega is
    tiny, deep ITM/OTM - the step falls back to bisection, so convergence is
    guaranteed.

    Converges when the price error is within tol or the sigma bracket has
    collapsed to machine precision (the best any solver can do where vega is
    tiny and price is locally flat in sigma). Note that IV is ill-conditioned
    in that regime: many sigmas reproduce the same float price, so the result
    is *a* consistent vol, not a well-identified one.

    Returns None when target_price is outside the no-arbitrage bounds
    [price(lo), price(hi)] for these inputs.
    """
    if not (price(option_type, S, K, T, lo, r, q) < target_price
            < price(option_type, S, K, T, hi, r, q)):
        return None

    sigma = float(np.clip(np.sqrt(2 * np.pi / T) * target_price / S, 0.05, hi))
    trace = []
    for i in range(max_iter):
        g = greeks(option_type, S, K, T, sigma, r, q)
        diff = g.price - target_price
        trace.append((i, sigma, g.price, diff))
        if abs(diff) < tol:
            return IVResult(sigma=sigma, iterations=i, trace=trace, converged=True)
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        if hi - lo < 1e-12:
            return IVResult(sigma=sigma, iterations=i, trace=trace, converged=True)
        vega = g.vega * 100.0  # back to per-1.0-vol units for the Newton step
        newton = sigma - diff / vega if vega > 1e-10 else None
        sigma = newton if (newton is not None and lo < newton < hi) else 0.5 * (lo + hi)

    return IVResult(sigma=sigma, iterations=max_iter, trace=trace, converged=False)
