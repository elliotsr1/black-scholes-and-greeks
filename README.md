# Black–Scholes explorer

An interactive, single-file visualization of European option pricing, Greeks,
and implied volatility under the Black–Scholes–Merton model — including a live
Newton-method IV solver and a real SPY volatility smile. No dependencies, no
build step — open `index.html` in any browser.

## Features

- **Live inputs** — sliders for spot `S`, strike `K`, time to expiry `T`,
  volatility `σ`, risk-free rate `r`, and continuous dividend yield `q`, plus a
  call/put toggle. Everything re-computes on drag.
- **Price + Greeks tiles** — option price, delta, gamma, vega (per 1 vol
  point), theta (per calendar day), and rho (per 1 rate point).
- **Value vs. spot chart** — today's value against the expiry payoff; the gap
  between the curves is time value. Hover (or focus and use arrow keys) for a
  crosshair readout; the current spot is marked on every chart.
- **Greek profiles** — small multiples of delta, gamma, vega, and theta across
  the spot range.
- **Implied volatility solver** — enter a market price and Newton's method
  inverts the pricer to recover the volatility, with the full iteration trace
  shown (σ, model price, error at each step). Seeded with the
  Brenner–Subrahmanyam approximation and safeguarded by a bisection bracket,
  so it converges even where vega is tiny; prices outside the no-arbitrage
  bounds are rejected.
- **Volatility smile from real data** — an embedded snapshot of SPY call
  mid-quotes (CBOE delayed quotes, 3 expiries × ~50 strikes) is inverted
  through that same solver in the browser, using the current r and q sliders
  as carry assumptions. The downward skew — and its steepening at short
  expiry — is the market pricing fat left tails that the lognormal model
  doesn't have.
- **Data tables** — the plotted greeks grid and the quotes/IVs are also
  available as tables, so no value is gated behind hover.
- Light and dark mode follow the system preference.

## The math

For a European option with continuous dividend yield `q`:

```
d1 = [ln(S/K) + (r − q + σ²/2)·T] / (σ·√T)
d2 = d1 − σ·√T

call = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
put  = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1)
```

Greeks are the partial derivatives of price, reported in trading conventions:
vega and rho are scaled per 1 percentage point, theta per calendar day
(annual / 365). The normal CDF uses the Abramowitz & Stegun 7.1.26 erf
approximation (|error| < 1.5e−7).

Implied volatility solves `BS(σ) = market price` for `σ` by Newton's method,

```
σ(n+1) = σ(n) − [BS(σ(n)) − market price] / vega(σ(n))
```

with each iterate also tightening a `[lo, hi]` bracket; any Newton step that
leaves the bracket is replaced by bisection, so convergence is guaranteed.
Typical convergence on the SPY chain is ≤ 10 iterations to ~1e-9.

Assumptions: European exercise, lognormal underlying, constant `σ`, `r`, `q`,
no transaction costs or early exercise. (SPY options are American-style, so
their Black–Scholes IVs are an approximation — good for calls on a
low-dividend underlying, where early exercise premium is small.)
