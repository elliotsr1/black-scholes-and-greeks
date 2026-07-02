# Black–Scholes explorer

An interactive Python app for European option pricing, Greeks, and implied
volatility under the Black–Scholes–Merton model — including a Newton-method
IV solver with a visible iteration trace, and a volatility smile computed
live from real SPY market data.

Built with NumPy/SciPy for the math, Streamlit + Plotly for the interface,
and pytest for verification (408 tests).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py       # opens the app in your browser
pytest                     # runs the test suite
```

## What's inside

| File | Role |
|---|---|
| `black_scholes.py` | Core library: pricing, Greeks, implied vol. Pure NumPy/SciPy, no UI dependencies — importable on its own. |
| `app.py` | Streamlit dashboard: sliders, Greeks profiles, IV solver, smile. |
| `test_black_scholes.py` | Pytest suite: parity, limits, finite differences, IV round-trips. |
| `data/spy_chain.json` | SPY call-chain snapshot (CBOE delayed quotes, 2026-07-02). |

## Features

- **Live pricing dashboard** — sliders for spot `S`, strike `K`, time to
  expiry `T`, volatility `σ`, risk-free rate `r`, and dividend yield `q`,
  with a call/put toggle. Price and all Greeks re-compute on drag, plus
  profiles of value, delta, gamma, vega, and theta across the spot range.
- **Implied volatility solver** — enter a market price and Newton's method
  inverts the pricer to recover the volatility, with the full iteration
  trace shown (σ, model price, error at each step). Seeded with the
  Brenner–Subrahmanyam approximation and safeguarded by a bisection bracket,
  so it converges even where vega is tiny; prices outside the no-arbitrage
  bounds are rejected.
- **Volatility smile from real data** — a snapshot of SPY call mid-quotes
  (CBOE delayed quotes, 3 expiries × ~50 liquid strikes) is inverted through
  that same solver, using the current `r` and `q` sliders as carry
  assumptions. The downward skew — and its steepening at short expiry — is
  the market pricing fat left tails that the lognormal model doesn't have.
- **Data tables** — the plotted grids and the quotes/solved IVs are all
  available as tables, not just charts.

## The math

For a European option with continuous dividend yield `q`:

```
d1 = [ln(S/K) + (r − q + σ²/2)·T] / (σ·√T)
d2 = d1 − σ·√T

call = S·e^(−qT)·N(d1) − K·e^(−rT)·N(d2)
put  = K·e^(−rT)·N(−d2) − S·e^(−qT)·N(−d1)
```

Greeks are the partial derivatives of price, reported in trading
conventions: vega and rho per 1 percentage point, theta per calendar day
(annual / 365).

Implied volatility solves `BS(σ) = market price` for `σ` by Newton's method,

```
σ(n+1) = σ(n) − [BS(σ(n)) − market price] / vega(σ(n))
```

with each iterate also tightening a `[lo, hi]` bracket (price is monotone in
σ); any Newton step that leaves the bracket is replaced by bisection, so
convergence is guaranteed. Typical convergence on the SPY chain is ≤ 10
iterations to ~1e-9.

## What the tests verify

- **Put–call parity** `C − P = S·e^(−qT) − K·e^(−rT)` holds to 1e-9 across
  200 random parameter sets.
- **Limits**: σ→0 gives the discounted intrinsic of the forward; T→0 gives
  the payoff; deep ITM/OTM deltas go to ±1/0.
- **Greeks match finite differences** of the price function (central
  differences, all five Greeks, both option types).
- **IV round-trips**: `price(σ) → implied_vol → σ` to 1e-6 — restricted to
  parameter sets where vega is meaningful, because where vega underflows,
  many σ values produce the identical float price and implied vol is
  mathematically unidentifiable (a property of the problem, not the solver).
- **No-arbitrage rejection**: prices below the call floor have no IV and are
  refused rather than fitted.

## Assumptions & caveats

European exercise, lognormal underlying, constant `σ`, `r`, `q`, no
frictions. SPY options are American-style, so their Black–Scholes IVs are an
approximation — reasonable for calls on a low-dividend underlying, where the
early-exercise premium is small. Smile quotes below the no-arbitrage floor
for the chosen `r`/`q` are omitted rather than force-fitted.
