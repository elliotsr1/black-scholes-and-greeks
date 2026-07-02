"""Interactive Black-Scholes explorer: pricing, Greeks, implied volatility,
and a real SPY volatility smile. Run with: streamlit run app.py"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from black_scholes import greeks, implied_vol, intrinsic, price

# Palette (colorblind-safe ordering; see README)
S1, S2, S3 = "#2a78d6", "#1baf7a", "#eda100"
GRID, BASELINE, MUTED = "#e1e0d9", "#c3c2b7", "#898781"

st.set_page_config(page_title="Black–Scholes explorer", layout="wide")


def styled(fig: go.Figure, *, height: int = 320, y_title: str = "") -> go.Figure:
    """Shared chart chrome: hairline grid, unified hover, quiet axes."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(size=13),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASELINE, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=BASELINE, zerolinecolor=BASELINE,
                     zerolinewidth=1, title_text=y_title)
    return fig


def line(x, y, name, color, width=2):
    return go.Scatter(x=x, y=y, name=name, mode="lines",
                      line=dict(color=color, width=width))


@st.cache_data
def load_chain():
    return json.loads((Path(__file__).parent / "data" / "spy_chain.json").read_text())


@st.cache_data
def smile_ivs(r: float, q: float):
    """Invert every chain quote through the solver at the given carry."""
    chain = load_chain()
    out = []
    for exp in chain["expiries"]:
        T = exp["days"] / 365.0
        ks, ivs = [], []
        for quote in exp["calls"]:
            res = implied_vol("call", quote["mid"], chain["spot"], quote["k"], T, r, q)
            if res is not None:
                ks.append(quote["k"])
                ivs.append(res.sigma * 100)
        out.append({"label": f"{exp['date']} · {exp['days']}d", "k": ks, "iv": ivs})
    return out


# ---------- controls ----------

st.title("Black–Scholes explorer")
st.caption("European option pricing and Greeks under Black–Scholes–Merton. "
           "Drag the sliders; everything below re-computes live.")

top = st.columns([1.2, 1, 1, 1, 1, 1, 1])
option_type = top[0].radio("Option type", ["call", "put"], horizontal=True)
S = top[1].slider("Spot S ($)", 20.0, 250.0, 100.0, 1.0)
K = top[2].slider("Strike K ($)", 20.0, 250.0, 100.0, 1.0)
T = top[3].slider("Expiry T (yr)", 0.02, 3.0, 1.0, 0.02)
sigma = top[4].slider("Vol σ (%)", 1.0, 100.0, 20.0, 1.0, key="sigma_pct") / 100
r = top[5].slider("Rate r (%)", 0.0, 15.0, 5.0, 0.25) / 100
q = top[6].slider("Div yield q (%)", 0.0, 10.0, 1.25, 0.25) / 100

# ---------- price + greeks tiles ----------

now = greeks(option_type, S, K, T, sigma, r, q)
tiles = st.columns(6)
tiles[0].metric("Option price", f"${now.price:,.2f}")
tiles[1].metric("Delta Δ", f"{now.delta:.4f}", help="price change per $1 of spot")
tiles[2].metric("Gamma Γ", f"{now.gamma:.4f}", help="delta change per $1 of spot")
tiles[3].metric("Vega", f"{now.vega:.4f}", help="per 1 vol point")
tiles[4].metric("Theta Θ", f"{now.theta:.4f}", help="per calendar day")
tiles[5].metric("Rho ρ", f"{now.rho:.4f}", help="per 1 rate point")

# ---------- value vs spot ----------

spots = np.linspace(min(0.4 * K, 0.75 * S), max(1.6 * K, 1.25 * S), 161)
curve = greeks(option_type, spots, K, T, sigma, r, q)
pay = intrinsic(option_type, spots, K)

st.subheader("Option value vs. spot price")
st.caption("Value today against the payoff locked in at expiry — the gap between the curves is time value.")
fig = go.Figure([
    line(spots, curve.price, "Value today", S1),
    line(spots, pay, "Payoff at expiry", S2),
])
fig.add_vline(x=S, line_width=1, line_color=BASELINE,
              annotation_text="S", annotation_font_color=MUTED)
st.plotly_chart(styled(fig, y_title="$"), width="stretch")

# ---------- greek profiles ----------

GREEK_PANELS = [
    ("delta", "Delta across spot", "Hedge ratio: steepens through the strike."),
    ("gamma", "Gamma across spot", "Peaks near the strike; grows as expiry nears."),
    ("vega", "Vega across spot", "Volatility sensitivity, largest at the money."),
    ("theta", "Theta across spot", "Daily time decay, usually deepest at the money."),
]
for row in (GREEK_PANELS[:2], GREEK_PANELS[2:]):
    cols = st.columns(2)
    for col, (attr, title, desc) in zip(cols, row):
        with col:
            st.subheader(title)
            st.caption(desc)
            g = go.Figure([line(spots, getattr(curve, attr), attr, S1)])
            g.add_vline(x=S, line_width=1, line_color=BASELINE)
            g.update_layout(showlegend=False)
            st.plotly_chart(styled(g, height=230), width="stretch")

with st.expander("Data table — value and Greeks across spot"):
    st.dataframe(pd.DataFrame({
        "Spot": spots, "Value": curve.price, "Payoff": pay, "Delta": curve.delta,
        "Gamma": curve.gamma, "Vega": curve.vega, "Theta": curve.theta, "Rho": curve.rho,
    }).iloc[::16].style.format(precision=4), hide_index=True, width="stretch")

# ---------- implied vol solver ----------

st.subheader("Implied volatility solver")
st.caption("Runs the pricer in reverse: given a market price, Newton's method finds the "
           "volatility that reproduces it, using vega as the derivative (with a bisection "
           "safeguard when a step leaves the bracket). Uses the current S, K, T, r, q and "
           "option type above.")

left, right = st.columns([1, 1.4])
with left:
    default_mkt = round(float(price(option_type, S, K, T, 0.30, r, q)), 2)
    mkt = st.number_input("Market option price ($)", min_value=0.0,
                          value=default_mkt, step=0.05)
    res = implied_vol(option_type, mkt, S, K, T, r, q)
    if res is None:
        st.metric("Implied volatility", "—")
        st.caption("Price is outside the no-arbitrage bounds for these inputs.")
    else:
        st.metric("Implied volatility", f"{res.sigma * 100:.2f}%",
                  help=f"converged in {res.iterations} iterations")
        st.button("Apply to σ slider", on_click=lambda v=res.sigma:
                  st.session_state.update(sigma_pct=round(np.clip(v, 0.01, 1.0) * 100)))
with right:
    if res is not None:
        st.dataframe(
            pd.DataFrame(res.trace, columns=["#", "σ", "Model price", "Error"])
            .style.format({"σ": "{:.4%}", "Model price": "${:,.4f}", "Error": "{:.2e}"}),
            hide_index=True, width="stretch")

# ---------- volatility smile ----------

st.subheader("Volatility smile — SPY calls")
chain = load_chain()
st.caption(f"Real market mid-quotes ({chain['source']} snapshot, {chain['asof']}, "
           f"spot ${chain['spot']:,.2f}) inverted through the solver above, using the "
           "current r and q as carry assumptions. If Black–Scholes described the market, "
           "each line would be flat — the skew is the market pricing fat left tails.")

smile = smile_ivs(r, q)
fig = go.Figure([line(s["k"], s["iv"], s["label"], c)
                 for s, c in zip(smile, (S1, S2, S3))])
fig.add_vline(x=chain["spot"], line_width=1, line_color=BASELINE,
              annotation_text="spot", annotation_font_color=MUTED)
st.plotly_chart(styled(fig, y_title="Implied vol (%)"), width="stretch")

with st.expander("Data table — quotes and solved IVs"):
    frames = []
    for exp, s in zip(chain["expiries"], smile):
        iv_by_k = dict(zip(s["k"], s["iv"]))
        frames.append(pd.DataFrame({
            "Expiry": exp["date"], "Strike": [c["k"] for c in exp["calls"]],
            "Bid": [c["bid"] for c in exp["calls"]],
            "Ask": [c["ask"] for c in exp["calls"]],
            "Mid": [c["mid"] for c in exp["calls"]],
            "IV (%)": [iv_by_k.get(c["k"]) for c in exp["calls"]],
        }))
    st.dataframe(pd.concat(frames).style.format(precision=2),
                 hide_index=True, width="stretch")

st.caption("Assumes a European option on a lognormal asset with constant volatility and "
           "rates, continuous dividend yield q, and no frictions. Vega per 1 vol point, "
           "theta per calendar day, rho per 1 rate point. SPY options are American-style, "
           "so Black–Scholes implied vols are an approximation; quotes below the "
           "no-arbitrage floor for the chosen r and q are omitted.")
