"""Build the deployed market-risk dashboard: one self-contained docs/index.html
covering Phases 3-10 (NAV, VaR/ES, vol engines, backtesting, model comparison,
stress). Data is embedded as JSON so the page needs no server and no CSVs at
view time; Plotly comes from CDN. Re-run after regenerating outputs/.

    /usr/local/bin/python3 build_dashboard.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs")
DOCS = Path("docs"); DOCS.mkdir(exist_ok=True)

WINDOWS = (250, 500, 750)
CONFS = (0.95, 0.99, 0.995)
HEADLINE_W, HEADLINE_C = 250, 0.99

# Entity -> color. Three validated categorical slots for the vol-scaled models;
# unscaled HS is the benchmark, so it wears neutral ink rather than a 4th slot
# (slot 4 puts yellow beside orange, which fails the all-pairs CVD floor).
COLOR = {"GJR": "#2a78d6", "GARCH": "#eb6834", "EWMA": "#1baf7a", "HS": "#52514e"}
LABEL = {"GJR": "GJR-GARCH", "GARCH": "GARCH(1,1)", "EWMA": "EWMA", "HS": "HS (unscaled)"}
ORDER = ["GJR", "GARCH", "EWMA", "HS"]
CSV_MODEL = {"Historical Simulation": "HS", "EWMA-scaled HS": "EWMA",
             "GARCH-scaled HS": "GARCH", "GJR-GARCH-scaled HS": "GJR"}

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
STATUS = {"GREEN": "#0ca30c", "AMBER": "#fab219", "YELLOW": "#fab219", "RED": "#d03b3b"}


def tag(c): return f"{100*c:g}".replace(".", "p")
def key(w, c): return f"{w}|{tag(c)}"
def clean(s): return [None if not np.isfinite(v) else round(float(v), r) for v, r in zip(s, [0]*len(s))]
def r2(s, nd=4): return [None if not np.isfinite(v) else round(float(v), nd) for v in s]


# ---------------------------------------------------------------- load
files = {"EWMA": "EWMA_Scaled_VaR.csv", "GARCH": "GARCH_Scaled_VaR.csv",
         "GJR": "GJR_GARCH_Scaled_VaR.csv"}
frames = {k: pd.read_csv(OUT / f, index_col=0, parse_dates=True) for k, f in files.items()}
base = frames["GJR"]
dates = [d.strftime("%Y-%m-%d") for d in base.index]
pnl = base["Portfolio_PnL"].astype(float)
nav_series = base["Start_of_Day_Value"].astype(float)

port = pd.read_csv("Portfolio_Construction.csv", index_col=0, parse_dates=True)
NAV = float(port["Value"].iloc[-1])

bt = pd.read_csv(OUT / "Backtest_Results.csv")
mc = pd.read_csv(OUT / "Model_Comparison.csv")
for df in (bt, mc):
    df["M"] = df["Model"].map(CSV_MODEL)

stress = pd.read_csv(OUT / "Stress_Historical.csv")
rev = pd.read_csv(OUT / "Stress_Reverse.csv")
gjr_p = pd.read_csv(OUT / "GJR_GARCH_Parameters.csv", index_col=0)["value"]

# portfolio beta (same CAPM overlay as Phase 10)
rets = pd.read_csv("Clean_Return_Matrix.csv", index_col=0, parse_dates=True)
bpx = pd.read_csv("00003069-Benchmark_prices.csv", index_col=0, parse_dates=True).sort_index()
bret = np.log(bpx / bpx.shift(1)).iloc[:, 0].reindex(rets.index).dropna()
com = rets.index.intersection(bret.index)
betas = {t: float(np.polyfit(bret.reindex(com).values, rets[t].reindex(com).values, 1)[0])
         for t in rets.columns}
BETA = float(np.mean(list(betas.values())))

# ---------------------------------------------------------------- slices
slices = {}
for w in WINDOWS:
    for c in CONFS:
        t = tag(c)
        var = {}
        for m in ORDER:
            src = base if m == "HS" else frames[m]
            col = f"{m}_VaR_{w}d_{t}"
            var[m] = [None if not np.isfinite(v) else round(float(v))
                      for v in src[col].astype(float).values] if col in src else None
        b = bt[(bt.Window == w) & (bt.Confidence == c)].set_index("M")
        k = mc[(mc.Window == w) & (mc.Confidence == c)].set_index("M")
        rows = []
        for m in ORDER:
            if m not in b.index:
                continue
            r, q = b.loc[m], (k.loc[m] if m in k.index else {})
            rows.append({
                "m": m, "n": int(r["N"]), "exc": int(r["Exceptions"]),
                "rate": round(float(r["Rate"]), 6), "exp": round(float(r["Expected"]), 1),
                "kp": round(float(r["Kupiec_p"]), 4), "ip": round(float(r["Ind_p"]), 4),
                "cc": round(float(r["CC_p"]), 4), "basel": str(r["Basel"]),
                "mvar": round(float(q.get("MeanVaR_pctNAV", np.nan)), 3),
                "pin": round(float(q.get("Pinball_bps", np.nan)), 3),
            })
        # composite rank: CC_p up, |rate-target| down, mean VaR down, pinball down
        kk = k.reindex([r["m"] for r in rows])
        rk = pd.DataFrame(index=kk.index)
        rk["cc"] = kk["CC_p"].rank(ascending=False, method="min")
        rk["ce"] = (kk["Rate"] - (1 - c)).abs().rank(ascending=True, method="min")
        rk["mv"] = kk["MeanVaR_pctNAV"].rank(ascending=True, method="min")
        rk["pb"] = kk["Pinball_bps"].rank(ascending=True, method="min")
        avg = rk.mean(axis=1)
        for r in rows:
            r["rank"] = round(float(avg.get(r["m"], np.nan)), 2)
        slices[key(w, c)] = {"var": var, "rows": rows,
                             "target": round(1 - c, 5), "best": str(avg.idxmin())}

vols = {m: r2(frames[m][f"{m}_Vol_Annualized"].astype(float).values * 100, 3)
        for m in ("EWMA", "GARCH", "GJR")}
loss = [None if not np.isfinite(v) else round(-float(v)) for v in pnl.values]

h = slices[key(HEADLINE_W, HEADLINE_C)]
hrow = {r["m"]: r for r in h["rows"]}
HVAR = next(v for v in reversed(h["var"]["GJR"]) if v is not None)
worst = stress.loc[stress["Loss_pctNAV"].idxmax()]

payload = {
    "dates": dates, "loss": loss, "vols": vols, "slices": slices,
    "color": COLOR, "label": LABEL, "order": ORDER,
    "stress": [{"s": r["Scenario"], "t": r["Type"], "p": round(float(r["Loss_pctNAV"]), 4),
                "x": round(float(r["xVaR"]), 2), "l": round(float(r["Loss"]))}
               for _, r in stress.iterrows()],
    "rev": [{"k": r["Loss trigger"], "l": round(float(r["Loss"])),
             "s": round(float(r["Required S&P shock"]), 4)} for _, r in rev.iterrows()],
    "nav": round(NAV), "beta": round(BETA, 3), "hvar": HVAR,
    "windows": list(WINDOWS), "confs": [tag(c) for c in CONFS],
    "confLabel": {tag(c): f"{100*c:g}%" for c in CONFS},
    "hw": HEADLINE_W, "hc": tag(HEADLINE_C),
    "span": f"{dates[0]} to {dates[-1]}",
    "gjr": {k: (round(float(gjr_p[k]), 4) if k in gjr_p.index else None)
            for k in ("alpha", "gamma", "beta", "persistence", "vol_half_life")},
    "status": STATUS, "ink": INK, "ink2": INK2, "muted": MUTED,
    "grid": GRID, "base": BASE, "surf": SURF,
}
enc = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
tpl = Path("_dashboard.html").read_text()
assert "__PAYLOAD__" in tpl, "_dashboard.html lost its payload slot"
html = tpl.replace("__PAYLOAD__", enc)
(DOCS / "index.html").write_text(html)
(DOCS / ".nojekyll").write_text("")
print(f"wrote docs/index.html  ({len(html)/1024:.0f} KB, data {len(enc)/1024:.0f} KB inline)")
print(f"headline: NAV {payload['nav']:,} · VaR {payload['hvar']:,} · beta {payload['beta']}")
