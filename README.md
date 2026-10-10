# Advanced Equity Portfolio Risk Platform

End-to-end market-risk analytics for a 10-stock equal-weighted US equity book: value-at-risk
across four model families, regulatory backtesting, model selection and stress testing.

**QFI Internship — Team 2 (Market Risk)** · Atharv Pratap Singh Chauhan · IIT Bombay

**[→ Live risk dashboard](https://atharv-iit-bombay.github.io/market-risk-var-platform/)**

---

## Headline results

| | |
|---|---|
| Book | 10 large-cap US equities, equal-weighted, monthly rebalance, ₹1M start |
| NAV (end of sample) | **₹2,995,852** |
| Headline VaR | **₹120,039** — 99%, 250-day, GJR-GARCH-scaled (4.01% of NAV) |
| Recommended model | **GJR-GARCH-scaled HS** — best composite rank at 99%/250d |
| Portfolio beta | 1.214 (vs S&P 500) |
| Unscaled HS | fails conditional coverage (CC p = 0.037) at the headline cell |

GJR-GARCH's leverage term is the substantive finding: γ = 0.0866 (p = 0.002) is significant,
while the symmetric ARCH term goes insignificant (p = 0.754) once γ absorbs the asymmetry.
Down-moves drive volatility more than up-moves of the same size.

## Pipeline

Ten phases, each a notebook in this repo:

| Phase | Notebook | What it does |
|---|---|---|
| 1 | `..._Task1.ipynb` | Data collection — prices, benchmark, returns |
| 2 | `..._Task2.ipynb` | Cleaning — outliers, missing data, corporate actions |
| 3 | `..._Task3.ipynb` | Portfolio construction — equal weight, monthly rebalance |
| 4 | `..._Task4.ipynb` | Historical-simulation VaR & ES |
| 5 | `..._Task5.ipynb` | EWMA volatility scaling |
| 6 | `..._Task6.ipynb` | GARCH(1,1) volatility scaling |
| 7 | `..._Task7.ipynb` | GJR-GARCH volatility scaling |
| 8 | `..._Task8.ipynb` | Backtesting — Kupiec, Christoffersen, Basel traffic light |
| 9 | `..._Task9.ipynb` | Model comparison — calibration, efficiency, scoring rules |
| 10 | `..._Task10.ipynb` | Stress testing — historical replay, shocks, reverse stress |

Notebooks are executed with outputs saved, so every figure and table is visible on GitHub
without running anything.

## Method notes

**VaR estimation.** Historical simulation at 95 / 99 / 99.5% confidence over 250 / 500 / 750-day
windows. The scaled variants divide each historical return by the ratio of current to
historical volatility (Hull–White) before taking the empirical quantile, which makes the
forecast responsive to volatility clusters without abandoning the empirical tail.

**Volatility models.** EWMA (λ = 0.94), GARCH(1,1) and GJR-GARCH, fitted with
[`arch`](https://github.com/bashtage/arch) — Bollerslev–Wooldridge robust standard errors,
with parameter t-statistics and p-values reported. Returns are scaled ×100 inside the fit for
numerical stability. A scipy SLSQP Gaussian-QMLE fallback keeps the notebooks runnable where
`arch` is unavailable.

**Backtesting.** Kupiec proportion-of-failures, Christoffersen independence and conditional
coverage, and the Basel traffic-light zones.

**Model selection.** Four headline criteria — conditional-coverage p-value (↑), coverage error
(↓), mean VaR as % of NAV (↓), pinball loss (↓) — ranked 1–4 and averaged. Calibration enters
twice because regulatory acceptance dominates the decision.

**Stress testing.** CAPM-beta overlay for out-of-sample crises (2008 GFC, COVID-2020),
instantaneous market shocks (−5/−10/−20/−30%), a volatility-doubling shock, a correlation
shock to 1, a sector shock, and reverse stress tests solving for the market move behind a
target loss.

## Dashboard

`docs/index.html` is a single self-contained page — all data inline, no server, no build step
at view time. Sections: VaR forecast paths against realised loss, the volatility engines, the
regulatory backtest, the composite ranking, stress results and reverse stress. The
window/confidence filters scope the first four sections; stress is absolute. Every chart has a
table view, and it follows the OS light/dark setting with a manual toggle.

To rebuild after regenerating `outputs/`:

```bash
/usr/local/bin/python3 build_dashboard.py
```

## Repository layout

```
*_TaskN.ipynb            ten phase notebooks (executed, outputs embedded)
build_dashboard.py       rebuilds docs/index.html from outputs/
_dashboard.html          dashboard template
outputs/                 CSVs + charts exported by the phases
  charts/                18 rendered figures
docs/                    deployed dashboard (GitHub Pages)
*.csv                    input data: clean returns, prices, benchmark, portfolio
```

## Running

Python 3.13 with `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `plotly` and `arch`.
Notebooks read the input CSVs from the repository root, so they run in place with no path
setup. Each notebook is self-contained and writes its exports into `outputs/`.

## Limitations

Rankings rest on one 2020–2026 realised path, dominated by an equity bull run; a crisis-heavy
regime could reorder the models. Scoring uses clean close-to-close returns with no fees or
slippage. Stress losses assume a constant linear beta and zero idiosyncratic move, which
understates a growth-tilted book in a real crash, and risk factors are estimated on trailing
windows that understate their own crisis values.
