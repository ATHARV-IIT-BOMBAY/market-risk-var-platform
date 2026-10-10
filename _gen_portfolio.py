"""Reproduce Task2 (clean) + Task3 (construct) to emit real Portfolio_Construction.csv.
Mirrors the user's own notebook logic exactly so Phases 6-10 run on the same numbers
as their submitted Phases 2-3. One-off generator; not a deliverable."""
import numpy as np, pandas as pd

RAW   = '00003070-Raw_Portfolio_Price.csv'
BENCH = '00003069-Benchmark_prices.csv'
INITIAL_VALUE = 1_000_000
REBALANCE     = 'ME'
WINSOR = (0.005, 0.995)

# --- Task 2: clean ------------------------------------------------------------
prices = pd.read_csv(RAW, index_col=0, parse_dates=True).sort_index()
bench  = pd.read_csv(BENCH, index_col=0, parse_dates=True).sort_index()
prices = prices.reindex(bench.index).ffill().dropna()        # align to benchmark calendar
tickers = list(prices.columns)

ret = np.log(prices / prices.shift(1)).dropna()
lo, hi = ret.quantile(WINSOR[0]), ret.quantile(WINSOR[1])    # per-column winsorize
returns = ret.clip(lower=lo, upper=hi, axis=1)

prices.to_csv('Clean_Price_Matrix.csv')
returns.to_csv('Clean_Return_Matrix.csv')

# --- Task 3: construct (equal weight, month-end rebalance) --------------------
n = len(tickers)
weights = pd.Series(1.0 / n, index=tickers)
rebal_dates = set(returns.resample(REBALANCE).last().index)
cur_w = weights.copy(); port_ret = []
for date, row in returns.iterrows():
    port_ret.append(float((cur_w * row).sum()))
    cur_w = cur_w * np.exp(row); cur_w /= cur_w.sum()
    if date in rebal_dates:
        cur_w = weights.copy()
portfolio_returns = pd.Series(port_ret, index=returns.index, name='Portfolio')

value = INITIAL_VALUE * (1 + portfolio_returns).cumprod()
start = value.index[0] - pd.Timedelta(days=1)
value_full = pd.concat([pd.Series({start: INITIAL_VALUE}), value])
pnl = value_full.diff().dropna()

result = pd.DataFrame({'Return': portfolio_returns, 'Value': value, 'PnL': pnl})
result.to_csv('Portfolio_Construction.csv')

# --- sanity check -------------------------------------------------------------
tot = value.iloc[-1] / INITIAL_VALUE - 1
yrs = (value.index[-1] - value.index[0]).days / 365.25
cagr = (value.iloc[-1] / INITIAL_VALUE) ** (1 / yrs) - 1
vol = portfolio_returns.std() * np.sqrt(252)
assert len(result) == len(returns) and result['Value'].notna().all()
assert result['Value'].iloc[-1] > 0
print(f'{len(result)} days x {n} stocks | {returns.index[0].date()}..{returns.index[-1].date()}')
print(f'Final value {value.iloc[-1]:,.0f} | total {tot:.2%} | CAGR {cagr:.2%} | vol {vol:.2%}')
print('wrote: Clean_Price_Matrix.csv, Clean_Return_Matrix.csv, Portfolio_Construction.csv')
