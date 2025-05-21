
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# --- Parameters ---
tickers         = ['AAPL', 'MSFT']
start_date      = '2022-01-01'
end_date        = '2024-01-01'
vol_target      = 0.02
rules           = ['A1', 'A2', 'B1']
rolling_windows = {'A1': 5, 'A2': 20, 'B1': 10}

# --- 1. Fetch data & compute raw returns ---
data     = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
close    = data['Close']
raw_rets = close.pct_change()  # keep NaNs for alignment

# --- 2. Build forecasts dict & DataFrame ---
forecast_data = {}
for rule in rules:
    window = rolling_windows[rule]
    for t in tickers:
        # simple MA forecast minus price
        f = close[t].rolling(window=window).mean() - close[t]
        forecast_data[(rule, t)] = f.shift(1)  # shift to avoid lookahead
forecasts = pd.DataFrame(forecast_data).dropna()
sim_index = forecasts.index  # base simulation index

# --- 3. Align returns & forecasts ---
rets = raw_rets.reindex(sim_index).dropna()       # drop first NaN
forecasts = forecasts.shift(1).loc[rets.index]    # shift forecasts to match returns
sim_index = rets.index

# --- 4. Vol-target and combine helpers ---
def vol_targeted(s):
    rv = s.std() * np.sqrt(252)
    return s * (vol_target / rv) if rv > 0 else s

def combine_forecasts(weights):
    df = pd.DataFrame(index=sim_index)
    for idx, rule in enumerate(rules):
        for t in tickers:
            df[t] = df.get(t, 0) + weights[idx] * forecasts[(rule, t)]
    return df

# --- 5. Standalone rule Sharpe calculation & plot forecasts ---
def calc_rule_sharpe(rule):
    df = pd.DataFrame({t: forecasts[(rule, t)] for t in tickers})
    vt = df.apply(vol_targeted)
    rts = (vt.shift(1) * rets).sum(axis=1)
    return rts.mean() / rts.std() * np.sqrt(252)

print("Standalone rule Sharpe ratios:")
for rule in rules:
    sharpe = calc_rule_sharpe(rule)
    print(f"  {rule}: {sharpe:.3f}")
    # Plot forecast series
    for t in tickers:
        plt.figure(figsize=(8, 3))
        plt.plot(forecasts[(rule, t)], label=f"{rule}-{t}")
        plt.title(f"Forecast Series for {rule}, {t}")
        plt.legend(); plt.grid(True)
        plt.show()

# --- 6. Objective: negative Sharpe for combined strategy ---
def neg_sharpe(weights):
    pos = combine_forecasts(weights).apply(vol_targeted)
    strat_ret = (pos.shift(1) * rets).sum(axis=1)
    ann_sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252)
    return -ann_sharpe

# --- 7. Optimize weights ---
x0   = np.repeat(1/len(rules), len(rules))
bnds = [(0, 1)] * len(rules)
cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
# enforce at least 20% per rule if desired:
# cons += [{'type':'ineq','fun': lambda w, i=i: w[i] - 0.2} for i in range(len(rules))]

res = minimize(neg_sharpe, x0, bounds=bnds, constraints=cons, method='SLSQP')
opt_w = res.x
print("Optimized Weights:", {r: float(w) for r, w in zip(rules, opt_w)})

# --- 8. Backtest & plot equity curve ---
final_pos = combine_forecasts(opt_w).apply(vol_targeted)
pnl       = (final_pos.shift(1) * rets).sum(axis=1)
eq_curve  = (1 + pnl).cumprod()

plt.figure(figsize=(10, 5))
plt.plot(eq_curve, label='Optimized Strategy Equity')
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend(); plt.grid(True)
plt.show()

for t in tickers:
    plt.figure(figsize=(8, 3))
    plt.plot(final_pos[t], label=f"{t}")
    plt.title(f"Position Series {t}")
    plt.legend(); plt.grid(True)
    plt.show()

