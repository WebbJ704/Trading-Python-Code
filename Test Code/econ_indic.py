import fredpy as fp
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
# Initialize FRED
fp.api_key = 'de118a94573cb3e139ad3fc2f0987e05'

# Define today's date and lookback window
today = dt.date.today()
lookback = today - dt.timedelta(days=730)
win = [lookback, today]

# Load indicators
sp500 = fp.series('SP500').window(win)#.as_frequency(freq='M')
unrate = fp.series('UNRATE').window(win)#.as_frequency(freq='M')
t10y2y = fp.series('T10Y2Y').window(win)#.as_frequency(freq='M')
sentiment = fp.series('UMCSENT').window(win)#.as_frequency(freq='M')
indprod = fp.series('INDPRO').window(win)#.as_frequency(freq='M')
stress = fp.series('STLFSI4').window(win)#.as_frequency(freq='M')
m2 = fp.series('M2SL').window(win)#.as_frequency(freq='M')

print(sp500.frequency)
print(unrate.frequency)
print(t10y2y.frequency)
print(sentiment.frequency)
print(indprod.frequency)
print(stress.frequency)
print(m2.frequency)


# Show recent data (optional)
print("S&P 500:\n", sp500.data.tail(), "\n")
print("Unemployment Rate (UNRATE):\n", unrate.data.tail(), "\n")
print("10Y-2Y Yield Spread (T10Y2Y):\n", t10y2y.data.tail(), "\n")
print("Consumer Sentiment (UMCSENT):\n", sentiment.data.tail(), "\n")
print("Industrial Production (INDPRO):\n", indprod.data.tail(), "\n")
print("Financial Stress Index (STLFSI4):\n", stress.data.tail(), "\n")
print("Money Supply (M2SL):\n", m2.data.tail(), "\n")

# plot data 
fig, axs = plt.subplots(7, figsize=(15, 12), sharex=True)

# Plot UNRATE
axs[0].plot(unrate.data, color='blue', label='UNRATE', lw=2, alpha=0.75)
axs[0].set_ylabel('Unemployment Rate')
axs[0].legend()
axs[0].grid()

# Plot S&P 500
axs[1].plot(sp500.data, color='green', label='S&P 500', lw=2, alpha=0.75)
axs[1].set_ylabel('S&P 500')
axs[1].legend()
axs[1].grid()

# Plot 10Y - 2Y Yield Spread
axs[2].plot(t10y2y.data, color='red', label='10Y-2Y Yield Spread', lw=2, alpha=0.75)
axs[2].set_ylabel('10Y-2Y Spread')
axs[2].legend()
axs[2].grid()

# Plot Consumer Sentiment
axs[3].plot(sentiment.data, color='purple', label='Consumer Sentiment', lw=2, alpha=0.75)
axs[3].set_ylabel('UM Sentiment')
axs[3].legend()
axs[3].grid()

# Plot Industrial Production
axs[4].plot(indprod.data, color='orange', label='Industrial Production', lw=2, alpha=0.75)
axs[4].set_ylabel('Industrial Production')
axs[4].legend()
axs[4].grid()

# Plot Financial Stress Index
axs[5].plot(stress.data, color='brown', label='Financial Stress Index', lw=2, alpha=0.75)
axs[5].set_ylabel('Stress Index')
axs[5].legend()
axs[5].grid()

# Plot Money Supply (M2SL)
axs[6].plot(m2.data, color='black', label='M2SL', lw=2, alpha=0.75)
axs[6].set_ylabel('Money Supply (M2SL)')
axs[6].legend()
axs[6].grid()


plt.tight_layout()
plt.show()


# Signal calculations
signals = {}

signals['Yield Curve'] = -1 if t10y2y.data.iloc[-1] < 0 else 1
signals['Unemployment'] = -1 if unrate.data.diff().rolling(3).mean().iloc[-1] > 0 else 1
signals['Sentiment'] = -1 if sentiment.data.pct_change().rolling(3).mean().iloc[-1] < 0 else 1
signals['Industrial Production'] = -1 if indprod.data.pct_change().rolling(3).mean().iloc[-1] < 0 else 1
signals['Stress Index'] = -1 if stress.data.iloc[-1] > 0 else 1
signals['Money Supply'] = -1 if m2.data.pct_change().rolling(3).mean().iloc[-1] < 0 else 1
signals['SP500 Trend'] = -1 if sp500.data.pct_change(fill_method=None).rolling(50).mean().iloc[-1] < 0 else 1

# Final outlook
score = sum(signals.values())
outlook = (
    "High Risk of Market Downturn" if score <= 2 else
    "Moderate Risk / Sideways Market" if 3 <= score <= 4 else
    "Healthy Market Outlook"
)

# Explanations for each signal
descriptions = {
    'Yield Curve': {
        1: "Yield curve is positive (no recession risk) — indicating a healthy economic outlook.",
        -1: "Yield curve inversion (negative = recession risk) — signaling potential economic downturn."
    },
    'Unemployment': {
        1: "Unemployment is stable or falling — generally a good sign for economic strength.",
        -1: "Unemployment trend is rising — signaling potential economic weakness."
    },
    'Sentiment': {
        1: "Consumer sentiment is rising or stable — positive outlook from the public on the economy.",
        -1: "Consumer sentiment is falling — suggesting growing pessimism about the economy."
    },
    'Industrial Production': {
        1: "Industrial production is stable or rising — indicating economic expansion.",
        -1: "Industrial production is declining — suggesting an economic slowdown."
    },
    'Stress Index': {
        1: "Financial stress is low — favorable market conditions with lower systemic risk.",
        -1: "Financial stress is rising — signaling potential financial risks and instability."
    },
    'Money Supply': {
        1: "Money supply is increasing — indicating loose monetary policy and liquidity in the market.",
        -1: "Money supply is contracting — signaling tightening financial conditions."
    },
    'SP500 Trend': {
        1: "Stock market is trending upward — indicating investor confidence and a healthy market.",
        -1: "Stock market is trending downward — signaling bearish sentiment and potential market weakness."
    }
}

# Output with descriptions
print("\nEconomic Market Risk Assessor")
print("----------------------------------")
for key, val in signals.items():
    state = 'Positive' if val == 1 else 'Negative'
    print(f"{key}: {state} — {descriptions[key][val]}")

print(f"\nFinal Market Outlook: {outlook}")
