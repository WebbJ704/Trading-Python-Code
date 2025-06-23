import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the stock ticker and the date range
ticker = 'AAPL'  # Stock symbol (e.g., Apple)
start_date = '2010-01-01'
end_date = '2022-12-31'

# Download stock data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Calculate short-term and long-term moving averages
short_window =  10 # 50 days
long_window = 400 # 200 days

data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

# Create Buy and Sell signals based on moving average crossover
data['Signal'] = 0  # Default signal: no action

# Buy signal when Short_MA crosses above Long_MA
data['Signal'] = np.where(data['Short_MA'] > data['Long_MA'], 1, 0)

# Sell signal when Short_MA crosses below Long_MA (i.e., when Signal switches from 1 to 0)
data['Position'] = data['Signal'].diff()  # Signal change (1 = Buy, -1 = Sell)

# Calculate daily returns
data['Daily_Return'] = data['Close'].pct_change()

# Strategy returns (based on buy/sell signals)
data['Strategy_Return'] = data['Daily_Return'] * data['Signal'].shift(1)  # Use previous signal for current day

# Calculate cumulative returns for both strategies
data['Buy_Hold_Return'] = (1 + data['Daily_Return']).cumprod()  # Buy and hold strategy
data['Strategy_Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()  # Strategy cumulative return

# Plotting the results
plt.figure(figsize=(14, 7))

# Plot 1: Stock price, moving averages, and buy/sell signals
plt.subplot(2, 1, 1)
plt.plot(data['Close'], label='Close Price', color='black', alpha=0.5)
plt.plot(data['Short_MA'], label=f'{short_window}-Day Moving Average', color='blue', alpha=0.7)
plt.plot(data['Long_MA'], label=f'{long_window}-Day Moving Average', color='red', alpha=0.7)

# Scatter the buy signals (green arrows) and sell signals (red arrows)
plt.scatter(data.index[data['Position'] == 1], data['Short_MA'][data['Position'] == 1], marker='^', color='g', label='Buy Signal', alpha=1)
plt.scatter(data.index[data['Position'] == -1], data['Short_MA'][data['Position'] == -1], marker='v', color='r', label='Sell Signal', alpha=1)

plt.title(f'{ticker} Price and Moving Averages with Buy/Sell Signals')
plt.legend(loc='best')

# Plot 2: Cumulative returns for Buy and Hold vs Strategy
plt.subplot(2, 1, 2)
plt.plot(data['Buy_Hold_Return'], label='Buy and Hold Return', color='blue')
plt.plot(data['Strategy_Cumulative_Return'], label='Strategy Return', color='orange')
plt.title(f'{ticker} Cumulative Returns: Buy and Hold vs Strategy')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

print(data['Signal'])