import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries

# Your Alpha Vantage API Key
API_KEY = 'SLNEQXVO3S7L9JTH'

import os

def download_data(ticker, start_date, end_date, cache_file='stock_data.csv'):
    # Check if cached file exists
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        print(f"Downloading data for {ticker}")
        ts = TimeSeries(key=API_KEY, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')  # Use get_daily instead of get_daily_adjusted
        data = data[['4. close', '2. high', '3. low']]  # We want 'close', 'high', and 'low' prices
        data = data.rename(columns={'4. close': 'Close', '2. high': 'High', '3. low': 'Low'})
        data = data[::-1]  # Reverse the data to have it in chronological order
        data = data[(data.index >= start_date) & (data.index <= end_date)]  # Filter data by the specified date range
        
        # Save the data to a file to avoid hitting the API again
        data.to_csv(cache_file)
        print(f"Data saved to {cache_file}")
    
    return data


# Calculate SMA Crossover strategy
def calculate_signals(data, short_window=10, long_window=100):
    # Calculate the short and long moving averages
    data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()

    # Initialize Signal column to 0 (no action)
    data['Signal'] = 0

    # Apply the trading signals: 1 for Buy, -1 for Sell, 0 for Hold
    for i in range(1, len(data)):
        if data['SMA_Short'][i] > data['SMA_Long'][i] and data['SMA_Short'][i-1] <= data['SMA_Long'][i-1]:
            data['Signal'][i] = 1  # Buy signal
        elif data['SMA_Short'][i] < data['SMA_Long'][i] and data['SMA_Short'][i-1] >= data['SMA_Long'][i-1]:
            data['Signal'][i] = -1  # Sell signal
        else:
            data['Signal'][i] = 0  # Hold position if no crossover

    return data

# Calculate ATR (Average True Range) for volatility-based filtering
def calculate_atr(data, window=14):
    # Calculate True Range (TR)
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
    data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
    data['True Range'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    # Calculate ATR as the rolling average of True Range
    data['ATR'] = data['True Range'].rolling(window=window).mean()
    return data

# Backtest the strategy with volatility filter
def backtest(data, volatility_threshold=2):
    # Shift signal to avoid lookahead bias
    data['Position'] = data['Signal'].shift(1)

    # Calculate daily returns
    data['Daily Return'] = data['Close'].pct_change()

    # Filter based on volatility: Avoid trades if ATR is too high (indicating high volatility)
    data['Volatility Filtered Return'] = np.where(data['ATR'] > volatility_threshold, 0, data['Daily Return'])

    # Calculate strategy returns based on positions and volatility filter
    data['Strategy Return'] = data['Volatility Filtered Return'] * data['Position']
    data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()
    data['Cumulative Buy and Hold Return'] = (1 + data['Daily Return']).cumprod()

    return data

# Plot the results
def plot_results(data):
    # Plot stock price
    plt.figure(figsize=(14,7))
    plt.plot(data['Close'], label='Stock Price', color='blue', alpha=0.6)
    plt.plot(data['SMA_Short'], label='50-Day SMA (Short)', color='orange', linestyle='--')
    plt.plot(data['SMA_Long'], label='200-Day SMA (Long)', color='green', linestyle='--')

    # Plot Buy and Sell signals using scatter
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]
    
    # Scatter plot for buy signals (green)
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', alpha=1)
    
    # Scatter plot for sell signals (red)
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', alpha=1)

    plt.title('Stock Price with Buy/Sell Signals and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot cumulative returns
    plt.figure(figsize=(14,7))
    plt.plot(data['Cumulative Strategy Return'], label='Cumulative Strategy Return')
    plt.plot(data['Cumulative Buy and Hold Return'], label='Cumulative Buy and Hold Return')
    plt.title('Cumulative Returns: Strategy vs Buy and Hold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    plt.show()

# Main function
def main():
    # Define stock data parameters
    ticker = 'NVDA'  # Example: Nvidia stock
    start_date = '2015-01-01'
    end_date = '2020-01-01'

    # Download data
    data = download_data(ticker, start_date, end_date)

    # Calculate signals
    data = calculate_signals(data)

    # Calculate ATR for volatility filter
    data = calculate_atr(data)

    # Backtest the strategy with volatility filter
    data = backtest(data)

    # Plot the results
    plot_results(data)

    # Display the final cumulative return data (returns as DataFrame)
    print(data[['Cumulative Strategy Return', 'Cumulative Buy and Hold Return']])

if __name__ == '__main__':
    main()
