import pandas as pd
import numpy as np
import requests
import yfinance as yf
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from bs4 import BeautifulSoup
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator

# ----------------- CONFIG -----------------
ALPHA_VANTAGE_API_KEY = 'SLNEQXVO3S7L9JTH'  # Replace with your real key
SYMBOL = 'AMD'
INTERVAL = '5min'
# ------------------------------------------

def fetch_alpha_vantage(ticker):
    filename = f'{ticker}_{INTERVAL}_stock_data.CSV'
    
    if os.path.exists(filename):
        print(f"{filename} already exists. Loading from disk...")
        df = pd.read_csv(filename, index_col=0)
        df = normalize_dataframe(df)
        return df
    else:
        print(f"Downloading data for {ticker}...")
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
        #data, meta_data = ts.get_intraday(symbol=ticker, interval=INTERVAL, outputsize="full")
        data, meta_data = ts.get_daily(symbol=ticker, outputsize="full")
        data.columns = ["Open", "High", "Low", "Close", "Volume"]
        data = data.reset_index()
        data["date"] = pd.to_datetime(data["date"])
        df = data.set_index("date")
        df = normalize_dataframe(df)
        df.to_csv(filename)
        return df

def normalize_dataframe(df):

    df = df.copy()

    # Fix index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Sort by time ascending
    df = df.sort_index()

    # Convert all numeric columns (handle str on first load)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def calculate_indicators(df):
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low']) / 2).cumsum() / df['Volume'].cumsum()
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
    df['Volume_Spike'] = df['Volume'] > (df['Volume'].rolling(window=20).mean() * 1.5)

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def generate_signals(df):
    df['Signal'] = 0
    for i in range(1, len(df)):
        if (
            df['EMA_9'].iloc[i] > df['EMA_21'].iloc[i] and
            df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and
            df['RSI'].iloc[i] > 70 #and
            #df['Close'].iloc[i] > df['VWAP'].iloc[i] and
            #df['Volume_Spike'].iloc[i] #and
            #df['ATR'].iloc[i] > df['ATR'].mean()
        ):
            df.loc[df.index[i], 'Signal'] = 1
    return df

def backtest(df):
    trades = []
    position = 0
    entry_price = 0
    entry_date = None

    for i in range(1, len(df)):
        if df['Signal'].iloc[i] == 1 and position == 0:
            position = 1
            entry_price = df['Close'].iloc[i]
            entry_date = df.index[i]
        elif position == 1:
            # Take Profit or Stop Loss exit
            if df['Close'].iloc[i] >= entry_price * 1.01 or df['Close'].iloc[i] <= entry_price * 0.985:
                exit_date = df.index[i]
                ret = (df['Close'].iloc[i] - entry_price) / entry_price
                trades.append({'EntryDate': entry_date, 'ExitDate': exit_date, 'Return': ret})
                position = 0
                entry_date = None

    # Return trades DataFrame indexed by ExitDate (sell date)
    return pd.DataFrame(trades).set_index('ExitDate')


def monte_carlo_bootstrap(trade_returns, n_simulations=1000, horizon=50):
    simulations = []
    for _ in range(n_simulations):
        sample = np.random.choice(trade_returns, size=horizon, replace=True)
        cumulative_return = np.prod(1 + sample)
        simulations.append(cumulative_return)
    return simulations


def plot_equity(returns):
    returns['Equity'] = (1 + returns['Return']).cumprod()
    returns['Equity'].plot(title='Cumulative Return', figsize=(10, 5))
    plt.grid()
    plt.show()

def get_vix_level():
    vix = yf.Ticker("^VIX")
    data = vix.history(period="1d", interval="1m")
    return data['Close'].iloc[-1] if not data.empty else 20

def get_fear_and_greed_index():
    url = "https://edition.cnn.com/markets/fear-and-greed"
    try:
        r = requests.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        # Find all elements that contain "Fear & Greed Index" text
        elems = soup.find_all(text=lambda t: "Fear & Greed Index" in t)
        for elem in elems:
            parent = elem.parent
            # Try to find next sibling or child containing the number
            next_elem = parent.find_next(string=lambda t: t.strip().isdigit())
            if next_elem:
                return int(next_elem.strip())

        return 61
    except Exception:
        return 61

def fear_filter(vix, fear_index):
    if vix > 30:
        return "BLOCK_ALL"
    elif vix > 25 or fear_index < 20:
        return "BLOCK_LONG"
    return "OK"


def plot_signals(df):
    trade_signals = df[df['Signal'] == 1]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot close price and trade signals
    ax.plot(df.index, df['Close'], label='Close Price', color='gray')
    ax.scatter(trade_signals.index, trade_signals['Close'], color='green', label='Signal', zorder=5)

    # Formatting
    ax.set_title("Trade Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))  # Set number of x-ticks

    plt.tight_layout()
    plt.show()

# ============ MAIN RUN =================
if __name__ == "__main__":
    print("Fetching Alpha Vantage minute data...")
    df = fetch_alpha_vantage(SYMBOL)
    df = calculate_indicators(df)
    df = generate_signals(df)

    print("Fetching VIX...")
    vix = get_vix_level()
    print(f"🔹 VIX: {vix:.2f}")

    print("Fetching CNN Fear & Greed Index...")
    fear_index = get_fear_and_greed_index()
    print(f"Fear & Greed Index: {fear_index}")

    status = fear_filter(vix, fear_index)
    print(f"Fear Filter Status: {status}")

    if status == "BLOCK_ALL":
        print("All trades blocked due to extreme market fear.")
    else:
        print("Running backtest...")
        returns = backtest(df)
        print(f"Completed {len(returns)} trades")
        plot_equity(returns)

    returns.to_csv('test')

    returns_series = returns['Return'].values
    simulated = monte_carlo_bootstrap(returns_series)
    sns.histplot(simulated, bins=50, kde=True)
    plt.title("Bootstrapped Return Distribution")
    plt.xlabel("Final Equity Multiplier")
    plt.show()

    plot_signals(df)




