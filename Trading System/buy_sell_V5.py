import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import seaborn as sns
import os
import matplotlib.dates as mdates
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime

# Alpaca API setup
API_KEY = ''
SECRET_KEY = ''
BASE_URL = 'https://paper-api.alpaca.markets'
api = REST(API_KEY, SECRET_KEY, BASE_URL)

# ----------------- CONFIG -----------------
ALPHA_VANTAGE_API_KEY = 'SLNEQXVO3S7L9JTH'  
SYMBOL = ['AMD','META','MSFT','IBM']
INTERVAL = 'Day'
# ------------------------------------------

def fetch_alpha_vantage_data(ticker):
    print(f"Downloading data for {ticker}...")
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
    if INTERVAL == 'Day':
        data, meta_data = ts.get_daily(symbol=ticker, outputsize="full")
    else:
        data, meta_data = ts.get_intraday(symbol=ticker, interval=INTERVAL, outputsize="full")
    data.columns = ["Open", "High", "Low", "Close", "Volume"]
    data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"])
    df = data.set_index("date")
    df = normalize_dataframe(df)
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

    # # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI(14)'] = 100 - (100 / (1 + rs))

    # ADX (Average Directional Index)
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    plus_di = 100 * plus_dm.rolling(window=14).sum() / atr
    minus_di = 100 * minus_dm.rolling(window=14).sum() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14).mean()
    df['ADX'] = adx

    return df

def stock_viability(df):
    df['Viabilty'] = False
    for i in range(1,len(df)):
        if df['ADX'].iloc[i] > 25:
            df.loc[df.index[i], 'Viabilty'] = True
        else:
            df.loc[df.index[i], 'Viabilty'] = False
    return df
        
def generate_signals(df):
    df['Signal'] = 0
    for i in range(1,len(df)):
        if (
            (df['ADX'].iloc[i] > 25 and df['RSI(14)'].iloc[i] >= 70) or
            (df['EMA_9'].iloc[i] > df['EMA_21'].iloc[i])
            #df['EMA_9'].iloc[i] > df['EMA_21'].iloc[i] or
            #df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] or
            #df['RSI(14)'].iloc[i] >= 70 and
            #df['ADX'].iloc[i] > 25 #and
            #df['Close'].iloc[i] > df['VWAP'].iloc[i] and
            #df['Volume_Spike'].iloc[i] #and
            #df['ATR'].iloc[i] > df['ATR'].mean()
        ):
            df.loc[df.index[i], 'Signal'] = 1
    return df


def trade(symbol, df):
    # Ensure latest signal is valid
    if 'Signal' not in df.columns or df.empty:
        print(f"No signal data for {symbol}")
        return

    # Latest row (most recent trading day)
    latest_signal = df['Signal'].iloc[-1]
    close_price = df['Close'].iloc[-1]
    today = df.index[-1].strftime('%Y-%m-%d')

    # Check current positions
    try:
        position = api.get_position(symbol)
        has_position = True
        entry_price = float(position.avg_entry_price)
        qty = float(position.qty)
    except:
        has_position = False
        entry_price = 0
        qty = 0

    # --- Entry Logic ---
    if latest_signal == 1 and not has_position:
        try:
            api.submit_order(
                symbol=symbol,
                qty=40,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            print(f"{today}: Buy {symbol} at approx {close_price}")
        except Exception as e:
            print(f"Buy order failed for {symbol}: {e}")

    # --- Exit Logic (Take Profit or Stop Loss) ---
    elif has_position:
        target_price = entry_price * 1.1
        stop_loss_price = entry_price * 0.92

        if close_price >= target_price or close_price <= stop_loss_price:
            try:
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                result = "Profit Target" if close_price >= target_price else "Stop Loss"
                print(f"{today}: Sell {symbol} at approx {close_price} - {result}")
            except Exception as e:
                print(f"Sell order failed for {symbol}: {e}")
        else:
            print(f"{today}: Holding {symbol} (entry: {entry_price}, now: {close_price})")


if __name__ == "__main__":
    for symbol in SYMBOL:
        print("Fetching Alpha Vantage data...")
        df = fetch_alpha_vantage_data(symbol)  
        df = calculate_indicators(df)
        df = stock_viability(df)
        df = generate_signals(df)
        print(df)
        trade(symbol, df)
   
