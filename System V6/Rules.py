import numpy as np
import pandas as pd

def Rules(df, **kwargs):
    EMA_short   = kwargs.get("EMA_short", 12)
    EMA_long    = kwargs.get("EMA_long", 26)
    EMA_signal  = kwargs.get("EMA_signal", 9)
    rsi_window  = kwargs.get("rsi_window", 14)
    k_period    = kwargs.get("k_period", 14)
    d_period    = kwargs.get("d_period", 3)
    Vol_window  = kwargs.get("Vol_window", 20)
    atr_window  = kwargs.get("atr_window", 14)
    adx_window  = kwargs.get("adx_window", atr_window)
    BB_window   = kwargs.get("BB_window", 35)
    BB_std_dev_top  = kwargs.get("BB_std_dev_top", 1.5)
    BB_std_dev_bottom  = kwargs.get("BB_std_dev_bottom", 0.5)

    df['EMA_short'] = df['Adj Close'].ewm(span=EMA_short, adjust=False).mean()
    df['EMA_long'] = df['Adj Close'].ewm(span=EMA_long, adjust=False).mean()

    # moving average crossover divergence 
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_Signal'] = df['MACD'].ewm(span=EMA_signal, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = df['Adj Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # stochiastic oscilator 
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    df['%K'] = ((df['Adj Close'] - low_min) / (high_max - low_min)) * 100
    df['%D'] = df['%K'].rolling(window=d_period).mean()

    # ADX (Average Directional Index)
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Adj Close'].shift(1))
    tr3 = abs(df['Low'] - df['Adj Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_window).mean()
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    plus_di = 100 * plus_dm.rolling(window=atr_window).sum() / atr
    minus_di = 100 * minus_dm.rolling(window=atr_window).sum() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=adx_window).mean()
    df['ADX'] = adx

    # VWAP
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low']) / 2).cumsum() / df['Volume'].cumsum()

    # ATR
    df['ATR'] = (df['High'] - df['Low']).rolling(window=atr_window).mean()

    # Voume spike
    df['Volume_Spike'] = df['Volume'] > (df['Volume'].rolling(window=Vol_window).mean() * 1.5)

    #Bollinger bands exponetaly weighted
    df['BB_Middle'] = df['Adj Close'].ewm(span=BB_window, adjust=False).mean()
    df['BB_StdDev'] = df['Adj Close'].ewm(span=BB_window, adjust=False).std()
    df['BB_Upper'] = df['BB_Middle'] + BB_std_dev_top * df['BB_StdDev']
    df['BB_Lower'] = df['BB_Middle'] - BB_std_dev_bottom * df['BB_StdDev']

    if kwargs.get("dropna", False):
        df.dropna(inplace=True)

    return df 
