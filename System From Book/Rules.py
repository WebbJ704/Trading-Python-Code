import numpy as np
import pandas as pd

def Rules(df):
    EMA_short = 12
    EMA_long = 26
    rsi_window = 14
    EMA_signal = 9
    k_period= 14
    d_period=3
    Vol_window = 20
    atr_window = 14
    adx_window = atr_window

    for i in range(len(EMA_short)):
        # Exponential moving average crossover
        df['EMA_short'] = df['Close'].ewm(span=EMA_short[i], adjust=False).mean()
        df['EMA_long'] = df['Close'].ewm(span=EMA_long[i], adjust=False).mean()

        # moving average crossover divergence 
        df['MACD'] = df['EMA_short'] - df['EMA_long']
        df['MACD_Signal'] = df['MACD'].ewm(span=EMA_signal[i], adjust=False).mean()

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=rsi_window[i]).mean()
        avg_loss = loss.rolling(window=rsi_window[i]).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # stochiastic oscilator 
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        df['%K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
        df['%D'] = df['%K'].rolling(window=d_period).mean()

        # ADX (Average Directional Index)
        high_diff = df['High'].diff()
        low_diff = df['Low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
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

    return df 
