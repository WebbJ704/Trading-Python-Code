def Rules(df):
    EMA_short = 12
    EMA_long = 26
    rsi_window = 14
    EMA_signal = 9
    k_period= 14
    d_period=3

    for i in range(len(EMA_short)):
        #Exponential moving average crossover
        df['EMA_short'] = df['Close'].ewm(span=EMA_short[i], adjust=False).mean()
        df['EMA_long'] = df['Close'].ewm(span=EMA_long[i], adjust=False).mean()

        #moving average crossover divergence 
        df['MACD'] = df['EMA_short'] - df['EMA_long']
        df['MACD_Signal'] = df['MACD'].ewm(span=EMA_signal[i], adjust=False).mean()

        # # RSI (Relative Strength Index)
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

    return df 
