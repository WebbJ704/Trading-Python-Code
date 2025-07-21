def generate_signals(df, **kwargs):
    # Define indicator toggles with default values
    use_stochastic = kwargs.get("use_stochastic", False)
    use_ADX = kwargs.get("use_ADX", False)
    use_RSI = kwargs.get("use_RSI", False)
    use_RSI_trend = kwargs.get("use_RSI_trend", False)
    use_EMA = kwargs.get("use_EMA", False)
    use_MACD = kwargs.get("use_MACD", False)
    use_ATR = kwargs.get("use_ATR", False)
    use_volume_spike = kwargs.get("use_volume_spike", False)
    use_BB = kwargs.get("use_BB", False)
    use_buy_sell = kwargs.get("use_buy_sell", False)

    # Thresholds and settings
    adx_threshold = kwargs.get("adx_threshold", 25)
    rsi_trend = kwargs.get("rsi_trend", 70)
    rsi_low = kwargs.get("rsi_low", 30)
    lower_k_threshold = kwargs.get("lower_k_threshold", 20)
    atr_mean = df['ATR'].mean()
    buy_factor = kwargs.get("buy_factor",0.95)

    df['Signal'] = 0

    for i in range(1, len(df)):
        conditions = []

        # Each block is optional based on kwargs
        if use_ADX:
            conditions.append(df['ADX'].iloc[i] > adx_threshold)

        if use_RSI_trend:
            conditions.append(df['RSI'].iloc[i] >= rsi_trend)

        if use_RSI:
            conditions.append(df['RSI'].iloc[i] <= rsi_low)

        if use_EMA:
            conditions.append(df['EMA_short'].iloc[i] > df['EMA_long'].iloc[i])

        if use_MACD:
            conditions.append(df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i])

        if use_ATR:
            conditions.append(df['ATR'].iloc[i] > atr_mean)

        if use_volume_spike:
            conditions.append(df['Volume_Spike'].iloc[i])

        if use_stochastic:
            conditions.append(
                df['%K'].iloc[i] > df['%D'].iloc[i] and
                df['%K'].iloc[i] < lower_k_threshold and
                df['%D'].iloc[i] < lower_k_threshold
            )

        if use_BB:
            conditions.append(df['Close'].iloc[i] < df['BB_Lower'].iloc[i] and 
                              df['Close'].iloc[i] > df['BB_Middle'].iloc[i])

        if use_buy_sell:
            conditions.append(df['Close'].iloc[i] < df['Close'].iloc[i-1]*buy_factor)

        # Only assign signal if ALL enabled conditions are met
        if any(conditions):
            df.loc[df.index[i], 'Signal'] = 1

    return df
