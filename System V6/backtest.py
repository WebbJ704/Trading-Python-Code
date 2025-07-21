import pandas as pd

def backtest(df, **kwargs):
    trades = []
    Signal = []
    position = 0
    entry_price = 0
    entry_date = None

    # Default exit condition toggles
    use_take_profit_stop_loss = kwargs.get("use_take_profit_stop_loss", False)
    take_profit = kwargs.get("take_profit", 1.1)
    stop_loss = kwargs.get("stop_loss", 0.92)

    use_macd_exit = kwargs.get("use_macd_exit", False)

    use_ADX_exit = kwargs.get('use_ADX_exit', False)
    adx_threshold = kwargs.get("ADX_threshold", 25)

    use_rsi_trend_exit = kwargs.get("use_rsi_trend_exit", False)
    rsi_trend_exit_threshold = kwargs.get("rsi_trend_exit_threshold", 30)
    
    use_rsi_exit = kwargs.get("use_rsi_exit", False)
    rsi_exit_threshold = kwargs.get("rsi_exit_threshold", 70)

    use_ema_exit = kwargs.get("use_ema_exit", False)

    use_stochastic_exit = kwargs.get("use_stochastic_exit", False)
    stochastic_upper = kwargs.get("stochastic_upper", 80)

    use_bb_exit = kwargs.get("use_bb_exit", False)

    use_no_signal = kwargs.get("use_no_signal", False)

    use_buy_sell_exit = kwargs.get("use_buy_sell_exit", False)
    sell_factor = kwargs.get("sell_factor",1.1)

    for i in range(1, len(df)-1):
        if df['Signal'].iloc[i-1] == 1 and position == 0:
            position = 1
            entry_price = df['Open'].iloc[i]
            entry_date = df.index[i]
            Signal.append(f'Buy - {entry_date, entry_price}')

        elif position == 1:
            exit_conditions = []

            if use_no_signal:
                exit_conditions.append(df['Signal'].iloc[i] == 0 )

            if use_take_profit_stop_loss:
                exit_conditions.append(
                    df['Close'].iloc[i] >= entry_price * take_profit or
                    df['Close'].iloc[i] <= entry_price * stop_loss
                )

            if use_macd_exit:
                exit_conditions.append(df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i])

            if use_ADX_exit:
                exit_conditions.append(df['ADX'].iloc[i] > adx_threshold)

            if use_rsi_trend_exit:
                exit_conditions.append(df['RSI'].iloc[i] <= rsi_trend_exit_threshold)

            if use_rsi_exit:
                exit_conditions.append(df['RSI'].iloc[i] >= rsi_exit_threshold)

            if use_ema_exit:
                exit_conditions.append(df['EMA_short'].iloc[i] < df['EMA_long'].iloc[i])

            if use_stochastic_exit:
                exit_conditions.append(
                    df['%K'].iloc[i] < df['%D'].iloc[i] and
                    df['%K'].iloc[i] > stochastic_upper and
                    df['%D'].iloc[i] > stochastic_upper
                )
            if use_bb_exit:
                exit_conditions.append(df['Close'].iloc[i] > df['BB_Upper'].iloc[i])
            
            if use_buy_sell_exit:
                exit_conditions.append(df['Close'].iloc[i] > df['Close'].iloc[i-1]*sell_factor)


            # Exit trade if ALL of the enabled conditions are True
            if any(exit_conditions):
                exit_price = df['Open'].iloc[i+1]
                exit_date = df.index[i+1]
                trade_return = (exit_price - entry_price) / entry_price
                trades.append({
                    'EntryDate': entry_date,
                    'ExitDate': exit_date,
                    'Return': trade_return
                })
                Signal.append(f'Sell - {exit_date, exit_price}')
                position = 0
                entry_date = None

    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.set_index('ExitDate', inplace=True)
        first_entry_date = trades_df['EntryDate'].iloc[0]
        buy_hold_start_price = df.loc[first_entry_date, 'Close']
        trades_df['BuyAndHold'] = df.loc[trades_df.index, 'Close'].values / buy_hold_start_price
        trades_df['StrategyEquity'] = (1 + trades_df['Return']).cumprod()
    else:
        trades_df = pd.DataFrame(columns=['EntryDate', 'ExitDate', 'Return', 'BuyAndHold'])

    return trades_df, Signal