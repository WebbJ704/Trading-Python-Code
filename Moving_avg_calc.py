import numpy as np 
import pandas as pd 

def moving_averages(data,short_window,long_window,tickers):
    for ticker in tickers: 
        # Calculate the short and long moving averages
        data[f'{ticker} SMA_Short'] = data[f'{ticker} Open'].rolling(window=short_window).mean()
        data[f'{ticker} SMA_Long'] = data[f'{ticker} Open'].rolling(window=long_window).mean()
        # Initialize Signal column to 0 (no action)
        signals = []
        last_signal = None

        for i in range(len(data[f'{ticker} Open'])):
            if i > long_window :
                if data.iloc[i,data.columns.get_loc(f'{ticker} SMA_Short')] > data.iloc[i,data.columns.get_loc(f'{ticker} SMA_Long')] and (last_signal != 'buy'):
                    signals.append('buy')
                    last_signal = 'buy'
                elif data.iloc[i,data.columns.get_loc(f'{ticker} SMA_Short')] < data.iloc[i,data.columns.get_loc(f'{ticker} SMA_Long')] and (last_signal != 'sell'):
                    signals.append('sell')
                    last_signal ='sell'
                else:
                 signals.append(None) 
            else:
                signals.append(None) 
        
        data[f'{ticker} MA Signals'] = signals
        data[f'{ticker} MA buy signals'] = data[f'{ticker} MA Signals'] == 'buy'
        data[f'{ticker} MA sell signals'] = data[f'{ticker} MA Signals'] == 'sell'
    
    return data

def calculate_cumulative_return(data,tickers):
    for ticker in tickers:
        data[f'{ticker} daily_return'] = data[f'{ticker} Open'].pct_change() # Calculate daily returns (percentage change)
        data[f'{ticker} MA position'] = 0 # Initialize a column to track positions (1 for holding, 0 for cash)

        MA_last_position = 0  # Initially, assume we're not in the market
        for i in range(len(data[f'{ticker} daily_return'])):
            if data[f'{ticker} MA buy signals'].iloc[i]: # Buy signal
                MA_last_position = 1  # Enter the market
            elif data[f'{ticker} MA sell signals'].iloc[i]:  # Sell signal
                MA_last_position = 0  # Exit the market
            
            data.iloc[i, data.columns.get_loc(f'{ticker} MA position')] = MA_last_position # Set the current position
        
        # Shift position to match the date of action
        data[f'{ticker} MA position'] = data[f'{ticker} MA position']#.shift(1, fill_value=0)

        # Calculate the strategy's daily return as the daily return * position
        data[f'{ticker} MA strategy_return'] = data[f'{ticker} daily_return'] * data[f'{ticker} MA position']

        # Calculate the cumulative return of the strategy (multiply daily returns)
        data[f'{ticker} MA cumulative_return'] = (1 + data[f'{ticker} MA strategy_return']).cumprod() 
        data[f'{ticker} Cumulative Buy and Hold Return'] = (1 + data[f'{ticker} daily_return']).cumprod()

    return data
