import numpy as np 
import pandas as pd 

def RSI(data,rolling_window,RSI_overbought,RSI_underbought,tickers):
  for ticker in tickers :
    data[f'{ticker} daily price change'] = data[f'{ticker} Open'].diff()

    data[f'{ticker} gain'] = np.where(data[f'{ticker} daily price change'] > 0, data[f'{ticker} daily price change'], 0)
    data[f'{ticker} loss'] = np.where(data[f'{ticker} daily price change'] < 0, -data[f'{ticker} daily price change'], 0)

    data[f'{ticker} avg_gain'] = data[f'{ticker} gain'].rolling(window=rolling_window, min_periods=1).mean()
    data[f'{ticker} avg_loss'] = data[f'{ticker} loss'].rolling(window=rolling_window, min_periods=1).mean()

    data[f'{ticker} RS'] = data[f'{ticker} avg_gain'] /  data[f'{ticker} avg_loss']
    data[f'{ticker} RSI'] = 100 - (100 / (1 + data[f'{ticker} RS']))
    data[f'{ticker} RSI'] = (data[f'{ticker} RSI'].replace(0,pd.NA)).dropna()
  
  
    last_signal = None
    signals = []
    for i in range(len(data[f'{ticker} RSI'])):
      if i >= rolling_window:
        if data.iloc[i, data.columns.get_loc(f'{ticker} RSI')] >= RSI_overbought and (last_signal != 'buy'):
          signals.append('buy')
          last_signal = 'buy'
        elif data.iloc[i, data.columns.get_loc(f'{ticker} RSI')] <= RSI_underbought and (last_signal != 'sell'):
          signals.append('sell')
          last_signal ='sell'
        else:
          signals.append(None)
      else:
          signals.append(None)
      
    if len(signals) < len(data):
      signals.append(None)

    data[f'{ticker} RSI signal'] = signals
    data[f'{ticker} RSI buy signals'] = data[f'{ticker} RSI signal'] == 'buy'
    data[f'{ticker} RSI sell signals'] = data[f'{ticker} RSI signal'] == 'sell'
  return data


def RSI_cumulative_return(data,tickers):
  
  for ticker in tickers:

    data[f'{ticker} daily_return'] = data[f'{ticker} Open'].pct_change() # Calculate daily returns (percentage change
    data[f'{ticker} RSI position'] = 0 # Initialize a column to track positions (1 for holding, 0 for cash)

    RSI_last_position = 0  # Initially, assume we're not in the market
    for i in range(len(data[f'{ticker} daily_return'])):
      if data[f'{ticker} RSI buy signals'].iloc[i]: # Buy signal
        RSI_last_position = 1  # Enter the market
      elif data[f'{ticker} RSI sell signals'].iloc[i]:  # Sell signal
        RSI_last_position = 0  # Exit the market
      
      data.iloc[i, data.columns.get_loc(f'{ticker} RSI position')] = RSI_last_position # Set the current position

    # Shift position to match the date of action
    data[f'{ticker} RSI position'] = data[f'{ticker} RSI position']#.shift(1, fill_value=0)

    # Calculate the strategy's daily return as the daily return * position
    data[f'{ticker} RSI strategy_return'] = data[f'{ticker} daily_return'] * data[f'{ticker} RSI position']

    # Calculate the cumulative return of the strategy (multiply daily returns)
    data[f'{ticker} RSI cumulative_return'] = (1 + data[f'{ticker} RSI strategy_return']).cumprod() 
    data[f'{ticker} Cumulative Buy and Hold Return'] = (1 + data[f'{ticker} daily_return']).cumprod() # cumulative return of buy and hold startegy 
      
  return data