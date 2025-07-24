import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import seaborn as sns
import os
from scipy.stats import skew
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator

def data_download_pl(ticker,api,start_date,end_date):

   
    filename= f'{ticker}_min_data.CSV'

    # Check if file exists
    if os.path.exists(filename):
        print(f" {filename} already exists. Skipping download.")
        df = pd.read_csv(filename)
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        #df.set_index("date", inplace=True)
        return df
    else :
        print(f"Downloading data for {ticker}...")
        ts = TimeSeries(key=api, output_format="pandas")
        data, meta_data = ts.get_intraday(symbol=ticker, outputsize="full")
        data.columns = ["Open", "High", "Low", "Close", "Volume"] # Rename columns for easier use
        data = data.reset_index()   # Reset index and format date column

        data["date"] = pd.to_datetime(data["date"])
        #data.set_index('date', inplace=True)
        #df = data.sort_values("date")  # Sort data in ascending order
        data.to_csv(filename) # saving to a csv file 
        return data
    

def moving_averages(data,short_window,long_window):
    # Calculate the short and long moving averages
    data['SMA_Short'] = data['Open'].rolling(window=short_window).mean()
    data['SMA_Long'] = data['Open'].rolling(window=long_window).mean()
    # Initialize Signal column to 0 (no action)
    signals = []
    last_signal = None

    for i in range(len(data['Open'])):
       if i > long_window :
        if data.iloc[i,data.columns.get_loc('SMA_Short')] > data.iloc[i,data.columns.get_loc('SMA_Long')] and (last_signal != 'buy'):
            signals.append('buy')
            last_signal = 'buy'
        elif data.iloc[i,data.columns.get_loc('SMA_Short')] < data.iloc[i,data.columns.get_loc('SMA_Long')] and (last_signal != 'sell'):
            signals.append('sell')
            last_signal ='sell'
        else:
           signals.append(None) 
       else:
           signals.append(None) 
    
    data['MA Signals'] = signals
    data['MA buy signals'] = data['MA Signals'] == 'buy'
    data['MA sell signals'] = data['MA Signals'] == 'sell'
    
    return data
    


def calculate_cumulative_return(data):
    data['daily_return'] = data['Open'].pct_change() # Calculate daily returns (percentage change)
    data['MA position'] = 0 # Initialize a column to track positions (1 for holding, 0 for cash)

    MA_last_position = 0  # Initially, assume we're not in the market
    for i in range(len(data['daily_return'])):
        if data['MA buy signals'].iloc[i]: # Buy signal
            MA_last_position = 1  # Enter the market
        elif data['MA sell signals'].iloc[i]:  # Sell signal
           MA_last_position = 0  # Exit the market
        
        data.iloc[i, data.columns.get_loc('MA position')] = MA_last_position # Set the current position
    
    # Shift position to match the date of action  
    #data['MA position'] = data['MA position'].shift(1, fill_value=0)

    # Calculate the strategy's daily return as the daily return * position
    data['MA strategy_return'] = data['daily_return'] * data['MA position']
    # Calculate the cumulative return of the strategy (multiply daily returns)
    data['MA cumulative_return'] = (1 + data['MA strategy_return']).cumprod() 
    data['Cumulative Buy and Hold Return'] = (1 + data['daily_return']).cumprod() 

    return data

def plotting(data,short_window,long_window):
    
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax[0].plot(data['SMA_Short'], label = f'{short_window} moving average' )
    ax[0].plot(data['SMA_Long'], label = f'{long_window} moving average')
    ax[1].scatter(data.index[data['MA buy signals']],data['Open_rolling'][data['MA buy signals']], marker='^', color='green', label='Buy Signal', zorder=5)
    ax[1].scatter(data.index[data['MA sell signals']],data['Open_rolling'][data['MA sell signals']], marker='v', color='red', label='Sell Signal', zorder=5)
    ax[1].plot(data['Open_rolling'], label = 'Open price' )
    ax[0].set_ylabel('Moving Averages')
    ax[1].set_ylabel('stock Price')
    plt.xticks(rotation=90)
    current_xlim = plt.gca().get_xlim()
    num_ticks = (current_xlim[1] - current_xlim[0]) // 100  # Adjust x-axis tick locations dynamically based on the zoom level. Adjust the denominator as needed
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
    plt.xticks(rotation=15, fontsize=10)# Rotate and reduce the font size for better readability
    plt.title(f'Moving avereages')
    plt.legend()
    plt.xlabel('date')
    plt.show()
  
    plt.figure()
    plt.plot(data['date'],data['MA cumulative_return'], label = ' Moving avereages')
    plt.plot(data['date'],data['Cumulative Buy and Hold Return'], label = 'buy/hold')
    num_ticks = (current_xlim[1] - current_xlim[0]) // 100 
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
    plt.xticks(rotation=15, fontsize=10)
    plt.legend(loc='upper left')
    plt.show()

def returns(data,rolling_window):
    data['Returns'] = data['Open'].pct_change(periods = rolling_window)# percentage return
    data['Returns'] = data['Returns']*100
    data['Open_rolling'] =  data['Open'].rolling(window = rolling_window).mean()
    return data

def main():
    ticker = 'GLD'
    api = ''
    start_date = "2005-10-01"
    end_date = "2025-03-10"
    short_window = 1
    long_window = 20

    data = data_download_pl(ticker,api,start_date,end_date)

    data = moving_averages(data,short_window,long_window)

    data = calculate_cumulative_return(data)

    data = returns(data,long_window)

    data = plotting(data,short_window,long_window)

if  __name__ == "__main__":
    main()


