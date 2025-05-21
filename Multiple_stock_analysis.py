import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time
import matplotlib.pyplot as plt
import numpy as np
import RSI_calc as RSI
from matplotlib.ticker import MaxNLocator
import Moving_avg_calc as MA

def donwload_data(tickers,start_date,end_date,api):
   for ticker in tickers:
    filename= f'{ticker}_stock_data.CSV'
    # Check if file exists
    if os.path.exists(filename):
        print(f" {filename} already exists. Skipping download.")
        df = pd.read_csv(filename)
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        df.set_index("date", inplace=True)
    else :
        print(f"Downloading data for {ticker}...")
        ts = TimeSeries(key=api, output_format="pandas")
        data, meta_data = ts.get_daily(symbol=ticker, outputsize="full")
        data.columns = ["Open", "High", "Low", "Close", "Volume"] # Rename columns for easier use
        data = data.reset_index()   # Reset index and format date column

        data["date"] = pd.to_datetime(data["date"])
        data.set_index('date', inplace=True)
        df = data.sort_values("date")  # Sort data in ascending order
        df.to_csv(f'{ticker}_stock_data.CSV') # saving to a csv file 


def combine_data(tickers,start_date,end_date):
   file = f'{tickers}_combined_data.csv'
   # Check if file exists
   if os.path.exists(file):
      print(f" {file} already exists. Skipping combine.")
      df = pd.read_csv(file, index_col='Date', parse_dates=True)
      return df
   else :
      combined_data = pd.DataFrame
      for ticker in tickers: 
         filename = f'{ticker}_stock_data.CSV'
         df = pd.read_csv(filename, index_col='date', parse_dates=True)
         df.index = pd.to_datetime(df.index)  # Ensure datetime index
         
         # Extract only the date part of the datetime index
         df.index = df.index.date
         
         # Rename the 'Open' column to the stock ticker name
         df = df[['Open']]  # Only keep the 'Close' column for simplicity
         df.rename(columns={'Open': f'{ticker} Open'}, inplace=True)
         # Merge the data
         if combined_data.empty:
            combined_data = df
         else:
            combined_data = combined_data.join(df, how='outer')

      combined_data.index = pd.to_datetime(combined_data.index) # ensures date time is the index
      combined_data.index.name = 'Date'
      combined_data = combined_data.dropna()
      combined_data = combined_data[(combined_data.index >= start_date) & (combined_data.index <= end_date)]
      
      # Save the combined data to a new CSV file
      combined_data.to_csv(f'{tickers}_combined_data.csv')
      print(f"Combined data saved to {tickers}_combined_data.csv")
      return combined_data
   
def SR(data,ticker):

   for tickers in ticker: 
      data[f'{tickers} Mean Anual'] =  (data[f'{tickers} Open'].pct_change(periods = 256)).mean()
      data[f'{tickers} std Anual'] = (data[f'{tickers} Open'].pct_change(periods = 256)).std()

      data[f'{tickers} daily mean'] = data[f'{tickers} Mean Anual']/256
      data[f'{tickers} daily std'] = data[f'{tickers} std Anual']/ np.sqrt(256)

      data[f'{tickers} SR anualised'] =  data[f'{tickers} Mean Anual'] /  data[f'{tickers} std Anual']
      data[f'{tickers} SR daily'] =   data[f'{tickers} daily mean']/ data[f'{tickers} daily std']
      print(data[f'{tickers} SR anualised'])
   return data 
         
def plot(data,ticker,rsi_window,short_window,long_window):
   for tickers in ticker:
      data[f'{tickers} Open'].plot(label=f'{tickers}')
      # Adding labels and title   
      plt.title('Stock Opening Prices over Time')
      plt.xlabel('Date')
      plt.ylabel('Opening Price')
      plt.legend()  # Display the legend
      plt.show()

   for tickers in ticker:
      plt.figure()
      plt.title(f'{tickers}')
      plt.plot(data[f'{tickers} RSI cumulative_return'], label = ' RSI')
      plt.plot(data[f'{tickers} Cumulative Buy and Hold Return'], label = 'buy/hold')
      current_xlim = plt.gca().get_xlim()
      num_ticks = (current_xlim[1] - current_xlim[0]) // 100 
      plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
      plt.xticks(rotation=45, fontsize=10)
      plt.legend(loc='upper left')
      plt.show()

   for tickers in ticker:
      # RSI PLOT
      fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
      ax[0].plot(data[f'{tickers} RSI'])
      ax[1].scatter(data.index[data[f'{tickers} RSI buy signals']],data[f'{tickers} Open_rolling'][data[f'{tickers} RSI buy signals']], marker='^', color='green', label='Buy Signal', zorder=5)
      ax[1].scatter(data.index[data[f'{tickers} RSI sell signals']],data[f'{tickers} Open_rolling'][data[f'{tickers} RSI sell signals']], marker='v', color='red', label='Sell Signal', zorder=5)
      ax[1].plot(data[f'{tickers} Open_rolling'], label = f'{tickers} Open price' )
      ax[0].set_ylabel(f'{tickers} RSI')
      ax[1].set_ylabel(f'{tickers} stock Price')
      plt.xticks(rotation=90)
      current_xlim = plt.gca().get_xlim()
      num_ticks = (current_xlim[1] - current_xlim[0]) // 100  # Adjust x-axis tick locations dynamically based on the zoom level. Adjust the denominator as needed
      plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
      plt.xticks(rotation=45, fontsize=10)# Rotate and reduce the font size for better readability
      plt.title(f'{tickers} RSI over {rsi_window} days')
      plt.xlabel('date')
      plt.show()

   for tickers in ticker :
      plt.figure()
      plt.title(f'{tickers}')
      plt.plot(data[f'{tickers} MA cumulative_return'], label = ' Moving avereages')
      plt.plot(data[f'{tickers} Cumulative Buy and Hold Return'], label = 'buy/hold')
      num_ticks = (current_xlim[1] - current_xlim[0]) // 100 
      plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
      plt.xticks(rotation=45, fontsize=10)
      plt.legend(loc='upper left')
      plt.show()

   for tickers in ticker :
      #Moving avg PLOT 
      fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
      ax[0].plot(data[f'{tickers} SMA_Short'],label = f'{short_window} moving average' )
      ax[0].plot(data[f'{tickers} SMA_Long'], label = f'{long_window} moving average')
      ax[1].scatter(data.index[data[f'{tickers} MA buy signals']],data[f'{tickers} Open_rolling'][data[f'{tickers} MA buy signals']], marker='^', color='green', label='Buy Signal', zorder=5)
      ax[1].scatter(data.index[data[f'{tickers} MA sell signals']],data[f'{tickers} Open_rolling'][data[f'{tickers} MA sell signals']], marker='v', color='red', label='Sell Signal', zorder=5)
      ax[1].plot(data[f'{tickers} Open_rolling'], label = 'Open price' )
      ax[0].set_ylabel(f'{tickers} Moving Averages')
      ax[1].set_ylabel(f'{tickers} stock Price')
      plt.xticks(rotation=90)
      current_xlim = plt.gca().get_xlim()
      num_ticks = (current_xlim[1] - current_xlim[0]) // 100  # Adjust x-axis tick locations dynamically based on the zoom level. Adjust the denominator as needed
      plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
      plt.xticks(rotation=45, fontsize=10)# Rotate and reduce the font size for better readability
      plt.title(f'{tickers} Moving avereages')
      plt.legend()
      plt.xlabel('date')
      plt.show()



   for tickers in ticker:
    s = data[f'{tickers} SR anualised'].head(1).iloc[0]
    r = data[f'{tickers} SR daily'].head(1).iloc[0]
    n = data[f'{tickers} RSI cumulative_return'].iloc[-1]
    t = data[f'{tickers} Cumulative Buy and Hold Return'].iloc[-1]
    print(f' Anualsied SR of {tickers} is {s:.2f}' )
    print(f' Daily SR of {tickers} is {r:.2f}' )
    print(f' RSI cumulative return for {tickers} is {n:.2f}' )
    print(f' buy and hold cumulative return for {tickers} is {t:.2f}' )


def returns(data,rolling_window,ticker):
   for tickers in ticker:
    data[f'{tickers} Returns'] = data[f'{tickers} Open'].pct_change(periods = rolling_window)# percentage return
    data[f'{tickers} Returns'] = data[f'{tickers} Returns']*100
    data[f'{tickers} Open_rolling'] =  data[f'{tickers} Open'].rolling(window = rolling_window).mean()
   return data


def main():
   ticker = ['AMD','NVDA','META','GOOGL','ASML','MSFT','AVGO','IBM','TSLA']
   start_date = "2010-10-01"
   end_date = "2025-14-04"
   api = 'SLNEQXVO3S7L9JTH'
   RSI_overbought = 70
   RSI_underbought =30
   RSI_window = 14
   short_window = 20
   long_window = 200

   data = donwload_data(ticker,start_date,end_date,api)

   data  =  combine_data(ticker,start_date,end_date)

   data = SR(data,ticker)

   data = RSI.RSI(data,RSI_window,RSI_overbought,RSI_underbought,ticker)

   data = RSI.RSI_cumulative_return(data,ticker)

   data = returns(data,RSI_window,ticker)

   data = MA.moving_averages(data,short_window,long_window,ticker)

   data = MA.calculate_cumulative_return(data,ticker)

   plot(data,ticker,RSI_window,short_window,long_window)

   
    
if __name__ == '__main__':
 main()
