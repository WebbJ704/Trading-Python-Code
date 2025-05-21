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

   
    filename= f'{ticker}_stock_data.CSV'

    # Check if file exists
    if os.path.exists(filename):
        print(f" {filename} already exists. Skipping download.")
        df = pd.read_csv(filename)
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        df.set_index("date", inplace=True)
        return df
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
        return df
    

def Sharp_ratio(data,rolling_window):
   
   data['Mean Anual'] =  (data['Close'].pct_change(periods = rolling_window)).mean()
   data['std Anual'] = (data['Close'].pct_change(periods = rolling_window)).std()
   data['daily mean'] = data['Mean Anual']/rolling_window
   data['daily std'] = data['std Anual']/ np.sqrt(rolling_window)
   

   data['SR anualised'] =  data['Mean Anual'] /  data['std Anual']
   data['SR daily'] =   data['daily mean']/ data['daily std']
   return data

def main():
    ticker = 'META'
    start_date = "2005-10-01"
    end_date = "2025-03-10"
    api = 'SLNEQXVO3S7L9JTH'
    return_window = 256
    RSI_window = 14
    short_window = 9
    long_window = 21
    RSI_overbought = 70
    RSI_underbought = 30
    
    adx_window = 20

    data  =  data_download_pl(ticker,api,start_date,end_date)

   # data = returns(data,return_window)

    #std_dev , anual_volatitly , data, low_volatility ,high_volatility = standard_deviation(data,return_window)

   # mean_results = mean(data)
    
   # x,y = prob_den_fs(mean_results,std_dev)

   # data = RSI(data,RSI_window,RSI_overbought,RSI_underbought)

   # data = moving_averages(data,short_window,long_window)
    
   # data = calculate_cumulative_return(data)
    
    data = Sharp_ratio(data,return_window)
    sr_daily = (data['SR daily'].iloc[0])
    print(f'{sr_daily:.3f}')
    print(data['SR anualised'])  ## do for min ADN MAX 
     
    

    #data = calculate_adx(data,adx_window)

    #plotting(data,ticker,return_window,std_dev,x,y,mean_results,std_dev,anual_volatitly,RSI_window,short_window,long_window, low_volatility ,high_volatility)
    
    
if __name__ == '__main__':
 main()
