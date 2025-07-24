import yfinance as yf
import pandas as pd 
import os
from datetime import date
today = date.today().strftime("%Y-%m-%d")

def fetch_yf_data(ticker,start = '2020-09-18', end = today):
    filename = f'{ticker}_data_for_book.csv'
    
    if os.path.exists(filename):
        print(f"{filename} already exists. Loading from disk...")
        df = pd.read_csv(filename)
        df.reset_index(inplace=True) 
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        return df
    else:
        print(f"Downloading data for {ticker}...")
        data = yf.download(ticker, auto_adjust=False) 
        data.reset_index(inplace=True) 
        df = data[["Date", "Open", "High", "Low", "Adj Close", "Volume"]]
        df.columns = ["date", "Open", "High", "Low", "Adj Close", "Volume"]
        df["date"] = pd.to_datetime(df["date"]) 
        df = df.set_index("date")
        df.to_csv(filename)
        df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        return df

