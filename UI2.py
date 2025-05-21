import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import seaborn as sns
import os
from scipy.stats import skew
import scipy.stats as stats
from matplotlib.ticker import MaxNLocator
import tkinter as tk
from tkinter import ttk
from datetime import datetime

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

def calculate_adx(data, period=14):
    # Calculate True Range (TR)
    data['High-Low'] = data['High'] - data['Low']
    data['High-Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-Close'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)

    # Calculate Directional Movement (DM)
    data['+DM'] = np.where((data['High'].diff() > data['Low'].diff()) & (data['High'].diff() > 0), data['High'].diff(), 0)
    data['-DM'] = np.where((data['Low'].diff() > data['High'].diff()) & (data['Low'].diff() > 0), data['Low'].diff(), 0)

    # Smooth the TR, +DM, -DM using an exponential moving average
    data['ATR'] = data['TR'].rolling(window=period).mean()
    data['+DI'] = (100 * (data['+DM'].rolling(window=period).mean() / data['ATR']))
    data['-DI'] = (100 * (data['-DM'].rolling(window=period).mean() / data['ATR']))

    # Calculate DX (Directional Index)
    data['DX'] = (abs(data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI'])) * 100

    # Calculate ADX (smoothed DX)
    data['ADX'] = data['DX'].rolling(window=period).mean()

    return data

def moving_averages(data,short_window,long_window):
    # Calculate the short and long moving averages
    data['SMA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_Long'] = data['Close'].rolling(window=long_window).mean()
    # Initialize Signal column to 0 (no action)
    signals = []
    last_signal = None

    for i in range(len(data['Close'])):
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


def RSI(data,rolling_window,RSI_overbought,RSI_underbought):
    data['daily price change'] = data['Close'].diff()

    data['gain'] = np.where(data['daily price change'] > 0, data['daily price change'], 0)
    data['loss'] = np.where(data['daily price change'] < 0, -data['daily price change'], 0)

    data['avg_gain'] = data['gain'].rolling(window=rolling_window, min_periods=1).mean()
    data['avg_loss'] = data['loss'].rolling(window=rolling_window, min_periods=1).mean()

    data['RS'] = data['avg_gain'] /  data['avg_loss']
    data['RSI'] = 100 - (100 / (1 + data['RS']))
    data['RSI'] = (data['RSI'].replace(0,pd.NA)).dropna()
     
    last_signal = None
    signals = []
    for i in range(len(data['RSI'])):
        if i >= rolling_window:
            if data.iloc[i, data.columns.get_loc('RSI')] >= RSI_overbought and (last_signal != 'buy'):
                signals.append('buy')
                last_signal = 'buy'
            elif data.iloc[i, data.columns.get_loc('RSI')] <= RSI_underbought and (last_signal != 'sell'):
                signals.append('sell')
                last_signal ='sell'
            else:
             signals.append(None)
        else:
         signals.append(None)
    
    if len(signals) < len(data):
     signals.append(None)

    data['RSI signal'] = signals
    data['RSI buy signals'] = data['RSI signal'] == 'buy'
    data['RSI sell signals'] = data['RSI signal'] == 'sell'
    return data


def Sharp_ratio(data,anual_window):
   
   data['Mean Anual'] =  (data['Close'].pct_change(periods = anual_window)).mean()
   data['std Anual'] = (data['Close'].pct_change(periods = anual_window)).std()
   data['daily mean'] = data['Mean Anual']/anual_window
   data['daily std'] = data['std Anual']/ np.sqrt(anual_window)
   

   data['SR anualised'] =  data['Mean Anual'] /  data['std Anual']
   data['SR daily'] =   data['daily mean']/ data['daily std']
   sr_daily = data['SR daily'].iloc[0]
   sr_anual = data['SR anualised'].iloc[0]
   print(f' daily sharp ratio is {sr_daily:.3f}')
   print(f' the anual sharp ratio is {sr_anual:.2f}') 

   return data

def returns(data,rolling_window):
    data['Returns'] = data['Close'].pct_change(periods = rolling_window)# percentage return
    data['Returns'] = data['Returns']*100
    data['Close_rolling'] =  data['Close'].rolling(window = rolling_window).mean()
    return data

def standard_deviation(data,rolling_window):
    std = data['Returns'].std()
    anual_volatiltiy = ((data['Close'].pct_change()).std())*np.sqrt(252)*100
    data['Rolling Volatility'] = (((data['Close'].pct_change()).rolling(window = rolling_window)).std())*100
    low_volatility = data['Rolling Volatility'].mean()-data['Rolling Volatility'].std()
    high_volatility = data['Rolling Volatility'].mean()+data['Rolling Volatility'].std()
    last_signal = None
    signals = []
    for i in range(len(data['Rolling Volatility'])):
        if i >= rolling_window:
            if data.iloc[i, data.columns.get_loc('Rolling Volatility')] <= low_volatility and (last_signal != 'buy'):
                signals.append('buy')
                last_signal = 'buy'
            elif data.iloc[i, data.columns.get_loc('Rolling Volatility')]  >= high_volatility and (last_signal != 'sell'):
                signals.append('sell')
                last_signal ='sell'
            else:
             signals.append(None)
        else:
         signals.append(None)
    
    if len(signals) < len(data):
     signals.append(None)

    data['Volatility signal'] = signals
    data['Volatility buy signals'] = data['Volatility signal'] == 'buy'
    data['Volatility sell signals'] = data['Volatility signal'] == 'sell'

    return std , anual_volatiltiy , data ,low_volatility ,high_volatility

def mean(data):
    mean_close = data['Returns'].mean()
    return mean_close

def prob_den_fs(mean_close,std):
    x = np.linspace(mean_close - 3*std, mean_close + 3*std, 1000)
    y = stats.norm.pdf(x, mean_close, std)
    return x,y

def calculate_cumulative_return(data):
    data['daily_return'] = data['Close'].pct_change() # Calculate daily returns (percentage change)
    data['RSI position'] = 0 # Initialize a column to track positions (1 for holding, 0 for cash)
    data['MA position'] = 0 # Initialize a column to track positions (1 for holding, 0 for cash)

    RSI_last_position = 0  # Initially, assume we're not in the market
    for i in range(len(data['daily_return'])):
        if data['RSI buy signals'].iloc[i]: # Buy signal
            RSI_last_position = 1  # Enter the market
        elif data['RSI sell signals'].iloc[i]:  # Sell signal
            RSI_last_position = 0  # Exit the market
        
        data.iloc[i, data.columns.get_loc('RSI position')] = RSI_last_position # Set the current position

    MA_last_position = 0  # Initially, assume we're not in the market
    for i in range(len(data['daily_return'])):
        if data['MA buy signals'].iloc[i]: # Buy signal
            MA_last_position = 1  # Enter the market
        elif data['MA sell signals'].iloc[i]:  # Sell signal
           MA_last_position = 0  # Exit the market
        
        data.iloc[i, data.columns.get_loc('MA position')] = MA_last_position # Set the current position
    
    # Shift position to match the date of action
    data['RSI position'] = data['RSI position'].shift(1, fill_value=0)
    data['MA position'] = data['MA position'].shift(1, fill_value=0)

    # Calculate the strategy's daily return as the daily return * position
    data['RSI strategy_return'] = data['daily_return'] * data['RSI position']
    data['MA strategy_return'] = data['daily_return'] * data['MA position']
    # Calculate the cumulative return of the strategy (multiply daily returns)
    data['RSI cumulative_return'] = (1 + data['RSI strategy_return']).cumprod() 
    data['MA cumulative_return'] = (1 + data['MA strategy_return']).cumprod() 
    data['Cumulative Buy and Hold Return'] = (1 + data['daily_return']).cumprod() 
    return data

def plotting(data, ticker, return_window, std_dev, x, y, mean_Return, std_dev_Return, anual_vol, rsi_window, short_window, long_window, low_volatility, high_volatility, log_widget):
   # Get the latest signals (last non-None)
    latest_signal_RSI = data['RSI signal'].dropna().iloc[-1] if not data['RSI signal'].dropna().empty else None
    latest_signal_MA = data['MA Signals'].dropna().iloc[-1] if not data['MA Signals'].dropna().empty else None
    if data['ADX'].iloc[-1] > 25:
        log_widget.insert(tk.END, f"Current ADX has a value of {data['ADX'].iloc[-1]:.2f} suggesting a strong trending market so use moving averages\n")
        log_widget.insert(tk.END, f"the last signal for moving averages is to {latest_signal_MA}\n")
    else:
        log_widget.insert(tk.END, f"Current ADX has a value of {data['ADX'].iloc[-1]:.2f} suggesting a choppy market so use RSI\n")
        log_widget.insert(tk.END, f"the last signal for RSI is to {latest_signal_RSI}\n")
    
    plt.figure()
    plt.plot(data['ADX'])
    current_xlim = plt.gca().get_xlim()
    num_ticks = (current_xlim[1] - current_xlim[0]) // 100 
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
    plt.xticks(rotation=45, fontsize=10)
    plt.axhline(y=25, color='red', linestyle='--', label="above = treding marke(Averages) , below = choppy market(RSI)")
    plt.ylabel('ADX index')
    plt.xlabel('Date')
    plt.legend()
    plt.show()
    
    plt.figure()
    data['Close'].plot(title='Close Prices', figsize=(10, 6))
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(8,5))
    sns.histplot(data['Returns'], bins=30, kde=True, color='blue')
    plt.xlabel(f'{return_window} day return')
    plt.axvline(data['Returns'].mean(), color='red', linestyle='dashed', label="Mean")
    plt.axvline(data['Returns'].mean() + std_dev, color='green', linestyle='dashed', label="+1 Std Dev")
    plt.axvline(data['Returns'].mean() - std_dev, color='green', linestyle='dashed', label="-1 Std Dev")
    plt.ylabel("Frequency")
    plt.title("Histogram of Data Distribution")
    plt.show()

    skew_value = skew(data['Returns'].dropna())
    print(f'{ticker} has a skewness of {skew_value:.2f}' )

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Normal Distribution', color='blue')
    plt.axvline(data['Returns'].mean(), color='red', linestyle='dashed', label="Mean")
    plt.axvline(data['Returns'].mean() + std_dev, color='green', linestyle='dashed', label="+1 Std Dev")
    plt.axvline(data['Returns'].mean() - std_dev, color='green', linestyle='dashed', label="-1 Std Dev")
    plt.title(f'Normal Distribution of {ticker} {return_window} day Return')
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend(loc='upper left')
    plt.show()
    
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax[0].plot(data['SMA_Short'],label = f'{short_window} moving average' )
    ax[0].plot(data['SMA_Long'], label = f'{long_window} moving average')
    ax[1].scatter(data.index[data['MA buy signals']],data['Close_rolling'][data['MA buy signals']], marker='^', color='green', label='Buy Signal', zorder=5)
    ax[1].scatter(data.index[data['MA sell signals']],data['Close_rolling'][data['MA sell signals']], marker='v', color='red', label='Sell Signal', zorder=5)
    ax[1].plot(data['Close_rolling'], label = 'Close price' )
    ax[0].set_ylabel('Moving Averages')
    ax[1].set_ylabel('stock Price')
    plt.xticks(rotation=90)
    current_xlim = plt.gca().get_xlim()
    num_ticks = (current_xlim[1] - current_xlim[0]) // 100  # Adjust x-axis tick locations dynamically based on the zoom level. Adjust the denominator as needed
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
    plt.xticks(rotation=45, fontsize=10)# Rotate and reduce the font size for better readability
    plt.title(f'Moving avereages')
    plt.legend()
    plt.xlabel('date')
    plt.show()

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax[0].plot(data['RSI'])
    ax[1].scatter(data.index[data['RSI buy signals']],data['Close_rolling'][data['RSI buy signals']], marker='^', color='green', label='Buy Signal', zorder=5)
    ax[1].scatter(data.index[data['RSI sell signals']],data['Close_rolling'][data['RSI sell signals']], marker='v', color='red', label='Sell Signal', zorder=5)
    ax[1].plot(data['Close_rolling'], label = 'Close price' )
    ax[0].set_ylabel('RSI')
    ax[1].set_ylabel('stock Price')
    plt.xticks(rotation=90)
    current_xlim = plt.gca().get_xlim()
    num_ticks = (current_xlim[1] - current_xlim[0]) // 100  # Adjust x-axis tick locations dynamically based on the zoom level. Adjust the denominator as needed
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
    plt.xticks(rotation=45, fontsize=10)# Rotate and reduce the font size for better readability
    plt.title(f'RSI over {rsi_window} days')
    plt.xlabel('date')
    plt.show()

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax[0].plot(data['Rolling Volatility'], label = 'Pirce volatitly')
    ax[1].scatter(data.index[data['Volatility buy signals']],data['Close_rolling'][data['Volatility buy signals']], marker='^', color='green', label='Buy Signal', zorder=5)
    ax[1].scatter(data.index[data['Volatility sell signals']],data['Close_rolling'][data['Volatility sell signals']], marker='v', color='red', label='Sell Signal', zorder=5)
    ax[1].plot(data['Close_rolling'], label = 'Close price' )
    ax[0].set_ylabel('Volatitly %')
    ax[1].set_ylabel('stock Price')
    ax[0].axhline(y=low_volatility, color='green', linestyle='--', label=f"low vol - 1 std = {low_volatility:.2f}")
    ax[0].axhline(y=high_volatility, color='red', linestyle='--', label=f"high vol + 1 std = {high_volatility:.2f}")
    current_xlim = plt.gca().get_xlim()
    num_ticks = (current_xlim[1] - current_xlim[0]) // 100 
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
    plt.xticks(rotation=45, fontsize=10)
    plt.show()

    plt.figure()
    plt.plot(data['RSI cumulative_return'], label = ' RSI')
    plt.plot(data['Cumulative Buy and Hold Return'], label = 'buy/hold')
    num_ticks = (current_xlim[1] - current_xlim[0]) // 100 
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(loc='upper left')
    plt.show()

    plt.figure()
    plt.plot(data['MA cumulative_return'], label = ' Moving avereages')
    plt.plot(data['Cumulative Buy and Hold Return'], label = 'buy/hold')
    num_ticks = (current_xlim[1] - current_xlim[0]) // 100 
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=num_ticks))
    plt.xticks(rotation=45, fontsize=10)
    plt.legend(loc='upper left')
    plt.show()

    # 5. Print the mean and standard deviation
    log_widget.insert(tk.END, f"Mean of {ticker} Returns: {mean_Return:.2f} %\n")
    log_widget.insert(tk.END, f"Overall Standard Deviation (volatility) of {ticker} Returns: {std_dev_Return:.2f}% over rolling window of {return_window} days\n")
    log_widget.insert(tk.END, f"The annual volatility for {ticker} is {anual_vol:.2f}%\n")


def launch_analysis():
    # Get user inputs
    ticker = ticker_var.get()
    api_key = api_key_var.get()
    start_date = start_date_var.get()
    end_date = end_date_var.get()
    rsi_window = int(rsi_window_var.get())
    rsi_overbought = float(rsi_overbought_var.get())
    rsi_underbought = float(rsi_underbought_var.get())
    short_window = int(short_window_var.get())
    long_window = int(long_window_var.get())
    return_window = int(return_window_var.get())
    annual_window = int(annual_window_var.get())

    # Clear previous logs
    log_widget.delete(1.0, tk.END)
    log_widget.insert(tk.END, "Analysis Started...\n")

    try:
        # Run the pipeline
        df = data_download_pl(ticker, api_key, start_date, end_date)
        df = calculate_adx(df)
        df = moving_averages(df, short_window, long_window)
        df = RSI(df, rsi_window, rsi_overbought, rsi_underbought)
        df = returns(df, return_window)
        df = Sharp_ratio(df, annual_window)
        std, anual_vol, df, low_vol, high_vol = standard_deviation(df, return_window)
        mean_return = mean(df)
        x, y = prob_den_fs(mean_return, std)
        df = calculate_cumulative_return(df)
        plotting(df, ticker, return_window, std, x, y, mean_return, std, anual_vol, rsi_window, short_window, long_window, low_vol, high_vol, log_widget)
        
        log_widget.insert(tk.END, "Analysis Completed.\n")
    
    except Exception as e:
        log_widget.insert(tk.END, f"Error occurred: {str(e)}\n")

# ---------------- GUI LAYOUT ----------------
root = tk.Tk()
root.title("Stock Analyzer")

mainframe = ttk.Frame(root, padding="10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Variables
ticker_var = tk.StringVar(value="AAPL")
api_key_var = tk.StringVar(value="SLNEQXVO3S7L9JTH")
start_date_var = tk.StringVar(value="2005-10-01") 
end_date_var = tk.StringVar(value=datetime.today().strftime("%Y-%m-%d"))
rsi_window_var = tk.StringVar(value="14")
rsi_overbought_var = tk.StringVar(value="70")
rsi_underbought_var = tk.StringVar(value="30")
short_window_var = tk.StringVar(value="20")
long_window_var = tk.StringVar(value="50")
return_window_var = tk.StringVar(value="5")
annual_window_var = tk.StringVar(value="252")

# Create a Text widget to display log messages
log_widget = tk.Text(root, height=15, width=100)
log_widget.grid(column=0, row=1, sticky=(tk.W, tk.E, tk.N, tk.S))
log_widget.insert(tk.END, "Analysis Started...\n")

# Entry form
inputs = [
    ("Ticker", ticker_var),
    ("API Key", api_key_var),
    ("Start Date (YYYY-MM-DD)", start_date_var),
    ("End Date (YYYY-MM-DD)", end_date_var),
    ("RSI Window", rsi_window_var),
    ("RSI Overbought", rsi_overbought_var),
    ("RSI Underbought", rsi_underbought_var),
    ("Short MA Window", short_window_var),
    ("Long MA Window", long_window_var),
    ("Return Window", return_window_var),
    ("Annual Window", annual_window_var),
]

for i, (label, var) in enumerate(inputs):
    ttk.Label(mainframe, text=label).grid(column=0, row=i, sticky=tk.W)
    ttk.Entry(mainframe, width=20, textvariable=var).grid(column=1, row=i)

ttk.Button(mainframe, text="Analyze", command=launch_analysis).grid(column=0, row=len(inputs), columnspan=2, pady=10)

root.mainloop()