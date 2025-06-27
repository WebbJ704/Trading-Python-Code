import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import seaborn as sns
import matplotlib.dates as mdates
from alpaca_trade_api.rest import REST, TimeFrame
from datetime import datetime

# Alpaca API setup
API_KEY = 'PKDYS2UOHMH6WP8MZF2V'
SECRET_KEY = 'UN1gUNnffY1iqJDsM8GemUGoO1KwpjWqhTYbLSex'
BASE_URL = 'https://paper-api.alpaca.markets'
api = REST(API_KEY, SECRET_KEY, BASE_URL)

# ----------------- CONFIG -----------------
ALPHA_VANTAGE_API_KEY = 'SLNEQXVO3S7L9JTH'  
SYMBOL = ['AMD','NVDA']
INTERVAL = 'Day'
# ------------------------------------------

def fetch_alpha_vantage_data(ticker):
    print(f"Downloading data for {ticker}...")
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format="pandas")
    if INTERVAL == 'Day':
        data, meta_data = ts.get_daily(symbol=ticker, outputsize="full")
    else:
        data, meta_data = ts.get_intraday(symbol=ticker, interval=INTERVAL, outputsize="full")
    data.columns = ["Open", "High", "Low", "Close", "Volume"]
    data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"])
    df = data.set_index("date")
    df = normalize_dataframe(df)
    return df

def normalize_dataframe(df):
    df = df.copy()
    # Fix index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # Sort by time ascending
    df = df.sort_index()
    # Convert all numeric columns (handle str on first load)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def calculate_indicators(df):
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low']) / 2).cumsum() / df['Volume'].cumsum()
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
    df['Volume_Spike'] = df['Volume'] > (df['Volume'].rolling(window=20).mean() * 1.5)

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI(14)'] = 100 - (100 / (1 + rs))

    # ADX (Average Directional Index)
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift(1))
    tr3 = abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    plus_di = 100 * plus_dm.rolling(window=14).sum() / atr
    minus_di = 100 * minus_dm.rolling(window=14).sum() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14).mean()
    df['ADX'] = adx

    return df

def stock_viability(df):
    df['Viabilty'] = False
    for i in range(1,len(df)):
        if df['ADX'].iloc[i] > 25:
            df.loc[df.index[i], 'Viabilty'] = True
        else:
            df.loc[df.index[i], 'Viabilty'] = False
    return df
        
def generate_signals(df):
    df['Signal'] = 0
    for i in range(1,len(df)):
        if (
            df['EMA_9'].iloc[i] > df['EMA_21'].iloc[i] or
            #df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i] and
            df['RSI(14)'].iloc[i] > 70 and
            df['ADX'].iloc[i] > 25 #and
            #df['Close'].iloc[i] > df['VWAP'].iloc[i] and
            #df['Volume_Spike'].iloc[i] #and
            #df['ATR'].iloc[i] > df['ATR'].mean()
        ):
            df.loc[df.index[i], 'Signal'] = 1
    return df

def backtest(df):
    trades = []
    Signal =[]
    position = 0
    entry_price = 0
    entry_date = None

    for i in range(1, len(df)):
        if df['Signal'].iloc[i] == 1 and position == 0: # i-1 to adjust for trade occuring day after close so dont get the close days returns 
            position = 1
            entry_price = df['Close'].iloc[i]
            entry_date = df.index[i]
            Signal.append(f'Buy - {entry_date}')
        elif position == 1:
            # Take Profit or Stop Loss exit
            if df['Close'].iloc[i] >= entry_price * 1.1 or df['Close'].iloc[i] <= entry_price * 0.92:
                exit_price = df['Close'].iloc[i]
                exit_date = df.index[i]
                trade_return = (exit_price - entry_price) / entry_price
                trades.append({'EntryDate': entry_date, 'ExitDate': exit_date, 'Return': trade_return})
                Signal.append(f'Sell - {exit_date}')
                position = 0
                entry_date = None

       # Only construct trades_df if trades exist
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.set_index('ExitDate', inplace=True)

        # Align Buy & Hold: From first entry date to each trade exit
        first_entry_date = trades_df['EntryDate'].iloc[0]
        buy_hold_start_price = df.loc[first_entry_date, 'Close']
        trades_df['BuyAndHold'] = df.loc[trades_df.index, 'Close'].values / buy_hold_start_price

    else:
        trades_df = pd.DataFrame(columns=['EntryDate', 'ExitDate', 'Return', 'BuyAndHold'])

    return trades_df , Signal

def bootstap(trades,n_simulations=1000):
    c0_means = []
    c0_deviations = []
    sharp_ratio = []
    for i in range(1, n_simulations):
        sample = trades['Return'].sample(10, replace = True, random_state = i)
        sample_mean = sample.mean()
        sample_std = sample.std()
        c0_means.append(sample_mean)
        c0_deviations.append(sample_std)
        sharp_ratio.append(sample_mean/sample_std)
    simulations = []
    for i in range(1, n_simulations):
        sample = trades['Return'].sample(10, replace = True, random_state = i)
        cumulative_return = np.prod(1 + sample)
        simulations.append(cumulative_return)
    return c0_means , c0_deviations, simulations, sharp_ratio

def sharp_ratio(df):
    c0_means = []
    c0_deviations = []
    sr = []
    returns = df['Close'].pct_change().dropna()
    for i in range(1000): 
        sample = returns.sample(10, replace=True, random_state = i)  
        sample_mean = sample.mean()
        sample_std = sample.std()
        if sample_std != 0:
            c0_means.append(sample_mean)
            c0_deviations.append(sample_std)
            sr.append(sample_mean / sample_std)
    return c0_means, c0_deviations, sr

def rolling_backtest_general(df):
    if INTERVAL == '60min':
        period_offset = pd.DateOffset(years=2)
    elif INTERVAL == '15min':
        period_offset = pd.DateOffset(weeks=15)
    elif INTERVAL == '5min':
        period_offset = pd.DateOffset(minutes=50)
    elif INTERVAL == '1min':
        period_offset = pd.DateOffset(minutes=10)
    elif INTERVAL == 'Day':
        period_offset = pd.DateOffset(years=2)

    results = []
    # Ensure datetime index is sorted and converted
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    start_date = df.index.min()
    end_date = df.index.max()
    current_start = start_date
    
    while current_start < end_date:
        current_end = current_start + period_offset
        period_df = df[(df.index >= current_start) & (df.index < current_end)].copy()

        if period_df.empty or len(period_df) < 20:
            current_start = current_end
            continue

        # Generate indicators & signals for this period
        period_df = calculate_indicators(period_df)
        period_df = stock_viability(period_df)
        period_df = generate_signals(period_df)

        # Backtest this period
        trades, date = backtest(period_df)

        if not trades.empty:
            mean_return = trades['Return'].mean()
            std_return = trades['Return'].std()
            sharpe_ratio = mean_return / std_return if std_return != 0 else np.nan

            results.append({
                'PeriodStart': current_start,
                'PeriodEnd': current_end,
                'MeanReturn': mean_return,
                'StdDev': std_return,
                'SharpeRatio': sharpe_ratio,
                'NumTrades': len(trades)
            })

        current_start = current_end

    return pd.DataFrame(results)

def get_bins(data):
    n = len(data)
    return min(max(int(np.sqrt(n)), 10), 100) 

def plot(mean_sys, std_sys, sims_sys, sharp_sys, mean_stock, std_stock, sharp_stock, results):


    #Plot stock data mean, std, sharp
    fig1, ax1 = plt.subplots(1, 3, figsize=(20, 4))
    sns.histplot(mean_stock, bins=get_bins(mean_stock), kde=True, ax=ax1[0])
    ax1[0].set_title('Distribution of Stocks return Means')
    sns.histplot(std_stock, bins=get_bins(std_stock), kde=True, ax=ax1[1])
    ax1[1].set_title('Distribution of Stock Return Standard Deviations')
    sns.histplot(sharp_stock, bins=get_bins(sharp_stock), kde=True, ax=ax1[2])
    ax1[2].set_title('Dsitribution of Sharp Ratios')
    plt.xlabel('%Return')
    plt.tight_layout()
    plt.show()

    #plot rolling window systm mean, std, sharp
    fig2, ax2 = plt.subplots(1, 3, figsize=(20, 4))
    sns.kdeplot(results['MeanReturn'], ax=ax2[0])
    ax2[0].set_title('Distribution of rolling window system return Means')
    sns.kdeplot(results['StdDev'], ax=ax2[1])
    ax2[1].set_title('Distribution of rolling window system retrun Standard Deviations')
    sns.kdeplot(results['SharpeRatio'], ax=ax2[2])
    ax2[2].set_title('Dsitribution of rolling window system Sharp Ratios')
    plt.tight_layout()
    plt.show()

    #Plot bootleg 1 system mean, std, sharp 
    fig3, ax3 = plt.subplots(1, 3, figsize=(20, 4))
    sns.histplot(mean_sys, bins=get_bins(mean_sys), kde=True, ax=ax3[0])
    ax3[0].set_title('Distribution of system return Means')
    sns.histplot(std_sys, bins=get_bins(std_sys), kde=True, ax=ax3[1])
    ax3[1].set_title('Distribution of system retrun Standard Deviations')
    sns.histplot(sharp_sys, bins=get_bins(sharp_sys), kde=True, ax=ax3[2])
    ax3[2].set_title('Dsitribution of system Sharp Ratios')
    plt.tight_layout()
    plt.show()

    # plot bootleg 2 
    sns.histplot(sims_sys, bins=50, kde=True)
    plt.title("Bootstrapped system Return Distribution")
    plt.xlabel("Final Equity Multiplier")
    plt.show()

    trade_signals = df[df['Signal'] == 1]
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    # Price + EMAs + Signals (Top)
    ax1.plot(df['Close'], label='Close Price', color='black')
    ax1.plot(df['EMA_9'], label='EMA 9', color='gold')
    ax1.plot(df['EMA_21'], label='EMA 21', color='blue')
    ax1.scatter(trade_signals.index, trade_signals['Close'], color='green', label='Signal', zorder=5)
    ax1.set_title("Trade Signals")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.tick_params(axis='x', rotation=0)
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.tick_params(axis='x', rotation=45)
    # RSI (Bottom)
    ax2.plot(df.index, df['RSI(14)'], label='RSI(14)', color='purple')
    ax2.axhline(70, linestyle='--', color='red', alpha=0.7)
    ax2.axhline(30, linestyle='--', color='green', alpha=0.7)
    ax2.set_title("Relative Strength Index (RSI)")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("RSI")
    ax2.legend()
    ax2.tick_params(axis='x', rotation=0)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, 100)
    plt.tight_layout()
    plt.show()


    trades['StrategyEquity'] = (1 + trades['Return']).cumprod()
    # Ceeate figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot the strategy and buy & hold
    ax.plot(trades.index, trades['StrategyEquity'], label='Strategy')
    ax.plot(trades.index, trades['BuyAndHold'], label='Buy & Hold', linestyle='--')
    # Format the x-axis as datetime
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.tick_params(axis='x', rotation=45)
    # Add titles and labels
    ax.set_title('Cumulative Returns Comparison')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for ticker in SYMBOL:
        #print("Fetching Alpha Vantage data...")
        df = fetch_alpha_vantage_data(ticker)  
        df = calculate_indicators(df)
        df = stock_viability(df)
        df = generate_signals(df)

        #print("Running backtests...")
        trades , BuySellDate = backtest(df)
       # print(f"Completed {len(trades)} trades")

        mean_sys, std_sys, sims_sys, sharp_sys = bootstap(trades)
        mean_stock, std_stock, sharp_stock = sharp_ratio(df)
        results = rolling_backtest_general(df)
        print(f'latest signal was {BuySellDate[-1]}')
        #print(trades)
    #plot(mean_sys,std_sys,sims_sys,sharp_sys, mean_stock, std_stock, sharp_stock, results)