import yfinance as yf
import pandas as pd 
import os
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ----------------- Data Acquisition & Preprocessing -----------------
def fetch_yf_data(ticker, start = '2020-09-18', end = date.today().strftime("%Y-%m-%d")):
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
        data = yf.download(ticker, period="max", interval="1d", auto_adjust=True)
        data.reset_index(inplace=True) 
        df = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "Open", "High", "Low", "Close", "Volume"]
        df["date"] = pd.to_datetime(df["date"]) 
        df = df.set_index("date")
        df.to_csv(filename)
        df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
        return df

# ----------------- Feature Engineering / Signal Generation -----------------

def Rules(df, **kwargs):
    EMA_short   = kwargs.get("EMA_short", 12)
    EMA_long    = kwargs.get("EMA_long", 26)
    EMA_signal  = kwargs.get("EMA_signal", 9)
    rsi_window  = kwargs.get("rsi_window", 14)
    BB_window   = kwargs.get("BB_window", 35)
    BB_std_dev_top  = kwargs.get("BB_std_dev_top", 1.5)
    BB_std_dev_bottom  = kwargs.get("BB_std_dev_bottom", 0.5)

    df['EMA_short'] = df['Close'].ewm(span=EMA_short, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=EMA_long, adjust=False).mean()

    # moving average crossover divergence 
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['MACD_Signal'] = df['MACD'].ewm(span=EMA_signal, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    #Bollinger bands exponetaly weighted
    df['BB_Middle'] = df['Close'].ewm(span=BB_window, adjust=False).mean()
    df['BB_StdDev'] = df['Close'].ewm(span=BB_window, adjust=False).std()
    df['BB_Upper'] = df['BB_Middle'] + BB_std_dev_top * df['BB_StdDev']
    df['BB_Lower'] = df['BB_Middle'] - BB_std_dev_bottom * df['BB_StdDev']

    if kwargs.get("dropna", False):
        df.dropna(inplace=True)

    return df 

# ----------------- Initial signal gneration -----------------
def generate_signals(df):
    df['Signal'] = 0
    for i in range(1, len(df)):
        # Enter the market on a Trend
        if (df['EMA_short'].iloc[i] > df['EMA_long'].iloc[i] and
           df['EMA_short'].iloc[i-1] <= df['EMA_long'].iloc[i-1]):
           df.loc[df.index[i], 'Signal'] = 1
    return df

# ------------------------------------------

# ----------------- Strategy Backtesting -----------------

def backtest(df_b, **kwargs):
    df = df_b.copy()
    trades = []
    Signal = []
    rsi_thresh = kwargs.get("rsi_thresh", 50)

    # Initialize columns
    df["Sell Signal"] = 0
    df["Position"] = 0

    # Identify basic signals
    buy_signal = (df['Signal'] == 1)
    breakout_signal = (df['Close'] > df['BB_Upper'])
    ema_exit = (df['EMA_short'] < df['EMA_long'])
    rsi_exit = (df['RSI'] < rsi_thresh) & (breakout_signal.shift(1).fillna(False))

    # Initialize position tracking
    position = 0
    in_breakout = False
    entry_price = 0
    entry_date = None

    # Precompute exits for speed
    exit_signal = pd.Series(False, index=df.index)

    for i in range(len(df) - 1):
        if not in_breakout:
            if buy_signal.iloc[i] and position == 0:
                # Enter position
                position = 1
                entry_price = df['Close'].iloc[i]
                entry_date = df.index[i]
                Signal.append(f'Buy - {entry_date, entry_price}')

            elif position == 1:
                # EMA crossover exit
                if ema_exit.iloc[i]:
                    exit_signal.iloc[i] = True
                # Breakout entry
                if breakout_signal.iloc[i]:
                    in_breakout = True

        else:
            # Breakout exit
            if rsi_exit.iloc[i]:
                exit_signal.iloc[i] = True
            if not breakout_signal.iloc[i]:
                in_breakout = False

        # Handle exits
        if exit_signal.iloc[i] and position == 1:
            exit_price = df['Close'].iloc[i]
            exit_date = df.index[i + 1] if i + 1 < len(df) else df.index[i]
            trades.append({
                'EntryDate': entry_date,
                'ExitDate': exit_date,
                'Return': (exit_price - entry_price) / entry_price
            })
            Signal.append(f'Sell - {exit_date, exit_price}')
            df.at[df.index[i], "Sell Signal"] = 1
            position = 0
            entry_date = None
            in_breakout = False

        # Daily portfolio update
        df.at[df.index[i], "Position"] = position

    # Build trades DataFrame
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.set_index('ExitDate', inplace=True)
        first_entry_date = trades_df['EntryDate'].iloc[0]
        buy_hold_start_price = df.loc[first_entry_date, 'Close']
        trades_df['BuyAndHold'] = df.loc[trades_df.index, 'Close'].values / buy_hold_start_price
        trades_df['StrategyEquity'] = (1 + trades_df['Return']).cumprod()
    else:
        trades_df = pd.DataFrame(columns=['EntryDate', 'ExitDate', 'Return', 'BuyAndHold'])

    # Equity curve
    df["MarketReturn"] = df["Close"].pct_change(fill_method=None).fillna(0)
    df["StrategyReturn"] = df["MarketReturn"] * df["Position"].shift(1).fillna(0)
    df["Equity"] = (1 + df["StrategyReturn"]).cumprod()
    df["BuyHold"] = df["Close"] / df["Close"].iloc[0]

    return trades_df, Signal, df


# ------------------------------------------

# ----------------- strategy Pipeline -----------------

def strat_pipeline(df, tickers, params):
    trades_df = {}
    signals_df = {}
    df_BT = {}
    for ticker in tickers:
        ticker_df = df[ticker].copy()
        fast, slow, signal, window,rsi_W, rsi_T ,lower_std, upper_std  = params[ticker]
        df_tick = Rules(ticker_df, EMA_short = fast, EMA_long = slow, EMA_signal = signal, 
                            BB_window = window, BB_std_dev_top = upper_std, BB_std_dev_bottom = lower_std, rsi_window = rsi_W, dropna=False)
        df_tick = generate_signals(df_tick)
        trades_df_tick, signals, BT_df = backtest(df_tick, rsi_thresh = rsi_T )
        trades_df[ticker] = trades_df_tick
        signals_df[ticker] = signals
        df_BT[ticker] =  BT_df
    return trades_df , signals_df, df_BT

# ----------------- Performance of different variations -----------------    
def variation_perfromace_plot(df,tickers,n_trials = 100):
    trade_means = {}
    for ticker in tickers:
        start = df[ticker].index.min()
        end   = df[ticker].index.max()
        total_days  = (end - start).days
        time_years  = total_days / 365.25
        total_return = []
        total_return_std = []
        means = []
        std= []
        ticker_df = df[ticker].copy()
        for _ in tqdm(range(n_trials)):
            rsi_W = int(np.random.randint(5, 25))
            rsi_T = int(np.random.randint(20, 50))
            fast  = int(np.random.randint(5, 20))
            slow  = int(np.random.randint(21, 50))
            signal = int(np.random.randint(5, 25))
            window = int(np.random.randint(10, 30))
            lower_std = float(np.round(np.random.uniform(0.5, 2), 1))
            upper_std = float(np.round(np.random.uniform(0.5, 2), 1))
            df_tick = Rules(ticker_df, EMA_short = fast, EMA_long = slow, EMA_signal = signal, 
                                BB_window = window, BB_std_dev_top = upper_std, BB_std_dev_bottom = lower_std, rsi_window = rsi_W, dropna=False)
            df_tick = generate_signals(df_tick)
            _, _, df_BT = backtest(df_tick, rsi_thresh = rsi_T )

            if df_BT.empty:
                continue
            else:
                daily_returns = df_BT["StrategyReturn"].dropna()
                returns_mean = (daily_returns.mean()*252)*100
                retrun_std = ((daily_returns.std())*np.sqrt(252))*100
                total = df_BT['Equity'].iloc[-1] 
                total_std = df_BT['Equity'].std()

            means.append(returns_mean)
            std.append(retrun_std)
            total_return.append(total)
            total_return_std.append(total_std)
        means = np.array(means, dtype=float)
        std = np.array(std, dtype=float)
        total_return = np.array(total_return, dtype=float)
        total_return_std = np.array(total_return_std, dtype=float)

        trade_means[ticker] = means

        plt.scatter(std, means, alpha = 0.4, edgecolor = 'black')
        plt.title(f'Risk vs. reward of Rule variations for {ticker}')
        plt.grid()
        plt.xlabel('Anual Volatility %')
        plt.ylabel('Mean Anual Return %')
        plt.show()
        plt.scatter(total_return_std, total_return, alpha = 0.4, edgecolor = 'black')
        plt.title(f'Risk vs. reward of Rule variations for {ticker} over {time_years:.1f} years')
        plt.grid()
        plt.xlabel('Overall Volatility')
        plt.ylabel('Overall Return')
        plt.show()
    return trade_means

# ----------------- BB_MACD Variations Performance Evaluation and optimisation with Optuna -----------------    

def evaluate_pipeline(df, EMA_short, EMA_long, EMA_signal, BB_window, BB_std_dev_top,BB_std_dev_bottom, rsi_W, rsi_T):

    df = Rules(df, EMA_short = EMA_short, EMA_long = EMA_long, EMA_signal = EMA_signal, 
                    BB_window = BB_window, BB_std_dev_top = BB_std_dev_top, BB_std_dev_bottom = BB_std_dev_bottom, rsi_window = rsi_W, dropna=False)
    df = generate_signals(df)
    _, _, df_BT = backtest(df, rsi_thresh = rsi_T)

    daily_returns = df_BT["StrategyReturn"].dropna()
    if daily_returns.std() == 0:
        return -999, daily_returns  # avoid divide by zero

    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)   

    return sharpe, daily_returns


def objective(trial, df):
    EMA_short = trial.suggest_int('EMA_short', 5, 20)
    EMA_long = trial.suggest_int('EMA_long', 21, 50)
    EMA_signal = trial.suggest_int('EMA_signal', 5, 25)
    BB_window = trial.suggest_int('BB_window', 10, 30)
    rsi_W = trial.suggest_int('RSI_window', 5, 25)
    rsi_T = trial.suggest_int('RSI_threshold', 20, 50)
    BB_std_dev_top = trial.suggest_float('BB_std_dev_top',0.5,2)
    BB_std_dev_bottom = trial.suggest_float('BB_std_dev_bottom',0.5,2)

    sharpe, daily_returns = evaluate_pipeline(df.copy(),  EMA_short, EMA_long, EMA_signal, BB_window, BB_std_dev_top, BB_std_dev_bottom, rsi_W, rsi_T)

    trial.set_user_attr("daily_returns", daily_returns)
    
    return sharpe

# ----------------- Portfolio Performance Evaluation and optimisation with Optuna -----------------    

def  portfolio_performance_pipeline(df,weights,tickers):
    weights = np.array(list(weights))
    weights = weights/sum(weights)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must sum to 1.")
    
    # Extract return columns
    return_cols = [f"{ticker}" for ticker in tickers]
    returns = df[return_cols].dropna()

    # Daily portfolio returns
    portfolio_returns = returns.dot(weights)

    # Calculate performance metrics
    mean_daily_return = portfolio_returns.mean()
    std_daily_return = portfolio_returns.std()

    #Anualsied sharpe ratio
    sharpe_ratio = mean_daily_return/ std_daily_return * np.sqrt(252)
    return sharpe_ratio , portfolio_returns

def objective_weights(trial, df, tickers):
    n = len(tickers)
    raw_weights = np.array([trial.suggest_float(f"w_{i}",0.0, 1.0) for i in range(n)])

    # Normalise so they sum to 1
    weights = raw_weights / np.sum(raw_weights)
    weights = weights/sum(weights)
    
    # Calculate Sharpe ratio
    sharpe, port_returns = portfolio_performance_pipeline(df, weights, tickers)
    trial.set_user_attr("portfolio_returns", port_returns)
    
    # Optuna minimizes objective, so return negative Sharpe ratio
    return sharpe

def bootstrap_weighted(df, n_simulations=1000):
    bootstrap_results = [] 
    for i in range(n_simulations):
        sample = df.sample(len(df), replace = True, random_state = i)
        sample_mean = sample.mean()
        sample_std = sample.std()
        sharpe = sample_mean/sample_std * np.sqrt(252)
        bootstrap_results.append({"c0_means":sample_mean,
                            "c0_dev":sample_std,
                            "sharpe":sharpe})
    return bootstrap_results

def get_bins(data):
    n = len(data)
    return min(max(int(np.sqrt(n)), 10), 100) 

def plot_bootstrap_weights(results,tickers,weights,title):
    # Helper function for title with weights
    weights = np.array(list(weights))
    weights = weights/sum(weights)
    results = pd.DataFrame(results)
    fig3, ax3 = plt.subplots(1, 3, figsize=(20, 4))
    sns.histplot(results['c0_means']*256, bins=get_bins(results['c0_means']), kde=True, ax=ax3[0], label = 'Portfolio')
    ax3[0].legend()
    ax3[0].set_title('Distribution of anual system return Means')
    sns.histplot(results['c0_dev']*np.sqrt(252), bins=get_bins(results['c0_dev']), kde=True, ax=ax3[1])
    ax3[1].set_title('Distribution of anual system retrun Standard Deviations')
    sns.histplot(results['sharpe'], bins=get_bins(results['sharpe']), kde=True, ax=ax3[2])
    ax3[2].axvline(np.mean(results['sharpe'])+ 2 * np.std(results['sharpe']), linestyle='--', label='+2σ')
    ax3[2].axvline(np.mean(results['sharpe']) - 2 * np.std(results['sharpe']), linestyle='--', label='-2σ')
    ax3[2].set_title(f'Dsitribution of system Anualised Sharp Ratios')
    ax3[2].legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def cumulative_return_plot(df, df_weights, tickers, weights, title):
    weights = np.array(list(weights))
    weights = weights/sum(weights)
    for ticker in tickers:
        plt.plot((1+df[f'{ticker}']).cumprod(), label = ticker)
    plt.plot((1+df_weights).cumprod(), label = 'Portfolio', color = 'yellow')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.show()

